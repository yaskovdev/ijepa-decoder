import json
import math
from pathlib import Path
import time

import torch as th

from .data import crop_targets, example_hashes, input_spec, load_cache
from .io import configure, save_cache
from .model import initialization_spec, load_decoder, load_image_prior


def perceptual_loss(device, models_dir):
    import lpips

    th.hub.set_dir(str(Path(models_dir) / "torch_hub"))
    return lpips.LPIPS(net="alex", verbose=False).eval().requires_grad_(False).to(device)


def image_errors(prediction, target, perceptual):
    if not th.isfinite(prediction).all():
        raise FloatingPointError("Decoder produced non-finite RGB.")
    return dict(
        raw_l1=(prediction - target).abs().mean((1, 2, 3)),
        rgb_mae=(prediction.clamp(0, 1) - target).abs().mean((1, 2, 3)),
        lpips=perceptual(prediction.clamp(0, 1) * 2 - 1, target * 2 - 1).flatten(),
    )


def averaged(errors, weight):
    result = {name: values.mean().item() for name, values in errors.items()}
    result["score"] = result["raw_l1"] + weight * result["lpips"]
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Reconstruction metrics are not finite.")
    return result


@th.no_grad()
def metrics(model, cache, crops, source, perceptual, device, weight, batch_size):
    previous = model.training
    model.eval()
    totals = {}
    for start in range(0, len(crops), batch_size):
        end = min(start + batch_size, len(crops))
        prediction = model(cache[f"{source}_tokens"][start:end].float().to(device), cache["boxes"][start:end])
        errors = image_errors(prediction, crops[start:end].float().to(device) / 255, perceptual)
        for name, values in errors.items():
            totals.setdefault(name, []).append(values.cpu())
    model.train(previous)
    return averaged({name: th.cat(values) for name, values in totals.items()}, weight)


def train(args):
    if (min(args.steps, args.batch_size) <= 0 or args.learning_rate <= 0
            or not math.isfinite(args.learning_rate) or args.lpips_weight < 0
            or not math.isfinite(args.lpips_weight)):
        raise ValueError("Training requires positive finite steps, batch size, and learning rate.")
    if args.feature_source == "mixed" and (args.batch_size < 2 or args.batch_size % 2):
        raise ValueError("Mixed training requires an even batch size of at least two.")
    device = configure(args)
    training = load_cache(Path(args.cache_dir) / "train.pt")
    validation = load_cache(Path(args.cache_dir) / "val.pt")
    if input_spec(training) != input_spec(validation):
        raise ValueError("Training and validation feature contracts differ.")
    if set(example_hashes(training)).intersection(example_hashes(validation)):
        raise ValueError("Training and validation share original image pixels.")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError("Checkpoints exist; choose a new experiment directory.")
    crops, held_out = crop_targets(training), crop_targets(validation)
    if args.init:
        model, _ = load_decoder(args.init, device, input_spec(training), freeze=False)
    else:
        model = load_image_prior(json.loads(Path(args.assets).read_text()),
                                 training["predictor_tokens"].shape[-1], training["spatial_shape"]).to(device)
    perceptual = perceptual_loss(device, args.models_dir)
    with th.no_grad():
        mean = crops.float().mean(0, keepdim=True).to(device) / 255
        baseline = averaged(image_errors(mean.expand(len(held_out), -1, -1, -1),
                                         held_out.float().to(device) / 255, perceptual), args.lpips_weight)
    source = "target" if args.feature_source == "target" else "predictor"
    initial = metrics(model, validation, held_out, source, perceptual, device,
                      args.lpips_weight, args.batch_size)
    projection = list(model.decoder.decoder_embed.parameters())
    other = [value for name, value in model.named_parameters()
             if value.requires_grad and not name.startswith("decoder.decoder_embed.")]
    optimizer = th.optim.AdamW([
        dict(params=projection, lr=args.learning_rate if args.init else args.learning_rate * 5),
        dict(params=other, lr=args.learning_rate),
    ], weight_decay=0.01)
    best, best_step, history = None, None, [dict(step=0, validation=initial)]
    started = time.monotonic()

    def snapshot(result, step):
        return dict(
            state_dict={name: value.cpu() for name, value in model.state_dict().items()},
            input_spec=input_spec(training), config=model.config.to_dict(), initialization=initialization_spec(),
            feature_source=args.feature_source, training_step=step, validation=result, baseline=baseline,
            training_examples=len(crops), validation_examples=len(held_out), seed=args.seed,
            lpips_weight=args.lpips_weight, init=args.init, learning_rate=args.learning_rate,
        )

    if initial["score"] < baseline["score"]:
        best, best_step = initial, 0
        save_cache(snapshot(best, best_step), output / "best.pt")
    print(f"{args.feature_source}: baseline {baseline}; initial {initial}", flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        indices = th.randint(len(crops), (args.batch_size,))
        tokens = training[f"{source}_tokens"][indices].float()
        if args.feature_source == "mixed":
            tokens[::2] = training["target_tokens"][indices[::2]].float()
        prediction = model(tokens.to(device), training["boxes"][indices])
        errors = image_errors(prediction, crops[indices].float().to(device) / 255, perceptual)
        loss = (errors["raw_l1"] + args.lpips_weight * errors["lpips"]).mean()
        if not th.isfinite(loss):
            raise FloatingPointError(f"Loss is not finite at step {step}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            result = metrics(model, validation, held_out, source, perceptual, device,
                             args.lpips_weight, args.batch_size)
            target = metrics(model, validation, held_out, "target", perceptual, device,
                             args.lpips_weight, args.batch_size) if source != "target" else result
            history.append(dict(step=step, loss=loss.item(), validation=result, target_oracle=target))
            if result["score"] < (baseline["score"] if best is None else best["score"]):
                best, best_step = result, step
                save_cache(snapshot(best, best_step), output / "best.pt")
            save_cache(snapshot(result, step), output / "last.pt")
            print(f"{args.feature_source} {step}/{args.steps}: {result}; target {target}; "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    (output / "metrics.json").write_text(json.dumps(
        dict(baseline=baseline, initial=initial, validation=best, best_step=best_step, history=history), indent=2
    ) + "\n")
    if best is None:
        raise RuntimeError("Decoder did not beat the mean-crop baseline; inspect last.pt.")
    print(f"Best checkpoint at step {best_step}: {output / 'best.pt'}", flush=True)


def evaluate(args):
    if args.batch_size <= 0:
        raise ValueError("Evaluation requires a positive batch size.")
    device = configure(args)
    cache = load_cache(args.cache)
    model, _ = load_decoder(args.decoder, device, input_spec(cache))
    result = metrics(model, cache, crop_targets(cache), "predictor",
                     perceptual_loss(device, args.models_dir), device, 0.5, args.batch_size)
    print(json.dumps(result, indent=2), flush=True)

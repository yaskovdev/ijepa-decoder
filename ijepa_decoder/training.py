import json
import math
from pathlib import Path
import time

import torch as th
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .data import crop_targets, example_hashes, input_spec, load_cache
from .io import configure, file_sha256, save_cache, save_json
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


def paired_difference(baseline, prediction, hashes, seed):
    photographs = {}
    for index, digest in enumerate(hashes):
        photographs.setdefault(digest, []).append(index)
    generator = th.Generator().manual_seed(seed)
    bootstrap = th.randint(len(photographs), (2000, len(photographs)), generator=generator)
    result = {}
    for name in ("rgb_mae", "lpips", "raw_l1"):
        delta = th.tensor(prediction["per_example"][name], dtype=th.float64) - th.tensor(
            baseline["per_example"][name], dtype=th.float64,
        )
        per_photo = th.stack([delta[indices].mean() for indices in photographs.values()])
        limits = th.quantile(per_photo[bootstrap].mean(1), th.tensor([0.025, 0.975], dtype=th.float64))
        result[name] = dict(new_minus_baseline=per_photo.mean().item(), paired_95_percent_ci=limits.tolist(),
                            photographs=len(photographs))
    return result


@th.no_grad()
def metrics(model, cache, crops, source, perceptual, device, weight, batch_size, per_example=False):
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
    errors = {name: th.cat(values) for name, values in totals.items()}
    result = averaged(errors, weight)
    if per_example:
        result["per_example"] = {name: values.tolist() for name, values in errors.items()}
    return result


def cpu_state(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def random_state(device):
    state = dict(cpu=th.get_rng_state())
    if device.type == "mps":
        state["mps"] = th.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = th.cuda.get_rng_state(device)
    return state


def restore_random_state(state, device):
    th.set_rng_state(state["cpu"])
    if device.type == "mps":
        th.mps.set_rng_state(state["mps"])
    elif device.type == "cuda":
        th.cuda.set_rng_state(state["cuda"], device)


def train(args):
    if (min(args.steps, args.batch_size) <= 0 or args.learning_rate <= 0
            or not math.isfinite(args.learning_rate) or args.lpips_weight < 0
            or not math.isfinite(args.lpips_weight) or args.eval_every <= 0
            or not 0 <= args.ema_decay < 1 or not math.isfinite(args.min_learning_rate)
            or not 0 <= args.min_learning_rate <= args.learning_rate):
        raise ValueError("Training requires positive finite steps, batch size, and learning rate.")
    if args.feature_source == "mixed" and (args.batch_size < 2 or args.batch_size % 2):
        raise ValueError("Mixed training requires an even batch size of at least two.")
    stop = args.steps if args.stop_after is None else args.stop_after
    if not 1 <= stop <= args.steps or (args.init and args.resume):
        raise ValueError("Stop step must be inside the full schedule; --init and --resume are mutually exclusive.")
    device = configure(args)
    training_path, validation_path = Path(args.cache_dir) / "train.pt", Path(args.cache_dir) / "val.pt"
    training, validation = load_cache(training_path), load_cache(validation_path)
    if input_spec(training) != input_spec(validation):
        raise ValueError("Training and validation feature contracts differ.")
    if set(example_hashes(training)).intersection(example_hashes(validation)):
        raise ValueError("Training and validation share original image pixels.")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError("Checkpoints exist; choose a new experiment directory.")
    if args.resume and Path(args.resume).resolve().parent != output.resolve():
        raise ValueError("Resume in the checkpoint's existing output directory; use --init for a new experiment.")
    contract = {name: getattr(args, name) for name in (
        "feature_source", "steps", "batch_size", "learning_rate", "min_learning_rate", "schedule",
        "ema_decay", "lpips_weight", "seed", "cpu_threads", "eval_every",
    )}
    contract.update(device=device.type, train_sha256=file_sha256(training_path),
                    val_sha256=file_sha256(validation_path), input_spec=input_spec(training))
    crops, held_out = crop_targets(training), crop_targets(validation)
    checkpoint = None
    if args.resume:
        model, checkpoint = load_decoder(args.resume, device, input_spec(training), freeze=False)
        if checkpoint.get("training_contract") != contract or "optimizer" not in checkpoint:
            raise ValueError("Resume checkpoint differs in data or training settings, or has no optimizer state.")
        if not checkpoint["training_step"] < stop:
            raise ValueError("Resume stop step must be later than the saved training step.")
    elif args.init:
        model, _ = load_decoder(args.init, device, input_spec(training), freeze=False)
    else:
        model = load_image_prior(json.loads(Path(args.assets).read_text()),
                                 training["predictor_tokens"].shape[-1], training["spatial_shape"]).to(device)
    perceptual = perceptual_loss(device, args.models_dir)
    source = "target" if args.feature_source == "target" else "predictor"
    if checkpoint is None:
        with th.no_grad():
            mean = crops.float().mean(0, keepdim=True).to(device) / 255
            baseline = averaged(image_errors(mean.expand(len(held_out), -1, -1, -1),
                                             held_out.float().to(device) / 255, perceptual), args.lpips_weight)
        initial = metrics(model, validation, held_out, source, perceptual, device,
                          args.lpips_weight, args.batch_size)
    else:
        baseline, initial = checkpoint["baseline"], checkpoint["initial"]
    initialized = args.init if checkpoint is None else checkpoint["init"]
    projection = list(model.decoder.decoder_embed.parameters())
    other = [value for name, value in model.named_parameters()
             if value.requires_grad and not name.startswith("decoder.decoder_embed.")]
    optimizer = th.optim.AdamW([
        dict(params=projection, lr=args.learning_rate if initialized else args.learning_rate * 5),
        dict(params=other, lr=args.learning_rate),
    ], weight_decay=0.01)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.min_learning_rate,
    ) if args.schedule == "cosine" else None
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay)) if args.ema_decay else None
    best, best_step, history = None, None, [dict(step=0, validation=initial)]
    start, elapsed = 0, 0.0
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if ema is not None:
            ema.load_state_dict(checkpoint["ema"], strict=True)
        history, start, elapsed = checkpoint["history"], checkpoint["training_step"], checkpoint["elapsed_seconds"]
        if checkpoint["best_step"] is not None:
            _, saved_best = load_decoder(output / "best.pt", "cpu", input_spec(training))
            if saved_best.get("training_contract") != contract:
                raise ValueError("Best checkpoint belongs to a different experiment.")
            best, best_step = saved_best["validation"], saved_best["training_step"]
        # Constructors consume random numbers; restore the sampler state only after setup.
        restore_random_state(checkpoint["rng"], device)
    started = time.monotonic()

    def snapshot(network, result, step, weight_kind):
        return dict(
            state_dict=cpu_state(network.state_dict()),
            input_spec=input_spec(training), config=model.config.to_dict(), initialization=initialization_spec(),
            feature_source=args.feature_source, training_step=step, validation=result, baseline=baseline,
            initial=initial, training_contract=contract, weight_kind=weight_kind, ema_decay=args.ema_decay,
            training_examples=len(crops), validation_examples=len(held_out), seed=args.seed,
            training_originals=len(training["originals"]), validation_originals=len(validation["originals"]),
            lpips_weight=args.lpips_weight, init=initialized, learning_rate=args.learning_rate,
        )

    if checkpoint is None and initial["score"] < baseline["score"]:
        best, best_step = initial, 0
        save_cache(snapshot(model, best, best_step, "raw"), output / "best.pt")
    print(f"{args.feature_source}: {len(crops)} examples / {len(training['originals'])} photographs; "
          f"baseline {baseline}; initial {initial}; starting step {start}", flush=True)
    model.train()
    for step in range(start + 1, stop + 1):
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
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update_parameters(model)
        if step % args.eval_every == 0 or step == stop:
            raw = metrics(model, validation, held_out, source, perceptual, device,
                          args.lpips_weight, args.batch_size)
            result, selected, weight_kind = raw, model, "raw"
            entry = dict(step=step, loss=loss.item(), raw_validation=raw,
                         learning_rates=[group["lr"] for group in optimizer.param_groups])
            if ema is not None:
                smoothed = metrics(ema.module, validation, held_out, source, perceptual, device,
                                   args.lpips_weight, args.batch_size)
                entry["ema_validation"] = smoothed
                if smoothed["score"] < raw["score"]:
                    result, selected, weight_kind = smoothed, ema.module, "ema"
            target = metrics(selected, validation, held_out, "target", perceptual, device,
                             args.lpips_weight, args.batch_size) if source != "target" else result
            entry.update(validation=result, target_oracle=target, weight_kind=weight_kind)
            history.append(entry)
            if result["score"] < (baseline["score"] if best is None else best["score"]):
                best, best_step = result, step
                save_cache(snapshot(selected, best, best_step, weight_kind), output / "best.pt")
            seconds = elapsed + time.monotonic() - started
            state = snapshot(model, raw, step, "raw")
            state.update(optimizer=cpu_state(optimizer.state_dict()), scheduler=scheduler.state_dict() if scheduler else None,
                         ema=cpu_state(ema.state_dict()) if ema is not None else None,
                         rng=random_state(device), history=history, best_step=best_step, elapsed_seconds=seconds)
            save_cache(state, output / "last.pt")
            save_json(dict(baseline=baseline, initial=initial, validation=best, best_step=best_step,
                           history=history, completed=step == args.steps, elapsed_seconds=seconds),
                      output / "metrics.json")
            print(f"{args.feature_source} {step}/{args.steps}: {weight_kind} {result}; "
                  f"target {target}; {seconds:.1f}s", flush=True)
    if best is None and stop == args.steps:
        raise RuntimeError("Decoder did not beat the mean-crop baseline; inspect last.pt.")
    if stop != args.steps:
        print(f"Paused at step {stop}; resume with --resume {output / 'last.pt'}.", flush=True)
    if best is not None:
        print(f"Best checkpoint at step {best_step}: {output / 'best.pt'}", flush=True)


def evaluate(args):
    if args.batch_size <= 0:
        raise ValueError("Evaluation requires a positive batch size.")
    device = configure(args)
    cache = load_cache(args.cache)
    model, _ = load_decoder(args.decoder, device, input_spec(cache))
    result = metrics(model, cache, crop_targets(cache), "predictor",
                     perceptual_loss(device, args.models_dir), device, 0.5, args.batch_size,
                     per_example=bool(args.output))
    if args.output:
        result.update(image_hashes=example_hashes(cache), boxes=cache["boxes"].tolist())
        save_json(result, args.output)
    print(json.dumps(result, indent=2), flush=True)

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch as th

from guided_diffusion_rcdm.feature_datasets import (
    different_image_index, distinct_examples, load_feature_cache, original_for_example,
)
from guided_diffusion_rcdm.ijepa_reconstruction import (
    NativeDecoder, download_assets, initialization_spec, load_image_prior, mask_order,
)
from scripts.ijepa_decoder import (
    comparison_row, feature_spec, prediction_in_context, save_cache, save_comparison_grid, to_image,
)
from scripts.ijepa_diagnostics import example_hashes
from scripts.ijepa_rae import configure, pixels


def native_spec(cache):
    return dict(ijepa=feature_spec(cache), shape=cache["spatial_shape"],
                feature_dim=cache["features"].shape[1], target_normalization=cache["target_normalization"],
                decoder_input="layer_norm_patch_tokens", output="native_rgb_patch14_fixed_imagenet")


def load_native_cache(path):
    cache = load_feature_cache(path)
    if (cache.get("target_normalization") != "layer_norm_patch_tokens"
            or (cache["input_size"], cache["patch_size"]) != (224, 14)):
        raise ValueError("Native cache must use the frozen I-JEPA/14 patch normalization.")
    count, dimension = cache["features"].shape
    height, width = cache["spatial_shape"]
    for name in ("predictor_tokens", "target_tokens"):
        values = cache[name]
        if (values.shape != (count, height * width, dimension) or not values.is_floating_point()
                or not th.isfinite(values).all()):
            raise ValueError(f"Native cache contains invalid {name}.")
    if cache["originals"].shape[-2:] != (224, 224):
        raise ValueError("Native reconstruction needs the aligned 224-pixel originals.")
    mask_order(cache["boxes"], cache["spatial_shape"])
    example_hashes(cache)
    return cache


def native_crops(cache):
    crops = []
    for index, (x0, y0, x1, y1) in enumerate(cache["boxes"].tolist()):
        crops.append(original_for_example(cache, index)[:, y0:y1, x0:x1])
    return th.stack(crops)


def perceptual_loss(device, models_dir):
    import lpips

    th.hub.set_dir(str(Path(models_dir) / "torch_hub"))
    return lpips.LPIPS(net="alex", verbose=False).eval().requires_grad_(False).to(device)


def assets(args):
    data = download_assets(Path(args.models_dir) / "mae_hub")
    path = Path(args.assets)
    if path.exists() and json.loads(path.read_text())["initialization"] != data["initialization"]:
        raise ValueError("Existing native assets belong to another image prior.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    perceptual_loss("cpu", args.models_dir)
    print(f"Verified MAE decoder and LPIPS assets: {path}", flush=True)


def image_errors(prediction, target, perceptual):
    if not th.isfinite(prediction).all():
        raise FloatingPointError("Native decoder produced non-finite RGB.")
    clipped = prediction.clamp(0, 1)
    return dict(
        raw_l1=(prediction - target).abs().mean((1, 2, 3)),
        rgb_mae=(clipped - target).abs().mean((1, 2, 3)),
        lpips=perceptual(clipped * 2 - 1, target * 2 - 1).flatten(),
    )


def averaged(errors, weight):
    result = {name: values.mean().item() for name, values in errors.items()}
    result["score"] = result["raw_l1"] + weight * result["lpips"]
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Native reconstruction metrics are not finite.")
    return result


@th.no_grad()
def evaluate(model, cache, crops, source, perceptual, device, weight, batch_size):
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


def restore_native(path, cache, device, freeze=True):
    checkpoint = th.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["input_spec"] != native_spec(cache)
            or checkpoint["initialization"] != initialization_spec()
            or checkpoint["feature_source"] not in ("target", "predictor", "mixed")):
        raise ValueError("Native checkpoint differs in frozen I-JEPA, geometry, normalization, or image prior.")
    model = NativeDecoder(cache["features"].shape[1], cache["spatial_shape"], checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if freeze:
        model.eval().requires_grad_(False)
    return model.to(device), checkpoint["feature_source"]


def train(args):
    if (min(args.steps, args.batch_size) <= 0 or args.learning_rate <= 0
            or not math.isfinite(args.learning_rate) or args.lpips_weight < 0
            or not math.isfinite(args.lpips_weight)):
        raise ValueError("Native training requires positive finite steps, batch size, and learning rate.")
    if args.feature_source == "mixed" and (args.batch_size < 2 or args.batch_size % 2):
        raise ValueError("Mixed training requires an even batch size of at least two.")
    device = configure(args)
    training = load_native_cache(Path(args.cache_dir) / "train.pt")
    validation = load_native_cache(Path(args.cache_dir) / "val.pt")
    if native_spec(training) != native_spec(validation):
        raise ValueError("Native training and validation have different I-JEPA feature contracts.")
    if set(example_hashes(training)).intersection(example_hashes(validation)):
        raise ValueError("Native training and validation share original image pixels.")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError("Native checkpoints already exist; choose a new experiment directory.")
    crops, held_out = native_crops(training), native_crops(validation)
    if args.init:
        model, _ = restore_native(args.init, training, device, freeze=False)
    else:
        model = load_image_prior(json.loads(Path(args.assets).read_text()), training["features"].shape[1],
                                 training["spatial_shape"]).to(device)
    perceptual = perceptual_loss(device, args.models_dir)
    with th.no_grad():
        mean = crops.float().mean(0, keepdim=True).to(device) / 255
        baseline = averaged(image_errors(mean.expand(len(held_out), -1, -1, -1),
                                         held_out.float().to(device) / 255, perceptual), args.lpips_weight)
    source = "target" if args.feature_source == "target" else "predictor"
    initial = evaluate(model, validation, held_out, source, perceptual, device, args.lpips_weight, args.batch_size)
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
            input_spec=native_spec(training), config=model.config.to_dict(), initialization=initialization_spec(),
            feature_source=args.feature_source, training_step=step, validation=result, baseline=baseline,
            training_examples=len(crops), validation_examples=len(held_out), seed=args.seed,
            lpips_weight=args.lpips_weight, init=args.init, learning_rate=args.learning_rate,
        )

    if initial["score"] < baseline["score"]:
        best, best_step = initial, 0
        save_cache(snapshot(best, best_step), output / "best.pt")
    print(f"Native {args.feature_source}: mean-crop baseline {baseline}; initial {initial}; "
          f"{sum(value.numel() for value in model.parameters()):,} parameters.", flush=True)
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
            raise FloatingPointError(f"Native decoder loss is not finite at step {step}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            result = evaluate(model, validation, held_out, source, perceptual, device,
                              args.lpips_weight, args.batch_size)
            target = evaluate(model, validation, held_out, "target", perceptual, device,
                              args.lpips_weight, args.batch_size) if source != "target" else result
            history.append(dict(step=step, loss=loss.item(), validation=result, target_oracle=target))
            if result["score"] < (baseline["score"] if best is None else best["score"]):
                best, best_step = result, step
                save_cache(snapshot(best, best_step), output / "best.pt")
            save_cache(snapshot(result, step), output / "last.pt")
            print(f"Native {args.feature_source} {step}/{args.steps}: {result}; target {target}; "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    (output / "metrics.json").write_text(json.dumps(
        dict(baseline=baseline, initial=initial, validation=best, best_step=best_step, history=history), indent=2
    ) + "\n")
    if best is None:
        raise RuntimeError("Native decoder did not beat the mean-crop baseline; inspect last.pt.")
    print(f"Native best at step {best_step}: {best}. Saved {output / 'best.pt'}", flush=True)


@th.no_grad()
def sample(args):
    device = configure(args)
    cache = load_native_cache(args.cache)
    model, source = restore_native(args.decoder, cache, device)
    teacher, teacher_source = restore_native(args.target_decoder, cache, device)
    if source not in ("predictor", "mixed") or teacher_source != "target":
        raise ValueError("Native comparison needs a predictor/mixed decoder and a target-trained decoder.")
    crops = native_crops(cache)
    indices = distinct_examples(cache, args.num_images)
    rows, samples, composites = [], [], []
    names = ["target oracle", "via target decoder", "native target", "native predictor",
             "wrong predictor", "shuffled tokens"]
    for index in indices:
        boxes = cache["boxes"][index:index + 1]
        predicted = cache["predictor_tokens"][index:index + 1].float().to(device)
        actual = cache["target_tokens"][index:index + 1].float().to(device)
        wrong = cache["predictor_tokens"][different_image_index(cache, index)].float().to(device)[None]
        permutation = th.randperm(predicted.shape[1], generator=th.Generator().manual_seed(args.seed)).to(device)
        decoded = pixels(th.cat([
            teacher(actual, boxes), teacher(predicted, boxes), model(actual, boxes),
            model(predicted, boxes), model(wrong, boxes), model(predicted[:, permutation], boxes),
        ]))
        composite = prediction_in_context(cache, index, decoded[3])
        rows.append(comparison_row(cache, index, crops.shape[-1]) + [
            to_image(decoded[0]), to_image(decoded[1]), to_image(decoded[2]), to_image(decoded[3]),
            composite, to_image(decoded[4]), to_image(decoded[5]),
        ])
        samples.append(decoded)
        composites.append(th.from_numpy(np.array(composite)).permute(2, 0, 1))
    samples = th.stack(samples)
    mae = (samples.float() / 255 - crops[indices, None].float() / 255).abs().mean((0, 2, 3, 4))
    save_comparison_grid(rows, ["original", "context", "true target", "target oracle", "via target",
                               "native target", "native predictor", "in context", "wrong predictor",
                               "shuffled tokens"], args.output, cell=160)
    result = dict(
        indices=indices, samples=samples, composites=th.stack(composites), sample_labels=names,
        rgb_mae=dict(zip(names, mae.tolist())), input_spec=native_spec(cache),
        primary_uses_hidden_rgb=False, conditioning="ijepa_predictor_native_patch_tokens",
        output="deterministic_native_rgb_crop",
    )
    save_cache(result, Path(args.output).with_suffix(".pt"))
    Path(args.output).with_suffix(".json").write_text(json.dumps(
        {name: value for name, value in result.items() if not isinstance(value, th.Tensor)}, indent=2
    ) + "\n")
    print(f"Native comparison RGB MAE: {result['rgb_mae']}. Saved {args.output}", flush=True)


def parser():
    result = argparse.ArgumentParser(description="Native I-JEPA-to-RGB decoder, initialized from a MAE image prior.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto")
    common.add_argument("--cpu-threads", type=int, default=4)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--assets", default="artifacts/ijepa_native/assets.json")
    common.add_argument("--models-dir", default="artifacts/checkpoints/native_decoder")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("assets", parents=[common])
    command = commands.add_parser("train", parents=[common])
    command.add_argument("--cache-dir", default="artifacts/ijepa_rae/features")
    command.add_argument("--out-dir", required=True)
    command.add_argument("--feature-source", choices=("target", "predictor", "mixed"), default="target")
    command.add_argument("--steps", type=int, default=1000)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--lpips-weight", type=float, default=0.5)
    command.add_argument("--init")
    command = commands.add_parser("sample", parents=[common])
    command.add_argument("--cache", default="artifacts/ijepa_rae/features/val.pt")
    command.add_argument("--decoder", required=True)
    command.add_argument("--target-decoder", required=True)
    command.add_argument("--num-images", type=int, default=8)
    command.add_argument("--output", required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    {"assets": assets, "train": train, "sample": sample}[args.command](args)

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F

from guided_diffusion_rcdm import dist_util
from guided_diffusion_rcdm.feature_datasets import (
    different_image_index, distinct_examples, load_feature_cache, original_for_example,
)
from guided_diffusion_rcdm.ijepa import (
    load_ijepa_with_target, make_rectangle_masks, predict_tokens, target_tokens,
)
from guided_diffusion_rcdm.rae import (
    SpatialAdapter, download_assets, load_decoder, load_encoder, rae_spec,
)
from scripts.ijepa_decoder import (
    comparison_row, feature_spec, hidden_crop, prediction_in_context, save_cache,
    save_comparison_grid, to_image,
)


def configure(args):
    if args.cpu_threads <= 0 or getattr(args, "num_images", 1) <= 0:
        raise ValueError("CPU thread and sampling counts must be positive.")
    th.set_num_threads(args.cpu_threads)
    th.manual_seed(args.seed)
    return dist_util.set_device(args.device)


def assets(args):
    data = download_assets(args.models_dir)
    path = Path(args.assets)
    if path.exists() and json.loads(path.read_text())["rae_spec"] != data["rae_spec"]:
        raise ValueError("Existing assets belong to another RAE feature family.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Verified pinned assets: {path}", flush=True)


def read_assets(path):
    data = json.loads(Path(path).read_text())
    if data["rae_spec"] != rae_spec():
        raise ValueError("RAE asset manifest differs from the pinned encoder/decoder.")
    return data


def pixels(images):
    if not th.isfinite(images).all():
        raise FloatingPointError("Cannot render non-finite RGB predictions.")
    return (images.cpu().clamp(0, 1) * 255).round().to(th.uint8)


@th.no_grad()
def oracle(args):
    device = configure(args)
    data = read_assets(args.assets)
    cache = load_feature_cache(args.cache)
    indices = distinct_examples(cache, args.num_images)
    if not indices:
        raise ValueError("Oracle requires at least one cached image.")
    crops = th.stack([th.from_numpy(np.array(hidden_crop(cache, index, 256))).permute(2, 0, 1)
                      for index in indices])
    encoder = load_encoder(data["encoder_dir"], device)
    decoder = load_decoder(args.rae_repo, data["decoder_path"], data["encoder_dir"], device)
    started = time.monotonic()
    tokens = encoder(crops.float().to(device) / 255)
    decoded = decoder(tokens).cpu()
    mae = (decoded - crops.float() / 255).abs().mean().item()
    restored = pixels(decoded)
    rows = [comparison_row(cache, index, 256) + [
        to_image(restored[row]), prediction_in_context(cache, index, restored[row])
    ] for row, index in enumerate(indices)]
    save_comparison_grid(rows, ["original", "context", "true target", "DINOv2 oracle", "oracle context"],
                         args.output, cell=160)
    save_cache(dict(indices=indices, tokens=tokens.cpu(), samples=restored, rae_spec=rae_spec(),
                    ground_truth_available=True, conditioning="true_hidden_rgb", rgb_mae=mae),
               Path(args.output).with_suffix(".pt"))
    print(f"RAE oracle RGB MAE: {mae:.4f}; elapsed {time.monotonic() - started:.1f}s. "
          f"Saved {args.output}", flush=True)


def spatial_spec(cache):
    return dict(ijepa=feature_spec(cache), shape=cache["spatial_shape"],
                target_normalization=cache["target_normalization"], rae=cache["rae_spec"])


def load_spatial_cache(path):
    cache = load_feature_cache(path)
    if cache.get("rae_spec") != rae_spec() or cache.get("target_normalization") != "layer_norm_patch_tokens":
        raise ValueError("Spatial cache uses incompatible RAE or I-JEPA normalization.")
    height, width = cache["spatial_shape"]
    count, dimension = cache["features"].shape
    for name in ("predictor_tokens", "target_tokens"):
        value = cache[name]
        if (value.shape != (count, height * width, dimension)
                or not value.is_floating_point() or not th.isfinite(value).all()):
            raise ValueError(f"Invalid cached {name}.")
    if cache["dino_tokens"].shape != (count, 256, 768) or not th.isfinite(cache["dino_tokens"]).all():
        raise ValueError("Cached DINOv2 crop tokens are invalid.")
    if cache["targets"].shape[2:] != (256, 256):
        raise ValueError("RAE targets must use the released decoder's 256-pixel resolution.")
    return cache


@th.no_grad()
def prepare(args):
    if min(args.batch_size, args.train_limit, args.val_limit) <= 0:
        raise ValueError("Feature extraction counts must be positive.")
    device = configure(args)
    data = read_assets(args.assets)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{split}.pt").exists() for split in ("train", "val")):
        raise FileExistsError("Spatial caches already exist; choose a new output directory.")
    encoder, predictor, teacher, epoch = load_ijepa_with_target(args.ijepa_repo, args.checkpoint, device)
    dino = load_encoder(data["encoder_dir"], device)
    mean = th.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = th.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    started, training_hashes = time.monotonic(), set()
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        source = load_feature_cache(Path(args.cache_dir) / f"{split}.pt")
        if (source["input_size"], source["patch_size"], source["block_size"], source["ijepa_epoch"],
                Path(source["ijepa_checkpoint"]).name) != (224, 14, 7, epoch, Path(args.checkpoint).name):
            raise ValueError("Source cache differs from the loaded official I-JEPA backbone or pilot mask.")
        indices = distinct_examples(source, limit)
        if len(indices) < 2:
            raise ValueError("Each spatial split requires at least two distinct original photographs.")
        originals = th.stack([original_for_example(source, index) for index in indices])
        crops = th.stack([th.from_numpy(np.array(hidden_crop(source, index, 256))).permute(2, 0, 1)
                          for index in indices])
        boxes = source["boxes"][indices]
        paths = [source["paths"][index] for index in indices]
        hashes = [source["image_hashes"][source["image_indices"][index].item()] for index in indices]
        if split == "train":
            training_hashes.update(hashes)
        elif training_hashes.intersection(hashes):
            raise ValueError("Spatial train and validation share original image pixels.")
        contexts, hidden = [], []
        for x0, y0, x1, y1 in boxes.tolist():
            if x0 % 14 or y0 % 14 or (x1 - x0, y1 - y0) != (98, 98):
                raise ValueError("This pilot requires aligned 7x7 target rectangles.")
            context, target = make_rectangle_masks(y0 // 14, x0 // 14, 7, 7)
            contexts.append(context)
            hidden.append(target)
        contexts, hidden = th.stack(contexts), th.stack(hidden)
        spec = feature_spec(source)
        del source
        predicted, actual, dino_tokens = [], [], []
        pooled = []
        for start in range(0, len(indices), args.batch_size):
            end = min(start + args.batch_size, len(indices))
            image = (originals[start:end].float().to(device) / 255 - mean) / std
            context, target = contexts[start:end].to(device), hidden[start:end].to(device)
            local = th.arange(end - start, device=device)
            pred = predict_tokens(encoder, predictor, image, context, target)
            truth = target_tokens(teacher, image, target, local)
            if split == "train" and start == 0:
                changed = image.clone()
                x0, y0, x1, y1 = boxes[0].tolist()
                changed[0, :, y0:y1, x0:x1] = 0
                if not th.equal(pred, predict_tokens(encoder, predictor, changed, context, target)):
                    raise AssertionError("Hidden pixels changed real predictor patch tokens.")
                if th.equal(truth[0], target_tokens(teacher, changed, target, local)[0]):
                    raise AssertionError("Target-encoder oracle did not respond to hidden pixels.")
                print("Verified real patch tokens: predictor excludes hidden pixels; target oracle sees them.",
                      flush=True)
            predicted.append(pred.cpu().half())
            actual.append(truth.cpu().half())
            pooled.append(pred.mean(1).cpu())
            dino_tokens.append(dino(crops[start:end].float().to(device) / 255).cpu().half())
            if end % 64 == 0 or end == len(indices):
                print(f"{split}: {end}/{len(indices)}, elapsed {time.monotonic() - started:.1f}s", flush=True)
        cache = dict(
            version=2, features=th.cat(pooled), targets=crops, originals=originals, boxes=boxes,
            image_indices=th.arange(len(indices)), image_paths=paths, paths=paths, image_hashes=hashes,
            predictor_tokens=th.cat(predicted), target_tokens=th.cat(actual), dino_tokens=th.cat(dino_tokens),
            spatial_shape=[7, 7], rae_spec=rae_spec(), target_normalization="layer_norm_patch_tokens",
            conditioning=spec["conditioning"], input_size=224, patch_size=14, block_size=7,
            ijepa_epoch=epoch, ijepa_checkpoint=str(Path(args.checkpoint).resolve()), seed=args.seed,
            source_indices=indices,
        )
        save_cache(cache, output / f"{split}.pt")
        print(f"Saved spatial cache {output / f'{split}.pt'}", flush=True)


@th.no_grad()
def metrics(model, cache, source, device, batch_size=16):
    squared, cosine, wrong_squared, count = 0.0, 0.0, 0.0, 0
    features, targets = cache[f"{source}_tokens"], cache["dino_tokens"]
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        x, y = features[start:end].float().to(device), targets[start:end].float().to(device)
        predicted = model(x, cache["spatial_shape"])
        wrong_indices = [different_image_index(cache, index) for index in range(start, end)]
        wrong = model(features[wrong_indices].float().to(device), cache["spatial_shape"])
        squared += (((predicted - y) / model.output_scale).square().mean().item()) * (end - start)
        wrong_squared += (((wrong - y) / model.output_scale).square().mean().item()) * (end - start)
        cosine += F.cosine_similarity(predicted, y, dim=-1).mean().item() * (end - start)
        count += end - start
    return dict(mse=squared / count, cosine=cosine / count, wrong_mse=wrong_squared / count)


def train(args):
    if args.ridge_alpha <= 0 or not math.isfinite(args.ridge_alpha):
        raise ValueError("Ridge alpha must be finite and positive.")
    device = configure(args)
    training = load_spatial_cache(Path(args.cache_dir) / "train.pt")
    validation = load_spatial_cache(Path(args.cache_dir) / "val.pt")
    if spatial_spec(training) != spatial_spec(validation):
        raise ValueError("Spatial train/validation use different frozen models or grids.")
    if set(training["image_hashes"]).intersection(validation["image_hashes"]):
        raise ValueError("Spatial train/validation share original images.")
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "last.pt").exists():
        raise FileExistsError("Adapter checkpoints already exist; use a new experiment directory.")
    inputs = training[f"{args.feature_source}_tokens"]
    model = SpatialAdapter(inputs.shape[-1], 768)
    model.fit_statistics(inputs, training["dino_tokens"], training["spatial_shape"])
    baseline = metrics(model.to(device), validation, args.feature_source, device)
    model.cpu()
    started = time.monotonic()
    model.fit_ridge(inputs, training["dino_tokens"], training["spatial_shape"], args.ridge_alpha)
    result = metrics(model.to(device), validation, args.feature_source, device)
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Spatial validation metrics are not finite.")
    payload = dict(
        state_dict={name: value.cpu() for name, value in model.state_dict().items()},
        input_spec=spatial_spec(training), feature_source=args.feature_source,
        input_dim=inputs.shape[-1], output_dim=768, hidden_dim=0, ridge_alpha=args.ridge_alpha,
        training_examples=len(inputs), validation_examples=len(validation["features"]),
        baseline=baseline, validation=result, seed=args.seed,
    )
    save_cache(payload, output / "last.pt")
    (output / "metrics.json").write_text(json.dumps(dict(baseline=baseline, validation=result), indent=2) + "\n")
    if result["mse"] >= baseline["mse"]:
        raise RuntimeError("Spatial ridge did not improve the mean-token baseline; inspect last.pt.")
    save_cache(payload, output / "best.pt")
    print(f"{args.feature_source} ridge: baseline {baseline}; validation {result}; "
          f"elapsed {time.monotonic() - started:.1f}s. Saved {output / 'best.pt'}", flush=True)


def restore_adapter(path, cache, device):
    checkpoint = th.load(path, weights_only=True, map_location="cpu")
    if checkpoint["input_spec"] != spatial_spec(cache) or checkpoint["input_dim"] != cache["features"].shape[-1]:
        raise ValueError("Spatial adapter checkpoint differs from the frozen model, normalization, or geometry.")
    if checkpoint["feature_source"] not in ("predictor", "target"):
        raise ValueError("Invalid spatial adapter feature source.")
    model = SpatialAdapter(checkpoint["input_dim"], checkpoint["output_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().requires_grad_(False).to(device), checkpoint["feature_source"]


@th.no_grad()
def sample(args):
    device = configure(args)
    data = read_assets(args.assets)
    cache = load_spatial_cache(args.cache)
    predictor_adapter, source = restore_adapter(args.adapter, cache, device)
    target_adapter, target_source = restore_adapter(args.target_adapter, cache, device)
    if (source, target_source) != ("predictor", "target"):
        raise ValueError("Comparison requires a predictor adapter and a target-trained oracle adapter.")
    decoder = load_decoder(args.rae_repo, data["decoder_path"], data["encoder_dir"], device)
    indices = distinct_examples(cache, args.num_images)
    rows, samples, composites = [], [], []
    labels = ["original", "context", "true target", "DINOv2 oracle", "IJEPA target",
              "via target", "predictor", "in context", "wrong predictor", "shuffled tokens"]
    started = time.monotonic()
    for index in indices:
        pred = cache["predictor_tokens"][index:index + 1].float().to(device)
        actual = cache["target_tokens"][index:index + 1].float().to(device)
        other = different_image_index(cache, index)
        wrong = cache["predictor_tokens"][other:other + 1].float().to(device)
        permutation = th.randperm(pred.shape[1], generator=th.Generator().manual_seed(args.seed))
        tokens = th.cat([
            cache["dino_tokens"][index:index + 1].float().to(device),
            target_adapter(actual, cache["spatial_shape"]), target_adapter(pred, cache["spatial_shape"]),
            predictor_adapter(pred, cache["spatial_shape"]), predictor_adapter(wrong, cache["spatial_shape"]),
            predictor_adapter(pred[:, permutation.to(device)], cache["spatial_shape"]),
        ])
        decoded = pixels(decoder(tokens))
        composite = prediction_in_context(cache, index, decoded[3])
        rows.append(comparison_row(cache, index, 256) + [
            to_image(decoded[0]), to_image(decoded[1]), to_image(decoded[2]), to_image(decoded[3]),
            composite, to_image(decoded[4]), to_image(decoded[5]),
        ])
        samples.append(decoded)
        composites.append(th.from_numpy(np.array(composite)).permute(2, 0, 1))
        print(f"Decoded spatial example {index}; elapsed {time.monotonic() - started:.1f}s", flush=True)
    save_comparison_grid(rows, labels, args.output, cell=160)
    save_cache(dict(indices=indices, samples=th.stack(samples), composites=th.stack(composites),
                    sample_labels=["DINOv2 oracle", "IJEPA target", "via target", "predictor",
                                   "wrong predictor", "shuffled tokens"],
                    input_spec=spatial_spec(cache), ground_truth_available=True,
                    conditioning="ijepa_predictor_patch_tokens"),
               Path(args.output).with_suffix(".pt"))
    print(f"Saved {args.output}", flush=True)


def parser():
    result = argparse.ArgumentParser(description="Frozen RAE pilot for spatial I-JEPA predictor tokens.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto")
    common.add_argument("--cpu-threads", type=int, default=4)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--assets", default="artifacts/ijepa_rae/assets.json")
    common.add_argument("--rae-repo", default="artifacts/rae")
    commands = result.add_subparsers(dest="command", required=True)
    command = commands.add_parser("assets", parents=[common])
    command.add_argument("--models-dir", default="artifacts/checkpoints/rae_hub")
    for name in ("oracle", "sample"):
        command = commands.add_parser(name, parents=[common])
        command.add_argument("--cache", required=True)
        command.add_argument("--num-images", type=int, default=4)
        command.add_argument("--output", required=True)
        if name == "sample":
            command.add_argument("--adapter", required=True)
            command.add_argument("--target-adapter", required=True)
    command = commands.add_parser("prepare", parents=[common])
    command.add_argument("--cache-dir", default="artifacts/ijepa_features_320_m3")
    command.add_argument("--out-dir", default="artifacts/ijepa_rae/features")
    command.add_argument("--ijepa-repo", default="artifacts/ijepa")
    command.add_argument("--checkpoint", default="artifacts/checkpoints/IN1K-vit.h.14-300e.pth.tar")
    command.add_argument("--train-limit", type=int, default=2048)
    command.add_argument("--val-limit", type=int, default=64)
    command.add_argument("--batch-size", type=int, default=4)
    command = commands.add_parser("train", parents=[common])
    command.add_argument("--cache-dir", default="artifacts/ijepa_rae/features")
    command.add_argument("--out-dir", required=True)
    command.add_argument("--feature-source", choices=("predictor", "target"), default="predictor")
    command.add_argument("--ridge-alpha", type=float, default=0.1)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    {"assets": assets, "oracle": oracle, "prepare": prepare, "train": train,
     "sample": sample}[arguments.command](arguments)

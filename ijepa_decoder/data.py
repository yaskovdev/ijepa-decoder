import hashlib
import json
from pathlib import Path
import time

import torch as th

from .encoder import load_ijepa, make_masks, predict_tokens, target_tokens
from .io import configure, image_pixels, normalized, save_cache
from .model import mask_order


def input_spec(cache):
    backbone = {key: cache[key] for key in (
        "conditioning", "input_size", "patch_size", "block_size", "ijepa_epoch",
    )}
    backbone["checkpoint"] = Path(cache["ijepa_checkpoint"]).name
    return dict(ijepa=backbone, shape=cache["spatial_shape"],
                feature_dim=cache["predictor_tokens"].shape[-1],
                target_normalization=cache["target_normalization"],
                decoder_input="layer_norm_patch_tokens", output="native_rgb_patch14_fixed_imagenet")


def example_hashes(cache):
    return [cache["image_hashes"][index] for index in cache["image_indices"].tolist()]


def load_cache(path):
    cache = th.load(path, map_location="cpu", weights_only=True)
    if (cache["target_normalization"] != "layer_norm_patch_tokens"
            or (cache["input_size"], cache["patch_size"]) != (224, 14)):
        raise ValueError("Cache has incompatible I-JEPA normalization or input geometry.")
    predicted, actual = cache["predictor_tokens"], cache["target_tokens"]
    height, width = cache["spatial_shape"]
    if (not isinstance(predicted, th.Tensor) or predicted.ndim != 3 or len(predicted) == 0
            or predicted.shape[1] != height * width or not predicted.is_floating_point()
            or not th.isfinite(predicted).all() or not isinstance(actual, th.Tensor)
            or actual.shape != predicted.shape or not actual.is_floating_point()
            or not th.isfinite(actual).all()):
        raise ValueError("Cache needs paired finite predictor/target patch tokens.")
    count = len(predicted)
    originals, indices = cache["originals"], cache["image_indices"]
    if (not isinstance(originals, th.Tensor) or originals.ndim != 4
            or originals.shape[1:] != (3, 224, 224) or originals.dtype != th.uint8
            or not isinstance(indices, th.Tensor) or indices.dtype != th.long
            or indices.shape != (count,) or indices.min() < 0 or indices.max() >= len(originals)
            or len(cache["image_hashes"]) != len(originals)):
        raise ValueError("Cache must map every token sequence to an original uint8 RGB image.")
    if cache["boxes"].shape != (count, 4):
        raise ValueError("Cache needs one box per patch-token sequence.")
    mask_order(cache["boxes"], cache["spatial_shape"])
    return cache


def original_for_example(cache, index):
    return cache["originals"][cache["image_indices"][index].item()]


def crop_targets(cache):
    return th.stack([original_for_example(cache, index)[:, y0:y1, x0:x1]
                     for index, (x0, y0, x1, y1) in enumerate(cache["boxes"].tolist())])


def read_manifest(path):
    manifest = json.loads(Path(path).read_text())
    seen = {}
    for split in ("train", "val"):
        entries = manifest["splits"][split]
        if len(entries) < 2:
            raise ValueError("Each manifest split needs at least two photographs.")
        seen[split] = set()
        for entry in entries:
            path = Path(entry["path"])
            if path.is_absolute() or ".." in path.parts or len(entry["sha256"]) != 64:
                raise ValueError("Manifest image paths must be relative and carry pixel hashes.")
            make_masks(entry["box"])
            if entry["sha256"] in seen[split]:
                raise ValueError("Manifest repeats an original photograph.")
            seen[split].add(entry["sha256"])
    if seen["train"].intersection(seen["val"]):
        raise ValueError("Manifest train/validation share image pixels.")
    return manifest


@th.no_grad()
def prepare(args):
    if args.batch_size <= 0 or (args.limit is not None and args.limit < 2):
        raise ValueError("Preparation needs positive batch size and at least two images per split.")
    device = configure(args)
    manifest = read_manifest(args.manifest)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{split}.pt").exists() for split in ("train", "val")):
        raise FileExistsError("Feature caches exist; choose a new output directory.")
    encoder, predictor, teacher, epoch = load_ijepa(
        args.ijepa_repo, args.ijepa_checkpoint, device, include_target=True,
    )
    if epoch != manifest["ijepa_epoch"]:
        raise ValueError("Loaded I-JEPA epoch differs from the reproduction manifest.")
    started = time.monotonic()
    for split in ("train", "val"):
        entries = manifest["splits"][split][:args.limit]
        originals, boxes, contexts, hidden = [], [], [], []
        for entry in entries:
            original = image_pixels(Path(args.data_dir) / entry["path"])
            digest = hashlib.sha256(original.permute(1, 2, 0).contiguous().numpy().tobytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Image pixels differ from the published manifest: {entry['path']}")
            context, target = make_masks(entry["box"])
            originals.append(original)
            boxes.append(entry["box"])
            contexts.append(context)
            hidden.append(target)
        originals, boxes = th.stack(originals), th.tensor(boxes)
        contexts, hidden = th.stack(contexts), th.stack(hidden)
        predicted, actual = [], []
        for start in range(0, len(entries), args.batch_size):
            end = min(start + args.batch_size, len(entries))
            image = normalized(originals[start:end], device)
            context, target = contexts[start:end].to(device), hidden[start:end].to(device)
            prediction = predict_tokens(encoder, predictor, image, context, target)
            truth = target_tokens(teacher, image, target)
            if split == "train" and start == 0:
                changed = image.clone()
                x0, y0, x1, y1 = boxes[0].tolist()
                changed[0, :, y0:y1, x0:x1] = 0
                if not th.equal(prediction, predict_tokens(encoder, predictor, changed, context, target)):
                    raise AssertionError("Hidden pixels changed predictor tokens.")
                if th.equal(truth[0], target_tokens(teacher, changed, target)[0]):
                    raise AssertionError("Target-encoder oracle did not respond to hidden pixels.")
                print("Verified hidden-pixel isolation of predictor tokens.", flush=True)
            predicted.append(prediction.cpu().half())
            actual.append(truth.cpu().half())
            if end % 64 == 0 or end == len(entries):
                print(f"{split}: {end}/{len(entries)}, {time.monotonic() - started:.1f}s", flush=True)
        cache = dict(
            predictor_tokens=th.cat(predicted), target_tokens=th.cat(actual),
            originals=originals, boxes=boxes, image_indices=th.arange(len(entries)),
            image_hashes=[entry["sha256"] for entry in entries],
            paths=[entry["path"] for entry in entries], spatial_shape=[7, 7],
            conditioning="ijepa_predictor", input_size=224, patch_size=14, block_size=7,
            ijepa_epoch=epoch, ijepa_checkpoint=str(Path(args.ijepa_checkpoint).resolve()),
            target_normalization="layer_norm_patch_tokens", seed=args.seed,
        )
        save_cache(cache, output / f"{split}.pt")
        print(f"Saved {output / f'{split}.pt'}", flush=True)

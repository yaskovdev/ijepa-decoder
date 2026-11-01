import hashlib
import json
from pathlib import Path
import random
import time

import torch as th

from .encoder import load_ijepa, make_masks, predict_tokens, target_tokens
from .io import configure, file_sha256, image_pixels, normalized, save_cache
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


def finite_tokens(tokens):
    return all(th.isfinite(tokens[start:start + 128]).all().item()
               for start in range(0, len(tokens), 128))


def load_cache(path):
    cache = th.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (cache["target_normalization"] != "layer_norm_patch_tokens"
            or (cache["input_size"], cache["patch_size"]) != (224, 14)):
        raise ValueError("Cache has incompatible I-JEPA normalization or input geometry.")
    predicted, actual = cache["predictor_tokens"], cache["target_tokens"]
    height, width = cache["spatial_shape"]
    if (not isinstance(predicted, th.Tensor) or predicted.ndim != 3 or len(predicted) == 0
            or predicted.shape[1] != height * width or not predicted.is_floating_point()
            or not finite_tokens(predicted) or not isinstance(actual, th.Tensor)
            or actual.shape != predicted.shape or not actual.is_floating_point()
            or not finite_tokens(actual)):
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


def entry_boxes(entry):
    if ("box" in entry) == ("boxes" in entry):
        raise ValueError("Manifest photograph must specify either box or boxes.")
    boxes = [entry["box"]] if "box" in entry else entry["boxes"]
    if not boxes or len({tuple(box) for box in boxes}) != len(boxes):
        raise ValueError("Manifest masks must be nonempty and distinct per photograph.")
    for box in boxes:
        make_masks(box)
    return boxes


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
            if (path.is_absolute() or ".." in path.parts or len(entry["sha256"]) != 64
                    or any(char not in "0123456789abcdef" for char in entry["sha256"])):
                raise ValueError("Manifest image paths must be relative and carry pixel hashes.")
            entry_boxes(entry)
            if entry["sha256"] in seen[split]:
                raise ValueError("Manifest repeats an original photograph.")
            seen[split].add(entry["sha256"])
    if seen["train"].intersection(seen["val"]):
        raise ValueError("Manifest train/validation share image pixels.")
    return manifest


def pixel_hash(original):
    return hashlib.sha256(original.permute(1, 2, 0).contiguous().numpy().tobytes()).hexdigest()


def generate_manifest(args):
    if not 1 <= args.masks_per_image <= 100 or args.validation_images < 2:
        raise ValueError("Use 1-100 training masks and at least two validation photographs.")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Manifest exists; choose a new output path.")
    reference = read_manifest(args.reference_manifest)
    root, rng = Path(args.data_dir), random.Random(args.seed)
    photographs, duplicate_counts = {}, {}
    for split in ("train", "val"):
        photographs[split], seen, duplicates = [], set(), 0
        for path in sorted((root / split).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            digest = pixel_hash(image_pixels(path))
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            photographs[split].append(dict(path=path.relative_to(root).as_posix(), sha256=digest))
        duplicate_counts[split] = duplicates
        print(f"{split}: {len(photographs[split])} distinct photographs; {duplicates} pixel duplicates.", flush=True)
    validation_hashes = {entry["sha256"] for entry in photographs["val"]}
    training = [entry for entry in photographs["train"] if entry["sha256"] not in validation_hashes]
    duplicate_counts["train_val_overlap"] = len(photographs["train"]) - len(training)
    print(f"Removed {duplicate_counts['train_val_overlap']} training/validation pixel duplicates.", flush=True)
    referenced = {split: {entry["path"]: entry for entry in reference["splits"][split]} for split in ("train", "val")}
    available = {split: {entry["path"]: entry for entry in photographs[split]} for split in ("train", "val")}
    for split in ("train", "val"):
        for path, entry in referenced[split].items():
            if path not in available[split] or available[split][path]["sha256"] != entry["sha256"]:
                raise ValueError(f"Reference photograph is missing or changed: {path}")
    if (args.validation_images < len(reference["splits"]["val"])
            or args.validation_images > len(photographs["val"])):
        raise ValueError("Validation size must include every reference image and fit the distinct validation split.")
    validation = [available["val"][entry["path"]].copy() for entry in reference["splits"]["val"]]
    extras = [entry for entry in photographs["val"] if entry["path"] not in referenced["val"]]
    rng.shuffle(extras)
    validation.extend(extras[:args.validation_images - len(validation)])
    rng.shuffle(training)
    positions = [[left * 14, top * 14, (left + 7) * 14, (top + 7) * 14]
                 for top in range(10) for left in range(10)]
    for split, entries in (("train", training), ("val", validation)):
        for entry in entries:
            boxes = entry_boxes(referenced[split][entry["path"]]) if entry["path"] in referenced[split] else []
            count = args.masks_per_image if split == "train" else max(1, len(boxes))
            if len(boxes) > count:
                raise ValueError("Requested masks cannot discard the reference training masks.")
            remaining = [box for box in positions if box not in boxes]
            entry["boxes"] = boxes + rng.sample(remaining, count - len(boxes))
    manifest = {key: value for key, value in reference.items() if key != "splits"}
    manifest.update(format_version=2, seed=args.seed, reference_sha256=file_sha256(args.reference_manifest),
                    duplicate_counts=duplicate_counts, splits=dict(train=training, val=validation))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    read_manifest(temporary)
    temporary.replace(output)
    print(f"Saved {output}: {len(training)} training photographs / "
          f"{sum(len(entry['boxes']) for entry in training)} masks; {len(validation)} validation photographs.",
          flush=True)


@th.no_grad()
def extract_cache(entries, args, encoder, predictor, teacher, epoch, device, provenance, isolation):
    originals, boxes, contexts, hidden, counts = [], [], [], [], []
    for entry in entries:
        original = image_pixels(Path(args.data_dir) / entry["path"])
        if pixel_hash(original) != entry["sha256"]:
            raise ValueError(f"Image pixels differ from the published manifest: {entry['path']}")
        originals.append(original)
        masks = entry_boxes(entry)
        counts.append(len(masks))
        for box in masks:
            context, target = make_masks(box)
            boxes.append(box)
            contexts.append(context)
            hidden.append(target)
    originals, boxes = th.stack(originals), th.tensor(boxes)
    contexts, hidden = th.stack(contexts), th.stack(hidden)
    image_indices = th.arange(len(entries)).repeat_interleave(th.tensor(counts))
    predicted, actual, offset = [], [], 0
    for start in range(0, len(entries), args.batch_size):
        end = min(start + args.batch_size, len(entries))
        example_end = offset + sum(counts[start:end])
        image = normalized(originals[start:end], device)
        local_indices = (image_indices[offset:example_end] - start).to(device)
        context, target = contexts[offset:example_end].to(device), hidden[offset:example_end].to(device)
        prediction = predict_tokens(encoder, predictor, image[local_indices], context, target)
        truth = target_tokens(teacher, image, target, local_indices)
        if isolation and start == 0:
            changed = image[:1].clone()
            x0, y0, x1, y1 = boxes[0].tolist()
            changed[:, :, y0:y1, x0:x1] = 0
            before = predict_tokens(encoder, predictor, image[:1], context[:1], target[:1])
            after = predict_tokens(encoder, predictor, changed, context[:1], target[:1])
            if not th.equal(before, after):
                raise AssertionError("Hidden pixels changed predictor tokens.")
            if th.equal(target_tokens(teacher, image[:1], target[:1]),
                        target_tokens(teacher, changed, target[:1])):
                raise AssertionError("Target-encoder oracle did not respond to hidden pixels.")
            print("Verified hidden-pixel isolation of predictor tokens.", flush=True)
        predicted.append(prediction.cpu().half())
        actual.append(truth.cpu().half())
        offset = example_end
    return dict(
        predictor_tokens=th.cat(predicted), target_tokens=th.cat(actual),
        originals=originals, boxes=boxes, image_indices=image_indices,
        image_hashes=[entry["sha256"] for entry in entries], paths=[entry["path"] for entry in entries],
        spatial_shape=[7, 7], conditioning="ijepa_predictor", input_size=224, patch_size=14, block_size=7,
        ijepa_epoch=epoch, ijepa_checkpoint=str(Path(args.ijepa_checkpoint).resolve()),
        target_normalization="layer_norm_patch_tokens", seed=args.seed, preparation=provenance,
    )


def verify_cached_entries(cache, entries, provenance):
    boxes = [box for entry in entries for box in entry_boxes(entry)]
    indices = th.arange(len(entries)).repeat_interleave(th.tensor([len(entry_boxes(entry)) for entry in entries]))
    if (cache.get("preparation") != provenance or cache["paths"] != [entry["path"] for entry in entries]
            or cache["image_hashes"] != [entry["sha256"] for entry in entries]
            or not th.equal(cache["boxes"], th.tensor(boxes))
            or not th.equal(cache["image_indices"], indices)):
        raise ValueError("Partial feature cache differs from the manifest or preparation settings.")
    if [pixel_hash(original) for original in cache["originals"]] != cache["image_hashes"]:
        raise ValueError("Partial feature cache original pixels differ from the manifest.")


def merge_parts(paths, entries, provenance):
    first = load_cache(paths[0])
    examples, photographs = sum(len(entry_boxes(entry)) for entry in entries), len(entries)
    result = {key: value for key, value in first.items()
              if key not in ("predictor_tokens", "target_tokens", "originals", "boxes", "image_indices",
                             "image_hashes", "paths")}
    for name in ("predictor_tokens", "target_tokens", "boxes", "image_indices"):
        result[name] = th.empty((examples, *first[name].shape[1:]), dtype=first[name].dtype)
    result["originals"] = th.empty((photographs, 3, 224, 224), dtype=th.uint8)
    result["image_hashes"], result["paths"] = [], []
    example_offset, image_offset = 0, 0
    for path in paths:
        cache = load_cache(path)
        count, images = len(cache["predictor_tokens"]), len(cache["originals"])
        verify_cached_entries(cache, entries[image_offset:image_offset + images], provenance)
        if input_spec(cache) != input_spec(first):
            raise ValueError("Preparation shards have different feature contracts.")
        for name in ("predictor_tokens", "target_tokens", "boxes"):
            result[name][example_offset:example_offset + count].copy_(cache[name])
        result["image_indices"][example_offset:example_offset + count] = cache["image_indices"] + image_offset
        result["originals"][image_offset:image_offset + images].copy_(cache["originals"])
        result["image_hashes"].extend(cache["image_hashes"])
        result["paths"].extend(cache["paths"])
        example_offset += count
        image_offset += images
    if (example_offset, image_offset) != (examples, photographs):
        raise ValueError("Preparation shards do not cover the complete manifest split.")
    return result


@th.no_grad()
def prepare(args):
    if (args.batch_size <= 0 or (args.limit is not None and args.limit < 2)
            or args.shard_size < 0 or (args.resume and not args.shard_size)):
        raise ValueError("Preparation needs positive batches, at least two images, and shards for resume.")
    device = configure(args)
    manifest = read_manifest(args.manifest)
    provenance = dict(manifest_sha256=file_sha256(args.manifest), checkpoint_sha256=file_sha256(args.ijepa_checkpoint),
                      batch_size=args.batch_size, shard_size=args.shard_size, limit=args.limit, seed=args.seed,
                      device=device.type)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"{split}.pt").exists() for split in ("train", "val")) and not args.resume:
        raise FileExistsError("Feature caches exist; choose a new output directory.")
    parts = output / ".parts"
    completed = set()
    if args.shard_size:
        marker = parts / "run.pt"
        if parts.exists() and not args.resume:
            raise FileExistsError("Partial preparation exists; use --resume or a new directory.")
        if marker.exists() and th.load(marker, weights_only=True) != provenance:
            raise ValueError("Preparation resume settings differ from the original run.")
        if parts.exists() and not marker.exists():
            raise ValueError("Partial preparation is missing its provenance marker.")
        for split in ("train", "val"):
            if (output / f"{split}.pt").exists():
                verify_cached_entries(load_cache(output / f"{split}.pt"),
                                      manifest["splits"][split][:args.limit], provenance)
                completed.add(split)
        if len(completed) == 2 and not parts.exists():
            print("Both feature caches are already complete and verified.", flush=True)
            return
        save_cache(provenance, marker)
    encoder, predictor, teacher, epoch = load_ijepa(
        args.ijepa_repo, args.ijepa_checkpoint, device, include_target=True,
    )
    if epoch != manifest["ijepa_epoch"]:
        raise ValueError("Loaded I-JEPA epoch differs from the reproduction manifest.")
    started, pending, part_paths = time.monotonic(), {}, []
    for split in ("train", "val"):
        entries = manifest["splits"][split][:args.limit]
        if split in completed:
            if args.shard_size:
                part_paths.extend(path for path in (
                    parts / f"{split}-{index:05d}.pt"
                    for index in range((len(entries) + args.shard_size - 1) // args.shard_size)
                ) if path.exists())
            print(f"Verified completed {split} cache.", flush=True)
            continue
        size = args.shard_size or len(entries)
        paths = []
        for start in range(0, len(entries), size):
            subset = entries[start:start + size]
            path = parts / f"{split}-{start // size:05d}.pt" if args.shard_size else output / f"{split}.pt"
            if path.exists():
                verify_cached_entries(load_cache(path), subset, provenance)
                print(f"Verified existing {path}", flush=True)
            else:
                cache = extract_cache(subset, args, encoder, predictor, teacher, epoch, device, provenance,
                                      isolation=split == "train" and start == 0)
                save_cache(cache, path)
                del cache
            paths.append(path)
            if args.shard_size:
                part_paths.append(path)
            print(f"{split}: {min(start + size, len(entries))}/{len(entries)} photographs, "
                  f"{time.monotonic() - started:.1f}s; saved {path}", flush=True)
        if args.shard_size:
            pending[split] = (paths, entries)
    del encoder, predictor, teacher
    if device.type == "mps":
        th.mps.empty_cache()
    for split, (paths, entries) in pending.items():
        cache = merge_parts(paths, entries, provenance)
        save_cache(cache, output / f"{split}.pt")
        del cache
        print(f"Saved consolidated {output / f'{split}.pt'}", flush=True)
    if args.shard_size:
        for path in part_paths:
            path.unlink()
        (parts / "run.pt").unlink()
        parts.rmdir()

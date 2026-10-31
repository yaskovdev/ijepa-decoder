import torch as th
from torch.utils.data import DataLoader, TensorDataset

from . import dist_util


def load_feature_cache(path):
    cache = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict) or "targets" not in cache or "features" not in cache:
        raise ValueError("Feature cache must contain targets and features.")
    targets, features = cache["targets"], cache["features"]
    if not isinstance(targets, th.Tensor) or not isinstance(features, th.Tensor):
        raise ValueError("Cached targets and features must be tensors.")
    if targets.ndim != 4 or targets.shape[1] != 3 or targets.shape[2] != targets.shape[3]:
        raise ValueError("Cached targets must be square RGB images in NCHW format.")
    if targets.dtype != th.uint8:
        raise ValueError("Cached targets must contain uint8 pixels.")
    if features.ndim != 2 or features.shape[0] != targets.shape[0] or targets.shape[0] == 0:
        raise ValueError("Feature cache must contain one nonempty feature vector per target.")
    if not features.is_floating_point() or not th.isfinite(features).all():
        raise ValueError("Cached features must be finite floating-point values.")
    if "image_indices" in cache:
        indices, originals = cache["image_indices"], cache.get("originals")
        if (not isinstance(indices, th.Tensor) or indices.dtype != th.int64
                or indices.shape != (len(features),)
                or not isinstance(originals, th.Tensor) or originals.ndim != 4
                or originals.shape[1] != 3 or originals.dtype != th.uint8
                or indices.min() < 0 or indices.max() >= len(originals)):
            raise ValueError("Cached image indices must map every example to an original RGB image.")
        if (len(cache["image_paths"]) != len(originals) or len(cache["paths"]) != len(features)
                or cache["paths"] != [cache["image_paths"][index] for index in indices.tolist()]):
            raise ValueError("Cached example paths do not match their original-image indices.")
    if "target_features" in cache:
        target = cache["target_features"]
        if (not isinstance(target, th.Tensor) or target.shape != features.shape
                or not target.is_floating_point() or not th.isfinite(target).all()
                or cache.get("target_normalization") != "layer_norm_patch_tokens"):
            raise ValueError("Target-encoder features must be paired, finite, and normalized like I-JEPA training.")
    if "image_hashes" in cache:
        hashes, originals = cache["image_hashes"], cache.get("originals")
        if (not isinstance(hashes, list) or not isinstance(originals, th.Tensor)
                or len(hashes) != len(originals) or any(not isinstance(value, str) for value in hashes)):
            raise ValueError("Image hashes must identify every cached original image.")
    return cache


def original_for_example(cache, index):
    original_index = cache["image_indices"][index].item() if "image_indices" in cache else index
    return cache["originals"][original_index]


def distinct_examples(cache, limit):
    seen, indices = set(), []
    for index, path in enumerate(cache["paths"]):
        if path not in seen:
            seen.add(path)
            indices.append(index)
            if len(indices) == limit:
                break
    return indices


def different_image_index(cache, index, direction=1):
    paths = cache["paths"]
    for offset in range(1, len(paths)):
        candidate = (index + direction * offset) % len(paths)
        if paths[candidate] != paths[index]:
            return candidate
    raise ValueError("Wrong-conditioning controls require at least two distinct original images.")


def load_feature_data(cache, batch_size, seed=0):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    rank, world_size = dist_util.get_rank(), dist_util.get_world_size()
    targets = cache["targets"][rank::world_size].float() / 127.5 - 1
    features = cache["features"][rank::world_size].float()
    if len(targets) < batch_size:
        raise ValueError(f"Only {len(targets)} cached images per rank, fewer than batch_size={batch_size}.")
    loader = DataLoader(
        TensorDataset(targets, features),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=th.Generator().manual_seed(seed + rank),
    )
    while True:
        for images, feat in loader:
            yield images, {"feat": feat}

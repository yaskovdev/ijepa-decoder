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
    return cache


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

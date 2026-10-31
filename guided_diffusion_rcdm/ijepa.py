import importlib
import sys
from pathlib import Path

import torch as th


def load_ijepa_models(repo_dir, checkpoint_path, device):
    encoder, predictor, _, epoch = _load_ijepa_models(repo_dir, checkpoint_path, device, False)
    return encoder, predictor, epoch


def load_ijepa_with_target(repo_dir, checkpoint_path, device):
    return _load_ijepa_models(repo_dir, checkpoint_path, device, True)


def _load_ijepa_models(repo_dir, checkpoint_path, device, include_target):
    repo_dir = Path(repo_dir).resolve()
    model_path = repo_dir / "src/models/vision_transformer.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Clone facebookresearch/ijepa first: {model_path}")

    # Upstream I-JEPA is not an installable package and imports its own `src`.
    sys.path.insert(0, str(repo_dir))
    try:
        vit = importlib.import_module("src.models.vision_transformer")
    finally:
        sys.path.remove(str(repo_dir))
    if Path(vit.__file__).resolve() != model_path:
        raise ImportError("Another package named src is shadowing the I-JEPA checkout.")

    checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    names = ("encoder", "predictor", "target_encoder") if include_target else ("encoder", "predictor")
    for name in names:
        if name not in checkpoint:
            raise ValueError(f"I-JEPA checkpoint is missing {name}; an encoder-only checkpoint is insufficient.")
    encoder = vit.vit_huge(img_size=[224], patch_size=14)
    predictor = vit.vit_predictor(
        num_patches=256,
        embed_dim=1280,
        predictor_embed_dim=384,
        depth=12,
        num_heads=16,
    )
    target_encoder = vit.vit_huge(img_size=[224], patch_size=14) if include_target else None
    models = [("encoder", encoder), ("predictor", predictor)]
    if target_encoder is not None:
        models.append(("target_encoder", target_encoder))
    for name, model in models:
        state = {key.removeprefix("module."): value for key, value in checkpoint[name].items()}
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
    epoch = checkpoint.get("epoch")
    del checkpoint, state
    return (encoder.to(device), predictor.to(device),
            target_encoder.to(device) if target_encoder is not None else None, epoch)


def make_block_masks(top, left, block_size=7, grid_size=16):
    if not 0 < block_size < grid_size:
        raise ValueError("block_size must leave both visible and hidden patches.")
    if not (0 <= top <= grid_size - block_size and 0 <= left <= grid_size - block_size):
        raise ValueError("Target block falls outside the patch grid.")
    indices = th.arange(grid_size * grid_size).reshape(grid_size, grid_size)
    hidden = th.zeros(grid_size, grid_size, dtype=th.bool)
    hidden[top:top + block_size, left:left + block_size] = True
    return indices[~hidden], indices[hidden]


@th.no_grad()
def predict_features(encoder, predictor, images, context_indices, target_indices):
    if images.shape[0] != context_indices.shape[0] or images.shape[0] != target_indices.shape[0]:
        raise ValueError("Images and masks must have the same batch size.")
    context = encoder(images, masks=[context_indices])
    predictions = predictor(context, [context_indices], [target_indices])
    return predictions.mean(dim=1)


@th.no_grad()
def target_features(encoder, images, target_indices, image_indices):
    if target_indices.ndim != 2 or image_indices.shape != (target_indices.shape[0],):
        raise ValueError("Target masks require one original-image index per example.")
    tokens = encoder(images)
    # Match I-JEPA's training target: normalize each token before masking/pooling.
    tokens = th.nn.functional.layer_norm(tokens, (tokens.shape[-1],))
    tokens = tokens[image_indices]
    selected = th.gather(tokens, 1, target_indices[..., None].expand(-1, -1, tokens.shape[-1]))
    return selected.mean(1)


class PredictorAdapter(th.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        if min(input_dim, output_dim) <= 0 or hidden_dim < 0 or not 0 <= dropout < 1:
            raise ValueError("Adapter dimensions must be positive, hidden_dim nonnegative, and dropout in [0,1).")
        self.register_buffer("input_mean", th.zeros(input_dim))
        self.register_buffer("input_scale", th.ones(input_dim))
        self.register_buffer("output_mean", th.zeros(output_dim))
        self.register_buffer("output_scale", th.ones(output_dim))
        self.net = th.nn.Sequential(th.nn.Linear(input_dim, output_dim)) if hidden_dim == 0 else th.nn.Sequential(
            th.nn.Linear(input_dim, hidden_dim),
            th.nn.GELU(),
            th.nn.Dropout(dropout),
            th.nn.Linear(hidden_dim, output_dim),
        )
        th.nn.init.zeros_(self.net[-1].weight)
        th.nn.init.zeros_(self.net[-1].bias)

    @th.no_grad()
    def fit_statistics(self, predictor_features, dino_features):
        if (predictor_features.ndim != 2 or dino_features.ndim != 2
                or predictor_features.shape[0] != dino_features.shape[0]
                or predictor_features.shape[0] < 2
                or predictor_features.shape[1] != self.input_mean.numel()
                or dino_features.shape[1] != self.output_mean.numel()):
            raise ValueError("Adapter statistics require paired, compatible training features.")
        if not th.isfinite(predictor_features).all() or not th.isfinite(dino_features).all():
            raise ValueError("Adapter training features must be finite.")
        if (dino_features < 0).any():
            raise ValueError("DINO ResNet trunk features must be nonnegative.")
        self.input_mean.copy_(predictor_features.mean(0))
        self.input_scale.copy_(predictor_features.std(0, unbiased=False).clamp_min(1e-4))
        self.output_mean.copy_(dino_features.mean(0))
        self.output_scale.copy_(dino_features.std(0, unbiased=False).clamp_min(1e-4))

    @th.no_grad()
    def fit_ridge(self, features, targets, alpha):
        if len(self.net) != 1 or alpha <= 0:
            raise ValueError("Ridge fitting requires a linear adapter and positive alpha.")
        # Keep the stable float64 solve on CPU: MPS does not support float64.
        x = ((features.cpu() - self.input_mean.cpu()) / self.input_scale.cpu()).double()
        y = ((targets.cpu() - self.output_mean.cpu()) / self.output_scale.cpu()).double()
        gram = x.T @ x
        gram.diagonal().add_(alpha * len(x))
        weights = th.linalg.solve(gram, x.T @ y)
        self.net[0].weight.copy_(weights.T.float())
        self.net[0].bias.zero_()

    def forward(self, features):
        normalized = (features - self.input_mean) / self.input_scale
        output = self.output_mean + self.output_scale * self.net(normalized)
        # ResNet trunk features follow a ReLU; preserve their support for RCDM.
        return output.relu()

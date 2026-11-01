import importlib
import sys
from pathlib import Path

import torch as th


IJEPA_COMMIT = "52c1ae95d05f743e000e8f10a1f3a79b10cff048"
IJEPA_CHECKPOINT = "IN1K-vit.h.14-300e.pth.tar"


def load_ijepa(repo_dir, checkpoint_path, device, include_target=False):
    repo_dir = Path(repo_dir).resolve()
    model_path = repo_dir / "src/models/vision_transformer.py"
    if not model_path.is_file():
        raise FileNotFoundError(f"Clone facebookresearch/ijepa first: {model_path}")
    sys.path.insert(0, str(repo_dir))
    try:
        vit = importlib.import_module("src.models.vision_transformer")
    finally:
        sys.path.remove(str(repo_dir))
    if Path(vit.__file__).resolve() != model_path:
        raise ImportError("Another src package is shadowing the I-JEPA checkout.")
    checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    names = ("encoder", "predictor", "target_encoder") if include_target else ("encoder", "predictor")
    for name in names:
        if name not in checkpoint:
            raise ValueError(f"Full I-JEPA checkpoint is missing {name}; encoder-only weights cannot work.")
    encoder = vit.vit_huge(img_size=[224], patch_size=14)
    predictor = vit.vit_predictor(num_patches=256, embed_dim=1280, predictor_embed_dim=384,
                                  depth=12, num_heads=16)
    teacher = vit.vit_huge(img_size=[224], patch_size=14) if include_target else None
    models = [("encoder", encoder), ("predictor", predictor)]
    if teacher is not None:
        models.append(("target_encoder", teacher))
    for name, model in models:
        state = {key.removeprefix("module."): value for key, value in checkpoint[name].items()}
        model.load_state_dict(state, strict=True)
        model.eval().requires_grad_(False)
    epoch = checkpoint.get("epoch")
    del checkpoint, state
    return encoder.to(device), predictor.to(device), teacher.to(device) if teacher is not None else None, epoch


def make_masks(box):
    x0, y0, x1, y1 = box
    if (min(box) < 0 or max(box) > 224 or any(value % 14 for value in box)
            or (x1 - x0, y1 - y0) != (98, 98)):
        raise ValueError("The released decoder needs a patch-aligned 98x98 box in the 224x224 view.")
    indices = th.arange(256).reshape(16, 16)
    hidden = th.zeros(16, 16, dtype=th.bool)
    hidden[y0 // 14:y1 // 14, x0 // 14:x1 // 14] = True
    return indices[~hidden], indices[hidden]


@th.no_grad()
def predict_tokens(encoder, predictor, images, context, target):
    if images.shape[0] != len(context) or images.shape[0] != len(target):
        raise ValueError("Images and mask batches differ.")
    return predictor(encoder(images, masks=[context]), [context], [target])


@th.no_grad()
def target_tokens(encoder, images, target, image_indices=None):
    tokens = encoder(images)
    tokens = th.nn.functional.layer_norm(tokens, (tokens.shape[-1],))
    if image_indices is not None:
        if (image_indices.dtype != th.long or image_indices.shape != (len(target),)
                or image_indices.min() < 0 or image_indices.max() >= len(images)):
            raise ValueError("Target masks must map to valid original images.")
        tokens = tokens[image_indices]
    if len(tokens) != len(target):
        raise ValueError("Target mask and original image batches differ.")
    return tokens.gather(1, target[..., None].expand(-1, -1, tokens.shape[-1]))

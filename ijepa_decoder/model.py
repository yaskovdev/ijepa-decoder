import json
from pathlib import Path

import torch as th
from torch import nn
from torch.nn import functional as F

from .io import file_sha256


MAE_MODEL = "facebook/vit-mae-base"
MAE_REVISION = "25b184bea5538bf5c4c852c79d221195fdd2778d"
MAE_SHA256 = "479dcef4bd5df06259399027b789f21e9d9a1b79f37155a64176d55bc26fdae8"


def initialization_spec():
    return dict(model=MAE_MODEL, revision=MAE_REVISION, sha256=MAE_SHA256,
                input_projection="new_ijepa_projection", positions="native_16x16_sincos",
                rgb_head="pretrained_16_to_14_bicubic", output_normalization="fixed_imagenet_rgb")


def download_assets(cache_dir):
    from huggingface_hub import snapshot_download

    directory = Path(snapshot_download(
        MAE_MODEL, revision=MAE_REVISION, token=False, cache_dir=str(cache_dir),
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    ))
    checkpoint = directory / "model.safetensors"
    if file_sha256(checkpoint) != MAE_SHA256:
        raise ValueError("MAE image prior differs from the pinned checkpoint checksum.")
    return dict(checkpoint=str(checkpoint), config=str(directory / "config.json"),
                processor=str(directory / "preprocessor_config.json"), initialization=initialization_spec())


def mask_order(boxes, shape, patch_size=14, grid_size=16):
    height, width = shape
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.dtype != th.long
            or min(height, width) <= 0 or max(height, width) > grid_size
            or (boxes < 0).any() or (boxes > patch_size * grid_size).any()
            or (boxes % patch_size != 0).any()
            or ((boxes[:, 2] - boxes[:, 0]) != width * patch_size).any()
            or ((boxes[:, 3] - boxes[:, 1]) != height * patch_size).any()):
        raise ValueError("Native reconstruction requires aligned boxes matching the patch-token rectangle.")
    rows = boxes[:, 1, None, None] // patch_size + th.arange(height, device=boxes.device)[None, :, None]
    columns = boxes[:, 0, None, None] // patch_size + th.arange(width, device=boxes.device)[None, None, :]
    indices = (rows * grid_size + columns).flatten(1)
    full = th.arange(grid_size**2, device=boxes.device).expand(len(boxes), -1)
    selected = th.zeros_like(full, dtype=th.bool).scatter_(1, indices, True)
    complement = full[~selected].reshape(len(boxes), -1)
    return th.cat([indices, complement], dim=1).argsort(1), indices


def resize_rgb_head(weight, bias, patch_size):
    old_patch = int((len(bias) / 3) ** 0.5)
    if (weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != len(bias)
            or old_patch**2 * 3 != len(bias) or patch_size <= 0):
        raise ValueError("RGB prediction head must contain complete row-major RGB patches.")
    channels = weight.shape[1]
    grid = weight.reshape(old_patch, old_patch, 3, channels).permute(2, 3, 0, 1)
    grid = F.interpolate(grid.reshape(3 * channels, 1, old_patch, old_patch),
                         (patch_size, patch_size), mode="bicubic", align_corners=False)
    resized = grid.reshape(3, channels, patch_size, patch_size).permute(2, 3, 0, 1)
    colors = bias.reshape(old_patch, old_patch, 3).permute(2, 0, 1)[None]
    colors = F.interpolate(colors, (patch_size, patch_size), mode="bicubic", align_corners=False)
    return resized.reshape(3 * patch_size**2, channels), colors[0].permute(1, 2, 0).reshape(-1)


class NativeDecoder(nn.Module):
    def __init__(self, input_dim=1280, source_shape=(7, 7), config=None):
        from transformers import ViTMAEConfig
        from transformers.models.vit_mae.modeling_vit_mae import ViTMAEDecoder

        super().__init__()
        if input_dim <= 0 or min(source_shape) <= 0 or max(source_shape) > 16:
            raise ValueError("Native decoder requires positive feature dimensions and a valid target grid.")
        self.source_shape = tuple(source_shape)
        self.config = ViTMAEConfig(**(config or {}))
        self.config.hidden_size, self.config.patch_size, self.config.image_size = input_dim, 14, 224
        self.config.num_channels, self.config.norm_pix_loss = 3, False
        self.config._attn_implementation = "eager"
        self.decoder = ViTMAEDecoder(self.config, num_patches=256)
        self.register_buffer("rgb_mean", th.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("rgb_std", th.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, tokens, boxes):
        if (tokens.ndim != 3 or tokens.shape[1:] != (
                self.source_shape[0] * self.source_shape[1], self.config.hidden_size)
                or len(boxes) != len(tokens)):
            raise ValueError("Native decoder token dimensions differ from its trained target grid.")
        restore, indices = mask_order(boxes.cpu(), self.source_shape)
        normalized = F.layer_norm(tokens, (tokens.shape[-1],), eps=1e-6)
        encoded = th.cat([normalized.mean(1, keepdim=True), normalized], dim=1)
        logits = self.decoder(encoded, restore).logits
        patches = logits.gather(1, indices.to(logits.device)[:, :, None].expand(-1, -1, logits.shape[-1]))
        height, width = self.source_shape
        rgb = patches.reshape(len(tokens), height, width, 14, 14, 3)
        rgb = rgb.permute(0, 5, 1, 3, 2, 4).reshape(len(tokens), 3, height * 14, width * 14)
        return rgb * self.rgb_std + self.rgb_mean


def load_image_prior(assets, input_dim, source_shape):
    from safetensors import safe_open

    if assets["initialization"] != initialization_spec() or file_sha256(assets["checkpoint"]) != MAE_SHA256:
        raise ValueError("Native decoder initialization is not the pinned MAE image prior.")
    config = json.loads(Path(assets["config"]).read_text())
    processor = json.loads(Path(assets["processor"]).read_text())
    if (config["norm_pix_loss"] or processor["image_mean"] != [0.485, 0.456, 0.406]
            or processor["image_std"] != [0.229, 0.224, 0.225]):
        raise ValueError("Native decoding requires fixed ImageNet normalization, not hidden patch statistics.")
    model = NativeDecoder(input_dim, source_shape, config)
    with safe_open(assets["checkpoint"], framework="pt", device="cpu") as weights:
        source = {name.removeprefix("decoder."): weights.get_tensor(name)
                  for name in weights.keys() if name.startswith("decoder.")}
    state = model.decoder.state_dict()
    if set(source) != set(state):
        raise ValueError("Released MAE decoder has unexpected parameter names.")
    source["decoder_pred.weight"], source["decoder_pred.bias"] = resize_rgb_head(
        source["decoder_pred.weight"], source["decoder_pred.bias"], 14,
    )
    for name, value in source.items():
        if name in ("decoder_embed.weight", "decoder_pos_embed"):
            continue
        if value.shape != state[name].shape:
            raise ValueError(f"Native image-prior parameter shape differs: {name}.")
        state[name] = value
    model.decoder.load_state_dict(state, strict=True)
    return model


def load_decoder(path, device, expected_spec=None, freeze=True):
    from safetensors.torch import load_file

    path = Path(path)
    if path.is_dir():
        metadata = json.loads((path / "config.json").read_text())
        state = load_file(str(path / "model.safetensors"), device="cpu")
    else:
        metadata = th.load(path, map_location="cpu", weights_only=True)
        state = metadata["state_dict"]
    spec = metadata["input_spec"]
    if (metadata["initialization"] != initialization_spec()
            or metadata["feature_source"] not in ("target", "predictor", "mixed")
            or spec["target_normalization"] != "layer_norm_patch_tokens"
            or spec["output"] != "native_rgb_patch14_fixed_imagenet"
            or (expected_spec is not None and spec != expected_spec)):
        raise ValueError("Decoder differs in frozen I-JEPA, geometry, normalization, or image prior.")
    model = NativeDecoder(spec["feature_dim"], spec["shape"], metadata["config"])
    model.load_state_dict(state, strict=True)
    if freeze:
        model.eval().requires_grad_(False)
    return model.to(device), metadata

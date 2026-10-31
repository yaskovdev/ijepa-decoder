import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch as th
from torch import nn
from torch.nn import functional as F


RAE_COMMIT = "a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041"
DINO_MODEL = "facebook/dinov2-with-registers-base"
DINO_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
RAE_COLLECTION = "nyu-visionx/RAE-collections"
RAE_REVISION = "1be4f03273523431f099a934da4cf1940dc6039f"
DECODER_FILE = "decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
DECODER_SHA256 = "5fedf7c9660476a709e122cef18385c917532914a23f034388e1c1a52bde2be6"


def rae_spec():
    return dict(encoder=DINO_MODEL, encoder_revision=DINO_REVISION,
                decoder_revision=RAE_REVISION, decoder_sha256=DECODER_SHA256,
                source_commit=RAE_COMMIT, encoder_input_size=224, output_size=256,
                normalization="imagenet_rgb_zero_one", token_normalization="layer_norm_no_affine",
                token_grid=[16, 16], token_dim=768, register_tokens=4)


def download_assets(cache_dir):
    from huggingface_hub import hf_hub_download, snapshot_download

    encoder = snapshot_download(
        DINO_MODEL, revision=DINO_REVISION, cache_dir=str(cache_dir), token=False,
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    )
    decoder = hf_hub_download(
        RAE_COLLECTION, DECODER_FILE, revision=RAE_REVISION, cache_dir=str(cache_dir), token=False,
    )
    if file_sha256(decoder) != DECODER_SHA256:
        raise ValueError("Downloaded RAE decoder does not match the pinned checksum.")
    return dict(encoder_dir=encoder, decoder_path=decoder, rae_spec=rae_spec())


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@th.no_grad()
def cache_dino_positions(model):
    positions = model.embeddings.position_embeddings
    if positions.device.type != "cpu":
        raise ValueError("Cache DINOv2's fixed-resolution positions before moving the encoder to MPS.")
    dummy = th.zeros(1, 257, model.config.hidden_size, dtype=positions.dtype)
    # Use the exact upstream antialiased interpolation once on CPU; MPS lacks it.
    fixed = model.embeddings.interpolate_pos_encoding(dummy, 224, 224)
    model.embeddings.position_embeddings = nn.Parameter(fixed, requires_grad=False)


class DinoPatchEncoder(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        if (model.config.hidden_size, model.config.patch_size, model.config.num_register_tokens) != (768, 14, 4):
            raise ValueError("RAE requires the matching DINOv2 base encoder with four registers.")
        self.model = model
        cache_dino_positions(model)
        # The released decoder was trained without the encoder's learned LN affine.
        model.layernorm = nn.LayerNorm(768, eps=model.config.layer_norm_eps, elementwise_affine=False)
        self.register_buffer("mean", th.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", th.tensor(std).reshape(1, 3, 1, 1))

    @th.no_grad()
    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
            raise ValueError("DINOv2 input must be floating-point RGB images.")
        if not th.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("DINOv2 input pixels must be finite and in [0,1].")
        images = F.interpolate(images, (224, 224), mode="bicubic", align_corners=False)
        tokens = self.model((images - self.mean) / self.std).last_hidden_state[:, 5:]
        if tokens.shape[1:] != (256, 768) or not th.isfinite(tokens).all():
            raise FloatingPointError("DINOv2 returned invalid RAE patch tokens.")
        return tokens


def load_encoder(encoder_dir, device):
    from transformers import Dinov2WithRegistersModel

    encoder_dir = Path(encoder_dir)
    processor = json.loads((encoder_dir / "preprocessor_config.json").read_text())
    model = Dinov2WithRegistersModel.from_pretrained(
        encoder_dir, local_files_only=True, use_safetensors=True, attn_implementation="eager",
    )
    encoder = DinoPatchEncoder(model, processor["image_mean"], processor["image_std"])
    return encoder.eval().requires_grad_(False).to(device)


class RaeDecoder(nn.Module):
    def __init__(self, decoder, mean, std):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("mean", th.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("std", th.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (256, 768) or not th.isfinite(tokens).all():
            raise ValueError("RAE decoding requires finite, unpooled 16x16 DINOv2 patch tokens.")
        logits = self.decoder(tokens, drop_cls_token=False).logits
        images = self.decoder.unpatchify(logits) * self.std + self.mean
        if not th.isfinite(images).all():
            raise FloatingPointError("RAE returned non-finite RGB images.")
        return images.clamp(0, 1)


def load_decoder(repo_dir, checkpoint, encoder_dir, device):
    repo_dir = Path(repo_dir).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != RAE_COMMIT:
        raise ValueError("RAE source checkout does not match the pinned decoder implementation.")
    if file_sha256(checkpoint) != DECODER_SHA256:
        raise ValueError("RAE decoder checkpoint differs from the pinned weights.")
    sys.path.insert(0, str(repo_dir / "src"))
    try:
        module = importlib.import_module("stage1.decoders.decoder")
    finally:
        sys.path.remove(str(repo_dir / "src"))
    if Path(module.__file__).resolve() != repo_dir / "src/stage1/decoders/decoder.py":
        raise ImportError("Another stage1 package is shadowing the RAE checkout.")
    config = module.ViTMAEConfig(**json.loads((repo_dir / "configs/decoder/ViTXL/config.json").read_text()))
    config.patch_size, config.image_size, config.hidden_size = 16, 256, 768
    config._attn_implementation = "eager"
    decoder = module.GeneralDecoder(config, num_patches=256)
    decoder.load_state_dict(th.load(checkpoint, map_location="cpu", weights_only=True, mmap=True), strict=True)
    processor = json.loads((Path(encoder_dir) / "preprocessor_config.json").read_text())
    model = RaeDecoder(decoder, processor["image_mean"], processor["image_std"])
    print(f"Loaded frozen RAE decoder: {sum(p.numel() for p in decoder.parameters()):,} parameters.", flush=True)
    return model.eval().requires_grad_(False).to(device)


def regrid(tokens, source_shape, output_size=16):
    height, width = source_shape
    if tokens.ndim != 3 or min(height, width) <= 0 or height * width != tokens.shape[1]:
        raise ValueError("Patch tokens must match their rectangular grid.")
    grid = tokens.transpose(1, 2).reshape(len(tokens), tokens.shape[-1], height, width)
    grid = F.interpolate(grid, (output_size, output_size), mode="bilinear", align_corners=False)
    return grid.flatten(2).transpose(1, 2)


class SpatialAdapter(nn.Module):
    def __init__(self, input_dim=1280, output_dim=768, hidden_dim=0):
        super().__init__()
        if min(input_dim, output_dim) <= 0 or hidden_dim < 0:
            raise ValueError("Spatial adapter dimensions must be positive.")
        self.register_buffer("input_mean", th.zeros(input_dim))
        self.register_buffer("input_scale", th.ones(input_dim))
        self.register_buffer("output_mean", th.zeros(output_dim))
        self.register_buffer("output_scale", th.ones(output_dim))
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim)) if hidden_dim == 0 else nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, tokens, source_shape):
        if tokens.shape[-1] != self.input_mean.numel():
            raise ValueError("Spatial adapter input feature dimension does not match training.")
        projected = self.net((tokens - self.input_mean) / self.input_scale)
        return self.output_mean + self.output_scale * regrid(projected, source_shape)

    @th.no_grad()
    def fit_statistics(self, inputs, targets, source_shape, batch_size=16):
        if (inputs.ndim != 3 or targets.ndim != 3 or len(inputs) != len(targets) or len(inputs) < 2
                or inputs.shape[-1] != self.input_mean.numel()
                or targets.shape[1:] != (256, self.output_mean.numel())
                or batch_size <= 0
                or not th.isfinite(inputs).all() or not th.isfinite(targets).all()):
            raise ValueError("Spatial adapter statistics require paired finite patch features.")
        xsum, xsquare = th.zeros_like(self.input_mean), th.zeros_like(self.input_mean)
        ysum, ysquare = th.zeros_like(self.output_mean), th.zeros_like(self.output_mean)
        for start in range(0, len(inputs), batch_size):
            x = regrid(inputs[start:start + batch_size].float(), source_shape)
            y = targets[start:start + batch_size].float()
            xsum += x.sum((0, 1))
            xsquare += x.square().sum((0, 1))
            ysum += y.sum((0, 1))
            ysquare += y.square().sum((0, 1))
        count = len(inputs) * 256
        self.input_mean.copy_(xsum / count)
        self.output_mean.copy_(ysum / count)
        self.input_scale.copy_((xsquare / count - self.input_mean.square()).clamp_min(1e-8).sqrt())
        self.output_scale.copy_((ysquare / count - self.output_mean.square()).clamp_min(1e-8).sqrt())

    @th.no_grad()
    def fit_ridge(self, inputs, targets, source_shape, alpha, batch_size=16):
        if len(self.net) != 1 or alpha <= 0 or not math.isfinite(alpha) or batch_size <= 0:
            raise ValueError("Use a linear adapter, positive ridge strength and positive batch size.")
        self.fit_statistics(inputs, targets, source_shape, batch_size)
        gram = th.zeros(self.input_mean.numel(), self.input_mean.numel(), dtype=th.float64)
        cross = th.zeros(self.input_mean.numel(), self.output_mean.numel(), dtype=th.float64)
        for start in range(0, len(inputs), batch_size):
            x = regrid((inputs[start:start + batch_size].float() - self.input_mean) / self.input_scale,
                       source_shape).reshape(-1, self.input_mean.numel()).double()
            y = ((targets[start:start + batch_size].float() - self.output_mean) / self.output_scale
                 ).reshape(-1, self.output_mean.numel()).double()
            gram += x.T @ x
            cross += x.T @ y
        gram.diagonal().add_(alpha * len(inputs) * 256)
        self.net[0].weight.copy_(th.linalg.solve(gram, cross).T.float())
        self.net[0].bias.zero_()

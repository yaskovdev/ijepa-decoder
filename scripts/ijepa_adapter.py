import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from guided_diffusion_rcdm import dist_util
from guided_diffusion_rcdm.feature_datasets import load_feature_cache
from guided_diffusion_rcdm.get_ssl_models import load_dino_trunk
from guided_diffusion_rcdm.ijepa import PredictorAdapter
from guided_diffusion_rcdm.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from scripts.ijepa_decoder import (
    comparison_row,
    feature_spec,
    hidden_crop,
    save_cache,
    save_comparison_grid,
    to_image,
)


def configure(args):
    th.set_num_threads(args.cpu_threads)
    th.manual_seed(args.seed)
    return dist_util.set_device(args.device)


def load_predictor_cache(path):
    cache = load_feature_cache(path)
    if cache.get("conditioning") != "ijepa_predictor":
        raise ValueError("The adapter requires frozen I-JEPA predictor outputs, not encoder features.")
    return cache


@th.no_grad()
def prepare(args):
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        if (out_dir / f"{split}.pt").exists():
            raise FileExistsError(f"{split}.pt already exists; choose a new output directory.")
    device = configure(args)
    model = load_dino_trunk(args.dino_checkpoint, device)
    started = time.monotonic()
    for split in ("train", "val"):
        cache = load_predictor_cache(Path(args.cache_dir) / f"{split}.pt")
        features = []
        for start in range(0, len(cache["features"]), args.batch_size):
            end = min(start + args.batch_size, len(cache["features"]))
            crops = [
                th.from_numpy(np.array(hidden_crop(cache, index, 224))).permute(2, 0, 1)
                for index in range(start, end)
            ]
            # Match the released RCDM's SSL inputs, not standard DINO ImageNet normalization.
            batch = th.stack(crops).float().to(device) / 127.5 - 1
            features.append(model(batch).cpu())
            print(f"{split}: {end}/{len(cache['features'])}, "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
        features = th.cat(features)
        if features.shape != (len(cache["features"]), 2048) or not th.isfinite(features).all():
            raise ValueError("Invalid DINO trunk features.")
        save_cache(dict(
            features=features,
            paths=cache["paths"],
            boxes=cache["boxes"],
            feature_spec=feature_spec(cache),
            dino_spec=dict(model="dino_resnet50_trunk", input_size=224,
                           normalization="rgb_minus_one_one", checkpoint=Path(args.dino_checkpoint).name),
        ), out_dir / f"{split}.pt")
    print(f"Saved aligned DINO feature caches to {out_dir}", flush=True)


def load_dino_cache(path, predictor_cache):
    cache = th.load(path, map_location="cpu", weights_only=True)
    if (cache["feature_spec"] != feature_spec(predictor_cache)
            or cache["paths"] != predictor_cache["paths"]
            or not th.equal(cache["boxes"], predictor_cache["boxes"])):
        raise ValueError("DINO targets do not match the predictor images, checkpoint, or hidden boxes.")
    features = cache["features"]
    if (features.ndim != 2 or features.shape[0] != len(predictor_cache["features"])
            or not features.is_floating_point() or not th.isfinite(features).all()
            or (features < 0).any()):
        raise ValueError("DINO targets must be finite, nonnegative paired feature vectors.")
    spec = cache["dino_spec"]
    if (spec["model"] != "dino_resnet50_trunk" or spec["input_size"] != 224
            or spec["normalization"] != "rgb_minus_one_one"):
        raise ValueError("DINO feature extraction does not match the released RCDM.")
    return cache


@th.no_grad()
def adapter_metrics(model, features, targets):
    prediction = model(features)
    return dict(
        mse=(((prediction - targets) / model.output_scale) ** 2).mean().item(),
        wrong_mse=(((prediction.roll(1, 0) - targets) / model.output_scale) ** 2).mean().item(),
        cosine=F.cosine_similarity(prediction, targets).mean().item(),
        wrong_cosine=F.cosine_similarity(prediction.roll(1, 0), targets).mean().item(),
    )


def train(args):
    if args.steps <= 0 or args.lr <= 0 or args.weight_decay < 0 or args.log_interval <= 0:
        raise ValueError("Use positive steps, learning rate, logging interval, and nonnegative weight decay.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "adapter_config.json").exists() or (out_dir / "best.pt").exists():
        raise FileExistsError("Adapter experiment already exists; choose a new output directory.")
    predictor = {
        split: load_predictor_cache(Path(args.cache_dir) / f"{split}.pt")
        for split in ("train", "val")
    }
    dino = {
        split: load_dino_cache(Path(args.dino_cache_dir) / f"{split}.pt", predictor[split])
        for split in ("train", "val")
    }
    if (feature_spec(predictor["train"]) != feature_spec(predictor["val"])
            or dino["train"]["dino_spec"] != dino["val"]["dino_spec"]
            or dino["train"]["features"].shape[1] != dino["val"]["features"].shape[1]):
        raise ValueError("Train and validation features must use the same frozen models and geometry.")
    if not 0 < args.batch_size <= len(predictor["train"]["features"]):
        raise ValueError("Batch size must be positive and no larger than the training dataset.")
    if len(predictor["val"]["features"]) < 2:
        raise ValueError("Validation needs at least two examples for the wrong-conditioning control.")
    device = configure(args)
    input_dim = predictor["train"]["features"].shape[1]
    output_dim = dino["train"]["features"].shape[1]
    model = PredictorAdapter(input_dim, output_dim, args.hidden_dim)
    model.fit_statistics(predictor["train"]["features"], dino["train"]["features"])
    model.to(device)
    features = {split: cache["features"].float().to(device) for split, cache in predictor.items()}
    targets = {split: cache["features"].float().to(device) for split, cache in dino.items()}
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    config = dict(
        input_dim=input_dim, output_dim=output_dim, hidden_dim=args.hidden_dim,
        feature_spec=feature_spec(predictor["train"]), dino_spec=dino["train"]["dino_spec"],
        seed=args.seed, steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay,
    )
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    model.eval()
    baseline = adapter_metrics(model, features["val"], targets["val"])
    print(f"Mean-feature baseline: {baseline}", flush=True)
    history = [dict(step=0, validation=baseline)]
    best = baseline["mse"]
    generator = th.Generator().manual_seed(args.seed)
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        indices = th.randint(len(features["train"]), (args.batch_size,), generator=generator).to(device)
        prediction = model(features["train"][indices])
        loss = (((prediction - targets["train"][indices]) / model.output_scale) ** 2).mean()
        if not th.isfinite(loss):
            raise FloatingPointError("Adapter training loss is not finite.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % args.log_interval == 0 or step == args.steps:
            model.eval()
            metrics = adapter_metrics(model, features["val"], targets["val"])
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Adapter validation metrics are not finite.")
            history.append(dict(step=step, training_mse=loss.item(), validation=metrics))
            payload = dict(config, state_dict={key: value.detach().cpu()
                                              for key, value in model.state_dict().items()},
                           completed_steps=step, validation=metrics)
            save_cache(payload, out_dir / "last.pt")
            if metrics["mse"] < best:
                best = metrics["mse"]
                save_cache(payload, out_dir / "best.pt")
            (out_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
            print(f"step {step}/{args.steps}: {metrics}, "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    if not (out_dir / "best.pt").exists():
        raise RuntimeError("Adapter did not improve on mean DINO features; inspect last.pt and metrics.json.")
    print(f"Saved best adapter to {out_dir / 'best.pt'}", flush=True)


def load_adapter(path, predictor_cache, dino_spec, device):
    checkpoint = th.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["feature_spec"] != feature_spec(predictor_cache)
            or checkpoint["dino_spec"] != dino_spec
            or checkpoint["input_dim"] != predictor_cache["features"].shape[1]):
        raise ValueError("Adapter checkpoint does not match the frozen predictor or DINO feature extraction.")
    model = PredictorAdapter(checkpoint["input_dim"], checkpoint["output_dim"], checkpoint["hidden_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_prior(path, device, sampling_steps):
    config = model_and_diffusion_defaults()
    config.update(
        image_size=128, num_channels=256, num_res_blocks=2,
        attention_resolutions="32,16,8", learn_sigma=True,
        resblock_updown=True, use_scale_shift_norm=True,
        timestep_respacing=f"ddim{sampling_steps}",
    )
    model, diffusion = create_model_and_diffusion(**config, feat_cond=True, G_shared=True, ssl_dim=2048)
    state = th.load(path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    print(f"Loaded frozen RCDM: {sum(parameter.numel() for parameter in model.parameters()):,} parameters.",
          flush=True)
    return model.to(device).eval().requires_grad_(False), diffusion


@th.no_grad()
def sample(args):
    if args.num_images <= 0 or args.samples_per_image <= 0 or args.sampling_steps <= 0:
        raise ValueError("Sampling counts and timesteps must be positive.")
    cache = load_predictor_cache(args.cache)
    dino = load_dino_cache(args.dino_cache, cache)
    if len(cache["features"]) < 2 or dino["features"].shape[1] != 2048:
        raise ValueError("The released prior needs 2048-dimensional features and two images for controls.")
    if args.oracle_only and args.adapter:
        raise ValueError("Choose either an adapter or the oracle-only decoder check.")
    if not args.oracle_only and not args.adapter:
        raise ValueError("Supply --adapter for predictor decoding, or --oracle-only for the baseline.")
    device = configure(args)
    adapter = None if args.oracle_only else load_adapter(args.adapter, cache, dino["dino_spec"], device)
    if adapter is not None and adapter.output_mean.numel() != 2048:
        raise ValueError("The released prior requires 2048-dimensional DINO conditioning.")
    model, diffusion = load_prior(args.rcdm_checkpoint, device, args.sampling_steps)
    features = dino["features"].to(device) if adapter is None else adapter(cache["features"].to(device))
    count = min(args.num_images, len(features))
    prefix = "oracle" if adapter is None else "predictor"
    labels = ["original", "context", "target"]
    if adapter is not None:
        labels.append("DINO oracle")
    labels += [f"{prefix} {index + 1}" for index in range(args.samples_per_image)] + [f"wrong {prefix}"]
    rows, samples, controls, oracles = [], [], [], []
    started = time.monotonic()
    for index in range(count):
        noise = th.randn(args.samples_per_image, 3, 128, 128,
                         generator=th.Generator().manual_seed(args.seed + index)).to(device)
        wrong_index = (index + 1) % len(features)
        conditioning = features[index:index + 1].repeat(args.samples_per_image, 1)
        conditioning = th.cat([conditioning, features[wrong_index:wrong_index + 1]])
        noises = th.cat([noise, noise[:1]])
        if adapter is not None:
            conditioning = th.cat([dino["features"][index:index + 1].to(device), conditioning])
            noises = th.cat([noise[:1], noises])
        decoded = diffusion.ddim_sample_loop(model, noises.shape, noise=noises,
                                            model_kwargs={"feat": conditioning})
        if not th.isfinite(decoded).all():
            raise FloatingPointError("Pretrained RCDM produced non-finite samples.")
        decoded = ((decoded.cpu() + 1) * 127.5).clamp(0, 255).to(th.uint8)
        if adapter is not None:
            oracles.append(decoded[0])
            decoded = decoded[1:]
        samples.append(decoded[:-1])
        controls.append(decoded[-1])
        row = comparison_row(cache, index, 128)
        if adapter is not None:
            row.append(to_image(oracles[-1]))
        rows.append(row + [to_image(image) for image in decoded])
        print(f"Decoded {index + 1}/{count}: {cache['paths'][index]}, "
              f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    output = Path(args.output)
    save_comparison_grid(rows, labels, output)
    save_cache(dict(samples=th.stack(samples), wrong_condition=th.stack(controls),
                    oracle=th.stack(oracles) if oracles else None,
                    paths=cache["paths"][:count], seed=args.seed,
                    conditioning="dino_oracle" if adapter is None else "ijepa_predictor_adapter"),
               output.with_suffix(".pt"))
    print(f"Saved {output}", flush=True)


def create_argparser():
    parser = argparse.ArgumentParser(description="Frozen I-JEPA predictor adapter to a released DINO-RCDM prior.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto")
    common.add_argument("--seed", default=0, type=int)
    common.add_argument("--cpu-threads", default=4, type=int)
    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--cache-dir", default="artifacts/ijepa_features")
    prepare_parser.add_argument("--dino-checkpoint", default="artifacts/checkpoints/dino_resnet50_pretrain.pth")
    prepare_parser.add_argument("--out-dir", default="artifacts/dino_crop_features")
    prepare_parser.add_argument("--batch-size", default=16, type=int)
    prepare_parser.set_defaults(func=prepare)
    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--cache-dir", default="artifacts/ijepa_features")
    train_parser.add_argument("--dino-cache-dir", default="artifacts/dino_crop_features")
    train_parser.add_argument("--out-dir", default="artifacts/ijepa_adapter")
    train_parser.add_argument("--steps", default=2000, type=int)
    train_parser.add_argument("--batch-size", default=128, type=int)
    train_parser.add_argument("--hidden-dim", default=512, type=int)
    train_parser.add_argument("--lr", default=1e-3, type=float)
    train_parser.add_argument("--weight-decay", default=0.01, type=float)
    train_parser.add_argument("--log-interval", default=100, type=int)
    train_parser.set_defaults(func=train)
    sample_parser = subparsers.add_parser("sample", parents=[common])
    sample_parser.add_argument("--cache", default="artifacts/ijepa_features/val.pt")
    sample_parser.add_argument("--dino-cache", default="artifacts/dino_crop_features/val.pt")
    sample_parser.add_argument("--rcdm-checkpoint", default="artifacts/checkpoints/rcdm_ema_dino_trunk.pt")
    sample_parser.add_argument("--adapter", default="")
    sample_parser.add_argument("--oracle-only", action="store_true")
    sample_parser.add_argument("--num-images", default=4, type=int)
    sample_parser.add_argument("--samples-per-image", default=2, type=int)
    sample_parser.add_argument("--sampling-steps", default=100, type=int)
    sample_parser.add_argument("--output", default="artifacts/ijepa_adapter/predictor_samples.png")
    sample_parser.set_defaults(func=sample)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    args.func(args)

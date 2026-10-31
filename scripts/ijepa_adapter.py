import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from guided_diffusion_rcdm import dist_util
from guided_diffusion_rcdm.feature_datasets import different_image_index, distinct_examples, load_feature_cache
from guided_diffusion_rcdm.get_ssl_models import load_dino_trunk
from guided_diffusion_rcdm.ijepa import PredictorAdapter
from guided_diffusion_rcdm.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from scripts.ijepa_decoder import (
    comparison_row,
    feature_spec,
    hidden_crop,
    prediction_in_context,
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
        payload = dict(
            features=features,
            paths=cache["paths"],
            boxes=cache["boxes"],
            feature_spec=feature_spec(cache),
            dino_spec=dict(model="dino_resnet50_trunk", input_size=224,
                           normalization="rgb_minus_one_one", checkpoint=Path(args.dino_checkpoint).name),
        )
        if "image_indices" in cache:
            payload["image_indices"] = cache["image_indices"]
        if "image_hashes" in cache:
            payload["image_hashes"] = cache["image_hashes"]
        save_cache(payload, out_dir / f"{split}.pt")
    print(f"Saved aligned DINO feature caches to {out_dir}", flush=True)


def load_dino_cache(path, predictor_cache):
    cache = th.load(path, map_location="cpu", weights_only=True)
    if (cache["feature_spec"] != feature_spec(predictor_cache)
            or cache["paths"] != predictor_cache["paths"]
            or not th.equal(cache["boxes"], predictor_cache["boxes"])
            or ("image_indices" in cache) != ("image_indices" in predictor_cache)
            or ("image_indices" in cache and not th.equal(cache["image_indices"],
                                                         predictor_cache["image_indices"]))
            or cache.get("image_hashes") != predictor_cache.get("image_hashes")):
        raise ValueError("DINO targets do not match the predictor images, checkpoint, or hidden boxes.")
    features = cache["features"]
    if (not isinstance(features, th.Tensor) or features.ndim != 2
            or features.shape[0] != len(predictor_cache["features"])
            or not features.is_floating_point() or not th.isfinite(features).all()
            or (features < 0).any()):
        raise ValueError("DINO targets must be finite, nonnegative paired feature vectors.")
    spec = cache["dino_spec"]
    if (spec["model"] != "dino_resnet50_trunk" or spec["input_size"] != 224
            or spec["normalization"] != "rgb_minus_one_one"):
        raise ValueError("DINO feature extraction does not match the released RCDM.")
    return cache


def source_features(cache, source):
    if source == "predictor":
        return cache["features"]
    if source != "target":
        raise ValueError("Feature source must be predictor or target.")
    if "target_features" not in cache or cache.get("target_normalization") != "layer_norm_patch_tokens":
        raise ValueError("Target adapters require cached per-token-normalized target-encoder features.")
    return cache["target_features"]


@th.no_grad()
def adapter_metrics(model, features, targets, wrong_indices=None, output_scale=None):
    prediction = model(features)
    wrong = prediction.roll(1, 0) if wrong_indices is None else prediction[wrong_indices]
    scale = model.output_scale if output_scale is None else output_scale
    return dict(
        mse=(((prediction - targets) / scale) ** 2).mean().item(),
        wrong_mse=(((wrong - targets) / scale) ** 2).mean().item(),
        cosine=F.cosine_similarity(prediction, targets).mean().item(),
        wrong_cosine=F.cosine_similarity(wrong, targets).mean().item(),
    )


def train(args):
    if args.steps <= 0 or args.lr <= 0 or args.weight_decay < 0 or args.log_interval <= 0:
        raise ValueError("Use positive steps, learning rate, logging interval, and nonnegative weight decay.")
    if args.kind == "ridge" and (not np.isfinite(args.ridge_alpha) or args.ridge_alpha <= 0):
        raise ValueError("Ridge alpha must be finite and positive.")
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
    if set(predictor["train"].get("image_hashes", [])) & set(predictor["val"].get("image_hashes", [])):
        raise ValueError("Train and validation contain the same original image.")
    if not 0 < args.batch_size <= len(predictor["train"]["features"]):
        raise ValueError("Batch size must be positive and no larger than the training dataset.")
    if len(predictor["val"]["features"]) < 2:
        raise ValueError("Validation needs at least two examples for the wrong-conditioning control.")
    device = configure(args)
    input_dim = predictor["train"]["features"].shape[1]
    output_dim = dino["train"]["features"].shape[1]
    hidden_dim = 0 if args.kind == "ridge" else args.hidden_dim
    steps = 1 if args.kind == "ridge" else args.steps
    model = PredictorAdapter(input_dim, output_dim, hidden_dim, args.dropout)
    inputs = {split: source_features(cache, args.feature_source) for split, cache in predictor.items()}
    model.fit_statistics(inputs["train"], dino["train"]["features"])
    model.to(device)
    features = {split: values.float().to(device) for split, values in inputs.items()}
    targets = {split: cache["features"].float().to(device) for split, cache in dino.items()}
    wrong_indices = th.tensor([different_image_index(predictor["val"], index, -1)
                               for index in range(len(features["val"]))], device=device)
    if args.kind == "mlp":
        optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    config = dict(
        input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
        feature_spec=feature_spec(predictor["train"]), dino_spec=dino["train"]["dino_spec"],
        seed=args.seed, steps=steps, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
        kind=args.kind, ridge_alpha=args.ridge_alpha, feature_source=args.feature_source,
        target_normalization=predictor["train"].get("target_normalization") if args.feature_source == "target" else None,
        training_examples=len(features["train"]), training_images=len(set(predictor["train"]["paths"])),
        validation_examples=len(features["val"]), validation_images=len(set(predictor["val"]["paths"])),
    )
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    model.eval()
    baseline = adapter_metrics(model, features["val"], targets["val"], wrong_indices)
    print(f"Mean-feature baseline: {baseline}", flush=True)
    history = [dict(step=0, validation=baseline)]
    best = baseline["mse"]
    generator = th.Generator().manual_seed(args.seed)
    started = time.monotonic()
    for step in range(1, steps + 1):
        loss = None
        if args.kind == "ridge":
            model.fit_ridge(features["train"], targets["train"], args.ridge_alpha)
        else:
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
        if step % args.log_interval == 0 or step == steps:
            model.eval()
            metrics = adapter_metrics(model, features["val"], targets["val"], wrong_indices)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Adapter validation metrics are not finite.")
            history.append(dict(step=step, training_mse=loss.item() if loss is not None else None, validation=metrics))
            payload = dict(config, state_dict={key: value.detach().cpu()
                                              for key, value in model.state_dict().items()},
                           completed_steps=step, validation=metrics)
            save_cache(payload, out_dir / "last.pt")
            if metrics["mse"] < best:
                best = metrics["mse"]
                save_cache(payload, out_dir / "best.pt")
            (out_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
            print(f"step {step}/{steps}: {metrics}, "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    if not (out_dir / "best.pt").exists():
        raise RuntimeError("Adapter did not improve on mean DINO features; inspect last.pt and metrics.json.")
    print(f"Saved best adapter to {out_dir / 'best.pt'}", flush=True)


def load_adapter(path, predictor_cache, dino_spec, device, *, allow_mask_transfer=False):
    checkpoint = th.load(path, map_location="cpu", weights_only=True)
    trained_spec, input_spec = checkpoint["feature_spec"].copy(), feature_spec(predictor_cache)
    if allow_mask_transfer:
        trained_spec.pop("block_size")
        input_spec.pop("block_size")
    if (trained_spec != input_spec
            or checkpoint["dino_spec"] != dino_spec
            or checkpoint["input_dim"] != predictor_cache["features"].shape[1]):
        raise ValueError("Adapter checkpoint does not match the frozen predictor or DINO feature extraction.")
    if allow_mask_transfer and checkpoint["feature_spec"]["block_size"] != predictor_cache["block_size"]:
        print(f"Mask geometry transfer: adapter trained on {checkpoint['feature_spec']['block_size']}x"
              f"{checkpoint['feature_spec']['block_size']} patches; inference uses a different target shape.",
              flush=True)
    source = checkpoint.get("feature_source", "predictor")
    if source not in ("predictor", "target") or (
            source == "target" and checkpoint.get("target_normalization") != "layer_norm_patch_tokens"):
        raise ValueError("Adapter checkpoint has an incompatible target-feature normalization or source.")
    model = PredictorAdapter(checkpoint["input_dim"], checkpoint["output_dim"], checkpoint["hidden_dim"],
                             checkpoint.get("dropout", 0.1))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.feature_source = source
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
    if args.oracle_only and (args.adapter or args.target_adapter or args.reference_adapter):
        raise ValueError("Choose either adapters or the oracle-only decoder check.")
    if not args.oracle_only and not args.adapter:
        raise ValueError("Supply --adapter for predictor decoding, or --oracle-only for the baseline.")
    device = configure(args)
    adapter = None if args.oracle_only else load_adapter(args.adapter, cache, dino["dino_spec"], device)
    target_adapter = load_adapter(args.target_adapter, cache, dino["dino_spec"], device) if args.target_adapter else None
    reference_adapter = load_adapter(args.reference_adapter, cache, dino["dino_spec"], device) if args.reference_adapter else None
    for candidate in (adapter, target_adapter, reference_adapter):
        if candidate is not None and candidate.output_mean.numel() != 2048:
            raise ValueError("The released prior requires 2048-dimensional DINO conditioning.")
    if target_adapter is not None and target_adapter.feature_source != "target":
        raise ValueError("The target-encoder oracle requires an adapter trained with --feature-source target.")
    target_inputs = source_features(cache, "target") if target_adapter is not None else None
    model, diffusion = load_prior(args.rcdm_checkpoint, device, args.sampling_steps)
    predictor_features = cache["features"].float().to(device)
    features = dino["features"].float().to(device) if adapter is None else adapter(predictor_features)
    target_oracle = target_adapter(target_inputs.float().to(device)) if target_adapter is not None else None
    via_target = target_adapter(predictor_features) if target_adapter is not None else None
    reference = reference_adapter(predictor_features) if reference_adapter is not None else None
    indices = distinct_examples(cache, args.num_images)
    count = len(indices)
    prefix = "oracle" if adapter is None else "predictor"
    labels = ["original", "context", "target"]
    if adapter is not None:
        labels.append("DINO oracle")
    if target_adapter is not None:
        labels += ["IJEPA target", "via target"]
    if reference_adapter is not None:
        labels.append("reference")
    reference_labels = labels[3:].copy()
    labels += [f"{prefix} {index + 1}" for index in range(args.samples_per_image)]
    if adapter is not None:
        labels.append("in context")
    labels.append(f"wrong {prefix}")
    rows, samples, controls, oracles, references, composites = [], [], [], [], [], []
    started = time.monotonic()
    for row_index, index in enumerate(indices):
        noise = th.randn(args.samples_per_image, 3, 128, 128,
                         generator=th.Generator().manual_seed(args.seed + row_index)).to(device)
        wrong_index = different_image_index(cache, index)
        conditioning = features[index:index + 1].repeat(args.samples_per_image, 1)
        conditioning = th.cat([conditioning, features[wrong_index:wrong_index + 1]])
        noises = th.cat([noise, noise[:1]])
        reference_conditions = []
        if adapter is not None:
            reference_conditions.append(dino["features"][index:index + 1].to(device))
        if target_adapter is not None:
            reference_conditions += [target_oracle[index:index + 1], via_target[index:index + 1]]
        if reference_adapter is not None:
            reference_conditions.append(reference[index:index + 1])
        if reference_conditions:
            conditioning = th.cat([*reference_conditions, conditioning])
            noises = th.cat([noise[:1].repeat(len(reference_conditions), 1, 1, 1), noises])
        decoded = diffusion.ddim_sample_loop(model, noises.shape, noise=noises,
                                            model_kwargs={"feat": conditioning})
        if not th.isfinite(decoded).all():
            raise FloatingPointError("Pretrained RCDM produced non-finite samples.")
        decoded = ((decoded.cpu() + 1) * 127.5).clamp(0, 255).to(th.uint8)
        row = comparison_row(cache, index, 128)
        if adapter is not None:
            oracles.append(decoded[0])
        if reference_conditions:
            references.append(decoded[:len(reference_conditions)])
            row += [to_image(image) for image in references[-1]]
            decoded = decoded[len(reference_conditions):]
        samples.append(decoded[:-1])
        controls.append(decoded[-1])
        row += [to_image(image) for image in decoded[:-1]]
        if adapter is not None:
            composite = prediction_in_context(cache, index, decoded[0])
            composites.append(th.from_numpy(np.array(composite)).permute(2, 0, 1))
            row.append(composite)
        rows.append(row + [to_image(decoded[-1])])
        print(f"Decoded {row_index + 1}/{count}: {cache['paths'][index]}, "
              f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    output = Path(args.output)
    save_comparison_grid(rows, labels, output)
    save_cache(dict(samples=th.stack(samples), wrong_condition=th.stack(controls),
                    oracle=th.stack(oracles) if oracles else None,
                    reference_samples=th.stack(references) if references else None,
                    reference_labels=reference_labels,
                    composites=th.stack(composites) if composites else None,
                    paths=[cache["paths"][index] for index in indices],
                    boxes=cache["boxes"][indices], example_indices=th.tensor(indices), seed=args.seed,
                    sampling_steps=args.sampling_steps, adapter=args.adapter,
                    target_adapter=args.target_adapter, reference_adapter=args.reference_adapter,
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
    train_parser.add_argument("--dropout", default=0.1, type=float)
    train_parser.add_argument("--kind", choices=("mlp", "ridge"), default="mlp")
    train_parser.add_argument("--ridge-alpha", default=1.0, type=float)
    train_parser.add_argument("--feature-source", choices=("predictor", "target"), default="predictor")
    train_parser.add_argument("--lr", default=1e-3, type=float)
    train_parser.add_argument("--weight-decay", default=0.01, type=float)
    train_parser.add_argument("--log-interval", default=100, type=int)
    train_parser.set_defaults(func=train)
    sample_parser = subparsers.add_parser("sample", parents=[common])
    sample_parser.add_argument("--cache", default="artifacts/ijepa_features/val.pt")
    sample_parser.add_argument("--dino-cache", default="artifacts/dino_crop_features/val.pt")
    sample_parser.add_argument("--rcdm-checkpoint", default="artifacts/checkpoints/rcdm_ema_dino_trunk.pt")
    sample_parser.add_argument("--adapter", default="")
    sample_parser.add_argument("--target-adapter", default="")
    sample_parser.add_argument("--reference-adapter", default="")
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

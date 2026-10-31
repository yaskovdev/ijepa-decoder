import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image, ImageDraw

from guided_diffusion_rcdm import dist_util
from guided_diffusion_rcdm.feature_datasets import load_feature_cache
from guided_diffusion_rcdm.ijepa import load_ijepa_models, make_block_masks, predict_features
from guided_diffusion_rcdm.image_datasets import _list_image_files_recursively, center_crop_arr
from guided_diffusion_rcdm.script_util import (
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from scripts import image_train


def save_cache(cache, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    th.save(cache, temporary)
    temporary.replace(path)


def feature_spec(cache):
    spec = {key: cache[key] for key in (
        "conditioning", "input_size", "patch_size", "block_size", "ijepa_epoch"
    )}
    spec["checkpoint"] = Path(cache["ijepa_checkpoint"]).name
    return spec


def prepare(args):
    if args.batch_size <= 0 or args.train_limit <= 0 or args.val_limit <= 0:
        raise ValueError("Batch size and dataset limits must be positive.")
    if args.image_size not in (64, 128):
        raise ValueError("The pilot supports 64 or 128 pixel targets.")
    make_block_masks(0, 0, args.block_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        if (out_dir / f"{split}.pt").exists():
            raise FileExistsError(f"{out_dir / f'{split}.pt'} already exists; choose a new output directory.")
        if not (Path(args.data_dir) / split).is_dir():
            raise FileNotFoundError(f"Expected an explicit {split} image directory.")
    device = dist_util.set_device(args.device)
    th.manual_seed(args.seed)
    th.set_num_threads(args.cpu_threads)
    print(f"Loading frozen official ViT-H/14 encoder and predictor on {device}.", flush=True)
    encoder, predictor, epoch = load_ijepa_models(args.ijepa_repo, args.checkpoint, device)
    print(f"I-JEPA checkpoint epoch: {epoch}", flush=True)
    mean = th.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = th.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    started = time.monotonic()
    for split, limit in (("train", args.train_limit), ("val", args.val_limit)):
        rng = random.Random(args.seed + (split == "val"))
        files = _list_image_files_recursively(str(Path(args.data_dir) / split))
        if not files:
            raise ValueError(f"No images found in {split}.")
        rng.shuffle(files)
        files = files[:limit]
        originals, targets, boxes, contexts, hidden = [], [], [], [], []
        for filename in files:
            with Image.open(filename) as image:
                arr = center_crop_arr(image.convert("RGB"), 224).copy()
            original = th.from_numpy(arr).permute(2, 0, 1).contiguous()
            top, left = rng.randrange(17 - args.block_size), rng.randrange(17 - args.block_size)
            context, target = make_block_masks(top, left, args.block_size)
            box = (left * 14, top * 14, (left + args.block_size) * 14, (top + args.block_size) * 14)
            crop = Image.fromarray(arr).crop(box).resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
            originals.append(original)
            targets.append(th.from_numpy(np.array(crop)).permute(2, 0, 1).contiguous())
            boxes.append(box)
            contexts.append(context)
            hidden.append(target)
        originals = th.stack(originals)
        context_indices, target_indices = th.stack(contexts), th.stack(hidden)
        features = []
        for start in range(0, len(files), args.batch_size):
            batch = (originals[start:start + args.batch_size].float() / 255 - mean) / std
            context = context_indices[start:start + args.batch_size].to(device)
            target = target_indices[start:start + args.batch_size].to(device)
            feat = predict_features(encoder, predictor, batch.to(device), context, target).cpu()
            if split == "train" and start == 0:
                changed = batch.clone()
                x0, y0, x1, y1 = boxes[0]
                changed[:1, :, y0:y1, x0:x1] = 0
                counterfactual = predict_features(
                    encoder, predictor, changed.to(device), context, target
                ).cpu()
                th.testing.assert_close(feat, counterfactual, rtol=0, atol=0)
                print("Verified: changing hidden pixels does not change predictor conditioning.", flush=True)
            features.append(feat)
            print(f"{split}: {min(start + args.batch_size, len(files))}/{len(files)}, "
                  f"elapsed {time.monotonic() - started:.1f}s", flush=True)
        cache = dict(
            version=1,
            conditioning="ijepa_predictor",
            targets=th.stack(targets),
            features=th.cat(features),
            originals=originals,
            boxes=th.tensor(boxes),
            paths=[str(Path(filename).relative_to(args.data_dir)) for filename in files],
            input_size=224,
            patch_size=14,
            block_size=args.block_size,
            ijepa_epoch=epoch,
            ijepa_checkpoint=str(Path(args.checkpoint).resolve()),
            seed=args.seed,
        )
        save_cache(cache, out_dir / f"{split}.pt")
        print(f"Saved {out_dir / f'{split}.pt'}", flush=True)


def train(args):
    cache = load_feature_cache(args.cache)
    if cache.get("conditioning") != "ijepa_predictor":
        raise ValueError("This pilot requires cached I-JEPA predictor outputs.")
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.num_channels < 32 or args.num_channels % 32:
        raise ValueError("num_channels must be a positive multiple of 32.")
    if args.lr <= 0 or not 0 < args.ema_rate < 1:
        raise ValueError("Use a positive learning rate and an EMA rate between zero and one.")
    if args.save_interval <= 0 or args.log_interval <= 0:
        raise ValueError("Logging and saving intervals must be positive.")
    training_args = image_train.create_argparser().parse_args([])
    training_args.feature_cache = args.cache
    training_args.out_dir = args.out_dir
    training_args.device = args.device
    training_args.image_size = cache["targets"].shape[-1]
    training_args.feat_cond = True
    training_args.type_model = "ijepa_predictor"
    training_args.num_channels = args.num_channels
    training_args.num_res_blocks = 1
    training_args.channel_mult = "1,2,2,4"
    training_args.attention_resolutions = "16,8"
    training_args.num_heads = 4
    training_args.resblock_updown = True
    training_args.learn_sigma = True
    training_args.predict_xstart = args.predict_xstart
    training_args.batch_size = args.batch_size
    training_args.lr = args.lr
    training_args.ema_rate = str(args.ema_rate)
    training_args.max_steps = args.steps
    training_args.save_interval = args.save_interval
    training_args.log_interval = args.log_interval
    training_args.resume_checkpoint = args.resume_checkpoint
    training_args.seed = args.seed
    th.manual_seed(args.seed)
    th.set_num_threads(args.cpu_threads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "decoder_config.json"
    config = dict(vars(training_args), ssl_dim=cache["features"].shape[1],
                  conditioning=cache["conditioning"], feature_spec=feature_spec(cache))
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        keys = list(model_and_diffusion_defaults()) + [
            "ssl_dim", "conditioning", "feature_cache", "ema_rate", "feature_spec"
        ]
        if any(previous[key] != config[key] for key in keys):
            raise ValueError("Existing decoder configuration differs; choose a new output directory.")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    image_train.main(training_args)


def to_image(tensor):
    return Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy())


def hidden_crop(cache, index, size):
    original = to_image(cache["originals"][index])
    x0, y0, x1, y1 = cache["boxes"][index].tolist()
    if not (0 <= x0 < x1 <= original.width and 0 <= y0 < y1 <= original.height):
        raise ValueError("Hidden crop lies outside the cached original image.")
    return original.crop((x0, y0, x1, y1)).resize((size, size), Image.Resampling.BICUBIC)


def comparison_row(cache, index, size):
    original = to_image(cache["originals"][index])
    x0, y0, x1, y1 = cache["boxes"][index].tolist()
    target = hidden_crop(cache, index, size)
    context = original.copy()
    ImageDraw.Draw(context).rectangle((x0, y0, x1 - 1, y1 - 1), fill=(127, 127, 127))
    ImageDraw.Draw(original).rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 200, 0), width=2)
    return [original, context, target]


def save_comparison_grid(rows, labels, output, cell=128):
    grid = Image.new("RGB", (len(labels) * cell, len(rows) * (cell + 20)), "white")
    draw = ImageDraw.Draw(grid)
    for index, row in enumerate(rows):
        if len(row) != len(labels):
            raise ValueError("Comparison row does not match the column labels.")
        for column, image in enumerate(row):
            y = index * (cell + 20)
            draw.text((column * cell + 3, y + 3), labels[column], fill="black")
            grid.paste(image.resize((cell, cell), Image.Resampling.NEAREST), (column * cell, y + 20))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


@th.no_grad()
def sample(args):
    if args.num_images <= 0 or args.samples_per_image <= 0:
        raise ValueError("Sampling counts must be positive.")
    cache = load_feature_cache(args.cache)
    if len(cache["features"]) < 2:
        raise ValueError("At least two images are needed for the wrong-conditioning control.")
    config = json.loads((Path(args.checkpoint).parent / "decoder_config.json").read_text())
    if cache.get("conditioning") != config["conditioning"] or cache["features"].shape[1] != config["ssl_dim"]:
        raise ValueError("Sampling features do not match the trained decoder.")
    if cache["targets"].shape[-1] != config["image_size"]:
        raise ValueError("Sampling resolution does not match the trained decoder.")
    if feature_spec(cache) != config["feature_spec"]:
        raise ValueError("Sampling I-JEPA checkpoint or block geometry differs from training.")
    device = dist_util.set_device(args.device)
    th.set_num_threads(args.cpu_threads)
    config["timestep_respacing"] = f"ddim{args.sampling_steps}"
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(argparse.Namespace(**config), model_and_diffusion_defaults()),
        feat_cond=True,
        G_shared=config["no_shared"],
        ssl_dim=config["ssl_dim"],
    )
    model.load_state_dict(th.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval().requires_grad_(False)
    size = config["image_size"]
    count = min(args.num_images, len(cache["features"]))
    labels = ["original", "context", "target"] + [
        f"sample {index + 1}" for index in range(args.samples_per_image)
    ] + ["wrong condition"]
    cell = max(size, 128)
    samples, controls, rows = [], [], []
    for index in range(count):
        feature = cache["features"][index:index + 1].repeat(args.samples_per_image, 1).to(device)
        noise = th.randn(
            args.samples_per_image, 3, size, size,
            generator=th.Generator().manual_seed(args.seed + index),
        ).to(device)
        decoded = diffusion.ddim_sample_loop(model, noise.shape, noise=noise, model_kwargs={"feat": feature})
        wrong_index = (index + 1) % len(cache["features"])
        wrong = diffusion.ddim_sample_loop(
            model, noise[:1].shape, noise=noise[:1].clone(),
            model_kwargs={"feat": cache["features"][wrong_index:wrong_index + 1].to(device)},
        )
        if not th.isfinite(decoded).all() or not th.isfinite(wrong).all():
            raise FloatingPointError("Decoder produced non-finite samples.")
        decoded = ((decoded.cpu() + 1) * 127.5).clamp(0, 255).to(th.uint8)
        wrong = ((wrong.cpu() + 1) * 127.5).clamp(0, 255).to(th.uint8)
        samples.append(decoded)
        controls.append(wrong)
        row = comparison_row(cache, index, size)
        row += [to_image(image) for image in decoded]
        row += [to_image(wrong[0])]
        rows.append(row)
        print(f"Decoded {index + 1}/{count}: {cache['paths'][index]}", flush=True)
    output = Path(args.output)
    save_comparison_grid(rows, labels, output, cell)
    save_cache(dict(samples=th.stack(samples), wrong_condition=th.stack(controls),
                    paths=cache["paths"][:count], seed=args.seed), output.with_suffix(".pt"))
    print(f"Saved {output}", flush=True)


def create_argparser():
    parser = argparse.ArgumentParser(description="Small frozen I-JEPA predictor to RCDM experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto")
    common.add_argument("--seed", default=0, type=int)
    common.add_argument("--cpu-threads", default=4, type=int)
    prepare_parser = subparsers.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--ijepa-repo", default="artifacts/ijepa")
    prepare_parser.add_argument("--checkpoint", default="artifacts/checkpoints/IN1K-vit.h.14-300e.pth.tar")
    prepare_parser.add_argument("--data-dir", default="artifacts/data/imagenette2-160")
    prepare_parser.add_argument("--out-dir", default="artifacts/ijepa_features")
    prepare_parser.add_argument("--image-size", default=64, type=int)
    prepare_parser.add_argument("--block-size", default=7, type=int)
    prepare_parser.add_argument("--batch-size", default=4, type=int)
    prepare_parser.add_argument("--train-limit", default=1000, type=int)
    prepare_parser.add_argument("--val-limit", default=64, type=int)
    prepare_parser.set_defaults(func=prepare)
    train_parser = subparsers.add_parser("train", parents=[common])
    train_parser.add_argument("--cache", default="artifacts/ijepa_features/train.pt")
    train_parser.add_argument("--out-dir", default="artifacts/ijepa_decoder")
    train_parser.add_argument("--num-channels", default=32, type=int)
    train_parser.add_argument("--batch-size", default=16, type=int)
    train_parser.add_argument("--steps", default=2000, type=int)
    train_parser.add_argument("--lr", default=2e-4, type=float)
    train_parser.add_argument("--predict-xstart", action="store_true",
                              help="Predict clean RGB directly instead of noise; useful for a short diagnostic run.")
    train_parser.add_argument("--ema-rate", default=0.99, type=float)
    train_parser.add_argument("--save-interval", default=500, type=int)
    train_parser.add_argument("--log-interval", default=50, type=int)
    train_parser.add_argument("--resume-checkpoint", default="")
    train_parser.set_defaults(func=train)
    sample_parser = subparsers.add_parser("sample", parents=[common])
    sample_parser.add_argument("--cache", default="artifacts/ijepa_features/val.pt")
    sample_parser.add_argument("--checkpoint", required=True)
    sample_parser.add_argument("--output", default="artifacts/ijepa_decoder/predictor_samples.png")
    sample_parser.add_argument("--num-images", default=8, type=int)
    sample_parser.add_argument("--samples-per-image", default=4, type=int)
    sample_parser.add_argument("--sampling-steps", default=50, type=int)
    sample_parser.set_defaults(func=sample)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    args.func(args)

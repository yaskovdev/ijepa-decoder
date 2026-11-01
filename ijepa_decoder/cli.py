import argparse
import json
from pathlib import Path

import torch as th
from PIL import ImageDraw

from .data import (
    crop_targets, different_image_index, example_hashes, generate_manifest, input_spec,
    load_cache, original_for_example, prepare,
)
from .encoder import IJEPA_CHECKPOINT, load_ijepa, make_masks, predict_tokens
from .io import (
    as_image, compose, configure, download_file, image_pixels, normalized, rgb_pixels, save_cache, save_grid, save_json,
)
from .model import download_assets, load_decoder


PACKAGE = Path(__file__).parent


def download(args):
    manifest = json.loads((PACKAGE / "pretrained.json").read_text())
    for name, asset in manifest["files"].items():
        path = download_file(asset["url"], Path(args.out_dir) / name, asset["sha256"])
        print(f"Verified {path}", flush=True)


def assets(args):
    from .training import perceptual_loss

    data = download_assets(Path(args.models_dir) / "mae_hub")
    path = Path(args.assets)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    perceptual_loss("cpu", args.models_dir)
    print(f"Verified MAE and LPIPS training assets: {path}", flush=True)


@th.no_grad()
def decode(args):
    device = configure(args)
    box = [args.left * 14, args.top * 14, (args.left + 7) * 14, (args.top + 7) * 14]
    context, target = make_masks(box)
    model, metadata = load_decoder(args.decoder, device)
    if (metadata["feature_source"] not in ("predictor", "mixed")
            or metadata["input_spec"]["shape"] != [7, 7] or metadata["input_spec"]["feature_dim"] != 1280):
        raise ValueError("Image inference requires the released 7x7 I-JEPA predictor decoder.")
    encoder, predictor, _, epoch = load_ijepa(args.ijepa_repo, args.ijepa_checkpoint, device)
    backbone = metadata["input_spec"]["ijepa"]
    if epoch != backbone["ijepa_epoch"] or Path(args.ijepa_checkpoint).name != backbone["checkpoint"]:
        raise ValueError("Frozen I-JEPA checkpoint differs from the decoder's training backbone.")
    original = image_pixels(args.image)
    image = normalized(original[None], device)
    x0, y0, x1, y1 = box
    image[:, :, y0:y1, x0:x1] = 0
    tokens = predict_tokens(encoder, predictor, image, context[None].to(device), target[None].to(device)).half().float()
    prediction = rgb_pixels(model(tokens, th.tensor([box])))[0]
    composite = compose(original, prediction, box)
    marked = as_image(original)
    ImageDraw.Draw(marked).rectangle((x0, y0, x1 - 1, y1 - 1), outline="lime", width=2)
    visible = original.clone()
    visible[:, y0:y1, x0:x1] = 128
    save_grid([[marked, as_image(visible), as_image(prediction), as_image(composite)]],
              ["original", "context", "native predictor", "in context"], args.output, cell=224)
    output = Path(args.output)
    as_image(prediction).save(output.with_name(output.stem + "_crop.png"))
    as_image(composite).save(output.with_name(output.stem + "_composite.png"))
    save_cache(dict(tokens=tokens.cpu(), prediction=prediction, composite=composite,
                    box=box, ijepa_epoch=epoch, prediction_uses_hidden_rgb=False), output.with_suffix(".pt"))
    print(f"Saved {output}; mask coordinates refer to the 224x224 center-cropped view.", flush=True)


@th.no_grad()
def compare(args):
    from .training import metrics, paired_difference, perceptual_loss

    if min(args.num_images, args.batch_size) <= 0:
        raise ValueError("Comparison needs positive image and batch counts.")
    device = configure(args)
    cache = load_cache(args.cache)
    if args.num_images > len(cache["predictor_tokens"]):
        raise ValueError("Requested comparison images exceed the validation cache.")
    model, metadata = load_decoder(args.decoder, device, input_spec(cache))
    baseline, baseline_metadata = load_decoder(args.baseline, device, input_spec(cache))
    if any(record["feature_source"] not in ("predictor", "mixed") for record in (metadata, baseline_metadata)):
        raise ValueError("Primary comparisons require predictor-trained decoders, not target-only oracles.")
    perceptual, crops = perceptual_loss(device, args.models_dir), crop_targets(cache)
    wrong = th.tensor([different_image_index(cache, index) for index in range(len(crops))])
    permutation = th.randperm(cache["predictor_tokens"].shape[1], generator=th.Generator().manual_seed(args.seed))
    wrong_cache = dict(cache, predictor_tokens=cache["predictor_tokens"][wrong])
    shuffled_cache = dict(cache, predictor_tokens=cache["predictor_tokens"][:, permutation])
    cases = [
        ("v0.1_predictor", baseline, cache, "predictor"),
        ("scaled_predictor", model, cache, "predictor"),
        ("target_encoder_control", model, cache, "target"),
        ("wrong_predictor", model, wrong_cache, "predictor"),
        ("shuffled_tokens", model, shuffled_cache, "predictor"),
    ]
    report = {}
    for name, network, features, source in cases:
        report[name] = metrics(network, features, crops, source, perceptual, device, 0.5, args.batch_size,
                               per_example=True)
        print(f"{name}: { {key: report[name][key] for key in ('rgb_mae', 'lpips', 'score')} }", flush=True)
    hashes = example_hashes(cache)
    summary = dict(
        metrics=report, paired_difference=paired_difference(report["v0.1_predictor"],
                                                           report["scaled_predictor"], hashes, args.seed),
        image_hashes=hashes, boxes=cache["boxes"].tolist(), training_step=metadata["training_step"],
        baseline_training_step=baseline_metadata["training_step"], primary_uses_hidden_rgb=False,
        oracle_uses_hidden_rgb=True, wrong_control_geometry_matched=th.equal(cache["boxes"], cache["boxes"][wrong]),
    )
    rows, samples, composites = [], [], []
    for index in range(args.num_images):
        box = cache["boxes"][index:index + 1]
        decoded = rgb_pixels(th.cat([
            network(features[f"{source}_tokens"][index:index + 1].float().to(device), box)
            for _, network, features, source in cases
        ]))
        original = original_for_example(cache, index)
        x0, y0, x1, y1 = box[0].tolist()
        marked = as_image(original)
        ImageDraw.Draw(marked).rectangle((x0, y0, x1 - 1, y1 - 1), outline="lime", width=2)
        context = original.clone()
        context[:, y0:y1, x0:x1] = 128
        composite = compose(original, decoded[1], box[0].tolist())
        rows.append([marked, as_image(context), as_image(crops[index]), as_image(decoded[0]),
                     as_image(decoded[1]), as_image(composite), as_image(decoded[3]), as_image(decoded[4])])
        samples.append(decoded)
        composites.append(composite)
    save_grid(rows, ["original", "context", "true target", "v0.1 predictor", "scaled predictor",
                     "in context", "wrong predictor", "shuffled tokens"], args.output)
    output = Path(args.output)
    save_json(summary, output.with_suffix(".json"))
    save_cache(dict(samples=th.stack(samples), sample_labels=[name for name, *_ in cases],
                    composites=th.stack(composites), primary_uses_hidden_rgb=False),
               output.with_suffix(".pt"))
    print(f"Saved {output}; paired differences below zero favor the scaled model.", flush=True)


def parser():
    result = argparse.ArgumentParser(description="Unofficial I-JEPA predictor-to-RGB decoder.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    common.add_argument("--cpu-threads", type=int, default=4)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--ijepa-repo", default="artifacts/ijepa")
    common.add_argument("--ijepa-checkpoint", default=f"artifacts/checkpoints/{IJEPA_CHECKPOINT}")
    common.add_argument("--models-dir", default="artifacts/checkpoints/training")
    common.add_argument("--assets", default="artifacts/training-assets.json")
    commands = result.add_subparsers(dest="command", required=True)
    command = commands.add_parser("download", parents=[common])
    command.add_argument("--out-dir", default="artifacts/checkpoints/native")
    commands.add_parser("assets", parents=[common])
    command = commands.add_parser("decode", parents=[common])
    command.add_argument("--image", required=True)
    command.add_argument("--top", type=int, default=4)
    command.add_argument("--left", type=int, default=4)
    command.add_argument("--decoder", default="artifacts/checkpoints/native")
    command.add_argument("--output", default="artifacts/prediction.png")
    command = commands.add_parser("manifest", parents=[common])
    command.add_argument("--data-dir", default="artifacts/data/imagenette2-320")
    command.add_argument("--reference-manifest", default=str(PACKAGE / "manifest.json"))
    command.add_argument("--masks-per-image", type=int, default=4)
    command.add_argument("--validation-images", type=int, default=512)
    command.add_argument("--output", default="artifacts/scaled/manifest.json")
    command = commands.add_parser("prepare", parents=[common])
    command.add_argument("--manifest", default=str(PACKAGE / "manifest.json"))
    command.add_argument("--data-dir", default="artifacts/data/imagenette2-320")
    command.add_argument("--out-dir", default="artifacts/features")
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--limit", type=int)
    command.add_argument("--shard-size", type=int, default=0,
                         help="Photographs per resumable preparation shard; zero keeps the small-cache recipe.")
    command.add_argument("--resume", action="store_true")
    command = commands.add_parser("train", parents=[common])
    command.add_argument("--cache-dir", default="artifacts/features")
    command.add_argument("--out-dir", required=True)
    command.add_argument("--feature-source", choices=("target", "predictor", "mixed"), default="target")
    command.add_argument("--steps", type=int, default=1000)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--lpips-weight", type=float, default=0.5)
    command.add_argument("--schedule", choices=("constant", "cosine"), default="constant")
    command.add_argument("--min-learning-rate", type=float, default=0)
    command.add_argument("--ema-decay", type=float, default=0)
    command.add_argument("--eval-every", type=int, default=100)
    command.add_argument("--stop-after", type=int, help="Pause at this absolute step without shortening --steps.")
    initialization = command.add_mutually_exclusive_group()
    initialization.add_argument("--init")
    initialization.add_argument("--resume", help="Restore weights, optimizer, schedule, EMA, and RNG from last.pt.")
    command = commands.add_parser("evaluate", parents=[common])
    command.add_argument("--cache", default="artifacts/features/val.pt")
    command.add_argument("--decoder", default="artifacts/checkpoints/native")
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--output", help="Save metrics and paired per-example errors as JSON.")
    command = commands.add_parser("compare", parents=[common])
    command.add_argument("--cache", default="artifacts/scaled/features/val.pt")
    command.add_argument("--decoder", required=True)
    command.add_argument("--baseline", default="artifacts/checkpoints/native")
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--num-images", type=int, default=8)
    command.add_argument("--output", default="artifacts/scaled/comparison.png")
    return result


def main():
    args = parser().parse_args()
    if args.command in ("train", "evaluate"):
        from .training import evaluate, train

        {"train": train, "evaluate": evaluate}[args.command](args)
    else:
        {"download": download, "assets": assets, "decode": decode, "compare": compare, "manifest": generate_manifest,
         "prepare": prepare}[args.command](args)


if __name__ == "__main__":
    main()

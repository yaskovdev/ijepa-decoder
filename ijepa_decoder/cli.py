import argparse
import json
from pathlib import Path

import torch as th
from PIL import ImageDraw

from .data import prepare
from .encoder import IJEPA_CHECKPOINT, load_ijepa, make_masks, predict_tokens
from .io import (
    as_image, compose, configure, download_file, image_pixels, normalized, rgb_pixels, save_cache, save_grid,
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
    command = commands.add_parser("prepare", parents=[common])
    command.add_argument("--manifest", default=str(PACKAGE / "manifest.json"))
    command.add_argument("--data-dir", default="artifacts/data/imagenette2-320")
    command.add_argument("--out-dir", default="artifacts/features")
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--limit", type=int)
    command = commands.add_parser("train", parents=[common])
    command.add_argument("--cache-dir", default="artifacts/features")
    command.add_argument("--out-dir", required=True)
    command.add_argument("--feature-source", choices=("target", "predictor", "mixed"), default="target")
    command.add_argument("--steps", type=int, default=1000)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--lpips-weight", type=float, default=0.5)
    command.add_argument("--init")
    command = commands.add_parser("evaluate", parents=[common])
    command.add_argument("--cache", default="artifacts/features/val.pt")
    command.add_argument("--decoder", default="artifacts/checkpoints/native")
    command.add_argument("--batch-size", type=int, default=8)
    return result


def main():
    args = parser().parse_args()
    if args.command in ("train", "evaluate"):
        from .training import evaluate, train

        {"train": train, "evaluate": evaluate}[args.command](args)
    else:
        {"download": download, "assets": assets, "decode": decode, "prepare": prepare}[args.command](args)


if __name__ == "__main__":
    main()

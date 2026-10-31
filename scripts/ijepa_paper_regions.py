import json
import math
import time
from pathlib import Path

import numpy as np
import torch as th
from PIL import Image, ImageDraw

from guided_diffusion_rcdm import dist_util
from guided_diffusion_rcdm.ijepa import load_ijepa_models, make_rectangle_masks, predict_features
from scripts.ijepa_adapter import load_adapter, load_prior
from scripts.ijepa_decoder import save_cache, save_comparison_grid, to_image


ROOT = Path("artifacts/ijepa_paper_regions")
REGIONS = [
    {"name": "dog", "crop_box": [1, 1, 198, 198], "target_box": [50, 11, 147, 109]},
    {"name": "bird", "crop_box": [202, 1, 399, 198], "target_box": [25, 88, 122, 187]},
    {"name": "wolf", "crop_box": [402, 1, 599, 199], "target_box": [12, 101, 110, 198]},
    {"name": "building", "crop_box": [603, 1, 800, 198], "target_box": [100, 32, 197, 132]},
]
MODELS = {
    "predictor": "artifacts/ijepa_adapter_320_ridge0.1/best.pt",
    "MLP": "artifacts/ijepa_adapter_320_mlp256/best.pt",
    "via target": "artifacts/ijepa_target_adapter_320_ridge0.1/best.pt",
}


def prepare(source, box):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= source.width and 0 <= y0 < y1 <= source.height):
        raise ValueError("Requested rectangle lies outside the supplied tile.")
    # Remove the published sketch before resizing, including interpolation halos.
    cleaned = source.copy()
    ImageDraw.Draw(cleaned).rectangle((x0, y0, x1 - 1, y1 - 1), fill=(128, 128, 128))
    pixels = np.array(cleaned.resize((224, 224), Image.Resampling.BICUBIC))
    left = math.floor(x0 * 224 / source.width / 14)
    top = math.floor(y0 * 224 / source.height / 14)
    right = math.ceil(x1 * 224 / source.width / 14)
    bottom = math.ceil(y1 * 224 / source.height / 14)
    context, target = make_rectangle_masks(top, left, bottom - top, right - left)
    aligned = (left * 14, top * 14, right * 14, bottom * 14)
    ax0, ay0, ax1, ay1 = aligned
    pixels[ay0:ay1, ax0:ax1] = 128
    return th.from_numpy(pixels).permute(2, 0, 1).contiguous(), context, target, aligned


def compose(source, box, aligned, prediction):
    x0, y0, x1, y1 = box
    ax0, ay0, ax1, ay1 = aligned
    patch = to_image(prediction)
    extent = ((x0 * 224 / source.width - ax0) * patch.width / (ax1 - ax0),
              (y0 * 224 / source.height - ay0) * patch.height / (ay1 - ay0),
              (x1 * 224 / source.width - ax0) * patch.width / (ax1 - ax0),
              (y1 * 224 / source.height - ay0) * patch.height / (ay1 - ay0))
    patch = patch.transform((x1 - x0, y1 - y0), Image.Transform.EXTENT, extent, Image.Resampling.BICUBIC)
    result = source.copy()
    result.paste(patch, (x0, y0))
    actual, original = np.array(result), np.array(source)
    visible = np.ones(actual.shape[:2], dtype=bool)
    visible[y0:y1, x0:x1] = False
    if not np.array_equal(actual[visible], original[visible]):
        raise AssertionError("Composition changed visible source pixels.")
    return result


@th.no_grad()
def main():
    if (ROOT / "predictions.png").exists():
        raise FileExistsError("This experiment already has results.")
    th.set_num_threads(4)
    th.manual_seed(0)
    device = dist_util.set_device("auto")
    with Image.open(ROOT / "source_figure.png") as image:
        figure = image.convert("RGB")
    if figure.size != (800, 199):
        raise ValueError("This experiment requires the original 800x199 I-JEPA README figure.")
    regions = [dict(region) for region in REGIONS]
    sources, inputs, contexts, targets, aligned_boxes = [], [], [], [], []
    rng = np.random.default_rng(0)
    for region in regions:
        source = figure.crop(region["crop_box"])
        pixels, context, target, aligned = prepare(source, region["target_box"])
        changed = source.copy()
        x0, y0, x1, y1 = region["target_box"]
        changed.paste(Image.fromarray(rng.integers(256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)),
                      (x0, y0))
        if not th.equal(pixels, prepare(changed, region["target_box"])[0]):
            raise AssertionError("Published hidden pixels leaked through image resizing.")
        sources.append(source)
        inputs.append(pixels)
        contexts.append(context)
        targets.append(target)
        aligned_boxes.append(aligned)
        region["model_box"] = list(aligned)
        region["target_grid_shape"] = [(aligned[3] - aligned[1]) // 14, (aligned[2] - aligned[0]) // 14]
        print(region, flush=True)
    encoder, predictor, epoch = load_ijepa_models(
        "artifacts/ijepa", "artifacts/checkpoints/IN1K-vit.h.14-300e.pth.tar", device
    )
    mean = th.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = th.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    features = []
    for index, pixels in enumerate(inputs):
        image = (pixels[None].float().to(device) / 255 - mean) / std
        context, target = contexts[index][None].to(device), targets[index][None].to(device)
        feature = predict_features(encoder, predictor, image, context, target)
        changed = image.clone()
        x0, y0, x1, y1 = aligned_boxes[index]
        changed[:, :, y0:y1, x0:x1] = th.randn_like(changed[:, :, y0:y1, x0:x1])
        if not th.equal(feature, predict_features(encoder, predictor, changed, context, target)):
            raise AssertionError("Hidden patch pixels changed predictor conditioning.")
        features.append(feature.cpu())
    cache = dict(features=th.cat(features), conditioning="ijepa_predictor", input_size=224,
                 patch_size=14, block_size=None, ijepa_epoch=epoch,
                 ijepa_checkpoint="IN1K-vit.h.14-300e.pth.tar")
    if not th.isfinite(cache["features"]).all():
        raise FloatingPointError("Non-finite predictor features.")
    save_cache(dict(cache, regions=regions, contexts=th.stack(inputs)), ROOT / "features.pt")
    (ROOT / "regions.json").write_text(json.dumps(regions, indent=2) + "\n")
    print("Verified all four cases: no sketch pixels enter the predictor.", flush=True)
    del encoder, predictor
    dino_spec = th.load(MODELS["predictor"], weights_only=True, map_location="cpu")["dino_spec"]
    if (dino_spec["model"], dino_spec["input_size"], dino_spec["normalization"]) != (
            "dino_resnet50_trunk", 224, "rgb_minus_one_one"):
        raise ValueError("Adapter feature family does not match the released prior.")
    adapters = {name: load_adapter(path, cache, dino_spec, device, allow_mask_transfer=True)
                for name, path in MODELS.items()}
    if adapters["via target"].feature_source != "target":
        raise ValueError("The comparison must use a target-trained adapter on predictor vectors.")
    if any(model.output_mean.numel() != 2048 for model in adapters.values()):
        raise ValueError("Released RCDM requires 2048-dimensional features.")
    mapped = {name: model(cache["features"].to(device)) for name, model in adapters.items()}
    prior, diffusion = load_prior("artifacts/checkpoints/rcdm_ema_dino_trunk.pt", device, 100)
    labels = ["Meta reference", "context", "predictor 1", "predictor 2", "in context", "in context 2",
              "MLP", "MLP context", "via target", "target context", "other region"]
    rows, samples, composites = [], [], []
    started = time.monotonic()
    for index, region in enumerate(regions):
        noise = th.randn(2, 3, 128, 128, generator=th.Generator().manual_seed(index)).to(device)
        features = th.cat([mapped["predictor"][index:index + 1].repeat(2, 1),
                           mapped["MLP"][index:index + 1], mapped["via target"][index:index + 1],
                           mapped["predictor"][(index + 1) % len(regions)][None]])
        noises = th.cat([noise, noise[:1].repeat(3, 1, 1, 1)])
        decoded = diffusion.ddim_sample_loop(prior, noises.shape, noise=noises, model_kwargs={"feat": features})
        if not th.isfinite(decoded).all():
            raise FloatingPointError("RCDM produced non-finite images.")
        decoded = ((decoded.cpu() + 1) * 127.5).clamp(0, 255).to(th.uint8)
        placed = [compose(sources[index], region["target_box"], aligned_boxes[index], decoded[slot])
                  for slot in (0, 1, 2, 3)]
        for slot, image in zip(("predictor", "predictor2", "mlp", "target"), placed):
            image.save(ROOT / f"{region['name']}_{slot}_context.png")
        rows.append([sources[index], to_image(inputs[index]), to_image(decoded[0]), to_image(decoded[1]),
                     placed[0], placed[1], to_image(decoded[2]), placed[2], to_image(decoded[3]), placed[3],
                     to_image(decoded[4])])
        samples.append(decoded)
        composites.append([th.from_numpy(np.array(image)).permute(2, 0, 1) for image in placed])
        print(f"Decoded {region['name']}: elapsed {time.monotonic() - started:.1f}s", flush=True)
    save_comparison_grid(rows, labels, ROOT / "predictions.png", cell=160)
    save_cache(dict(regions=regions, samples=th.stack(samples), composites=composites,
                    composite_labels=["predictor 1", "predictor 2", "MLP", "via target"],
                    adapters=MODELS, conditioning="ijepa_predictor", ground_truth_available=False,
                    sampling_steps=100, seed=0, mask_geometry_transfer=True),
               ROOT / "predictions.pt")
    print(f"Saved {ROOT / 'predictions.png'}", flush=True)


if __name__ == "__main__":
    main()

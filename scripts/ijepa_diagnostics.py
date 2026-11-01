import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F

from guided_diffusion_rcdm.feature_datasets import load_feature_cache
from guided_diffusion_rcdm.rae import load_decoder, regrid
from scripts.ijepa_decoder import (
    comparison_row, feature_spec, hidden_crop, prediction_in_context, save_cache,
    save_comparison_grid, to_image,
)
from scripts.ijepa_rae import configure, load_spatial_cache, pixels, read_assets, spatial_spec


@th.no_grad()
def project_grid(tokens, source_shape):
    if (tokens.ndim != 3 or tokens.shape[1] != 256 or not tokens.is_floating_point()
            or not th.isfinite(tokens).all() or min(source_shape) <= 0 or max(source_shape) > 16):
        raise ValueError("Spatial projection needs finite 16x16 tokens and a source grid no larger than 16x16.")
    basis = regrid(th.eye(source_shape[0] * source_shape[1], dtype=th.float64)[None], source_shape)[0]
    orthogonal, _ = th.linalg.qr(basis, mode="reduced")
    projected = orthogonal @ (orthogonal.T @ tokens.cpu().double())
    return projected.to(tokens)


def example_hashes(cache):
    if "image_indices" not in cache or "image_hashes" not in cache:
        raise ValueError("Retrieval requires original-image indices and pixel hashes.")
    return [cache["image_hashes"][index] for index in cache["image_indices"].tolist()]


def retrieval_features(cache, source):
    if source == "predictor":
        values = cache["features"]
    elif source == "target":
        if "target_features" in cache:
            values = cache["target_features"]
        elif "target_tokens" in cache:
            values = cache["target_tokens"].float().mean(1)
        else:
            raise ValueError("Retrieval requires cached frozen target-encoder features.")
    else:
        raise ValueError("Retrieval source must be predictor or target.")
    if (values.shape != cache["features"].shape or not values.is_floating_point()
            or not th.isfinite(values).all()):
        raise ValueError("Retrieval features must be paired finite pooled I-JEPA vectors.")
    return values.float()


def validate_retrieval_caches(bank, query):
    if (feature_spec(bank) != feature_spec(query)
            or bank["features"].shape[1] != query["features"].shape[1]
            or any(cache.get("target_normalization") != "layer_norm_patch_tokens" for cache in (bank, query))):
        raise ValueError("Retrieval caches differ in frozen I-JEPA model, geometry, or normalization.")
    bank_hashes, query_hashes = example_hashes(bank), example_hashes(query)
    if set(bank_hashes).intersection(query_hashes):
        raise ValueError("Retrieval bank and query share original image pixels.")
    retrieval_features(bank, "target")
    retrieval_features(query, "target")
    return bank_hashes, query_hashes


@th.no_grad()
def nearest_examples(bank, queries, image_hashes, top_k, center=False):
    bank, queries = bank.float().cpu(), queries.float().cpu()
    if (bank.ndim != 2 or queries.ndim != 2 or bank.shape[1] != queries.shape[1]
            or min(len(bank), len(queries)) <= 0 or len(image_hashes) != len(bank)
            or not 0 < top_k <= len(set(image_hashes))
            or not th.isfinite(bank).all() or not th.isfinite(queries).all()):
        raise ValueError("Nearest-neighbor search needs paired finite vectors and enough distinct original images.")
    if center:
        mean = bank.mean(0, keepdim=True)
        bank, queries = bank - mean, queries - mean
    norms = (bank.norm(dim=1), queries.norm(dim=1))
    if any((value == 0).any() or not th.isfinite(value).all() for value in norms):
        raise ValueError("Cannot compare zero-norm or non-finite-norm retrieval vectors.")
    similarities = F.normalize(queries, dim=1) @ F.normalize(bank, dim=1).T
    ranked = similarities.argsort(dim=1, descending=True, stable=True)
    indices = []
    for order in ranked.tolist():
        seen, chosen = set(), []
        for index in order:
            if image_hashes[index] not in seen:
                seen.add(image_hashes[index])
                chosen.append(index)
                if len(chosen) == top_k:
                    break
        indices.append(chosen)
    indices = th.tensor(indices, dtype=th.long)
    return indices, similarities.gather(1, indices)


@th.no_grad()
def spatial_oracle(args):
    if args.batch_size <= 0:
        raise ValueError("Oracle batch size must be positive.")
    device = configure(args)
    assets = read_assets(args.assets)
    cache = load_spatial_cache(args.cache)
    decoder = load_decoder(args.rae_repo, assets["decoder_path"], assets["encoder_dir"], device)
    height, width = cache["spatial_shape"]
    if height != width:
        raise ValueError("The resize control currently requires a square source grid.")
    names = ["native DINO", "resize 7x7", "resize + LN", "best 7x7 grid", "best + LN", "pooled DINO"]
    names = [name.replace("7x7", f"{height}x{width}") for name in names]
    samples, errors, feature_errors, cosines = [], [], [], []
    started = time.monotonic()
    for start in range(0, len(cache["features"]), args.batch_size):
        end = min(start + args.batch_size, len(cache["features"]))
        true = cache["dino_tokens"][start:end].float()
        resized = regrid(regrid(true, (16, 16), output_size=height), (height, width))
        projected = project_grid(true, (height, width))
        tokens = th.stack([
            true, resized, F.layer_norm(resized, (768,), eps=1e-6),
            projected, F.layer_norm(projected, (768,), eps=1e-6),
            true.mean(1, keepdim=True).expand_as(true),
        ])
        decoded = pixels(decoder(tokens.flatten(0, 1).to(device))).reshape(
            len(names), end - start, 3, 256, 256).transpose(0, 1)
        target = cache["targets"][start:end].float()[:, None] / 255
        samples.append(decoded)
        errors.append((decoded.float() / 255 - target).abs().mean((2, 3, 4)))
        feature_errors.append((tokens - true[None]).square().mean((2, 3)).T)
        cosines.append(F.cosine_similarity(tokens, true[None], dim=-1).mean(-1).T)
        print(f"Spatial oracle: {end}/{len(cache['features'])}; elapsed {time.monotonic() - started:.1f}s",
              flush=True)
    samples, errors = th.cat(samples), th.cat(errors)
    feature_errors, cosines = th.cat(feature_errors), th.cat(cosines)
    indices = list(range(min(args.num_images, len(samples))))
    rows = [comparison_row(cache, index, 256) + [to_image(image) for image in samples[index]]
            for index in indices]
    save_comparison_grid(rows, ["original", "context", "true target"] + names, args.output, cell=160)
    result = dict(
        examples=len(samples), source_shape=[height, width], input_spec=spatial_spec(cache),
        conditioning="true_hidden_rgb_dino_tokens", ground_truth_available=True,
        rgb_mae=dict(zip(names, errors.mean(0).tolist())),
        token_mse=dict(zip(names, feature_errors.mean(0).tolist())),
        token_cosine=dict(zip(names, cosines.mean(0).tolist())),
        projection_note="CPU float64 QR gives the closest bilinear source-grid representation in token MSE, "
                        "not the best possible RGB reconstruction. LayerNorm variants leave that linear subspace.",
    )
    save_cache(dict(result, samples=samples, image_rgb_mae=errors, grid_indices=indices,
                    sample_labels=names), Path(args.output).with_suffix(".pt"))
    Path(args.output).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Spatial RGB MAE: {result['rgb_mae']}. Saved {args.output}", flush=True)


@th.no_grad()
def retrieve(args):
    if args.top_k <= 0:
        raise ValueError("Retrieval neighbor count must be positive.")
    configure(args)
    started = time.monotonic()
    bank, query = load_feature_cache(args.bank), load_feature_cache(args.cache)
    bank_hashes, query_hashes = validate_retrieval_caches(bank, query)
    first_by_hash = {}
    for index, value in enumerate(query_hashes):
        first_by_hash.setdefault(value, index)
    indices = list(first_by_hash.values())
    if len(indices) < 2:
        raise ValueError("Retrieval controls need at least two distinct query images.")
    wrong_indices = indices[1:] + indices[:1]
    targets = retrieval_features(bank, "target")
    predicted = retrieval_features(query, "predictor")[indices]
    actual = retrieval_features(query, "target")[indices]
    wrong = retrieval_features(query, "predictor")[wrong_indices]
    features = dict(predictor=predicted, centered_predictor=predicted, target_oracle=actual,
                    centered_target_oracle=actual, wrong_predictor=wrong, centered_wrong_predictor=wrong)
    found, similarities = {}, {}
    for name, values in features.items():
        found[name], similarities[name] = nearest_examples(
            targets, values, bank_hashes, args.top_k, center=name.startswith("centered"))
    labels = ["original", "context", "true target", "predictor NN1", "retrieved context"]
    labels += [f"predictor NN{k + 1}" for k in range(1, args.top_k)]
    labels += ["target oracle NN1", "wrong predictor NN1"]
    centered_labels = [label.replace("predictor NN", "centered NN") for label in labels]
    rows, centered_rows, records, composites, crop_samples = [], [], [], [], []
    for row, index in enumerate(indices):
        crops = {name: [hidden_crop(bank, neighbor, 256) for neighbor in neighbors[row].tolist()]
                 for name, neighbors in found.items()}
        top1 = {name: th.from_numpy(np.array(crops[name][0])).permute(2, 0, 1) for name in features}
        composite = prediction_in_context(query, index, top1["predictor"])
        centered_composite = prediction_in_context(query, index, top1["centered_predictor"])
        if row < args.num_images:
            rows.append(comparison_row(query, index, 256) + [
                crops["predictor"][0], composite,
            ] + crops["predictor"][1:] + [
                crops["target_oracle"][0], crops["wrong_predictor"][0],
            ])
            centered_rows.append(comparison_row(query, index, 256) + [
                crops["centered_predictor"][0], centered_composite,
            ] + crops["centered_predictor"][1:] + [
                crops["centered_target_oracle"][0], crops["centered_wrong_predictor"][0],
            ])
        composites.append(th.stack([
            th.from_numpy(np.array(composite)).permute(2, 0, 1),
            th.from_numpy(np.array(centered_composite)).permute(2, 0, 1),
        ]))
        crop_samples.append(th.stack(list(top1.values())))
        record = dict(query_index=index, query_path=query["paths"][index],
                      query_box=query["boxes"][index].tolist(), wrong_query_index=wrong_indices[row])
        for name in features:
            record[name] = [dict(index=neighbor, path=bank["paths"][neighbor],
                                 box=bank["boxes"][neighbor].tolist(), image_hash=bank_hashes[neighbor],
                                 cosine=similarities[name][row, rank].item())
                            for rank, neighbor in enumerate(found[name][row].tolist())]
        records.append(record)
    samples = th.stack(crop_samples)
    truth = th.stack([th.from_numpy(np.array(hidden_crop(query, index, 256))).permute(2, 0, 1)
                      for index in indices])
    errors = (samples.float() / 255 - truth.float()[:, None] / 255).abs().mean((2, 3, 4))
    summary = dict(
        examples=len(indices), bank_crops=len(targets), bank_originals=len(set(bank_hashes)),
        rgb_mae=dict(zip(features, errors.mean(0).tolist())),
        predictor_target_cosine=F.cosine_similarity(predicted, actual, dim=1).mean().item(),
        predictor_target_same_top1=(found["predictor"][:, 0] == found["target_oracle"][:, 0]).float().mean().item(),
        centering="Subtract the training target-bank mean before cosine normalization.",
    )
    bank_classes = [Path(path).parent.name for path in bank["paths"]]
    query_classes = [Path(query["paths"][index]).parent.name for index in indices]
    if all(re.fullmatch(r"n[0-9]{8}", value) for value in bank_classes + query_classes):
        summary["original_photo_class_top1"] = {
            name: sum(bank_classes[neighbor[0]] == label
                      for neighbor, label in zip(neighbors.tolist(), query_classes)) / len(indices)
            for name, neighbors in found.items()
        }
        summary["original_photo_class_topk"] = {
            name: sum(label in [bank_classes[neighbor] for neighbor in neighbors]
                      for neighbors, label in zip(matches.tolist(), query_classes)) / len(indices)
            for name, matches in found.items()
        }
        summary["class_note"] = (
            "Imagenette classes label original photos, not necessarily the masked crop; "
            "class agreement is not proof of matching pose or pixels."
        )
    else:
        summary["class_note"] = "Class metrics omitted: paths do not expose consistent ImageNet synset labels."
        print(summary["class_note"], flush=True)
    save_comparison_grid(rows, labels, args.output, cell=160)
    output = Path(args.output)
    centered_output = output.with_name(output.stem + "_centered" + output.suffix)
    save_comparison_grid(centered_rows, centered_labels, centered_output, cell=160)
    payload = dict(
        input_spec=feature_spec(query), target_normalization=query["target_normalization"],
        bank_path=str(Path(args.bank).resolve()), query_path=str(Path(args.cache).resolve()),
        seed=args.seed, conditioning="ijepa_predictor_nearest_training_target_features",
        primary_uses_hidden_rgb=False, output_type="retrieved_training_rgb_crops", metrics=summary, records=records,
        images=dict(cosine=str(output), centered_cosine=str(centered_output)),
    )
    Path(args.output).with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    save_cache(dict(payload, samples=samples, sample_labels=list(features),
                    composites=th.stack(composites), indices=indices,
                    grid_indices=indices[:args.num_images]), Path(args.output).with_suffix(".pt"))
    print(f"Retrieval: {summary}; elapsed {time.monotonic() - started:.1f}s. Saved {args.output}", flush=True)


def parser():
    result = argparse.ArgumentParser(description="Independent spatial and retrieval controls for I-JEPA decoding.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", default="artifacts/ijepa_rae/features/val.pt")
    common.add_argument("--num-images", type=int, default=8)
    common.add_argument("--output", required=True)
    common.add_argument("--device", default="cpu")
    common.add_argument("--cpu-threads", type=int, default=4)
    common.add_argument("--seed", type=int, default=0)
    commands = result.add_subparsers(dest="command", required=True)
    command = commands.add_parser("spatial-oracle", parents=[common])
    command.add_argument("--assets", default="artifacts/ijepa_rae/assets.json")
    command.add_argument("--rae-repo", default="artifacts/rae")
    command.add_argument("--batch-size", type=int, default=4)
    command = commands.add_parser("retrieve", parents=[common])
    command.add_argument("--bank", default="artifacts/ijepa_features_320_m3/train.pt")
    command.add_argument("--top-k", type=int, default=3)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    {"spatial-oracle": spatial_oracle, "retrieve": retrieve}[args.command](args)

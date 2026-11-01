import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch as th
from PIL import Image
from safetensors.torch import save_file

from ijepa_decoder import cli, data
from ijepa_decoder.encoder import IJEPA_CHECKPOINT
from ijepa_decoder.io import compose, download_file, image_pixels
from ijepa_decoder.model import NativeDecoder, initialization_spec, load_decoder, mask_order, resize_rgb_head
from ijepa_decoder.training import evaluate, train


def fake_encoder(images, masks=None):
    values = th.nn.functional.avg_pool2d(images, 14).flatten(2).transpose(1, 2)
    values = th.cat([values, th.ones_like(values[..., :1])], dim=-1)
    if masks is not None:
        values = values.gather(1, masks[0][..., None].expand(-1, -1, 4))
    return values


def fake_predictor(context, context_indices, target_indices):
    return context.mean(1, keepdim=True) + target_indices[0][..., None].float() / 256


class DecoderTests(unittest.TestCase):
    def setUp(self):
        th.set_num_threads(2)
        th.manual_seed(0)

    def test_native_positions_and_head_channel_order(self):
        boxes = th.tensor([[28, 14, 126, 112]])
        restore, indices = mask_order(boxes, (7, 7))
        self.assertEqual(indices[0, :7].tolist(), list(range(18, 25)))
        sequence = th.full((1, 256), -1)
        sequence[:, :49] = th.arange(49)
        th.testing.assert_close(sequence.gather(1, restore).gather(1, indices), th.arange(49)[None])
        bias = th.tensor([1., 10., 100., 2., 20., 200., 3., 30., 300., 4., 40., 400.])
        weight = th.stack([bias, bias * 2], dim=1)
        resized, colors = resize_rgb_head(weight, bias, 1)
        th.testing.assert_close(colors, th.tensor([2.5, 25., 250.]))
        th.testing.assert_close(resized, th.stack([colors, colors * 2], dim=1))

    def test_safetensors_restore_gradients_and_reject_other_geometry(self):
        model = NativeDecoder(8, (7, 7), dict(decoder_hidden_size=24, decoder_num_hidden_layers=1,
                                           decoder_num_attention_heads=4, decoder_intermediate_size=48))
        tokens, boxes = th.randn(1, 49, 8, requires_grad=True), th.tensor([[28, 14, 126, 112]])
        expected = model(tokens, boxes)
        expected.square().mean().backward()
        self.assertGreater(tokens.grad.abs().sum().item(), 0)
        spec = dict(feature_dim=8, shape=[7, 7], target_normalization="layer_norm_patch_tokens",
                    output="native_rgb_patch14_fixed_imagenet")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file(model.state_dict(), str(path / "model.safetensors"))
            (path / "config.json").write_text(json.dumps(dict(
                config=model.config.to_dict(), input_spec=spec, initialization=initialization_spec(),
                feature_source="mixed",
            )))
            restored, _ = load_decoder(path, "cpu", spec)
            th.testing.assert_close(restored(tokens.detach(), boxes), expected.detach())
            with self.assertRaisesRegex(ValueError, "geometry"):
                load_decoder(path, "cpu", dict(spec, shape=[1, 49]))

    def test_native_rgb_crops_use_mapped_originals_and_compositing_preserves_visible_pixels(self):
        originals = th.zeros(2, 3, 224, 224, dtype=th.uint8)
        originals[0] = 11
        originals[1] = 99
        cache = dict(originals=originals, image_indices=th.tensor([1, 0]),
                     boxes=th.tensor([[28, 14, 126, 112]] * 2))
        targets = data.crop_targets(cache)
        self.assertEqual(targets.shape, (2, 3, 98, 98))
        self.assertTrue((targets[0] == 99).all())
        output = compose(originals[0], targets[0], cache["boxes"][0].tolist())
        expected = originals[0].clone()
        expected[:, 14:112, 28:126] = 99
        th.testing.assert_close(output, expected)

    def test_manifest_preparation_preserves_exact_masks_and_rejects_changed_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(ijepa_epoch=299, splits={})
            for split, colors in (("train", [(100, 30, 50), (40, 120, 80)]),
                                  ("val", [(80, 40, 180), (10, 200, 90)])):
                (root / split).mkdir()
                manifest["splits"][split] = []
                for index, color in enumerate(colors):
                    path = root / split / f"{index}.png"
                    Image.new("RGB", (224, 224), color).save(path)
                    digest = hashlib.sha256(image_pixels(path).permute(1, 2, 0).contiguous().numpy().tobytes()).hexdigest()
                    manifest["splits"][split].append(dict(path=f"{split}/{index}.png", sha256=digest,
                                                          box=[28, 14, 126, 112]))
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            checkpoint = root / "checkpoint"
            checkpoint.write_bytes(b"frozen test checkpoint")
            args = cli.parser().parse_args(["prepare", "--device", "cpu", "--manifest", str(path),
                                            "--data-dir", str(root), "--out-dir", str(root / "features"),
                                            "--batch-size", "2", "--ijepa-checkpoint", str(checkpoint)])
            with patch.object(data, "load_ijepa", return_value=(fake_encoder, fake_predictor, fake_encoder, 299)):
                data.prepare(args)
            cache = data.load_cache(root / "features/val.pt")
            self.assertEqual(cache["predictor_tokens"].shape, (2, 49, 4))
            self.assertEqual(data.crop_targets(cache).shape, (2, 3, 98, 98))
            Image.new("RGB", (224, 224), (0, 0, 0)).save(root / "train/0.png")
            args.out_dir = str(root / "changed")
            with patch.object(data, "load_ijepa", return_value=(fake_encoder, fake_predictor, fake_encoder, 299)):
                with self.assertRaisesRegex(ValueError, "pixels differ"):
                    data.prepare(args)
            manifest["splits"]["val"][0]["sha256"] = manifest["splits"]["train"][0]["sha256"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "share image pixels"):
                data.read_manifest(path)

    def test_download_rejects_corrupted_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"correct")
            target = root / "downloaded"
            digest = hashlib.sha256(b"correct").hexdigest()
            download_file(source.as_uri(), target, digest)
            self.assertEqual(target.read_bytes(), b"correct")
            with self.assertRaisesRegex(ValueError, "wrong checksum"):
                download_file(source.as_uri(), root / "invalid", hashlib.sha256(b"other").hexdigest())

    def test_image_inference_never_uses_hidden_pixels_or_a_target_encoder(self):
        def predictor(context, context_indices, target_indices):
            return fake_predictor(context, context_indices, target_indices).repeat(1, 1, 320)

        def decoder(tokens, boxes):
            return tokens[:, :1, :3].transpose(1, 2).sigmoid()[..., None].expand(-1, -1, 98, 98)

        metadata = dict(feature_source="mixed", input_spec=dict(
            shape=[7, 7], feature_dim=1280, ijepa=dict(ijepa_epoch=299, checkpoint=IJEPA_CHECKPOINT),
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (224, 224), (70, 100, 170))
            path = root / "image.png"
            image.save(path)
            for name, source in (("original", image), ("changed", image.copy())):
                if name == "changed":
                    source.paste((255, 0, 0), (56, 56, 154, 154))
                source.save(path)
                args = cli.parser().parse_args(["decode", "--device", "cpu", "--image", str(path),
                                                "--output", str(root / f"{name}.png")])
                with patch.object(cli, "load_decoder", return_value=(decoder, metadata)), \
                        patch.object(cli, "load_ijepa",
                                     return_value=(fake_encoder, predictor, None, 299)) as loader:
                    cli.decode(args)
                    self.assertFalse(loader.call_args.kwargs.get("include_target", False))
            original = th.load(root / "original.pt", weights_only=True)
            changed = th.load(root / "changed.pt", weights_only=True)
            th.testing.assert_close(original["prediction"], changed["prediction"], atol=0, rtol=0)
            th.testing.assert_close(original["tokens"], changed["tokens"], atol=0, rtol=0)
            visible = th.ones(224, 224, dtype=th.bool)
            visible[56:154, 56:154] = False
            th.testing.assert_close(original["composite"][:, visible], image_pixels(path)[:, visible])

    def test_evaluation_rejects_empty_batches_before_loading_weights(self):
        args = cli.parser().parse_args(["evaluate", "--batch-size", "0"])
        with self.assertRaisesRegex(ValueError, "positive batch size"):
            evaluate(args)

    def test_scaled_manifest_preserves_reference_masks_and_adds_distinct_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root, reference = Path(directory), dict(ijepa_epoch=299, splits={})
            for split, colors in (("train", [(100, 30, 50), (40, 120, 80), (210, 90, 40)]),
                                  ("val", [(80, 40, 180), (10, 200, 90), (20, 30, 220)])):
                (root / split).mkdir()
                entries = []
                for index, color in enumerate(colors):
                    path = root / split / f"{index}.png"
                    Image.new("RGB", (224, 224), color).save(path)
                    entries.append(dict(path=f"{split}/{index}.png", sha256=data.pixel_hash(image_pixels(path)),
                                        box=[28, 14, 126, 112]))
                reference["splits"][split] = entries[:2]
            source, output = root / "reference.json", root / "scaled.json"
            source.write_text(json.dumps(reference))
            args = cli.parser().parse_args(["manifest", "--data-dir", str(root), "--reference-manifest", str(source),
                                            "--validation-images", "3", "--output", str(output)])
            data.generate_manifest(args)
            result = data.read_manifest(output)
            self.assertEqual(len(result["splits"]["train"]), 3)
            self.assertTrue(all(len(entry["boxes"]) == 4 for entry in result["splits"]["train"]))
            self.assertEqual([entry["path"] for entry in result["splits"]["val"][:2]],
                             [entry["path"] for entry in reference["splits"]["val"]])
            for split in ("train", "val"):
                by_path = {entry["path"]: entry for entry in result["splits"][split]}
                for entry in reference["splits"][split]:
                    self.assertEqual(by_path[entry["path"]]["boxes"][0], entry["box"])

    def test_multimask_preparation_deduplicates_rgb_and_resumes_after_partial_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = Path(directory), dict(ijepa_epoch=299, splits={})
            for split, colors in (("train", [(100, 30, 50), (40, 120, 80)]),
                                  ("val", [(80, 40, 180), (10, 200, 90)])):
                (root / split).mkdir()
                manifest["splits"][split] = []
                for index, color in enumerate(colors):
                    path = root / split / f"{index}.png"
                    Image.new("RGB", (224, 224), color).save(path)
                    manifest["splits"][split].append(dict(
                        path=f"{split}/{index}.png", sha256=data.pixel_hash(image_pixels(path)),
                        boxes=[[28, 14, 126, 112], [56, 56, 154, 154]],
                    ))
            path, checkpoint = root / "manifest.json", root / "checkpoint"
            path.write_text(json.dumps(manifest))
            checkpoint.write_bytes(b"frozen test checkpoint")
            args = cli.parser().parse_args(["prepare", "--device", "cpu", "--manifest", str(path),
                                            "--data-dir", str(root), "--out-dir", str(root / "features"),
                                            "--batch-size", "2", "--shard-size", "1",
                                            "--ijepa-checkpoint", str(checkpoint)])
            batches = []

            def teacher(images):
                batches.append(len(images))
                return fake_encoder(images)

            merge = data.merge_parts

            def interrupted(paths, entries, provenance):
                if paths[0].name.startswith("val"):
                    raise RuntimeError("simulated interruption")
                return merge(paths, entries, provenance)

            with patch.object(data, "load_ijepa", return_value=(fake_encoder, fake_predictor, teacher, 299)), \
                    patch.object(data, "merge_parts", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    data.prepare(args)
            count = len(batches)
            args.resume = True
            with patch.object(data, "load_ijepa", return_value=(fake_encoder, fake_predictor, teacher, 299)):
                data.prepare(args)
            self.assertEqual(len(batches), count)
            self.assertEqual(max(batches), 1)
            cache = data.load_cache(root / "features/train.pt")
            self.assertEqual(len(cache["originals"]), 2)
            self.assertEqual(cache["predictor_tokens"].shape, (4, 49, 4))
            th.testing.assert_close(cache["image_indices"], th.tensor([0, 0, 1, 1]))
            self.assertEqual(data.crop_targets(cache).shape, (4, 3, 98, 98))
            self.assertFalse((root / "features/.parts").exists())
            args.batch_size = 1
            with self.assertRaisesRegex(ValueError, "preparation settings"):
                data.prepare(args)

    def test_cosine_ema_training_resumes_exact_optimizer_and_sampler_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, colors in (("train", [(90, 140, 180), (160, 150, 90)]),
                                  ("val", [(124, 116, 104), (125, 116, 104)])):
                originals = th.stack([th.tensor(color, dtype=th.uint8)[:, None, None].expand(3, 224, 224).clone()
                                      for color in colors])
                save = dict(predictor_tokens=th.randn(2, 49, 4).half(), target_tokens=th.randn(2, 49, 4).half(),
                            originals=originals, boxes=th.tensor([[28, 14, 126, 112]] * 2),
                            image_indices=th.arange(2), image_hashes=[data.pixel_hash(image) for image in originals],
                            conditioning="ijepa_predictor", input_size=224, patch_size=14, block_size=7,
                            ijepa_epoch=299, ijepa_checkpoint=IJEPA_CHECKPOINT, spatial_shape=[7, 7],
                            target_normalization="layer_norm_patch_tokens")
                th.save(save, root / f"{split}.pt")
            model = NativeDecoder(4, (7, 7), dict(decoder_hidden_size=24, decoder_num_hidden_layers=1,
                                                decoder_num_attention_heads=4, decoder_intermediate_size=48))
            th.nn.init.zeros_(model.decoder.decoder_pred.weight)
            th.nn.init.zeros_(model.decoder.decoder_pred.bias)
            initial = root / "initial.pt"
            th.save(dict(state_dict=model.state_dict(), input_spec=data.input_spec(save),
                         config=model.config.to_dict(), initialization=initialization_spec(), feature_source="mixed"),
                    initial)
            common = ["train", "--device", "cpu", "--cache-dir", str(root), "--feature-source", "mixed",
                      "--steps", "6", "--batch-size", "2", "--schedule", "cosine", "--ema-decay", "0.9",
                      "--min-learning-rate", "1e-6", "--eval-every", "3"]

            def perceptual(prediction, target):
                return (prediction - target).square().mean((1, 2, 3), keepdim=True)

            with patch("ijepa_decoder.training.perceptual_loss", return_value=perceptual):
                train(cli.parser().parse_args(common + ["--init", str(initial), "--out-dir", str(root / "full")]))
                train(cli.parser().parse_args(common + ["--init", str(initial), "--out-dir", str(root / "resumed"),
                                                        "--stop-after", "3"]))
                train(cli.parser().parse_args(common + ["--out-dir", str(root / "resumed"),
                                                        "--resume", str(root / "resumed/last.pt")]))
            full = th.load(root / "full/last.pt", weights_only=True)
            resumed = th.load(root / "resumed/last.pt", weights_only=True)
            for name, value in full["state_dict"].items():
                th.testing.assert_close(resumed["state_dict"][name], value, atol=0, rtol=0)
            for name, value in full["ema"].items():
                th.testing.assert_close(resumed["ema"][name], value, atol=0, rtol=0)
            th.testing.assert_close(resumed["rng"]["cpu"], full["rng"]["cpu"], atol=0, rtol=0)
            self.assertEqual(resumed["scheduler"], full["scheduler"])
            self.assertEqual(resumed["history"], full["history"])
            incompatible = cli.parser().parse_args(common + ["--steps", "9", "--out-dir", str(root / "resumed"),
                                                              "--resume", str(root / "resumed/last.pt")])
            with self.assertRaisesRegex(ValueError, "training settings"):
                train(incompatible)


if __name__ == "__main__":
    unittest.main()

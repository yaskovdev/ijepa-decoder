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
from ijepa_decoder.training import evaluate


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
            args = cli.parser().parse_args(["prepare", "--device", "cpu", "--manifest", str(path),
                                            "--data-dir", str(root), "--out-dir", str(root / "features"),
                                            "--batch-size", "2"])
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


if __name__ == "__main__":
    unittest.main()

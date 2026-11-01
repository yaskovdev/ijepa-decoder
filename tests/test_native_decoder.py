import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch as th

from guided_diffusion_rcdm.ijepa_reconstruction import (
    NativeDecoder, initialization_spec, mask_order, resize_rgb_head,
)
from scripts.ijepa_native import load_native_cache, native_crops, native_spec, restore_native


class NativeDecoderTests(unittest.TestCase):
    def setUp(self):
        th.set_num_threads(2)
        th.manual_seed(0)

    def test_mask_restoration_preserves_native_patch_positions(self):
        boxes = th.tensor([[28, 14, 70, 42], [140, 140, 182, 168]])
        restore, selected = mask_order(boxes, (2, 3))
        self.assertEqual(selected[0].tolist(), [18, 19, 20, 34, 35, 36])
        sequence = th.full((2, 256), -1)
        sequence[:, :6] = th.arange(6)
        restored = sequence.gather(1, restore)
        th.testing.assert_close(restored.gather(1, selected), th.arange(6).expand(2, 6))
        with self.assertRaisesRegex(ValueError, "aligned boxes"):
            mask_order(boxes + 1, (2, 3))

    def test_pretrained_rgb_head_resize_keeps_channels_and_row_major_pixels(self):
        bias = th.tensor([1., 10., 100., 2., 20., 200., 3., 30., 300., 4., 40., 400.])
        weight = th.stack([bias, bias * 2], dim=1)
        resized_weight, resized_bias = resize_rgb_head(weight, bias, 1)
        th.testing.assert_close(resized_bias, th.tensor([2.5, 25., 250.]))
        th.testing.assert_close(resized_weight, th.stack([resized_bias, resized_bias * 2], dim=1))

    def test_native_targets_slice_mapped_originals_not_upscaled_rae_targets(self):
        originals = th.zeros(2, 3, 224, 224, dtype=th.uint8)
        originals[0, :, 14:42, 28:70] = 11
        originals[1, :, 14:42, 28:70] = 99
        cache = dict(originals=originals, features=th.zeros(2, 8), image_indices=th.tensor([1, 0]),
                     boxes=th.tensor([[28, 14, 70, 42], [28, 14, 70, 42]]),
                     targets=th.full((2, 3, 256, 256), 255, dtype=th.uint8))
        actual = native_crops(cache)
        expected = th.tensor([99, 11], dtype=th.uint8)[:, None, None, None].expand(2, 3, 28, 42)
        th.testing.assert_close(actual, expected)

    def test_native_cache_rejects_unpaired_boxes_and_non_tensor_tokens(self):
        cache = dict(features=th.zeros(2, 8), targets=th.zeros(2, 3, 256, 256, dtype=th.uint8),
                     originals=th.zeros(2, 3, 224, 224, dtype=th.uint8), image_hashes=["first", "second"],
                     image_indices=th.tensor([0, 1]), image_paths=["first", "second"], paths=["first", "second"],
                     predictor_tokens=th.randn(2, 6, 8), target_tokens=th.randn(2, 6, 8),
                     boxes=th.tensor([[28, 14, 70, 42], [140, 140, 182, 168]]),
                     input_size=224, patch_size=14, spatial_shape=[2, 3],
                     target_normalization="layer_norm_patch_tokens")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            th.save(cache, path)
            th.testing.assert_close(load_native_cache(path)["boxes"], cache["boxes"])
            for change, message in [
                (dict(boxes=cache["boxes"][:1]), "one aligned box"),
                (dict(predictor_tokens=None), "invalid predictor_tokens"),
            ]:
                th.save(dict(cache, **change), path)
                with self.assertRaisesRegex(ValueError, message):
                    load_native_cache(path)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Native optional dependencies are absent")
    def test_native_rgb_forward_and_backward_need_only_tokens_and_geometry(self):
        config = dict(decoder_hidden_size=24, decoder_num_hidden_layers=1,
                      decoder_num_attention_heads=4, decoder_intermediate_size=48)
        model = NativeDecoder(8, (2, 3), config)
        tokens = th.randn(2, 6, 8, requires_grad=True)
        boxes = th.tensor([[28, 14, 70, 42], [140, 140, 182, 168]])
        rgb = model(tokens, boxes)
        self.assertEqual(rgb.shape, (2, 3, 28, 42))
        rgb.square().mean().backward()
        self.assertGreater(tokens.grad.abs().sum().item(), 0)
        if th.backends.mps.is_available():
            model.eval()
            with th.no_grad():
                expected = model(tokens.detach(), boxes)
                actual = model.to("mps")(tokens.detach().to("mps"), boxes).cpu()
            th.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "Native optional dependencies are absent")
    def test_native_restore_rejects_other_feature_geometry_or_image_prior(self):
        config = dict(decoder_hidden_size=24, decoder_num_hidden_layers=1,
                      decoder_num_attention_heads=4, decoder_intermediate_size=48)
        model = NativeDecoder(8, (2, 3), config)
        cache = dict(features=th.zeros(2, 8), conditioning="ijepa_predictor", input_size=224,
                     patch_size=14, block_size=7, ijepa_epoch=299, ijepa_checkpoint="frozen.pt",
                     spatial_shape=[2, 3], target_normalization="layer_norm_patch_tokens")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.pt"
            checkpoint = dict(state_dict=model.state_dict(), config=model.config.to_dict(),
                              input_spec=native_spec(cache), initialization=initialization_spec(),
                              feature_source="target")
            th.save(checkpoint, path)
            restored, source = restore_native(path, cache, "cpu")
            tokens, boxes = th.randn(2, 6, 8), th.tensor([[28, 14, 70, 42], [140, 140, 182, 168]])
            th.testing.assert_close(restored(tokens, boxes), model(tokens, boxes))
            self.assertEqual(source, "target")
            with self.assertRaisesRegex(ValueError, "differs"):
                restore_native(path, dict(cache, ijepa_epoch=300), "cpu")
            with self.assertRaisesRegex(ValueError, "differs"):
                restore_native(path, dict(cache, spatial_shape=[3, 2]), "cpu")
            checkpoint["initialization"] = dict(initialization_spec(), revision="other")
            th.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "differs"):
                restore_native(path, cache, "cpu")

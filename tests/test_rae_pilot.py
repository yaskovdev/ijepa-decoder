import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch as th

from guided_diffusion_rcdm.rae import SpatialAdapter, cache_dino_positions, rae_spec, regrid
from scripts.ijepa_rae import restore_adapter, spatial_spec


class RaePilotTests(unittest.TestCase):
    def setUp(self):
        th.set_num_threads(2)
        th.manual_seed(0)

    def test_spatial_adapter_retains_positions_and_signed_features(self):
        model = SpatialAdapter(2, 2)
        with th.no_grad():
            model.net[0].weight.copy_(th.eye(2))
        tokens = th.tensor([[[-2., -1.], [0., 1.], [2., 3.], [4., 5.]]])
        flipped = tokens.flip(1)
        th.testing.assert_close(tokens.mean(1), flipped.mean(1))
        output = model(tokens, (2, 2))
        changed = model(flipped, (2, 2))
        self.assertFalse(th.equal(output, changed))
        th.testing.assert_close(output[:, 0], tokens[:, 0])
        th.testing.assert_close(changed[:, 0], flipped[:, 0])
        with self.assertRaisesRegex(ValueError, "rectangular grid"):
            model(tokens, (2, 3))

    def test_streamed_ridge_learns_a_signed_spatial_mapping(self):
        inputs = th.randn(32, 6, 3)
        weights = th.tensor([[2., -1.], [0.5, 3.], [-2., 0.25]])
        targets = regrid(inputs, (2, 3)) @ weights + th.tensor([-1., 2.])
        model = SpatialAdapter(3, 2)
        model.fit_ridge(inputs, targets, (2, 3), alpha=1e-6, batch_size=5)
        held_out = th.randn(4, 6, 3)
        expected = regrid(held_out, (2, 3)) @ weights + th.tensor([-1., 2.])
        th.testing.assert_close(model(held_out, (2, 3)), expected, atol=1e-4, rtol=1e-4)
        with self.assertRaisesRegex(ValueError, "positive"):
            model.fit_ridge(inputs, targets, (2, 3), alpha=float("nan"))

    def test_spatial_residual_starts_at_the_fitted_ridge_and_receives_gradients(self):
        inputs = th.randn(8, 4, 3)
        targets = regrid(inputs, (2, 2)) @ th.randn(3, 4)
        linear, residual = SpatialAdapter(3, 4), SpatialAdapter(3, 4, hidden_dim=5)
        for model in (linear, residual):
            model.fit_ridge(inputs, targets, (2, 2), alpha=0.1)
        th.testing.assert_close(residual(inputs, (2, 2)), linear(inputs, (2, 2)), rtol=0, atol=0)
        residual.net.requires_grad_(False)
        loss = (residual(inputs, (2, 2)) - targets).square().mean()
        loss.backward()
        self.assertGreater(residual.residual[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(residual.net[0].weight.grad)

    def test_spatial_output_normalization_is_explicit_and_restored(self):
        model = SpatialAdapter(3, 768, hidden_dim=5, normalize_output=True)
        with th.no_grad():
            model.net[0].weight.normal_()
        features = th.randn(2, 4, 3)
        expected = model(features, (2, 2))
        th.testing.assert_close(expected.mean(-1), th.zeros(2, 256), atol=1e-6, rtol=0)
        th.testing.assert_close(expected.var(-1, unbiased=False), th.ones(2, 256), atol=2e-5, rtol=0)
        cache = dict(features=th.zeros(2, 3), conditioning="ijepa_predictor", input_size=224,
                     patch_size=14, block_size=7, ijepa_epoch=299, ijepa_checkpoint="frozen.pt",
                     spatial_shape=[2, 2], target_normalization="layer_norm_patch_tokens", rae_spec=rae_spec())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            th.save(dict(state_dict=model.state_dict(), input_spec=spatial_spec(cache),
                         input_dim=3, output_dim=768, hidden_dim=5, normalize_output=True,
                         feature_source="predictor"), path)
            restored, _ = restore_adapter(path, cache, "cpu")
            th.testing.assert_close(restored(features, (2, 2)), expected, rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "RAE optional dependencies are absent")
    def test_cpu_position_cache_matches_unmodified_dino_forward(self):
        from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel

        config = Dinov2WithRegistersConfig(
            hidden_size=24, num_hidden_layers=1, num_attention_heads=4, intermediate_size=48,
            image_size=28, patch_size=14, num_register_tokens=4,
        )
        config._attn_implementation = "eager"
        model = Dinov2WithRegistersModel(config).eval()
        images = th.randn(2, 3, 224, 224)
        with th.no_grad():
            expected = model(images).last_hidden_state
            cache_dino_positions(model)
            actual = model(images).last_hidden_state
        th.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_spatial_restore_rejects_changed_frozen_models_or_geometry(self):
        model = SpatialAdapter(3, 768)
        cache = dict(features=th.zeros(2, 3), conditioning="ijepa_predictor", input_size=224,
                     patch_size=14, block_size=7, ijepa_epoch=299, ijepa_checkpoint="frozen.pt",
                     spatial_shape=[2, 2], target_normalization="layer_norm_patch_tokens", rae_spec=rae_spec())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            th.save(dict(state_dict=model.state_dict(), input_spec=spatial_spec(cache),
                         input_dim=3, output_dim=768, hidden_dim=0, feature_source="predictor"), path)
            restored, source = restore_adapter(path, cache, "cpu")
            features = th.randn(2, 4, 3)
            th.testing.assert_close(restored(features, (2, 2)), model(features, (2, 2)))
            self.assertEqual(source, "predictor")
            for changed in (dict(cache, spatial_shape=[1, 4]), dict(cache, ijepa_checkpoint="other.pt"),
                            dict(cache, rae_spec=dict(rae_spec(), encoder_revision="different"))):
                with self.assertRaisesRegex(ValueError, "differs"):
                    restore_adapter(path, changed, "cpu")

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch as th

from guided_diffusion_rcdm.rae import SpatialAdapter, cache_dino_positions, rae_spec, regrid
from scripts.ijepa_diagnostics import (
    nearest_examples, parser as diagnostics_parser, project_grid, retrieve, retrieval_features,
    validate_retrieval_caches,
)
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

    def test_grid_projection_preserves_representable_maps_and_beats_plain_resize(self):
        coarse = th.randn(2, 6, 3)
        representable = regrid(coarse, (2, 3))
        th.testing.assert_close(project_grid(representable, (2, 3)), representable, atol=1e-6, rtol=1e-6)
        full = th.randn(2, 256, 3)
        projected = project_grid(full, (7, 7))
        resized = regrid(regrid(full, (16, 16), output_size=7), (7, 7))
        self.assertLess((projected - full).square().mean().item(), (resized - full).square().mean().item())
        th.testing.assert_close(project_grid(projected, (7, 7)), projected, atol=1e-6, rtol=1e-6)
        if th.backends.mps.is_available():
            th.testing.assert_close(project_grid(full.to("mps"), (7, 7)).cpu(), projected,
                                    atol=1e-6, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, "source grid"):
            project_grid(full, (17, 7))

    def test_retrieval_cosine_returns_distinct_originals_not_duplicate_masks(self):
        bank = th.tensor([[2., 0.], [1., 0.01], [0.8, 0.6], [-1., 0.]])
        queries = th.tensor([[1., 0.], [-1., 0.]])
        indices, scores = nearest_examples(bank, queries, ["a", "a", "b", "c"], top_k=2)
        self.assertEqual(indices.tolist(), [[0, 2], [3, 2]])
        th.testing.assert_close(scores[0], th.tensor([1., 0.8]))
        with self.assertRaisesRegex(ValueError, "distinct original"):
            nearest_examples(bank, queries, ["a", "a", "b", "c"], top_k=4)
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            nearest_examples(th.ones(3, 2), queries, ["a", "b", "c"], top_k=1, center=True)

    def test_retrieval_supports_pooled_and_spatial_targets_and_rejects_leakage(self):
        spec = dict(conditioning="ijepa_predictor", input_size=224, patch_size=14, block_size=7,
                    ijepa_epoch=299, ijepa_checkpoint="frozen.pt", target_normalization="layer_norm_patch_tokens")
        bank = dict(spec, features=th.ones(3, 2), target_features=th.randn(3, 2),
                    image_indices=th.tensor([0, 0, 1]), image_hashes=["train-a", "train-b"])
        query = dict(spec, features=th.ones(2, 2), target_tokens=th.randn(2, 4, 2),
                     image_indices=th.tensor([0, 1]), image_hashes=["val-a", "val-b"])
        bank_hashes, query_hashes = validate_retrieval_caches(bank, query)
        self.assertEqual(bank_hashes, ["train-a", "train-a", "train-b"])
        self.assertEqual(query_hashes, ["val-a", "val-b"])
        th.testing.assert_close(retrieval_features(query, "target"), query["target_tokens"].mean(1))
        for changed, message in (
            (dict(query, image_hashes=["train-a", "val-b"]), "share original"),
            (dict(query, ijepa_epoch=300), "differ"),
            (dict(query, target_normalization="pooled_layer_norm"), "differ"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                validate_retrieval_caches(bank, changed)

    def test_retrieval_driver_renders_real_crops_without_changing_visible_pixels(self):
        def cache(paths, hashes, features):
            count = len(paths)
            originals = th.arange(count, dtype=th.uint8)[:, None, None, None].expand(
                count, 3, 224, 224).clone() * 40
            return dict(
                features=features, target_features=features.clone(),
                originals=originals, targets=th.zeros(count, 3, 128, 128, dtype=th.uint8),
                image_indices=th.arange(count), image_paths=paths, paths=paths, image_hashes=hashes,
                boxes=th.tensor([[14, 14, 112, 112]]).expand(count, 4).clone(),
                conditioning="ijepa_predictor", input_size=224, patch_size=14, block_size=7,
                ijepa_epoch=299, ijepa_checkpoint="frozen.pt", target_normalization="layer_norm_patch_tokens",
            )

        bank = cache(["train/a", "train/b", "train/c"], ["a", "b", "c"],
                     th.tensor([[1., 0.], [0., 1.], [-1., 0.]]))
        query = cache(["val/d", "val/e"], ["d", "e"], th.tensor([[0.9, 0.1], [-0.9, 0.1]]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            th.save(bank, root / "bank.pt")
            th.save(query, root / "query.pt")
            args = diagnostics_parser().parse_args([
                "retrieve", "--bank", str(root / "bank.pt"), "--cache", str(root / "query.pt"),
                "--top-k", "2", "--output", str(root / "retrieval.png"),
            ])
            retrieve(args)
            self.assertTrue((root / "retrieval_centered.png").is_file())
            output = th.load(root / "retrieval.pt", weights_only=True)
            self.assertEqual(output["output_type"], "retrieved_training_rgb_crops")
            self.assertFalse(output["primary_uses_hidden_rgb"])
            visible = th.ones(224, 224, dtype=th.bool)
            visible[14:112, 14:112] = False
            for row in range(2):
                for composite in output["composites"][row]:
                    th.testing.assert_close(composite[:, visible], query["originals"][row, :, visible])

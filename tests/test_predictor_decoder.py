import tempfile
import unittest
import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
from torch import nn

from guided_diffusion_rcdm import dist_util, logger
from guided_diffusion_rcdm.feature_datasets import (
    different_image_index, distinct_examples, load_feature_cache, load_feature_data,
)
from guided_diffusion_rcdm.gaussian_diffusion import _extract_into_tensor
from guided_diffusion_rcdm.ijepa import PredictorAdapter, make_block_masks, predict_features, target_features
from guided_diffusion_rcdm.script_util import create_gaussian_diffusion, create_model
from guided_diffusion_rcdm.train_util import TrainLoop
from scripts.ijepa_decoder import feature_spec, sample
from scripts.ijepa_decoder import hidden_crop, prediction_in_context
from scripts import ijepa_adapter


class PatchEncoder(nn.Module):
    def forward(self, images, masks=None):
        patches = th.nn.functional.avg_pool2d(images, 2).flatten(2).transpose(1, 2)
        if masks is None:
            return patches
        return th.gather(patches, 1, masks[0].unsqueeze(-1).expand(-1, -1, patches.shape[-1]))


class ContextPredictor(nn.Module):
    def forward(self, context, context_indices, target_indices):
        return context.mean(dim=1, keepdim=True).expand(-1, target_indices[0].shape[1], -1)


class PredictorDecoderTests(unittest.TestCase):
    def setUp(self):
        th.set_num_threads(2)
        dist_util.set_device("cpu")

    def test_hidden_pixels_do_not_enter_predictor_conditioning(self):
        context, target = make_block_masks(1, 1, block_size=2, grid_size=4)
        image = th.randn(1, 3, 8, 8)
        changed = image.clone()
        changed[:, :, 2:6, 2:6] = 1000
        encoder, predictor = PatchEncoder(), ContextPredictor()
        original = predict_features(encoder, predictor, image, context[None], target[None])
        hidden_changed = predict_features(encoder, predictor, changed, context[None], target[None])
        th.testing.assert_close(original, hidden_changed, rtol=0, atol=0)
        changed[:, :, :2, :2] += 10
        visible_changed = predict_features(encoder, predictor, changed, context[None], target[None])
        self.assertFalse(th.equal(original, visible_changed))

    def test_cache_rejects_unpaired_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cache.pt"
            th.save(dict(targets=th.zeros(2, 3, 8, 8, dtype=th.uint8), features=th.ones(1, 3)), path)
            with self.assertRaisesRegex(ValueError, "one nonempty feature vector"):
                load_feature_cache(path)

    def test_cache_does_not_loop_forever_when_batch_is_too_large(self):
        cache = dict(targets=th.zeros(1, 3, 8, 8, dtype=th.uint8), features=th.ones(1, 3))
        with self.assertRaisesRegex(ValueError, "fewer than batch_size"):
            next(load_feature_data(cache, 2))

    def test_hidden_crop_uses_aligned_original_pixels(self):
        original = th.arange(3 * 8 * 8, dtype=th.uint8).reshape(3, 8, 8)
        cache = dict(originals=original[None], boxes=th.tensor([[2, 2, 6, 6]]),
                     targets=th.zeros(1, 3, 4, 4, dtype=th.uint8))
        crop = th.from_numpy(np.array(hidden_crop(cache, 0, 4))).permute(2, 0, 1)
        th.testing.assert_close(crop, original[:, 2:6, 2:6], rtol=0, atol=0)
        cache["boxes"][0] = th.tensor([7, 7, 9, 9])
        with self.assertRaisesRegex(ValueError, "outside"):
            hidden_crop(cache, 0, 4)

    def test_multi_mask_cache_and_composite_preserve_visible_pixels(self):
        originals = th.arange(2 * 3 * 8 * 8).remainder(256).to(th.uint8).reshape(2, 3, 8, 8)
        cache = dict(originals=originals, image_indices=th.tensor([0, 0, 1, 1]),
                     image_paths=["a.jpg", "b.jpg"], paths=["a.jpg", "a.jpg", "b.jpg", "b.jpg"],
                     boxes=th.tensor([[2, 1, 6, 5]] * 4), features=th.ones(4, 3),
                     targets=th.zeros(4, 3, 4, 4, dtype=th.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            th.save(cache, path)
            loaded = load_feature_cache(path)
            patch = th.full((3, 6, 6), 37, dtype=th.uint8)
            composite = th.from_numpy(np.array(prediction_in_context(loaded, 2, patch))).permute(2, 0, 1)
            expected = originals[1].clone()
            expected[:, 1:5, 2:6] = 37
            th.testing.assert_close(composite, expected, rtol=0, atol=0)
            th.testing.assert_close(loaded["originals"], originals, rtol=0, atol=0)
            self.assertEqual(distinct_examples(loaded, 4), [0, 2])
            self.assertEqual(different_image_index(loaded, 0), 2)
            self.assertEqual(different_image_index(loaded, 2, -1), 1)
            cache["image_indices"][2] = 5
            th.save(cache, path)
            with self.assertRaisesRegex(ValueError, "map every example"):
                load_feature_cache(path)

    def test_target_encoder_features_normalize_tokens_before_pooling(self):
        images = th.randn(2, 3, 8, 8)
        indices = th.tensor([[0, 1], [2, 3], [0, 1]])
        image_indices = th.tensor([0, 1, 0])
        encoder = PatchEncoder()
        actual = target_features(encoder, images, indices, image_indices)
        tokens = th.nn.functional.layer_norm(encoder(images), (3,))
        expected = th.stack([tokens[image, mask].mean(0) for image, mask in zip(image_indices, indices)])
        th.testing.assert_close(actual, expected)
        raw = encoder(images)[0, indices[0]].mean(0)
        self.assertFalse(th.allclose(actual[0], th.nn.functional.layer_norm(raw, (3,))))

    def test_ridge_adapter_uses_cpu_solve_even_with_mps_features(self):
        th.manual_seed(2)
        features = th.rand(32, 3)
        targets = 1 + features @ th.rand(3, 4)
        for device in ("cpu", "mps") if th.backends.mps.is_available() else ("cpu",):
            model = PredictorAdapter(3, 4, hidden_dim=0)
            model.fit_statistics(features, targets)
            model.to(device)
            model.fit_ridge(features.to(device), targets.to(device), 0.001)
            th.testing.assert_close(model(features.to(device)).cpu(), targets, rtol=0.01, atol=0.01)

    def test_dino_cache_rejects_different_boxes_and_normalization(self):
        predictor = dict(
            features=th.ones(2, 3), paths=["a.jpg", "b.jpg"], boxes=th.tensor([[0, 0, 4, 4]] * 2),
            conditioning="ijepa_predictor", input_size=224, patch_size=14,
            block_size=7, ijepa_epoch=299, ijepa_checkpoint="official.pt",
        )
        dino = dict(
            features=th.ones(2, 4), paths=predictor["paths"], boxes=predictor["boxes"].clone(),
            feature_spec=feature_spec(predictor),
            dino_spec=dict(model="dino_resnet50_trunk", input_size=224,
                           normalization="rgb_minus_one_one", checkpoint="dino.pt"),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "dino.pt"
            dino["boxes"][0, 0] = 1
            th.save(dino, path)
            with self.assertRaisesRegex(ValueError, "hidden boxes"):
                ijepa_adapter.load_dino_cache(path, predictor)
            dino["boxes"] = predictor["boxes"]
            dino["dino_spec"]["normalization"] = "imagenet"
            th.save(dino, path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                ijepa_adapter.load_dino_cache(path, predictor)

    def test_adapter_trains_restores_statistics_and_rejects_other_predictor(self):
        th.manual_seed(4)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "predictor").mkdir()
            (root / "dino").mkdir()
            predictor = {}
            dino_spec = dict(model="dino_resnet50_trunk", input_size=224,
                             normalization="rgb_minus_one_one", checkpoint="dino.pt")
            mapping = th.tensor([[1., 0., 0., 1.], [0., 1., 0., -1.], [0., 0., 1., 0.5]])
            for split, count in (("train", 32), ("val", 8)):
                features = th.randn(count, 3)
                cache = dict(
                    features=features, targets=th.zeros(count, 3, 8, 8, dtype=th.uint8),
                    paths=[f"{split}/{index}.jpg" for index in range(count)],
                    boxes=th.tensor([[0, 0, 4, 4]] * count),
                    conditioning="ijepa_predictor", input_size=224, patch_size=14,
                    block_size=7, ijepa_epoch=299, ijepa_checkpoint="official.pt",
                )
                predictor[split] = cache
                th.save(cache, root / "predictor" / f"{split}.pt")
                th.save(dict(
                    features=(1 + features @ mapping).relu(),
                    paths=cache["paths"], boxes=cache["boxes"],
                    feature_spec=feature_spec(cache), dino_spec=dino_spec,
                ), root / "dino" / f"{split}.pt")
            args = ijepa_adapter.create_argparser().parse_args([
                "train", "--device", "cpu", "--cache-dir", str(root / "predictor"),
                "--dino-cache-dir", str(root / "dino"), "--out-dir", str(root / "adapter"),
                "--steps", "100", "--batch-size", "16", "--hidden-dim", "32", "--lr", "0.01",
            ])
            ijepa_adapter.train(args)
            history = json.loads((root / "adapter" / "metrics.json").read_text())
            self.assertLess(history[-1]["validation"]["mse"], history[0]["validation"]["mse"])
            self.assertLess(history[-1]["validation"]["mse"], history[-1]["validation"]["wrong_mse"])
            checkpoint = root / "adapter" / "best.pt"
            model = ijepa_adapter.load_adapter(checkpoint, predictor["val"], dino_spec, "cpu")
            th.testing.assert_close(model.input_mean, predictor["train"]["features"].mean(0))
            self.assertTrue(th.isfinite(model(predictor["val"]["features"])).all().item())
            legacy = th.load(checkpoint, weights_only=True)
            for key in ("feature_source", "dropout", "kind"):
                legacy.pop(key)
            th.save(legacy, root / "legacy.pt")
            restored = ijepa_adapter.load_adapter(root / "legacy.pt", predictor["val"], dino_spec, "cpu")
            th.testing.assert_close(model(predictor["val"]["features"]), restored(predictor["val"]["features"]))
            with self.assertRaisesRegex(ValueError, "does not match"):
                ijepa_adapter.load_adapter(
                    checkpoint, dict(predictor["val"], ijepa_checkpoint="different.pt"), dino_spec, "cpu"
                )

    def test_target_ridge_uses_teacher_features_and_wrong_originals(self):
        th.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "predictor").mkdir()
            (root / "dino").mkdir()
            predictor, targets = {}, {}
            dino_spec = dict(model="dino_resnet50_trunk", input_size=224,
                             normalization="rgb_minus_one_one", checkpoint="dino.pt")
            mapping = th.tensor([[1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 1.]])
            for split, images in (("train", 16), ("val", 4)):
                count = images * 2
                teacher = th.rand(images, 3).repeat_interleave(2, 0)
                cache = dict(features=th.rand(images, 3).repeat_interleave(2, 0),
                             target_features=teacher, target_normalization="layer_norm_patch_tokens",
                             targets=th.zeros(count, 3, 8, 8, dtype=th.uint8),
                             paths=[f"{split}/{index // 2}.jpg" for index in range(count)],
                             boxes=th.tensor([[0, 0, 4, 4]] * count),
                             conditioning="ijepa_predictor", input_size=224, patch_size=14,
                             block_size=7, ijepa_epoch=299, ijepa_checkpoint="official.pt")
                predictor[split] = cache
                targets[split] = 1 + teacher @ mapping
                th.save(cache, root / "predictor" / f"{split}.pt")
                th.save(dict(features=targets[split], paths=cache["paths"], boxes=cache["boxes"],
                             feature_spec=feature_spec(cache), dino_spec=dino_spec),
                        root / "dino" / f"{split}.pt")
            args = ijepa_adapter.create_argparser().parse_args([
                "train", "--device", "cpu", "--kind", "ridge", "--ridge-alpha", "0.001",
                "--feature-source", "target", "--batch-size", "16",
                "--cache-dir", str(root / "predictor"), "--dino-cache-dir", str(root / "dino"),
                "--out-dir", str(root / "adapter"),
            ])
            ijepa_adapter.train(args)
            model = ijepa_adapter.load_adapter(root / "adapter" / "best.pt", predictor["val"], dino_spec, "cpu")
            actual = model(predictor["val"]["target_features"])
            metrics = json.loads((root / "adapter" / "metrics.json").read_text())[-1]["validation"]
            self.assertLess(metrics["mse"], 0.01)
            wrong = actual[th.tensor([7, 7, 1, 1, 3, 3, 5, 5])]
            expected_wrong_mse = (((wrong - targets["val"]) / model.output_scale) ** 2).mean().item()
            self.assertAlmostEqual(metrics["wrong_mse"], expected_wrong_mse)
            self.assertGreater(ijepa_adapter.adapter_metrics(model, predictor["val"]["features"],
                                                           targets["val"])["mse"], 0.25)

    def test_sampling_rejects_features_from_a_different_ijepa_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = dict(
                targets=th.zeros(2, 3, 8, 8, dtype=th.uint8), features=th.ones(2, 3),
                conditioning="ijepa_predictor", input_size=224, patch_size=14,
                block_size=7, ijepa_epoch=299, ijepa_checkpoint="another_checkpoint.pt",
            )
            th.save(cache, root / "cache.pt")
            training_spec = feature_spec(dict(cache, ijepa_checkpoint="training_checkpoint.pt"))
            (root / "decoder_config.json").write_text(json.dumps(dict(
                conditioning="ijepa_predictor", ssl_dim=3, image_size=8, feature_spec=training_spec,
            )))
            args = argparse.Namespace(cache=str(root / "cache.pt"), checkpoint=str(root / "ema.pt"),
                                      num_images=1, samples_per_image=1)
            with self.assertRaisesRegex(ValueError, "checkpoint or block geometry differs"):
                sample(args)

    def test_local_training_saves_and_resumes_without_distributed_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            logger.configure(dir=folder, format_strs=["csv"])
            cache = dict(targets=th.randint(256, (2, 3, 8, 8), dtype=th.uint8), features=th.randn(2, 3))
            diffusion = create_gaussian_diffusion()
            model = create_model(8, 32, 1, channel_mult="1,2", feat_cond=True, ssl_dim=3,
                                 attention_resolutions="4", use_scale_shift_norm=True)
            initial = model.out[-1].weight.detach().clone()
            loop = TrainLoop(
                model=model, diffusion=diffusion, data=load_feature_data(cache, 2),
                batch_size=2, microbatch=-1, lr=1e-4, ema_rate=0.99,
                log_interval=1, save_interval=10, resume_checkpoint="", max_steps=2,
            )
            loop.run_loop()
            checkpoint = Path(folder) / "model000002.pt"
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((Path(folder) / "ema_0.99_000002.pt").is_file())
            self.assertFalse(th.equal(initial, model.out[-1].weight))
            resumed = TrainLoop(
                model=model, diffusion=diffusion, data=load_feature_data(cache, 2),
                batch_size=2, microbatch=-1, lr=1e-4, ema_rate=0.99,
                log_interval=1, save_interval=10, resume_checkpoint=str(checkpoint), max_steps=3,
            )
            self.assertEqual(resumed.resume_step, 2)
            resumed.run_loop()
            self.assertTrue((Path(folder) / "model000003.pt").is_file())

    @unittest.skipUnless(th.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_diffusion_training_and_sampling(self):
        device = dist_util.set_device("mps")
        timesteps = th.tensor([1, 9], device=device)
        coefficients = np.linspace(0.1, 0.9, 10, dtype=np.float64)
        actual = _extract_into_tensor(coefficients, timesteps, (2, 3, 8, 8)).cpu()
        expected = th.tensor(coefficients[[1, 9]], dtype=th.float32)[:, None, None, None].expand_as(actual)
        th.testing.assert_close(actual, expected, rtol=0, atol=0)
        model = create_model(8, 32, 1, channel_mult="1,2", feat_cond=True, ssl_dim=3,
                             attention_resolutions="4", use_scale_shift_norm=True).to(device)
        diffusion = create_gaussian_diffusion(timestep_respacing="ddim10")
        images, features = th.randn(2, 3, 8, 8, device=device), th.randn(2, 3, device=device)
        losses = diffusion.training_losses(model, images, th.tensor([2, 8], device=device),
                                           model_kwargs={"feat": features})
        losses["loss"].mean().backward()
        self.assertTrue(th.isfinite(model.out[-1].weight.grad).all().item())
        sampled = diffusion.ddim_sample_loop(model, images.shape, model_kwargs={"feat": features})
        self.assertTrue(th.isfinite(sampled).all().item())
        dist_util.set_device("cpu")


if __name__ == "__main__":
    unittest.main()

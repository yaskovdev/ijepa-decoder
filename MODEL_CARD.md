# Native I-JEPA Predictor Decoder v0.2.0

An **unofficial research model**, not a Meta release or a reproduction of its
unpublished RCDM decoder. The released model is the best **predictor** checkpoint,
selected at mixed-training **EMA step 10,750**, not a target-encoder oracle.
The v0.1.0 model remains available in its original release.

## Model

- Frozen backbone: official I-JEPA ViT-H/14, `IN1K-vit.h.14-300e.pth.tar`,
  actual checkpoint epoch 299; source commit
  `52c1ae95d05f743e000e8f10a1f3a79b10cff048`.
- Decoder: 26,309,708 parameters, initialized from
  `facebook/vit-mae-base` revision `25b184bea5538bf5c4c852c79d221195fdd2778d`.
- Input: 49 unpooled, 1280-dimensional predictor tokens and their known
  positions in the native 16x16 patch grid.
- Output: deterministic 98x98 RGB, using fixed ImageNet denormalization.
  Hidden-patch color statistics are never supplied.
- Format: `model.safetensors` plus `config.json`; checksums are published in
  `SHA256SUMS` and packaged in `ijepa_decoder/pretrained.json`.

The decoder transformer, resized 14-pixel RGB head, and new input projection
are trainable. The I-JEPA encoders/predictor and AlexNet LPIPS network are frozen.
Inference loads no target encoder and uses no DINO, RAE, or diffusion model.

## Training and Evaluation

The v0.2 manifest contains **9,469 distinct Imagenette 320 training photographs,
four masks each (37,876 examples), and 512 validation photographs**. It is
generated deterministically from the packaged v0.1 reference manifest; the
original 2,048 training masks and 64 validation photographs/masks are retained.
Training and validation have no preprocessed-pixel overlap. Hashes refer to
224x224 RGB after OpenAI/RCDM center-crop preprocessing, not JPEG file bytes.

The exact expanded manifest is included in the v0.2 release. Its SHA256 is
`41f1a58ebe053bcac55b921b44a89a7fd9a3521e1608a11d54d44636409e8fbc`.
Newly extracted predictor and target tokens for the original 64 photographs
were bit-identical to the v0.1 cache.

Both experiments start from the released v0.1 mixed-step-400 decoder, with a
new optimizer. Each runs 15,000 steps: batch size 16, seed 0, AdamW weight decay
0.01, learning rate 5e-5 with cosine decay to 1e-6, EMA decay 0.999, and
predictor validation every 250 steps. Loss is raw RGB L1 + 0.5 frozen AlexNet
LPIPS. Mixed batches alternate target/predictor examples 50/50. Raw and EMA
weights are compared using the same `raw_l1 + 0.5 * LPIPS` validation criterion.

| Predictor evaluation on all 512 validation photographs | Selected step | RGB MAE | LPIPS |
|---|---:|---:|---:|
| v0.1 released initializer | 400 | 0.131718 | 0.324860 |
| Expanded predictor-only, EMA | 6,250 | 0.121533 | 0.280126 |
| **Expanded mixed, EMA (v0.2 release)** | **10,750** | **0.120092** | **0.275880** |

On the original 64 validation photographs, v0.2 predictor RGB MAE is
**0.118100** and LPIPS **0.286794**, versus v0.1 **0.128089 / 0.335584**.
The README shows the same four photographs selected for the earlier v0.1
illustration, not new examples selected after expanded training.

Paired bootstrap intervals use 2,000 resamples grouped by original photograph.
On 512 photographs, new-minus-v0.1 LPIPS is -0.048980 (95% interval
[-0.051724, -0.046105]); RGB MAE difference is -0.011626 (95% interval
[-0.013154, -0.010198]). These are **descriptive validation intervals**, not
independent-test confidence intervals: checkpoint selection uses this same
validation set, and the original 64 photographs are a subset of it.

Conditioning controls on the same 512 photographs:

| Input to v0.2 decoder | RGB MAE | LPIPS |
|---|---:|---:|
| Correct predictor tokens | 0.120092 | 0.275880 |
| Wrong-photo predictor tokens | 0.303337 | 0.650313 |
| Shuffled predictor tokens | 0.201301 | 0.492012 |
| Target encoder, **hidden-RGB oracle** | 0.086206 | 0.146810 |

Wrong-photo tokens may also come from a different mask position, so this is
not a geometry-matched semantic ablation. The oracle is not used for ordinary
prediction. Pasting predictions into context preserves visible uint8 pixels
exactly.

Training loops took 3,374 seconds (predictor-only) and 3,452 seconds (mixed)
on an Apple M3 Max with 36 GiB unified memory, including periodic validation
and checkpointing after feature caching. Frozen feature extraction logged
2,806 seconds before consolidation. Downloads and initialization are excluded.
Exact bitwise retraining is not promised across devices, versions, or kernels.

The v0.1 initializer was itself MAE-initialized: 1,000 target-token steps,
then 800 mixed steps with predictor validation selecting step 400. Its complete
recipe remains in the README and the preserved v0.1 release.

Feature extraction uses float32. Both image inference and cached evaluation
quantize predictor tokens to float16, then convert them to float32 for the
decoder, matching the training representation.

## Limits and License

This ten-class pilot visualizes features; it does not reveal a unique hidden
image or provide general-purpose inpainting. Texture, colors, alignment, and
patch seams can be inaccurate. Only patch-aligned 7x7 masks on the 224x224
center-cropped view are supported by this release. Validation is separate from
our decoder training, but may overlap I-JEPA/MAE ImageNet pretraining.

Released decoder weights are licensed **CC-BY-NC 4.0** for noncommercial research;
the full model license accompanies the release. Official I-JEPA code/weights
retain their own CC-BY-NC 4.0 license. The pinned Hugging Face MAE model card
lists Apache-2.0. Repository code retains the original MIT license and
Meta/OpenAI copyright notices. Dataset photographs retain their own rights.

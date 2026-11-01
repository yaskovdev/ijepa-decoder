# Native I-JEPA Predictor Decoder v0.1.0

An **unofficial research model**, not a Meta release or a reproduction of its
unpublished RCDM decoder. The released model is the best **predictor** checkpoint,
selected at mixed-training step 400, not a target-encoder oracle.

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

The shipped `ijepa_decoder/manifest.json` identifies the exact 2,048 training
and 64 validation Imagenette 320 photographs, preprocessed-pixel hashes, and
patch-aligned masks. Hashes refer to 224x224 RGB after the included
OpenAI/RCDM center-crop preprocessing, not to the JPEG file bytes.

Training uses native original RGB crops and L1 + 0.5 AlexNet LPIPS. First,
1,000 target-token steps initialize a reconstruction oracle. Then 800 mixed
steps alternate target/predictor examples within each batch; predictor
validation selects step 400. Batch size is 8, seed 0, AdamW weight decay 0.01;
learning rates are 1e-4 initially (5x for the new projection) and 5e-5 for
mixed continuation.

On all 64 validation photographs, the released predictor model has RGB MAE
**0.128089** and LPIPS **0.335584**. The measured training loops took 118 and
95 seconds on an M3 Max with already-cached features, excluding downloads and
I-JEPA feature extraction. Exact bitwise retraining is not promised across
different devices, package versions, or numerical kernels.

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

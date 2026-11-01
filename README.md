# I-JEPA Decoder (Unofficial)

**Turn Meta I-JEPA predictor features into an RGB visualization.** This
independent research prototype includes only the best native decoder,
image inference, and the code/data manifest needed to reproduce its training.
No RCDM, DINO, RAE, or diffusion generator is required.

![Best native predictor checkpoint: original, context, true target, predictor, and prediction pasted into context](docs/result.png)

Four illustrated validation examples from the **same best predictor model**.
The predictor sees only visible patches and known mask positions; the
**in context** column inserts its decoded crop without changing visible pixels.
The true target is a comparison, not an input to prediction.

## Pretrained Decoder

**[Download v0.2.0](https://github.com/yaskovdev/ijepa-decoder/releases/tag/v0.2.0)**
or use `ijepa-decoder download` below. The release contains
[105 MB safetensors weights](https://github.com/yaskovdev/ijepa-decoder/releases/download/v0.2.0/model.safetensors),
[config](https://github.com/yaskovdev/ijepa-decoder/releases/download/v0.2.0/config.json),
the model license, and
[SHA256SUMS](https://github.com/yaskovdev/ijepa-decoder/releases/download/v0.2.0/SHA256SUMS).
Checksums are verified by the downloader. Weights are release assets, not git blobs.

The decoder has **26.3 million parameters**, is initialized from MAE-base, and
maps 49 unpooled I-JEPA ViT-H/14 predictor tokens plus their native positions
to a **98x98 RGB crop**. The released checkpoint is **mixed-training EMA at step
10,750**, selected from a 15,000-step run on 9,469 photographs with four masks each.
It is not a target-encoder oracle.

| Predictor evaluation (lower is better) | v0.1 RGB MAE | v0.2 RGB MAE | v0.1 LPIPS | v0.2 LPIPS |
|---|---:|---:|---:|---:|
| All 512 validation photographs | 0.131718 | **0.120092** | 0.324860 | **0.275880** |
| Original 64 validation photographs | 0.128089 | **0.118100** | 0.335584 | **0.286794** |

LPIPS improved by **15.1%** on the expanded validation set. These are validation
results, not an independent test or a guarantee that every example improves.
The illustrated photographs are the same four shown for v0.1.
[Model card, training details, limitations, and licensing](MODEL_CARD.md).

This is **not an official Meta decoder**, general-purpose inpainting, or the
unique hidden image. Colors, details, alignment, and seams may be wrong.
Only patch-aligned 7x7 masks in the 224x224 center-cropped view are supported.

## Install and Visualize Your Image

Python 3.12 / PyTorch 2.7.1 was tested on Apple Silicon MPS. CUDA and CPU are
supported code paths, but their runtime was not benchmarked. Choose appropriate
Linux wheels using the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```shell
git clone https://github.com/yaskovdev/ijepa-decoder.git
cd ijepa-decoder
python3 -m venv .venv
source .venv/bin/activate
pip install -c constraints.txt -e .
ijepa-decoder download
```

You also need the **full official I-JEPA encoder/predictor checkpoint**
(about 10.4 GB), not an encoder-only Hugging Face model. I-JEPA is downloaded
separately and retains its own noncommercial license:

```shell
mkdir -p artifacts/checkpoints
git clone https://github.com/facebookresearch/ijepa.git artifacts/ijepa
git -C artifacts/ijepa checkout 52c1ae95d05f743e000e8f10a1f3a79b10cff048
curl --fail --location --continue-at - --output artifacts/checkpoints/IN1K-vit.h.14-300e.pth.tar https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
ijepa-decoder decode --image your-image.jpg --top 4 --left 4 --output artifacts/prediction.png
```

`--top` / `--left` are patch coordinates (0 through 9) in the **224x224
center-cropped view**, not original-image pixel coordinates. The hidden block
is always 7x7 patches. `--device auto` selects CUDA, MPS, or CPU; use an explicit
device when needed. Inference loads the frozen context encoder and predictor,
but **no target encoder**. It also saves the native RGB crop and a 224x224
composite beside the comparison grid.

If v0.1 already occupies `artifacts/checkpoints/native`, download v0.2 into a
new directory with `ijepa-decoder download --out-dir artifacts/checkpoints/native-v0.2`
and pass `--decoder artifacts/checkpoints/native-v0.2` to inference. The
[v0.1.0 release](https://github.com/yaskovdev/ijepa-decoder/releases/tag/v0.1.0)
is preserved and can still be downloaded explicitly with `--version v0.1.0`.

## Reproduce Training

The shipped [reference manifest](ijepa_decoder/manifest.json) preserves the
original v0.1 photographs and masks. The v0.2 generator expands it deterministically
to **9,469 training photographs, 37,876 masks, and 512 validation photographs**.
The [exact expanded manifest](https://github.com/yaskovdev/ijepa-decoder/releases/download/v0.2.0/manifest.json)
is also a release asset. Neither manifest contains dataset images. Preparation
rejects changed pixels or train/validation overlap and verifies hidden-pixel isolation.

```shell
pip install -c constraints.txt -e '.[train]'
mkdir -p artifacts/data
curl --fail --location --continue-at - --output artifacts/data/imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf artifacts/data/imagenette2-320.tgz -C artifacts/data
ijepa-decoder assets
ijepa-decoder download --version v0.1.0 --out-dir artifacts/checkpoints/v0.1.0
ijepa-decoder manifest --output artifacts/scaled/manifest.json
ijepa-decoder prepare --manifest artifacts/scaled/manifest.json --out-dir artifacts/scaled/features --batch-size 8 --shard-size 128
ijepa-decoder train --feature-source mixed --cache-dir artifacts/scaled/features \
  --init artifacts/checkpoints/v0.1.0 --out-dir artifacts/scaled/mixed \
  --steps 15000 --batch-size 16 --learning-rate 5e-5 --schedule cosine \
  --min-learning-rate 1e-6 --ema-decay 0.999 --eval-every 250
ijepa-decoder evaluate --cache artifacts/scaled/features/val.pt
ijepa-decoder compare --decoder artifacts/scaled/mixed/best.pt --output artifacts/scaled/comparison.png
```

`evaluate` defaults to the released model; use
`--decoder artifacts/scaled/mixed/best.pt` to evaluate a retrained checkpoint.
Training uses original native RGB crops, L1 + 0.5 frozen AlexNet LPIPS,
and fixed ImageNet normalization, not unknown hidden-patch statistics.
Mixed batches alternate target/predictor examples 50/50; selection always uses
predictor validation. All I-JEPA models remain frozen.

The mixed-training loop took **3,452 seconds on an M3 Max after feature caching**;
the comparable predictor-only run took 3,374 seconds. Frozen feature extraction
took approximately 47 minutes before consolidation. Exact bitwise
retraining is not promised across hardware or package versions. Use new output
directories for repeat experiments; `--init` loads weights, not optimizer state.
Datasets, features, weights, and generated results stay under ignored `artifacts/`.

Preparation processes bounded shards, reuses each target-encoder pass across
its photograph's masks, and consolidates memory-mappable caches after freeing
the frozen networks. Originals are stored once; final caches occupy about
11 GB. Temporary shards need another approximately 11 GB during preparation.
Repeat the preparation command with `--resume` after interruption.

Training compares raw and EMA weights on the same validation data. `best.pt`
contains the selected decoder; `last.pt` also stores optimizer, cosine schedule,
EMA, and random states. Resume in the same output directory with the original
arguments, replacing `--init ...` with `--resume .../last.pt`. `--stop-after`
pauses at an absolute step without shortening the full `--steps` schedule.
Constant learning rate and no EMA remain the CLI defaults for v0.1 reproduction;
the v0.2 commands above enable cosine decay and EMA explicitly.
Comparisons use identical predictor features for both models, include wrong-photo
and shuffled-token controls, and report paired bootstrap intervals grouped by
original photograph. These intervals do not account for checkpoint selection on
the same validation set. The target-encoder control is explicitly separate from
the predictor-only visualization.

To reproduce the predictor-only comparison, repeat the 15,000-step command with
`--feature-source predictor --out-dir artifacts/scaled/predictor`. It uses the
same initializer, data, seed, schedule, and EMA settings; its best checkpoint was
EMA step 6,250. Mixed training performed slightly better.

To reproduce the v0.1 initializer from the MAE prior, use `ijepa-decoder prepare`
with the packaged reference manifest, then train 1,000 target-token steps followed
by 800 mixed steps at learning rate 5e-5:

```shell
ijepa-decoder prepare
ijepa-decoder train --feature-source target --steps 1000 --out-dir artifacts/target_1000
ijepa-decoder train --feature-source mixed --steps 800 --learning-rate 5e-5 --init artifacts/target_1000/best.pt --out-dir artifacts/mixed_800
```

Predictor validation selected mixed step 400. Its original training loops took
118 + 95 seconds after feature caching. Use `compare --cache artifacts/features/val.pt`
for the fixed 64-image comparison against the preserved v0.1 release.

## Tests and Attribution

```shell
python -m unittest discover -s tests -v
```

The focused tests use synthetic data and small models without model downloads.
This standalone project retains the git history and original MIT notices from
[RCDM](https://github.com/facebookresearch/RCDM) /
[OpenAI guided-diffusion](https://github.com/openai/guided-diffusion);
unrelated experiment code is absent from the current tree.

**Code: [MIT](LICENSE). Released model weights: CC-BY-NC 4.0.**
Official [I-JEPA code/weights](https://github.com/facebookresearch/ijepa/blob/main/LICENSE)
also remain CC-BY-NC 4.0. The pinned
[MAE model card](https://huggingface.co/facebook/vit-mae-base/blob/25b184bea5538bf5c4c852c79d221195fdd2778d/README.md)
lists Apache-2.0. This I-JEPA-based workflow is for noncommercial research;
the code license does not relicense models or
[Imagenette/ImageNet](https://github.com/fastai/imagenette) photographs.

Related papers: [I-JEPA](https://arxiv.org/abs/2301.08243),
[MAE](https://arxiv.org/abs/2111.06377), [RCDM](https://arxiv.org/abs/2112.09164).

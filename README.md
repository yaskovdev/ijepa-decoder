# An Attempt to Create a Decoder for I-JEPA

An attempt to create a decoder for [I-JEPA](https://github.com/facebookresearch/ijepa) based on RCDM, work in progress.

Models requiring `vissl` currently can't be sampled from.

## Small I-JEPA Predictor Decoder on Apple Silicon

This pilot trains a small RCDM on **frozen official I-JEPA predictor outputs**,
not on full-image encoder features. It is not Meta's unpublished decoder.
The context encoder receives only visible patch tokens. The predictor receives
the hidden block's positions, and its outputs are average-pooled into a
1280-dimensional conditioning vector. The diffusion target is the corresponding
RGB crop. Preparation verifies that changing hidden pixels does not change the
conditioning.

Use a separate environment. The commands below reuse an existing PyTorch 2.7.1
installation; otherwise install the matching PyTorch/torchvision versions there.

```shell
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e . "torch==2.7.1" "torchvision==0.22.1"

mkdir -p artifacts/checkpoints artifacts/data
git clone --depth 1 https://github.com/facebookresearch/ijepa.git artifacts/ijepa
curl --fail --location --continue-at - --output artifacts/checkpoints/IN1K-vit.h.14-300e.pth.tar https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
curl --fail --location --continue-at - --output artifacts/data/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf artifacts/data/imagenette2-160.tgz -C artifacts/data

.venv/bin/python -m scripts.ijepa_decoder prepare --device mps --train-limit 1000 --val-limit 64
.venv/bin/python -m scripts.ijepa_decoder train --device mps --steps 2000
.venv/bin/python -m scripts.ijepa_decoder sample --device mps --checkpoint artifacts/ijepa_decoder/ema_0.99_002000.pt
```

The full I-JEPA checkpoint is approximately 10.4 GB and includes the predictor;
an encoder-only Hugging Face checkpoint is insufficient. Prepared features,
checkpoints and grids are saved under the ignored `artifacts/` directory.
Training and validation are cached separately. Once preparation exits, decoder
training no longer needs I-JEPA in GPU memory.

The short pilot uses 64x64 targets, a 32-channel U-Net, and EMA 0.99 rather than
the original long-run EMA 0.9999. The grid shows the original, masked context,
true target, multiple generated samples, and a different conditioning vector
with the same initial noise. Generalization and sensitivity to conditioning,
not only training loss, determine whether a larger run is worthwhile.

An optional `train --predict-xstart` selects RCDM's existing clean-image
prediction objective instead of noise prediction. Use a separate `--out-dir`
when comparing objectives; sampling reads the matching objective from the saved
configuration.

**Pilot limitation:** With 1,000 Imagenette training crops, the initial
2,000-step noise-prediction run still yields strongly colored, noisy images.
A separate 500-step clean-image run yields generic blurred patterns. Neither
is yet a convincing visualization of I-JEPA predictions; the wrong-conditioning
control remains visually similar. This establishes local execution, not decoder
quality.

Resume with `train --steps TOTAL_STEPS --resume-checkpoint PATH_TO_MODEL_PT`.
Keep the matching optimizer and EMA files in the same directory. Model settings
are recorded in `decoder_config.json` and checked before resuming.
The standard `scripts/image_train.py` also accepts `--device mps`,
`--feature_cache PATH`, and `--max_steps TOTAL_STEPS`.

The official I-JEPA source and weights are licensed
[CC-BY-NC 4.0](https://github.com/facebookresearch/ijepa/blob/main/LICENSE).
Their use is restricted to noncommercial purposes.

### Frozen RCDM Prior and a Small Predictor Adapter

This is the more useful starting point after preparing the I-JEPA caches.
Keep the released DINO-RCDM generator frozen, and train only a small
`1280 -> 512 -> 2048` adapter from predictor outputs to DINO features of the
corresponding hidden RGB crops. I-JEPA is also frozen. Learning an image prior
from scratch is no longer necessary.

```shell
curl --fail --location --continue-at - --output artifacts/checkpoints/dino_resnet50_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
curl --fail --location --continue-at - --output artifacts/checkpoints/rcdm_ema_dino_trunk.pt https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_dino_trunk.pt

.venv/bin/python -m scripts.ijepa_adapter prepare --device mps
.venv/bin/python -m scripts.ijepa_adapter sample --device mps --oracle-only --output artifacts/ijepa_adapter/oracle_baseline.png
.venv/bin/python -m scripts.ijepa_adapter train --device mps --steps 2000
.venv/bin/python -m scripts.ijepa_adapter sample --device mps --adapter artifacts/ijepa_adapter/best.pt
```

The additional weights total approximately 1.8 GB. DINO targets use 224x224 RGB
in `[-1, 1]`, **without ImageNet or L2 normalization**, matching the original
RCDM extraction code. Crops come from the cached original images and exact hidden
boxes, not from upsampling the low-resolution diffusion targets. Model/geometry,
image paths, and boxes are checked when pairing caches.

Adapter normalization statistics come only from training features. `best.pt`
is selected using validation feature error; `last.pt` is not necessarily better.
In the initial 1,000-image run, the best recorded checkpoint was step 100 rather
than step 2000. Its validation normalized MSE was 0.795 versus 0.961 for constant
mean features; cosine similarity was 0.810 versus 0.641 with shuffled pairing.

The grid columns are original, masked context, true target, **DINO oracle**,
predictor samples, **in context**, and wrong-predictor control. The oracle uses the true hidden
RGB only as a decoder-quality reference. Predictor columns receive only the
adapted predictor vector. DDIM uses identical initial noise for oracle,
predictor sample 1, and the wrong-predictor control.

The **in context** column pastes predictor sample 1 into its exact hidden
rectangle at the original image resolution. All visible pixels remain unchanged;
the hidden rectangle is fully replaced, without blending with its true pixels.
This is a visualization, not a spatially conditioned inpainting model, so seams
and inconsistent object alignment are expected. Oracle-only grids do not label
ground-truth-conditioned generations as predictions.

The initial predictor grid contains recognizable vehicle, golf-ball and dog
hypotheses, with a less faithful audio-equipment example. This is a small
Imagenette proof of concept, not Meta's decoder or a full ImageNet evaluation.
Validation is withheld from adapter training, not necessarily from the
pretrained backbones or image prior. The released
[RCDM weights are also noncommercial](https://dl.fbaipublicfiles.com/rcdm/LICENSE).

### More Data, Multiple Masks, and a Target-Encoder Control

Use Imagenette 320 for the next bounded Mac experiment. The preparation command
below uses all 9,469 training photographs and 256 validation photographs, with
three distinct hidden-block positions per photograph: 28,407 and 768 examples.
Split photographs before adding masks. Original images are stored only once
and accessed through `image_indices`; DINO pairing also checks this mapping and
the hashes of the preprocessed original pixels.

```shell
.venv/bin/python -m scripts.ijepa_decoder prepare --device mps --data-dir artifacts/data/imagenette2-320 --out-dir artifacts/ijepa_features_320_m3 --train-limit 10000 --val-limit 256 --masks-per-image 3 --with-target-encoder
.venv/bin/python -m scripts.ijepa_adapter prepare --device mps --cache-dir artifacts/ijepa_features_320_m3 --out-dir artifacts/dino_crop_features_320_m3

.venv/bin/python -m scripts.ijepa_adapter train --device mps --cache-dir artifacts/ijepa_features_320_m3 --dino-cache-dir artifacts/dino_crop_features_320_m3 --kind ridge --ridge-alpha 1 --out-dir artifacts/ijepa_adapter_320_ridge1
.venv/bin/python -m scripts.ijepa_adapter train --device mps --cache-dir artifacts/ijepa_features_320_m3 --dino-cache-dir artifacts/dino_crop_features_320_m3 --hidden-dim 128 --dropout 0.25 --weight-decay 0.1 --lr 0.0003 --log-interval 25 --out-dir artifacts/ijepa_adapter_320_mlp128
.venv/bin/python -m scripts.ijepa_adapter train --device mps --cache-dir artifacts/ijepa_features_320_m3 --dino-cache-dir artifacts/dino_crop_features_320_m3 --feature-source target --kind ridge --ridge-alpha 1 --out-dir artifacts/ijepa_target_adapter_320_ridge1

.venv/bin/python -m scripts.ijepa_adapter sample --device mps --cache artifacts/ijepa_features_320_m3/val.pt --dino-cache artifacts/dino_crop_features_320_m3/val.pt --adapter artifacts/ijepa_adapter_320_ridge1/best.pt --target-adapter artifacts/ijepa_target_adapter_320_ridge1/best.pt --reference-adapter artifacts/ijepa_adapter/best.pt --output artifacts/ijepa_adapter_320_ridge1/comparison.png
```

Ridge fits a regularized linear adapter using a float64 solve on CPU; inference
and MLP training still use MPS float32. Compare ridge strengths such as 0.1, 1
and 10, and smaller MLPs, selecting `best.pt` by held-out feature error rather
than training duration. Sampling chooses distinct photographs by default, and
wrong-conditioning controls use another photograph, not another mask of the
same one.

The optional **IJEPA target** column uses the actual target encoder, which sees
hidden pixels, through a target-trained adapter. **Via target** applies that
same adapter to predictor outputs instead. These help diagnose mapping quality
and predictor/target mismatch; the gap can reflect information loss or
distribution shift, not just decoder error. Neither reference is an input to
the primary predictor adapter. Target tokens are layer-normalized individually before
masking and pooling, as in official I-JEPA training. **Reference** uses the
previous adapter on the same examples and boxes. All reference columns and
predictor sample 1 share the same DDIM noise. Use `--sampling-steps 250` for a
separate sampling-budget comparison without changing the pretrained resolution.

On the expanded caches, the best ridge strength was **0.1**. The smaller
256-hidden-unit MLP was nearly tied. The following comparisons use the same
768 validation regions, and all errors use the same DINO normalization scales
fitted on the expanded training data:

| Decoder input / adapter | Normalized feature MSE | DINO cosine |
| --- | ---: | ---: |
| Predictor / original 1,000-image MLP | 0.870 | 0.769 |
| Predictor / expanded ridge 0.1 | **0.698** | **0.823** |
| Predictor / expanded MLP 256 | 0.701 | 0.822 |
| Actual I-JEPA target / target ridge 0.1 (oracle) | 0.619 | 0.847 |
| Predictor / that same target ridge 0.1 | 0.752 | 0.805 |

The expanded pipeline reduces held-out **feature** error by approximately 20%
relative to the original adapter; this is not a pixel-reconstruction score.
Predictor and target pooled vectors have mean cosine 0.961, but even small
representation errors can affect the decoded image. The experiment remains
restricted to Imagenette's ten classes and does not establish a universal decoder.
Full metrics are in `artifacts/ijepa_adapter_320_results.json`.

The selected predictor and target checkpoints are
`artifacts/ijepa_adapter_320_ridge0.1/best.pt` and
`artifacts/ijepa_target_adapter_320_ridge0.1/best.pt`.
Controlled grids are `artifacts/ijepa_adapter_320_ridge0.1/comparison_100.png`
and `comparison_250.png` in the same directory. The 100-step grid includes the
old adapter on the new examples, target controls, and the in-context column;
the 250-step grid uses the same first two examples and initial noises.

The context column exposes persistent object-scale and alignment errors,
including dog-part crops: this is still semantic visualization, not faithful
spatial inpainting. In the two inspected regions, 250 DDIM steps offered no
clear semantic improvement over 100.

## Sample Using PyCharm on macOS with Remote Linux Interpreter

```shell
# Run the below commands on the Linux machine.
# Tested with "Linux pop-os 6.9.3-76060903-generic #202405300957~1726766035~22.04~4092a0e SMP PREEMPT_DYNAMIC Thu S x86_64 x86_64 x86_64 GNU/Linux" and "conda 25.11.1".

conda create -n rcdm python=3.12
conda activate rcdm

# Taken from https://pytorch.org/get-started/locally/, to pick CUDA version check it with `nvidia-smi` command.
# Or, actually, that's probably not the real CUDA version, see https://stackoverflow.com/a/55717476/1862286.
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

mkdir ~/rcdm
cd ~/rcdm
# Create a new Conda SSH interpreter in PyCharm and specify the newly created as the target of "Sync folders" while creating your SSH interpreter in PyCharm
pip install -e .

# Note: Make sure that the `/home/yaskovdev/rcdm_images` contains at least one image, otherwise sampling will get stuck.
python scripts/image_sample.py --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --batch_size 8 --num_images 1 --timestep_respacing 100 --data_dir "/home/yaskovdev/rcdm_images" --type_model supervised
```

You can then run `scp yaskovdev@192.168.1.186:/home/yaskovdev/rcdm/samples.jpeg ~/Downloads/rcdm_samples` on macOS.

# Train Using PyCharm on macOS with Remote Linux Interpreter

Download the dataset: https://www.kaggle.com/datasets/thbdh5765/ilsvrc2012/data.

Extract the archive to a `/home/yaskovdev/images` folder.

```shell
# Run the below commands on the Linux machine.
cd ~/rcdm
conda activate rcdm

export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 8 --save_interval 5000"

# Train on a PC with 2 GPUs:
torchrun --nproc_per_node=2 scripts/image_train.py $MODEL_FLAGS_128 $TRAIN_FLAGS --feat_cond --data_dir /home/yaskovdev/images --type_model dino --out_dir /home/yaskovdev/save_dir_128

# To resume an interrupted run, add --resume_checkpoint /home/yaskovdev/save_dir_128/model050000.pt to the training command, using your actual checkpoint path.

# Sample from the trained model:
python scripts/image_sample.py $MODEL_FLAGS_128 --batch_size 8 --num_images 1 --timestep_respacing 100 --data_dir "/home/yaskovdev/rcdm_images" --model_path /home/yaskovdev/save_dir_128/model090000.pt --type_model dino
```

## Sample on Windows

```powershell
# Start Anaconda Prompt (tested with Miniconda 23.7.4)

conda create -n rcdm python=3.12
conda activate rcdm

# Taken from https://pytorch.org/get-started/locally/, to pick CUDA version check it with `nvidia-smi` command
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

cd C:\dev\git_home\RCDM
pip install -e .

python scripts/image_sample.py --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --batch_size 8 --num_images 1 --timestep_respacing 100 --data_dir "c:\dev\rcdm_images_3" --type_model supervised
```

Tried with `--type_model dino` as well.

# Train on Windows

```powershell
# Start Anaconda Prompt (tested with Miniconda 23.7.4)

python scripts/image_train.py
```

# Representation Conditional Diffusion Model (RCDM)

![RCDM's samples of earth](images/RCDM_front.png?raw=true "samples earth")

This is the codebase for [High Fidelity Visualization of What Your Self-Supervised Representation Knows About](https://arxiv.org/abs/2112.09164)

This repo is a fork of the github repository [guided-diffusion](https://github.com/openai/guided-diffusion) which is the codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

Instead of using the gradient of a classifier to guide the sampling, our model uses the features of supervised or self-supervised models as conditioning.

# How to install

Create a new conda environement
```
conda create --name RCDM python=3.9
conda activate RCDM
```
Clone this repository, navigate to it, then run:
```
pip install -e .
```

# Repository structure
All the training and sampling scripts are located in scripts/. The SSL models that we used to compute the representations from are defined in the file ./guided_diffusion_rcdm/get_ssl_models.py. The links of the pretrained RCDM models are located in ./guided_diffusion/get_rcdm_models.py.

## Model FLAGS used in this repository
All model availables through this repository were trained on a resolution of 128x128 pixels. In order to use the pretrained models, you'll need to export these following FLAGS:
```
export MODEL_FLAGS_128="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma True --noise_schedule linear --num_channels 256 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
export TRAIN_FLAGS="--lr 1e-4 --batch_size 8"
```

If you don't want to use the pretrained models, you can use the FLAGS used in the original github repository [guided-diffusion](https://github.com/openai/guided-diffusion) to train RCDM in 64x64 or 256x256.

# Training RCDM with self-supervised representations
We use submitit to run the distributed training accross several nodes. You'll probably need to adapt the file run_with_submitit.py depending on your cluster configuration. The command below run an RCDM training over the features extracted from Dino (you can change the feature extractor by changing the --type_model argument or by updating the file ./guided_diffusion/get_ssl_models.py).
```
python run_with_submitit.py --nodes 4 --ngpus 8 --use_volta32 $MODEL_FLAGS_128 $TRAIN_FLAGS --feat_cond --data_dir DATASET_DIR --type_model dino --out_dir SAVE_DIR
```

# Use a pretrained model

To use any of the following pretrained models, you will need to install vissl:
```
pip install vissl
```

This repository comes with a set of model that were pretrained on ImageNet (please note that even if the code is released under the MIT licence, the weights of the model are only available under the [CC-BY-NC license](https://dl.fbaipublicfiles.com/rcdm/LICENSE)).  The models will be automatically downloaded when using any of the sampling scripts. However, you can also download them manually from the following links:

| Model | SSL Model | Type (Trunk/Head) | url |
|-------------------|-------------------|-------------------|---------------------|
| RCDM | SimCLR | Trunk | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_simclr_trunk.pt) | 
| RCDM | SimCLR | Head | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_simclr_head.pt) | 
| RCDM | Barlow Twins | Trunk | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_barlow_trunk.pt) | 
| RCDM | Barlow Twins | Head | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_barlow_head.pt) | 
| RCDM | VICReg | Trunk | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_vicreg_trunk.pt) | 
| RCDM | VICReg | Head | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_vicreg_head.pt) | 
| RCDM | Dino | Trunk | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_dino_trunk.pt) | 
| RCDM | Dino | Head | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_dino_head.pt) | 
| RCDM | Supervised | N/A | [model](https://dl.fbaipublicfiles.com/rcdm/rcdm_ema_supervised.pt) | 

# Conditional sampling with RCDM
![RCDM's sampling with different SSL representations](images/samples.jpeg?raw=true "sampling")

To sample from these models, you can use the `image_sample.py` script. You need to replace DATA_PATH by the directory in which the images that will be used to compute the ssl representations are saved. The num_images parameters indicates how many images in the directory DATA_PATH you want to use. The type_model parameters indicates which SSL model to use as feature extractor. You can add --model_path to specify which RCDM checkpoints to use (Otherwise, one of the pretrained model will be downloaded automatically). You can use the argument --use_head to use the head instead of the trunk as feature extrator. The command bellow will generated 16 samples, with 8 samples par images in the DATA_PATH directory by using the representation given by Barlow Twins.

```
python scripts/image_sample.py $MODEL_FLAGS_128 --batch_size 8 --num_images 2 --timestep_respacing 100 --data_dir DATA_PATH --type_model barlow
```

## Interpolation of two images
![RCDM's sampling with interpolation](images/interpolation.jpeg?raw=true "interpolation")

To interpolate between the representations of two images, you can use the `image_sample_interpolation.py` script.
Replace MODEL_PATH by the relevant model checkpoint (or leave blank to use the pretrained model) and FIRST_IMAGE_PATH/SECOND_IMAGE_PATH by the path of the images you want to interpolate in the ssl representation space.

```
python scripts/image_sample_interpolation.py $MODEL_FLAGS_128 --batch_size 8 --timestep_respacing 100 --type_model barlow --first_image_path FIRST_IMAGE_PATH --second_image_path SECOND_IMAGE_PATH
```

## Manipulation of representations
![RCDM's manipulation of representations](images/manipulation.jpeg?raw=true "Dog's clothes")

To manipulate representations, you can use the `image_sample_manipulation.py` script.
Replace MODEL_PATH by the relevant model checkpoint, IMAGE_PATH by the path of the images you want to extract the attributes from, NN_PATH by the folder's path of the images in which to find the nearest neigbors and TARGET_PATH by the path of the image on which to apply the attribute on.

```
python scripts/image_sample_manipulation.py $MODEL_FLAGS_128 --batch_size 8 --timestep_respacing 100 --dist --type_model simclr --image_path IMAGE_PATH --folder_nn_path NN_PATH --image_target_path TARGET_PATH
```

If you want to reproduce the image above, just use:
```
python scripts/image_sample_manipulation.py $MODEL_FLAGS_128 --batch_size 8 --timestep_respacing 100 --type_model simclr  --image_path IMAGENET_PATH/val/n02099601/ILSVRC2012_val_00027833.JPEG --folder_nn_path IMAGENET_PATH/val/n02099601/ --image_target_path IMAGENET_PATH/val/n02099601/ILSVRC2012_val_00001123.JPEG
```

# LICENSE
The code is distributed under the MIT license. However the weights of the pretrained models that are available for download are under the CC-BY-NC license and should not be use for commercial purpose.

# Cite the paper
```
@article{
      bordes2022high,
      title={High Fidelity Visualization of What Your Self-Supervised Representation Knows About},
      author={Florian Bordes and Randall Balestriero and Pascal Vincent},
      journal={Transactions on Machine Learning Research},
      year={2022},
      url={https://openreview.net/forum?id=urfWb7VjmL},
      note={}
}

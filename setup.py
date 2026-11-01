# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from setuptools import setup

setup(
    name="guided-diffusion-rcdm",
    py_modules=["guided_diffusion_rcdm"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "numpy", "torchvision"],
    extras_require={
        "rae": ["transformers>=4.51,<5", "huggingface_hub>=0.30,<1"],
        "native": ["transformers>=4.51,<5", "huggingface_hub>=0.30,<1", "lpips>=0.1.4,<0.2"],
    },
)

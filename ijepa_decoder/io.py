import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch as th
from PIL import Image, ImageDraw


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, path, sha256):
    path = Path(path)
    if path.exists():
        if file_sha256(path) != sha256:
            raise ValueError(f"Existing file has the wrong checksum: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)
    if file_sha256(temporary) != sha256:
        temporary.unlink()
        raise ValueError(f"Downloaded file has the wrong checksum: {url}")
    temporary.replace(path)
    return path


def configure(args):
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive.")
    th.set_num_threads(args.cpu_threads)
    th.manual_seed(args.seed)
    name = args.device
    if name == "auto":
        name = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    if name == "cuda" and not th.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    if name == "mps" and not th.backends.mps.is_available():
        raise ValueError("MPS is unavailable.")
    return th.device(name)


def save_cache(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    th.save(data, temporary)
    temporary.replace(path)


def image_pixels(path, size=224):
    # Preserve the OpenAI/RCDM center-crop preprocessing used by the manifest.
    with Image.open(path) as source:
        image = source.convert("RGB")
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(value // 2 for value in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(value * scale) for value in image.size), Image.Resampling.BICUBIC)
    array = np.array(image)
    y, x = (array.shape[0] - size) // 2, (array.shape[1] - size) // 2
    return th.from_numpy(array[y:y + size, x:x + size].copy()).permute(2, 0, 1).contiguous()


def normalized(pixels, device):
    image = pixels.float().to(device) / 255
    mean = th.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = th.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    return (image - mean) / std


def rgb_pixels(images):
    if not th.isfinite(images).all():
        raise FloatingPointError("Cannot render non-finite RGB.")
    return (images.cpu().clamp(0, 1) * 255).round().to(th.uint8)


def as_image(pixels):
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def compose(original, prediction, box):
    x0, y0, x1, y1 = box
    if prediction.shape != (3, y1 - y0, x1 - x0) or prediction.dtype != th.uint8:
        raise ValueError("Prediction must be native-size uint8 RGB matching the mask.")
    result = original.clone()
    result[:, y0:y1, x0:x1] = prediction
    return result


def save_grid(rows, labels, path, cell=160):
    canvas = Image.new("RGB", (len(labels) * cell, len(rows) * (cell + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        if len(row) != len(labels):
            raise ValueError("Image row does not match its labels.")
        for column, image in enumerate(row):
            y = index * (cell + 20)
            draw.text((column * cell + 3, y + 3), labels[column], fill="black")
            canvas.paste(image.resize((cell, cell), Image.Resampling.NEAREST), (column * cell, y + 20))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)

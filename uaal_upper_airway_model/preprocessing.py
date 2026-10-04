"""Shared image preprocessing for the UAAL upper-airway model.

Used both when building the training set (prepare_data.py) and at inference time,
so live bronchoscope frames go through exactly the same steps as training images:

    frame -> grayscale -> crop to the scope view -> resize (longest side <= max_side)
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from scipy import ndimage

DARK_THRESHOLD = 20    # gray level at or below which a pixel counts as black screen border
MIN_SCOPE_AREA = 0.25  # a detected view smaller than this fraction of the frame is not trusted
DETECT_SIDE = 480      # detection runs on a downscaled copy for speed


def detect_scope_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the (x0, y0, x1, y1) box of the endoscope view inside a screen capture.

    Clinical captures show the scope image (octagon or rectangle) on a black screen with
    patient text, icons and sometimes a picture-in-picture thumbnail around it. The scope
    image is the largest bright connected region; the overlays are small separate regions
    and fall outside its box. Returns None when no reliable region is found (very dark
    frame), in which case the caller should keep the full frame.
    """
    h, w = gray.shape
    scale = min(1.0, DETECT_SIDE / max(h, w))
    if scale < 1.0:
        small = np.asarray(
            Image.fromarray(gray).resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        )
    else:
        small = gray
    mask = ndimage.binary_opening(small > DARK_THRESHOLD, iterations=2)
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    ys, xs = np.nonzero(labels == sizes.argmax())
    pad = 2  # pixels of the downscaled image, compensates for the opening
    x0 = max(0, math.floor((xs.min() - pad) / scale))
    y0 = max(0, math.floor((ys.min() - pad) / scale))
    x1 = min(w, math.ceil((xs.max() + 1 + pad) / scale))
    y1 = min(h, math.ceil((ys.max() + 1 + pad) / scale))
    if (x1 - x0) * (y1 - y0) < MIN_SCOPE_AREA * w * h:
        return None
    return x0, y0, x1, y1


def resize_to_max_side(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Downscale so the longest side is at most max_side (never upscales). Returns (image, scale)."""
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img, scale


def preprocess_frame(
    img: Image.Image, max_side: int = 1024, grayscale: bool = True, crop: bool = True
) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """Inference-time preprocessing. Returns (model_input, crop_box, scale).

    A detection (x, y) on model_input maps back to the original frame as
    (x / scale + crop_box[0], y / scale + crop_box[1]).
    """
    gray = np.asarray(img.convert("L"))
    box = (detect_scope_box(gray) if crop else None) or (0, 0, img.width, img.height)
    out = (img.convert("L") if grayscale else img.convert("RGB")).crop(box)
    out, scale = resize_to_max_side(out, max_side)
    return out, box, scale

"""Shared image preprocessing for the BM-BronchoLC lower-airway model.

Used both when building the training set (prepare_data.py) and at inference time, so live
frames go through the same steps as training images:

    frame -> find scope view -> remove burned-in text -> black outside the view
          -> crop to the view -> grayscale -> resize (longest side <= max_side)

BM-BronchoLC frames are screen captures: an octagonal (or rectangular) scope view on a black
screen with patient fields, date/time, scope settings and icons around it. In most videos
part of that text is drawn over the left edge of the view itself.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

DARK_THRESHOLD = 20      # gray level at or below which a pixel is black screen
PROFILE_FRACTION = 0.5   # a column/row belongs to the view if it is at least this bright relative to the brightest
RED_WARNING = 60         # R - B above this = red on-screen warning box ("Lamp Error", "WHT BAL incomplete")
MAX_WARNING_AREA = 0.2   # a red region larger than this fraction of the view is tissue, not a warning box
TEXT_BAND = 0.25         # burned-in text only overlaps the left 25% of the view
TEXT_STROKE_PX = 11      # text strokes and marks are thinner than this (morphological top-hat size)
TEXT_TOPHAT = 30         # a text stroke is this much brighter than its local background
TEXT_REL_SATURATION = 0.5  # ...and less saturated than half the frame's typical tissue saturation
TEXT_RED_RESIDUAL = 25   # anti-aliased text edges: thin structures much redder than their surroundings
MIN_VIEW_AREA = 0.2      # a view smaller than this fraction of the frame is not trusted


def _run_around_peak(profile: np.ndarray, fraction: float) -> tuple[int, int]:
    """Contiguous index range around the maximum where profile >= fraction * max."""
    ok = profile >= profile.max() * fraction
    a = b = int(np.argmax(profile))
    while a > 0 and ok[a - 1]:
        a -= 1
    while b < len(profile) - 1 and ok[b + 1]:
        b += 1
    return a, b + 1


def detect_view_mask(rgb: np.ndarray) -> np.ndarray | None:
    """Boolean mask of the scope view, or None if no reliable view is found.

    1. bright = not black screen (red warning boxes count as bright so a banner crossing the
       view does not cut it in two; they are removed later by detect_text_overlay)
    2. the view spans most rows of its columns (and vice versa); text columns are mostly black,
       so a column/row brightness profile isolates the view's bounding range
    3. inside that range, an opening removes thin text strokes attached to the view edge,
       the largest region is the view, and its convex hull restores the octagon shape
       (dark lumen areas touching the edge would otherwise cut notches into it)
    """
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    bright = gray > DARK_THRESHOLD
    bright = ndimage.binary_opening(bright, iterations=2)
    if not bright.any():
        return None
    x0, x1 = _run_around_peak(bright.mean(axis=0), PROFILE_FRACTION)
    y0, y1 = _run_around_peak(bright[:, x0:x1].mean(axis=1), PROFILE_FRACTION)
    inner = np.zeros_like(bright)
    inner[y0:y1, x0:x1] = bright[y0:y1, x0:x1]
    inner = ndimage.binary_opening(inner, iterations=max(2, round(min(gray.shape) / 80)))
    labels, n = ndimage.label(inner)
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    ys, xs = np.nonzero(labels == sizes.argmax())
    try:
        hull = ConvexHull(np.c_[xs, ys])
    except QhullError:
        return None
    mask_img = Image.new("L", (gray.shape[1], gray.shape[0]), 0)
    ImageDraw.Draw(mask_img).polygon([(float(xs[i]), float(ys[i])) for i in hull.vertices], fill=255)
    mask = np.asarray(mask_img) > 0
    if mask.mean() < MIN_VIEW_AREA:
        return None
    return mask


def detect_text_overlay(rgb: np.ndarray, view: np.ndarray) -> np.ndarray:
    """Mask of burned-in text inside the view's left band, plus red warning boxes anywhere in the view.

    Text = thin bright strokes (white top-hat) that are less saturated than this frame's tissue,
    plus anti-aliased stroke edges (thin, much redder than the surroundings on blue NBI tissue).
    Using the frame's own tissue saturation keeps pale NBI frames from being treated as text.
    """
    a = rgb.astype(np.int16)
    xs = np.nonzero(view.any(axis=0))[0]
    band = np.zeros_like(view)
    band[:, xs.min(): xs.min() + int(TEXT_BAND * (xs.max() + 1 - xs.min()))] = True
    brightest = a.max(axis=2)
    saturation = brightest - a.min(axis=2)
    tophat = brightest - ndimage.grey_opening(brightest, size=(TEXT_STROKE_PX, TEXT_STROKE_PX))
    neutral = saturation < max(12.0, TEXT_REL_SATURATION * float(np.median(saturation[view])))
    red = a[..., 0]
    red_residual = (red - ndimage.median_filter(red, size=11)) > TEXT_RED_RESIDUAL
    text = ((tophat > TEXT_TOPHAT) & neutral | red_residual & (tophat > TEXT_TOPHAT / 2)) & band & view
    # red warning boxes anywhere in the view, including the white text written inside them
    warning = ndimage.binary_fill_holes(ndimage.binary_dilation((a[..., 0] - a[..., 2]) > RED_WARNING, iterations=3)) & view
    if 0 < warning.sum() < MAX_WARNING_AREA * view.sum():
        text |= warning
    return ndimage.binary_dilation(text, iterations=2) & view


def clean_frame(img: Image.Image, remove_text: bool = True) -> tuple[np.ndarray, tuple[int, int, int, int], dict]:
    """Return (cleaned grayscale array cropped to the view, crop box, info).

    Pixels outside the view are black; text pixels inside the view are filled with the
    nearest clean tissue pixel. The crop box maps back to the original frame.
    """
    rgb = np.asarray(img.convert("RGB"))
    gray = np.asarray(img.convert("L")).copy()
    h, w = gray.shape
    view = detect_view_mask(rgb)
    info = {"view_found": view is not None, "view_fraction": 1.0, "text_fraction": 0.0}
    if view is None:
        return gray, (0, 0, w, h), info
    text = detect_text_overlay(rgb, view) if remove_text else np.zeros_like(view)
    fill_from = text | ~view
    if text.any():
        iy, ix = ndimage.distance_transform_edt(fill_from, return_distances=False, return_indices=True)
        gray = gray[iy, ix]
    gray[~view] = 0
    ys, xs = np.nonzero(view)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    info.update(view_fraction=float(view.mean()), text_fraction=float(text.sum() / view.sum()))
    return gray[box[1]:box[3], box[0]:box[2]], box, info


def resize_to_max_side(img: Image.Image, max_side: int) -> tuple[Image.Image, float]:
    """Downscale so the longest side is at most max_side (never upscales). Returns (image, scale)."""
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img, scale


def preprocess_frame(img: Image.Image, max_side: int = 640, remove_text: bool = True
                     ) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    """Inference-time preprocessing. Returns (model_input, crop_box, scale).

    A detection (x, y) on model_input maps back to the original frame as
    (x / scale + crop_box[0], y / scale + crop_box[1]).
    Set remove_text=False for a clean camera feed without screen overlays (e.g. the trainer
    camera): the text filter assumes blue NBI tissue and could touch pale tissue otherwise.
    """
    gray, box, _ = clean_frame(img, remove_text)
    out, scale = resize_to_max_side(Image.fromarray(gray), max_side)
    return out, box, scale

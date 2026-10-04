#!/usr/bin/env python3
"""UAAL dataset preparation: analyze -> preprocess -> split -> verify.

Reads the original UAAL zips directly (nothing is extracted or modified in dataset/)
and writes a YOLO-format detection dataset that mixes the clinical and phantom subsets.

Stages
  analyze   statistics of the raw dataset (classes, masks, rotated boxes, groups,
            frame similarity, scope-crop behaviour) -> <out>/reports/raw_*
  prepare   class mapping, box fixing, scope crop, grayscale, resize, near-duplicate
            removal, leakage-free group split, YOLO export -> <out>/images, labels, ...
  verify    integrity checks on the prepared dataset + preview images with boxes
  all       analyze + prepare + verify (default)

Usage
  python prepare_data.py                     # run everything with defaults
  python prepare_data.py analyze
  python prepare_data.py prepare --seed 7 --max-side 1024
  python prepare_data.py verify --previews 30
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from preprocessing import detect_scope_box, resize_to_max_side

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "dataset" / "26342779"
OUT_DIR = ROOT / "data" / "uaal_det"

# subset name -> (zip file, top-level folder inside the zip)
SOURCES = {
    "clinical": ("coco_ins_clinical.zip", "coco_ins_clinical"),
    "phantom": ("coco_ins_phantom.zip", "coco_ins_phantom"),
}
OFFICIAL_SPLITS = ("train", "val")
SPLITS = ("train", "val", "test")

# Original UAAL class name -> training class name (None = drop the class).
# Mapping is by NAME because category ids differ between the clinical and phantom files.
# Nostrils are merged (192 + 195 examples, only 16 right nostrils in phantom);
# Nose is kept separate because its box contains the nostril boxes.
CLASS_MAP = {
    "Epiglottis": "epiglottis",
    "Glottis": "glottis",
    "Glottic aperture": "glottic_aperture",
    "Left vocal fold": "left_vocal_fold",
    "Right vocal fold": "right_vocal_fold",
    "Trachea": "trachea",
    "Channel": "channel",
    "Nose": "nose",
    "Left nostril": "nostril",
    "Right nostril": "nostril",
}
CLASS_NAMES = [
    "epiglottis", "glottis", "glottic_aperture", "left_vocal_fold", "right_vocal_fold",
    "trachea", "channel", "nose", "nostril",
]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

FILE_RE = re.compile(r"^(?P<group>.+)_(?P<frame>\d+)\.png$", re.IGNORECASE)
THUMB = 32           # thumbnail side used for frame-similarity / near-duplicate checks
MIN_BOX_PX = 2       # boxes thinner than this (output pixels) are dropped
MAX_BOX_LOSS = 0.3   # a box may lose up to 30% of its area when clipped to the scope view
SCAN_VERSION = 1     # bump when scan_image() changes, invalidates the cache

# Chart colours (validated categorical palette, light surface)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


# --------------------------------------------------------------------------- loading

def load_records(raw_dir: Path) -> list[dict]:
    """One record per image with its annotations converted to axis-aligned xyxy boxes."""
    records = []
    for domain, (zip_name, top) in SOURCES.items():
        with zipfile.ZipFile(raw_dir / zip_name) as z:
            for official in OFFICIAL_SPLITS:
                coco = json.load(z.open(f"{top}/annotations/instances_{official}2017.json"))
                cats = {c["id"]: c["name"] for c in coco["categories"]}
                anns_by_img = defaultdict(list)
                for a in coco["annotations"]:
                    anns_by_img[a["image_id"]].append(a)
                for im in coco["images"]:
                    m = FILE_RE.match(im["file_name"])
                    if not m:
                        raise ValueError(f"unexpected file name {im['file_name']}")
                    w, h = im["width"], im["height"]
                    boxes = []
                    for a in anns_by_img[im["id"]]:
                        xyxy, kind = ann_to_xyxy(a)
                        boxes.append({
                            "orig_class": cats[a["category_id"]],
                            "xyxy": xyxy,
                            "kind": kind,  # polygon | rotated | box
                            "out_of_bounds": xyxy[0] < -0.5 or xyxy[1] < -0.5 or xyxy[2] > w + 0.5 or xyxy[3] > h + 0.5,
                        })
                    records.append({
                        "uid": f"{domain}_{m['group']}_{int(m['frame']):04d}",
                        "domain": domain,
                        "official_split": official,
                        "zip": zip_name,
                        "member": f"{top}/{official}2017/{im['file_name']}",
                        "group": f"{domain}/{m['group']}",
                        "frame": int(m["frame"]),
                        "width": w,
                        "height": h,
                        "boxes": boxes,
                    })
    records.sort(key=lambda r: (r["group"], r["frame"]))
    uids = Counter(r["uid"] for r in records)
    dup = [u for u, c in uids.items() if c > 1]
    if dup:
        raise ValueError(f"duplicate image ids: {dup[:5]}")
    return records


def ann_to_xyxy(a: dict) -> tuple[list[float], str]:
    """Axis-aligned box for a COCO annotation.

    - Polygon masks give the tightest box.
    - Box-only annotations exported by CVAT can be rotated rectangles: `bbox` is the
      UNROTATED rectangle and attributes.rotation the clockwise angle in degrees around
      its centre. The raw bbox is wrong for those (verified visually), so we use the
      axis-aligned box enclosing the rotated rectangle.
    """
    seg = a.get("segmentation") or []
    if isinstance(seg, list) and seg and any(len(p) >= 6 for p in seg):
        xs = [v for p in seg for v in p[0::2]]
        ys = [v for p in seg for v in p[1::2]]
        return [min(xs), min(ys), max(xs), max(ys)], "polygon"
    x, y, w, h = a["bbox"]
    rot = float((a.get("attributes") or {}).get("rotation", 0) or 0)
    if rot % 360:
        t = math.radians(rot)
        cx, cy = x + w / 2, y + h / 2
        pts = [(cx + dx * math.cos(t) - dy * math.sin(t), cy + dx * math.sin(t) + dy * math.cos(t))
               for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
        return [min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts)], "rotated"
    return [x, y, x + w, y + h], "box"


# --------------------------------------------------------------------------- image workers

_ZIPS: dict[str, zipfile.ZipFile] = {}


def _init_worker(raw_dir: str) -> None:
    for zip_name, _ in SOURCES.values():
        _ZIPS[zip_name] = zipfile.ZipFile(Path(raw_dir) / zip_name)


def _read_image(rec: dict) -> Image.Image:
    img = Image.open(io.BytesIO(_ZIPS[rec["zip"]].read(rec["member"])))
    img.load()
    return img


def scan_image(rec: dict) -> dict:
    """Cheap per-image facts: scope-view box and a small grayscale thumbnail."""
    img = _read_image(rec)
    gray = np.asarray(img.convert("L"))
    box = detect_scope_box(gray)
    thumb = np.asarray(Image.fromarray(gray).resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8)
    return {"uid": rec["uid"], "size": list(img.size), "mode": img.mode,
            "scope_box": list(box) if box else None, "thumb": thumb}


def write_image(job: dict) -> dict:
    """Crop, convert, resize and save one image; returns its YOLO label lines."""
    rec, crop, out_img, gray, max_side = job["rec"], job["crop"], job["out_img"], job["gray"], job["max_side"]
    img = _read_image(rec)
    img = img.convert("L") if gray else img.convert("RGB")
    img = img.crop(tuple(crop))
    img, scale = resize_to_max_side(img, max_side)
    img.save(out_img, compress_level=3)
    W, H = img.size
    # exact per-axis factors: the resized size is rounded, so `scale` alone can overshoot the edge
    sx, sy = W / (crop[2] - crop[0]), H / (crop[3] - crop[1])
    lines, dropped = [], []
    for b in rec["boxes"]:
        cls = CLASS_MAP.get(b["orig_class"])
        if cls is None:
            dropped.append((b["orig_class"], "class not used"))
            continue
        x0, y0, x1, y1 = b["xyxy"]
        x0 = (min(max(x0, crop[0]), crop[2]) - crop[0]) * sx
        x1 = (min(max(x1, crop[0]), crop[2]) - crop[0]) * sx
        y0 = (min(max(y0, crop[1]), crop[3]) - crop[1]) * sy
        y1 = (min(max(y1, crop[1]), crop[3]) - crop[1]) * sy
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            dropped.append((b["orig_class"], "too small after resize"))
            continue
        lines.append(f"{CLASS_ID[cls]} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                     f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return {"uid": rec["uid"], "out_size": [W, H], "scale": scale, "lines": lines, "dropped": dropped}


def run_parallel(fn, items, raw_dir: Path, workers: int, label: str) -> list:
    t0, out = time.time(), []
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(raw_dir),)) as ex:
        for i, r in enumerate(ex.map(fn, items, chunksize=16), 1):
            out.append(r)
            if i % 500 == 0 or i == len(items):
                print(f"  {label}: {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def get_scans(records: list[dict], args) -> dict[str, dict]:
    """Scan all images once and cache the result (scope boxes + thumbnails)."""
    cache = args.out_dir / "cache" / "scan.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if int(data["version"]) == SCAN_VERSION and set(data["uids"]) == {r["uid"] for r in records}:
            meta = json.loads(str(data["meta"]))
            return {u: {**meta[u], "thumb": t} for u, t in zip(data["uids"], data["thumbs"])}
    print("Scanning images (scope view + thumbnails)...")
    scans = run_parallel(scan_image, records, args.raw_dir, args.workers, "scan")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, version=SCAN_VERSION,
        uids=np.array([s["uid"] for s in scans]),
        thumbs=np.stack([s["thumb"] for s in scans]),
        meta=json.dumps({s["uid"]: {k: v for k, v in s.items() if k not in ("uid", "thumb")} for s in scans}),
    )
    return {s["uid"]: s for s in scans}


# --------------------------------------------------------------------------- processing steps

def crop_box_for(rec: dict, scan: dict, use_crop: bool) -> tuple[list[int], str]:
    """Scope-view crop, widened when it would cut away a real part of an annotation.

    Nothing anatomical lies outside the scope view, so boxes that only stick out a little
    (typically the loose enclosing boxes of rotated rectangles) are clipped to the view.
    If a box would lose more than MAX_BOX_LOSS of its area, the view detection is not
    trusted and the crop is widened to include the box.
    """
    W, H = rec["width"], rec["height"]
    full = [0, 0, W, H]
    if not use_crop:
        return full, "disabled"
    if scan["scope_box"] is None:
        return full, "no_view_found"
    sx0, sy0, sx1, sy1 = x0, y0, x1, y1 = scan["scope_box"]
    status = "cropped"
    for b in rec["boxes"]:
        bx0, by0, bx1, by1 = b["xyxy"]
        bx0, by0, bx1, by1 = max(bx0, 0), max(by0, 0), min(bx1, W), min(by1, H)
        area = max(bx1 - bx0, 0) * max(by1 - by0, 0)
        inside = max(min(bx1, sx1) - max(bx0, sx0), 0) * max(min(by1, sy1) - max(by0, sy0), 0)
        if area and inside / area < 1 - MAX_BOX_LOSS:
            status = "widened_for_annotation"
            x0, y0 = min(x0, math.floor(bx0)), min(y0, math.floor(by0))
            x1, y1 = max(x1, math.ceil(bx1)), max(y1, math.ceil(by1))
    if [x0, y0, x1, y1] == full:
        status = "full_frame"
    return [x0, y0, x1, y1], status


def mapped_classes(rec: dict) -> frozenset:
    return frozenset(CLASS_MAP[b["orig_class"]] for b in rec["boxes"] if CLASS_MAP.get(b["orig_class"]))


def find_near_duplicates(records: list[dict], scans: dict, threshold: float) -> set[str]:
    """Consecutive frames of a video that are almost identical to the last kept frame.

    Similarity = mean absolute difference of 32x32 grayscale thumbnails (0-255 scale).
    A frame is only removed if it also has the same set of classes as the kept frame.
    """
    removed = set()
    if threshold <= 0:
        return removed
    by_group = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)
    for recs in by_group.values():
        recs.sort(key=lambda r: r["frame"])
        kept = recs[0]
        for r in recs[1:]:
            diff = np.abs(scans[r["uid"]]["thumb"].astype(np.float32) - scans[kept["uid"]]["thumb"]).mean()
            if diff < threshold and mapped_classes(r) == mapped_classes(kept):
                removed.add(r["uid"])
            else:
                kept = r
    return removed


def assign_groups(records: list[dict], ratios: tuple[float, ...], seed: int, trials: int = 200_000) -> dict[str, str]:
    """Assign whole videos (groups) to train/val/test.

    Frames inside a video are near-identical, so splitting frames randomly would leak the
    same scene into train and test. Groups are assigned per subset (clinical, phantom) so
    every split contains both subsets, choosing the assignment whose per-class and
    per-image shares are closest to the target ratios (random search, seeded).
    """
    rng = np.random.default_rng(seed)
    target = np.array(ratios, dtype=np.float64)
    assignment = {}
    for domain in SOURCES:
        groups = sorted({r["group"] for r in records if r["domain"] == domain})
        G, K = len(groups), len(CLASS_NAMES)
        counts = np.zeros((G, K + 1))  # per-class instances + image count
        gi = {g: i for i, g in enumerate(groups)}
        for r in records:
            if r["domain"] != domain:
                continue
            counts[gi[r["group"]], K] += 1
            for b in r["boxes"]:
                if CLASS_MAP.get(b["orig_class"]):
                    counts[gi[r["group"]], CLASS_ID[CLASS_MAP[b["orig_class"]]]] += 1
        present = counts.sum(0) > 0
        weights = np.where(present, 1.0, 0.0)
        weights[K] = 3.0  # image share matters most
        best, best_loss = None, np.inf
        for _ in range(0, trials, 20_000):
            A = rng.integers(0, 3, size=(20_000, G))
            onehot = (A[:, :, None] == np.arange(3)[None, None, :]).astype(np.float64)  # n,G,3
            split_counts = np.einsum("ngs,gk->nsk", onehot, counts)                      # n,3,K+1
            totals = split_counts.sum(1, keepdims=True)
            share = np.divide(split_counts, totals, out=np.zeros_like(split_counts), where=totals > 0)
            loss = (np.abs(share - target[None, :, None]).sum(1) * weights).sum(1)
            # hard constraints: every split has a group, every present class appears in train
            loss[(onehot.sum(1) == 0).any(1)] = np.inf
            loss[((split_counts[:, 0, :K] == 0) & present[None, :K]).any(1)] = np.inf
            # soft: classes missing from val/test cannot be evaluated
            loss += 0.5 * ((split_counts[:, 1:, :K] == 0) & present[None, None, :K]).sum((1, 2))
            i = int(np.argmin(loss))
            if loss[i] < best_loss:
                best, best_loss = A[i], loss[i]
        if best is None or not np.isfinite(best_loss):
            raise RuntimeError(f"no valid split found for {domain}")
        for g, s in zip(groups, best):
            assignment[g] = SPLITS[s]
    return assignment


# --------------------------------------------------------------------------- reporting helpers

def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_barh(path: Path, title: str, categories: list[str], series: dict[str, list[float]], xlabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(series)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(categories) * max(1, n * 0.6) + 1.2), facecolor=SURFACE)
    _style_axes(ax)
    height = 0.8 / n
    y = np.arange(len(categories))
    for i, (name, vals) in enumerate(series.items()):
        pos = y - 0.4 + height * (i + 0.5)
        ax.barh(pos, vals, height=height * 0.85, color=SERIES[i % len(SERIES)], label=name)
        for p, v in zip(pos, vals):
            if v:
                ax.text(v, p, f" {int(v)}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, categories)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    if n > 1:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def size_histogram(path: Path, title: str, series: dict[str, list[float]], xlabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style_axes(ax)
    allv = [v for vals in series.values() for v in vals if v > 0]
    bins = np.logspace(np.log10(min(allv)), np.log10(max(allv)), 40)
    for i, (name, vals) in enumerate(series.items()):
        ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=SERIES[i], label=name)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel("boxes", color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def pct(values, qs=(10, 50, 90)) -> str:
    if not len(values):
        return "-"
    return " / ".join(f"{v:.3g}" for v in np.percentile(values, qs))


# --------------------------------------------------------------------------- stages

def stage_analyze(args) -> None:
    print("== analyze")
    records = load_records(args.raw_dir)
    scans = get_scans(records, args)
    rep_dir = args.out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    stats: dict = {"images": {}, "annotations": {}, "per_class": {}, "groups": {}}
    md = ["# UAAL raw dataset analysis", "",
          f"Generated by `prepare_data.py analyze` from `{args.raw_dir}`.", ""]

    # images / annotations per subset and official split
    rows = []
    for d in SOURCES:
        for s in OFFICIAL_SPLITS:
            recs = [r for r in records if r["domain"] == d and r["official_split"] == s]
            n_ann = sum(len(r["boxes"]) for r in recs)
            stats["images"][f"{d}/{s}"] = len(recs)
            stats["annotations"][f"{d}/{s}"] = n_ann
            sizes = Counter(f"{r['width']}x{r['height']}" for r in recs)
            rows.append([d, s, len(recs), n_ann, ", ".join(f"{k} ({v})" for k, v in sizes.most_common())])
    md += ["## Images and annotations", "", md_table(["Subset", "Official split", "Images", "Annotations", "Sizes"], rows), ""]

    # per original class: counts, annotation kind, out-of-bounds
    per_class = defaultdict(Counter)
    for r in records:
        for b in r["boxes"]:
            c = per_class[b["orig_class"]]
            c[f"{r['domain']}/{r['official_split']}"] += 1
            c[b["kind"]] += 1
            c["out_of_bounds"] += b["out_of_bounds"]
    stats["per_class"] = {k: dict(v) for k, v in per_class.items()}
    cols = [f"{d}/{s}" for d in SOURCES for s in OFFICIAL_SPLITS]
    rows = [[k, CLASS_MAP.get(k) or "(dropped)"] + [v[c] for c in cols] + [v["polygon"], v["rotated"], v["box"], v["out_of_bounds"]]
            for k, v in sorted(per_class.items(), key=lambda kv: -sum(kv[1][c] for c in cols))]
    md += ["## Classes (original names)", "",
           "`polygon` = has mask (box taken from polygon), `rotated` = CVAT rotated rectangle "
           "(converted to enclosing box), `box` = plain box, `oob` = box extends outside the image (clipped).", "",
           md_table(["Class", "Maps to"] + cols + ["polygon", "rotated", "box", "oob"], rows), ""]
    grouped_barh(rep_dir / "raw_class_counts.png", "UAAL annotations per class (raw)",
                 sorted(per_class), {d: [sum(per_class[k][f"{d}/{s}"] for s in OFFICIAL_SPLITS) for k in sorted(per_class)]
                                     for d in SOURCES}, "annotations")

    # groups (= videos)
    groups = defaultdict(lambda: {"images": 0, "official_split": set(), "classes": Counter()})
    for r in records:
        g = groups[r["group"]]
        g["images"] += 1
        g["official_split"].add(r["official_split"])
        for b in r["boxes"]:
            g["classes"][b["orig_class"]] += 1
    stats["groups"] = {k: {"images": v["images"], "official_split": sorted(v["official_split"]),
                           "classes": dict(v["classes"])} for k, v in groups.items()}
    rows = [[k, v["images"], "/".join(sorted(v["official_split"])),
             ", ".join(f"{c} {n}" for c, n in v["classes"].most_common())] for k, v in sorted(groups.items())]
    leaking = [k for k, v in groups.items() if len(v["official_split"]) > 1]
    md += ["## Groups (one group = one continuous recording)", "",
           f"Groups in both official train and val: **{len(leaking)}** {leaking or ''}", "",
           md_table(["Group", "Images", "Official split", "Annotations"], rows), ""]

    # frame similarity -> near duplicates
    diffs = {d: [] for d in SOURCES}
    rand = {d: [] for d in SOURCES}
    rng = np.random.default_rng(0)
    by_group = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)
    for g, recs in by_group.items():
        th = [scans[r["uid"]]["thumb"].astype(np.float32) for r in recs]
        d = recs[0]["domain"]
        diffs[d] += [float(np.abs(a - b).mean()) for a, b in zip(th, th[1:])]
        for _ in range(min(50, len(th))):
            i, j = rng.choice(len(th), 2, replace=False)
            rand[d].append(float(np.abs(th[i] - th[j]).mean()))
    rows = []
    for d in SOURCES:
        rows.append([d, pct(diffs[d]), pct(rand[d])] +
                    [int((np.array(diffs[d]) < t).sum()) for t in (1, 2, 3, 5)])
    stats["frame_similarity"] = {d: {"consecutive_p10_50_90": pct(diffs[d]), "random_p10_50_90": pct(rand[d])} for d in SOURCES}
    md += ["## Frame similarity", "",
           "Mean absolute difference between 32x32 grayscale thumbnails (0-255). Consecutive frames are far "
           "more similar than random frames of the same recording, so a random frame-level split would leak.", "",
           md_table(["Subset", "Consecutive p10/50/90", "Random same-group p10/50/90",
                     "pairs < 1", "pairs < 2", "pairs < 3", "pairs < 5"], rows), ""]

    # scope crop behaviour
    crop_stats = defaultdict(Counter)
    area = defaultdict(list)
    for r in records:
        box, status = crop_box_for(r, scans[r["uid"]], True)
        crop_stats[r["domain"]][status] += 1
        area[r["domain"]].append((box[2] - box[0]) * (box[3] - box[1]) / (r["width"] * r["height"]))
    statuses = ["cropped", "widened_for_annotation", "full_frame", "no_view_found"]
    rows = [[d] + [crop_stats[d][s] for s in statuses] + [pct(area[d])] for d in SOURCES]
    stats["scope_crop"] = {d: dict(v) for d, v in crop_stats.items()}
    md += ["## Scope-view crop", "",
           "Crop = largest bright region of the frame (removes black borders, side text, icons and "
           "picture-in-picture thumbnails), widened when an annotation reaches outside it.", "",
           md_table(["Subset"] + statuses + ["kept area fraction p10/50/90"], rows), ""]

    # relative box size
    rel = {d: [] for d in SOURCES}
    for r in records:
        for b in r["boxes"]:
            x0, y0, x1, y1 = b["xyxy"]
            rel[r["domain"]].append(max(x1 - x0, 1e-6) * max(y1 - y0, 1e-6) / (r["width"] * r["height"]))
    size_histogram(rep_dir / "raw_box_sizes.png", "Box area as a fraction of the frame (raw)", rel, "box area / image area")
    md += ["## Charts", "", "![classes](raw_class_counts.png)", "", "![sizes](raw_box_sizes.png)", ""]

    (rep_dir / "raw_stats.json").write_text(json.dumps(stats, indent=2, default=list))
    (rep_dir / "raw_analysis.md").write_text("\n".join(md))
    print(f"  wrote {rep_dir / 'raw_analysis.md'}")


def stage_prepare(args) -> None:
    print("== prepare")
    records = load_records(args.raw_dir)
    scans = get_scans(records, args)

    # 1. near-duplicate removal (before splitting, within each recording)
    removed = find_near_duplicates(records, scans, args.dedup_threshold)
    kept = [r for r in records if r["uid"] not in removed]
    print(f"  near-duplicates removed: {len(removed)} of {len(records)}")

    # 2. split
    if args.split_strategy == "official":
        split_of = {r["uid"]: r["official_split"] for r in kept}
        group_split = {r["group"]: r["official_split"] for r in kept}
    else:
        group_split = assign_groups(kept, tuple(args.ratios), args.seed)
        split_of = {r["uid"]: group_split[r["group"]] for r in kept}

    # 3. crop boxes
    crops = {r["uid"]: crop_box_for(r, scans[r["uid"]], not args.no_crop) for r in kept}

    # 4. write images + labels (old output is replaced)
    for sub in ("images", "labels", "lists", "previews"):
        shutil.rmtree(args.out_dir / sub, ignore_errors=True)
    for s in SPLITS:
        (args.out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
    jobs = [{"rec": r, "crop": crops[r["uid"]][0], "gray": not args.color, "max_side": args.max_side,
             "out_img": str(args.out_dir / "images" / split_of[r["uid"]] / f"{r['uid']}.png")} for r in kept]
    results = {res["uid"]: res for res in run_parallel(write_image, jobs, args.raw_dir, args.workers, "write")}
    for r in kept:
        res = results[r["uid"]]
        lbl = args.out_dir / "labels" / split_of[r["uid"]] / f"{r['uid']}.txt"
        lbl.write_text("\n".join(res["lines"]) + ("\n" if res["lines"] else ""))

    # 5. per-subset image lists + dataset yamls
    (args.out_dir / "lists").mkdir(exist_ok=True)
    present_splits = [s for s in SPLITS if any(split_of[r["uid"]] == s for r in kept)]
    names = {i: n for i, n in enumerate(CLASS_NAMES)}
    header = ("# UAAL upper-airway detection dataset (generated by prepare_data.py).\n"
              "# IMPORTANT: train with fliplr=0.0 - a horizontal flip turns a left vocal fold into a right one.\n"
              "# Images are grayscale and cropped to the scope view: apply preprocessing.preprocess_frame() at inference.\n")
    base = {f: f"images/{f}" for f in present_splits}
    (args.out_dir / "data.yaml").write_text(header + yaml.safe_dump({**base, "nc": len(CLASS_NAMES), "names": names}, sort_keys=False))
    for d in SOURCES:
        entry = {}
        for s in present_splits:
            files = sorted(f"./../images/{s}/{r['uid']}.png" for r in kept if r["domain"] == d and split_of[r["uid"]] == s)
            (args.out_dir / "lists" / f"{s}_{d}.txt").write_text("\n".join(files) + "\n")
            entry[s] = f"lists/{s}_{d}.txt"
        (args.out_dir / f"data_{d}.yaml").write_text(
            f"# Evaluation-only view of the {d} subset (use with model.val(data=..., split='val'|'test')).\n"
            + yaml.safe_dump({**entry, "nc": len(CLASS_NAMES), "names": names}, sort_keys=False))

    # 6. manifest + dropped log
    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "split", "domain", "group", "frame", "official_split", "source", "orig_w", "orig_h",
                    "crop_x0", "crop_y0", "crop_x1", "crop_y1", "crop_status", "out_w", "out_h", "n_boxes", "classes"])
        for r in kept:
            res, (crop, status) = results[r["uid"]], crops[r["uid"]]
            classes = sorted({CLASS_NAMES[int(l.split()[0])] for l in res["lines"]})
            w.writerow([r["uid"], split_of[r["uid"]], r["domain"], r["group"], r["frame"], r["official_split"],
                        f"{r['zip']}:{r['member']}", r["width"], r["height"], *crop, status, *res["out_size"],
                        len(res["lines"]), ";".join(classes)])
    with open(args.out_dir / "dropped.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "what", "class", "reason"])
        for u in sorted(removed):
            w.writerow([u, "image", "", f"near-duplicate (thumb diff < {args.dedup_threshold})"])
        for r in kept:
            for cls, reason in results[r["uid"]]["dropped"]:
                w.writerow([r["uid"], "box", cls, reason])

    # 7. report
    write_prepare_report(args, records, kept, removed, split_of, group_split, crops, results)
    (args.out_dir / "config_used.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "raw_dir": str(args.raw_dir), "seed": args.seed,
        "split_strategy": args.split_strategy, "ratios": args.ratios, "max_side": args.max_side,
        "grayscale": not args.color, "scope_crop": not args.no_crop, "dedup_threshold": args.dedup_threshold,
        "class_map": CLASS_MAP, "class_names": CLASS_NAMES, "group_split": group_split,
    }, indent=2))
    print(f"  wrote dataset to {args.out_dir}")


def write_prepare_report(args, records, kept, removed, split_of, group_split, crops, results) -> None:
    rep_dir = args.out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    present = [s for s in SPLITS if any(split_of[r["uid"]] == s for r in kept)]
    md = ["# UAAL prepared dataset report", "",
          f"Split strategy: **{args.split_strategy}** (seed {args.seed}, target ratios {args.ratios}), "
          f"grayscale: **{not args.color}**, scope crop: **{not args.no_crop}**, max side: **{args.max_side}px**, "
          f"near-duplicate threshold: **{args.dedup_threshold}**.", ""]

    rows = []
    for s in present:
        row = [s]
        for d in SOURCES:
            row.append(sum(1 for r in kept if r["domain"] == d and split_of[r["uid"]] == s))
        row.append(sum(1 for r in kept if split_of[r["uid"]] == s))
        rows.append(row)
    total = len(kept)
    rows = [row + [f"{row[-1] / total:.1%}"] for row in rows]
    md += ["## Images per split", "", md_table(["Split"] + list(SOURCES) + ["Total", "Share"], rows), ""]

    inst = {s: Counter() for s in present}
    inst_dom = {(s, d): Counter() for s in present for d in SOURCES}
    for r in kept:
        s = split_of[r["uid"]]
        for line in results[r["uid"]]["lines"]:
            c = CLASS_NAMES[int(line.split()[0])]
            inst[s][c] += 1
            inst_dom[(s, r["domain"])][c] += 1
    rows = [[c] + [f"{inst[s][c]} ({inst_dom[(s, 'clinical')][c]}/{inst_dom[(s, 'phantom')][c]})" for s in present]
            + [sum(inst[s][c] for s in present)] for c in CLASS_NAMES]
    md += ["## Box instances per class and split", "", "Format: total (clinical/phantom).", "",
           md_table(["Class"] + present + ["Total"], rows), ""]
    grouped_barh(rep_dir / "prepared_class_counts.png", "Box instances per class and split",
                 CLASS_NAMES, {s: [inst[s][c] for c in CLASS_NAMES] for s in present}, "boxes")

    rows = [[s, d, ", ".join(sorted(g.split("/")[1] for g, gs in group_split.items() if gs == s and g.startswith(d + "/")))]
            for s in present for d in SOURCES]
    md += ["## Recordings (groups) per split", "",
           "Whole recordings are kept together so near-identical consecutive frames never end up in different splits.", "",
           md_table(["Split", "Subset", "Groups"], rows), ""]

    status = Counter(st for _, st in crops.values())
    drops = Counter(reason for r in kept for _, reason in results[r["uid"]]["dropped"])
    md += ["## Processing summary", "",
           f"- Raw images: {len(records)}; near-duplicates removed: {len(removed)}; written: {len(kept)}",
           f"- Scope crop: " + ", ".join(f"{k} {v}" for k, v in status.most_common()),
           f"- Boxes written: {sum(len(r['lines']) for r in results.values())}; boxes dropped: "
           + (", ".join(f"{k} {v}" for k, v in drops.items()) or "0"),
           f"- Output sizes: " + ", ".join(f"{k} ({v})" for k, v in Counter(
               f"{res['out_size'][0]}x{res['out_size'][1]}" for res in results.values()).most_common(6)), ""]

    rel = {d: [] for d in SOURCES}
    for r in kept:
        W, H = results[r["uid"]]["out_size"]
        for line in results[r["uid"]]["lines"]:
            _, _, _, bw, bh = map(float, line.split())
            rel[r["domain"]].append(math.sqrt(bw * bh) * 640)
    size_histogram(rep_dir / "prepared_box_sizes.png", "Box size at 640px training input (sqrt of area, px)", rel, "pixels")
    md += ["## Box size at a 640px training input", "",
           "sqrt(box area) in pixels if the image is letterboxed to 640 (p10 / p50 / p90): " +
           "; ".join(f"{d} {pct(v)}" for d, v in rel.items()), "",
           "![classes](prepared_class_counts.png)", "", "![sizes](prepared_box_sizes.png)", "",
           "## Training notes", "",
           "- Set `fliplr=0.0` (left/right vocal folds). Vertical flips are also anatomically wrong: keep `flipud=0.0`.",
           "- Images are single-channel PNG; YOLO loaders read them as 3 identical channels. Use "
           "`preprocessing.preprocess_frame()` on live frames so inference matches training.",
           "- Report validation/test metrics separately for clinical and phantom with `data_clinical.yaml` / `data_phantom.yaml`.", ""]
    (rep_dir / "prepared_report.md").write_text("\n".join(md))


def stage_verify(args) -> None:
    print("== verify")
    out = args.out_dir
    errors, warnings = [], []
    manifest = list(csv.DictReader(open(out / "manifest.csv")))
    by_uid = {m["uid"]: m for m in manifest}

    # every image <-> label, label syntax, sizes
    hashes = defaultdict(list)
    n_boxes = 0
    for s in SPLITS:
        imgs = {p.stem: p for p in (out / "images" / s).glob("*.png")}
        lbls = {p.stem: p for p in (out / "labels" / s).glob("*.txt")}
        for u in imgs.keys() - lbls.keys():
            errors.append(f"{s}/{u}: image without label file")
        for u in lbls.keys() - imgs.keys():
            errors.append(f"{s}/{u}: label without image")
        for u, p in imgs.items():
            m = by_uid.get(u)
            if m is None or m["split"] != s:
                errors.append(f"{s}/{u}: not in manifest for this split")
                continue
            with Image.open(p) as im:
                if [im.width, im.height] != [int(m["out_w"]), int(m["out_h"])]:
                    errors.append(f"{u}: size {im.size} differs from manifest")
                hashes[hashlib.md5(im.tobytes()).hexdigest()].append((s, u))
            lines = [l for l in lbls.get(u, p).read_text().splitlines() if l.strip()] if u in lbls else []
            if not lines:
                warnings.append(f"{u}: no boxes (background image)")
            for l in lines:
                parts = l.split()
                n_boxes += 1
                if len(parts) != 5:
                    errors.append(f"{u}: bad label line '{l}'")
                    continue
                c, *v = parts
                if not c.isdigit() or int(c) >= len(CLASS_NAMES):
                    errors.append(f"{u}: invalid class {c}")
                x, y, w, h = map(float, v)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors.append(f"{u}: coordinates out of range '{l}'")
                if x - w / 2 < -1e-4 or x + w / 2 > 1 + 1e-4 or y - h / 2 < -1e-4 or y + h / 2 > 1 + 1e-4:
                    errors.append(f"{u}: box extends outside image '{l}'")

    # leakage: a recording in more than one split, identical images across splits
    group_splits = defaultdict(set)
    for m in manifest:
        group_splits[m["group"]].add(m["split"])
    for g, ss in group_splits.items():
        if len(ss) > 1:
            errors.append(f"group {g} appears in splits {sorted(ss)}")
    for h, items in hashes.items():
        if len({s for s, _ in items}) > 1:
            errors.append(f"identical image in several splits: {items}")
        elif len(items) > 1:
            warnings.append(f"identical images within {items[0][0]}: {[u for _, u in items]}")

    # every class present in train, and each subset present in every split
    train_classes = Counter()
    for p in (out / "labels" / "train").glob("*.txt"):
        for l in p.read_text().splitlines():
            if l.strip():
                train_classes[CLASS_NAMES[int(l.split()[0])]] += 1
    for c in CLASS_NAMES:
        if not train_classes[c]:
            errors.append(f"class {c} has no training boxes")
    for s in {m["split"] for m in manifest}:
        for d in SOURCES:
            if not any(m["split"] == s and m["domain"] == d for m in manifest):
                warnings.append(f"split {s} has no {d} images")

    write_previews(args, manifest)
    print(f"  images: {len(manifest)}, boxes: {n_boxes}, warnings: {len(warnings)}, errors: {len(errors)}")
    for w in warnings[:10]:
        print("  WARN ", w)
    for e in errors[:30]:
        print("  ERROR", e)
    (out / "reports" / "verify.txt").write_text(
        f"images {len(manifest)}\nboxes {n_boxes}\n\nERRORS ({len(errors)})\n" + "\n".join(errors)
        + f"\n\nWARNINGS ({len(warnings)})\n" + "\n".join(warnings) + "\n")
    print("  RESULT:", "PASS" if not errors else "FAIL")
    if errors:
        sys.exit(1)


def write_previews(args, manifest: list[dict]) -> None:
    """Draw the boxes on a few random images per split and subset, for a visual check."""
    rng = np.random.default_rng(args.seed)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948", "#4a3aa7"]
    for s in SPLITS:
        for d in SOURCES:
            rows = [m for m in manifest if m["split"] == s and m["domain"] == d]
            if not rows:
                continue
            dst = args.out_dir / "previews" / s
            dst.mkdir(parents=True, exist_ok=True)
            for m in rng.choice(rows, min(args.previews, len(rows)), replace=False):
                im = Image.open(args.out_dir / "images" / s / f"{m['uid']}.png").convert("RGB")
                draw = ImageDraw.Draw(im)
                lw = max(2, im.width // 300)
                for l in (args.out_dir / "labels" / s / f"{m['uid']}.txt").read_text().splitlines():
                    c, x, y, w, h = l.split()
                    c, x, y, w, h = int(c), float(x) * im.width, float(y) * im.height, float(w) * im.width, float(h) * im.height
                    draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline=colors[c], width=lw)
                    draw.text((x - w / 2 + 3, y - h / 2 + 2), CLASS_NAMES[c], fill=colors[c])
                im.save(dst / f"{m['uid']}.jpg", quality=85)


# --------------------------------------------------------------------------- CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="all", choices=["analyze", "prepare", "verify", "all"])
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="folder with the UAAL zips")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR, help="output dataset folder")
    p.add_argument("--split-strategy", choices=["grouped", "official"], default="grouped",
                   help="grouped: new train/val/test by recording (default); official: UAAL train/val, no test")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-side", type=int, default=1024, help="longest image side after crop (no upscaling)")
    p.add_argument("--dedup-threshold", type=float, default=2.0,
                   help="drop consecutive frames whose thumbnail difference is below this (0 = keep all)")
    p.add_argument("--color", action="store_true", help="keep RGB instead of converting to grayscale")
    p.add_argument("--no-crop", action="store_true", help="do not crop to the scope view")
    p.add_argument("--previews", type=int, default=12, help="preview images per split and subset")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    if abs(sum(args.ratios) - 1) > 1e-6:
        p.error("--ratios must sum to 1")
    args.out_dir = args.out_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if args.stage in ("analyze", "all"):
        stage_analyze(args)
    if args.stage in ("prepare", "all"):
        stage_prepare(args)
    if args.stage in ("verify", "all"):
        stage_verify(args)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

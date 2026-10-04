#!/usr/bin/env python3
"""BM-BronchoLC dataset preparation: analyze -> clean -> split -> export -> verify.

Reads the original dataset folders (nothing in dataset/ is modified) and writes a YOLO
dataset with polygon labels (usable for both detection and segmentation training) plus
COCO json files that keep every class for later phases.

Stages
  analyze   statistics of the raw data (patients, classes, polygons, sizes, duplicates,
            scope-view and text-removal behaviour) -> <out>/reports/raw_*
  prepare   text removal + view crop, grayscale, renaming, near-duplicate removal,
            patient-level split, polygon conversion, export -> <out>/images, labels, ...
  verify    integrity and leakage checks + preview images
  all       analyze + prepare + verify (default)

Tasks (which classes go into the YOLO labels; the split is the same for every task)
  anatomy        11 airway landmarks (default; images with only lesion labels are excluded)
  lesion         7 lesion classes, all images (images without lesions are negatives)
  lesion_binary  lesions merged into one "abnormal" class, all images
  all            all 18 classes

Usage
  python prepare_data.py
  python prepare_data.py prepare --task lesion_binary
  python prepare_data.py verify --previews 20
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from preprocessing import clean_frame, resize_to_max_side

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "dataset"
COHORTS = {"lc": "Lung_cancer", "nlc": "Non_lung_cancer"}   # short name -> folder
SPLITS = ("train", "val", "test")

# Original label -> clean class name. Two original names are misspelled ("Muscosal", "Anthrocosis").
ANATOMY = {
    "Vocal cords": "vocal_cords",
    "Trachea": "trachea",
    "Main carina": "main_carina",
    "Right main bronchus": "right_main_bronchus",
    "Left main bronchus": "left_main_bronchus",
    "Intermediate bronchus": "intermediate_bronchus",
    "Right superior lobar bronchus": "right_superior_lobar_bronchus",
    "Right middle lobar bronchus": "right_middle_lobar_bronchus",
    "Right inferior lobar bronchus": "right_inferior_lobar_bronchus",
    "Left superior lobar bronchus": "left_superior_lobar_bronchus",
    "Left inferior lobar bronchus": "left_inferior_lobar_bronchus",
}
LESION = {
    "Tumor": "tumor",
    "Mucosal infiltration": "mucosal_infiltration",
    "Mucosal edema of carina": "mucosal_edema_of_carina",
    "Stenosis": "stenosis",
    "Vascular growth": "vascular_growth",
    "Muscosal erythema": "mucosal_erythema",
    "Anthrocosis": "anthracosis",
}
ALL_CLASSES = {**ANATOMY, **LESION}
TASKS = {
    "anatomy": list(ANATOMY.values()),
    "lesion": list(LESION.values()),
    "lesion_binary": ["abnormal"],
    "all": list(ANATOMY.values()) + list(LESION.values()),
}

THUMB = 32                # thumbnail side for similarity checks
CROSS_PATIENT_DUP = 1.0   # thumbnails closer than this in different patients = same recording
MIN_POLY_AREA = 4.0       # px^2 in the output image; smaller polygons are dropped
SIMPLIFY_EPS = 0.75       # px; polygon simplification tolerance in the output image
MIN_TEXTURE = 3.0         # gray-level std in the view centre; real tissue frames are > 14
SCAN_VERSION = 4
DUPLICATE_POLYGONS: Counter = Counter()  # filled by load_records

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


# --------------------------------------------------------------------------- loading

def load_records(raw_dir: Path) -> list[dict]:
    """One record per image, named bm_<cohort>_p<patient>_f<frame> (frame = order in objects.json)."""
    records = []
    for cohort, folder in COHORTS.items():
        base = raw_dir / folder / folder
        label_name = {l["id"]: l["name"] for l in json.load(open(base / "labels.json"))}
        unknown = set(label_name.values()) - set(ALL_CLASSES)
        if unknown:
            raise ValueError(f"unknown labels in {folder}: {unknown}")
        polys, seen = defaultdict(list), set()
        for a in json.load(open(base / "annotation.json")):
            labels = [label_name[l] for l in a["label_ids"]]
            points = [(p["x"], p["y"]) for p in a["data"]]
            # ~14% of polygons are exact copies of another polygon on the same image: keep one
            key = (a["object_id"], tuple(sorted(labels)), tuple((round(x, 1), round(y, 1)) for x, y in points))
            if key in seen:
                DUPLICATE_POLYGONS[folder] += 1
                continue
            seen.add(key)
            polys[a["object_id"]].append({"id": a["finding_id"], "labels": labels, "points": points})
        patients = [p for p in json.load(open(base / "objects.json"))
                    if any(v["images"] for v in p["videos"])]
        for pidx, patient in enumerate(sorted(patients, key=lambda p: p["id"]), 1):
            frame = 0
            for video in patient["videos"]:
                for im in video["images"]:
                    frame += 1
                    path = base / "imgs" / patient["id"] / video["video_id"] / f"{im['image_id']}.png"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    with Image.open(path) as f:
                        w, h = f.size
                    records.append({
                        "uid": f"bm_{cohort}_p{pidx:03d}_f{frame:02d}",
                        "cohort": cohort, "patient": f"{cohort}_p{pidx:03d}", "patient_id": patient["id"],
                        "video_id": video["video_id"], "image_id": im["image_id"],
                        "path": str(path), "width": w, "height": h, "polygons": polys.get(im["image_id"], []),
                    })
    return records


# --------------------------------------------------------------------------- geometry

def clip_polygon(points: list[tuple[float, float]], box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clipping of a polygon to an axis-aligned box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    edges = [(lambda p: p[0] >= x0, lambda p, q: _cut_x(p, q, x0)),
             (lambda p: p[0] <= x1, lambda p, q: _cut_x(p, q, x1)),
             (lambda p: p[1] >= y0, lambda p, q: _cut_y(p, q, y0)),
             (lambda p: p[1] <= y1, lambda p, q: _cut_y(p, q, y1))]
    out = list(points)
    for inside, cut in edges:
        if not out:
            break
        src, out = out, []
        for i, cur in enumerate(src):
            prev = src[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cut(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cut(prev, cur))
    return out


def _cut_x(p, q, x):
    t = (x - p[0]) / (q[0] - p[0])
    return x, p[1] + t * (q[1] - p[1])


def _cut_y(p, q, y):
    t = (y - p[1]) / (q[1] - p[1])
    return p[0] + t * (q[0] - p[0]), y


def simplify(points: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of a closed polygon (iterative)."""
    if len(points) <= 4:
        return points
    # split the ring at the two most distant points so both halves are open polylines
    start = 0
    end = int(np.argmax(((points - points[start]) ** 2).sum(1)))
    keep = np.zeros(len(points), bool)
    keep[[start, end]] = True
    ring = np.vstack([points, points[:1]])
    for a, b in ((start, end), (end, len(points))):
        stack = [(a, b)]
        while stack:
            i, j = stack.pop()
            if j - i < 2:
                continue
            seg = ring[i + 1:j]
            p, q = ring[i], ring[j]
            d = q - p
            norm = math.hypot(*d)
            dist = (np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / norm
                    if norm > 0 else np.hypot(*(seg - p).T))
            k = int(np.argmax(dist))
            if dist[k] > eps:
                keep[(i + 1 + k) % len(points)] = True
                stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return points[keep]


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --------------------------------------------------------------------------- image workers

def scan_image(rec: dict) -> dict:
    """Raw pixel hash, raw thumbnail, and the cleaning result (crop box, view/text fractions)."""
    img = Image.open(rec["path"])
    img.load()
    raw = np.asarray(img.convert("RGB"))
    gray, box, info = clean_frame(img)
    thumb = np.asarray(Image.fromarray(gray).resize((THUMB, THUMB), Image.BILINEAR), dtype=np.uint8)
    h, w = gray.shape
    texture = float(gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4].std())  # ~0 for a blank/solid frame
    return {"uid": rec["uid"], "md5": hashlib.md5(raw.tobytes()).hexdigest(), "box": list(box),
            "mode": img.mode, **info, "texture": texture, "thumb": thumb}


def write_image(job: dict) -> dict:
    """Clean, resize and save one image; convert its polygons to output pixel coordinates."""
    rec, max_side = job["rec"], job["max_side"]
    img = Image.open(rec["path"])
    gray, box, _ = clean_frame(img, remove_text=job["remove_text"])
    out, _ = resize_to_max_side(Image.fromarray(gray), max_side)
    out.save(job["out_img"], compress_level=3)
    W, H = out.size
    sx, sy = W / (box[2] - box[0]), H / (box[3] - box[1])
    polys, dropped = [], []
    for p in rec["polygons"]:
        pts = clip_polygon([(min(max(x, 0), rec["width"]), min(max(y, 0), rec["height"])) for x, y in p["points"]], box)
        if len(pts) < 3:
            dropped.append((p["labels"], "outside scope view"))
            continue
        arr = np.array([((x - box[0]) * sx, (y - box[1]) * sy) for x, y in pts])
        arr = simplify(arr, SIMPLIFY_EPS)
        arr[:, 0] = arr[:, 0].clip(0, W)
        arr[:, 1] = arr[:, 1].clip(0, H)
        bw, bh = np.ptp(arr[:, 0]), np.ptp(arr[:, 1])
        if len(arr) < 3 or polygon_area(arr) < MIN_POLY_AREA or bw < 2 or bh < 2:
            dropped.append((p["labels"], "degenerate after clipping/simplification"))
            continue
        polys.append({"labels": p["labels"], "points": arr.round(2).tolist()})
    return {"uid": rec["uid"], "out_size": [W, H], "polygons": polys, "dropped": dropped}


def run_parallel(fn, items, workers: int, label: str) -> list:
    t0, out = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(fn, items, chunksize=8), 1):
            out.append(r)
            if i % 500 == 0 or i == len(items):
                print(f"  {label}: {i}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def get_scans(records: list[dict], args) -> dict[str, dict]:
    cache = args.cache_dir / "scan.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if int(data["version"]) == SCAN_VERSION and set(data["uids"]) == {r["uid"] for r in records}:
            meta = json.loads(str(data["meta"]))
            return {u: {**meta[u], "thumb": t} for u, t in zip(data["uids"], data["thumbs"])}
    print("Scanning images (view detection, text removal, thumbnails)...")
    scans = run_parallel(scan_image, records, args.workers, "scan")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, version=SCAN_VERSION, uids=np.array([s["uid"] for s in scans]),
                        thumbs=np.stack([s["thumb"] for s in scans]),
                        meta=json.dumps({s["uid"]: {k: v for k, v in s.items() if k not in ("uid", "thumb")} for s in scans}))
    return {s["uid"]: s for s in scans}


# --------------------------------------------------------------------------- duplicates / grouping / split

def thumb_matrix(records, scans) -> np.ndarray:
    return np.stack([scans[r["uid"]]["thumb"].reshape(-1).astype(np.float32) for r in records])


def cross_patient_pairs(records, scans, threshold: float) -> list[tuple[str, str, float]]:
    """Near-identical images that belong to different patients (same recording uploaded twice)."""
    X = thumb_matrix(records, scans)
    patient = np.array([r["patient"] for r in records])
    pairs = []
    for i0 in range(0, len(X), 16):  # chunked: a full pairwise tensor would not fit in memory
        d = np.abs(X[i0:i0 + 16, None, :] - X[None, :, :]).mean(2)
        for a, b in zip(*np.nonzero(d < threshold)):
            a += i0
            if a < b and patient[a] != patient[b]:
                pairs.append((records[a]["uid"], records[b]["uid"], float(d[a - i0, b])))
    return pairs


def split_groups(records, pairs) -> dict[str, str]:
    """Patients linked by near-identical images form one split group (union-find)."""
    parent = {r["patient"]: r["patient"] for r in records}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pat = {r["uid"]: r["patient"] for r in records}
    for a, b, _ in pairs:
        ra, rb = find(pat[a]), find(pat[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {p: find(p) for p in parent}


def class_set(rec) -> frozenset:
    return frozenset(l for p in rec["polygons"] for l in p["labels"])


def near_duplicates(records, scans, threshold: float) -> set[str]:
    """Within a patient, drop an image nearly identical to an earlier kept one with the same classes.

    Frame order in BM-BronchoLC is not guaranteed to be temporal, so all pairs are compared.
    """
    removed = set()
    if threshold <= 0:
        return removed
    by_patient = defaultdict(list)
    for r in records:
        by_patient[r["patient"]].append(r)
    for recs in by_patient.values():
        kept = []
        for r in recs:
            t = scans[r["uid"]]["thumb"].astype(np.float32)
            if any(np.abs(t - scans[k["uid"]]["thumb"]).mean() < threshold and class_set(k) == class_set(r) for k in kept):
                removed.add(r["uid"])
            else:
                kept.append(r)
    return removed


def assign_split(records, group_of: dict[str, str], ratios, seed: int, trials: int = 300_000) -> dict[str, str]:
    """Assign split groups (patients) to train/val/test, separately per cohort.

    Balances the image count and the share of every one of the 18 classes, so the same split
    serves the anatomy and the lesion tasks.
    """
    rng = np.random.default_rng(seed)
    target = np.array(ratios)
    names = list(ALL_CLASSES)
    K = len(names)
    out = {}
    for cohort in COHORTS:
        # a group belongs to the cohort of its root patient (groups linking cohorts are kept whole)
        recs = [r for r in records if group_of[r["patient"]].startswith(cohort + "_")]
        groups = sorted({group_of[r["patient"]] for r in recs})
        gi = {g: i for i, g in enumerate(groups)}
        # columns: 18 class instance counts, all images, images with anatomy labels
        counts = np.zeros((len(groups), K + 2))
        for r in recs:
            g = gi[group_of[r["patient"]]]
            counts[g, K] += 1
            counts[g, K + 1] += bool(class_set(r) & set(ANATOMY))
            for p in r["polygons"]:
                for l in p["labels"]:
                    counts[g, names.index(l)] += 1
        present = counts[:, :K].sum(0) > 0
        weights = np.r_[np.where(present, 1.0, 0.0), 4.0, 4.0]

        def losses(A):  # A: (n, groups) split index per group
            onehot = (A[:, :, None] == np.arange(3)).astype(np.float64)
            sc = np.einsum("ngs,gk->nsk", onehot, counts)
            tot = sc.sum(1, keepdims=True)
            share = np.divide(sc, tot, out=np.zeros_like(sc), where=tot > 0)
            loss = (np.abs(share - target[None, :, None]).sum(1) * weights).sum(1)
            loss[((sc[:, 0, :K] == 0) & present).any(1)] = np.inf      # every class in train
            loss += 0.5 * ((sc[:, 1:, :K] == 0) & present).sum((1, 2))  # prefer every class in val/test
            return loss

        best, best_loss = None, np.inf
        for _ in range(0, trials, 10_000):  # 1) random search
            A = rng.integers(0, 3, size=(10_000, len(groups)))
            loss = losses(A)
            i = int(np.argmin(loss))
            if loss[i] < best_loss:
                best, best_loss = A[i].copy(), loss[i]
        improved = True
        while improved:  # 2) local search: move single groups while the loss decreases
            cand = np.repeat(best[None], 2 * len(groups), 0)
            for g in range(len(groups)):
                cand[2 * g, g] = (best[g] + 1) % 3
                cand[2 * g + 1, g] = (best[g] + 2) % 3
            loss = losses(cand)
            i = int(np.argmin(loss))
            improved = loss[i] < best_loss - 1e-9
            if improved:
                best, best_loss = cand[i].copy(), loss[i]
        for g, s in zip(groups, best):
            out[g] = SPLITS[s]
    return out


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
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(categories) * max(1, n * 0.6) + 1.2), facecolor=SURFACE)
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


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def pct(values, qs=(10, 50, 90)) -> str:
    return " / ".join(f"{v:.3g}" for v in np.percentile(values, qs)) if len(values) else "-"


# --------------------------------------------------------------------------- stages

def stage_analyze(args) -> None:
    print("== analyze")
    records = load_records(args.raw_dir)
    scans = get_scans(records, args)
    rep = args.out_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    md = ["# BM-BronchoLC raw dataset analysis", "", f"Generated by `prepare_data.py analyze` from `{args.raw_dir}`.", ""]

    rows = []
    for c in COHORTS:
        recs = [r for r in records if r["cohort"] == c]
        per_pat = Counter(r["patient"] for r in recs)
        rows.append([c, len(per_pat), len(recs), sum(len(r["polygons"]) for r in recs),
                     pct(list(per_pat.values()), (0, 50, 100)),
                     ", ".join(f"{w}x{h} ({n})" for (w, h), n in Counter((r["width"], r["height"]) for r in recs).most_common())])
    md += ["## Patients and images", "", md_table(["Cohort", "Patients", "Images", "Polygons", "Images/patient min/med/max", "Sizes"], rows), ""]

    per_class = defaultdict(Counter)
    for r in records:
        seen = set()
        for p in r["polygons"]:
            for l in p["labels"]:
                per_class[l][f"{r['cohort']}_polygons"] += 1
                if l not in seen:
                    per_class[l][f"{r['cohort']}_images"] += 1
                    seen.add(l)
    pats = defaultdict(set)
    for r in records:
        for l in class_set(r):
            pats[l].add(r["patient"])
    rows = [[ALL_CLASSES[k], "anatomy" if k in ANATOMY else "lesion", v["lc_polygons"], v["nlc_polygons"],
             v["lc_images"] + v["nlc_images"], len(pats[k])] for k, v in per_class.items()]
    rows.sort(key=lambda r: (r[1], -(r[2] + r[3])))
    md += ["## Classes", "", md_table(["Class", "Type", "Polygons LC", "Polygons NLC", "Images", "Patients"], rows), ""]
    grouped_barh(rep / "raw_class_counts.png", "Polygons per class (raw)", [r[0] for r in rows],
                 {"lung cancer": [r[2] for r in rows], "non lung cancer": [r[3] for r in rows]}, "polygons")

    multi = Counter(tuple(p["labels"]) for r in records for p in r["polygons"] if len(p["labels"]) > 1)
    npts = [len(p["points"]) for r in records for p in r["polygons"]]
    small = sum(1 for r in records for p in r["polygons"] if len(p["points"]) < 3)
    outside = sum(1 for r in records for p in r["polygons"]
                  if any(x < 0 or y < 0 or x > r["width"] or y > r["height"] for x, y in p["points"]))
    cats = Counter("anatomy+lesion" if (s := class_set(r)) & set(ANATOMY) and s & set(LESION)
                   else "anatomy only" if s & set(ANATOMY) else "lesion only" for r in records)
    md += ["## Polygons", "",
           f"- Exact duplicate polygons removed while loading (same image, labels and points): "
           + ", ".join(f"{k} {v}" for k, v in DUPLICATE_POLYGONS.items()),
           f"- Points per polygon p10/50/90: {pct(npts)}; polygons with < 3 points: {small}; with points outside the image: {outside}",
           f"- Polygons carrying two labels (anatomy + lesion): {sum(multi.values())}; most common: "
           + ", ".join(f"{' + '.join(k)} ({v})" for k, v in multi.most_common(5)),
           f"- Images: " + ", ".join(f"{k} {v}" for k, v in cats.most_common()), ""]

    md5s = defaultdict(list)
    for r in records:
        md5s[scans[r["uid"]]["md5"]].append(r["uid"])
    exact = [v for v in md5s.values() if len(v) > 1]
    pairs = cross_patient_pairs(records, scans, CROSS_PATIENT_DUP)
    linked = sorted({tuple(sorted((a.rsplit("_f", 1)[0], b.rsplit("_f", 1)[0]))) for a, b, _ in pairs})
    md += ["## Duplicates", "",
           f"- Exactly identical images: {len(exact)} groups {exact}",
           f"- Near-identical images in different patients (thumbnail diff < {CROSS_PATIENT_DUP}): {len(pairs)} pairs, "
           f"linking patients {linked}. These patients are kept in the same split.", ""]
    diffs = []
    by_patient = defaultdict(list)
    for r in records:
        by_patient[r["patient"]].append(scans[r["uid"]]["thumb"].astype(np.float32))
    for th in by_patient.values():
        diffs += [float(np.abs(a - b).mean()) for i, a in enumerate(th) for b in th[i + 1:]]
    md += [f"- Within-patient image pairs: thumbnail diff p10/50/90 {pct(diffs)}; pairs < 2: {sum(d < 2 for d in diffs)}, "
           f"< 3: {sum(d < 3 for d in diffs)}, < 5: {sum(d < 5 for d in diffs)}", ""]

    vf = [scans[r["uid"]]["view_fraction"] for r in records]
    tf = [scans[r["uid"]]["text_fraction"] for r in records]
    nf = sum(1 for r in records if not scans[r["uid"]]["view_found"])
    blank = [r["uid"] for r in records if scans[r["uid"]]["texture"] < MIN_TEXTURE]
    md += ["## Scope view and text removal", "",
           f"- View not found (kept full frame): {nf}; blank frames (no image detail, excluded): {blank}",
           f"- View area as a fraction of the frame p10/50/90: {pct(vf)}",
           f"- Fraction of view pixels filled as burned-in text p10/50/90: {pct(tf)}; max {max(tf):.3f}", "",
           "![classes](raw_class_counts.png)", ""]

    (rep / "raw_analysis.md").write_text("\n".join(md))
    print(f"  wrote {rep / 'raw_analysis.md'}")


def stage_prepare(args) -> None:
    print(f"== prepare (task: {args.task})")
    records = load_records(args.raw_dir)
    scans = get_scans(records, args)
    classes = TASKS[args.task]

    # 1. patients linked by duplicate images -> one split group
    pairs = cross_patient_pairs(records, scans, CROSS_PATIENT_DUP)
    group_of = split_groups(records, pairs)

    # 2. split on ALL images and classes, so every task uses the same patients per split
    split_of_group = assign_split(records, group_of, args.ratios, args.seed)
    split_of = {r["uid"]: split_of_group[group_of[r["patient"]]] for r in records}

    # 3. which images belong to this task
    excluded = {}
    for r in records:
        if not scans[r["uid"]]["view_found"] or scans[r["uid"]]["texture"] < MIN_TEXTURE:
            excluded[r["uid"]] = "blank / broken frame (no scope view or no image detail)"
        elif args.task == "anatomy" and not class_set(r) & set(ANATOMY):
            excluded[r["uid"]] = "no anatomy label (lesion-only image)"
    removed = near_duplicates([r for r in records if r["uid"] not in excluded], scans, args.dedup_threshold)
    for u in removed:
        excluded[u] = f"near-duplicate (thumbnail diff < {args.dedup_threshold})"
    kept = [r for r in records if r["uid"] not in excluded]
    print(f"  images kept {len(kept)} / {len(records)}; excluded: {dict(Counter(excluded.values()))}")

    # 4. write images
    for sub in ("images", "labels", "lists", "previews", "annotations"):
        shutil.rmtree(args.out_dir / sub, ignore_errors=True)
    for s in SPLITS:
        (args.out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
    jobs = [{"rec": r, "max_side": args.max_side, "remove_text": not args.keep_text,
             "out_img": str(args.out_dir / "images" / split_of[r["uid"]] / f"{r['uid']}.png")} for r in kept]
    results = {x["uid"]: x for x in run_parallel(write_image, jobs, args.workers, "write")}

    # 5. YOLO polygon labels for the task
    def task_class(label: str) -> int | None:
        name = ALL_CLASSES[label]
        if args.task == "lesion_binary":
            return 0 if label in LESION else None
        return classes.index(name) if name in classes else None

    for r in kept:
        lines = []
        for p in results[r["uid"]]["polygons"]:
            W, H = results[r["uid"]]["out_size"]
            for label in p["labels"]:
                c = task_class(label)
                if c is not None:
                    coords = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in p["points"])
                    lines.append(f"{c} {coords}")
        (args.out_dir / "labels" / split_of[r["uid"]] / f"{r['uid']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    # 6. COCO json with all 18 classes (for later phases / other tools)
    (args.out_dir / "annotations").mkdir(exist_ok=True)
    cat_ids = {name: i + 1 for i, name in enumerate(TASKS["all"])}
    for s in SPLITS:
        coco = {"info": {"description": f"BM-BronchoLC cleaned ({s})", "version": "1"}, "licenses": [],
                "categories": [{"id": i, "name": n, "supercategory": "anatomy" if n in ANATOMY.values() else "lesion"}
                               for n, i in cat_ids.items()], "images": [], "annotations": []}
        for img_id, r in enumerate([r for r in kept if split_of[r["uid"]] == s], 1):
            res = results[r["uid"]]
            coco["images"].append({"id": img_id, "file_name": f"images/{s}/{r['uid']}.png", "width": res["out_size"][0],
                                   "height": res["out_size"][1], "cohort": r["cohort"], "patient": r["patient"]})
            for p in res["polygons"]:
                pts = np.array(p["points"])
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                for label in p["labels"]:
                    coco["annotations"].append({
                        "id": len(coco["annotations"]) + 1, "image_id": img_id, "category_id": cat_ids[ALL_CLASSES[label]],
                        "segmentation": [pts.reshape(-1).tolist()], "area": round(polygon_area(pts), 2),
                        "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)], "iscrowd": 0})
        (args.out_dir / "annotations" / f"instances_{s}.json").write_text(json.dumps(coco))

    # 7. yaml files + per-cohort lists
    names = dict(enumerate(classes))
    header = (f"# BM-BronchoLC {args.task} dataset (generated by prepare_data.py). Labels are polygons:\n"
              "# usable for detection (YOLO converts them to boxes) and segmentation (task=segment).\n"
              "# IMPORTANT: train with fliplr=0.0 - left/right bronchi swap under a horizontal flip.\n"
              "# Apply preprocessing.preprocess_frame() at inference so frames match training.\n")
    (args.out_dir / "data.yaml").write_text(header + yaml.safe_dump(
        {**{s: f"images/{s}" for s in SPLITS}, "nc": len(classes), "names": names}, sort_keys=False))
    (args.out_dir / "lists").mkdir(exist_ok=True)
    for c in COHORTS:
        entry = {}
        for s in SPLITS:
            files = sorted(f"./../images/{s}/{r['uid']}.png" for r in kept if r["cohort"] == c and split_of[r["uid"]] == s)
            (args.out_dir / "lists" / f"{s}_{c}.txt").write_text("\n".join(files) + "\n")
            entry[s] = f"lists/{s}_{c}.txt"
        (args.out_dir / f"data_{c}.yaml").write_text(
            f"# Evaluation-only view of the {COHORTS[c]} cohort.\n"
            + yaml.safe_dump({**entry, "nc": len(classes), "names": names}, sort_keys=False))

    # 8. manifest, dropped, config, report
    with open(args.out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "split", "cohort", "patient", "split_group", "orig_patient_id", "orig_video_id", "orig_image_id",
                    "source", "orig_w", "orig_h", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "view_found",
                    "text_fraction", "out_w", "out_h", "n_polygons", "task_labels"])
        for r in kept:
            res, sc = results[r["uid"]], scans[r["uid"]]
            lbl = args.out_dir / "labels" / split_of[r["uid"]] / f"{r['uid']}.txt"
            task_labels = sorted({classes[int(l.split()[0])] for l in lbl.read_text().splitlines() if l.strip()})
            w.writerow([r["uid"], split_of[r["uid"]], r["cohort"], r["patient"], group_of[r["patient"]], r["patient_id"],
                        r["video_id"], r["image_id"], os.path.relpath(r["path"], args.raw_dir), r["width"], r["height"],
                        *sc["box"], sc["view_found"], round(sc["text_fraction"], 4), *res["out_size"],
                        len(res["polygons"]), ";".join(task_labels)])
    with open(args.out_dir / "dropped.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "what", "labels", "reason"])
        for u, reason in sorted(excluded.items()):
            w.writerow([u, "image", "", reason])
        for r in kept:
            for labels, reason in results[r["uid"]]["dropped"]:
                w.writerow([r["uid"], "polygon", "+".join(labels), reason])
    (args.out_dir / "config_used.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "task": args.task, "classes": classes, "seed": args.seed,
        "ratios": args.ratios, "max_side": args.max_side, "remove_text": not args.keep_text,
        "dedup_threshold": args.dedup_threshold, "class_map": ALL_CLASSES,
        "linked_patients": sorted({tuple(sorted((a.rsplit('_f', 1)[0], b.rsplit('_f', 1)[0]))) for a, b, _ in pairs}),
        "group_split": split_of_group,
    }, indent=2))
    write_prepare_report(args, records, kept, excluded, split_of, results, classes)
    print(f"  wrote dataset to {args.out_dir}")


def write_prepare_report(args, records, kept, excluded, split_of, results, classes) -> None:
    rep = args.out_dir / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    md = ["# BM-BronchoLC prepared dataset report", "",
          f"Task **{args.task}** ({len(classes)} classes), patient-level split (seed {args.seed}, ratios {args.ratios}), "
          f"grayscale, text removal **{not args.keep_text}**, max side {args.max_side}px, near-duplicate threshold {args.dedup_threshold}.", ""]
    rows = []
    for s in SPLITS:
        pats = {r["patient"] for r in kept if split_of[r["uid"]] == s}
        row = [s, len(pats)] + [sum(1 for r in kept if r["cohort"] == c and split_of[r["uid"]] == s) for c in COHORTS]
        row.append(sum(row[2:]))
        rows.append(row + [f"{row[-1] / len(kept):.1%}"])
    md += ["## Images per split", "", md_table(["Split", "Patients"] + list(COHORTS) + ["Total", "Share"], rows), ""]

    inst = {s: Counter() for s in SPLITS}
    for r in kept:
        for l in (args.out_dir / "labels" / split_of[r["uid"]] / f"{r['uid']}.txt").read_text().splitlines():
            if l.strip():
                inst[split_of[r["uid"]]][classes[int(l.split()[0])]] += 1
    rows = [[c] + [inst[s][c] for s in SPLITS] + [sum(inst[s][c] for s in SPLITS)] for c in classes]
    md += ["## Polygon instances per class and split", "", md_table(["Class"] + list(SPLITS) + ["Total"], rows), ""]
    grouped_barh(rep / "prepared_class_counts.png", f"Instances per class and split ({args.task})",
                 classes, {s: [inst[s][c] for c in classes] for s in SPLITS}, "polygons")

    drops = Counter(reason for r in kept for _, reason in results[r["uid"]]["dropped"])
    md += ["## Processing summary", "",
           f"- Raw images {len(records)}, written {len(kept)}; excluded: " + ", ".join(f"{k} ({v})" for k, v in Counter(excluded.values()).items()),
           f"- Polygons written {sum(len(x['polygons']) for x in results.values())}; dropped: "
           + (", ".join(f"{k} ({v})" for k, v in drops.items()) or "0"),
           "- Output sizes: " + ", ".join(f"{k} ({v})" for k, v in Counter(f"{x['out_size'][0]}x{x['out_size'][1]}"
                                                                             for x in results.values()).most_common(6)),
           "", "![classes](prepared_class_counts.png)", "",
           "## Training notes", "",
           "- `fliplr=0.0`, `flipud=0.0`: flips swap left/right bronchi.",
           "- Report metrics per cohort with `data_lc.yaml` / `data_nlc.yaml`.",
           "- `annotations/instances_*.json` (COCO) keep all 18 classes for the lesion phase.", ""]
    (rep / "prepared_report.md").write_text("\n".join(md))


def stage_verify(args) -> None:
    print("== verify")
    out = args.out_dir
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    nc = cfg["nc"]
    manifest = list(csv.DictReader(open(out / "manifest.csv")))
    by_uid = {m["uid"]: m for m in manifest}
    errors, warnings, n_poly = [], [], 0
    hashes = defaultdict(list)
    for s in SPLITS:
        imgs = {p.stem: p for p in (out / "images" / s).glob("*.png")}
        lbls = {p.stem: p for p in (out / "labels" / s).glob("*.txt")}
        errors += [f"{s}/{u}: image without label" for u in imgs.keys() - lbls.keys()]
        errors += [f"{s}/{u}: label without image" for u in lbls.keys() - imgs.keys()]
        for u, p in imgs.items():
            m = by_uid.get(u)
            if m is None or m["split"] != s:
                errors.append(f"{s}/{u}: not in manifest for this split")
                continue
            with Image.open(p) as im:
                if [im.width, im.height] != [int(m["out_w"]), int(m["out_h"])]:
                    errors.append(f"{u}: size differs from manifest")
                if im.mode != "L":
                    errors.append(f"{u}: not grayscale ({im.mode})")
                hashes[hashlib.md5(im.tobytes()).hexdigest()].append((s, u))
            lines = [l for l in lbls[u].read_text().splitlines() if l.strip()] if u in lbls else []
            if not lines:
                warnings.append(f"{u}: no labels (negative image)")
            for l in lines:
                parts = l.split()
                n_poly += 1
                if not parts[0].isdigit() or int(parts[0]) >= nc:
                    errors.append(f"{u}: invalid class '{parts[0]}'")
                v = parts[1:]
                if len(v) < 6 or len(v) % 2:
                    errors.append(f"{u}: polygon needs >= 3 points: '{l[:60]}'")
                elif any(not 0 <= float(x) <= 1 for x in v):
                    errors.append(f"{u}: coordinate outside [0, 1]")
    for key in ("patient", "split_group"):
        seen = defaultdict(set)
        for m in manifest:
            seen[m[key]].add(m["split"])
        errors += [f"{key} {k} in several splits {sorted(v)}" for k, v in seen.items() if len(v) > 1]
    for items in hashes.values():
        if len({s for s, _ in items}) > 1:
            errors.append(f"identical image in several splits: {items}")
    train_cls = Counter(int(l.split()[0]) for p in (out / "labels" / "train").glob("*.txt") for l in p.read_text().splitlines() if l.strip())
    errors += [f"class {cfg['names'][c]} has no training polygons" for c in range(nc) if not train_cls[c]]
    for s in SPLITS:
        val_cls = Counter(int(l.split()[0]) for p in (out / "labels" / s).glob("*.txt") for l in p.read_text().splitlines() if l.strip())
        warnings += [f"class {cfg['names'][c]} has no polygons in {s}" for c in range(nc) if not val_cls[c]]

    write_previews(args, manifest, cfg["names"])
    print(f"  images {len(manifest)}, polygons {n_poly}, warnings {len(warnings)}, errors {len(errors)}")
    for w in warnings[:8]:
        print("  WARN ", w)
    for e in errors[:30]:
        print("  ERROR", e)
    (out / "reports" / "verify.txt").write_text(f"images {len(manifest)}\npolygons {n_poly}\n\nERRORS ({len(errors)})\n"
                                                + "\n".join(errors) + f"\n\nWARNINGS ({len(warnings)})\n" + "\n".join(warnings) + "\n")
    print("  RESULT:", "PASS" if not errors else "FAIL")
    if errors:
        sys.exit(1)


def write_previews(args, manifest, names) -> None:
    rng = np.random.default_rng(args.seed)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948", "#4a3aa7",
              "#00a3c4", "#a35c00", "#6b6b6b", "#b8007a", "#3c8c00", "#c47a00", "#5a00b8", "#008c7a", "#8c0000"]
    for s in SPLITS:
        for c in COHORTS:
            rows = [m for m in manifest if m["split"] == s and m["cohort"] == c]
            if not rows:
                continue
            dst = args.out_dir / "previews" / s
            dst.mkdir(parents=True, exist_ok=True)
            for m in rng.choice(rows, min(args.previews, len(rows)), replace=False):
                im = Image.open(args.out_dir / "images" / s / f"{m['uid']}.png").convert("RGB")
                d = ImageDraw.Draw(im)
                for l in (args.out_dir / "labels" / s / f"{m['uid']}.txt").read_text().splitlines():
                    v = l.split()
                    k = int(v[0])
                    pts = [(float(v[i]) * im.width, float(v[i + 1]) * im.height) for i in range(1, len(v), 2)]
                    d.polygon(pts, outline=colors[k % len(colors)], width=2)
                    d.text(pts[0], names[k], fill=colors[k % len(colors)])
                im.save(dst / f"{m['uid']}.jpg", quality=88)


# --------------------------------------------------------------------------- CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", nargs="?", default="all", choices=["analyze", "prepare", "verify", "all"])
    p.add_argument("--task", choices=list(TASKS), default="anatomy")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="folder containing Lung_cancer/ and Non_lung_cancer/")
    p.add_argument("--out-dir", type=Path, help="output folder (default data/bm_<task>)")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-side", type=int, default=640, help="longest image side after crop (no upscaling)")
    p.add_argument("--dedup-threshold", type=float, default=2.0, help="near-duplicate threshold within a patient (0 = off)")
    p.add_argument("--keep-text", action="store_true", help="do not remove burned-in text inside the view")
    p.add_argument("--previews", type=int, default=12)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args()
    if abs(sum(args.ratios) - 1) > 1e-6:
        p.error("--ratios must sum to 1")
    args.out_dir = (args.out_dir or ROOT / "data" / f"bm_{args.task}").resolve()
    args.cache_dir = ROOT / "data" / "cache"
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

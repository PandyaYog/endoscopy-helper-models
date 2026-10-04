#!/usr/bin/env python3
"""Charts for a finished training run.

    python plot_results.py --run uaal_runs/yolo26s --data data/uaal_det/data.yaml

Writes PNGs to <run>/plots/:
  map50_by_split.png       mAP50 on train / val / test, overall and per subset
  map50_per_class.png      mAP50 per class on train / val / test
  loss_curves.png          train vs val box / classification / L1 loss per epoch
  val_metrics_curves.png   validation mAP50, mAP50-95, precision, recall per epoch
  learning_rate.png        learning-rate schedule
  export_speed_accuracy.png  CPU speed vs accuracy of the exported formats (if export_report.json exists)

mAP50 for train/val/test is computed with the given model (default: the OpenVINO FP32 export, which
scores the same as best.pt) and cached in <run>/plots/split_metrics.json; use --recompute to refresh.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

SPLITS = ("train", "val", "test")
COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}   # validated categorical slots 1-3
EXTRA = ["#eda100", "#e87ba4"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e0"


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, color=INK, fontsize=11, loc="left")


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- metrics

def split_metrics(model_path: str, data_yaml: Path, imgsz: int, cache: Path, recompute: bool) -> dict:
    """mAP50 (overall + per class) on every split, for all images and for each subset yaml."""
    if cache.exists() and not recompute:
        return json.loads(cache.read_text())
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    root = data_yaml.parent
    cfg = yaml.safe_load(data_yaml.read_text())
    abs_yaml = cache.parent / "data_abs.yaml"
    abs_yaml.write_text(yaml.safe_dump({**{s: str(root / cfg[s]) for s in SPLITS if s in cfg},
                                        "nc": cfg["nc"], "names": cfg["names"]}))
    views = {"all": abs_yaml, **{p.stem.removeprefix("data_"): p for p in sorted(root.glob("data_*.yaml"))}}
    out = {}
    for view, yml in views.items():
        for split in SPLITS:
            if split not in yaml.safe_load(Path(yml).read_text()):
                continue
            r = model.val(data=str(yml), split=split, imgsz=imgsz, batch=1, device="cpu", plots=False,
                          verbose=False, project=str(cache.parent / "val_runs"), name=f"{view}_{split}", exist_ok=True)
            ap50 = r.box.all_ap[:, 0] if len(r.box.all_ap) else []
            out[f"{view}/{split}"] = {"mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
                                      "per_class_mAP50": {r.names[c]: round(float(a), 4) for c, a in zip(r.ap_class_index, ap50)}}
            print(f"  {view:9s} {split:5s} mAP50 {r.box.map50:.3f}")
    cache.write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- charts

def plot_map50_by_split(m: dict, path: Path):
    import matplotlib.pyplot as plt
    views = list(dict.fromkeys(k.split("/")[0] for k in m))
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(views))
    w = 0.26
    for i, s in enumerate(SPLITS):
        vals = [m.get(f"{v}/{s}", {}).get("mAP50", np.nan) for v in views]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.9, color=COLORS[s], label=s)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [v if v != "all" else "all images" for v in views])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mAP50", color=INK2)
    title(ax, "mAP50 on train / validation / test")
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, loc="upper right")
    save(fig, path)
    plt.close(fig)


def plot_map50_per_class(m: dict, path: Path):
    import matplotlib.pyplot as plt
    classes = list(m["all/train"]["per_class_mAP50"])
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(classes) + 1.4), facecolor=SURFACE)
    style(ax, "x")
    y = np.arange(len(classes))
    h = 0.26
    for i, s in enumerate(SPLITS):
        vals = [m.get(f"all/{s}", {}).get("per_class_mAP50", {}).get(c, np.nan) for c in classes]
        ax.barh(y + (i - 1) * h, vals, h * 0.9, color=COLORS[s], label=s)
    ax.set_yticks(y, classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("mAP50", color=INK2)
    title(ax, "mAP50 per class")
    ax.legend(frameon=False, labelcolor=INK2, ncol=3, loc="lower right")
    save(fig, path)
    plt.close(fig)


def read_results(run: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(run / "results.csv")))
    return {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}


def best_epoch(r) -> int:
    fitness = 0.1 * r["metrics/mAP50(B)"] + 0.9 * r["metrics/mAP50-95(B)"]  # Ultralytics' selection rule
    return int(r["epoch"][int(np.argmax(fitness))])


def mark_best(ax, ep):
    ax.axvline(ep, color=MUTED, linestyle="--", linewidth=1)
    ax.text(ep, ax.get_ylim()[1], f" best (epoch {ep})", color=INK2, fontsize=8, va="top")


def plot_losses(r, path: Path):
    import matplotlib.pyplot as plt
    names = [(k.split("/")[1]) for k in r if k.startswith("train/") and f"val/{k.split('/')[1]}" in r]
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), facecolor=SURFACE)
    ep, best = r["epoch"], best_epoch(r)
    for ax, n in zip(np.atleast_1d(axes), names):
        style(ax)
        ax.plot(ep, r[f"train/{n}"], color=COLORS["train"], linewidth=2, label="train")
        ax.plot(ep, r[f"val/{n}"], color=COLORS["val"], linewidth=2, label="val")
        ax.set_xlabel("epoch", color=INK2)
        if n == "cls_loss":  # epoch 1 is ~10x higher and flattens the rest of the curve
            ax.set_ylim(0, max(r[f"val/{n}"][1:].max(), r[f"train/{n}"][1:].max()) * 1.15)
        title(ax, n.replace("_", " "))
        mark_best(ax, best)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=8)
    save(fig, path)
    plt.close(fig)


def plot_val_metrics(r, path: Path):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    ep, best = r["epoch"], best_epoch(r)
    for ax, keys, ttl in ((a1, [("mAP50(B)", "mAP50"), ("mAP50-95(B)", "mAP50-95")], "Validation mAP per epoch"),
                          (a2, [("precision(B)", "precision"), ("recall(B)", "recall")], "Validation precision / recall per epoch")):
        style(ax)
        for (k, label), c in zip(keys, [COLORS["train"], COLORS["val"]]):
            ax.plot(ep, r[f"metrics/{k}"], color=c, linewidth=2, label=label)
        ax.set_ylim(0, 1)
        ax.set_xlabel("epoch", color=INK2)
        title(ax, ttl)
        mark_best(ax, best)
        ax.legend(frameon=False, labelcolor=INK2, fontsize=8, loc="lower right")
    save(fig, path)
    plt.close(fig)


def plot_lr(r, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    style(ax)
    ax.plot(r["epoch"], r["lr/pg0"], color=COLORS["train"], linewidth=2)
    ax.set_xlabel("epoch", color=INK2)
    ax.set_ylabel("learning rate", color=INK2)
    title(ax, "Learning-rate schedule (warm-up, then cosine decay)")
    save(fig, path)
    plt.close(fig)


def plot_export(report: Path, path: Path):
    import matplotlib.pyplot as plt
    rep = json.loads(report.read_text())
    fmts = [f for f in rep["accuracy"] if "fps" in rep["speed"].get(f, {}) and "mAP50" in rep["accuracy"][f]]
    fps = [rep["speed"][f]["fps"] for f in fmts]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(fmts) + 1.3), facecolor=SURFACE)
    style(ax, "x")
    y = np.arange(len(fmts))
    ax.barh(y, fps, 0.6, color=COLORS["train"])
    for i, f in enumerate(fmts):
        a = rep["accuracy"][f]
        ax.text(fps[i] + max(fps) * 0.01, i, f"{fps[i]:.0f} FPS  |  test mAP50 {a['mAP50']:.3f}, mAP50-95 {a['mAP50-95']:.3f}",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, fmts)
    ax.invert_yaxis()
    ax.set_xlim(0, max(fps) * 1.75)
    ax.set_xlabel(f"frames per second, model only ({rep['cpu']})", color=INK2)
    title(ax, "Exported models: CPU speed and test accuracy")
    save(fig, path)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=here / "uaal_runs" / "yolo26s", help="training run folder (has results.csv)")
    p.add_argument("--data", type=Path, default=here / "data" / "uaal_det" / "data.yaml")
    p.add_argument("--weights", help="model used for train/val/test mAP50 (default: export_dynamic/best_openvino_model)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--recompute", action="store_true", help="recompute the cached split metrics")
    args = p.parse_args()
    out = args.run / "plots"
    out.mkdir(exist_ok=True)
    weights = args.weights or str(args.run / "export_dynamic" / "best_openvino_model")
    if not Path(weights).exists():
        weights = str(args.run / "weights" / "best.pt")

    print(f"computing mAP50 per split with {weights} (cached after the first run)")
    m = split_metrics(weights, args.data.resolve(), args.imgsz, out / "split_metrics.json", args.recompute)
    plot_map50_by_split(m, out / "map50_by_split.png")
    plot_map50_per_class(m, out / "map50_per_class.png")
    r = read_results(args.run)
    plot_losses(r, out / "loss_curves.png")
    plot_val_metrics(r, out / "val_metrics_curves.png")
    plot_lr(r, out / "learning_rate.png")
    report = args.run / "export_dynamic" / "export_report.json"
    if report.exists():
        plot_export(report, out / "export_speed_accuracy.png")


if __name__ == "__main__":
    main()

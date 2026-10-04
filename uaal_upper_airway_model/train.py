#!/usr/bin/env python3
"""Train, evaluate and export the UAAL upper-airway landmark detector.

Model: Ultralytics YOLO26 (default yolo26s). YOLO26 is the Ultralytics family designed for
edge/CPU deployment: NMS-free end-to-end head (no post-processing step on the CPU) and
faster CPU inference than YOLO11 at similar accuracy. yolo26n is the faster/smaller option.

Pipeline
  1. find the prepared dataset (data.yaml from prepare_data.py; auto-detected on Kaggle)
  2. train with endoscopy-specific augmentation
  3. evaluate the best checkpoint on val and test, overall and per subset (clinical / phantom)
  4. export to ONNX and OpenVINO (FP32 + INT8) for CPU inference
  5. benchmark CPU speed of each format and write report.md + summary.json

Kaggle (GPU + Internet on):
  !pip install -q -U "ultralytics>=8.4" "albumentations>=2.0" openvino nncf onnx onnxruntime
  !python /kaggle/input/<your-dataset>/code/train.py
  # results end up in /kaggle/working/uaal_runs (also zipped as uaal_runs.zip)

Local CPU benchmark of an exported model:
  python train.py benchmark --weights runs/yolo26s/weights/best_openvino_model --data data/uaal_det/data.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
ON_KAGGLE = KAGGLE_INPUT.exists()
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- data

def find_data_yaml(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    candidates = []
    for root in ([KAGGLE_INPUT] if ON_KAGGLE else []) + [HERE / "data", HERE.parent]:
        if root.exists():
            candidates += sorted(root.glob("**/uaal_det/data.yaml"))
    if not candidates:
        sys.exit("data.yaml not found - pass --data /path/to/uaal_det/data.yaml")
    return candidates[0].resolve()


def make_train_yaml(data_yaml: Path, out_dir: Path) -> Path:
    """Write a dataset yaml with absolute paths into a writable folder.

    Kaggle inputs are read-only, so the yaml we train from must live in /kaggle/working
    and point back to the images with absolute paths.
    """
    cfg = yaml.safe_load(data_yaml.read_text())
    root = data_yaml.parent
    out = {k: str(root / cfg[k]) for k in ("train", "val", "test") if k in cfg}
    out.update(nc=cfg["nc"], names=cfg["names"])
    dst = out_dir / "data_abs.yaml"
    dst.write_text(yaml.safe_dump(out, sort_keys=False))
    return dst


# --------------------------------------------------------------------------- augmentation

def build_hyperparameters(args) -> dict:
    """Augmentation and training settings chosen for grayscale bronchoscopy frames.

    Geometric (built into Ultralytics):
      - rotation +-15 deg: the bronchoscope rotates around its axis during navigation
      - scale 0.5 / translate 0.1: distance to the anatomy and framing vary
      - mosaic: more objects and contexts per batch; switched off for the last 15 epochs
      - NO horizontal/vertical flips: they would turn a left vocal fold into a right one
    Photometric:
      - brightness (hsv_v); hue/saturation are useless on grayscale images
    Endoscopy-specific (Albumentations, pixel-level so boxes are unaffected):
      - motion/defocus blur (scope movement, out-of-focus frames), sensor noise,
        CLAHE / gamma / contrast (illumination differences between scopes),
        JPEG compression (video capture artefacts)
    Class imbalance (nose/nostril/channel/trachea are 5-10x rarer than glottic classes):
      - cls_pw: inverse-frequency class weights in the classification loss
    """
    return dict(
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
        optimizer="auto", cos_lr=True, seed=args.seed, deterministic=False, fraction=args.fraction,
        # geometry
        degrees=15.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=1.0, close_mosaic=15, mixup=0.0, cutmix=0.0,
        # photometric
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
        # imbalance
        cls_pw=args.cls_pw,
    )


def endoscopy_augmentations() -> list | None:
    try:
        import albumentations as A
    except ImportError:
        print("WARNING: albumentations not installed - training without endoscopy-specific augmentation")
        return None
    return [
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7)), A.Defocus(radius=(1, 3))], p=0.25),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.15),
        A.CLAHE(clip_limit=(1, 3), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.25, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.2),
    ]


# --------------------------------------------------------------------------- progress

def train_box_loss(trainer) -> float | None:
    """Mean training box loss of the epoch (the loss container differs between Ultralytics versions)."""
    try:
        items = trainer.label_loss_items(trainer.tloss, prefix="train")
        return round(float(items["train/box_loss"]), 4)
    except Exception:
        return None


def add_progress_logging(model, log_path: Path, total_epochs: int) -> None:
    """One compact line per epoch on the console + progress.jsonl (useful when the log is long)."""
    t0 = time.time()
    state = {"stopped": False}

    def on_fit_epoch_end(trainer):
        if int(os.environ.get("RANK", -1)) > 0:  # multi-GPU (DDP): only the main process has the metrics
            return
        m = trainer.metrics or {}
        if state["stopped"]:  # Ultralytics re-runs this callback once for the final validation of best.pt
            print(f"[progress] best.pt final validation | mAP50 {m.get('metrics/mAP50(B)', 0):.3f} "
                  f"| mAP50-95 {m.get('metrics/mAP50-95(B)', 0):.3f}", flush=True)
            return
        state["stopped"] = bool(trainer.stop)
        ep = trainer.epoch + 1
        elapsed = time.time() - t0
        eta = elapsed / ep * (total_epochs - ep)
        row = {
            "epoch": ep, "elapsed_min": round(elapsed / 60, 1), "eta_min": round(eta / 60, 1),
            "train_box_loss": train_box_loss(trainer),
            "precision": round(m.get("metrics/precision(B)", 0), 4), "recall": round(m.get("metrics/recall(B)", 0), 4),
            "mAP50": round(m.get("metrics/mAP50(B)", 0), 4), "mAP50-95": round(m.get("metrics/mAP50-95(B)", 0), 4),
            "best_fitness": round(float(trainer.best_fitness or 0), 4),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[progress] epoch {ep}/{total_epochs} | mAP50 {row['mAP50']:.3f} | mAP50-95 {row['mAP50-95']:.3f} "
              f"| P {row['precision']:.3f} R {row['recall']:.3f} | {row['elapsed_min']} min elapsed, ~{row['eta_min']} min left",
              flush=True)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


# --------------------------------------------------------------------------- evaluation / export / speed

def evaluate(weights: Path, data_yaml: Path, train_yaml: Path, imgsz: int, out_dir: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    views = {"all": train_yaml}
    for subset in ("clinical", "phantom"):
        p = data_yaml.parent / f"data_{subset}.yaml"
        if p.exists():
            views[subset] = p
    results = {}
    for split in ("val", "test"):
        for view, yml in views.items():
            if split not in yaml.safe_load(Path(yml).read_text()):
                continue
            r = model.val(data=str(yml), split=split, imgsz=imgsz, batch=16, plots=view == "all",
                          project=str(out_dir), name=f"eval_{split}_{view}", exist_ok=True, verbose=False)
            per_class = {r.names[c]: round(float(r.box.maps[c]), 4) for c in r.ap_class_index}
            results[f"{split}/{view}"] = {
                "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
                "mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
                "per_class_mAP50-95": per_class,
            }
            print(f"[eval] {split:4s} {view:8s} mAP50 {r.box.map50:.3f}  mAP50-95 {r.box.map:.3f}")
    return results


def export_models(weights: Path, train_yaml: Path, imgsz: int) -> dict[str, str]:
    from ultralytics import YOLO

    exports = {"pytorch": str(weights)}
    # dynamic=True: rectangular frames are not padded to a square. A fixed 640x640 export lost
    # 3 mAP50-95 points on the UAAL test set (0.440 -> 0.408) and was slower.
    jobs = {
        "onnx": dict(format="onnx", simplify=True, dynamic=True),
        "openvino_fp32": dict(format="openvino", dynamic=True),
        "openvino_int8": dict(format="openvino", dynamic=True, int8=True, data=str(train_yaml), fraction=1.0),
    }
    for name, kw in jobs.items():
        try:
            path = YOLO(str(weights)).export(imgsz=imgsz, **kw)
            # int8 and fp32 OpenVINO use different folder names, keep both
            exports[name] = str(path)
            print(f"[export] {name}: {path}")
        except Exception as e:  # an exporter failing must not lose the trained model
            print(f"[export] {name} FAILED: {e}")
    return exports


def evaluate_exports(exports: dict[str, str], train_yaml: Path, imgsz: int, task: str) -> dict:
    """Test-set accuracy of every exported format, so a lossy export (e.g. INT8) is visible."""
    from ultralytics import YOLO

    split = "test" if "test" in yaml.safe_load(Path(train_yaml).read_text()) else "val"
    out = {}
    for fmt, path in exports.items():
        try:
            r = YOLO(path, task=task).val(data=str(train_yaml), split=split, imgsz=imgsz, batch=1, device="cpu",
                                          plots=False, verbose=False, project=str(Path(path).parent / "export_eval"))
            m = r.seg if task == "segment" else r.box
            out[fmt] = {"split": split, "mAP50": round(float(m.map50), 4), "mAP50-95": round(float(m.map), 4)}
            print(f"[export-eval] {fmt}: {split} mAP50 {m.map50:.3f}  mAP50-95 {m.map:.3f}", flush=True)
        except Exception as e:
            out[fmt] = {"error": str(e)}
            print(f"[export-eval] {fmt} FAILED: {e}", flush=True)
    return out


def benchmark_cpu(model_path: str, data_yaml: Path, imgsz: int, n_images: int = 100) -> dict:
    """Average per-image latency on CPU, measured on test (or val) images."""
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(data_yaml).read_text())
    split = cfg.get("test") or cfg["val"]
    img_dir = Path(split) if Path(split).is_absolute() else Path(data_yaml).parent / split
    images = sorted(img_dir.glob("*.png"))[:n_images]
    model = YOLO(model_path, task="detect")
    for im in images[:5]:  # warm-up
        model.predict(str(im), imgsz=imgsz, device="cpu", verbose=False)
    pre = inf = post = 0.0
    for im in images:
        s = model.predict(str(im), imgsz=imgsz, device="cpu", verbose=False)[0].speed
        pre, inf, post = pre + s["preprocess"], inf + s["inference"], post + s["postprocess"]
    n = len(images)
    total = (pre + inf + post) / n
    return {"images": n, "preprocess_ms": round(pre / n, 1), "inference_ms": round(inf / n, 1),
            "postprocess_ms": round(post / n, 1), "total_ms": round(total, 1), "fps": round(1000 / total, 1)}


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or "unknown"


# --------------------------------------------------------------------------- report

def write_report(out_dir: Path, summary: dict) -> None:
    lines = ["# UAAL upper-airway detector - training report", "",
             f"- Model: **{summary['model']}**, image size {summary['imgsz']}, epochs run {summary['epochs_run']} "
             f"(max {summary['epochs']}), training time {summary['train_minutes']} min",
             f"- Best checkpoint: `{summary['best_weights']}`", ""]
    lines += ["## Accuracy", "", "| Split / subset | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for k, v in summary["metrics"].items():
        lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")
    test = summary["metrics"].get("test/all") or summary["metrics"].get("val/all")
    if test:
        lines += ["", "## Per-class mAP50-95 (test, all)", "", "| Class | mAP50-95 |", "|---|---|"]
        lines += [f"| {c} | {v:.3f} |" for c, v in test["per_class_mAP50-95"].items()]
    lines += ["", f"## CPU speed ({summary['cpu']})", "",
              "| Format | Test mAP50-95 | Pre (ms) | Inference (ms) | Post (ms) | Total (ms) | FPS |", "|---|---|---|---|---|---|---|"]
    for k, v in summary["cpu_benchmark"].items():
        acc = summary.get("export_accuracy", {}).get(k, {}).get("mAP50-95", "-")
        if "error" in v:
            lines.append(f"| {k} | {acc} | failed: {v['error'][:60]} | | | | |")
        else:
            lines.append(f"| {k} | {acc} | {v['preprocess_ms']} | {v['inference_ms']} | {v['postprocess_ms']} | {v['total_ms']} | {v['fps']} |")
    lines += ["", "Kaggle CPUs are slow, shared server cores; re-run `python train.py benchmark` on the target "
              "machine for real numbers.", ""]
    (out_dir / "report.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- commands

def cmd_train(args) -> None:
    from ultralytics import YOLO
    import ultralytics

    out_root = Path(args.out_dir or (KAGGLE_WORKING / "uaal_runs" if ON_KAGGLE else HERE / "runs")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_yaml = find_data_yaml(args.data)
    train_yaml = make_train_yaml(data_yaml, out_root)
    print(f"ultralytics {ultralytics.__version__} | data {data_yaml} | output {out_root}")

    all_summaries = {}
    for model_name in args.models:
        run_dir = out_root / model_name
        if args.resume_from:  # checkpoint saved from an earlier Kaggle session
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copy(args.resume_from, run_dir / "weights" / "last.pt")
            args.resume = True
        if run_dir.exists() and not args.resume:
            shutil.rmtree(run_dir)
        progress = out_root / f"{model_name}_progress.jsonl"
        if not args.resume:
            progress.unlink(missing_ok=True)

        model = YOLO(str(run_dir / "weights" / "last.pt") if args.resume else f"{model_name}.pt")
        add_progress_logging(model, progress, args.epochs)
        hyp = build_hyperparameters(args)
        augs = endoscopy_augmentations()
        print(f"[train] {model_name}: {json.dumps(hyp)}")
        t0 = time.time()
        model.train(data=str(train_yaml), project=str(out_root), name=model_name, exist_ok=True,
                    device=args.device, workers=args.workers, cache=args.cache, resume=args.resume,
                    plots=True, **({"augmentations": augs} if augs else {}), **hyp)
        train_min = round((time.time() - t0) / 60, 1)
        best = run_dir / "weights" / "best.pt"

        print(f"[train] done in {train_min} min - evaluating {best}")
        metrics = evaluate(best, data_yaml, train_yaml, args.imgsz, run_dir)
        exports = export_models(best, train_yaml, args.imgsz)
        export_acc = evaluate_exports(exports, train_yaml, args.imgsz, "detect")
        speed = {}
        for fmt, path in exports.items():
            try:
                speed[fmt] = benchmark_cpu(path, train_yaml, args.imgsz)
                print(f"[cpu] {fmt}: {speed[fmt]['total_ms']} ms/img ({speed[fmt]['fps']} FPS)")
            except Exception as e:
                speed[fmt] = {"error": str(e)}
                print(f"[cpu] {fmt} FAILED: {e}")

        epochs_run = len({json.loads(l)["epoch"] for l in open(progress) if l.strip()}) if progress.exists() else None
        summary = {"model": model_name, "imgsz": args.imgsz, "epochs": args.epochs, "epochs_run": epochs_run,
                   "train_minutes": train_min, "best_weights": str(best), "metrics": metrics,
                   "exports": exports, "export_accuracy": export_acc, "cpu": cpu_name(), "cpu_benchmark": speed,
                   "hyperparameters": hyp, "ultralytics": ultralytics.__version__}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_report(run_dir, summary)
        all_summaries[model_name] = summary
        print(f"[done] {model_name}: report at {run_dir / 'report.md'}")

    if len(all_summaries) > 1:
        print("\nmodel comparison (test/all):")
        for name, s in all_summaries.items():
            t = s["metrics"].get("test/all", {})
            fps = s["cpu_benchmark"].get("openvino_int8", {}).get("fps") or s["cpu_benchmark"].get("onnx", {}).get("fps")
            print(f"  {name:10s} mAP50 {t.get('mAP50', 0):.3f}  mAP50-95 {t.get('mAP50-95', 0):.3f}  CPU FPS {fps}")
    if ON_KAGGLE:
        archive = shutil.make_archive(str(KAGGLE_WORKING / "uaal_runs"), "zip", out_root)
        print(f"[done] download {archive}")


def cmd_export(args) -> None:
    """Re-export a trained best.pt (dynamic ONNX/OpenVINO FP32/INT8), check accuracy and CPU speed."""
    if not args.weights or not args.weights.endswith(".pt"):
        sys.exit("--weights path/to/best.pt is required for export")
    weights = Path(args.weights).resolve()
    train_yaml = make_train_yaml(find_data_yaml(args.data), weights.parent)
    exports = export_models(weights, train_yaml, args.imgsz)
    acc = evaluate_exports(exports, train_yaml, args.imgsz, "detect")
    speed = {}
    for fmt, path in exports.items():
        try:
            speed[fmt] = benchmark_cpu(path, train_yaml, args.imgsz)
        except Exception as e:
            speed[fmt] = {"error": str(e)}
    report = {"cpu": cpu_name(), "imgsz": args.imgsz, "exports": exports, "accuracy": acc, "speed": speed}
    (weights.parent / "export_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{'format':15s} {'test mAP50-95':>14s} {'ms/img':>8s} {'FPS':>6s}  ({cpu_name()})")
    for fmt in exports:
        print(f"{fmt:15s} {str(acc[fmt].get('mAP50-95', 'err')):>14s} {str(speed[fmt].get('total_ms', 'err')):>8s} "
              f"{str(speed[fmt].get('fps', 'err')):>6s}")


def cmd_benchmark(args) -> None:
    if not args.weights:
        sys.exit("--weights is required for benchmark")
    data_yaml = find_data_yaml(args.data)
    r = benchmark_cpu(args.weights, data_yaml, args.imgsz, args.n_images)
    print(f"{cpu_name()}\n{json.dumps(r, indent=2)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="train", choices=["train", "export", "benchmark"])
    p.add_argument("--data", help="path to uaal_det/data.yaml (auto-detected if omitted)")
    p.add_argument("--models", nargs="+", default=["yolo26s"],
                   help="one or more models to train in sequence, e.g. yolo26s yolo26n")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30, help="early stop after N epochs without improvement")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cls-pw", type=float, default=0.5, help="class-imbalance weighting power (0 = off)")
    p.add_argument("--device", default=None, help="'0', '0,1' (both Kaggle T4s) or 'cpu'; default auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="ram", help="'ram', 'disk' or 'false'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the train set (quick tests)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run from <out-dir>/<model>/weights/last.pt")
    p.add_argument("--resume-from", help="continue from a last.pt saved elsewhere (e.g. a previous Kaggle session)")
    p.add_argument("--out-dir", help="output folder (default /kaggle/working/uaal_runs or ./runs)")
    p.add_argument("--weights", help="export: best.pt; benchmark: model file or exported model folder")
    p.add_argument("--n-images", type=int, default=100, help="benchmark: number of images")
    args = p.parse_args()
    args.cache = False if str(args.cache).lower() in ("false", "0", "no") else args.cache
    return args


if __name__ == "__main__":
    a = parse_args()
    {"train": cmd_train, "export": cmd_export, "benchmark": cmd_benchmark}[a.command](a)

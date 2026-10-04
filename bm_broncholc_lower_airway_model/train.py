#!/usr/bin/env python3
"""Train, evaluate and export the BM-BronchoLC lower-airway landmark model.

Model: Ultralytics YOLO26 (default yolo26s, detection). YOLO26 is built for edge/CPU deployment:
NMS-free end-to-end head and fast CPU inference. The dataset labels are polygons, so the same
data also trains a segmentation model (--task segment -> yolo26s-seg).

Pipeline
  1. find and pre-flight check the prepared dataset (data.yaml from prepare_data.py)
  2. record the run configuration (arguments, library versions, GPUs, dataset fingerprint)
  3. train with endoscopy-specific augmentation (no flips: left/right bronchi)
  4. evaluate best.pt on val and test, overall and per cohort (lung cancer / non lung cancer)
  5. export to ONNX and OpenVINO (FP32 + INT8), benchmark CPU speed
  6. write report.md, summary.json and model_card.json; zip everything on Kaggle

Kaggle (GPU + Internet on):
  !pip install -q -U "ultralytics>=8.4" "albumentations>=2.0" openvino nncf onnx onnxruntime onnxslim
  !python /kaggle/input/<dataset>/code/train.py
  # results: /kaggle/working/bm_runs (zipped as /kaggle/working/bm_runs.zip)

Local CPU benchmark of an exported model:
  python train.py benchmark --weights runs/yolo26s/weights/best_int8_openvino_model --data data/bm_anatomy/data.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import yaml

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
ON_KAGGLE = KAGGLE_INPUT.exists()
HERE = Path(__file__).resolve().parent
COHORTS = {"lc": "lung cancer", "nlc": "non lung cancer"}
RUN_NAME = "bm_runs"


def log(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def is_main_process() -> bool:
    """False inside the extra worker processes of multi-GPU (DDP) training."""
    return int(os.environ.get("RANK", -1)) <= 0


# --------------------------------------------------------------------------- data

def find_data_yaml(arg: str | None) -> Path:
    if arg:
        p = Path(arg).resolve()
        if not p.exists():
            sys.exit(f"--data {p} does not exist")
        return p
    candidates = []
    for root in ([KAGGLE_INPUT] if ON_KAGGLE else []) + [HERE / "data", HERE.parent]:
        if root.exists():
            candidates += sorted(root.glob("**/bm_*/data.yaml"))
    if not candidates:
        sys.exit("No bm_*/data.yaml found - pass --data /path/to/bm_anatomy/data.yaml")
    if len(candidates) > 1:
        log("data", f"several datasets found, using the first (pass --data to choose): {[str(c) for c in candidates]}")
    return candidates[0].resolve()


def preflight_check(data_yaml: Path) -> dict:
    """Fail fast (before any GPU time is spent) if the dataset is incomplete or inconsistent."""
    cfg = yaml.safe_load(data_yaml.read_text())
    root = data_yaml.parent
    problems, stats = [], {}
    for key in ("train", "val", "nc", "names"):
        if key not in cfg:
            problems.append(f"data.yaml has no '{key}'")
    if not problems and len(cfg["names"]) != cfg["nc"]:
        problems.append(f"nc={cfg['nc']} but {len(cfg['names'])} names")
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        imgs = {p.stem for p in (root / cfg[split]).glob("*.png")}
        lbl_dir = root / str(cfg[split]).replace("images", "labels", 1)
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}
        missing = imgs - lbls
        if not imgs:
            problems.append(f"{split}: no images in {root / cfg[split]}")
        if missing:
            problems.append(f"{split}: {len(missing)} images without labels (e.g. {sorted(missing)[:3]})")
        n_inst = sum(1 for p in lbl_dir.glob("*.txt") for l in p.read_text().splitlines() if l.strip())
        stats[split] = {"images": len(imgs), "instances": n_inst}
    if problems:
        sys.exit("Dataset pre-flight check failed:\n  " + "\n  ".join(problems))
    log("data", f"pre-flight OK: {json.dumps(stats)} | classes: {list(cfg['names'].values()) if isinstance(cfg['names'], dict) else cfg['names']}")
    return stats


def dataset_fingerprint(data_yaml: Path) -> dict:
    """Identify exactly which prepared dataset a model was trained on."""
    root = data_yaml.parent
    info = {"data_yaml": str(data_yaml)}
    manifest = root / "manifest.csv"
    if manifest.exists():
        info["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]
    cfg = root / "config_used.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        info.update({k: c.get(k) for k in ("created", "task", "seed", "ratios", "max_side", "remove_text")})
    return info


def make_train_yaml(data_yaml: Path, out_dir: Path) -> Path:
    """Dataset yaml with absolute paths in a writable folder (Kaggle inputs are read-only)."""
    cfg = yaml.safe_load(data_yaml.read_text())
    out = {k: str(data_yaml.parent / cfg[k]) for k in ("train", "val", "test") if k in cfg}
    out.update(nc=cfg["nc"], names=cfg["names"])
    dst = out_dir / "data_abs.yaml"
    dst.write_text(yaml.safe_dump(out, sort_keys=False))
    return dst


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu_name()}
    for mod in ("ultralytics", "torch", "albumentations", "openvino", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        info["gpus"] = []
    return info


# --------------------------------------------------------------------------- augmentation

def build_hyperparameters(args) -> dict:
    """Training and augmentation settings for grayscale bronchoscopy frames (~1,500 training images).

    Geometric (Ultralytics):
      - rotation +-15 deg: the scope rotates around its axis
      - scale 0.5 / translate 0.1: distance to the carina/bronchi and framing vary
      - mosaic: more context per batch, important for a small dataset; off for the last 20 epochs
        so the model finishes on realistic single frames
      - NO flips: a horizontal flip turns the left main bronchus into a right one
    Photometric: brightness only (hsv_v); hue/saturation do nothing on grayscale images.
    Endoscopy-specific: see endoscopy_augmentations().
    Class imbalance: cls_pw weights rare classes (trachea, intermediate bronchus) up.
    """
    return dict(
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, patience=args.patience,
        optimizer="auto", cos_lr=True, seed=args.seed, deterministic=False, fraction=args.fraction,
        degrees=15.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        fliplr=0.0, flipud=0.0,
        mosaic=1.0, close_mosaic=20, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
        cls_pw=args.cls_pw,
    )


def endoscopy_augmentations() -> list | None:
    """Pixel-level augmentations (labels are unaffected), matched to bronchoscopy video:
    motion/defocus blur (scope movement), sensor noise, CLAHE/gamma/contrast (light source and
    NBI vs white light differences), JPEG artefacts (video capture)."""
    try:
        import albumentations as A
    except ImportError:
        log("augment", "WARNING: albumentations not installed - training without endoscopy-specific augmentation")
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

def metric_key(task: str, name: str) -> str:
    return f"metrics/{name}({'M' if task == 'segment' else 'B'})"


def add_progress_logging(model, log_path: Path, total_epochs: int, task: str) -> None:
    """One compact line per epoch + progress.jsonl. Only the main process logs (DDP-safe)."""
    t0 = time.time()
    state = {"stopped": False}

    def on_fit_epoch_end(trainer):
        if not is_main_process():
            return
        m = trainer.metrics or {}
        m50, m5095 = m.get(metric_key(task, "mAP50"), 0), m.get(metric_key(task, "mAP50-95"), 0)
        if state["stopped"]:  # Ultralytics re-runs the callback once for the final validation of best.pt
            log("progress", f"best.pt final validation | mAP50 {m50:.3f} | mAP50-95 {m5095:.3f}")
            return
        state["stopped"] = bool(trainer.stop)
        ep = trainer.epoch + 1
        elapsed = time.time() - t0
        row = {"epoch": ep, "elapsed_min": round(elapsed / 60, 1),
               "eta_min": round(elapsed / ep * (total_epochs - ep) / 60, 1),
               "precision": round(m.get(metric_key(task, "precision"), 0), 4),
               "recall": round(m.get(metric_key(task, "recall"), 0), 4),
               "mAP50": round(m50, 4), "mAP50-95": round(m5095, 4),
               "best_fitness": round(float(trainer.best_fitness or 0), 4)}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        log("progress", f"epoch {ep}/{total_epochs} | mAP50 {row['mAP50']:.3f} | mAP50-95 {row['mAP50-95']:.3f} "
                        f"| P {row['precision']:.3f} R {row['recall']:.3f} | {row['elapsed_min']} min elapsed, "
                        f"~{row['eta_min']} min left (upper bound, early stopping may end sooner)")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


# --------------------------------------------------------------------------- evaluation / export / speed

def _metrics(r, task: str) -> dict:
    m = r.seg if task == "segment" else r.box
    out = {"precision": round(float(m.mp), 4), "recall": round(float(m.mr), 4),
           "mAP50": round(float(m.map50), 4), "mAP50-95": round(float(m.map), 4),
           "per_class_mAP50-95": {r.names[c]: round(float(m.maps[c]), 4) for c in r.ap_class_index}}
    if task == "segment":
        out["box_mAP50-95"] = round(float(r.box.map), 4)
    return out


def evaluate(weights: Path, data_yaml: Path, train_yaml: Path, imgsz: int, out_dir: Path, task: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    views = {"all": train_yaml}
    views.update({c: data_yaml.parent / f"data_{c}.yaml" for c in COHORTS if (data_yaml.parent / f"data_{c}.yaml").exists()})
    results = {}
    for split in ("val", "test"):
        for view, yml in views.items():
            if split not in yaml.safe_load(Path(yml).read_text()):
                continue
            r = model.val(data=str(yml), split=split, imgsz=imgsz, batch=16, plots=view == "all",
                          project=str(out_dir), name=f"eval_{split}_{view}", exist_ok=True, verbose=False)
            results[f"{split}/{view}"] = _metrics(r, task)
            log("eval", f"{split:4s} {view:4s} mAP50 {results[f'{split}/{view}']['mAP50']:.3f}  "
                        f"mAP50-95 {results[f'{split}/{view}']['mAP50-95']:.3f}")
    return results


def export_models(weights: Path, train_yaml: Path, imgsz: int) -> dict[str, str]:
    from ultralytics import YOLO

    exports = {"pytorch": str(weights)}
    # dynamic=True: rectangular frames are not padded to a square. A fixed-size export lost
    # 3 mAP50-95 points on the UAAL test set (0.440 -> 0.408) and was slower.
    jobs = {"onnx": dict(format="onnx", simplify=True, dynamic=True),
            "openvino_fp32": dict(format="openvino", dynamic=True),
            "openvino_int8": dict(format="openvino", dynamic=True, int8=True, data=str(train_yaml), fraction=1.0)}
    for name, kw in jobs.items():
        try:
            exports[name] = str(YOLO(str(weights)).export(imgsz=imgsz, **kw))
            log("export", f"{name}: {exports[name]}")
        except Exception as e:  # an exporter failing must never lose the trained model
            log("export", f"{name} FAILED: {e}")
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


def benchmark_cpu(model_path: str, data_yaml: Path, imgsz: int, task: str, n_images: int = 100) -> dict:
    """Average per-image CPU latency (pre-process + inference + post-process) on test images."""
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(data_yaml).read_text())
    split = cfg.get("test") or cfg["val"]
    img_dir = Path(split) if Path(split).is_absolute() else Path(data_yaml).parent / split
    images = sorted(img_dir.glob("*.png"))[:n_images]
    model = YOLO(model_path, task=task)
    for im in images[:5]:
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
    return platform.processor() or "unknown"


# --------------------------------------------------------------------------- report

def write_report(out_dir: Path, s: dict) -> None:
    lines = [f"# BM-BronchoLC lower-airway {s['task']} model - training report", "",
             f"- Model **{s['model']}**, image size {s['imgsz']}, epochs run {s['epochs_run']} (max {s['epochs']}), "
             f"training time {s['train_minutes']} min",
             f"- Dataset: `{s['dataset'].get('data_yaml')}` (manifest {s['dataset'].get('manifest_sha256')}, "
             f"prepared {s['dataset'].get('created')})",
             f"- Best checkpoint: `{s['best_weights']}`", "",
             "## Accuracy", "", "| Split / cohort | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for k, v in s["metrics"].items():
        lines.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")
    ref = s["metrics"].get("test/all") or s["metrics"].get("val/all")
    if ref:
        lines += ["", "## Per-class mAP50-95 (test, all)", "", "| Class | mAP50-95 |", "|---|---|"]
        lines += [f"| {c} | {v:.3f} |" for c, v in sorted(ref["per_class_mAP50-95"].items(), key=lambda kv: -kv[1])]
    lines += ["", f"## CPU speed ({s['environment']['cpu']})", "",
              "| Format | Test mAP50-95 | Pre (ms) | Inference (ms) | Post (ms) | Total (ms) | FPS |", "|---|---|---|---|---|---|---|"]
    for k, v in s["cpu_benchmark"].items():
        acc = s.get("export_accuracy", {}).get(k, {}).get("mAP50-95", "-")
        lines.append(f"| {k} | {acc} | failed: {v['error'][:60]} | | | | |" if "error" in v else
                     f"| {k} | {acc} | {v['preprocess_ms']} | {v['inference_ms']} | {v['postprocess_ms']} | {v['total_ms']} | {v['fps']} |")
    lines += ["", "Kaggle CPUs are shared server cores: re-run `python train.py benchmark` on the target machine.",
              "", "Plots (confusion matrix, PR curves, training curves) are in the run folder and `eval_*` folders.", ""]
    (out_dir / "report.md").write_text("\n".join(lines))


def write_model_card(run_dir: Path, s: dict, names) -> None:
    """Everything an inference application needs to use the model correctly."""
    test = s["metrics"].get("test/all", {})
    card = {
        "model": s["model"], "task": s["task"], "imgsz": s["imgsz"],
        "classes": names if isinstance(names, dict) else dict(enumerate(names)),
        "input": "single-channel grayscale replicated to 3 channels; frames must go through "
                 "preprocessing.preprocess_frame() (scope-view crop + text removal) first",
        "training_data": "BM-BronchoLC (Scientific Data 2024), patient-level split", "dataset": s["dataset"],
        "test_metrics": {k: v for k, v in test.items() if k != "per_class_mAP50-95"},
        "exports": s["exports"], "intended_use": "research / education (bronchoscopy training), not clinical diagnosis",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "model_card.json").write_text(json.dumps(card, indent=2))


# --------------------------------------------------------------------------- commands

def cmd_train(args) -> None:
    from ultralytics import YOLO

    out_root = Path(args.out_dir or (KAGGLE_WORKING / RUN_NAME if ON_KAGGLE else HERE / "runs")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    data_yaml = find_data_yaml(args.data)
    preflight_check(data_yaml)
    train_yaml = make_train_yaml(data_yaml, out_root)
    env, fingerprint = environment_info(), dataset_fingerprint(data_yaml)
    names = yaml.safe_load(train_yaml.read_text())["names"]
    log("setup", f"{json.dumps(env)}")
    log("setup", f"dataset {json.dumps(fingerprint)} | output {out_root}")
    if not env["gpus"] and args.device != "cpu":
        log("setup", "WARNING: no GPU visible - in Kaggle set Accelerator to GPU (Settings -> Accelerator)")

    summaries = {}
    for base in args.models:
        model_name = f"{base}-seg" if args.task == "segment" and not base.endswith("-seg") else base
        run_dir = out_root / model_name
        if args.resume_from:  # checkpoint carried over from an earlier Kaggle session
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copy(args.resume_from, run_dir / "weights" / "last.pt")
            args.resume = True
        if run_dir.exists() and not args.resume:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        progress = out_root / f"{model_name}_progress.jsonl"
        if not args.resume:
            progress.unlink(missing_ok=True)

        hyp = build_hyperparameters(args)
        (run_dir / "run_config.json").write_text(json.dumps(
            {"args": vars(args), "model": model_name, "hyperparameters": hyp, "environment": env, "dataset": fingerprint},
            indent=2, default=str))

        model = YOLO(str(run_dir / "weights" / "last.pt") if args.resume else f"{model_name}.pt")
        add_progress_logging(model, progress, args.epochs, args.task)
        augs = endoscopy_augmentations()
        log("train", f"{model_name}: {json.dumps(hyp)}")
        t0 = time.time()
        model.train(data=str(train_yaml), project=str(out_root), name=model_name, exist_ok=True,
                    device=args.device, workers=args.workers, cache=args.cache, resume=args.resume,
                    plots=True, **({"augmentations": augs} if augs else {}), **hyp)
        train_min = round((time.time() - t0) / 60, 1)
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            sys.exit(f"training produced no {best}")

        log("train", f"done in {train_min} min - evaluating {best}")
        metrics = evaluate(best, data_yaml, train_yaml, args.imgsz, run_dir, args.task)
        exports = export_models(best, train_yaml, args.imgsz) if not args.no_export else {"pytorch": str(best)}
        export_acc = evaluate_exports(exports, train_yaml, args.imgsz, args.task)
        speed = {}
        for fmt, path in exports.items():
            try:
                speed[fmt] = benchmark_cpu(path, train_yaml, args.imgsz, args.task)
                log("cpu", f"{fmt}: {speed[fmt]['total_ms']} ms/img ({speed[fmt]['fps']} FPS)")
            except Exception as e:
                speed[fmt] = {"error": str(e)}
                log("cpu", f"{fmt} FAILED: {e}")

        epochs_run = len({json.loads(l)["epoch"] for l in open(progress) if l.strip()}) if progress.exists() else None
        summary = {"model": model_name, "task": args.task, "imgsz": args.imgsz, "epochs": args.epochs,
                   "epochs_run": epochs_run, "train_minutes": train_min, "best_weights": str(best),
                   "metrics": metrics, "exports": exports, "export_accuracy": export_acc, "cpu_benchmark": speed,
                   "hyperparameters": hyp,
                   "environment": env, "dataset": fingerprint}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_report(run_dir, summary)
        write_model_card(run_dir, summary, names)
        summaries[model_name] = summary
        log("done", f"{model_name}: report at {run_dir / 'report.md'}")
        try:
            import torch
            torch.cuda.empty_cache()  # free GPU memory before the next model
        except Exception:
            pass

    if len(summaries) > 1:
        log("compare", "test/all:")
        for name, s in summaries.items():
            t = s["metrics"].get("test/all", {})
            fps = s["cpu_benchmark"].get("openvino_int8", {}).get("fps") or s["cpu_benchmark"].get("pytorch", {}).get("fps")
            print(f"  {name:12s} mAP50 {t.get('mAP50', 0):.3f}  mAP50-95 {t.get('mAP50-95', 0):.3f}  CPU FPS {fps}")
    if ON_KAGGLE:
        log("done", f"download {shutil.make_archive(str(KAGGLE_WORKING / RUN_NAME), 'zip', out_root)}")


def cmd_export(args) -> None:
    """Re-export a trained best.pt (dynamic ONNX/OpenVINO FP32/INT8), check accuracy and CPU speed."""
    if not args.weights or not args.weights.endswith(".pt"):
        sys.exit("--weights path/to/best.pt is required for export")
    weights = Path(args.weights).resolve()
    train_yaml = make_train_yaml(find_data_yaml(args.data), weights.parent)
    exports = export_models(weights, train_yaml, args.imgsz)
    acc = evaluate_exports(exports, train_yaml, args.imgsz, args.task)
    speed = {}
    for fmt, path in exports.items():
        try:
            speed[fmt] = benchmark_cpu(path, train_yaml, args.imgsz, args.task)
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
    r = benchmark_cpu(args.weights, find_data_yaml(args.data), args.imgsz, args.task, args.n_images)
    print(f"{cpu_name()}\n{json.dumps(r, indent=2)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="train", choices=["train", "export", "benchmark"])
    p.add_argument("--data", help="path to bm_<task>/data.yaml (auto-detected if omitted)")
    p.add_argument("--task", choices=["detect", "segment"], default="detect",
                   help="detect (boxes, fastest on CPU) or segment (masks, uses the polygon labels)")
    p.add_argument("--models", nargs="+", default=["yolo26s"], help="e.g. yolo26s yolo26n (trained in sequence)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50, help="early stop after N epochs without improvement")
    p.add_argument("--imgsz", type=int, default=512, help="images are <= 640px; 512 keeps CPU inference fast")
    p.add_argument("--batch", type=int, default=32, help="total over all GPUs")
    p.add_argument("--cls-pw", type=float, default=0.5, help="class-imbalance weighting power (0 = off)")
    p.add_argument("--device", default=None, help="'0', '0,1' (both Kaggle T4s) or 'cpu'; default auto")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="ram", help="'ram', 'disk' or 'false'")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the train set (quick tests)")
    p.add_argument("--no-export", action="store_true", help="skip ONNX/OpenVINO export")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/<model>/weights/last.pt")
    p.add_argument("--resume-from", help="continue from a last.pt saved elsewhere (e.g. an earlier Kaggle session)")
    p.add_argument("--out-dir", help=f"output folder (default /kaggle/working/{RUN_NAME} or ./runs)")
    p.add_argument("--weights", help="export: best.pt; benchmark: model file or exported model folder")
    p.add_argument("--n-images", type=int, default=100, help="benchmark: number of images")
    args = p.parse_args()
    args.cache = False if str(args.cache).lower() in ("false", "0", "no") else args.cache
    if args.resume_from and not Path(args.resume_from).exists():
        p.error(f"--resume-from {args.resume_from} does not exist")
    if args.resume_from and len(args.models) > 1:
        p.error("--resume-from works with a single model")
    return args


if __name__ == "__main__":
    a = parse_args()
    {"train": cmd_train, "export": cmd_export, "benchmark": cmd_benchmark}[a.command](a)

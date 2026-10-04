#!/usr/bin/env python3
"""Run the trained BM-BronchoLC lower-airway detector on images.

    python infer.py                      # all images in inference_images/
    python infer.py path/to/frame.png
    python infer.py frame1.png frame2.jpg some_folder/ --conf 0.35 --out predictions/

For every image it:
  1. applies the same preprocessing as training (scope view, burned-in text removal, crop, grayscale)
  2. runs the detector on the CPU
  3. maps the boxes back to the ORIGINAL image, draws them and saves <out>/<name>_pred.jpg
  4. prints the detected landmarks (and writes them to <out>/predictions.json)

Default model: OpenVINO FP32 with dynamic input size
(bm_runs/yolo26s/export_dynamic/best_openvino_model). Test mAP50 0.554 (best.pt: 0.560), ~48 FPS
model-only on a Ryzen 7 7445HS. The INT8 model is faster (~59 FPS) but scores 0.531 mAP50.

Note: neighbouring bronchi (left/right main, lobar bronchi) are often confused from a single frame;
the navigation layer should use the scope's position in the airway tree to resolve them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocessing import preprocess_frame

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "bm_runs" / "yolo26s" / "export_dynamic" / "best_openvino_model"
FALLBACK_MODEL = HERE / "bm_runs" / "yolo26s" / "weights" / "best.pt"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#9085e9", "#e34948", "#4a3aa7",
          "#00a3c4", "#a35c00"]
MAX_SIDE = 640  # same as prepare_data.py


def collect_images(inputs: list[str]) -> list[Path]:
    files = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"not found: {p}")
    if not files:
        sys.exit("no images found")
    return files


def load_model(path: Path | None):
    from ultralytics import YOLO

    path = path or (DEFAULT_MODEL if DEFAULT_MODEL.exists() else FALLBACK_MODEL)
    if not path.exists():
        sys.exit(f"model not found: {path}\nrun `python train.py export --weights bm_runs/yolo26s/weights/best.pt "
                 f"--data data/bm_anatomy/data.yaml --imgsz 512` first, or pass --model")
    print(f"model: {path}")
    return YOLO(str(path), task="detect")


def detect(model, img: Image.Image, imgsz: int, conf: float, remove_text: bool) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    model_input, box, scale = preprocess_frame(img, max_side=MAX_SIDE, remove_text=remove_text)
    arr = np.repeat(np.asarray(model_input)[:, :, None], 3, axis=2)  # gray -> 3 identical channels, as in training
    t1 = time.perf_counter()
    result = model.predict(arr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    t2 = time.perf_counter()
    detections = []
    for xyxy, c, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        x0, y0, x1, y1 = xyxy
        detections.append({
            "class": result.names[int(c)], "confidence": round(float(p), 3),
            # back to original-image pixels: undo the resize, then the crop offset
            "box": [round(x0 / scale + box[0], 1), round(y0 / scale + box[1], 1),
                    round(x1 / scale + box[0], 1), round(y1 / scale + box[1], 1)],
        })
    detections.sort(key=lambda d: -d["confidence"])
    timing = {"preprocess_ms": round((t1 - t0) * 1000, 1), "model_ms": round((t2 - t1) * 1000, 1)}
    return detections, timing


def draw(img: Image.Image, detections: list[dict], names: list[str]) -> Image.Image:
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    lw = max(2, out.width // 400)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(14, out.width // 70))
    except OSError:
        font = ImageFont.load_default()
    for det in detections:
        color = COLORS[names.index(det["class"]) % len(COLORS)] if det["class"] in names else COLORS[0]
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0, y0, x1, y1], outline=color, width=lw)
        label = f"{det['class']} {det['confidence']:.2f}"
        tx0, ty0, tx1, ty1 = d.textbbox((x0, y0), label, font=font)
        ty = y0 - (ty1 - ty0) - 4 if y0 - (ty1 - ty0) - 4 > 0 else y0
        d.rectangle([x0, ty, x0 + (tx1 - tx0) + 6, ty + (ty1 - ty0) + 4], fill=color)
        d.text((x0 + 3, ty), label, fill="white", font=font)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="*", default=[str(HERE / "inference_images")],
                   help="image files and/or folders (default: the inference_images/ folder)")
    p.add_argument("--model", type=Path, help=f"model file/folder (default {DEFAULT_MODEL.relative_to(HERE)})")
    p.add_argument("--conf", type=float, default=0.25, help="minimum confidence (default 0.25)")
    p.add_argument("--imgsz", type=int, default=512, help="inference size used in training (512)")
    p.add_argument("--out", type=Path, default=HERE / "predictions", help="output folder")
    p.add_argument("--keep-text", action="store_true",
                   help="skip burned-in text removal (for clean camera frames without screen overlays)")
    args = p.parse_args()

    files = collect_images(args.images)
    model = load_model(args.model)
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    args.out.mkdir(parents=True, exist_ok=True)
    # warm-up on the first real image: OpenVINO compiles the model for each new input shape, so the
    # first frames of a given size are slow (~100 ms); a video stream of fixed size settles at ~30 ms
    with Image.open(files[0]) as first:
        first.load()
        for _ in range(2):
            detect(model, first, args.imgsz, args.conf, not args.keep_text)

    all_results = {}
    for f in files:
        img = Image.open(f)
        img.load()
        dets, timing = detect(model, img, args.imgsz, args.conf, not args.keep_text)
        draw(img, dets, names).save(args.out / f"{f.stem}_pred.jpg", quality=92)
        all_results[str(f)] = {"detections": dets, **timing}
        top = f"{dets[0]['class']} ({dets[0]['confidence']:.2f})" if dets else "nothing above threshold"
        print(f"\n{f.name}  [{img.width}x{img.height}]  preprocess {timing['preprocess_ms']} ms, model {timing['model_ms']} ms")
        print(f"  most confident landmark: {top}")
        for d in dets:
            print(f"  - {d['class']:18s} {d['confidence']:.2f}  box {d['box']}")
    (args.out / "predictions.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nannotated images and predictions.json written to {args.out}")


if __name__ == "__main__":
    main()

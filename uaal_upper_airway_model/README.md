# UAAL Upper-Airway Model

Detection model for upper-airway landmarks (nose/nostrils → epiglottis → glottis → vocal folds → trachea), trained on the UAAL dataset (`dataset/26342779/`, clinical + phantom mixed).

Target: real-time CPU inference. See `../DATASET_ANALYSIS.md` §2 and §5.1 for dataset details.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Raw analysis → preprocessing → group-level split → YOLO export → verification |
| `preprocessing.py` | Shared frame preprocessing (scope crop, grayscale, resize). Use `preprocess_frame()` at inference too |
| `train.py` | Training (YOLO26 + endoscopy augmentation), evaluation per subset, ONNX/OpenVINO export, CPU benchmark |
| `infer.py` | Run the trained model on images: `python3 infer.py frame.png` |
| `plot_results.py` | Charts for a finished run (mAP50 per split/class, losses, metrics, LR, export speed) |
| `package_kaggle.py` | Builds `dist/uaal_det_kaggle.zip` (dataset + code) for a Kaggle dataset upload |
| `requirements.txt` | Dependencies for data preparation |

## Data preparation

```bash
python3 prepare_data.py            # analyze + prepare + verify (~1.5 min)
python3 prepare_data.py --help     # options: --seed, --ratios, --max-side, --dedup-threshold, --color, --no-crop, --split-strategy
```

Output in `data/uaal_det/` (not tracked):

```
data.yaml                    train/val/test, 9 classes (mixed clinical + phantom)
data_clinical.yaml           evaluation-only views per subset
data_phantom.yaml
images/{train,val,test}/     grayscale PNG, cropped to the scope view, longest side ≤ 1024
labels/{train,val,test}/     YOLO txt: class cx cy w h (normalised)
manifest.csv                 one row per image: source file, group, split, crop box, sizes, classes
dropped.csv                  removed near-duplicate frames
reports/                     raw_analysis.md, prepared_report.md, charts, verify.txt
previews/                    sample images with boxes drawn
config_used.json             all parameters + group→split assignment
```

## Training notes

- `fliplr=0.0` and `flipud=0.0`: flips swap left/right vocal folds.
- Report metrics per subset with `data_clinical.yaml` / `data_phantom.yaml`.

## Training on Kaggle

1. `python3 package_kaggle.py` → upload `dist/uaal_det_kaggle.zip` as a new Kaggle dataset.
2. In a notebook (Accelerator **GPU T4 x2** or **P100**, Internet **on**), add the dataset and run:

```python
!pip install -q -U "ultralytics>=8.4" "albumentations>=2.0" openvino nncf onnx onnxruntime onnxslim
!python /kaggle/input/<dataset-slug>/code/train.py                      # yolo26s, 150 epochs, 640px
# !python /kaggle/input/<dataset-slug>/code/train.py --models yolo26s yolo26n   # train and compare both
```

3. Download `/kaggle/working/uaal_runs.zip` (weights, ONNX/OpenVINO exports, `report.md`, `summary.json`, plots).

Progress prints one `[progress]` line per epoch (also saved to `<model>_progress.jsonl`). An interrupted run continues with `--resume`.

## CPU benchmark on this laptop

```bash
python3 train.py benchmark --weights runs/yolo26s/weights/best_int8_openvino_model --data data/uaal_det/data.yaml
```

## Using the trained model

Model to use: **`uaal_runs/yolo26s/export_dynamic/best_openvino_model`** (OpenVINO FP32, dynamic input).
It has the same accuracy as `best.pt` (test mAP50 0.723), at ~35 FPS model-only on a Ryzen 7 7445HS (~23 FPS with preprocessing).

```bash
python3 infer.py path/to/frame.png                 # annotated image + detections -> predictions/
python3 infer.py folder_of_frames/ --conf 0.35     # several images, higher threshold
python3 infer.py frame.png --model uaal_runs/yolo26s/export_dynamic/best_int8_openvino_model   # faster, slightly less accurate
python3 plot_results.py                            # charts -> uaal_runs/yolo26s/plots/
```

Needs `pip install ultralytics openvino` (plus the packages in requirements.txt).

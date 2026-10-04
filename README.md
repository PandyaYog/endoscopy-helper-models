# AI-Assisted Bronchoscopy Training Platform: Vision Models

This repository contains the computer-vision part of a bronchoscopy **training platform**. A trainee moves a bronchoscope through a physical airway trainer (ESP32, ultrasonic level sensors, touch and buzzer feedback). The AI identifies the anatomical landmark visible in the camera frame, so a navigation layer can follow the route the trainee takes and flag wrong turns.

There are two detection models, one for each part of the airway. Both are small enough to run in **real time on a laptop CPU**.

> Intended use: education and research (simulation training). These models are **not** clinically validated and must not be used for diagnosis.

## The two models

| | Upper-airway model | Lower-airway model |
|---|---|---|
| Folder | [`uaal_upper_airway_model/`](uaal_upper_airway_model/) | [`bm_broncholc_lower_airway_model/`](bm_broncholc_lower_airway_model/) |
| Purpose | Recognises the entry route: **nose → nostrils → epiglottis → glottis → vocal folds → trachea** | Recognises the bronchial tree: **vocal cords → trachea → main carina → main, intermediate and lobar bronchi** |
| Trained on | UAAL dataset: clinical videos + airway-phantom images (phantom = closest to a physical trainer) | BM-BronchoLC dataset: 208 patients' bronchoscopy videos (anatomy labels) |
| Classes | 9: epiglottis, glottis, glottic_aperture, left/right_vocal_fold, trachea, channel, nose, nostril | 11: vocal_cords, trachea, main_carina, right/left_main_bronchus, intermediate_bronchus, right superior/middle/inferior and left superior/inferior lobar bronchus |
| Model | YOLO26s detector, 640 px input | YOLO26s detector, 512 px input |
| **Test mAP50** | **0.723** (clinical 0.787, phantom 0.711) | **0.554** (lung cancer 0.605, non-cancer 0.535) |
| Speed (Ryzen 7 7445HS, OpenVINO FP32) | ~35 FPS model-only, ~23 FPS including preprocessing | ~48 FPS model-only |

The two models meet at the **vocal cords → trachea** transition, where the navigation layer switches from one model to the other.

**What the numbers mean.** The upper-airway model recognises the glottic structures reliably (epiglottis 0.96, glottis 0.91, glottic aperture 0.85 test mAP50). The lower-airway model finds bronchial openings but often confuses *neighbouring* bronchi (e.g. left vs right main bronchus), because in a single frame they look alike. That is why the platform uses the trainee's position in the airway tree (navigation context) to resolve the exact bronchus, rather than relying on single frames.

## Repository layout

```
├── README.md                          this file
├── requirements.txt                   packages for inference and plots
├── DATASET_ANALYSIS.md                full analysis of both datasets (counts, quirks, decisions)
├── AI_Assisted_Bronchoscopy_Project_Document.docx   project concept document
├── dataset/README.md                  where to download the raw datasets (not included)
│
├── uaal_upper_airway_model/           upper-airway model
│   ├── results-ppt/                   ★ selected result charts (see below)
│   ├── uaal_runs/                     full training run: reports, curves, evaluations, models
│   ├── inference_images/              15 test-set frames the model handles well, for demos
│   ├── infer.py                       run the model on images
│   ├── preprocessing.py               frame preprocessing shared by training and inference
│   ├── prepare_data.py                dataset analysis, cleaning, split, YOLO export
│   ├── train.py                       training / evaluation / export (run on Kaggle GPUs)
│   ├── plot_results.py                creates the charts in uaal_runs/yolo26s/plots/
│   ├── package_kaggle.py              bundles dataset + code for a Kaggle upload
│   ├── training-logs.md               complete console log of the Kaggle training run
│   └── README.md                      model-specific details
│
└── bm_broncholc_lower_airway_model/   lower-airway model (same structure)
    ├── results-ppt/                   ★ selected result charts
    ├── bm_runs/                       full training run
    ├── inference_images/              22 test-set frames for demos
    ├── results.md                     complete console log of the Kaggle training run
    └── ...
```

## Results: start with the `results-ppt` folders

The charts selected for presentation are in **`uaal_upper_airway_model/results-ppt/`** and **`bm_broncholc_lower_airway_model/results-ppt/`**. Each image shows the following:

| Image | What it shows | In folder |
|---|---|---|
| `map50_by_split.png` | mAP50 on the **train, validation and test** sets, overall and per subset (clinical / phantom). The train-vs-test gap shows how well the model generalises to unseen recordings | UAAL |
| `map50_per_class.png` | mAP50 of **every class** on train / validation / test, showing which landmarks are recognised reliably and which are weak | UAAL, BM |
| `loss_curves.png` | Training vs validation **loss per epoch** (box, classification, L1), with the selected best epoch marked. Diverging curves indicate overfitting | UAAL, BM |
| `val_metrics_curves.png` | Validation **mAP50, mAP50-95, precision and recall per epoch**: how quickly the model learned and where it plateaued | UAAL, BM |
| `results.png` | Ultralytics' standard training summary: all losses and metrics per epoch in one grid | UAAL, BM |
| `BoxPR_curve.png` | **Precision-recall curve** per class on the validation set (best model). The area under each curve is that class's AP | UAAL, BM |
| `BoxF1_curve.png` | **F1 score vs confidence threshold** per class (validation set), useful for choosing the `--conf` threshold at inference | BM |
| `confusion_matrix_normalized.png` | For each true class (columns), which class the model predicted (rows), including missed objects and false detections (*background*), on the validation set | UAAL |
| `labels.jpg` | Training-label statistics: number of boxes per class, and where in the frame and how large the boxes are | UAAL |
| `export_speed_accuracy.png` | Each exported model format (PyTorch, ONNX, OpenVINO FP32/INT8): **CPU speed in FPS** on the development laptop and test accuracy | UAAL, BM |

mAP50 = mean average precision at 50% box overlap (1.0 = perfect), the main detection accuracy measure.

## More detail: the run folders

Everything produced by training is in **`uaal_upper_airway_model/uaal_runs/yolo26s/`** and **`bm_broncholc_lower_airway_model/bm_runs/yolo26s/`**:

| File / folder | Content |
|---|---|
| `report.md` | Training summary: accuracy per split and subset, per-class scores, CPU speed |
| `summary.json`, `model_card.json`, `run_config.json`, `args.yaml` | Machine-readable metrics, model card (classes, input, intended use), exact training configuration and library versions |
| `plots/` | All charts made by `plot_results.py`, plus `split_metrics.json` (the numbers behind them) |
| `eval_val_*`, `eval_test_*` | Separate evaluations on the validation and test sets, overall and per subset, each with its confusion matrix and PR/F1 curves |
| `results.csv`, `*_progress.jsonl` | Per-epoch metrics |
| `train_batch*.jpg`, `val_batch*_labels/pred.jpg` | Example augmented training batches, and validation labels vs predictions |
| `weights/best.pt` | Trained PyTorch model |
| `export_dynamic/best_openvino_model/` | **Model used by `infer.py`**: OpenVINO FP32, variable input size, same accuracy as `best.pt` |
| `export_dynamic/best_int8_openvino_model/` | Faster INT8 version (about 20–30% faster, 1–4 mAP points lower) |
| `export_dynamic/export_report.json` | Accuracy and CPU speed of every exported format |

The full console logs of the Kaggle training runs are in `uaal_upper_airway_model/training-logs.md` and `bm_broncholc_lower_airway_model/results.md`.

## Running inference

### Setup (once)

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### What `infer.py` does with an image

1. **Preprocesses the frame exactly like the training data.**
   - Finds the scope view inside the screen capture and crops to it, removing the black border, patient text, icons and picture-in-picture thumbnails.
   - The lower-airway model also removes burned-in text and red warning banners drawn over the view, filling those pixels from the surrounding tissue.
   - Converts the frame to grayscale and resizes it.
2. **Runs the detector on the CPU** with the OpenVINO model.
3. **Maps the boxes back to the original image**, undoing the crop and resize, and draws them with class name and confidence.
4. **Saves and reports.** Writes `<name>_pred.jpg` for each image and a `predictions.json` with every detection (class, confidence, box in original pixels, timing). It also prints the most confident landmark and all detections to the terminal.

### Commands

Run them from the model's folder:

```bash
cd uaal_upper_airway_model            # or: cd bm_broncholc_lower_airway_model

# No image given: uses every image in the model's inference_images/ folder
python3 infer.py

# One image
python3 infer.py path/to/frame.png

# Several images and/or folders, higher confidence threshold, custom output folder
python3 infer.py frame1.png frame2.jpg some_folder/ --conf 0.4 --out my_predictions
```

Results are written to `predictions/` by default.

Useful options:
- `--conf`: minimum confidence (default 0.25; **0.4** gives cleaner pictures).
- `--model`: use another model, e.g. `uaal_runs/yolo26s/export_dynamic/best_int8_openvino_model` for more speed.
- Frames from a clean camera without screen overlays:
  - upper-airway model: `--no-crop`
  - lower-airway model: `--keep-text`

Example output:

```
pt_AI019_0030.PNG  [1920x1080]  preprocess 17.7 ms, model 31.1 ms
  most confident landmark: epiglottis (0.80)
  - epiglottis         0.80  box [573.5, 675.2, 1170.6, 869.1]
  - glottis            0.77  box [617.1, 604.6, 903.9, 721.4]
```

**Which model is used:** by default, `export_dynamic/best_openvino_model` in each run folder (OpenVINO FP32, variable input size). It is as accurate as the PyTorch weights and 2–3× faster on a CPU. If it is missing, `infer.py` falls back to `weights/best.pt`.

The `inference_images/` folders contain raw frames from the **test** set (never seen in training) on which the model finds every landmark correctly. Their `index.csv` lists the expected landmarks. They are good for demos, but chosen as easy cases; see the results above for typical accuracy.

## Reproducing the models

1. Download the datasets into `dataset/` (see [`dataset/README.md`](dataset/README.md)).
2. In each model folder, run `python3 prepare_data.py`. It analyses, cleans and splits the data, then writes `data/`.
3. Run `python3 package_kaggle.py`, upload `dist/*.zip` as a Kaggle dataset, and run `train.py` on a GPU notebook. The model READMEs contain the exact steps.
4. Run `python3 train.py export --weights <best.pt> --data <data.yaml>` to create the dynamic OpenVINO models, then `python3 plot_results.py` to create the charts.

[`DATASET_ANALYSIS.md`](DATASET_ANALYSIS.md) explains the dataset problems found and how they were handled: burned-in text, rotated boxes in UAAL, duplicate polygons in BM-BronchoLC, patient-level splits, and more.

## Datasets and credits

- **UAAL**: *Upper Airway Anatomical Landmark Dataset for Automated Bronchoscopy and Intubation*, Scientific Data (2025), doi:10.1038/s41597-025-06169-0; data doi:10.6084/m9.figshare.26342779.
- **BM-BronchoLC**: *A rich bronchoscopy dataset for anatomical landmarks and lung cancer lesion recognition*, Scientific Data (2024), doi:10.1038/s41597-024-03145-y.
- Models trained with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

The `inference_images/` folders contain a small number of frames from these datasets for demonstration; all rights remain with the dataset authors.

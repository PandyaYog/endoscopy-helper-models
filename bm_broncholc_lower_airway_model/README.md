# BM-BronchoLC Lower-Airway Model

Landmark model for the lower airway (vocal cords → trachea → carina → main, intermediate and lobar bronchi), trained on BM-BronchoLC (`dataset/Lung_cancer/`, `dataset/Non_lung_cancer/`). Lesion detection is a later, optional phase; the same pipeline already supports it.

Target: real-time CPU inference. See `../DATASET_ANALYSIS.md` §3 and §5.2.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Raw analysis → cleaning → patient-level split → YOLO + COCO export → verification |
| `preprocessing.py` | Scope-view detection, burned-in text removal, crop, grayscale. Use `preprocess_frame()` at inference too |
| `train.py` | Pre-flight check → training (YOLO26 + endoscopy augmentation) → evaluation per cohort → ONNX/OpenVINO export → CPU benchmark → report + model card |
| `plot_results.py` | Charts for a finished run: `python3 plot_results.py` → `bm_runs/yolo26s/plots/` |
| `package_kaggle.py` | Builds `dist/bm_anatomy_kaggle.zip` for a Kaggle dataset upload |
| `requirements.txt` | Dependencies for data preparation |

## Data preparation

```bash
python3 prepare_data.py                          # anatomy task: analyze + prepare + verify (~4 min)
python3 prepare_data.py --task lesion_binary     # later: one "abnormal" class, same patient split
python3 prepare_data.py --help                   # --task, --seed, --ratios, --max-side, --dedup-threshold, --keep-text
```

What it does, in order:

1. **Load and clean annotations.** Maps labels by name, fixes misspelled class names, removes exact duplicate polygons (860 of 6,225).
2. **Clean each frame.**
   - Finds the scope view with column/row brightness profiles plus a convex hull.
   - Blacks out everything outside the view (side text, icons, anonymisation blur).
   - Fills burned-in text inside the view and red warning banners from neighbouring tissue.
   - Crops to the view and converts to grayscale.
3. **Filter images.** Drops blank or broken frames, near-duplicate frames of the same patient, and (for the anatomy task) lesion-only images, whose anatomy is not labelled.
4. **Rename** to `bm_<lc|nlc>_p<patient>_f<frame>`. `manifest.csv` maps every file back to its original UUIDs.
5. **Split by patient** (70/15/15) per cohort, balancing image counts and all 18 classes. Patients that share near-identical frames are kept together.
6. **Export.**
   - YOLO polygon labels, usable for detection or segmentation.
   - COCO JSON with all 18 classes.
   - Per-cohort evaluation yamls (`data_lc.yaml`, `data_nlc.yaml`).
   - Reports, previews, and `dropped.csv`.

Output: `data/bm_<task>/` (not tracked).

## Training notes

- `fliplr=0.0`, `flipud=0.0`: flips swap left/right bronchi.
- For a clean camera feed without screen overlays, use `preprocess_frame(img, remove_text=False)`.

## Training on Kaggle

1. `python3 package_kaggle.py` → upload `dist/bm_anatomy_kaggle.zip` as a Kaggle dataset.
2. Notebook: Accelerator **GPU T4 x2**, Internet **on**, add the dataset, then:

```python
!pip install -q -U "ultralytics>=8.4" "albumentations>=2.0" openvino nncf onnx onnxruntime onnxslim
!find /kaggle/input -name train.py -path "*code*"        # shows the exact path to use below
!python <path-from-find>/train.py                          # detection, yolo26s, 512px, up to 200 epochs
```

3. Download `/kaggle/working/bm_runs.zip`.

Options: `--task segment` (masks), `--models yolo26s yolo26n` (train and compare), `--resume` / `--resume-from last.pt`.

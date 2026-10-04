# BM-BronchoLC lower-airway detect model - training report

- Model **yolo26s**, image size 512, epochs run 74 (max 200), training time 20.2 min
- Dataset: `/kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/data.yaml` (manifest 6d3d82a01d8d5a4e, prepared 2026-10-05 02:27:22)
- Best checkpoint: `/kaggle/working/bm_runs/yolo26s/weights/best.pt`

## Accuracy

| Split / cohort | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val/all | 0.650 | 0.570 | 0.632 | 0.306 |
| val/lc | 0.721 | 0.535 | 0.632 | 0.310 |
| val/nlc | 0.655 | 0.575 | 0.645 | 0.317 |
| test/all | 0.635 | 0.516 | 0.561 | 0.278 |
| test/lc | 0.608 | 0.607 | 0.615 | 0.298 |
| test/nlc | 0.692 | 0.463 | 0.543 | 0.281 |

## Per-class mAP50-95 (test, all)

| Class | mAP50-95 |
|---|---|
| vocal_cords | 0.378 |
| right_superior_lobar_bronchus | 0.358 |
| right_main_bronchus | 0.310 |
| left_main_bronchus | 0.306 |
| left_superior_lobar_bronchus | 0.304 |
| right_inferior_lobar_bronchus | 0.295 |
| intermediate_bronchus | 0.288 |
| trachea | 0.266 |
| left_inferior_lobar_bronchus | 0.232 |
| right_middle_lobar_bronchus | 0.166 |
| main_carina | 0.157 |

## CPU speed (Intel(R) Xeon(R) CPU @ 2.00GHz)

| Format | Pre (ms) | Inference (ms) | Post (ms) | Total (ms) | FPS |
|---|---|---|---|---|---|
| pytorch | 1.9 | 102.9 | 0.9 | 105.7 | 9.5 |
| onnx | 2.2 | 82.0 | 1.2 | 85.4 | 11.7 |
| openvino_fp32 | 2.2 | 78.8 | 1.2 | 82.2 | 12.2 |
| openvino_int8 | 2.1 | 64.1 | 1.2 | 67.4 | 14.8 |

Kaggle CPUs are shared server cores: re-run `python train.py benchmark` on the target machine.

Plots (confusion matrix, PR curves, training curves) are in the run folder and `eval_*` folders.

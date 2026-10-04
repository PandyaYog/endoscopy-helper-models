# UAAL upper-airway detector - training report

- Model: **yolo26s**, image size 640, epochs run 165 (max 150), training time 63.8 min
- Best checkpoint: `/kaggle/working/uaal_runs/yolo26s/weights/best.pt`

## Accuracy

| Split / subset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| val/all | 0.795 | 0.771 | 0.806 | 0.472 |
| val/clinical | 0.883 | 0.812 | 0.867 | 0.527 |
| val/phantom | 0.675 | 0.730 | 0.706 | 0.416 |
| test/all | 0.683 | 0.731 | 0.723 | 0.439 |
| test/clinical | 0.750 | 0.779 | 0.784 | 0.491 |
| test/phantom | 0.640 | 0.746 | 0.716 | 0.428 |

## Per-class mAP50-95 (test, all)

| Class | mAP50-95 |
|---|---|
| epiglottis | 0.657 |
| glottis | 0.598 |
| glottic_aperture | 0.614 |
| left_vocal_fold | 0.520 |
| right_vocal_fold | 0.483 |
| trachea | 0.364 |
| channel | 0.090 |
| nose | 0.365 |
| nostril | 0.256 |

## CPU speed (Intel(R) Xeon(R) CPU @ 2.00GHz)

| Format | Pre (ms) | Inference (ms) | Post (ms) | Total (ms) | FPS |
|---|---|---|---|---|---|
| pytorch | 2.6 | 120.8 | 1.0 | 124.4 | 8.0 |
| onnx | 3.5 | 117.4 | 1.3 | 122.1 | 8.2 |
| openvino_fp32 | 3.3 | 116.1 | 1.4 | 120.8 | 8.3 |
| openvino_int8 | 3.3 | 87.6 | 1.2 | 92.1 | 10.9 |

Kaggle CPUs are slow, shared server cores; re-run `python train.py benchmark` on the target machine for real numbers.

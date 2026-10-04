# Inference demo images

22 raw frames (original resolution, with the screen border and text still present) from the **test split**, which the model never saw during training or model selection.
The trained model finds every labelled landmark in each of them (IoU ≥ 0.5) with no wrong detections at confidence ≥ 0.4.

`index.csv` lists, per image: subset, original id, expected landmarks, the lowest confidence among them, and the mean box overlap with the label.

Run from the model folder:

```bash
python3 infer.py inference_images --conf 0.4 --out predictions
```

Selection is biased toward easy frames by design. For honest overall accuracy, see `report.md` and the plots in the run folder.

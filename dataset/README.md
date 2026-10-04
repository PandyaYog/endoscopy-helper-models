# Datasets (not included in this repository)

The raw datasets are large (~8 GB) and are distributed by their authors. Download them and place them here:

```
dataset/
├── 26342779/                       UAAL (figshare 10.6084/m9.figshare.26342779)
│   ├── coco_ins_clinical.zip
│   ├── coco_ins_phantom.zip
│   └── annotation visualization.zip
├── Lung_cancer/Lung_cancer/        BM-BronchoLC (Scientific Data, doi:10.1038/s41597-024-03145-y)
│   ├── imgs/  labels.json  objects.json  annotation.json
└── Non_lung_cancer/Non_lung_cancer/
    └── imgs/  labels.json  objects.json  annotation.json
```

Then rebuild the training-ready datasets with `python3 prepare_data.py` inside each model folder.

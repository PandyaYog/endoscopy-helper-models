Creating new Ultralytics Settings v0.0.8 file ✅ 
View Ultralytics Settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.json'
Update Settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'. For help see https://docs.ultralytics.com/usage/settings.
[data] several datasets found, using the first (pass --data to choose): ['/kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/data.yaml', '/kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/data.yaml']
[data] pre-flight OK: {"train": {"images": 1481, "instances": 3041}, "val": {"images": 321, "instances": 632}, "test": {"images": 325, "instances": 641}} | classes: ['vocal_cords', 'trachea', 'main_carina', 'right_main_bronchus', 'left_main_bronchus', 'intermediate_bronchus', 'right_superior_lobar_bronchus', 'right_middle_lobar_bronchus', 'right_inferior_lobar_bronchus', 'left_superior_lobar_bronchus', 'left_inferior_lobar_bronchus']
[setup] {"python": "3.13.15", "platform": "Linux-6.18.48+-x86_64-with-glibc2.39", "cpu": "Intel(R) Xeon(R) CPU @ 2.00GHz", "ultralytics": "8.4.173", "torch": "2.11.0+cu128", "albumentations": "2.0.8", "openvino": "2026.4.1-22982-e213a147257-releases/2026/4", "onnxruntime": "1.30.0", "gpus": ["Tesla T4", "Tesla T4"]}
[setup] dataset {"data_yaml": "/kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/data.yaml", "manifest_sha256": "6d3d82a01d8d5a4e", "created": "2026-10-05 02:27:22", "task": "anatomy", "seed": 42, "ratios": [0.7, 0.15, 0.15], "max_side": 640, "remove_text": true} | output /kaggle/working/bm_runs
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt to 'yolo26s.pt': 100% ━━━━━━━━━━━━ 19.5MB 58.3MB/s 0.3s 0.3s<0.1s
[train] yolo26s: {"epochs": 200, "imgsz": 512, "batch": 32, "patience": 50, "optimizer": "auto", "cos_lr": true, "seed": 42, "deterministic": false, "fraction": 1.0, "degrees": 15.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0, "fliplr": 0.0, "flipud": 0.0, "mosaic": 1.0, "close_mosaic": 20, "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.4, "cls_pw": 0.5}
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
                                                        CUDA:1 (Tesla T4, 14912MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, augmentations=[{'__version__': '2.0.8', 'transform': {'__class_fullname__': 'OneOf', 'p': 0.25, 'transforms': [{'__class_fullname__': 'MotionBlur', 'p': 0.5, 'allow_shifted': True, 'angle_range': (0.0, 360.0), 'blur_limit': (3, 9), 'direction_range': (-1.0, 1.0)}, {'__class_fullname__': 'GaussianBlur', 'p': 0.5, 'blur_limit': (3, 7), 'sigma_limit': (0.5, 3.0)}, {'__class_fullname__': 'Defocus', 'p': 0.5, 'alias_blur': (0.1, 0.5), 'radius': (1, 3)}]}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'GaussNoise', 'p': 0.15, 'mean_range': (0.0, 0.0), 'noise_scale_factor': 1.0, 'per_channel': True, 'std_range': (0.02, 0.06)}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'CLAHE', 'p': 0.2, 'clip_limit': (1.0, 3.0), 'tile_grid_size': (8, 8)}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'RandomBrightnessContrast', 'p': 0.3, 'brightness_by_max': True, 'brightness_limit': (-0.2, 0.2), 'contrast_limit': (-0.25, 0.25), 'ensure_safe_range': False}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'RandomGamma', 'p': 0.2, 'gamma_limit': (80.0, 120.0)}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'ImageCompression', 'p': 0.2, 'compression_type': 'jpeg', 'quality_range': (50, 95)}}], auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=ram, cfg=None, channels_last=None, classes=None, close_mosaic=20, cls=0.5, cls_pw=0.5, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/kaggle/working/bm_runs/data_abs.yaml, degrees=15.0, deterministic=False, device=0,1, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=200, erasing=0.4, exist_ok=True, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, imgsz=512, iou=0.7, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26s.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=yolo26s, nbs=64, nms=None, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=50, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/kaggle/working/bm_runs, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/kaggle/working/bm_runs/yolo26s, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=42, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
Downloading https://ultralytics.com/assets/Arial.ttf to '/root/.config/Ultralytics/Arial.ttf': 100% ━━━━━━━━━━━━ 755.1KB 5.1MB/s 0.1s 0.1s<0.2s
Overriding model.yaml nc=80 with nc=11

                   from  n    params  module                                       arguments                     
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]                 
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]     
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]    
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]           
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5, 3, True]        
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1    477184  ultralytics.nn.modules.block.C3k2            [768, 256, 1, True]           
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    136192  ultralytics.nn.modules.block.C3k2            [512, 128, 1, True]           
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1    378880  ultralytics.nn.modules.block.C3k2            [384, 256, 1, True]           
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1843712  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True, 0.5, True]
 23        [16, 19, 22]  1    940378  ultralytics.nn.modules.head.Detect           [11, 1, True, [128, 256, 512]]
YOLO26s summary: 260 layers, 9,956,378 parameters, 9,956,378 gradients, 22.8 GFLOPs

Transferred 696/708 items from pretrained weights
DDP: debug command /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 26808 /root/.config/Ultralytics/DDP/_temp_0wr9orrr138915629010288.py
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
                                                        CUDA:1 (Tesla T4, 14912MiB)
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: downloading yolo26n.pt for AMP checks (one-time, not used for training)...
Downloading https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt to 'weights/yolo26n.pt': 100% ━━━━━━━━━━━━ 5.3MB 21.4MB/s 0.2s.2s<0.1s8s
AMP: checks passed ✅
WARNING ⚠️ train: Slow image access detected (ping: 0.1±0.2 ms, read: 21.8±4.7 MB/s, size: 69.1 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips
train: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels/train... 1481 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1481/1481 631.1it/s 2.3s0.0s
WARNING ⚠️ train: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels is not writable, cache not saved.
train: Caching images (0.8GB RAM): 100% ━━━━━━━━━━━━ 1481/1481 500.7it/s 3.0s0.0s
albumentations: OneOf([
  MotionBlur(p=0.5, allow_shifted=True, angle_range=(0.0, 360.0), blur_limit=(3, 9), direction_range=(-1.0, 1.0)),
  GaussianBlur(p=0.5, blur_limit=(3, 7), sigma_limit=(0.5, 3.0)),
  Defocus(p=0.5, alias_blur=(0.1, 0.5), radius=(1, 3)),
], p=0.25), GaussNoise(p=0.15, mean_range=(0.0, 0.0), noise_scale_factor=1.0, per_channel=True, std_range=(0.02, 0.06)), CLAHE(p=0.2, clip_limit=(1.0, 3.0), tile_grid_size=(8, 8)), RandomBrightnessContrast(p=0.3, brightness_by_max=True, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.25, 0.25), ensure_safe_range=False), RandomGamma(p=0.2, gamma_limit=(80.0, 120.0)), ImageCompression(p=0.2, compression_type='jpeg', quality_range=(50, 95))
WARNING ⚠️ val: Slow image access detected (ping: 0.1±0.2 ms, read: 24.7±3.6 MB/s, size: 67.7 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels/val... 321 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 321/321 682.7it/s 0.5s0.1s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels is not writable, cache not saved.
val: Caching images (0.2GB RAM): 100% ━━━━━━━━━━━━ 321/321 424.9it/s 0.8s0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and determining best 'optimizer' and 'lr0' automatically... 
optimizer: AdamW(lr=0.000667, momentum=0.9) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias(decay=0.0)
Class weights: [      0.925       1.657       0.862       0.838       0.766       1.224       0.964       0.936       0.957       1.002       0.869]
Plotting labels to /kaggle/working/bm_runs/yolo26s/labels.jpg... 
Using 1481 train, 321 val images for fraction=1.0 at imgsz=512
Using 4 dataloader workers
Logging results to /kaggle/working/bm_runs/yolo26s
Starting training for 200 epochs...

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      1/200       2.9G      1.769      7.102    0.02639         20        512: 100% ━━━━━━━━━━━━ 47/47 1.5it/s 31.4s0.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.5s/it 15.3s0.9ss
                   all        321        632       0.12      0.156     0.0926     0.0327
[progress] epoch 1/200 | mAP50 0.093 | mAP50-95 0.033 | P 0.120 R 0.156 | 1.4 min elapsed, ~283.2 min left (upper bound, early stopping may end sooner)
/usr/local/lib/python3.13/dist-packages/ray/train/_internal/session.py:683: UserWarning: `get_trial_id` is meant to only be called inside a function that is executed by a Tuner or Trainer. Returning `None`.
  warnings.warn(

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      2/200      4.01G      1.562      3.839    0.02216         23        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.5s0.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632      0.278      0.338      0.249      0.117
[progress] epoch 2/200 | mAP50 0.249 | mAP50-95 0.117 | P 0.278 R 0.338 | 1.7 min elapsed, ~166.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      3/200      4.01G      1.602      3.083     0.0236         22        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.8it/s 1.2s0.2s
                   all        321        632      0.464      0.366      0.335      0.153
[progress] epoch 3/200 | mAP50 0.336 | mAP50-95 0.153 | P 0.464 R 0.366 | 1.9 min elapsed, ~126.8 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      4/200      4.01G        1.6      2.731    0.02295         25        512: 100% ━━━━━━━━━━━━ 47/47 3.4it/s 13.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.9it/s 1.2s0.2s
                   all        321        632      0.301      0.407      0.304      0.121
[progress] epoch 4/200 | mAP50 0.303 | mAP50-95 0.121 | P 0.300 R 0.406 | 2.2 min elapsed, ~107.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      5/200      4.01G       1.62      2.585    0.02259         23        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.433      0.456       0.41      0.186
[progress] epoch 5/200 | mAP50 0.410 | mAP50-95 0.186 | P 0.433 R 0.456 | 2.4 min elapsed, ~95.3 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      6/200      4.01G      1.575       2.41    0.02271         22        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.419       0.46      0.412      0.191
[progress] epoch 6/200 | mAP50 0.412 | mAP50-95 0.191 | P 0.419 R 0.460 | 2.7 min elapsed, ~87.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      7/200      4.01G       1.56      2.315    0.02198         16        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632      0.525      0.487        0.5      0.211
[progress] epoch 7/200 | mAP50 0.499 | mAP50-95 0.211 | P 0.525 R 0.487 | 3.0 min elapsed, ~81.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      8/200      4.01G      1.591      2.193    0.02251         26        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.7it/s 1.3s0.2s
                   all        321        632      0.505      0.438      0.436      0.191
[progress] epoch 8/200 | mAP50 0.436 | mAP50-95 0.191 | P 0.505 R 0.438 | 3.2 min elapsed, ~77.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      9/200      4.01G      1.576      2.199    0.02238         29        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.472       0.56      0.497      0.228
[progress] epoch 9/200 | mAP50 0.497 | mAP50-95 0.228 | P 0.472 R 0.560 | 3.5 min elapsed, ~73.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     10/200      4.01G      1.558      2.164    0.02205         26        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.534      0.546      0.526      0.243
[progress] epoch 10/200 | mAP50 0.526 | mAP50-95 0.243 | P 0.534 R 0.546 | 3.7 min elapsed, ~70.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     11/200      4.01G       1.57      2.112    0.02269         34        512: 100% ━━━━━━━━━━━━ 47/47 3.4it/s 13.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632      0.527      0.552      0.535      0.241
[progress] epoch 11/200 | mAP50 0.535 | mAP50-95 0.241 | P 0.527 R 0.552 | 4.0 min elapsed, ~68.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     12/200      4.01G      1.557      2.094    0.02235         23        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s0.2s
                   all        321        632      0.543      0.522      0.526      0.256
[progress] epoch 12/200 | mAP50 0.526 | mAP50-95 0.256 | P 0.543 R 0.522 | 4.2 min elapsed, ~66.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     13/200      4.01G      1.548      2.103    0.02198         20        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s0.2s
                   all        321        632      0.612       0.56      0.587      0.283
[progress] epoch 13/200 | mAP50 0.587 | mAP50-95 0.283 | P 0.612 R 0.560 | 4.5 min elapsed, ~64.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     14/200      4.01G      1.532      2.024    0.02142         22        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s0.2s
                   all        321        632       0.59      0.544      0.564      0.246
[progress] epoch 14/200 | mAP50 0.564 | mAP50-95 0.246 | P 0.590 R 0.544 | 4.7 min elapsed, ~62.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     15/200      4.01G      1.523      1.977     0.0212         18        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632       0.55      0.533      0.546      0.254
[progress] epoch 15/200 | mAP50 0.546 | mAP50-95 0.255 | P 0.550 R 0.533 | 5.0 min elapsed, ~61.3 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     16/200      4.01G      1.525      1.922    0.02234         18        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.596      0.537      0.574      0.265
[progress] epoch 16/200 | mAP50 0.574 | mAP50-95 0.265 | P 0.596 R 0.537 | 5.2 min elapsed, ~60.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     17/200      4.01G      1.515      1.895    0.02203         26        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.0s0.2s
                   all        321        632      0.596      0.494      0.537      0.244
[progress] epoch 17/200 | mAP50 0.537 | mAP50-95 0.244 | P 0.596 R 0.494 | 5.5 min elapsed, ~58.9 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     18/200      4.01G       1.51      1.915    0.02157         27        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.554      0.534       0.56      0.264
[progress] epoch 18/200 | mAP50 0.560 | mAP50-95 0.264 | P 0.554 R 0.534 | 5.7 min elapsed, ~57.9 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     19/200      4.01G      1.524      1.878    0.02184         19        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s0.2s
                   all        321        632      0.551      0.587      0.573       0.27
[progress] epoch 19/200 | mAP50 0.573 | mAP50-95 0.270 | P 0.551 R 0.587 | 6.0 min elapsed, ~56.9 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     20/200      4.01G      1.533      1.854    0.02184         20        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.606      0.577      0.587      0.278
[progress] epoch 20/200 | mAP50 0.587 | mAP50-95 0.278 | P 0.606 R 0.578 | 6.2 min elapsed, ~56.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     21/200      4.01G      1.523      1.863    0.02112         12        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.599      0.479      0.522      0.253
[progress] epoch 21/200 | mAP50 0.522 | mAP50-95 0.253 | P 0.599 R 0.479 | 6.5 min elapsed, ~55.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     22/200      4.01G      1.526      1.844    0.02174         17        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s0.2s
                   all        321        632      0.609      0.603      0.608      0.287
[progress] epoch 22/200 | mAP50 0.608 | mAP50-95 0.287 | P 0.609 R 0.603 | 6.7 min elapsed, ~54.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     23/200      4.01G      1.526      1.827    0.02212         18        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632       0.51      0.564       0.53      0.241
[progress] epoch 23/200 | mAP50 0.530 | mAP50-95 0.241 | P 0.510 R 0.564 | 7.0 min elapsed, ~53.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     24/200      4.01G       1.52      1.804     0.0214         24        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.9it/s 1.2s0.2s
                   all        321        632      0.662      0.566      0.633      0.306
[progress] epoch 24/200 | mAP50 0.633 | mAP50-95 0.306 | P 0.662 R 0.566 | 7.2 min elapsed, ~53.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     25/200      4.01G      1.497      1.747    0.02077         20        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.1it/s 1.2s0.2s
                   all        321        632      0.666      0.563      0.602      0.273
[progress] epoch 25/200 | mAP50 0.602 | mAP50-95 0.273 | P 0.665 R 0.564 | 7.5 min elapsed, ~52.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     26/200      4.01G       1.49      1.788    0.02145         20        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.607      0.591      0.617      0.287
[progress] epoch 26/200 | mAP50 0.617 | mAP50-95 0.287 | P 0.607 R 0.591 | 7.7 min elapsed, ~51.8 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     27/200      4.01G       1.49       1.75    0.02089         21        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s0.2s
                   all        321        632      0.616      0.564      0.594      0.279
[progress] epoch 27/200 | mAP50 0.594 | mAP50-95 0.279 | P 0.616 R 0.564 | 8.0 min elapsed, ~51.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     28/200      4.01G      1.481      1.747    0.02073         26        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.9it/s 1.2s0.2s
                   all        321        632      0.648      0.582      0.627      0.304
[progress] epoch 28/200 | mAP50 0.627 | mAP50-95 0.304 | P 0.647 R 0.582 | 8.2 min elapsed, ~50.6 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     29/200      4.01G      1.478      1.768    0.02123         17        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.655      0.582      0.615      0.297
[progress] epoch 29/200 | mAP50 0.615 | mAP50-95 0.297 | P 0.655 R 0.582 | 8.5 min elapsed, ~50.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     30/200      4.01G       1.46       1.72    0.02011         20        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.1it/s 1.2s0.2s
                   all        321        632      0.622      0.593      0.602      0.283
[progress] epoch 30/200 | mAP50 0.602 | mAP50-95 0.283 | P 0.622 R 0.593 | 8.7 min elapsed, ~49.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     31/200      4.01G      1.488      1.728    0.02129         25        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.9it/s 1.2s0.2s
                   all        321        632       0.61      0.598      0.627      0.293
[progress] epoch 31/200 | mAP50 0.627 | mAP50-95 0.293 | P 0.610 R 0.598 | 9.0 min elapsed, ~49.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     32/200      4.01G      1.448      1.676    0.02056         21        512: 100% ━━━━━━━━━━━━ 47/47 3.4it/s 13.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632      0.622      0.573      0.611      0.284
[progress] epoch 32/200 | mAP50 0.611 | mAP50-95 0.284 | P 0.623 R 0.573 | 9.2 min elapsed, ~48.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     33/200      4.01G      1.472      1.694    0.02067         31        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.682      0.581        0.6      0.283
[progress] epoch 33/200 | mAP50 0.600 | mAP50-95 0.283 | P 0.682 R 0.581 | 9.5 min elapsed, ~48.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     34/200      4.01G      1.432      1.673    0.01982         29        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.638      0.583      0.604      0.279
[progress] epoch 34/200 | mAP50 0.604 | mAP50-95 0.279 | P 0.638 R 0.583 | 9.8 min elapsed, ~47.6 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     35/200      4.01G      1.457      1.664    0.02035         29        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.8it/s 1.0s0.2s
                   all        321        632      0.719      0.579      0.631      0.293
[progress] epoch 35/200 | mAP50 0.631 | mAP50-95 0.293 | P 0.719 R 0.579 | 10.0 min elapsed, ~47.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     36/200      4.01G      1.443      1.654       0.02         26        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.667      0.581      0.631      0.289
[progress] epoch 36/200 | mAP50 0.631 | mAP50-95 0.289 | P 0.666 R 0.581 | 10.3 min elapsed, ~46.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     37/200      4.01G      1.425      1.683    0.01995         18        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.609      0.548      0.576      0.259
[progress] epoch 37/200 | mAP50 0.576 | mAP50-95 0.259 | P 0.609 R 0.548 | 10.5 min elapsed, ~46.3 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     38/200      4.01G      1.472      1.633    0.02023         28        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.653      0.627      0.637      0.303
[progress] epoch 38/200 | mAP50 0.637 | mAP50-95 0.303 | P 0.653 R 0.627 | 10.8 min elapsed, ~45.9 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     39/200      4.01G      1.429      1.598    0.01999         15        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.6s0.2s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.644      0.574       0.62      0.284
[progress] epoch 39/200 | mAP50 0.620 | mAP50-95 0.284 | P 0.643 R 0.574 | 11.0 min elapsed, ~45.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     40/200      4.01G      1.448      1.578    0.01987         19        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s0.2s
                   all        321        632      0.702      0.582      0.632      0.284
[progress] epoch 40/200 | mAP50 0.632 | mAP50-95 0.284 | P 0.702 R 0.582 | 11.3 min elapsed, ~45.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     41/200      4.01G      1.434      1.545    0.01986         20        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632      0.705      0.572      0.626      0.295
[progress] epoch 41/200 | mAP50 0.626 | mAP50-95 0.295 | P 0.705 R 0.572 | 11.5 min elapsed, ~44.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     42/200      4.01G       1.44      1.515    0.01967         29        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632       0.67       0.61       0.63       0.29
[progress] epoch 42/200 | mAP50 0.630 | mAP50-95 0.290 | P 0.670 R 0.610 | 11.8 min elapsed, ~44.3 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     43/200      4.01G      1.397       1.53    0.01918         25        512: 100% ━━━━━━━━━━━━ 47/47 3.4it/s 13.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632       0.64      0.596       0.61      0.286
[progress] epoch 43/200 | mAP50 0.610 | mAP50-95 0.286 | P 0.640 R 0.596 | 12.0 min elapsed, ~43.9 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     44/200      4.01G      1.381      1.523    0.01908         19        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.693      0.597      0.634      0.291
[progress] epoch 44/200 | mAP50 0.634 | mAP50-95 0.290 | P 0.693 R 0.597 | 12.3 min elapsed, ~43.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     45/200      4.01G      1.397      1.512    0.01979         16        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.655      0.573      0.604      0.264
[progress] epoch 45/200 | mAP50 0.604 | mAP50-95 0.264 | P 0.655 R 0.573 | 12.5 min elapsed, ~43.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     46/200      4.01G      1.432      1.574    0.01998         14        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632      0.668      0.611       0.63      0.289
[progress] epoch 46/200 | mAP50 0.630 | mAP50-95 0.289 | P 0.668 R 0.611 | 12.8 min elapsed, ~42.8 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     47/200      4.01G      1.384      1.528    0.01944         20        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.0s0.2s
                   all        321        632      0.657      0.581      0.623      0.274
[progress] epoch 47/200 | mAP50 0.623 | mAP50-95 0.274 | P 0.656 R 0.581 | 13.0 min elapsed, ~42.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     48/200      4.01G      1.396      1.534    0.01965         24        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.691       0.57      0.612      0.285
[progress] epoch 48/200 | mAP50 0.612 | mAP50-95 0.285 | P 0.691 R 0.570 | 13.3 min elapsed, ~42.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     49/200      4.01G       1.38      1.527    0.01941         18        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.665       0.58      0.625      0.281
[progress] epoch 49/200 | mAP50 0.625 | mAP50-95 0.281 | P 0.665 R 0.581 | 13.5 min elapsed, ~41.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     50/200      4.01G      1.381      1.548    0.01953         22        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.667      0.587      0.628      0.293
[progress] epoch 50/200 | mAP50 0.628 | mAP50-95 0.293 | P 0.667 R 0.587 | 13.8 min elapsed, ~41.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     51/200      4.01G      1.375      1.477    0.01928         17        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.677      0.572      0.602      0.277
[progress] epoch 51/200 | mAP50 0.602 | mAP50-95 0.277 | P 0.677 R 0.572 | 14.0 min elapsed, ~41.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     52/200      4.01G      1.405      1.507    0.01947         28        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s0.2s
                   all        321        632      0.646      0.602      0.611      0.278
[progress] epoch 52/200 | mAP50 0.611 | mAP50-95 0.278 | P 0.646 R 0.603 | 14.3 min elapsed, ~40.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     53/200      4.01G       1.38      1.499    0.01836         29        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.657       0.61      0.621       0.29
[progress] epoch 53/200 | mAP50 0.621 | mAP50-95 0.290 | P 0.657 R 0.610 | 14.5 min elapsed, ~40.3 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     54/200      4.01G      1.349      1.437    0.01872         28        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s0.2s
                   all        321        632      0.659       0.54      0.588      0.278
[progress] epoch 54/200 | mAP50 0.588 | mAP50-95 0.278 | P 0.659 R 0.540 | 14.8 min elapsed, ~40.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     55/200      4.01G      1.359      1.471    0.01893         32        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s0.2s
                   all        321        632      0.682      0.581      0.623      0.266
[progress] epoch 55/200 | mAP50 0.623 | mAP50-95 0.266 | P 0.682 R 0.581 | 15.0 min elapsed, ~39.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     56/200      4.01G      1.355      1.424    0.01826         38        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.4it/s 1.1s0.2s
                   all        321        632      0.693      0.594      0.617      0.285
[progress] epoch 56/200 | mAP50 0.617 | mAP50-95 0.285 | P 0.693 R 0.594 | 15.3 min elapsed, ~39.3 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     57/200      4.01G      1.339      1.446    0.01817         33        512: 100% ━━━━━━━━━━━━ 47/47 3.4it/s 13.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.8it/s 1.0s0.2s
                   all        321        632      0.682      0.548      0.611      0.276
[progress] epoch 57/200 | mAP50 0.611 | mAP50-95 0.276 | P 0.682 R 0.548 | 15.6 min elapsed, ~39.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     58/200      4.01G      1.358      1.457     0.0186         28        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.708      0.605      0.638       0.29
[progress] epoch 58/200 | mAP50 0.638 | mAP50-95 0.290 | P 0.708 R 0.605 | 15.8 min elapsed, ~38.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     59/200      4.01G      1.355      1.425     0.0185         30        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.2s0.2s
                   all        321        632      0.691      0.574      0.627      0.286
[progress] epoch 59/200 | mAP50 0.627 | mAP50-95 0.286 | P 0.691 R 0.574 | 16.1 min elapsed, ~38.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     60/200      4.01G      1.356      1.453     0.0188         19        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 4.5it/s 1.3s0.2s
                   all        321        632      0.704      0.587      0.632      0.284
[progress] epoch 60/200 | mAP50 0.632 | mAP50-95 0.284 | P 0.704 R 0.588 | 16.3 min elapsed, ~38.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     61/200      4.01G       1.33      1.418    0.01824         19        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.674      0.573      0.598      0.263
[progress] epoch 61/200 | mAP50 0.598 | mAP50-95 0.263 | P 0.674 R 0.573 | 16.6 min elapsed, ~37.7 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     62/200      4.01G      1.323      1.379    0.01804         24        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s0.2s
                   all        321        632      0.672      0.608      0.621      0.284
[progress] epoch 62/200 | mAP50 0.621 | mAP50-95 0.284 | P 0.672 R 0.608 | 16.8 min elapsed, ~37.4 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     63/200      4.01G      1.333      1.378    0.01853         35        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.6it/s 1.1s0.2s
                   all        321        632      0.699      0.587      0.609      0.281
[progress] epoch 63/200 | mAP50 0.609 | mAP50-95 0.281 | P 0.699 R 0.587 | 17.1 min elapsed, ~37.1 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     64/200      4.01G      1.364      1.374    0.01799         26        512: 100% ━━━━━━━━━━━━ 47/47 3.5it/s 13.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.0s0.2s
                   all        321        632      0.633      0.602      0.607      0.271
[progress] epoch 64/200 | mAP50 0.607 | mAP50-95 0.271 | P 0.633 R 0.602 | 17.3 min elapsed, ~36.8 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     65/200      4.01G      1.334      1.397    0.01841         33        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.652      0.595       0.62      0.296
[progress] epoch 65/200 | mAP50 0.620 | mAP50-95 0.296 | P 0.652 R 0.595 | 17.6 min elapsed, ~36.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     66/200      4.01G      1.325      1.349    0.01774         33        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.675      0.583      0.608      0.274
[progress] epoch 66/200 | mAP50 0.608 | mAP50-95 0.274 | P 0.675 R 0.583 | 17.8 min elapsed, ~36.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     67/200      4.01G      1.319      1.354    0.01805         27        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.2it/s 1.1s0.2s
                   all        321        632      0.662      0.609      0.611      0.275
[progress] epoch 67/200 | mAP50 0.611 | mAP50-95 0.275 | P 0.662 R 0.609 | 18.1 min elapsed, ~35.8 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     68/200      4.01G      1.306      1.311    0.01787         26        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.3it/s 1.1s0.2s
                   all        321        632      0.698      0.555      0.598      0.273
[progress] epoch 68/200 | mAP50 0.598 | mAP50-95 0.273 | P 0.698 R 0.555 | 18.3 min elapsed, ~35.5 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     69/200      4.01G      1.305      1.348    0.01776         23        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.637      0.603       0.62      0.286
[progress] epoch 69/200 | mAP50 0.620 | mAP50-95 0.286 | P 0.637 R 0.603 | 18.6 min elapsed, ~35.2 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     70/200      4.01G        1.3      1.332     0.0178         35        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.659      0.601      0.618      0.284
[progress] epoch 70/200 | mAP50 0.618 | mAP50-95 0.284 | P 0.659 R 0.601 | 18.8 min elapsed, ~34.9 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     71/200      4.01G      1.291      1.316    0.01765         16        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.671      0.562      0.597      0.279
[progress] epoch 71/200 | mAP50 0.597 | mAP50-95 0.279 | P 0.671 R 0.562 | 19.0 min elapsed, ~34.6 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     72/200      4.01G      1.299      1.337    0.01785         16        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 13.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.7it/s 1.1s0.2s
                   all        321        632      0.702      0.554      0.587      0.267
[progress] epoch 72/200 | mAP50 0.587 | mAP50-95 0.268 | P 0.702 R 0.554 | 19.3 min elapsed, ~34.3 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     73/200      4.01G      1.283      1.287    0.01757         28        512: 100% ━━━━━━━━━━━━ 47/47 3.7it/s 12.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.639      0.572      0.595      0.281
[progress] epoch 73/200 | mAP50 0.595 | mAP50-95 0.281 | P 0.639 R 0.572 | 19.5 min elapsed, ~34.0 min left (upper bound, early stopping may end sooner)

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     74/200      4.01G      1.283      1.292    0.01753         21        512: 100% ━━━━━━━━━━━━ 47/47 3.6it/s 12.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 5.5it/s 1.1s0.2s
                   all        321        632      0.621      0.622      0.611      0.276
EarlyStopping: Training stopped early as no improvement observed in last 50 epochs. Best results observed at epoch 24, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
[progress] epoch 74/200 | mAP50 0.611 | mAP50-95 0.276 | P 0.621 R 0.622 | 19.8 min elapsed, ~33.7 min left (upper bound, early stopping may end sooner)

74 epochs completed in 0.319 hours.
Optimizer stripped from /kaggle/working/bm_runs/yolo26s/weights/last.pt, 20.3MB
Optimizer stripped from /kaggle/working/bm_runs/yolo26s/weights/best.pt, 20.3MB

Validating /kaggle/working/bm_runs/yolo26s/weights/best.pt...
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 20.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 6/6 2.2s/it 13.4s1.7ss
                   all        321        632      0.662      0.568      0.635      0.306
           vocal_cords         36         45      0.895      0.756      0.847      0.393
               trachea         19         19      0.517      0.579      0.589      0.284
           main_carina         60         75      0.714      0.333      0.473      0.135
   right_main_bronchus         56         67      0.767      0.492      0.627      0.333
    left_main_bronchus         69         83      0.832      0.482      0.614      0.319
 intermediate_bronchus         39         44      0.643      0.591      0.685      0.342
right_superior_lobar_bronchus         48         54      0.544      0.751      0.708       0.36
right_middle_lobar_bronchus         48         60      0.491      0.533      0.513      0.216
right_inferior_lobar_bronchus         55         64      0.511      0.672      0.652      0.364
left_superior_lobar_bronchus         41         51      0.709      0.529      0.642      0.286
left_inferior_lobar_bronchus         61         70      0.662      0.531      0.632      0.337
Speed: 0.2ms preprocess, 38.2ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to /kaggle/working/bm_runs/yolo26s
[progress] best.pt final validation | mAP50 0.635 | mAP50-95 0.306
[train] done in 20.2 min - evaluating /kaggle/working/bm_runs/yolo26s/weights/best.pt
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.3±0.7 ms, read: 112.2±57.1 MB/s, size: 65.4 KB)
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels/val... 321 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 321/321 1.1Kit/s 0.3s<0.1s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 5.4it/s 3.9s0.1s
                   all        321        632       0.65       0.57      0.632      0.306
Speed: 0.9ms preprocess, 6.0ms inference, 0.0ms loss, 1.4ms postprocess per image
Results saved to /kaggle/working/bm_runs/yolo26s/eval_val_all
[eval] val  all  mAP50 0.632  mAP50-95 0.306
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 128.5±37.9 MB/s, size: 59.2 KB)
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels/val... 141 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 141/141 938.2it/s 0.2s2s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 6.9it/s 1.3s0.1s
                   all        141        239      0.721      0.535      0.632      0.311
Speed: 0.4ms preprocess, 5.0ms inference, 0.0ms loss, 0.9ms postprocess per image
[eval] val  lc   mAP50 0.632  mAP50-95 0.310
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 100.9±38.3 MB/s, size: 61.8 KB)
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels/val... 180 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 180/180 1.1Kit/s 0.2s<0.2s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 12/12 7.4it/s 1.6s0.1s
                   all        180        393      0.655      0.575      0.645      0.317
Speed: 0.4ms preprocess, 4.7ms inference, 0.0ms loss, 0.9ms postprocess per image
[eval] val  nlc  mAP50 0.645  mAP50-95 0.317
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 20.8 GFLOPs
WARNING ⚠️ val: Slow image access detected (ping: 0.2±0.4 ms, read: 18.6±9.9 MB/s, size: 70.7 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels/test... 325 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 325/325 546.0it/s 0.6s0.1s
val: /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/images/test/bm_lc_p057_f04.png: 1 duplicate labels removed
val: /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/images/test/bm_lc_p057_f13.png: 1 duplicate labels removed
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 5.5it/s 3.8s0.1s
                   all        325        639      0.635      0.516      0.561      0.278
Speed: 0.9ms preprocess, 5.4ms inference, 0.0ms loss, 1.4ms postprocess per image
Results saved to /kaggle/working/bm_runs/yolo26s/eval_test_all
[eval] test all  mAP50 0.561  mAP50-95 0.278
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 78.8±30.4 MB/s, size: 59.3 KB)
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels/test... 144 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 144/144 1.1Kit/s 0.1s<0.1s
val: /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../images/test/bm_lc_p057_f04.png: 1 duplicate labels removed
val: /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../images/test/bm_lc_p057_f13.png: 1 duplicate labels removed
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 9/9 6.7it/s 1.4s0.1ss
                   all        144        236      0.608      0.607      0.615      0.298
Speed: 0.4ms preprocess, 5.1ms inference, 0.0ms loss, 0.9ms postprocess per image
[eval] test lc   mAP50 0.615  mAP50-95 0.298
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.1±0.1 ms, read: 128.8±36.0 MB/s, size: 62.0 KB)
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels/test... 181 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 181/181 1.1Kit/s 0.2s<0.2s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 12/12 7.6it/s 1.6s0.1s
                   all        181        403      0.692      0.463      0.543      0.281
Speed: 0.4ms preprocess, 4.5ms inference, 0.0ms loss, 1.0ms postprocess per image
[eval] test nlc  mAP50 0.543  mAP50-95 0.281
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CPU (Intel Xeon CPU @ 2.00GHz)
💡 ProTip: Export to OpenVINO format for best performance on Intel hardware. Learn more at https://docs.ultralytics.com/integrations/openvino
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 13.2 GFLOPs

PyTorch: starting from '/kaggle/working/bm_runs/yolo26s/weights/best.pt' with input shape (1, 3, 512, 512) BCHW and output shape(s) (1, 15, 5376) (19.4 MB)

ONNX: starting export with onnx 1.23.1 opset 18...
ONNX: slimming with onnxslim 0.1.97...
ONNX: export success ✅ 2.4s, saved as '/kaggle/working/bm_runs/yolo26s/weights/best.onnx' (36.3 MB)

Export complete (2.9s)
Results saved to /kaggle/working/bm_runs/yolo26s/weights/best.onnx
Predict:         yolo predict task=detect model=/kaggle/working/bm_runs/yolo26s/weights/best.onnx imgsz=512 
Validate:        yolo val task=detect model=/kaggle/working/bm_runs/yolo26s/weights/best.onnx imgsz=512 data=/kaggle/working/bm_runs/data_abs.yaml  
Visualize:       https://netron.app
[export] onnx: /kaggle/working/bm_runs/yolo26s/weights/best.onnx
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CPU (Intel Xeon CPU @ 2.00GHz)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 13.2 GFLOPs

PyTorch: starting from '/kaggle/working/bm_runs/yolo26s/weights/best.pt' with input shape (1, 3, 512, 512) BCHW and output shape(s) (1, 15, 5376) (19.4 MB)

OpenVINO: starting export with openvino 2026.4.1-22982-e213a147257-releases/2026/4...
OpenVINO: export success ✅ 3.1s, saved as '/kaggle/working/bm_runs/yolo26s/weights/best_openvino_model/' (36.5 MB)

Export complete (3.6s)
Results saved to /kaggle/working/bm_runs/yolo26s/weights/best_openvino_model
Predict:         yolo predict task=detect model=/kaggle/working/bm_runs/yolo26s/weights/best_openvino_model/ imgsz=512 
Validate:        yolo val task=detect model=/kaggle/working/bm_runs/yolo26s/weights/best_openvino_model/ imgsz=512 data=/kaggle/working/bm_runs/data_abs.yaml  
Visualize:       https://netron.app
[export] openvino_fp32: /kaggle/working/bm_runs/yolo26s/weights/best_openvino_model/
WARNING ⚠️ 'int8' is deprecated and will be removed in the future. Use 'quantize' instead.
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CPU (Intel Xeon CPU @ 2.00GHz)
YOLO26s summary (fused): 120 layers, 9,469,437 parameters, 0 gradients, 13.2 GFLOPs

PyTorch: starting from '/kaggle/working/bm_runs/yolo26s/weights/best.pt' with input shape (1, 3, 512, 512) BCHW and output shape(s) (1, 15, 5376) (19.4 MB)
OpenVINO: collecting INT8 calibration images from 'data=/kaggle/working/bm_runs/data_abs.yaml'
val: Fast image access ✅ (ping: 0.5±1.0 ms, read: 100.7±26.9 MB/s, size: 62.2 KB)
val: Scanning /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels/val... 160 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 160/160 941.6it/s 0.2s0.2s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya22/bm-dataset/bm_anatomy/labels is not writable, cache not saved.
WARNING ⚠️ OpenVINO: >300 images recommended for INT8 calibration, found 160 images.

OpenVINO: starting export with openvino 2026.4.1-22982-e213a147257-releases/2026/4...
INFO:nncf:28 ignored nodes were found by names in the NNCFGraph
INFO:nncf:Not adding activation input quantizer for operation: 149 __module.model.23/aten::view/Reshape
INFO:nncf:Not adding activation input quantizer for operation: 243 __module.model.23/aten::view/Reshape_3
INFO:nncf:Not adding activation input quantizer for operation: 309 __module.model.23/aten::view/Reshape_1
INFO:nncf:Not adding activation input quantizer for operation: 371 __module.model.23/aten::view/Reshape_4
INFO:nncf:Not adding activation input quantizer for operation: 382 __module.model.23/aten::view/Reshape_2
INFO:nncf:Not adding activation input quantizer for operation: 389 __module.model.23/aten::view/Reshape_5
INFO:nncf:Not adding activation input quantizer for operation: 158 __module.model.23/aten::cat/Concat
INFO:nncf:Not adding activation input quantizer for operation: 170 __module.model.23/prim::ListUnpack/VariadicSplit
INFO:nncf:Not adding activation input quantizer for operation: 184 __module.model.23/aten::sub/Subtract
INFO:nncf:Not adding activation input quantizer for operation: 185 __module.model.23/aten::add/Add
INFO:nncf:Not adding activation input quantizer for operation: 204 __module.model.23/aten::add/Add_1
222 __module.model.23/aten::div/Divide

INFO:nncf:Not adding activation input quantizer for operation: 205 __module.model.23/aten::sub/Subtract_1
INFO:nncf:Not adding activation input quantizer for operation: 223 __module.model.23/aten::cat/Concat_1
INFO:nncf:Not adding activation input quantizer for operation: 242 __module.model.23/aten::mul/Multiply
INFO:nncf:Not adding activation input quantizer for operation: 257 __module.model.23/aten::cat/Concat_2
INFO:nncf:Not adding activation input quantizer for operation: 272 __module.model.23/aten::sigmoid/Sigmoid
INFO:nncf:Not adding activation input quantizer for operation: 256 __module.model.23/aten::cat/Concat_3
Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 160/160 • 0:00:31 • 0:00:0000:0100:02
Applying Fast Bias correction ━━━━━━━━━━━━━━━━━━━ 100% 94/94 • 0:00:03 • 0:00:0000:0100:01━━━━━━━━━━━━━━━━━━━━━━━━━   • 0:00:00  
OpenVINO: export success ✅ 44.8s, saved as '/kaggle/working/bm_runs/yolo26s/weights/best_int8_openvino_model/' (9.9 MB)

Export complete (45.2s)
Results saved to /kaggle/working/bm_runs/yolo26s/weights/best_int8_openvino_model
Predict:         yolo predict task=detect model=/kaggle/working/bm_runs/yolo26s/weights/best_int8_openvino_model/ imgsz=512 
Validate:        yolo val task=detect model=/kaggle/working/bm_runs/yolo26s/weights/best_int8_openvino_model/ imgsz=512 data=/kaggle/working/bm_runs/data_abs.yaml  
Visualize:       https://netron.app
[export] openvino_int8: /kaggle/working/bm_runs/yolo26s/weights/best_int8_openvino_model/
[cpu] pytorch: 105.7 ms/img (9.5 FPS)
Loading /kaggle/working/bm_runs/yolo26s/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime 1.30.0 with CPUExecutionProvider
[cpu] onnx: 85.4 ms/img (11.7 FPS)
Loading /kaggle/working/bm_runs/yolo26s/weights/best_openvino_model/ for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on CPU...
[cpu] openvino_fp32: 82.2 ms/img (12.2 FPS)
Loading /kaggle/working/bm_runs/yolo26s/weights/best_int8_openvino_model/ for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on CPU...
[cpu] openvino_int8: 67.4 ms/img (14.8 FPS)
[done] yolo26s: report at /kaggle/working/bm_runs/yolo26s/report.md
[done] download /kaggle/working/bm_runs.zip
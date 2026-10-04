ultralytics 8.4.173 | data /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/data.yaml | output /kaggle/working/uaal_runs
[train] yolo26s: {"epochs": 150, "imgsz": 640, "batch": 32, "patience": 30, "optimizer": "auto", "cos_lr": true, "seed": 42, "deterministic": false, "fraction": 1.0, "degrees": 15.0, "translate": 0.1, "scale": 0.5, "shear": 0.0, "perspective": 0.0, "fliplr": 0.0, "flipud": 0.0, "mosaic": 1.0, "close_mosaic": 15, "mixup": 0.0, "cutmix": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.4, "cls_pw": 0.5}
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
                                                        CUDA:1 (Tesla T4, 14912MiB)
engine/trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, augmentations=[{'__version__': '2.0.8', 'transform': {'__class_fullname__': 'OneOf', 'p': 0.25, 'transforms': [{'__class_fullname__': 'MotionBlur', 'p': 0.5, 'allow_shifted': True, 'angle_range': (0.0, 360.0), 'blur_limit': (3, 9), 'direction_range': (-1.0, 1.0)}, {'__class_fullname__': 'GaussianBlur', 'p': 0.5, 'blur_limit': (3, 7), 'sigma_limit': (0.5, 3.0)}, {'__class_fullname__': 'Defocus', 'p': 0.5, 'alias_blur': (0.1, 0.5), 'radius': (1, 3)}]}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'GaussNoise', 'p': 0.15, 'mean_range': (0.0, 0.0), 'noise_scale_factor': 1.0, 'per_channel': True, 'std_range': (0.02, 0.06)}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'CLAHE', 'p': 0.2, 'clip_limit': (1.0, 3.0), 'tile_grid_size': (8, 8)}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'RandomBrightnessContrast', 'p': 0.3, 'brightness_by_max': True, 'brightness_limit': (-0.2, 0.2), 'contrast_limit': (-0.25, 0.25), 'ensure_safe_range': False}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'RandomGamma', 'p': 0.2, 'gamma_limit': (80.0, 120.0)}}, {'__version__': '2.0.8', 'transform': {'__class_fullname__': 'ImageCompression', 'p': 0.2, 'compression_type': 'jpeg', 'quality_range': (50, 95)}}], auto_augment=randaugment, batch=32, bgr=0.0, box=7.5, cache=ram, cfg=None, channels_last=None, classes=None, close_mosaic=15, cls=0.5, cls_pw=0.5, cls_remap=True, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=/kaggle/working/uaal_runs/data_abs.yaml, degrees=15.0, deterministic=False, device=0,1, dfl=1.5, dgrad=0.5, dis=6.0, distill_model=None, dlam=1.0, dlog=1.0, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=150, erasing=0.4, exist_ok=True, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, imgsz=640, iou=0.7, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo26s.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=yolo26s, nbs=64, nms=None, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=30, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=/kaggle/working/uaal_runs, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=/kaggle/working/uaal_runs/yolo26s, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=42, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=4, workspace=None
Overriding model.yaml nc=80 with nc=9

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
 23        [16, 19, 22]  1    938830  ultralytics.nn.modules.head.Detect           [9, 1, True, [128, 256, 512]] 
YOLO26s summary: 260 layers, 9,954,830 parameters, 9,954,830 gradients, 22.8 GFLOPs

Transferred 696/708 items from pretrained weights
DDP: debug command /usr/bin/python3 -m torch.distributed.run --nproc_per_node 2 --master_port 31238 /root/.config/Ultralytics/DDP/_temp_45uf3wet138517152988336.py
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
                                                        CUDA:1 (Tesla T4, 14912MiB)
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 194.6±165.3 MB/s, size: 124.0 KB)
train: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels/train... 4064 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4064/4064 1.0Kit/s 4.0s0.0ss
WARNING ⚠️ train: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels is not writable, cache not saved.
train: Caching images (4.3GB RAM): 100% ━━━━━━━━━━━━ 4064/4064 201.2it/s 20.2s0.1ss
albumentations: OneOf([
  MotionBlur(p=0.5, allow_shifted=True, angle_range=(0.0, 360.0), blur_limit=(3, 9), direction_range=(-1.0, 1.0)),
  GaussianBlur(p=0.5, blur_limit=(3, 7), sigma_limit=(0.5, 3.0)),
  Defocus(p=0.5, alias_blur=(0.1, 0.5), radius=(1, 3)),
], p=0.25), GaussNoise(p=0.15, mean_range=(0.0, 0.0), noise_scale_factor=1.0, per_channel=True, std_range=(0.02, 0.06)), CLAHE(p=0.2, clip_limit=(1.0, 3.0), tile_grid_size=(8, 8)), RandomBrightnessContrast(p=0.3, brightness_by_max=True, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.25, 0.25), ensure_safe_range=False), RandomGamma(p=0.2, gamma_limit=(80.0, 120.0)), ImageCompression(p=0.2, compression_type='jpeg', quality_range=(50, 95))
val: Fast image access ✅ (ping: 1.1±2.5 ms, read: 217.6±176.9 MB/s, size: 208.0 KB)
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels/val... 870 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 870/870 1.0Kit/s 0.9s0.0ss
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels is not writable, cache not saved.
val: Caching images (0.9GB RAM): 100% ━━━━━━━━━━━━ 870/870 148.3it/s 5.9s0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and determining best 'optimizer' and 'lr0' automatically... 
optimizer: AdamW(lr=0.000769, momentum=0.9) with parameter groups 114 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias(decay=0.0)
Class weights: [      0.557       0.526       0.568       0.594       0.649       1.294       1.487       1.808       1.518]
Plotting labels to /kaggle/working/uaal_runs/yolo26s/labels.jpg... 
Using 4064 train, 870 val images for fraction=1.0 at imgsz=640
Using 4 dataloader workers
Logging results to /kaggle/working/uaal_runs/yolo26s
Starting training for 150 epochs...

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      1/150      4.71G      1.634      5.895    0.02229         58        640: 100% ━━━━━━━━━━━━ 127/127 2.2it/s 57.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 1.1s/it 15.4s0.3s
                   all        870       2060      0.575      0.352      0.393      0.175
[progress] epoch 1/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 2.6 min elapsed, ~390.1 min left
[progress] epoch 1/150 | mAP50 0.393 | mAP50-95 0.175 | P 0.575 R 0.352 | 2.6 min elapsed, ~391.6 min left
/usr/local/lib/python3.13/dist-packages/ray/train/_internal/session.py:683: UserWarning: `get_trial_id` is meant to only be called inside a function that is executed by a Tuner or Trainer. Returning `None`.
  warnings.warn(

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      2/150      5.86G      1.546       1.72    0.02102         65        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.613      0.522      0.556      0.234
[progress] epoch 2/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 3.4 min elapsed, ~250.0 min left
[progress] epoch 2/150 | mAP50 0.556 | mAP50-95 0.234 | P 0.614 R 0.522 | 3.4 min elapsed, ~250.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      3/150      5.88G      1.565      1.352    0.02155         73        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.612      0.587      0.613      0.281
[progress] epoch 3/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 4.1 min elapsed, ~202.4 min left
[progress] epoch 3/150 | mAP50 0.613 | mAP50-95 0.281 | P 0.612 R 0.587 | 4.1 min elapsed, ~203.3 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      4/150      5.88G      1.547      1.212    0.02109         98        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.742      0.652      0.709       0.35
[progress] epoch 4/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 4.9 min elapsed, ~178.5 min left
[progress] epoch 4/150 | mAP50 0.709 | mAP50-95 0.350 | P 0.742 R 0.652 | 4.9 min elapsed, ~178.8 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      5/150      5.88G      1.486      1.096    0.02029         84        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.722      0.654      0.672      0.331
[progress] epoch 5/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 5.6 min elapsed, ~163.7 min left
[progress] epoch 5/150 | mAP50 0.672 | mAP50-95 0.331 | P 0.722 R 0.654 | 5.7 min elapsed, ~164.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      6/150      5.88G      1.473      1.023    0.01991         81        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060       0.82      0.593      0.685      0.354
[progress] epoch 6/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 6.4 min elapsed, ~153.5 min left
[progress] epoch 6/150 | mAP50 0.685 | mAP50-95 0.353 | P 0.820 R 0.593 | 6.4 min elapsed, ~153.8 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      7/150      5.88G      1.435     0.9587    0.01924         47        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060       0.68      0.656      0.696      0.362
[progress] epoch 7/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 7.1 min elapsed, ~146.0 min left
[progress] epoch 7/150 | mAP50 0.696 | mAP50-95 0.362 | P 0.679 R 0.656 | 7.2 min elapsed, ~146.2 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      8/150      5.88G      1.405     0.9144    0.01889         69        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.4s0.3s
                   all        870       2060      0.729      0.701      0.744      0.385
[progress] epoch 8/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 7.9 min elapsed, ~140.2 min left
[progress] epoch 8/150 | mAP50 0.744 | mAP50-95 0.385 | P 0.729 R 0.701 | 7.9 min elapsed, ~140.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
      9/150      5.88G      1.422     0.9126     0.0187         73        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.823      0.604      0.703      0.371
[progress] epoch 9/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 8.6 min elapsed, ~135.3 min left
[progress] epoch 9/150 | mAP50 0.703 | mAP50-95 0.371 | P 0.823 R 0.604 | 8.6 min elapsed, ~135.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     10/150      5.88G      1.388      0.868    0.01837         84        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.725      0.646      0.699      0.373
[progress] epoch 10/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 9.4 min elapsed, ~131.4 min left
[progress] epoch 10/150 | mAP50 0.699 | mAP50-95 0.373 | P 0.725 R 0.646 | 9.4 min elapsed, ~131.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     11/150      5.88G      1.355     0.8522    0.01812         75        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.708      0.767      0.784      0.425
[progress] epoch 11/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 10.1 min elapsed, ~128.0 min left
[progress] epoch 11/150 | mAP50 0.784 | mAP50-95 0.425 | P 0.708 R 0.767 | 10.1 min elapsed, ~128.1 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     12/150      5.88G      1.356     0.8158    0.01793         82        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.715      0.714      0.729      0.392
[progress] epoch 12/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 10.9 min elapsed, ~125.2 min left
[progress] epoch 12/150 | mAP50 0.729 | mAP50-95 0.392 | P 0.715 R 0.714 | 10.9 min elapsed, ~125.3 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     13/150      5.88G      1.321     0.7971    0.01728         78        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.801      0.723      0.759      0.403
[progress] epoch 13/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 11.6 min elapsed, ~122.7 min left
[progress] epoch 13/150 | mAP50 0.759 | mAP50-95 0.403 | P 0.801 R 0.723 | 11.7 min elapsed, ~122.8 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     14/150      5.88G       1.31     0.7799    0.01691         79        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.717      0.758      0.754      0.413
[progress] epoch 14/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 12.4 min elapsed, ~120.5 min left
[progress] epoch 14/150 | mAP50 0.754 | mAP50-95 0.413 | P 0.717 R 0.758 | 12.4 min elapsed, ~120.6 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     15/150      5.88G      1.313      0.782    0.01713         65        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.779      0.726      0.784      0.424
[progress] epoch 15/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 13.2 min elapsed, ~118.4 min left
[progress] epoch 15/150 | mAP50 0.784 | mAP50-95 0.424 | P 0.779 R 0.726 | 13.2 min elapsed, ~118.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     16/150      5.88G      1.283     0.7552    0.01662         71        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.4it/s 3.2s0.3s
                   all        870       2060      0.766      0.759      0.791      0.428
[progress] epoch 16/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 13.9 min elapsed, ~116.4 min left
[progress] epoch 16/150 | mAP50 0.791 | mAP50-95 0.428 | P 0.766 R 0.759 | 13.9 min elapsed, ~116.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     17/150      5.88G      1.293     0.7513    0.01727         76        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.2s0.3s
                   all        870       2060        0.8      0.667      0.735      0.412
[progress] epoch 17/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 14.7 min elapsed, ~114.7 min left
[progress] epoch 17/150 | mAP50 0.735 | mAP50-95 0.412 | P 0.800 R 0.667 | 14.7 min elapsed, ~114.8 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     18/150      5.88G      1.277     0.7555    0.01644         71        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.695      0.709      0.741      0.396
[progress] epoch 18/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 15.4 min elapsed, ~113.0 min left
[progress] epoch 18/150 | mAP50 0.741 | mAP50-95 0.396 | P 0.695 R 0.709 | 15.4 min elapsed, ~113.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     19/150      5.88G      1.236     0.7051    0.01608         77        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.766      0.741      0.768      0.418
[progress] epoch 19/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 16.1 min elapsed, ~111.3 min left
[progress] epoch 19/150 | mAP50 0.768 | mAP50-95 0.418 | P 0.766 R 0.741 | 16.2 min elapsed, ~111.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     20/150      5.88G       1.25      0.712    0.01603         66        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.768      0.758      0.784       0.42
[progress] epoch 20/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 16.9 min elapsed, ~109.8 min left
[progress] epoch 20/150 | mAP50 0.784 | mAP50-95 0.420 | P 0.768 R 0.758 | 16.9 min elapsed, ~109.8 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     21/150      5.88G      1.221      0.705    0.01548         71        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.771      0.759      0.816      0.445
[progress] epoch 21/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 17.6 min elapsed, ~108.4 min left
[progress] epoch 21/150 | mAP50 0.816 | mAP50-95 0.445 | P 0.771 R 0.759 | 17.7 min elapsed, ~108.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     22/150      5.88G      1.215     0.6893    0.01553         52        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.8s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.4s0.3s
                   all        870       2060      0.734      0.747      0.752      0.417
[progress] epoch 22/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 18.4 min elapsed, ~107.0 min left
[progress] epoch 22/150 | mAP50 0.752 | mAP50-95 0.417 | P 0.734 R 0.747 | 18.4 min elapsed, ~107.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     23/150      5.88G      1.219       0.69    0.01541         66        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.0it/s 3.5s0.3s
                   all        870       2060      0.702      0.789      0.784      0.426
[progress] epoch 23/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 19.1 min elapsed, ~105.6 min left
[progress] epoch 23/150 | mAP50 0.784 | mAP50-95 0.426 | P 0.702 R 0.789 | 19.1 min elapsed, ~105.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     24/150      5.88G      1.174     0.6683    0.01505         86        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.728       0.74       0.76      0.411
[progress] epoch 24/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 19.9 min elapsed, ~104.4 min left
[progress] epoch 24/150 | mAP50 0.760 | mAP50-95 0.411 | P 0.728 R 0.740 | 19.9 min elapsed, ~104.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     25/150      5.88G      1.205     0.6738    0.01513         67        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.0it/s 3.5s0.3s
                   all        870       2060      0.777      0.755      0.779      0.439
[progress] epoch 25/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 20.6 min elapsed, ~103.2 min left
[progress] epoch 25/150 | mAP50 0.779 | mAP50-95 0.439 | P 0.777 R 0.755 | 20.6 min elapsed, ~103.2 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     26/150      5.88G      1.198     0.6733    0.01506         70        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.726      0.746      0.767      0.428
[progress] epoch 26/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 21.4 min elapsed, ~102.0 min left
[progress] epoch 26/150 | mAP50 0.767 | mAP50-95 0.428 | P 0.726 R 0.746 | 21.4 min elapsed, ~102.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     27/150      5.88G      1.157     0.6272    0.01487         93        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.768       0.81      0.802      0.436
[progress] epoch 27/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 22.1 min elapsed, ~100.8 min left
[progress] epoch 27/150 | mAP50 0.802 | mAP50-95 0.436 | P 0.768 R 0.810 | 22.1 min elapsed, ~100.9 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     28/150      5.88G      1.184     0.6473    0.01517         78        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.733      0.761      0.757      0.432
[progress] epoch 28/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 22.9 min elapsed, ~99.7 min left
[progress] epoch 28/150 | mAP50 0.757 | mAP50-95 0.432 | P 0.733 R 0.761 | 22.9 min elapsed, ~99.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     29/150      5.88G       1.18     0.6505    0.01509         87        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.732      0.792      0.779      0.441
[progress] epoch 29/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 23.6 min elapsed, ~98.6 min left
[progress] epoch 29/150 | mAP50 0.779 | mAP50-95 0.441 | P 0.732 R 0.792 | 23.6 min elapsed, ~98.6 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     30/150      5.88G      1.155     0.6136     0.0149         64        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.716      0.763      0.748      0.426
[progress] epoch 30/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 24.4 min elapsed, ~97.5 min left
[progress] epoch 30/150 | mAP50 0.749 | mAP50-95 0.426 | P 0.716 R 0.763 | 24.4 min elapsed, ~97.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     31/150      5.88G      1.154      0.621    0.01484         74        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.734      0.765      0.781      0.434
[progress] epoch 31/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 25.1 min elapsed, ~96.4 min left
[progress] epoch 31/150 | mAP50 0.781 | mAP50-95 0.434 | P 0.734 R 0.765 | 25.1 min elapsed, ~96.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     32/150      5.88G      1.152     0.5947    0.01456         75        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.717       0.75      0.777      0.428
[progress] epoch 32/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 25.9 min elapsed, ~95.4 min left
[progress] epoch 32/150 | mAP50 0.777 | mAP50-95 0.428 | P 0.717 R 0.750 | 25.9 min elapsed, ~95.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     33/150      5.88G      1.146     0.6138    0.01468         48        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.743       0.81      0.797      0.448
[progress] epoch 33/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 26.6 min elapsed, ~94.4 min left
[progress] epoch 33/150 | mAP50 0.797 | mAP50-95 0.448 | P 0.743 R 0.810 | 26.6 min elapsed, ~94.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     34/150      5.88G      1.137     0.6199    0.01448         83        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060       0.78      0.765      0.801      0.459
[progress] epoch 34/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 27.4 min elapsed, ~93.3 min left
[progress] epoch 34/150 | mAP50 0.801 | mAP50-95 0.459 | P 0.780 R 0.765 | 27.4 min elapsed, ~93.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     35/150      5.88G      1.129     0.5882    0.01413         83        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060       0.74      0.752      0.772      0.437
[progress] epoch 35/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 28.1 min elapsed, ~92.3 min left
[progress] epoch 35/150 | mAP50 0.772 | mAP50-95 0.437 | P 0.740 R 0.752 | 28.1 min elapsed, ~92.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     36/150      5.88G      1.118     0.5944    0.01419         64        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.8s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 3.8it/s 3.7s0.3s
                   all        870       2060      0.769      0.738      0.746      0.423
[progress] epoch 36/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 28.9 min elapsed, ~91.4 min left
[progress] epoch 36/150 | mAP50 0.746 | mAP50-95 0.423 | P 0.769 R 0.738 | 28.9 min elapsed, ~91.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     37/150      5.88G      1.117     0.6104    0.01435         86        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.715      0.754      0.777      0.434
[progress] epoch 37/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 29.6 min elapsed, ~90.4 min left
[progress] epoch 37/150 | mAP50 0.777 | mAP50-95 0.434 | P 0.716 R 0.754 | 29.6 min elapsed, ~90.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     38/150      5.88G      1.121     0.5624    0.01409         68        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.751      0.746      0.754      0.424
[progress] epoch 38/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 30.4 min elapsed, ~89.5 min left
[progress] epoch 38/150 | mAP50 0.754 | mAP50-95 0.424 | P 0.751 R 0.746 | 30.4 min elapsed, ~89.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     39/150      5.88G      1.126     0.5902     0.0142         70        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.728      0.752      0.754      0.437
[progress] epoch 39/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 31.1 min elapsed, ~88.5 min left
[progress] epoch 39/150 | mAP50 0.754 | mAP50-95 0.437 | P 0.728 R 0.752 | 31.1 min elapsed, ~88.6 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     40/150      5.88G      1.108     0.5776    0.01427         70        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.744      0.734      0.755       0.43
[progress] epoch 40/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 31.9 min elapsed, ~87.6 min left
[progress] epoch 40/150 | mAP50 0.755 | mAP50-95 0.430 | P 0.744 R 0.734 | 31.9 min elapsed, ~87.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     41/150      5.88G      1.083       0.56    0.01364         60        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.793      0.782      0.797      0.454
[progress] epoch 41/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 32.6 min elapsed, ~86.7 min left
[progress] epoch 41/150 | mAP50 0.797 | mAP50-95 0.454 | P 0.793 R 0.782 | 32.6 min elapsed, ~86.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     42/150      5.88G      1.069     0.5623    0.01367         57        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.776      0.788      0.793      0.456
[progress] epoch 42/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 33.4 min elapsed, ~85.8 min left
[progress] epoch 42/150 | mAP50 0.793 | mAP50-95 0.456 | P 0.776 R 0.788 | 33.4 min elapsed, ~85.9 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     43/150      5.88G      1.072     0.5632    0.01332         78        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.712      0.756      0.725      0.422
[progress] epoch 43/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 34.1 min elapsed, ~85.0 min left
[progress] epoch 43/150 | mAP50 0.725 | mAP50-95 0.422 | P 0.713 R 0.756 | 34.2 min elapsed, ~85.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     44/150      5.88G      1.085     0.5559    0.01374         72        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.5s0.3s
                   all        870       2060      0.753      0.719      0.771       0.44
[progress] epoch 44/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 34.9 min elapsed, ~84.1 min left
[progress] epoch 44/150 | mAP50 0.771 | mAP50-95 0.440 | P 0.753 R 0.719 | 34.9 min elapsed, ~84.1 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     45/150      5.88G      1.068     0.5351    0.01331         59        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.693      0.799      0.789      0.439
[progress] epoch 45/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 35.7 min elapsed, ~83.2 min left
[progress] epoch 45/150 | mAP50 0.789 | mAP50-95 0.439 | P 0.693 R 0.799 | 35.7 min elapsed, ~83.2 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     46/150      5.88G      1.072     0.5325    0.01338         85        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.8s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.708      0.751      0.747      0.439
[progress] epoch 46/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 36.4 min elapsed, ~82.3 min left
[progress] epoch 46/150 | mAP50 0.747 | mAP50-95 0.439 | P 0.708 R 0.751 | 36.4 min elapsed, ~82.3 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     47/150      5.88G      1.071     0.5594    0.01348         82        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.774      0.739      0.761      0.434
[progress] epoch 47/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 37.2 min elapsed, ~81.5 min left
[progress] epoch 47/150 | mAP50 0.761 | mAP50-95 0.434 | P 0.774 R 0.739 | 37.2 min elapsed, ~81.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     48/150      5.88G      1.061     0.5465    0.01311         68        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.781      0.751      0.793      0.445
[progress] epoch 48/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 37.9 min elapsed, ~80.6 min left
[progress] epoch 48/150 | mAP50 0.793 | mAP50-95 0.445 | P 0.781 R 0.751 | 37.9 min elapsed, ~80.6 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     49/150      5.88G      1.066      0.524    0.01328         83        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.772      0.773      0.799      0.455
[progress] epoch 49/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 38.7 min elapsed, ~79.7 min left
[progress] epoch 49/150 | mAP50 0.799 | mAP50-95 0.455 | P 0.772 R 0.773 | 38.7 min elapsed, ~79.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     50/150      5.88G      1.044     0.5212    0.01307         62        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.797      0.738       0.78      0.435
[progress] epoch 50/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 39.4 min elapsed, ~78.9 min left
[progress] epoch 50/150 | mAP50 0.780 | mAP50-95 0.435 | P 0.797 R 0.738 | 39.4 min elapsed, ~78.9 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     51/150      5.88G      1.053     0.5481    0.01331         66        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.723       0.79      0.791      0.461
[progress] epoch 51/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 40.2 min elapsed, ~78.0 min left
[progress] epoch 51/150 | mAP50 0.791 | mAP50-95 0.461 | P 0.723 R 0.790 | 40.2 min elapsed, ~78.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     52/150      5.88G      1.041      0.533    0.01259         62        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.2s0.3s
                   all        870       2060      0.796      0.771      0.806      0.473
[progress] epoch 52/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 40.9 min elapsed, ~77.1 min left
[progress] epoch 52/150 | mAP50 0.806 | mAP50-95 0.473 | P 0.796 R 0.771 | 40.9 min elapsed, ~77.1 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     53/150      5.88G      1.035     0.5208    0.01283         85        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060       0.72      0.805      0.787       0.45
[progress] epoch 53/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 41.7 min elapsed, ~76.2 min left
[progress] epoch 53/150 | mAP50 0.787 | mAP50-95 0.450 | P 0.720 R 0.805 | 41.7 min elapsed, ~76.2 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     54/150      5.88G      1.032     0.5101    0.01305         73        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.745      0.758      0.792      0.457
[progress] epoch 54/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 42.4 min elapsed, ~75.4 min left
[progress] epoch 54/150 | mAP50 0.792 | mAP50-95 0.457 | P 0.745 R 0.758 | 42.4 min elapsed, ~75.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     55/150      5.88G      1.043     0.5331    0.01296         73        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.4it/s 3.2s0.3s
                   all        870       2060      0.757      0.736      0.784      0.461
[progress] epoch 55/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 43.1 min elapsed, ~74.5 min left
[progress] epoch 55/150 | mAP50 0.784 | mAP50-95 0.461 | P 0.757 R 0.736 | 43.2 min elapsed, ~74.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     56/150      5.88G       1.02     0.4943    0.01254         58        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.768      0.742      0.767      0.437
[progress] epoch 56/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 43.9 min elapsed, ~73.7 min left
[progress] epoch 56/150 | mAP50 0.767 | mAP50-95 0.437 | P 0.768 R 0.742 | 43.9 min elapsed, ~73.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     57/150      5.88G       1.02     0.5174    0.01289         72        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060      0.751      0.725       0.76      0.437
[progress] epoch 57/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 44.6 min elapsed, ~72.8 min left
[progress] epoch 57/150 | mAP50 0.760 | mAP50-95 0.437 | P 0.751 R 0.725 | 44.6 min elapsed, ~72.8 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     58/150      5.88G      1.029     0.4946    0.01257         67        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.4s0.3s
                   all        870       2060      0.776      0.769      0.791      0.455
[progress] epoch 58/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 45.4 min elapsed, ~72.0 min left
[progress] epoch 58/150 | mAP50 0.791 | mAP50-95 0.455 | P 0.776 R 0.769 | 45.4 min elapsed, ~72.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     59/150      5.88G       1.01     0.4923    0.01244         49        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.5it/s 3.1s0.2s
                   all        870       2060      0.777      0.753       0.79      0.453
[progress] epoch 59/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 46.1 min elapsed, ~71.1 min left
[progress] epoch 59/150 | mAP50 0.790 | mAP50-95 0.453 | P 0.777 R 0.753 | 46.1 min elapsed, ~71.2 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     60/150      5.88G      1.003     0.4816    0.01242         54        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.3s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.0it/s 3.5s0.3s
                   all        870       2060      0.781       0.75      0.778      0.435
[progress] epoch 60/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 46.9 min elapsed, ~70.3 min left
[progress] epoch 60/150 | mAP50 0.778 | mAP50-95 0.434 | P 0.781 R 0.750 | 46.9 min elapsed, ~70.3 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     61/150      5.88G      1.001     0.4938    0.01264         58        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.791      0.798      0.809      0.448
[progress] epoch 61/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 47.6 min elapsed, ~69.5 min left
[progress] epoch 61/150 | mAP50 0.809 | mAP50-95 0.448 | P 0.791 R 0.798 | 47.6 min elapsed, ~69.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     62/150      5.88G     0.9929     0.4901    0.01235         96        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.4it/s 3.2s0.2s
                   all        870       2060      0.772      0.761      0.784      0.452
[progress] epoch 62/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 48.4 min elapsed, ~68.7 min left
[progress] epoch 62/150 | mAP50 0.784 | mAP50-95 0.452 | P 0.772 R 0.761 | 48.4 min elapsed, ~68.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     63/150      5.88G     0.9842     0.4771    0.01207         78        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060       0.78      0.754      0.785      0.458
[progress] epoch 63/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 49.1 min elapsed, ~67.9 min left
[progress] epoch 63/150 | mAP50 0.785 | mAP50-95 0.458 | P 0.780 R 0.754 | 49.1 min elapsed, ~67.9 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     64/150      5.88G     0.9832     0.4627    0.01196         58        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.6s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.759      0.755      0.772      0.453
[progress] epoch 64/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 49.9 min elapsed, ~67.0 min left
[progress] epoch 64/150 | mAP50 0.772 | mAP50-95 0.453 | P 0.759 R 0.755 | 49.9 min elapsed, ~67.0 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     65/150      5.88G     0.9712     0.4718    0.01205         66        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.794      0.747      0.767      0.445
[progress] epoch 65/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 50.6 min elapsed, ~66.2 min left
[progress] epoch 65/150 | mAP50 0.767 | mAP50-95 0.445 | P 0.794 R 0.747 | 50.6 min elapsed, ~66.2 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     66/150      5.88G     0.9849     0.4932    0.01228         75        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.0it/s 3.5s0.3s
                   all        870       2060      0.777      0.741      0.771      0.448
[progress] epoch 66/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 51.4 min elapsed, ~65.4 min left
[progress] epoch 66/150 | mAP50 0.771 | mAP50-95 0.448 | P 0.777 R 0.741 | 51.4 min elapsed, ~65.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     67/150      5.88G     0.9652      0.461    0.01168         84        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060       0.77      0.767       0.78      0.447
[progress] epoch 67/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 52.1 min elapsed, ~64.5 min left
[progress] epoch 67/150 | mAP50 0.780 | mAP50-95 0.447 | P 0.769 R 0.767 | 52.1 min elapsed, ~64.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     68/150      5.88G      0.958     0.4552    0.01156         68        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.4it/s 3.2s0.3s
                   all        870       2060      0.807      0.762      0.771      0.455
[progress] epoch 68/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 52.8 min elapsed, ~63.7 min left
[progress] epoch 68/150 | mAP50 0.771 | mAP50-95 0.455 | P 0.807 R 0.762 | 52.8 min elapsed, ~63.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     69/150      5.88G     0.9809     0.4761    0.01222         64        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.796      0.735      0.774      0.444
[progress] epoch 69/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 53.6 min elapsed, ~62.9 min left
[progress] epoch 69/150 | mAP50 0.774 | mAP50-95 0.444 | P 0.796 R 0.735 | 53.6 min elapsed, ~62.9 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     70/150      5.88G     0.9613      0.446     0.0119         72        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 40.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.784      0.752      0.781      0.446
[progress] epoch 70/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 54.3 min elapsed, ~62.1 min left
[progress] epoch 70/150 | mAP50 0.781 | mAP50-95 0.446 | P 0.784 R 0.751 | 54.3 min elapsed, ~62.1 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     71/150      5.88G     0.9719     0.4504    0.01175         65        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.2s
                   all        870       2060      0.791      0.772      0.789      0.459
[progress] epoch 71/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 55.1 min elapsed, ~61.3 min left
[progress] epoch 71/150 | mAP50 0.789 | mAP50-95 0.459 | P 0.791 R 0.772 | 55.1 min elapsed, ~61.3 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     72/150      5.88G     0.9469     0.4468    0.01156         73        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.757      0.796      0.786      0.459
[progress] epoch 72/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 55.8 min elapsed, ~60.5 min left
[progress] epoch 72/150 | mAP50 0.786 | mAP50-95 0.459 | P 0.757 R 0.796 | 55.8 min elapsed, ~60.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     73/150      5.88G     0.9509     0.4485    0.01147         65        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.1s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.776      0.749      0.774      0.445
[progress] epoch 73/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 56.6 min elapsed, ~59.7 min left
[progress] epoch 73/150 | mAP50 0.774 | mAP50-95 0.445 | P 0.776 R 0.749 | 56.6 min elapsed, ~59.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     74/150      5.88G      0.952     0.4444    0.01151         62        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.7s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.791      0.741      0.768      0.443
[progress] epoch 74/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 57.3 min elapsed, ~58.9 min left
[progress] epoch 74/150 | mAP50 0.768 | mAP50-95 0.443 | P 0.791 R 0.741 | 57.3 min elapsed, ~58.9 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     75/150      5.88G     0.9385     0.4411    0.01129         59        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.8s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.781      0.798      0.775      0.449
[progress] epoch 75/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 58.1 min elapsed, ~58.1 min left
[progress] epoch 75/150 | mAP50 0.775 | mAP50-95 0.449 | P 0.781 R 0.797 | 58.1 min elapsed, ~58.1 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     76/150      5.88G     0.9483      0.438     0.0115         76        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.5s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060      0.765       0.75      0.754      0.441
[progress] epoch 76/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 58.8 min elapsed, ~57.3 min left
[progress] epoch 76/150 | mAP50 0.754 | mAP50-95 0.441 | P 0.765 R 0.750 | 58.9 min elapsed, ~57.3 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     77/150      5.88G     0.9271     0.4297    0.01134         94        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.5s0.4s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.0it/s 3.5s0.3s
                   all        870       2060      0.787      0.779      0.782      0.449
[progress] epoch 77/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 59.6 min elapsed, ~56.5 min left
[progress] epoch 77/150 | mAP50 0.782 | mAP50-95 0.449 | P 0.787 R 0.779 | 59.6 min elapsed, ~56.5 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     78/150      5.88G     0.9311      0.431    0.01132         71        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.4s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.3s0.3s
                   all        870       2060      0.804      0.761      0.781      0.448
[progress] epoch 78/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 60.4 min elapsed, ~55.7 min left
[progress] epoch 78/150 | mAP50 0.781 | mAP50-95 0.448 | P 0.804 R 0.761 | 60.4 min elapsed, ~55.7 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     79/150      5.88G     0.9339      0.439     0.0113         80        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.0s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.3it/s 3.3s0.3s
                   all        870       2060       0.76      0.773       0.78      0.446
[progress] epoch 79/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 61.1 min elapsed, ~54.9 min left
[progress] epoch 79/150 | mAP50 0.780 | mAP50-95 0.446 | P 0.760 R 0.773 | 61.1 min elapsed, ~54.9 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     80/150      5.88G     0.9272     0.4249    0.01136         86        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 40.8s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060       0.76      0.758      0.786      0.448
[progress] epoch 80/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 61.9 min elapsed, ~54.1 min left
[progress] epoch 80/150 | mAP50 0.785 | mAP50-95 0.448 | P 0.760 R 0.758 | 61.9 min elapsed, ~54.1 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     81/150      5.88G     0.9158     0.4275    0.01082         74        640: 100% ━━━━━━━━━━━━ 127/127 3.1it/s 41.2s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.1it/s 3.4s0.3s
                   all        870       2060       0.79      0.721      0.762      0.444
[progress] epoch 81/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 62.6 min elapsed, ~53.4 min left
[progress] epoch 81/150 | mAP50 0.762 | mAP50-95 0.444 | P 0.790 R 0.721 | 62.6 min elapsed, ~53.4 min left

      Epoch    GPU_mem   box_loss   cls_loss    l1_loss  Instances       Size
     82/150      5.88G     0.9088     0.4124    0.01093         69        640: 100% ━━━━━━━━━━━━ 127/127 3.2it/s 39.9s0.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 4.2it/s 3.4s0.3s
                   all        870       2060      0.811      0.761      0.794      0.456
EarlyStopping: Training stopped early as no improvement observed in last 30 epochs. Best results observed at epoch 52, best model saved as best.pt.
To update EarlyStopping(patience=30) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
[progress] epoch 82/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 63.4 min elapsed, ~52.6 min left
[progress] epoch 82/150 | mAP50 0.794 | mAP50-95 0.456 | P 0.811 R 0.761 | 63.4 min elapsed, ~52.6 min left

82 epochs completed in 1.033 hours.
Optimizer stripped from /kaggle/working/uaal_runs/yolo26s/weights/last.pt, 20.3MB
Optimizer stripped from /kaggle/working/uaal_runs/yolo26s/weights/best.pt, 20.3MB

Validating /kaggle/working/uaal_runs/yolo26s/weights/best.pt...
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 14/14 1.1it/s 12.9s0.3s
[progress] epoch 83/150 | mAP50 0.000 | mAP50-95 0.000 | P 0.000 R 0.000 | 63.7 min elapsed, ~51.4 min left
                   all        870       2060      0.795      0.771      0.806      0.473
            epiglottis        379        379      0.969      0.978      0.991       0.66
               glottis        394        394      0.819      0.794      0.793      0.493
      glottic_aperture        424        425      0.915      0.868      0.934       0.65
       left_vocal_fold        349        349      0.774      0.805      0.802      0.435
      right_vocal_fold        320        320      0.797      0.816      0.862      0.491
               trachea         72         72      0.649      0.597      0.697      0.349
               channel         47         47      0.619      0.617      0.585      0.206
                  nose         24         24      0.683      0.917      0.889       0.55
               nostril         46         50      0.932      0.544      0.701      0.425
Speed: 0.2ms preprocess, 12.0ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to /kaggle/working/uaal_runs/yolo26s
[progress] best.pt final validation | mAP50 0.806 | mAP50-95 0.473
[train] done in 63.8 min - evaluating /kaggle/working/uaal_runs/yolo26s/weights/best.pt
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 371.8±153.7 MB/s, size: 213.6 KB)
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels/val... 870 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 870/870 1.1Kit/s 0.8s0.1ss
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 55/55 4.3it/s 12.9s0.2s
                   all        870       2060      0.796      0.771      0.806      0.472
Speed: 0.9ms preprocess, 9.8ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to /kaggle/working/uaal_runs/yolo26s/eval_val_all
[eval] val  all      mAP50 0.806  mAP50-95 0.472
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.2±0.3 ms, read: 354.3±109.0 MB/s, size: 227.5 KB)
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels/val... 487 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 487/487 1.1Kit/s 0.4s<0.1s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 31/31 5.1it/s 6.1s0.2s
                   all        487       1432      0.883      0.812      0.867      0.527
Speed: 0.7ms preprocess, 8.7ms inference, 0.0ms loss, 0.9ms postprocess per image
[eval] val  clinical mAP50 0.867  mAP50-95 0.527
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 97.6±48.3 MB/s, size: 70.2 KB)
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels/val... 383 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 383/383 1.1Kit/s 0.3s<0.1s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 4.0it/s 6.0s0.2s
                   all        383        628      0.675      0.729      0.706      0.417
Speed: 0.8ms preprocess, 11.9ms inference, 0.0ms loss, 0.8ms postprocess per image
[eval] val  phantom  mAP50 0.706  mAP50-95 0.417
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs
WARNING ⚠️ val: Slow image access detected (ping: 0.0±0.0 ms, read: 6.0±1.8 MB/s, size: 233.5 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels/test... 899 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 899/899 51.8it/s 17.4s0.1s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 57/57 4.4it/s 13.1s0.2s
                   all        899       2058      0.683      0.731      0.723      0.439
Speed: 0.9ms preprocess, 9.4ms inference, 0.0ms loss, 1.2ms postprocess per image
Results saved to /kaggle/working/uaal_runs/yolo26s/eval_test_all
[eval] test all      mAP50 0.723  mAP50-95 0.439
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 371.6±98.7 MB/s, size: 228.3 KB)
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels/test... 576 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 576/576 1.1Kit/s 0.5s<0.0s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 36/36 5.3it/s 6.8s0.2s
                   all        576       1516       0.75      0.779      0.784      0.491
Speed: 0.6ms preprocess, 8.2ms inference, 0.0ms loss, 0.8ms postprocess per image
[eval] test clinical mAP50 0.784  mAP50-95 0.491
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14912MiB)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 98.2±22.6 MB/s, size: 55.7 KB)
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels/test... 323 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 323/323 1.1Kit/s 0.3s<0.2s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/lists/../labels is not writable, cache not saved.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 21/21 4.1it/s 5.1s0.2s
                   all        323        542       0.64      0.746      0.716      0.428
Speed: 0.8ms preprocess, 12.0ms inference, 0.0ms loss, 0.8ms postprocess per image
[eval] test phantom  mAP50 0.716  mAP50-95 0.428
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CPU (Intel Xeon CPU @ 2.00GHz)
💡 ProTip: Export to OpenVINO format for best performance on Intel hardware. Learn more at https://docs.ultralytics.com/integrations/openvino
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs

PyTorch: starting from '/kaggle/working/uaal_runs/yolo26s/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 13, 8400) (19.4 MB)

ONNX: starting export with onnx 1.23.1 opset 18...
ONNX: slimming with onnxslim 0.1.97...
ONNX: export success ✅ 2.9s, saved as '/kaggle/working/uaal_runs/yolo26s/weights/best.onnx' (36.4 MB)

Export complete (3.5s)
Results saved to /kaggle/working/uaal_runs/yolo26s/weights/best.onnx
Predict:         yolo predict task=detect model=/kaggle/working/uaal_runs/yolo26s/weights/best.onnx imgsz=640 
Validate:        yolo val task=detect model=/kaggle/working/uaal_runs/yolo26s/weights/best.onnx imgsz=640 data=/kaggle/working/uaal_runs/data_abs.yaml  
Visualize:       https://netron.app
[export] onnx: /kaggle/working/uaal_runs/yolo26s/weights/best.onnx
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CPU (Intel Xeon CPU @ 2.00GHz)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs

PyTorch: starting from '/kaggle/working/uaal_runs/yolo26s/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 13, 8400) (19.4 MB)

OpenVINO: starting export with openvino 2026.4.1-22982-e213a147257-releases/2026/4...
OpenVINO: export success ✅ 3.8s, saved as '/kaggle/working/uaal_runs/yolo26s/weights/best_openvino_model/' (36.6 MB)

Export complete (4.4s)
Results saved to /kaggle/working/uaal_runs/yolo26s/weights/best_openvino_model
Predict:         yolo predict task=detect model=/kaggle/working/uaal_runs/yolo26s/weights/best_openvino_model/ imgsz=640 
Validate:        yolo val task=detect model=/kaggle/working/uaal_runs/yolo26s/weights/best_openvino_model/ imgsz=640 data=/kaggle/working/uaal_runs/data_abs.yaml  
Visualize:       https://netron.app
[export] openvino_fp32: /kaggle/working/uaal_runs/yolo26s/weights/best_openvino_model/
WARNING ⚠️ 'int8' is deprecated and will be removed in the future. Use 'quantize' instead.
Ultralytics 8.4.173 🚀 Python-3.13.15 torch-2.11.0+cu128 CPU (Intel Xeon CPU @ 2.00GHz)
YOLO26s summary (fused): 120 layers, 9,468,663 parameters, 0 gradients, 20.8 GFLOPs

PyTorch: starting from '/kaggle/working/uaal_runs/yolo26s/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 13, 8400) (19.4 MB)
OpenVINO: collecting INT8 calibration images from 'data=/kaggle/working/uaal_runs/data_abs.yaml'
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 377.9±167.9 MB/s, size: 233.9 KB)
val: Scanning /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels/val... 218 images, 0 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 218/218 997.2it/s 0.2s0.0s
WARNING ⚠️ val: Cache directory /kaggle/input/datasets/yogpandya/uaal-dataset/uaal_det/labels is not writable, cache not saved.
WARNING ⚠️ OpenVINO: >300 images recommended for INT8 calibration, found 218 images.

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
Statistics collection ━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 218/218 • 0:00:58 • 0:00:0000:0100:03
Applying Fast Bias correction ━━━━━━━━━━━━━━━━━━━ 100% 94/94 • 0:00:03 • 0:00:0000:0100:01━━━━━━━━━━━━━━━━━━━━━━━━━   • 0:00:00  
OpenVINO: export success ✅ 73.6s, saved as '/kaggle/working/uaal_runs/yolo26s/weights/best_int8_openvino_model/' (10.0 MB)

Export complete (74.1s)
Results saved to /kaggle/working/uaal_runs/yolo26s/weights/best_int8_openvino_model
Predict:         yolo predict task=detect model=/kaggle/working/uaal_runs/yolo26s/weights/best_int8_openvino_model/ imgsz=640 
Validate:        yolo val task=detect model=/kaggle/working/uaal_runs/yolo26s/weights/best_int8_openvino_model/ imgsz=640 data=/kaggle/working/uaal_runs/data_abs.yaml  
Visualize:       https://netron.app
[export] openvino_int8: /kaggle/working/uaal_runs/yolo26s/weights/best_int8_openvino_model/
[cpu] pytorch: 124.4 ms/img (8.0 FPS)
Loading /kaggle/working/uaal_runs/yolo26s/weights/best.onnx for ONNX Runtime inference...
Using ONNX Runtime 1.30.0 with CPUExecutionProvider
[cpu] onnx: 122.1 ms/img (8.2 FPS)
Loading /kaggle/working/uaal_runs/yolo26s/weights/best_openvino_model/ for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on CPU...
[cpu] openvino_fp32: 120.8 ms/img (8.3 FPS)
Loading /kaggle/working/uaal_runs/yolo26s/weights/best_int8_openvino_model/ for OpenVINO inference...
Using OpenVINO LATENCY mode for batch=1 inference on CPU...
[cpu] openvino_int8: 92.1 ms/img (10.9 FPS)
[done] yolo26s: report at /kaggle/working/uaal_runs/yolo26s/report.md
[done] download /kaggle/working/uaal_runs.zip
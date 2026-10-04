# Dataset Analysis — UAAL and BM-BronchoLC

Analysis of the datasets in `dataset/`, done to plan the two separate models:

- **Upper-airway model** trained on UAAL → `uaal_upper_airway_model/`
- **Lower-airway model** trained on BM-BronchoLC → `bm_broncholc_lower_airway_model/`

All numbers below were computed directly from the downloaded files (annotation JSONs and image files), not copied from the papers. Where the papers report numbers, they match.

---

## 1. Summary

| | UAAL clinical | UAAL phantom | BM-BronchoLC |
|---|---|---|---|
| Location | `dataset/26342779/coco_ins_clinical.zip` | `dataset/26342779/coco_ins_phantom.zip` | `dataset/Lung_cancer/`, `dataset/Non_lung_cancer/` |
| Images | 3,814 (2,683 train / 1,131 val) | 2,746 (2,267 train / 479 val) | 2,606 (1,348 cancer / 1,258 non-cancer) |
| Annotations | 10,330 | 4,526 | 6,225 polygons (6,426 class labels) |
| Classes (with data) | 8 | 9 | 18 (11 anatomy + 7 lesion) |
| Format | COCO instance (JSON) | COCO instance (JSON) | Custom JSON (polygons, no boxes) |
| Masks | Partial (only some classes) | Partial (only some classes) | All annotations are polygons |
| Image type | PNG, white light, full frame | PNG, white light, 400×400 | PNG, blue-tinted (NBI-like), octagonal scope view with on-screen text |
| Official split | Yes (train/val) | Yes (train/val) | **No**, must split by patient |
| Anatomy covered | Nose → nostrils → epiglottis → glottis → vocal folds | Nose → nostrils → glottis → vocal folds → trachea | Vocal cords → trachea → carina → main and lobar bronchi |

---

## 2. UAAL — Upper Airway Anatomical Landmark Dataset

### 2.1 Files and format

```
dataset/26342779/
├── coco_ins_clinical.zip        (train2017/ 2,683 PNG, val2017/ 1,131 PNG, annotations/instances_{train,val}2017.json)
├── coco_ins_phantom.zip         (train2017/ 2,267 PNG, val2017/   479 PNG, annotations/instances_{train,val}2017.json)
└── annotation visualization.zip (vis-data.py + ~45 sample visualisations, box and mask)
```

- Standard COCO keys: `licenses, info, categories, images, annotations`.
- Each annotation has `id, image_id, category_id, segmentation, area, bbox, iscrowd, attributes`.
- `bbox` is COCO `[x, y, w, h]` in pixels. `iscrowd` is always 0. `attributes` contains `occluded` (always False) and `rotation`.
- `segmentation` is a list of polygons, or an **empty list** for classes that only have boxes (see 2.4).
- The category IDs **differ between clinical and phantom** (e.g. id 1 = "Right vocal fold" in clinical, "Nose" in phantom). Always map by name, never by id.
- Every image has at least one annotation (no background-only images).

### 2.2 Categories

| Class | Clinical id | Phantom id | Present in clinical | Present in phantom |
|---|---|---|---|---|
| Nose | 8 | 1 | yes (few) | yes (few) |
| Right nostril | 6 | 2 | yes (few) | yes (very few) |
| Left nostril | 7 | 3 | yes (few) | yes (few) |
| Epiglottis | 5 | — | yes | **no** |
| Glottis | 4 | 7 | yes | yes |
| Glottic aperture | 3 | 6 | yes | yes |
| Right vocal fold | 1 | 4 | yes | yes |
| Left vocal fold | 2 | 5 | yes | yes |
| Trachea | 9 | 8 | **declared but 0 annotations** | yes |
| Channel | 10 | 9 | **declared but 0 annotations** | yes |

### 2.3 Annotation and image counts per class

Number of annotations; image count is identical except Glottic aperture in clinical train (1,040 images), so every class appears at most once per image.

| Class | Clinical train | Clinical val | Clinical total | Phantom train | Phantom val | Phantom total | **UAAL total** |
|---|---|---|---|---|---|---|---|
| Epiglottis | 2,003 | 680 | 2,683 | — | — | — | **2,683** |
| Glottis | 1,821 | 742 | 2,563 | 484 | 102 | 586 | **3,149** |
| Glottic aperture | 1,041 | 490 | 1,531 | 1,059 | 251 | 1,310 | **2,841** |
| Left vocal fold | 1,043 | 532 | 1,575 | 808 | 196 | 1,004 | **2,579** |
| Right vocal fold | 1,015 | 513 | 1,528 | 413 | 137 | 550 | **2,078** |
| Trachea | 0 | 0 | 0 | 451 | 88 | 539 | **539** |
| Channel | 0 | 0 | 0 | 299 | 53 | 352 | **352** |
| Nose | 30 | 144 | 174 | 61 | 13 | 74 | **248** |
| Right nostril | 36 | 140 | 176 | 12 | 4 | 16 | **192** |
| Left nostril | 46 | 54 | 100 | 78 | 17 | 95 | **195** |
| **Total** | **7,035** | **3,295** | **10,330** | **3,665** | **861** | **4,526** | **14,856** |

Observations:
- Glottic structures (epiglottis, glottis, aperture, vocal folds) are well represented (2,000–3,000 each).
- **Nose and nostrils are rare**, and in clinical data most of them are in **val** (144 Nose in val vs 30 in train), so the official split is unbalanced for these classes. Phantom right nostril has only 16 examples in total.
- **Trachea and Channel exist only in the phantom subset.**

### 2.4 Masks vs boxes (important)

Only some classes have segmentation polygons; the rest are **bounding-box only**. This holds for both subsets and both splits:

| Has polygon mask | Box only (empty `segmentation`) |
|---|---|
| Glottic aperture, Left vocal fold, Right vocal fold, Left nostril, Right nostril | Epiglottis, Glottis, Nose, Trachea, Channel |

| Subset / split | Annotations with mask |
|---|---|
| Clinical train | 3,181 / 7,035 (45%) |
| Clinical val | 1,729 / 3,295 (52%) |
| Phantom train | 2,370 / 3,665 (65%) |
| Phantom val | 605 / 861 (70%) |

**Consequence:** UAAL cannot train a full instance-segmentation model for all classes. The UAAL model should be a **detection (bounding box) model**. Masks could only be used for the 5 mask classes as an optional extra.

### 2.5 Image sizes

| Subset | Sizes |
|---|---|
| Clinical train | 1920×1080 (1,856), 1280×720 (556), 720×576 (271) |
| Clinical val | 1920×1080 (728), 1280×720 (403) |
| Phantom (all) | 400×400 (2,746) |

Within each recording group, all images have the same size (i.e. size depends on the recording device).

### 2.6 Annotations per image

| Annotations per image | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Clinical train | 948 | 615 | 131 | 481 | 508 |
| Clinical val | 245 | 291 | 107 | 293 | 195 |
| Phantom train | 1,319 | 546 | 354 | 48 | 0 |
| Phantom val | 247 | 93 | 128 | 11 | 0 |

Most common class combinations per image:
- Clinical: `Epiglottis` alone (774 train), `Epiglottis + Glottis` (570), all five glottic classes together (507), `Glottic aperture + Glottis + L/R vocal fold` (394).
- Phantom: `Glottic aperture + Left vocal fold` (399 train), `Trachea` alone (396), `Glottis` alone (369), `Channel` alone (299).

### 2.7 Object size (bounding-box area as a fraction of the image)

Median (p10 – p90):

| Class | Clinical train | Phantom train |
|---|---|---|
| Nose | 0.334 (0.188 – 0.733) | 0.184 (0.087 – 0.484) |
| Epiglottis | 0.079 (0.027 – 0.293) | — |
| Left / Right nostril | 0.083 / 0.074 | 0.071 / 0.006 |
| Glottis | 0.020 (0.005 – 0.083) | 0.096 (0.046 – 0.404) |
| Glottic aperture | 0.013 (0.002 – 0.044) | 0.172 (0.027 – 0.563) |
| Left vocal fold | 0.007 (0.002 – 0.028) | 0.373 (0.041 – 0.756) |
| Right vocal fold | 0.006 (0.002 – 0.024) | 0.186 (0.052 – 0.807) |
| Trachea | — | 0.037 (0.019 – 0.096) |
| Channel | — | 0.097 (0.045 – 0.346) |

**Consequence:** in the clinical images, the vocal folds and glottic aperture are **very small** (often under 1% of a 1920×1080 frame). Training at a low input size (e.g. 320 px) will lose them; 512–640 px input is needed for the clinical subset. Phantom objects are much larger.

### 2.8 Recording groups and split

File names are `<group>_<frame>.PNG`. The official split keeps groups apart (**no group appears in both train and val**), so it is leakage-free at the group level.

| Subset | Train groups (images) | Val groups (images) |
|---|---|---|
| Clinical | b_AI042 (275), c_AI003 (271), c_AI069 (294), n_AI002 (112), n_AI004 (555), n_AI013 (517), n_AI023 (321), n_AI065 (281), pt_AI019 (57) | c_AI038 (107), n_AI011 (296), n_AI031 (250), n_AI046 (478) |
| Phantom | try1, try2, try3, try4, try6, try7, try9, try10, try12, try13 | try5, try8, try11, try14 |

Note: the paper reports 82 patients for the clinical subset, but the file names only show 13 groups. The groups are probably recording sessions or devices rather than single patients. Patient-level IDs are not available in the files.

Nose/nostril annotations are concentrated in a few groups (clinical val `n_AI046` alone has 137 Nose and 136 Right nostril), which explains the train/val imbalance in 2.3.

### 2.9 Visual characteristics

- White-light (normal colour) endoscopy.
- **Correction:** most clinical frames are **screen captures**, not clean full-frame images. The scope view (octagon or rectangle) covers only 40–90% of the frame (median 58%). The rest is black border with burned-in patient fields, date/time, scope settings, icons and sometimes a picture-in-picture thumbnail. In some recordings the text is drawn **over** the scope image itself.
- The screen layout changes **within** a group, so a group is not one fixed device setup. The crop has to be detected per frame (see `uaal_upper_airway_model/preprocessing.py`).
- Phantom images are from a commercial airway phantom: uniform pink/red plastic, small 400×400 frames, no overlays.

### 2.10 Annotation quirks

- **Rotated boxes:** 1,388 box-only annotations are CVAT rotated rectangles (`attributes.rotation` ≠ 0): 1,330 Epiglottis (half of all Epiglottis boxes), 58 Glottis, 1 Nose. Their `bbox` field is the *unrotated* rectangle and is wrong as given. It must be converted to the box enclosing the rotated rectangle.
- 262 boxes extend outside the image (mostly rotated Epiglottis) and must be clipped.
- Consecutive frames are near-identical: the median thumbnail difference is 9.4 (clinical) / 7.3 (phantom) between consecutive frames vs 40 / 55 between random frames of the same group. 906 consecutive pairs differ by < 2/255. A random frame-level split would therefore leak.

---

## 3. BM-BronchoLC — Bronchoscopy Anatomy and Lung Cancer Lesion Dataset

### 3.1 Files and format

```
dataset/Lung_cancer/Lung_cancer/            (and the same for Non_lung_cancer/Non_lung_cancer/)
├── labels.json        18 class definitions (same IDs in both folders)
├── objects.json       patient → video → list of image_ids
├── annotation.json    list of polygon annotations
└── imgs/<patient_id>/<video_id>/<image_id>.png
```

(`__MACOSX/` and `.DS_Store` files are macOS junk and should be ignored.)

- `labels.json`: `{id, type: "FINDING", scope: "IMAGE", annotation_type: "POLYGON", name}`.
- `objects.json`: `[{id: patient_id, videos: [{video_id, images: [{image_id}]}]}]`. **Exactly 1 video per patient.**
- `annotation.json`: `{finding_id (unique annotation id), object_id (= image_id), data: [{x, y}, ...] polygon in pixels, label_ids: [class ids], study_id}`.
- There are **no bounding boxes**; boxes must be computed from the polygons.
- Polygons are dense: median 140 points (p10 24–44, p90 ~260), so simplify them during conversion.
- One polygon can carry **2 labels** (one anatomy and one lesion on the same region): 182 polygons in Lung_cancer, 19 in Non_lung_cancer. E.g. `Left main bronchus + Mucosal infiltration` (31), `Left main bronchus + Tumor` (15).
- Every image file has annotations, and every annotated image has a file.

### 3.2 Patients, videos and images

| | Lung_cancer | Non_lung_cancer | Total |
|---|---|---|---|
| Patients in `objects.json` | 108 | 104 | 212 |
| Patients with ≥1 image | 106 | 102 | **208** |
| Videos | 108 (1 per patient) | 104 | 212 |
| Images (PNG) | 1,348 | 1,258 | **2,606** |
| Images per patient (mean / min / max) | 12.5 / 0 / 28 | 12.1 / 0 / 33 | ~12 |
| Polygon annotations | 3,290 | 2,935 | **6,225** |

### 3.3 Classes and counts

Counts are per class label (a 2-label polygon counts once for each label).

| Class | Type | Annotations (LC / NLC / total) | Images (LC / NLC / total) | Patients |
|---|---|---|---|---|
| Left main bronchus | Anatomy | 303 / 366 / **669** | 199 / 246 / **445** | 182 |
| Main carina | Anatomy | 239 / 304 / **543** | 163 / 191 / **354** | 178 |
| Left inferior lobar bronchus | Anatomy | 233 / 310 / **543** | 171 / 224 / **395** | 165 |
| Right main bronchus | Anatomy | 247 / 295 / **542** | 175 / 199 / **374** | 175 |
| Right middle lobar bronchus | Anatomy | 203 / 280 / **483** | 134 / 187 / **321** | 155 |
| Right inferior lobar bronchus | Anatomy | 182 / 278 / **460** | 125 / 200 / **325** | 156 |
| Vocal cords | Anatomy | 192 / 260 / **452** | 120 / 149 / **269** | 132 |
| Right superior lobar bronchus | Anatomy | 179 / 268 / **447** | 117 / 202 / **319** | 148 |
| Left superior lobar bronchus | Anatomy | 197 / 202 / **399** | 146 / 144 / **290** | 149 |
| Intermediate bronchus | Anatomy | 142 / 152 / **294** | 98 / 111 / **209** | 122 |
| Trachea | Anatomy | 77 / 82 / **159** | 50 / 75 / **125** | 85 |
| Mucosal infiltration | Lesion | 536 / 10 / **546** | 355 / 9 / **364** | 79 |
| Tumor | Lesion | 229 / 0 / **229** | 183 / 0 / **183** | 40 |
| Mucosal edema of carina | Lesion | 154 / 32 / **186** | 128 / 28 / **156** | 51 |
| Stenosis | Lesion | 145 / 8 / **153** | 111 / 8 / **119** | 49 |
| Vascular growth | Lesion | 150 / 2 / **152** | 111 / 2 / **113** | 41 |
| Muscosal erythema *(sic)* | Lesion | 47 / 55 / **102** | 35 / 42 / **77** | 35 |
| Anthrocosis *(sic)* | Lesion | 17 / 50 / **67** | 13 / 40 / **53** | 15 |

Notes:
- "Muscosal erythema" and "Anthrocosis" are misspelled in the original labels; keep the original strings when parsing and fix them only in our class names.
- Unlike UAAL, the same class can appear **several times in one image** (e.g. 669 Left main bronchus annotations across 445 images).
- **Anatomy** classes are fairly balanced (209–445 images, 120–180 patients each). Only **Trachea** is small (125 images).
- **Lesion** classes are small and come from few patients (Tumor: 183 images from 40 patients; Anthracosis: 15 patients), almost entirely from the Lung_cancer folder.

### 3.4 Image categories

| | Lung_cancer | Non_lung_cancer | Total |
|---|---|---|---|
| Anatomy only | 675 | 1,142 | 1,817 |
| Lesion only | 419 | 55 | 474 |
| Anatomy + lesion | 254 | 61 | 315 |
| **Images with any anatomy** | 929 | 1,203 | **2,132** |
| **Images with any lesion** | 673 | 116 | **789** |

For the anatomy model, the 474 lesion-only images contain no anatomy labels. They can be used as background images, or left out. Note that anatomy may be visible but unlabelled in them.

### 3.5 Annotations per image

| Annotations per image | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|
| Lung_cancer | 498 | 390 | 200 | 136 | 26 | 98 |
| Non_lung_cancer | 476 | 393 | 145 | 126 | 24 | 94 |

### 3.6 Image sizes

| Size | Lung_cancer | Non_lung_cancer |
|---|---|---|
| 480×480 | 1,260 | 1,144 |
| 720×480 | 40 | 114 |
| 352×240 | 48 | 0 |

### 3.7 Object size (polygon bounding-box area as a fraction of the image)

Overall: median 0.090 (p10 0.028 – p90 0.333). Objects are much larger, relative to the frame, than in UAAL clinical.

| Class | Median | p10 – p90 |
|---|---|---|
| Trachea | 0.283 | 0.104 – 0.422 |
| Mucosal infiltration | 0.170 | 0.056 – 0.416 |
| Right superior lobar bronchus | 0.148 | 0.033 – 0.466 |
| Right inferior lobar bronchus | 0.139 | 0.062 – 0.432 |
| Tumor | 0.119 | 0.031 – 0.347 |
| Vocal cords | 0.114 | 0.034 – 0.272 |
| Left inferior lobar bronchus | 0.095 | 0.025 – 0.396 |
| Intermediate bronchus | 0.094 | 0.046 – 0.326 |
| Vascular growth | 0.090 | 0.026 – 0.266 |
| Right middle lobar bronchus | 0.088 | 0.031 – 0.266 |
| Muscosal erythema | 0.084 | 0.035 – 0.300 |
| Left superior lobar bronchus | 0.080 | 0.023 – 0.276 |
| Left main bronchus | 0.076 | 0.035 – 0.207 |
| Mucosal edema of carina | 0.075 | 0.021 – 0.232 |
| Right main bronchus | 0.074 | 0.038 – 0.151 |
| Anthrocosis | 0.054 | 0.014 – 0.152 |
| Main carina | 0.037 | 0.015 – 0.098 |
| Stenosis | 0.032 | 0.009 – 0.200 |

### 3.8 Visual characteristics

- **Colour:** 2,605 of 2,606 images are blue-dominant (mean blue > mean red), i.e. NBI-like or a strongly blue colour profile. They look very different from UAAL white-light images. → Plan: train in **grayscale**.
- **Scope view:** the image is an octagon on a black background.
- **Burned-in text:** patient-info fields, date/time, frame counter and scope settings are printed in the black border (left side, and sometimes icons on the right). Their position varies by image.
  - A test of automatic masking (threshold the dark border → keep the largest bright region → fill holes → crop) removed nearly all the text. Small pieces remained where the text touches the octagon edge. Shrinking the mask by a few pixels should remove those, but that step hasn't been tested yet.
  - No manual work is needed.

### 3.9 Split

- There is no official split.
- Each patient has exactly one video, so a **patient-level split = video-level split**. Use `objects.json` to group images by patient.
- Stratify across the Lung_cancer / Non_lung_cancer folders so both are represented in train/val/test.

### 3.10 Data-quality findings (from `bm_broncholc_lower_airway_model/prepare_data.py`)

- **Duplicate polygons:** 860 of 6,225 polygons (14%) are exact copies of another polygon on the same image, sometimes repeated 4×. Lung_cancer has 693 and Non_lung_cancer 167. They are removed on load, so the real counts are lower than in §3.3 (e.g. 4,314 anatomy polygons remain in the prepared anatomy set).
- **Same recording under two patient IDs:** lc_p027 ↔ lc_p061 and nlc_p055 ↔ nlc_p100 share near-identical frames (2 pixel-identical). Each pair is kept in the same split.
- **Overlays:** besides the side text, some videos show a red warning banner ("WHT BAL incomplete", "Lamp Error") drawn across the view, and in most videos text overlaps the left edge of the view itself. Both are removed (filled from neighbouring tissue); typically 0.7% of the view pixels are filled, at most 8%.
- **Broken frame:** `bm_nlc_p093_f01` is a solid red image that still carries a bronchus label. It is excluded.
- Frames are ~1 fps, so near-duplicates are rare: only 4 frames removed.

---

## 4. Overlap between the two datasets

| Structure | UAAL | BM-BronchoLC |
|---|---|---|
| Vocal folds | `Left vocal fold`, `Right vocal fold` (separate, masks) | `Vocal cords` (single polygon) |
| Trachea | Phantom only (box only), 539 annotations | 159 annotations (polygons) |
| Everything else | Upper airway only | Lower airway + lesions only |

The two datasets meet at the vocal cords → trachea transition. This is the handover point between the two models.

---

## 5. Implications for the two models

### 5.1 UAAL upper-airway model (`uaal_upper_airway_model/`)

- **Task:** object detection (boxes). Half of the classes have no masks.
- **Classes (implemented):** epiglottis, glottis, glottic_aperture, left_vocal_fold, right_vocal_fold, trachea, channel, nose, **nostril** (left + right merged because counts are very low). Nose stays separate because its box contains the nostril boxes.
- **Data:** train on clinical + phantom together, and report val metrics **separately** for clinical and phantom.
- **Split (implemented):** new train/val/test (70/15/15) built from whole recordings, with clinical and phantom in every split and every class in every split. The official split has no test set and puts most nose/nostril examples in val.
- **Input size:** ≥ 512 px because clinical vocal folds and aperture are tiny.
- **Preprocessing (implemented in `uaal_upper_airway_model/prepare_data.py`):** map categories by **name** (IDs differ between subsets); convert rotated boxes; crop each frame to its scope view; grayscale; longest side ≤ 1024; remove near-duplicate consecutive frames.

### 5.2 BM-BronchoLC lower-airway model (`bm_broncholc_lower_airway_model/`)

- **Task:** detection or instance segmentation (polygons are available for every class).
- **Classes (phase 1):** the 11 anatomy classes only. Lesions are an optional later phase (merged into one "abnormal mucosa" class).
- **Preprocessing:** automatic scope-region masking/cropping to remove burned-in text; grayscale; convert polygons → YOLO format (boxes or simplified polygons).
- **2-label polygons:** keep the anatomy label for the anatomy model; the lesion label goes to the lesion phase.
- **Split:** by patient (≈ 70/15/15), stratified across the cancer / non-cancer folders.
- **Input size:** 416–512 px is enough (objects are relatively large).

### 5.3 Shared

- Both models output to a common landmark vocabulary for the navigation state machine. `vocal_cords` and `trachea` are the shared handover classes.
- Later (joint model): use each model to pseudo-label the missing classes in the other dataset (see project discussion).

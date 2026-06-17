 # 🛰️ Gaza Damage Assessment System Using Deep Learning

A deep learning framework for assessing building damage in Al-Shati Camp, Gaza, using satellite imagery captured before and after the conflict.

---

## 📌 Project Overview

This project aims to automatically detect buildings and estimate damage levels using satellite images taken before and after the war in Gaza.

The proposed system focuses on **Al-Shati Camp**, combining satellite imagery analysis with deep learning techniques to identify damaged structures and estimate the percentage of affected buildings.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Detect buildings from pre-event satellite imagery.
- Identify damaged regions by comparing pre-war and post-war images.
- Estimate the percentage of damage occurring inside buildings.
- Generate visual overlays to support damage assessment and decision making.

---

## 🗺️ Study Area

The study focuses on:

**Al-Shati Camp (Beach Camp), Gaza Strip, Palestine**

Satellite images belonging to this area were extracted and analyzed throughout the project.

---

## 📂 Dataset

### Satellite Images

The satellite imagery was obtained from the Kaggle dataset:

**Gaza Before and After Dataset**

The dataset contains satellite images captured before and after the conflict.

---

### Building Footprints

Building information was extracted from:

**OpenStreetMap (OSM)**

using the OSMnx library.

These footprints were used to generate building masks for training the building segmentation model.

---

# ⚙️ System Architecture

The proposed system consists of three main stages.

---

# Stage 1: Building Detection

## Objective

Identify building locations from pre-event satellite images.

---

## Methodology

### Data Preparation

The system performs the following steps:

- Reads metadata associated with satellite images.
- Filters images belonging to Al-Shati Camp.
- Removes invalid images.
- Selects pre-war images.

---

### Building Mask Generation

Building polygons are downloaded from OpenStreetMap.

The geographic coordinates are converted into image coordinates to generate binary masks representing building footprints.

---

### Patch Extraction

Large satellite images are divided into smaller patches.

Patch size:

```
256 × 256 pixels
```

Stride:

```
64 pixels
```

Only patches containing sufficient building information are selected.

---

## Building Segmentation Model

### Model Architecture

U-Net

Encoder:

```
ResNet34
```

Pretrained weights:

```
ImageNet
```

Input Channels:

```
3 (RGB)
```

Output:

```
Binary building mask
```

---

## Data Augmentation

The following augmentation techniques were applied:

- Horizontal Flip
- Vertical Flip
- Random Rotation
- Shift and Scale
- Brightness Adjustment
- Contrast Adjustment
- Gaussian Noise

---

## Loss Function

The training loss combines:

### Dice Loss

Used to improve segmentation overlap.

### Binary Cross Entropy Loss

Used for pixel-wise classification.

Total loss:

```
Loss = Dice Loss + 0.5 × BCE Loss
```

---

## Optimizer

AdamW

Learning Rate:

```
1e-4
```

Weight Decay:

```
1e-4
```

Scheduler:

```
Cosine Annealing Learning Rate
```

---

## Training Configuration

Epochs:

```
50
```

Batch Size:

```
8
```

Train/Validation Split:

```
80% / 20%
```

---

## Evaluation Metric

The building segmentation model was evaluated using:

### F1-Score

which measures the overlap between predicted and true building masks.

---

# Stage 2: Damage Detection

## Objective

Detect damaged regions by comparing satellite images captured before and after the conflict.

---

## Pair Generation

For each geographical section:

The latest pre-war image was paired with the earliest post-war image.

---

## Damage Mask Generation

Pseudo-labels were generated automatically using image differences.

The following procedure was applied:

1. Convert images into floating-point representation.
2. Compute intensity differences.
3. Apply Gaussian filtering.
4. Threshold the differences.
5. Remove noise using morphological operations.
6. Exclude water regions.

---

## Important Note

The damage masks used in this project are automatically generated pseudo-labels.

They are not manually annotated ground-truth labels.

Therefore, the results should be interpreted as preliminary damage estimations.

---

## Damage Segmentation Model

### Model Architecture

U-Net

Encoder:

```
EfficientNet-B4
```

Input Channels:

```
6 Channels
(Post-event RGB + Pre-event RGB)
```

Output:

```
Damage mask
```

---

# Stage 3: Final Damage Assessment

## Objective

Combine building detection and damage detection results.

---

## Procedure

The final system performs the following:

- Detect buildings.
- Detect damaged regions.
- Calculate damage occurring inside buildings only.
- Estimate damage percentage.

---

## Visualization

Final outputs include:

### Yellow Contours

Represent detected buildings.

### Red Regions

Represent damaged areas.

### Overlay Maps

Combine both outputs into a single visualization.

---

# 📊 Results

The system produces:

- Building segmentation masks.
- Damage masks.
- Building contour maps.
- Damage overlays.
- Estimated building damage percentages.

Example interpretation:

```
Yellow → Buildings
Red → Damage
```

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Segmentation Models PyTorch
- OpenCV
- NumPy
- Matplotlib
- Albumentations
- OSMnx
- OpenStreetMap
- Scikit-learn
- Google Colab

---

# 📁 Project Structure

```
Project
│
├── Cell 0
│   └── Install libraries and download dataset
│
├── Cell 1
│   └── Metadata exploration and image pairing
│
├── Cell 2
│   └── Building dataset preparation
│
├── Cell 3
│   └── Building model training
│
├── Cell 4
│   └── Building prediction and visualization
│
├── Cell 5
│   └── Damage dataset preparation
│
├── Cell 6
│   └── Damage model training
│
├── Cell 7
│   └── Final evaluation
│
└── Cell 8
    └── Full-image overlay generation
```

---

# 🚧 Limitations

This work has several limitations:

- Damage labels are generated automatically rather than manually annotated.
- The study area is limited to Al-Shati Camp.
- Satellite image quality affects performance.
- Building footprints obtained from OpenStreetMap may contain inaccuracies.

---

# 🚀 Future Work

Potential improvements include:

- Using expert-annotated damage datasets.
- Expanding the study to all regions of Gaza.
- Integrating higher-resolution satellite imagery.
- Developing a real-time web application.
- Incorporating temporal analysis using multiple time periods.

---

# 👩‍💻 Author

Developed as an academic deep learning project for satellite-based damage assessment.

Study Area:
Al-Shati Camp, Gaza Strip, Palestine.

## data set 
https://www.kaggle.com/datasets/abdoomoh/gaza-before-and-after-2

#  Breast Ultrasound Image Classification
### Using Depthwise Separable CNNs with Class-Imbalance Mitigation Strategies

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-BUSI-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

**Author:** Harsh Tripathi
**Institution:** Indian Institute of Information Technology Raichur, Karnataka, India
**Contact:** [tripathiharsh2104@gmail.com](mailto:tripathiharsh2104@gmail.com)

</div>

---

##  Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Experiments](#-experiments)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Visualisations](#-results--visualisations)
- [Research Paper](#-research-paper)
- [References](#-references)

---

##  Overview

Breast cancer is the leading cause of cancer-related mortality among women globally. This project builds an automated **computer-aided diagnosis (CAD)** system for classifying breast ultrasound images into three clinically significant categories:

| Class | Description |
|-------|-------------|
| 🟢 **Benign** | Well-defined hypoechoic masses with smooth margins |
| 🔴 **Malignant** | Irregular borders, heterogeneous echogenicity, acoustic shadowing |
| 🔵 **Normal** | Homogeneous parenchymal patterns; no discrete masses |

The project systematically compares **5 experimental configurations** across two CNN architectures, specifically investigating how **Depthwise Separable Convolutions** and **class-imbalance strategies** affect diagnostic accuracy.

---

##  Key Results

| Rank | Method | Accuracy | Precision | Recall | **F1 Score** |
|------|--------|----------|-----------|--------|-------------|
| 🥇 | **Depthwise Separable CNN** | **79.49%** | **0.7897** | **0.7949** | **0.7732** |
| 🥈 | DS-CNN + Focal Loss | 72.65% | 0.7582 | 0.7265 | 0.6976 |
| 🥉 | DS-CNN + Oversampling | 69.23% | 0.7261 | 0.6923 | 0.6790 |
| 4th | Baseline CNN | 72.65% | 0.6021 | 0.7265 | 0.6585 |
| 5th | DS-CNN + Augmentation | 68.38% | 0.5849 | 0.6838 | 0.6144 |

> ✅ **DS-CNN achieves +6.84% accuracy and +0.1147 F1 over Baseline CNN — with 4× fewer parameters (~0.3M vs ~1.2M)**

---

##  Dataset

### BUSI — Breast Ultrasound Images Dataset

| Property | Details |
|----------|---------|
| **Source** | Baheya Hospital, Cairo, Egypt |
| **Total Images** | 780 grayscale ultrasound images |
| **Patients** | 600 female patients, aged 25–75 |
| **Equipment** | LOGIQ E9 ultrasound machine, ML6-15-D Matrix probe |
| **Format** | Grayscale PNG with paired segmentation masks |
| **Citation** | Al-Dhabyani et al., *Data in Brief*, 2020 |

### Class Distribution

```
Benign    ████████████████████████████  437 images  (56.0%)
Malignant ████████████                 210 images  (26.9%)
Normal    ████████                     133 images  (17.1%)
```

**Imbalance Ratios:**
- Benign : Normal    → **3.29 : 1** (most severe)
- Benign : Malignant → **2.08 : 1** (moderate)

### Data Split

```
Total: 780 images
├── Train  (70%)  →  ~546 images  [stratified]
├── Val    (15%)  →  ~117 images  [stratified]
└── Test   (15%)  →  ~117 images  [stratified]
```

---

##  Project Structure

```
breast-ultrasound-classification/
│
├──  BUSI_Classification_Pointwise_Depthwise.ipynb   # Main notebook
│
├──  breast_ultrasound_ieee_paper.tex                # IEEE paper (LaTeX)
├──  README.md                                       # This file
│
├──  outputs/
│   ├── class_distribution.png                        # EDA class distribution plot
│   ├── curves_baseline_cnn.png                       # Training curves — Baseline
│   ├── curves_depthwise_separable_cnn.png            # Training curves — DS-CNN
│   ├── curves_ds-cnn_+_focal_loss.png                # Training curves — Focal Loss
│   ├── curves_ds-cnn_+_oversampling.png              # Training curves — Oversampling
│   ├── curves_ds-cnn_+_augmentation.png              # Training curves — Augmentation
│   └── all_confusion_matrices.png                    # All 5 confusion matrices
│
└──  requirements.txt                               # Python dependencies
```

---

##  Methodology

### Preprocessing Pipeline

```
Raw Images (varied sizes)
        │
        ▼
  Resize → 224×224 px  (bilinear interpolation)
        │
        ▼
  Normalize → ÷255 → ImageNet μ/σ standardisation
        │
        ▼
  Stratified Split  →  70% Train | 15% Val | 15% Test
        │
        ▼
  Label Encode  →  Benign=0 | Malignant=1 | Normal=2
```

### Model Architectures

#### 1. Plain CNN (Baseline)
```
Input (3×224×224)
    │
    ├─ Conv Block 1:  Conv2d(3→32)   + BN + ReLU + MaxPool
    ├─ Conv Block 2:  Conv2d(32→64)  + BN + ReLU + MaxPool
    ├─ Conv Block 3:  Conv2d(64→128) + BN + ReLU + MaxPool
    ├─ Conv Block 4:  Conv2d(128→256)+ BN + ReLU + MaxPool
    │
    ├─ Flatten
    ├─ FC(1024→256) + ReLU + Dropout(0.4)
    └─ FC(256→3)  →  Softmax
                          Parameters: ~1.2M
```

#### 2. Depthwise Separable CNN (DS-CNN)
```
Input (3×224×224)
    │
    ├─ DS Block 1:  DepthwiseConv(3→3)   + PointwiseConv(3→32)   + BN + ReLU + MaxPool
    ├─ DS Block 2:  DepthwiseConv(32→32) + PointwiseConv(32→64)  + BN + ReLU + MaxPool
    ├─ DS Block 3:  DepthwiseConv(64→64) + PointwiseConv(64→128) + BN + ReLU + MaxPool
    ├─ DS Block 4:  DepthwiseConv(128→128)+PointwiseConv(128→256)+ BN + ReLU + MaxPool
    │
    ├─ Flatten
    ├─ FC(1024→256) + ReLU + Dropout(0.4)
    └─ FC(256→3)  →  Softmax
                          Parameters: ~0.3M  (4× reduction)
```

**Efficiency gain per block:**

$$\frac{C_{DS}}{C_{std}} = \frac{1}{N} + \frac{1}{k^2} \approx \frac{1}{8.9} \quad \text{for } k=3,\; N=64$$

---

##  Experiments

### Training Setup

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Epochs | 10 |
| LR Scheduler | ReduceLROnPlateau (patience=3) |
| Model Selection | Max validation accuracy |
| Device | CPU |

### Experiment Configurations

| # | Experiment | Architecture | Strategy |
|---|-----------|-------------|----------|
| 1 | **Baseline CNN** | PlainCNN | Standard CrossEntropy, uniform sampling |
| 2 | **DS-CNN** | DS-CNN | Standard CrossEntropy |
| 3 | **DS-CNN + Oversampling** | DS-CNN | `WeightedRandomSampler` — equal batch exposure |
| 4 | **DS-CNN + Augmentation** | DS-CNN | Flip, Rotation, Affine, ColorJitter |
| 5 | **DS-CNN + Focal Loss** | DS-CNN | Focal Loss (γ=2) — focus on hard examples |

### Imbalance Strategies Explained

<details>
<summary><b>🔵 WeightedRandomSampler (Oversampling)</b></summary>

Each sample is assigned an inverse-frequency weight:

```
w_i = N / count(class_i)
```

This ensures that each mini-batch sees approximately equal numbers of all classes, without duplicating data on disk.

</details>

<details>
<summary><b>🟡 Data Augmentation</b></summary>

Applied only to the training set:
- `RandomHorizontalFlip(p=0.5)`
- `RandomVerticalFlip(p=0.5)`
- `RandomRotation(±15°)`
- `RandomAffine(scale=[0.9,1.1], translate≤10%)`
- `ColorJitter(brightness=0.2, contrast=0.2)`

</details>

<details>
<summary><b>🔴 Focal Loss (γ=2)</b></summary>

Modifies cross-entropy to down-weight easy examples:

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```

Well-classified majority-class samples contribute near-zero gradient, allowing the model to concentrate on hard malignant/normal examples.

</details>

---

##  Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/breast-ultrasound-classification.git
cd breast-ultrasound-classification

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
kagglehub>=0.2.0
```

### Dataset Download

```python
import kagglehub

# Download BUSI dataset from Kaggle
path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
print("Dataset path:", path)
```

> **Note:** A Kaggle account and API token (`~/.kaggle/kaggle.json`) are required.

---

##  Usage

### Run the Full Notebook

```bash
jupyter notebook BUSI_Classification_Pointwise_Depthwise.ipynb
```

### Run All Experiments Sequentially

The notebook is structured in 15 sections. Run cells top-to-bottom to:
1. Install dependencies
2. Load and preprocess the BUSI dataset
3. Perform EDA and visualise class distribution
4. Define PlainCNN and DS-CNN architectures
5. Train all 5 experimental configurations
6. Evaluate and compare results
7. Generate confusion matrices and training curves
8. Print the final summary table

### Expected Output (Final Summary)

```
========================================================================
      BUSI CLASSIFICATION PROJECT — FINAL RESULTS
========================================================================
  Image Size  : 224×224  |  Batch: 32  |  Epochs: 10
  Device      : cpu

                 Method  Accuracy  Precision   Recall  F1 Score
Depthwise Separable CNN  0.794872   0.789726 0.794872  0.773189
    DS-CNN + Focal Loss  0.726496   0.758160 0.726496  0.697567
  DS-CNN + Oversampling  0.692308   0.726063 0.692308  0.678969
           Baseline CNN  0.726496   0.602102 0.726496  0.658467
  DS-CNN + Augmentation  0.683761   0.584934 0.683761  0.614435

  ★ Best F1 Score : Depthwise Separable CNN  (0.7732)
  ★ Best Accuracy : Depthwise Separable CNN  (0.7949)
========================================================================
```

---

##  Results & Visualisations

### Performance Comparison

```
Accuracy
DS-CNN              ████████████████████████████████████████ 79.49%
DS-CNN+Focal        ████████████████████████████████████     72.65%
Baseline CNN        ████████████████████████████████████     72.65%
DS-CNN+Oversamp     ██████████████████████████████████       69.23%
DS-CNN+Augment      █████████████████████████████████        68.38%

F1 Score
DS-CNN              ████████████████████████████████████████ 0.7732
DS-CNN+Focal        ████████████████████████████████████     0.6976
DS-CNN+Oversamp     ███████████████████████████████████      0.6790
Baseline CNN        █████████████████████████████████        0.6585
DS-CNN+Augment      ███████████████████████████████          0.6144
```

### Key Observations

1. **DS-CNN > PlainCNN** despite 4× fewer parameters — factorised convolutions are highly efficient for small medical datasets.

2. **Focal Loss** improves precision (+15.6% over Baseline) but sacrifices some recall — useful when false positives are costly.

3. **Oversampling** provides a balanced improvement, particularly benefiting Malignant and Normal class recall.

4. **Augmentation** without pretrained weights **hurts** performance — geometric transforms introduce too much variance for a ~546-image training set.

5. **Clinical Priority:** Missing Malignant cases is life-threatening. Recall on the Malignant class should be the primary optimisation target in production.

---

##  Research Paper

This project is accompanied by a full **IEEE Conference-format research paper** written in LaTeX.

| File | Description |
|------|-------------|
| `breast_ultrasound_ieee_paper.tex` | Complete LaTeX source |

### Compile the Paper

```bash
# Using pdflatex (run twice for references)
pdflatex breast_ultrasound_ieee_paper.tex
pdflatex breast_ultrasound_ieee_paper.tex

# Or open directly in Overleaf (recommended)
# Upload the .tex file at https://www.overleaf.com
```

> **Note:** Place all plot `.png` files (from notebook outputs) in the same directory as the `.tex` file before compiling.

---

##  Future Work

- [ ] **Pretrained Backbones** — ResNet-50, EfficientNet-B0, ViT with ImageNet weights
- [ ] **Combined Strategies** — DS-CNN + Oversampling + Focal Loss jointly
- [ ] **k-Fold Cross-Validation** — More robust performance estimation
- [ ] **Attention Mechanisms** — SE blocks (channel) or CBAM (spatial + channel)
- [ ] **Mask-Guided Training** — Use BUSI segmentation masks as weak supervision
- [ ] **Grad-CAM Explainability** — Visual explanations for clinical trust

---

##  References

1. Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. *Dataset of breast ultrasound images.* Data in Brief, 28, 104863, **2020**.

2. Howard AG et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861, **2017**.

3. Lin TY, Goyal P, Girshick R, He K, Dollár P. *Focal Loss for Dense Object Detection.* IEEE ICCV, pp. 2980–2988, **2017**.

4. Paszke A et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS, vol. 32, **2019**.

5. Litjens G et al. *A Survey on Deep Learning in Medical Image Analysis.* Medical Image Analysis, 42, 60–88, **2017**.

6. Loshchilov I, Hutter F. *Decoupled Weight Decay Regularization.* ICLR, **2019**.

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with  by **Harsh Tripathi**
Department of AI & Data Science · IIIT Raichur · Karnataka, India

 *If you found this useful, please consider starring the repository!*

</div>

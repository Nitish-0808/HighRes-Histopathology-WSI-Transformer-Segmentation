# 🧬 HighRes Histopathology Semantic Segmentation using Transformer

> **A high-resolution transformer-based framework for efficient and precise segmentation of histopathology whole-slide images (WSIs)**🧠💻

## 🧪 Overview
**HighRes-Histopathology-WSI-Transformer-Segmentation** provides an end-to-end deep learning pipeline for histopathology WSI segmentation.  
It leverages **transformer-based contextual encoding** and **boundary-aware learning** to achieve high accuracy and robust generalization across tissue slides.

## 🚀 Features

✅ End-to-end pipeline for WSI segmentation  
✅ Transformer backbone with boundary & dilation-aware training   
✅ Configurable training hyperparameters and flexible dataset options  
✅ Visual sanity checks for colored mask generation  
✅ Evaluation and overlay visualization for predictions  

## 🧰 Prerequisites
Before proceeding, ensure the following resources are available in your environment:
- **Pretrained Backbone Weights:**
Download the pretrained backbone weights for model initialization from [this link](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth) and place the file at:
  ```bash
  <repo_root>/
  ```
**Dataset Location:**
Store the raw Whole Slide Images (WSIs) at:
  ```bash
  <repo_root>/
  └── datasets/
      └── raw/
          ├── Training/                 # Training WSIs or tiles
          │   ├── sample_001.png
          │   ├── sample_001_mask.png
          │   ├── sample_002.png
          │   ├── sample_002_mask.png
          │   └── ...
          ├── Validation/               # Validation set
          │   ├── slide_101.png
          │   ├── slide_101_mask.png
          │   └── ...
          └── Extra/                    # Optional: test or unseen slides
              ├── slide_201.png
              ├── slide_201_mask.png
              └── ...

  ```
## 📂 Directory Structure

```bash
FCBFormer/
├── Data_prep/                     # Dataset preparation utilities
│   ├── prepare_dataset.py
│   ├── make_colored_previews.py
│   └── check_manifests.py
├── train.py                       # Main training entry point
├── eval.py                        # Evaluation script
├── pvt_v2_b3.pth                  # Initial weights
├── datasets/
│   ├── raw/                       # Original WSIs or tiles
│   └── processed/                 # Preprocessed and tiled data
└── outputs/
    ├── <timestamp>/               # Model checkpoints and logs
    ├── EvalWSI/                   # Evaluation results on validation/test data
    └── extra_preds/               # Predictions for extra/unseen slides
...
```  
## 🛠️ Training and validation Setup (Conda)
### 1) Clone the repository
```bash
git clone https://github.com/Nitish-0808/HighRes-Histopathology-WSI-Transformer-Segmentation.git
cd HighRes-Histopathology-WSI-Transformer-Segmentation
```

### 2) Create environment
```bash
# From repo root
conda env create -f environment.yml
conda activate fcbformer
```

### 3) One-shot sanity check
```bash
python - <<'PY'
import torch, torchvision, timm, cv2, sklearn, skimage, tqdm, numpy as np
print("CUDA available:", torch.cuda.is_available())
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("opencv:", cv2.__version__)
print("sklearn:", sklearn.__version__, "skimage:", skimage.__version__, "numpy:", np.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```
### 4) Prepare data

Keep raw delivery intact and generate a processed set by running 

```bash
python Data_prep/prepare_dataset.py \
  --root ./datasets \
  --out-name tiles_512_o64 \
  --patch-size 256 \
  --overlap 64
```

### 5) Visualize colored mask previews & Sanity-check manifests

```bash
python Data_prep/make_colored_previews.py \
  --mask-dir ./datasets/processed/tiles_512_o64/train/masks \
  --limit 50

python Data_prep/check_manifests.py \
  --manifests-dir ./datasets/processed/manifests \
  --samples 5
```

### 6) Train

```bash
python train.py \
  --dataset AIRA \
  --data-root ./datasets \
  --epochs 60 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --img-size 256 \
  --boundary-weight 0.7 \
  --tumor-dil-weight 0.7 \
  --benign-dil-weight 0.7 \
  --dil-iters 5 \
  --dil-kernel 3 \
  --prob-power 1.0
```

Artifacts are saved in
```bash
./outputs/<timestamp>/
```

### 7) Evaluate (val/test)

```bash
python eval.py \
  --data-root ./datasets \
  --checkpoint ./outputs/<timestamp>/best_FCBFormer.pt \
  --img-size 256 \
  --batch-size 4 \
  --save-png \
  --out-dir ./outputs/EvalWSI
  ```

### 8) Evaluate on extra slides
```bash
python eval.py \
  --data-root ./datasets \
  --checkpoint ./outputs/<timestamp>/best_FCBFormer.pt \
  --img-size 256 \
  --batch-size 4 \
  --tiles-manifest ./datasets/processed/manifests/extra.csv \
  --out-dir ./outputs/extra_preds \
  --overlay-alpha 0.4
```
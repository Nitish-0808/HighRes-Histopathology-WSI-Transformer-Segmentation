from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TilesDatasetJoint(Dataset):
    """
    Manifest-based dataset with joint (image, mask) Albumentations.
    Assumes CSV columns: image_path, mask_path
    """
    def __init__(self, csv_path, mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5),
                 img_size=352, is_train=True):
        self.df = pd.read_csv(csv_path)

        base = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=0, value=(255,255,255), mask_value=0),
        ]

        aug = []
        if is_train:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
                                   border_mode=0, value=(255,255,255), mask_value=0, p=0.5),
                # color ops apply to image only (Albumentations handles that)
                A.ColorJitter(0.1, 0.1, 0.1, 0.02, p=0.5),
                A.GaussianBlur(blur_limit=(3,7), sigma_limit=(0.1,2.0), p=0.2),
            ]

        tail = [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),  # image -> float32 tensor (C,H,W); mask -> torch.uint8 tensor (H,W)
        ]

        self.tf = A.Compose(base + aug + tail)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(row["mask_path"]), dtype=np.uint8)

        out = self.tf(image=img, mask=mask)
        x = out["image"]                                 # float32, (3,H,W)
        y = out["mask"].to(torch.long).clamp_(0, 3)      # int64,   (H,W) in {0..3}
        return x, y

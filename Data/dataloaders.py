import json
import numpy as np
import random
from PIL import Image
import multiprocessing
from pathlib import Path

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils import data
import torch

from Data.dataset import SegDataset


def split_ids(len_ids):
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_dataloaders(input_paths, target_paths, batch_size):

    transform_input4train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01
            ),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_input4test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((352, 352)), transforms.Grayscale()]
    )

    train_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4train,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=True,
    )

    test_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    val_dataset = SegDataset(
        input_paths=input_paths,
        target_paths=target_paths,
        transform_input=transform_input4test,
        transform_target=transform_target,
    )

    train_indices, test_indices, val_indices = split_ids(len(input_paths))

    train_dataset = data.Subset(train_dataset, train_indices)
    val_dataset = data.Subset(val_dataset, val_indices)
    test_dataset = data.Subset(test_dataset, test_indices)

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=multiprocessing.Pool()._processes,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=multiprocessing.Pool()._processes,
    )

    return train_dataloader, test_dataloader, val_dataloader
    

# ---------------- helpers for our AIRA manifests ---------------- #
def _read_manifest(csv_path: str):
    import csv
    inputs, targets = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        has_mask = "mask_path" in reader.fieldnames
        for r in reader:
            inputs.append(r["image_path"])
            if has_mask:
                targets.append(r["mask_path"])
    return inputs, targets
    

def _make_input_transforms(img_size: int, mean, std, is_train: bool):
    base = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
    ]
    aug = []
    if is_train:
        aug = [
            transforms.GaussianBlur((5, 5), sigma=(0.001, 1.5)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    tail = [transforms.Normalize(mean, std)]
    return transforms.Compose(base + aug + tail)
    

def _make_target_transform(img_size: int):
    # returns a callable: np.ndarray -> LongTensor(H,W) with NEAREST resize
    def _tf(mask_np):
        m = Image.fromarray(mask_np.astype(np.uint8), mode="L")
        m = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)(m)
        m = np.array(m, dtype=np.uint8)
        m = np.clip(m, 0, 3)
        return torch.from_numpy(m).long()
    return _tf
    

def get_dataloaders_from_manifests(
    train_csv: str,
    val_csv: str,
    stats_json: str,
    batch_size: int = 4,
    img_size: int = 352,
    num_workers: int = multiprocessing.Pool()._processes,
):
    # load manifest rows
    train_inputs, train_targets = _read_manifest(train_csv)
    val_inputs,   val_targets   = _read_manifest(val_csv)

    # load dataset mean/std
    stats = json.load(open(stats_json))
    mean, std = stats["rgb_mean"], stats["rgb_std"]

    transform_input_train = _make_input_transforms(img_size, mean, std, is_train=True)
    transform_input_eval  = _make_input_transforms(img_size, mean, std, is_train=False)
    transform_target      = _make_target_transform(img_size)

    train_dataset = SegDataset(
        input_paths=train_inputs,
        target_paths=train_targets,
        transform_input=transform_input_train,
        transform_target=transform_target,
        hflip=False, vflip=False, affine=False,  # flips already in input transform; mask sync handled by Resize only
    )

    val_dataset = SegDataset(
        input_paths=val_inputs,
        target_paths=val_targets,
        transform_input=transform_input_eval,
        transform_target=transform_target,
        hflip=False, vflip=False, affine=False,
    )

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,   # you can use 1 as well
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader




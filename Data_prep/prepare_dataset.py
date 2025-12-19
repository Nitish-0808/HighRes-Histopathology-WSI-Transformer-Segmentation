#!/usr/bin/env python3
"""
prepare_dataset.py

End-to-end data preparation for WSI-style segmentation:
1) Keeps raw data intact under: dataset/raw/{Training,Validation,Extra}
2) Converts colored masks -> single-channel class IDs {0:bg,1:stroma,2:benign,3:tumor}
3) Crops to ROI (mask-driven; fallback tissue detector)
4) Tiles to 512x512 with 64px overlap, pads when needed, skips low-foreground tiles
5) Writes tiles to: dataset/processed/tiles_512_o64/{train,val,extra}
6) Creates manifests: dataset/processed/manifests/{train.csv,val.csv,extra.csv}
7) Computes train mean/std and class counts; writes dataset/processed/stats.json
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None
# -------------------------
# Config: color palette ? IDs
# -------------------------
# 0: Background (black)   = (0, 0, 0)
# 1: Stroma (blue)        = (0, 0, 255)
# 2: Benign (green)       = (0, 255, 0)
# 3: Tumor (yellow)       = (255, 255, 0)
PALETTE = np.array([
    [0,   0,   0],   # 0 background
    [0,   0, 255],   # 1 stroma
    [0, 255,   0],   # 2 benign
    [255, 255, 0],   # 3 tumor
], dtype=np.int16)


# ---------- I/O helpers ----------
def load_rgb(path: Path) -> np.ndarray:
    """Load image as RGB uint8 (H,W,3)."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def load_mask_to_ids(path: Path, tol: int = 10) -> np.ndarray:
    """
    Load a mask image; if RGB, map colors to IDs {0..3} with tolerance.
    If single-channel (L/I), clamp to 0..3.
    """
    m = Image.open(path)
    if m.mode in ("L", "I;16", "I"):
        arr = np.array(m)
        arr = np.where(arr > 3, 0, arr)
        return arr.astype(np.uint8)
    rgb = np.array(m.convert("RGB"), dtype=np.uint8)
    return rgb_to_id(rgb, tol)

def save_png(path: Path, arr: np.ndarray):
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    Image.fromarray(arr).save(path)


# ---------- mask color mapping ----------
def rgb_to_id(mask_rgb: np.ndarray, tol: int = 10) -> np.ndarray:
    """
    Map RGB mask to IDs {0..3} using tolerance against PALETTE.
    Unmatched pixels fall back to background (0).
    """
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    flat = mask_rgb.astype(np.int16).reshape(-1, 3)
    outf = out.reshape(-1)
    # assign by nearest palette color within tolerance (L1 sum <= 3*tol)
    for cls_id, col in enumerate(PALETTE):
        hits = (np.abs(flat - col).sum(axis=1) <= 3 * tol)
        outf[hits] = cls_id
    return out


# ---------- ROI & tiling ----------
def crop_roi_from_mask(label: np.ndarray, pad: int = 64) -> Tuple[int, int, int, int]:
    """Compute bbox around non-background labels, expanded by pad."""
    ys, xs = np.where(label != 0)
    if ys.size == 0:
        return 0, label.shape[0], 0, label.shape[1]
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad); y1 = min(label.shape[0], y1 + pad + 1)
    x0 = max(0, x0 - pad); x1 = min(label.shape[1], x1 + pad + 1)
    return y0, y1, x0, x1

def fallback_tissue_roi(img: np.ndarray, pad: int = 64) -> Tuple[int, int, int, int]:
    """
    Fallback ROI via tissue detection: consider non-white pixels as tissue.
    White-ish ~ (R,G,B > 240).
    """
    not_white = ~((img[..., 0] > 240) & (img[..., 1] > 240) & (img[..., 2] > 240))
    ys, xs = np.where(not_white)
    if ys.size == 0:
        return 0, img.shape[0], 0, img.shape[1]
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad); y1 = min(img.shape[0], y1 + pad + 1)
    x0 = max(0, x0 - pad); x1 = min(img.shape[1], x1 + pad + 1)
    return y0, y1, x0, x1

def tile_coords(h: int, w: int, patch: int, overlap: int):
    """
    Yield top-left (y,x) for overlapping tiles.
    Ensures last tiles align to image borders (full coverage).
    """
    step = patch - overlap
    ys = list(range(0, max(1, h - patch + 1), step))
    xs = list(range(0, max(1, w - patch + 1), step))
    if ys[-1] != h - patch:
        ys.append(h - patch)
    if xs[-1] != w - patch:
        xs.append(w - patch)
    for y in ys:
        for x in xs:
            yield y, x

def non_bg_ratio(lbl: np.ndarray, bg_id: int = 0) -> float:
    total = lbl.size
    if total == 0:
        return 0.0
    return 1.0 - (np.count_nonzero(lbl == bg_id) / float(total))


# ---------- discovery ----------
def collect_pairs(folder: Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    For Training/Validation: returns list of (image_path, mask_path).
    For Extra: returns list of (image_path, None).
    Images = files not ending with '_mask' and standard image suffixes.
    Mask = <stem>_mask.<ext> (try several ext).
    """
    imgs = [p for p in folder.iterdir()
            if p.is_file()
            and not p.stem.endswith("_mask")
            and p.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg")]
    pairs = []
    for ip in imgs:
        maskp = None
        cand = ip.with_name(ip.stem + "_mask" + ip.suffix)
        if cand.exists():
            maskp = cand
        else:
            for ext in (".png", ".tif", ".tiff", ".jpg"):
                alt = ip.with_name(ip.stem + "_mask" + ext)
                if alt.exists():
                    maskp = alt; break
        pairs.append((ip, maskp))
    return pairs


# ---------- core processing ----------
def process_split(
    split: str,
    in_dir: Path,
    out_img_dir: Path,
    out_msk_dir: Optional[Path],
    manifest_csv: Path,
    patch: int,
    overlap: int,
    min_fg: float,
    pad: int,
    tol: int,
):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    if out_msk_dir is not None:
        out_msk_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(in_dir)
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.writer(f)
        if split in ("train", "val"):
            writer.writerow(["image_path", "mask_path", "src_id", "h", "w", "y", "x"])
        else:
            writer.writerow(["image_path", "src_id", "h", "w", "y", "x"])

        for img_path, mask_path in tqdm(pairs, desc=f"{split}: files"):
            src_id = img_path.stem
            img = load_rgb(img_path)

            label = None
            if mask_path is not None:
                label = load_mask_to_ids(mask_path, tol=tol)
                # Align sizes if needed (prefer not to, but safe fallback)
                if label.shape[:2] != img.shape[:2]:
                    label = np.array(
                        Image.fromarray(label).resize(
                            (img.shape[1], img.shape[0]), resample=Image.NEAREST
                        ),
                        dtype=np.uint8,
                    )

            # ROI from mask or fallback tissue ROI
            if label is not None:
                y0, y1, x0, x1 = crop_roi_from_mask(label, pad=pad)
                if (y0 == 0 and y1 == label.shape[0] and x0 == 0 and x1 == label.shape[1]
                        and np.count_nonzero(label) == 0):
                    y0, y1, x0, x1 = fallback_tissue_roi(img, pad=pad)
            else:
                y0, y1, x0, x1 = fallback_tissue_roi(img, pad=pad)

            img_c = img[y0:y1, x0:x1]
            if label is not None:
                lbl_c = label[y0:y1, x0:x1]

            H, W = img_c.shape[:2]
            # minimal padding if ROI smaller than patch
            if H < patch or W < patch:
                py, px = max(0, patch - H), max(0, patch - W)
                img_c = np.pad(img_c, ((0, py), (0, px), (0, 0)), mode="constant", constant_values=255)
                if label is not None:
                    lbl_c = np.pad(lbl_c, ((0, py), (0, px)), mode="constant", constant_values=0)
                H, W = img_c.shape[:2]

            for y, x in tile_coords(H, W, patch, overlap):
                img_t = img_c[y:y+patch, x:x+patch]

                if split in ("train", "val"):
                    lbl_t = lbl_c[y:y+patch, x:x+patch]
                    if non_bg_ratio(lbl_t) < min_fg:
                        continue

                img_name = f"img_{src_id}_{y0+y}_{x0+x}.png"
                save_png(out_img_dir / img_name, img_t)

                if split in ("train", "val"):
                    msk_name = f"mask_{src_id}_{y0+y}_{x0+x}.png"
                    save_png(out_msk_dir / msk_name, lbl_t)
                    writer.writerow([str(out_img_dir / img_name),
                                     str(out_msk_dir / msk_name),
                                     src_id, patch, patch, y0+y, x0+x])
                else:
                    writer.writerow([str(out_img_dir / img_name),
                                     src_id, patch, patch, y0+y, x0+x])


# ---------- stats computation ----------
def compute_train_mean_std(train_images_dir: Path) -> Tuple[List[float], List[float]]:
    """
    Compute per-channel mean/std over all train tiles.
    Uses streaming sums to avoid big memory use.
    Returns mean/std in [0..1] range.
    """
    # list all PNGs
    paths = list(train_images_dir.glob("*.png"))
    if not paths:
        return [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]

    n_pixels = 0
    sum_rgb = np.zeros(3, dtype=np.float64)
    sumsq_rgb = np.zeros(3, dtype=np.float64)

    for p in tqdm(paths, desc="compute mean/std (train)"):
        im = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = im.shape
        n = h * w
        n_pixels += n
        sum_rgb += im.reshape(-1, 3).sum(axis=0)
        sumsq_rgb += (im.reshape(-1, 3) ** 2).sum(axis=0)

    mean = (sum_rgb / n_pixels).tolist()
    var = (sumsq_rgb / n_pixels) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12)).tolist()
    return mean, std

def compute_class_counts(train_masks_dir: Path, n_classes: int = 4) -> List[int]:
    """Count pixel frequency per class over train mask tiles."""
    paths = list(train_masks_dir.glob("*.png"))
    counts = np.zeros(n_classes, dtype=np.int64)
    for p in tqdm(paths, desc="compute class counts (train)"):
        m = np.array(Image.open(p), dtype=np.uint8)
        for c in range(n_classes):
            counts[c] += np.count_nonzero(m == c)
    return counts.tolist()


# ---------- optional stain normalization hook ----------
def apply_stain_norm_if_hook(img: np.ndarray) -> np.ndarray:
    """
    Placeholder for stain normalization (e.g., Macenko).
    Currently returns img unchanged. If you implement later, drop it here.
    """
    return img


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Prepare tiled dataset from raw WSI-like images and masks.")
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing raw/ and processed/")
    ap.add_argument("--out-name", type=str, default="tiles_512_o64", help="Processed subfolder name under processed/")
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--min-foreground", type=float, default=0.05,
                    help="Skip train/val tiles with < this fraction of non-background (ID!=0)")
    ap.add_argument("--pad", type=int, default=64, help="Margin (px) to expand ROI bbox")
    ap.add_argument("--tol", type=int, default=10, help="RGB tolerance when mapping colors?IDs")
    ap.add_argument("--no-stats", action="store_true", help="Skip computing mean/std and class counts")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    raw = root / "raw"
    processed = root / "processed" / args.out_name
    manifests = root / "processed" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    # inputs
    train_in = raw / "Training"
    val_in   = raw / "Validation"
    extra_in = raw / "Extra"

    # outputs
    train_img_out = processed / "train" / "images"
    train_msk_out = processed / "train" / "masks"
    val_img_out   = processed / "val"   / "images"
    val_msk_out   = processed / "val"   / "masks"
    extra_img_out = processed / "extra" / "images"

    # manifests
    train_csv = manifests / "train.csv"
    val_csv   = manifests / "val.csv"
    extra_csv = manifests / "extra.csv"

    # process splits
    if train_in.exists():
        process_split("train", train_in, train_img_out, train_msk_out, train_csv,
                      patch=args.patch_size, overlap=args.overlap,
                      min_fg=args.min_foreground, pad=args.pad, tol=args.tol)
    if val_in.exists():
        process_split("val", val_in, val_img_out, val_msk_out, val_csv,
                      patch=args.patch_size, overlap=args.overlap,
                      min_fg=args.min_foreground, pad=args.pad, tol=args.tol)
    if extra_in.exists():
        process_split("extra", extra_in, extra_img_out, None, extra_csv,
                      patch=args.patch_size, overlap=args.overlap,
                      min_fg=0.0, pad=args.pad, tol=args.tol)

    print(f"\nTiling done at: {processed}")
    print(f"Manifests at:   {manifests}")

    # compute stats (optional)
    if not args.no_stats and train_img_out.exists() and train_msk_out.exists():
        mean, std = compute_train_mean_std(train_img_out)
        counts = compute_class_counts(train_msk_out, n_classes=4)
        stats_path = root / "processed" / "stats.json"
        stats = {
            "processed_subdir": args.out_name,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "min_foreground": args.min_foreground,
            "rgb_mean": mean,       # in [0,1]
            "rgb_std": std,         # in [0,1]
            "class_counts": {
                "background": counts[0],
                "stroma": counts[1],
                "benign": counts[2],
                "tumor": counts[3],
                "total_pixels": int(sum(counts)),
            },
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats written to: {stats_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

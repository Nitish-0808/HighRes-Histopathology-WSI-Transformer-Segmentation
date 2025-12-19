#!/usr/bin/env python3
"""
make_colored_previews.py

Quick sanity check for segmentation masks:
- Loads processed single-channel ID masks (0=bg,1=stroma,2=benign,3=tumor)
- Converts them into RGB colors for visualization
- Saves colored previews into a sibling folder (e.g. masks_color/)
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import os, glob

# ID -> RGB mapping
ID2COLOR = np.array([
    [0,   0,   0],   # 0 background
    [0,   0, 255],   # 1 stroma (blue)
    [0, 255,   0],   # 2 benign (green)
    [255, 255, 0],   # 3 tumor (yellow)
], dtype=np.uint8)


def convert_masks(mask_dir: Path, out_dir: Path, limit: int = 50):
    """Convert grayscale ID masks to RGB previews."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = sorted(glob.glob(str(mask_dir / "*.png")))
    if limit:
        mask_paths = mask_paths[:limit]

    for p in mask_paths:
        m = np.array(Image.open(p))
        rgb = ID2COLOR[m]
        Image.fromarray(rgb).save(out_dir / Path(p).name)

    print(f"Saved {len(mask_paths)} colored previews to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Make colored previews of mask IDs.")
    parser.add_argument("--mask-dir", required=True, type=Path,
                        help="Directory with single-channel ID masks (e.g., processed/train/masks/)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for colored masks (default: <mask-dir>_color)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Limit number of masks to convert (default: 50)")
    args = parser.parse_args()

    out_dir = args.out_dir or Path(str(args.mask_dir) + "_color")
    convert_masks(args.mask_dir, out_dir, args.limit)


if __name__ == "__main__":
    main()

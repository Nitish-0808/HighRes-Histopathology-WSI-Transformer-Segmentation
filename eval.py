#############################################################################################################
import os, csv, json, argparse, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
##############################################################################################################
# --- constants ---
NUM_CLASSES = 4
ID2COLOR = np.array([
    [0, 0, 0],       # bg
    [0, 0, 255],     # stroma
    [0, 255, 0],     # benign
    [255, 255, 0],   # tumor
], dtype=np.uint8)
############################################################################################################
# --- io helpers ---
def read_manifest(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # val.csv has image_path,mask_path,src_id,h,w,y,x
            row["h"] = int(row["h"]); row["w"] = int(row["w"])
            row["y"] = int(row["y"]); row["x"] = int(row["x"])
            rows.append(row)
    return rows
def load_rgb(path):
    return Image.open(path).convert("RGB")

def load_mask_ids(path):
    arr = np.array(Image.open(path), dtype=np.uint8)
    return arr

def make_val_transform(img_size, mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(mean, std),
    ])

def upsample_to(arr_chw, out_h, out_w, mode="bilinear"):
    # arr_chw: Tensor (C,H,W)
    t = arr_chw.unsqueeze(0)  # (1,C,H,W)
    t = F.interpolate(t, size=(out_h, out_w),
                      mode=mode, align_corners=False if mode=="bilinear" else None)
    return t.squeeze(0)
##############################################################################################################  
# --- metrics ---
def iou_per_class(pred_ids, gt_ids, num_classes=NUM_CLASSES):
    ious = []
    for c in range(num_classes):
        pred_c = (pred_ids == c)
        gt_c   = (gt_ids == c)
        inter = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or (pred_c, gt_c).sum()
        ious.append( (inter / union) if union > 0 else 0.0 )
    return ious
#############################################################################################################################
def dice_per_class(pred_ids, gt_ids, num_classes=NUM_CLASSES):
    dices = []
    for c in range(num_classes):
        pred_c = (pred_ids == c)
        gt_c = (gt_ids == c)
        inter = np.logical_and(pred_c, gt_c).sum()
        denom = pred_c.sum() + gt_c.sum()
        dices.append((2 * inter / denom) if denom > 0 else 0.0)
    return dices
#############################################################################################################################
def precision_recall_per_class(pred_ids, gt_ids, num_classes=NUM_CLASSES):
    precisions, recalls = [], []
    for c in range(num_classes):
        pred_c = (pred_ids == c)
        gt_c = (gt_ids == c)
        tp = np.logical_and(pred_c, gt_c).sum()
        fp = np.logical_and(pred_c, ~gt_c).sum()
        fn = np.logical_and(~pred_c, gt_c).sum()
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    return precisions, recalls
##############################################################################################################################
# --- stitching core ---
@torch.no_grad()
def stitch_and_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load stats for normalization
    stats_path = os.path.join(args.data_root, "processed", "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean, std = stats["rgb_mean"], stats["rgb_std"]
############################################################################################################
    # model
    from Models import models as modelz
    model = modelz.FCBFormer(size=args.img_size, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
##############################################################################################################################
    # val manifest rows -> group by WSI (src_id)
    val_csv = os.path.join(args.data_root, "processed", "manifests", "val.csv")
    rows = read_manifest(val_csv)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["src_id"]].append(r)

    tf = make_val_transform(args.img_size, mean, std)
    os.makedirs(args.out_dir, exist_ok=True)
    per_wsi_ious, per_wsi_dices = [], []
    per_wsi_precisions, per_wsi_recalls = [], []

    class_names = ["bg","stroma","benign","tumor"]

    for idx, (src_id, tiles) in enumerate(sorted(by_src.items())):
        # determine stitched canvas size from tiles
        H = max(r["y"] + r["h"] for r in tiles)
        W = max(r["x"] + r["w"] for r in tiles)
        # accumulate per-class logits and counts for averaging overlaps
        logit_acc = torch.zeros(NUM_CLASSES, H, W, dtype=torch.float32, device=device)
        count_acc = torch.zeros(1, H, W, dtype=torch.float32, device=device)
        # stitch GT mask too
        gt_stitched = np.zeros((H, W), dtype=np.uint8)

        # process tiles (optional mini-batching for speed)
        batch_imgs, batch_meta = [], []
        BATCH = args.batch_size
        def flush_batch():
            if not batch_imgs: return
            x = torch.stack(batch_imgs, dim=0).to(device)  # (B,3,sz,sz)
            logits = model(x)                              # (B,C,sz,sz)
            if isinstance(logits, (tuple, list)):  # safety if model returns tuple
                logits = logits[0]
            # upsample logits back to original tile HxW (usually 512x512)
            for b in range(logits.shape[0]):
                r = batch_meta[b]
                logit_tile = logits[b]
                logit_tile = upsample_to(logit_tile, r["h"], r["w"], mode="bilinear")
                y0, x0 = r["y"], r["x"]
                logit_acc[:, y0:y0+r["h"], x0:x0+r["w"]] += logit_tile
                count_acc[:, y0:y0+r["h"], x0:x0+r["w"]] += 1.0
            batch_imgs.clear(); batch_meta.clear()

        for r in tiles:
            img = load_rgb(r["image_path"])
            gt  = load_mask_ids(r["mask_path"])
            # place GT (simple overwrite is fine; tiles are consistent)
            gt_stitched[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]] = gt

            x = tf(img)  # (3,sz,sz)
            batch_imgs.append(x)
            batch_meta.append(r)

            if len(batch_imgs) == BATCH:
                flush_batch()
        flush_batch()

        # average overlaps
        count_acc = torch.clamp(count_acc, min=1.0)
        logit_acc /= count_acc

        # argmax -> predicted IDs
        pred_ids = torch.argmax(logit_acc, dim=0).cpu().numpy().astype(np.uint8)

        # IoU per class on this WSI
        ious = iou_per_class(pred_ids, gt_stitched, NUM_CLASSES)
        dices = dice_per_class(pred_ids, gt_stitched, NUM_CLASSES)
        precs, recs = precision_recall_per_class(pred_ids, gt_stitched, NUM_CLASSES)

        per_wsi_ious.append(ious)
        per_wsi_dices.append(dices)
        per_wsi_precisions.append(precs)
        per_wsi_recalls.append(recs)

        # save stitched prediction (colored) if requested
        if args.save_png:
            color = ID2COLOR[pred_ids]
            Image.fromarray(color).save(os.path.join(args.out_dir, f"{src_id}_pred.png"))

        # progress line
        miou = float(np.mean(ious))
        #print(f"[{idx+1}/{len(by_src)}] {src_id}  IoU: "
              #+ ", ".join(f"{n}:{v:.3f}" for n,v in zip(class_names, ious))
              #+ f"  | mIoU: {miou:.4f}")
        
        mdice = float(np.mean(dices))
        #print(f"[{idx+1}/{len(by_src)}] {src_id}  Dice: "
              #+ ", ".join(f"{n}:{v:.3f}" for n,v in zip(class_names, dices))
              #+ f"  | mDice: {mdice:.4f}")
        print(f"[{idx+1}/{len(by_src)}] {src_id}")
        print("  IoU : " + ", ".join(f"{n}:{v:.3f}" for n,v in zip(class_names, ious))
              + f"  | mIoU: {miou:.4f}")
        print("  Dice: " + ", ".join(f"{n}:{v:.3f}" for n,v in zip(class_names, dices))
              + f"  | mDice: {mdice:.4f}")
    # dataset summary
    per_wsi_ious = np.array(per_wsi_ious)  # (N,4)
    mean_per_class_iou = per_wsi_ious.mean(0).tolist()
    mean_miou = float(np.mean(mean_per_class_iou))
###########################################################################################################################
    per_wsi_dices = np.array(per_wsi_dices)
    mean_per_class_dice = per_wsi_dices.mean(0).tolist()
    mean_mdice = float(np.mean(mean_per_class_dice))
###########################################################################################################################
    per_wsi_precisions = np.array(per_wsi_precisions)
    per_wsi_recalls = np.array(per_wsi_recalls)
                               
    print("\n=== Validation (WSI-level) IoU ===")
    for n,v in zip(class_names, mean_per_class_iou):
        print(f"{n:>7}: {v:.4f}")
    print(f"mIoU  : {mean_miou:.4f}")

    print("\n=== Validation (WSI-level) Dice ===")
    for n,v in zip(class_names, mean_per_class_dice):
        print(f"{n:>7}: {v:.4f}")
    print(f"mDice  : {mean_mdice:.4f}")

    mean_precision = per_wsi_precisions.mean(0).tolist()
    mean_recall    = per_wsi_recalls.mean(0).tolist()
    print("\n=== Validation (WSI-level) Precision ===")
    for n, v in zip(class_names, mean_precision):
        print(f"{n:>7}: {v:.4f}")

    print("\n=== Validation (WSI-level) Recall ===")
    for n, v in zip(class_names, mean_recall):
        print(f"{n:>7}: {v:.4f}")
######################################################################################################################################    
@torch.no_grad()
def stitch_and_save_no_gt(args):
    """
    Stitch predictions from a tiles manifest (NO GT) and save per-WSI:
      - <src_id>_pred.png     (full-WSI colored prediction)
      - <src_id>_overlay.png  (full-WSI overlay = RGB stitched image + colored mask)

    Manifest must have columns: image_path,src_id,h,w,y,x
    (same geometry as val.csv, but no mask_path).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load stats for normalization (as in eval)
    stats_path = os.path.join(args.data_root, "processed", "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean, std = stats["rgb_mean"], stats["rgb_std"]

    # model
    from Models import models as modelz
    model = modelz.FCBFormer(size=args.img_size, num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # read tiles manifest (no masks)
    rows = []
    with open(args.tiles_manifest, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            row["h"] = int(row["h"]); row["w"] = int(row["w"])
            row["y"] = int(row["y"]); row["x"] = int(row["x"])
            rows.append(row)

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["src_id"]].append(r)

    tf = make_val_transform(args.img_size, mean, std)
    os.makedirs(args.out_dir, exist_ok=True)

    for idx, (src_id, tiles) in enumerate(sorted(by_src.items())):
        # canvas size from tiles
        H = max(r["y"] + r["h"] for r in tiles)
        W = max(r["x"] + r["w"] for r in tiles)

        # accumulators for logits + overlap counts
        logit_acc = torch.zeros(NUM_CLASSES, H, W, dtype=torch.float32, device=device)
        count_acc = torch.zeros(1, H, W, dtype=torch.float32, device=device)

        # stitch RGB image for overlay
        rgb_canvas = np.zeros((H, W, 3), dtype=np.uint8)
        
        # batched inference
        batch_imgs, batch_meta = [], []
        BATCH = args.batch_size

        def flush():
            if not batch_imgs:
                return
            x = torch.stack(batch_imgs, dim=0).to(device)   # (B,3,sz,sz)
            logits = model(x)                               # (B,C,sz,sz)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            for b in range(logits.shape[0]):
                r = batch_meta[b]
                logit_tile = upsample_to(logits[b], r["h"], r["w"], mode="bilinear")
                y0, x0 = r["y"], r["x"]
                logit_acc[:, y0:y0+r["h"], x0:x0+r["w"]] += logit_tile
                count_acc[:, y0:y0+r["h"], x0:x0+r["w"]] += 1.0
            batch_imgs.clear(); batch_meta.clear()

        for r in tiles:
            img = load_rgb(r["image_path"])
            # place RGB tile for overlay (ensure size matches manifest)
            rgb_canvas[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]] = np.array(img.resize((r["w"], r["h"])))

            x = tf(img)
            batch_imgs.append(x)
            batch_meta.append(r)
            if len(batch_imgs) == BATCH:
                flush()
        flush()
        # average overlaps
        count_acc = torch.clamp(count_acc, min=1.0)
        logit_acc /= count_acc
        # argmax -> pred ids + color
        pred_ids = torch.argmax(logit_acc, dim=0).cpu().numpy().astype(np.uint8)
        color = ID2COLOR[pred_ids]

        #ious = iou_per_class(pred_ids, rgb_canvas)
        #dices = dice_per_class(pred_ids, rgb_canvas)
        #precs, recs = precision_recall_per_class(pred_ids, rgb_canvas)

        # save colored prediction
        pred_path = os.path.join(args.out_dir, f"{src_id}_pred.png")
        Image.fromarray(color).save(pred_path)

        # save overlay
        alpha = getattr(args, "overlay_alpha", 0.4)
        overlay = (rgb_canvas * (1.0 - alpha) + color * alpha).astype(np.uint8)
        overlay_path = os.path.join(args.out_dir, f"{src_id}_overlay.png")
        Image.fromarray(overlay).save(overlay_path)

        print(f"[{idx+1}/{len(by_src)}] {src_id} -> saved pred + overlay")
######################################################################################################################################
def parse_args():
    ap = argparse.ArgumentParser("WSI-level evaluation for AIRA (FCBFormer)")
    ap.add_argument("--data-root", required=True,
                    help="datasets/ root that contains processed/{manifests,tiles...}")
    ap.add_argument("--checkpoint", required=True,
                    help="path to trained model checkpoint (*.pt)")
    ap.add_argument("--img-size", type=int, default=352,
                    help="model input size used in training (e.g., 352)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="./EvalWSI")
    ap.add_argument("--save-png", action="store_true",
                    help="save colored stitched predictions")
    ap.add_argument("--tiles-manifest", type=str, default="",
                    help="CSV manifest with tiles but no masks (image_path,src_id,h,w,y,x). "
                         "If set, stitches full-WSI predictions and saves pred & overlay.")
    ap.add_argument("--overlay-alpha", type=float, default=0.4,
                    help="Alpha for overlay blending (0..1). Used in tiles-manifest mode.")
    return ap.parse_args()

def main():
    # keep single-threaded CPU ops stable
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = parse_args()

    if args.tiles_manifest:
        # inference-only stitching with no GT; saves <src_id>_pred.png and <src_id>_overlay.png
        stitch_and_save_no_gt(args)
    else:
        # validation mode using processed/manifests/val.csv with GT
        stitch_and_eval(args)

if __name__ == "__main__":
    main()





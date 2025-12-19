import sys
import os
import argparse
import time
import numpy as np
import glob
from pathlib import Path
import csv
from datetime import datetime
import subprocess

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
from Metrics.log_helper import AvgMeter

from Data.dataloaders_joint import get_loaders_from_manifests
import json, random


def train_epoch(model, device, train_loader, optimizer, epoch, Dice_loss, BCE_loss, 
                ce_loss=None, dice_loss_mc=None, bnd_loss=None, tumor_dil_loss=None, benign_dil_loss=None,
                boundary_weight=0.2, tumor_dil_weight=0.2, benign_dil_weight=0.2, num_classes=1):
    t = time.time()
    model.train()
    loss_accumulator = []
    
    # new meters
    m_ce   = AvgMeter(); m_dice = AvgMeter()
    m_bnd  = AvgMeter(); m_tum  = AvgMeter(); m_ben = AvgMeter()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        if num_classes > 1:
            # AIRA (4 classes): use CE on logits + multiclass Dice
            target_long = target.long()
            l_dice = dice_loss_mc(output, target_long); m_dice.update(l_dice.item(), data.size(0))
            l_ce   = ce_loss(output, target_long);      m_ce.update(l_ce.item(),   data.size(0))
            loss = l_dice + l_ce
            if bnd_loss is not None and boundary_weight > 0:
                l_bnd = bnd_loss(output, target_long);  m_bnd.update(l_bnd.item(), data.size(0))
                loss = loss + boundary_weight * l_bnd
            else:
                l_bnd = torch.tensor(0.0, device=device)
            if tumor_dil_loss is not None and tumor_dil_weight > 0:
                l_tum = tumor_dil_loss(output, target_long); m_tum.update(l_tum.item(), data.size(0))
                loss = loss + tumor_dil_weight * l_tum
            else:
                l_tum = torch.tensor(0.0, device=device)
            if benign_dil_loss is not None and benign_dil_weight > 0:
                l_ben = benign_dil_loss(output, target_long); m_ben.update(l_ben.item(), data.size(0))
                loss = loss + benign_dil_weight * l_ben
            else:
                l_ben = torch.tensor(0.0, device=device)
                
            # Debug print every 50 batches
            if (batch_idx % 50) == 0:
                def _f(x): return float(x.detach().cpu()) if isinstance(x, torch.Tensor) else float(x)
                print(f"\n [losses] CE:{_f(l_ce):.4f} Dice:{_f(l_dice):.4f} Bnd:{_f(l_bnd):.4f} TumDil:{_f(l_tum):.4f} BenDil:{_f(l_ben):.4f}")

        else:
            # Binary path (Kvasir/CVC): keep original behavior
            l_dice = Dice_loss(output, target); m_dice.update(l_dice.item(), data.size(0))
            l_ce   = BCE_loss(output, target);  m_ce.update(l_ce.item(),   data.size(0))
            loss = l_dice + l_ce

        loss.backward()
        optimizer.step()
        loss_accumulator.append(loss.item())

        # (optional) pretty printing kept as-is...
        if batch_idx + 1 < len(train_loader):
            print("\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100.0 * (batch_idx + 1) / len(train_loader),
                loss.item(), time.time() - t,
            ), end="")
        else:
            print("\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100.0 * (batch_idx + 1) / len(train_loader),
                np.mean(loss_accumulator), time.time() - t,
            ))

    # return epoch averages (for CSV)
    return {
        "loss": float(np.mean(loss_accumulator)),
        "ce": m_ce.avg, "dice": m_dice.avg,
        "bnd": m_bnd.avg, "tum": m_tum.avg, "ben": m_ben.avg,
    }


@torch.no_grad()
def iou_per_class(logits, targets, num_classes=4):
    preds = torch.argmax(logits, dim=1)  # (B,H,W)
    ious = []
    for c in range(num_classes):
        pred_c = (preds == c)
        targ_c = (targets == c)
        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()
        ious.append(inter / union if union > 0 else 0.0)
    return ious, sum(ious)/len(ious)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure=None, num_classes=4):
    t = time.time()
    model.eval()
    perf_accumulator = []
    all_ious = []

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        if perf_measure is not None and num_classes == 1:
            perf_accumulator.append(perf_measure(output, target).item())

        if num_classes > 1:
            ious, miou = iou_per_class(output, target, num_classes=num_classes)
            all_ious.append(ious)

    if num_classes > 1 and all_ious:
        mean_ious = np.mean(all_ious, axis=0).tolist()
        miou = float(np.mean(mean_ious))
        print(f"[Val IoU] per-class: {mean_ious}, mIoU: {miou:.4f}")
        # Return mIoU so scheduler/checkpoint use it
        return miou, 0.0, { "bg": mean_ious[0], "stroma": mean_ious[1], "benign": mean_ious[2], "tumor": mean_ious[3] }

    # binary fallback
    return (
        float(np.mean(perf_accumulator)) if perf_accumulator else 0.0,
        float(np.std(perf_accumulator)) if perf_accumulator else 0.0,
        {}, 
    )



def build(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset in ["Kvasir", "CVC"]:
        # existing logic
        if args.dataset == "Kvasir":
            img_path = args.root + "images/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "masks/*"
            target_paths = sorted(glob.glob(depth_path))
        elif args.dataset == "CVC":
            img_path = args.root + "Original/*"
            input_paths = sorted(glob.glob(img_path))
            depth_path = args.root + "Ground Truth/*"
            target_paths = sorted(glob.glob(depth_path))
        train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
            input_paths, target_paths, batch_size=args.batch_size
        )
        num_classes = 1  # original binary
        ce_loss = nn.BCEWithLogitsLoss()
        dice_loss = losses.SoftDiceLoss()
        bnd_loss  = None
        tumor_dil_loss  = None
        benign_dil_loss = None

    elif args.dataset == "AIRA":
        train_csv = os.path.join(args.root, "processed", "manifests", "train.csv")
        val_csv   = os.path.join(args.root, "processed", "manifests", "val.csv")
        stats_json = os.path.join(args.root, "processed", "stats.json")

        train_dataloader, val_dataloader = get_loaders_from_manifests(
            train_csv, val_csv, stats_json,
            img_size=args.img_size, batch_size=args.batch_size,
            num_workers=0
        )
        num_classes = 4

        # ---- NEW: class-weighted CE from TRAIN stats ----
        with open(stats_json) as f:
            stats = json.load(f)
        cc = stats["class_counts"]
        freq = torch.tensor(
            [cc["background"], cc["stroma"], cc["benign"], cc["tumor"]],
            dtype=torch.float
        )
        # log-balancing, then normalize to mean=1
        weights = 1.0 / torch.log(1.02 + freq / freq.sum())
        weights = weights / weights.mean()
        weights = weights.to(device)

        ce_loss = nn.CrossEntropyLoss(weight=weights)
        dice_loss = losses.MultiClassDiceLoss(num_classes=num_classes)
        bnd_loss = losses.BoundaryDiceLossAgnostic(num_classes=num_classes, kernel_size=3)
        
        # NEW class-focused dilated losses
        tumor_dil_loss  = losses.TumorDilatedDiceLoss(
            num_classes=num_classes, iters=args.dil_iters, 
            kernel_size=args.dil_kernel, prob_power=args.prob_power
        )
        benign_dil_loss = losses.BenignDilatedDiceLoss(
            num_classes=num_classes, iters=args.dil_iters, 
            kernel_size=args.dil_kernel, prob_power=args.prob_power
        )
        
    else:
        raise ValueError("Unknown dataset")

    # model
    model = models.FCBFormer(size=args.img_size, num_classes=num_classes)  # ensure your FCBFormer accepts num_classes
    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    
    # ---- OPTIONAL: warm-start from a checkpoint if present ----
    ckpt_path = os.getenv("WARM_START_CKPT", "")
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[warm-start] Loaded weights from {ckpt_path}")
    # ------------------------------------------------------------

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # perf metric (DiceScore in your repo is probably binary; for AIRA use CE + Dice and report IoU in test())
    perf = performance_metrics.DiceScore() if num_classes == 1 else None

    return device, train_dataloader, val_dataloader, dice_loss, ce_loss, bnd_loss, tumor_dil_loss, benign_dil_loss, perf, model, optimizer, num_classes



def train(args, run_dir: Path):
    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        CE_loss,
        BND_loss,
        TumorDilLoss,
        BenignDilLoss, 
        perf,
        model,
        optimizer,
        num_classes, 
    ) = build(args)
    
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    train_csv = run_dir / "logs" / "train_log.csv"
    val_csv   = run_dir / "logs" / "val_log.csv"
    
    # create headers if new
    if not train_csv.exists():
        with open(train_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","loss","ce","dice","bnd","tum","ben","lr"])
    if not val_csv.exists():
        with open(val_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","miou","bg","stroma","benign","tumor"])

    prev_best_test = None
    scheduler = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, verbose=True
            )
    for epoch in range(1, args.epochs + 1):
        try:
            train_stats = train_epoch(
                model, device, train_dataloader, optimizer, epoch,
                Dice_loss, CE_loss,
                ce_loss=CE_loss if num_classes > 1 else None,
                dice_loss_mc=Dice_loss if num_classes > 1 else None,
                bnd_loss=BND_loss if num_classes > 1 else None,
                tumor_dil_loss=TumorDilLoss if num_classes > 1 else None,
                benign_dil_loss=BenignDilLoss if num_classes > 1 else None,
                boundary_weight=args.boundary_weight,
                tumor_dil_weight=args.tumor_dil_weight,
                benign_dil_weight=args.benign_dil_weight,
                num_classes=num_classes,
            )

            test_measure_mean, test_measure_std, val_detail = test(
                model, device, val_dataloader, epoch,
                perf_measure=perf,
                num_classes=num_classes,
            )
            
            # CSV LOGGING
            lr_now = optimizer.param_groups[0]["lr"]
            with open(train_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, train_stats["loss"], train_stats["ce"], train_stats["dice"],
                            train_stats["bnd"], train_stats["tum"], train_stats["ben"], lr_now])

            if val_detail:
                with open(val_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([epoch, test_measure_mean,
                            val_detail["bg"], val_detail["stroma"], val_detail["benign"], val_detail["tumor"]])
            else:
                with open(val_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([epoch, test_measure_mean, "", "", "", ""])
            
        except KeyboardInterrupt:
            print("Training interrupted by user")
            sys.exit(0)
        if scheduler is not None:
            scheduler.step(test_measure_mean)
        # state dict (shared by both saves)
        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict() if args.mgpu == "false" else model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "loss": train_stats["loss"],
            "val_mIoU": test_measure_mean,
            "val_mIoU_std": test_measure_std,
            "args": vars(args),
        }
        torch.save(state_dict, run_dir / "last_FCBFormer.pt")
        if prev_best_test is None or test_measure_mean > prev_best_test:
            print("Saving best...")
            torch.save(state_dict, run_dir / "best_FCBFormer.pt")
            prev_best_test = test_measure_mean



def get_args():
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--dataset", type=str, required=True, choices=["Kvasir","CVC","AIRA"])
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-2, dest="lr")
    parser.add_argument(
        "--learning-rate-scheduler", type=str, default="true", dest="lrs"
    )
    parser.add_argument(
        "--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min"
    )
    parser.add_argument(
        "--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"]
    )
    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--tumor-dil-weight", type=float, default=0.2)
    parser.add_argument("--benign-dil-weight", type=float, default=0.1)
    parser.add_argument("--dil-iters", type=int, default=3)
    parser.add_argument("--dil-kernel", type=int, default=3)
    parser.add_argument("--prob-power", type=float, default=1.0)

    return parser.parse_args()


def main():
    args = get_args()

    run_dir = Path("outputs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # save args for reproducibility
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # set & save seeds
    seed = 1337
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    with open(run_dir / "seed.txt", "w") as f: f.write(str(seed))
    
    # environment snapshot
    freeze = subprocess.run(["pip","freeze"], capture_output=True, text=True).stdout
    with open(run_dir / "requirements_freeze.txt", "w") as f:
        f.write(freeze)
    
    train(args, run_dir)


if __name__ == "__main__":
    main()

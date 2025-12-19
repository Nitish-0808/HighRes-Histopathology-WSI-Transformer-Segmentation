import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score
        

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=4, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    def forward(self, logits, targets):
        # logits: (B,C,H,W), targets: (B,H,W) long
        probs = torch.softmax(logits, dim=1)
        target_1h = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()
        

class BoundaryDiceLoss(nn.Module):
    """
    Differentiable boundary loss for multi-class seg.
    - Extracts per-class 'soft edges' via morphological gradient:
        edge(x) = maxpool(x) - minpool(x)  (minpool implemented as -maxpool(-x))
    - Computes Dice on edge maps between softmax(pred) and one-hot(GT).
    """
    def __init__(self, num_classes=4, ignore_background=True, smooth=1.0, kernel_size=3):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.smooth = smooth
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def _morph_gradient(self, x):
        # x: (B, C, H, W), values in [0,1]
        x_max = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        x_min = -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        return x_max - x_min

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W) int64 in [0...C-1]
        probs = torch.softmax(logits, dim=1)                         # (B,C,H,W)
        tgt_1h = F.one_hot(targets, num_classes=self.num_classes)    # (B,H,W,C)
        tgt_1h = tgt_1h.permute(0, 3, 1, 2).float()                  # (B,C,H,W)

        # soft edges
        pred_edge = self._morph_gradient(probs)
        tgt_edge  = self._morph_gradient(tgt_1h)

        # optionally drop background channel from the edge dice
        if self.ignore_background and self.num_classes > 1:
            pred_edge = pred_edge[:, 1:, ...]
            tgt_edge  = tgt_edge[:,  1:, ...]

        # edge dice
        dims = (0, 2, 3)
        inter = (pred_edge * tgt_edge).sum(dims)
        denom = pred_edge.sum(dims) + tgt_edge.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BoundaryDiceLossAgnostic(nn.Module):
    """
    Class-agnostic boundary loss for multi-class segmentation.

    Steps:
      1) Softmax -> per-class probability maps (B,C,H,W)
      2) Morphological gradient per class: maxpool - minpool (differentiable)
      3) Max across classes -> single edge map (B,1,H,W) for pred & GT
      4) Dice loss between predicted edge map and GT edge map.

    This supervises *geometry* of boundaries independent of class labels.
    """
    def __init__(self, num_classes=4, smooth=1.0, kernel_size=3, reduction="mean"):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.reduction = reduction

    def _morph_grad(self, x):
        # x: (B,C,H,W), values in [0,1]
        x_max = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        x_min = -F.max_pool2d(-x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        return x_max - x_min

    def _edge_map_agnostic(self, per_class):
        # per_class: (B,C,H,W)
        # compute per-class gradients then take max across channels -> (B,1,H,W)
        g = self._morph_grad(per_class)          # (B,C,H,W)
        g_max = g.max(dim=1, keepdim=True).values
        return g_max

    def forward(self, logits, targets):
        """
        logits:  (B,C,H,W)
        targets: (B,H,W) int64 in [0..C-1]
        """
        # prediction edges (soft)
        probs = torch.softmax(logits, dim=1)                        # (B,C,H,W)
        pred_edge = self._edge_map_agnostic(probs)                  # (B,1,H,W)

        # GT edges (from one-hot)
        tgt_1h = F.one_hot(targets, num_classes=self.num_classes)   # (B,H,W,C)
        tgt_1h = tgt_1h.permute(0,3,1,2).float()                    # (B,C,H,W)
        tgt_edge = self._edge_map_agnostic(tgt_1h)                  # (B,1,H,W)

        # Dice on edge maps
        dims = (0,2,3)  # sum over batch and spatial
        inter = (pred_edge * tgt_edge).sum(dims)
        denom = pred_edge.sum(dims) + tgt_edge.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)  # scalar
        loss = 1.0 - dice

        if self.reduction == "none":
            return loss
        # here loss is already reduced across dims; just ensure tensor type
        return loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=logits.device)
        

class ClassFocusDilatedDiceLoss(nn.Module):
    """
    Soft, class-specific Dice on *dilated* masks.
    - Picks one class from softmax(logits).
    - Dilation via repeated max-pooling (differentiable).
    - Computes soft Dice between dilated pred-prob and dilated GT one-hot.

    Args:
        class_id: which class channel to focus on (e.g., tumor=3, benign=2 by your mapping)
        num_classes: total classes in logits
        iters: number of dilation iterations (e.g., 3~5)
        kernel_size: pooling kernel used as dilation structuring element
        smooth: dice smoothing
        prob_power: raise preds to this power to sharpen/soften (gamma-like). 1.0 = none.
    """
    def __init__(self, class_id: int, num_classes: int = 4,
                 iters: int = 3, kernel_size: int = 3, smooth: float = 1.0,
                 prob_power: float = 1.0):
        super().__init__()
        self.class_id = class_id
        self.num_classes = num_classes
        self.iters = iters
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.smooth = smooth
        self.prob_power = prob_power

    def _dilate(self, x):
        # x: (B,1,H,W), values in [0,1]
        for _ in range(self.iters):
            x = F.max_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        return x

    def forward(self, logits, targets_long):
        """
        logits: (B,C,H,W)
        targets_long: (B,H,W) int64
        """
        probs = torch.softmax(logits, dim=1)                            # (B,C,H,W)
        pred_c = probs[:, self.class_id:self.class_id+1, :, :]           # (B,1,H,W)

        if self.prob_power != 1.0:
            pred_c = torch.clamp(pred_c, 1e-6, 1-1e-6) ** self.prob_power

        tgt_1h = F.one_hot(targets_long, num_classes=self.num_classes)   # (B,H,W,C)
        tgt_c = tgt_1h[..., self.class_id].unsqueeze(1).float()          # (B,1,H,W)

        # dilate both
        pred_c_d = self._dilate(pred_c)
        tgt_c_d  = self._dilate(tgt_c)

        # soft dice on dilated maps
        inter = (pred_c_d * tgt_c_d).sum(dim=(0,2,3))
        denom = pred_c_d.sum(dim=(0,2,3)) + tgt_c_d.sum(dim=(0,2,3))
        dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class TumorDilatedDiceLoss(ClassFocusDilatedDiceLoss):
    def __init__(self, num_classes=4, iters=3, kernel_size=3, smooth=1.0, prob_power=1.0):
        # tumor = class id 3 in your mapping
        super().__init__(class_id=3, num_classes=num_classes, iters=iters,
                         kernel_size=kernel_size, smooth=smooth, prob_power=prob_power)


class BenignDilatedDiceLoss(ClassFocusDilatedDiceLoss):
    def __init__(self, num_classes=4, iters=3, kernel_size=3, smooth=1.0, prob_power=1.0):
        # benign = class id 2 in your mapping
        super().__init__(class_id=2, num_classes=num_classes, iters=iters,
                         kernel_size=kernel_size, smooth=smooth, prob_power=prob_power)


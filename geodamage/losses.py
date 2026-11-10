"""Masked weighted BCE + Dice loss (paper Sec. 3.4).

"the objective combines a masked, weighted binary cross-entropy term (class
weights 1.0 for damaged and approximately 2.0969 for likely-intact pixels)
with a Dice-overlap term (weight 0.25) ... ignore pixels are masked out of
both terms."

The paper's 2.0969 is a property of their own training label distribution; we
provide `compute_class_weight` to derive the analogous weight from whatever
training masks are actually produced by `weak_labels.py`, rather than
hardcoding their number.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


def compute_class_weight(label_masks, damaged_value=config.LABEL_DAMAGED,
                          intact_value=config.LABEL_LIKELY_INTACT):
    """Inverse-frequency weight for the minority (likely-intact) class: ratio
    of damaged-pixel count to likely-intact-pixel count across one or more
    label masks (numpy arrays with values in {LABEL_DAMAGED, LABEL_LIKELY_INTACT,
    LABEL_IGNORE})."""
    n_damaged = sum((m == damaged_value).sum() for m in label_masks)
    n_intact = sum((m == intact_value).sum() for m in label_masks)
    if n_intact == 0:
        raise ValueError("No likely-intact pixels found; cannot compute class weight")
    return float(n_damaged) / float(n_intact)


class MaskedWeightedBCEDiceLoss(nn.Module):
    def __init__(self, weight_damaged=1.0, weight_intact=2.0969,
                 dice_weight=config.LOSS_DICE_WEIGHT, ignore_value=config.LABEL_IGNORE, eps=1e-6):
        super().__init__()
        self.weight_damaged = weight_damaged
        self.weight_intact = weight_intact
        self.dice_weight = dice_weight
        self.ignore_value = ignore_value
        self.eps = eps

    def forward(self, logits: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        """logits: (N,1,H,W) raw scores. label_mask: (N,1,H,W) or (N,H,W) with
        values in {0 (likely-intact), 1 (damaged), LABEL_IGNORE}."""
        if label_mask.dim() == 3:
            label_mask = label_mask.unsqueeze(1)
        valid = (label_mask != self.ignore_value).float()
        targets = torch.where(label_mask == self.ignore_value, torch.zeros_like(label_mask), label_mask).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weight_map = torch.where(targets == 1, self.weight_damaged, self.weight_intact)
        bce = (bce * weight_map * valid).sum() / valid.sum().clamp_min(self.eps)

        probs = torch.sigmoid(logits) * valid
        targets_masked = targets * valid
        intersection = (probs * targets_masked).sum()
        union = probs.sum() + targets_masked.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)

        return bce + self.dice_weight * dice_loss

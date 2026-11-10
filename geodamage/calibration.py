"""Threshold calibration and evaluation metrics (paper Sec. 3.5, 4.1-4.3).

Two calibration stages, both validation-only and frozen before touching test
data (paper: "validation-only calibration of both a pixel-level decision
threshold and building-level confidence tiers, frozen before test inference"):

1. Pixel threshold: swept against full-section inference on the validation
   section, selected by max macro-F1 (the paper doesn't fully spell out its
   selection rule beyond "selected threshold (0.23)" from the sweep figure, so
   max-macro-F1 is our own documented, explicit choice).
2. Building-level thresholds: screening threshold + 90%/95%-specificity
   thresholds for the likely-intact class, from the ROC of building-level
   scores (Eq. 1) on validation buildings.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, jaccard_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, average_precision_score,
)

from . import config


@dataclass
class PixelMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    macro_iou: float
    balanced_accuracy: float
    roc_auc: float
    average_precision: float
    mcc: float
    overall_accuracy: float


def _valid_mask(labels):
    return labels != config.LABEL_IGNORE


def compute_pixel_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> PixelMetrics:
    """probs, labels: flat arrays (or any shape, will be flattened). labels in
    {LABEL_DAMAGED, LABEL_LIKELY_INTACT, LABEL_IGNORE}; ignore pixels excluded."""
    probs = probs.ravel()
    labels = labels.ravel()
    valid = _valid_mask(labels)
    p, y = probs[valid], labels[valid].astype(int)
    pred = (p >= threshold).astype(int)

    roc_auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")

    return PixelMetrics(
        threshold=threshold,
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        macro_iou=jaccard_score(y, pred, average="macro", zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        roc_auc=roc_auc,
        average_precision=ap,
        mcc=matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else float("nan"),
        overall_accuracy=(pred == y).mean(),
    )


def sweep_pixel_threshold(probs: np.ndarray, labels: np.ndarray,
                           thresholds=np.linspace(0.01, 0.99, 99)):
    """Returns (list[PixelMetrics], best: PixelMetrics) selecting by max macro-F1."""
    results = [compute_pixel_metrics(probs, labels, t) for t in thresholds]
    best = max(results, key=lambda r: r.macro_f1)
    return results, best


def building_level_score(pixel_probs: np.ndarray, footprint_mask: np.ndarray) -> float:
    """Eq. (1): mean predicted pixel probability within a building's footprint."""
    vals = pixel_probs[footprint_mask]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def calibrate_building_thresholds(scores: np.ndarray, weak_labels: np.ndarray,
                                   specificities=(0.90, 0.95)):
    """Calibrate on validation buildings (weak_labels: 0=likely-intact,
    1=damaged, ignore entries excluded beforehand by the caller).

    Returns a dict with:
      - 'screening': threshold maximizing building-level macro-F1 (paper's
        broad "screening threshold")
      - one key per requested specificity, e.g. 'spec_90', 'spec_95': the score
        threshold at which that fraction of likely-intact validation buildings
        score below it (i.e. that level of specificity for the likely-intact
        class), mirroring the paper's 90%/95%-specificity thresholds.
    """
    valid = ~np.isnan(scores)
    s, y = scores[valid], weak_labels[valid].astype(int)

    candidate_thresholds = np.unique(s)
    best_t, best_f1 = 0.5, -1.0
    for t in candidate_thresholds:
        pred = (s >= t).astype(int)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    intact_scores = s[y == 0]
    out = {"screening": float(best_t)}
    for spec in specificities:
        out[f"spec_{int(spec*100)}"] = float(np.quantile(intact_scores, spec)) if intact_scores.size else float("nan")
    return out


def assign_tier(score: float, thresholds: dict) -> str:
    if np.isnan(score):
        return "unscored"
    if score >= thresholds.get("spec_95", np.inf):
        return config.TIER_VERY_HIGH_CONF
    if score >= thresholds.get("spec_90", np.inf):
        return config.TIER_HIGH_CONF
    if score >= thresholds.get("screening", np.inf):
        return config.TIER_SCREENING
    return config.TIER_LOW

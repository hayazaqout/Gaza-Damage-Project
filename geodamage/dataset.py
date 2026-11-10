"""128x128x6 patch extraction and a PyTorch Dataset (paper Sec. 3.3).

"Standardized rasters were cut into 128x128 patches with six channels --
Before-R, Before-G, Before-B, After-R, After-G, After-B -- normalized to
[0,1]." The paper doesn't specify a patch stride; we default to non-overlapping
tiling (stride = patch size), which is the simplest reading and avoids the
"optimistic bias" the paper explicitly warns about for overlapping patches at
the section/split level (Sec. "spatially isolated... avoiding optimistic bias
of random overlapping-patch validation"). A stride can still be supplied
explicitly for overlapping training patches if desired.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


def stack_before_after(before_img: np.ndarray, after_img: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 + (H,W,3) uint8 -> (H,W,6) float32 in [0,1]."""
    stacked = np.concatenate([before_img, after_img], axis=-1).astype(np.float32) / 255.0
    return stacked


def extract_patches(array: np.ndarray, patch_size=config.PATCH_SIZE, stride=None):
    """Tile a (H,W) or (H,W,C) array into patch_size x patch_size windows,
    padding the bottom/right edge with zeros (image) / ignore value (mask) as
    needed so every patch is full-size. Returns a list of patches."""
    stride = stride or patch_size
    h, w = array.shape[:2]
    pad_h = (-h) % patch_size if h % patch_size else 0
    pad_w = (-w) % patch_size if w % patch_size else 0
    if array.ndim == 3:
        padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    else:
        padded = np.pad(array, ((0, pad_h), (0, pad_w)), mode="constant",
                         constant_values=config.LABEL_IGNORE)

    patches = []
    H, W = padded.shape[:2]
    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            patches.append(padded[row:row + patch_size, col:col + patch_size])
    return patches


class SectionPatchDataset(Dataset):
    """Wraps one or more (6-channel image, label mask) section pairs as a flat
    collection of aligned 128x128x6 image patches + 128x128 label patches."""

    def __init__(self, section_pairs, patch_size=config.PATCH_SIZE, stride=None):
        self.image_patches = []
        self.label_patches = []
        for image6ch, label_mask in section_pairs:
            img_patches = extract_patches(image6ch, patch_size, stride)
            lbl_patches = extract_patches(label_mask, patch_size, stride)
            assert len(img_patches) == len(lbl_patches)
            self.image_patches.extend(img_patches)
            self.label_patches.extend(lbl_patches)

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.image_patches[idx]).permute(2, 0, 1).float()  # (6,H,W)
        lbl = torch.from_numpy(self.label_patches[idx]).long()  # (H,W)
        return img, lbl

"""
PyTorch Dataset & DataLoader utilities for cell cycle images.
Supports both synthetic and real microscopy datasets with the same folder layout.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

from src.config import (
    PHASES, PHASE_TO_IDX, IMG_SIZE, IMG_MEAN, IMG_STD,
    BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED, SYNTHETIC_DIR
)


class CellCycleDataset(Dataset):
    """
    Loads cell images from a directory structure:
        root/<phase_name>/img_XXXX.png
    """

    def __init__(self, root: str = SYNTHETIC_DIR, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []   # list of (path, label_idx)

        for phase in PHASES:
            phase_dir = os.path.join(root, phase)
            if not os.path.isdir(phase_dir):
                continue
            label = PHASE_TO_IDX[phase]
            for fname in sorted(os.listdir(phase_dir)):
                if fname.lower().endswith((".png", ".jpg", ".tif")):
                    self.samples.append((os.path.join(phase_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ────────────────────────────────────────────────

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])


# ── DataLoader factory ───────────────────────────────────────

def create_dataloaders(root: str = SYNTHETIC_DIR,
                       batch_size: int = BATCH_SIZE,
                       seed: int = RANDOM_SEED):
    """
    Create train / val / test DataLoaders with stratified random splits.
    Returns (train_loader, val_loader, test_loader).
    """
    full_ds = CellCycleDataset(root, transform=None)
    n = len(full_ds)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_ds = _TransformSubset(full_ds, train_idx, get_train_transform())
    val_ds   = _TransformSubset(full_ds, val_idx,   get_eval_transform())
    test_ds  = _TransformSubset(full_ds, test_idx,  get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    print(f"  Dataset splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_loader, val_loader, test_loader


class _TransformSubset(Dataset):
    """Subset wrapper that applies a transform."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

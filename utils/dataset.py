"""
utils/dataset.py
Data loading, splitting, and augmentation for the PlantVillage tomato dataset.

Augmentation uses torchvision.transforms.v2 (ships with torchvision) rather than
albumentations, so the project installs cleanly on any machine that can install
torch — no compiler toolchain required.
"""

import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

import config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ─── Augmentation pipelines ──────────────────────────────────────────────────
def get_train_transforms() -> v2.Compose:
    """Geometric + photometric augmentation, then normalise to ImageNet stats."""
    return v2.Compose([
        v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.3),
        v2.RandomRotation(degrees=30),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        # Coarse dropout equivalent — applied after normalisation, like the original.
        v2.RandomErasing(p=0.3, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0.0),
    ])


def get_val_transforms() -> v2.Compose:
    """Deterministic: resize + normalise only."""
    return v2.Compose([
        v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────
class TomatoDiseaseDataset(Dataset):
    """
    Folder structure expected:
        PlantVillage/
            Tomato_Early_blight/   ← folder name = class name
                img001.jpg
                ...
            Tomato_healthy/
                ...
    """

    def __init__(self, samples: List[Tuple[str, int]], transforms=None):
        self.samples    = samples      # [(path, label), ...]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, label


# ─── Build samples list from disk ─────────────────────────────────────────────
def build_samples(data_dir: Path, verbose: bool = True) -> List[Tuple[str, int]]:
    """
    Scan data_dir subfolders; folder name must match config.CLASS_NAMES.
    Returns list of (image_path, class_index).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            f"Set the TOMATO_DATA_DIR environment variable to your PlantVillage "
            f"folder, or place it at {config.BASE_DIR / 'PlantVillage'}."
        )

    samples: List[Tuple[str, int]] = []
    class_to_idx = {c: i for i, c in enumerate(config.CLASS_NAMES)}
    found_classes = set()
    cap = config.MAX_IMAGES_PER_CLASS

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in class_to_idx:
            if verbose:
                print(f"  [SKIP] Unknown folder: {folder.name}")
            continue

        label = class_to_idx[folder.name]
        imgs  = sorted(str(p) for p in folder.iterdir()
                       if p.suffix.lower() in IMG_EXTENSIONS)
        if cap:
            imgs = imgs[:cap]

        samples.extend((img, label) for img in imgs)
        found_classes.add(folder.name)
        if verbose:
            print(f"  {folder.name}: {len(imgs)} images")

    missing = [c for c in config.CLASS_NAMES if c not in found_classes]
    if missing:
        raise FileNotFoundError(
            f"Dataset at {data_dir} is missing {len(missing)} class folder(s): "
            f"{missing}"
        )
    if not samples:
        raise FileNotFoundError(f"No images found under {data_dir}")

    return samples


# ─── Split into train / val / test ───────────────────────────────────────────
def split_samples(samples):
    """
    Stratified split — each class keeps the same train/val/test proportions.
    Matters here because the classes are imbalanced (373 mosaic-virus images
    vs 3209 yellow-leaf-curl); a plain random split can starve small classes.
    Deterministic given config.RANDOM_SEED.
    """
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))

    rng = random.Random(config.RANDOM_SEED)
    train, val, test = [], [], []

    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)
        n       = len(items)
        n_train = int(n * config.TRAIN_SPLIT)
        n_val   = int(n * config.VAL_SPLIT)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Shuffle across classes so batches aren't class-ordered.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ─── DataLoader factory ───────────────────────────────────────────────────────
def get_dataloaders(data_dir: Path, batch_size: int = None, verbose: bool = True):
    if verbose:
        print(f"\n📂  Scanning dataset … ({data_dir})")
    samples = build_samples(data_dir, verbose=verbose)
    if verbose:
        print(f"\n  Total images: {len(samples)}")

    train_s, val_s, test_s = split_samples(samples)
    if verbose:
        print(f"  Train: {len(train_s)}  |  Val: {len(val_s)}  |  Test: {len(test_s)}")

    train_tf = get_train_transforms() if config.USE_AUGMENTATION else get_val_transforms()
    val_tf   = get_val_transforms()

    train_ds = TomatoDiseaseDataset(train_s, train_tf)
    val_ds   = TomatoDiseaseDataset(val_s,   val_tf)
    test_ds  = TomatoDiseaseDataset(test_s,  val_tf)

    num_workers = config.NUM_WORKERS
    loader_kwargs = dict(
        batch_size  = batch_size or config.BATCH_SIZE,
        num_workers = num_workers,
        pin_memory  = config.PIN_MEMORY,
        persistent_workers = num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=False, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ─── Torchvision transform for inference (single PIL image) ──────────────────
inference_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

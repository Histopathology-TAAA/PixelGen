"""
Paired H&E → IHC Dataset for StarDiff Training.
Synchronized random crop + geometric augmentation + residual computation.
"""
import os
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm
from typing import List, Tuple, Optional, Any


# ── Kaggle download helpers ──

STAIN_TO_SLUG = {
    "KI67": "mist-preprocessed-512-ki67",
    "ER": "mist-preprocessed-512-er",
    "PR": "mist-preprocessed-512-pr",
}


def download_dataset(dataset_root: str, stains):
    """Download preprocessed 512×512 paired patches from Kaggle."""
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    for stain in stains:
        stain_path = root / stain
        if not stain_path.exists():
            slug = STAIN_TO_SLUG[stain]
            print(f"📥 Downloading preprocessed {stain} dataset...")
            os.system(f"kaggle datasets download -d mohamedtarek26/{slug} -p {root} --unzip")
        else:
            print(f"✓ {stain} already exists at {stain_path}")

    return root


def get_dataset_paths(dataset_root, stains):
    """Return dict of {stain: {train_he, train_ihc, val_he, val_ihc}} paths."""
    root = Path(dataset_root)
    paths = {}
    for stain in stains:
        paths[stain] = {
            "train_he": root / stain / "train" / "he",
            "train_ihc": root / stain / "train" / "ihc",
            "val_he": root / stain / "val" / "he",
            "val_ihc": root / stain / "val" / "ihc",
        }
    return paths


# ── Dataset class ──

class StarDiffPairedDataset(Dataset):
    """
    Paired H&E → IHC dataset with:
    - Synchronized random crop (512→256)
    - Synchronized geometric augmentations (flips, 90° rotations)
    - Medical-safe: NO color jitter on IHC
    - Computes restoration residual: I_res = I_ihc - I_he
    """

    def __init__(self, he_dir, ihc_dir, resolution=256, source_resolution=512, augment=True, preload_to_ram=False):
        self.he_dir = Path(he_dir)
        self.ihc_dir = Path(ihc_dir)
        self.resolution = resolution
        self.source_resolution = source_resolution
        self.augment = augment

        # Match filenames across he/ihc dirs
        he_files = set(f.name for f in self.he_dir.glob("*.png"))
        he_files.update(f.name for f in self.he_dir.glob("*.jpg"))
        ihc_files = set(f.name for f in self.ihc_dir.glob("*.png"))
        ihc_files.update(f.name for f in self.ihc_dir.glob("*.jpg"))
        self.paired_files = sorted(list(he_files & ihc_files))
        print(f"  Loaded {len(self.paired_files)} paired H&E/IHC images from {he_dir}")

        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.preload_to_ram = preload_to_ram
        self.preloaded_data: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if self.preload_to_ram:
            print(f"🚀 Preloading {len(self.paired_files)} images into System RAM (as uint8) to eliminate IO bottleneck...")
            self.preloaded_data = [] # type: ignore
            for filename in tqdm(self.paired_files, desc="Preloading"):
                he = read_image(str(self.he_dir / filename), mode=ImageReadMode.RGB)
                ihc = read_image(str(self.ihc_dir / filename), mode=ImageReadMode.RGB)
                self.preloaded_data.append((he, ihc)) # type: ignore

    def __len__(self):
        return len(self.paired_files)

    def _synchronized_crop(self, he_tensor, ihc_tensor):
        h, w = he_tensor.shape[-2:]
        if w >= self.resolution and h >= self.resolution and self.augment:
            top = random.randint(0, h - self.resolution)
            left = random.randint(0, w - self.resolution)
        else:
            top = max(0, (h - self.resolution) // 2)
            left = max(0, (w - self.resolution) // 2)
        he_tensor = TF.crop(he_tensor, top, left, self.resolution, self.resolution)
        ihc_tensor = TF.crop(ihc_tensor, top, left, self.resolution, self.resolution)
        return he_tensor, ihc_tensor

    def _synchronized_augment(self, he_tensor, ihc_tensor):
        if random.random() > 0.5:
            he_tensor = TF.hflip(he_tensor)
            ihc_tensor = TF.hflip(ihc_tensor)
        if random.random() > 0.5:
            he_tensor = TF.vflip(he_tensor)
            ihc_tensor = TF.vflip(ihc_tensor)
        rotation = random.choice([0, 90, 180, 270])
        if rotation > 0:
            he_tensor = TF.rotate(he_tensor, rotation)
            ihc_tensor = TF.rotate(ihc_tensor, rotation)
        return he_tensor, ihc_tensor

    def __getitem__(self, idx):
        if self.preload_to_ram and self.preloaded_data is not None:
            he_tensor, ihc_tensor = self.preloaded_data[idx]
            he_tensor = he_tensor.float() / 255.0
            ihc_tensor = ihc_tensor.float() / 255.0
            filename = self.paired_files[idx]
        else:
            filename = self.paired_files[idx]
            # Load directly into torch tensors to bypass slow PIL Image.open
            he_tensor = read_image(str(self.he_dir / filename), mode=ImageReadMode.RGB).float() / 255.0
            ihc_tensor = read_image(str(self.ihc_dir / filename), mode=ImageReadMode.RGB).float() / 255.0

        # Resize if needed (torchvision transforms are much faster than PIL)
        if he_tensor.shape[1:] != (self.source_resolution, self.source_resolution):
            he_tensor = TF.resize(he_tensor, (self.source_resolution, self.source_resolution), antialias=True)
            ihc_tensor = TF.resize(ihc_tensor, (self.source_resolution, self.source_resolution), antialias=True)

        # Resize 512→256 (instead of cropping)
        if self.resolution < self.source_resolution:
            he_tensor = TF.resize(he_tensor, (self.resolution, self.resolution), antialias=True)
            ihc_tensor = TF.resize(ihc_tensor, (self.resolution, self.resolution), antialias=True)

        # Synchronized geometric augmentation
        if self.augment:
            he_tensor, ihc_tensor = self._synchronized_augment(he_tensor, ihc_tensor)

        # Restoration residual (StarDiff key innovation)
        residual_tensor = ihc_tensor - he_tensor  # in [-1, 1]

        # Normalize to [-1, 1]
        he_tensor = self.normalize(he_tensor)
        ihc_tensor = self.normalize(ihc_tensor)

        return {
            "he": he_tensor,
            "ihc": ihc_tensor,
            "residual": residual_tensor,
            "filename": filename,
        }


def create_dataloaders(config):
    """Create train and validation dataloaders from config."""
    dataset_paths = get_dataset_paths(config.dataset_root, config.stains)

    # Training datasets
    train_datasets = []
    for stain in config.stains:
        he_path = dataset_paths[stain]["train_he"]
        ihc_path = dataset_paths[stain]["train_ihc"]
        if he_path.exists() and len(list(he_path.glob("*.png"))) > 0:
            ds = StarDiffPairedDataset(
                he_path, ihc_path,
                resolution=config.image_size,
                source_resolution=config.source_image_size,
                augment=True,
                preload_to_ram=config.preload_to_ram,
            )
            train_datasets.append(ds)
        else:
            print(f"⚠️ {stain}: No data found at {he_path}")

    if not train_datasets:
        raise ValueError("No training data found! Check download step.")

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

    # Validation dataset (first stain)
    val_stain = config.stains[0]
    val_dataset = StarDiffPairedDataset(
        dataset_paths[val_stain]["val_he"],
        dataset_paths[val_stain]["val_ihc"],
        resolution=config.image_size,
        source_resolution=config.source_image_size,
        augment=False,
        preload_to_ram=config.preload_to_ram,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    print(f"✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"✓ Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    return train_loader, val_loader

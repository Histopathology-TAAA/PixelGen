"""
Paired H&E -> IHC dataset for DABPixelGen training.

Each sample returns:
  - h_he_density:   H PSPStain density from H&E deconvolution     [1, H, W]
  - dab_gt:         Raw PSPStain DAB density from IHC              [1, H, W]
                    (same scale as model output; used for recomposition)
  - dab_gt_fod:     FOD-transformed DAB from IHC                   [1, H, W]
                    (expression-level signal; used for DAB prediction loss)

The two DAB representations are complementary:
  dab_gt      → Beer-Lambert recomposition needs raw stain density
  dab_gt_fod  → FOD emphasises positive nuclei/cells, suppresses background;
                used by DABPredictionLoss for perceptually meaningful supervision

Stain deconvolution uses the PSPStain pipeline (round-trip + FOD) for
accurate brown-stain separation.  See stain_utils.PSPStainDABExtractor.

Synchronized geometric augmentation (flips + 90° rotations) is applied before
deconvolution so the density maps stay aligned with the cropped image pair.

Dataset layout (same Kaggle slugs as stardiff_pixelgen):
  dataset_root/<STAIN>/train/he/   dataset_root/<STAIN>/train/ihc/
  dataset_root/<STAIN>/val/he/     dataset_root/<STAIN>/val/ihc/

Or original MIST 1024×1024 layout (TrainValAB):
  dataset_root/<STAIN>/TrainValAB/trainA/   (H&E)
  dataset_root/<STAIN>/TrainValAB/trainB/   (IHC)
"""
import os
import sys
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

PIXELGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PIXELGEN_ROOT not in sys.path:
    sys.path.insert(0, PIXELGEN_ROOT)

from dab_pixelgen.stain_utils import StainDeconvolution, PSPStainDABExtractor


# ── Kaggle dataset slugs (mirrors stardiff_pixelgen/dataset.py) ──────────────

STAIN_TO_SLUG = {
    "KI67": "mist-preprocessed-512-ki67",
    "ER":   "mist-preprocessed-512-er",
    "PR":   "mist-preprocessed-512-pr",
}
MIST_1024_SLUG = "mist-er-pr-ki67"


# ── Path helpers ──────────────────────────────────────────────────────────────

def download_dataset(dataset_root: str, stains):
    """Download preprocessed 512×512 paired patches from Kaggle."""
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    for stain in stains:
        stain_path = root / stain
        if not stain_path.exists():
            slug = STAIN_TO_SLUG[stain]
            print(f"Downloading preprocessed {stain} dataset...")
            os.system(f"kaggle datasets download -d mohamedtarek26/{slug} -p {root} --unzip")
        else:
            print(f"  {stain} already exists at {stain_path}")
    return root


def _find_trainvalab(root: Path, stain: str):
    """Locate TrainValAB folder, handling Kaggle nested extraction."""
    flat = root / stain / "TrainValAB"
    if flat.exists() and (flat / "trainA").exists():
        return flat
    stain_dir = root / stain
    if stain_dir.is_dir():
        for child in sorted(stain_dir.iterdir()):
            if child.is_dir() and child.name != "TrainValAB":
                candidate = child / "TrainValAB"
                if candidate.exists() and (candidate / "trainA").exists():
                    return candidate
    return None


def get_dataset_paths(dataset_root, stains):
    """Return {stain: {train_he, train_ihc, val_he, val_ihc}} for preprocessed data."""
    root = Path(dataset_root)
    return {
        stain: {
            "train_he":  root / stain / "train" / "he",
            "train_ihc": root / stain / "train" / "ihc",
            "val_he":    root / stain / "val"   / "he",
            "val_ihc":   root / stain / "val"   / "ihc",
        }
        for stain in stains
    }


def get_dataset_paths_1024(dataset_root, stains):
    """Return paths for original MIST 1024×1024 data."""
    root = Path(dataset_root)
    paths = {}
    for stain in stains:
        base = _find_trainvalab(root, stain)
        if base is None:
            raise ValueError(
                f"Cannot find TrainValAB for stain '{stain}' under {root / stain}."
            )
        paths[stain] = {
            "train_he":  base / "trainA",
            "train_ihc": base / "trainB",
            "val_he":    base / "valA",
            "val_ihc":   base / "valB",
        }
    return paths


# ── Dataset ───────────────────────────────────────────────────────────────────

class DABPairedDataset(Dataset):
    """
    Paired H&E / IHC dataset with stain-density outputs.

    Each sample returns:
        he:            [3, H, W]  H&E image in [-1, 1]
        ihc:           [3, H, W]  IHC image in [-1, 1]
        ihc_01:        [3, H, W]  IHC image in [0, 1]   (for recomposition loss)
        h_he_density:  [1, H, W]  H OD from H&E deconvolution  (>= 0)
        dab_gt:        [1, H, W]  DAB OD from IHC deconvolution (>= 0)
        filename:      str
    """

    def __init__(
        self,
        he_dir:           str,
        ihc_dir:          str,
        resolution:       int = 256,
        source_resolution:int = 512,
        augment:          bool = True,
        preload_to_ram:   bool = False,
    ):
        self.he_dir  = Path(he_dir)
        self.ihc_dir = Path(ihc_dir)
        self.resolution        = resolution
        self.source_resolution = source_resolution
        self.augment  = augment

        # Paired filename matching
        he_files  = set(f.name for f in self.he_dir.glob("*.png"))
        he_files.update(f.name for f in self.he_dir.glob("*.jpg"))
        ihc_files = set(f.name for f in self.ihc_dir.glob("*.png"))
        ihc_files.update(f.name for f in self.ihc_dir.glob("*.jpg"))
        self.paired_files = sorted(list(he_files & ihc_files))
        print(f"  {len(self.paired_files)} paired files from {he_dir}")

        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Stain deconvolution (CPU, float32)
        # PSPStainDABExtractor: accurate round-trip DAB separation + FOD
        # StainDeconvolution:   used only for H_HE (hematoxylin from H&E)
        self._deconv    = StainDeconvolution()
        self._dab_extractor = PSPStainDABExtractor()

        # Preload raw uint8 images into RAM (optional)
        self.preload_to_ram = preload_to_ram
        self.preloaded: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if preload_to_ram:
            print(f"  Preloading {len(self.paired_files)} images into RAM...")
            self.preloaded = []
            for fn in tqdm(self.paired_files, desc="Preloading"):
                he  = read_image(str(self.he_dir  / fn), mode=ImageReadMode.RGB)
                ihc = read_image(str(self.ihc_dir / fn), mode=ImageReadMode.RGB)
                self.preloaded.append((he, ihc))

    def __len__(self):
        return len(self.paired_files)

    # ── Augmentation helpers ──────────────────────────────────────────────────

    def _resize_pair(self, he, ihc):
        if he.shape[1:] != (self.source_resolution, self.source_resolution):
            he  = TF.resize(he,  (self.source_resolution, self.source_resolution), antialias=True)
            ihc = TF.resize(ihc, (self.source_resolution, self.source_resolution), antialias=True)
        if self.resolution < self.source_resolution:
            he  = TF.resize(he,  (self.resolution, self.resolution), antialias=True)
            ihc = TF.resize(ihc, (self.resolution, self.resolution), antialias=True)
        return he, ihc

    def _augment(self, he, ihc):
        if random.random() > 0.5:
            he, ihc = TF.hflip(he), TF.hflip(ihc)
        if random.random() > 0.5:
            he, ihc = TF.vflip(he), TF.vflip(ihc)
        angle = random.choice([0, 90, 180, 270])
        if angle:
            he  = TF.rotate(he,  angle)
            ihc = TF.rotate(ihc, angle)
        return he, ihc

    # ── Stain deconvolution ───────────────────────────────────────────────────

    def _deconvolve_pair(self, he_01: torch.Tensor, ihc_01: torch.Tensor):
        """
        Stain deconvolution on [3, H, W] float32 in [0, 1] tensors.

        Returns:
            h_he_density: [1, H, W]  H density from H&E (PSPStain scale)
            dab_gt:       [1, H, W]  raw DAB density from IHC (for recomposition)
            dab_gt_fod:   [1, H, W]  FOD-transformed DAB (for expression loss)
        """
        he_b  = he_01.unsqueeze(0)   # [1, 3, H, W]
        ihc_b = ihc_01.unsqueeze(0)  # [1, 3, H, W]

        # H from H&E: standard PSPStain deconv, take H channel (index 0)
        h_he = self._deconv(he_b)["hematoxylin"].squeeze(0)          # [1, H, W]

        # DAB from IHC: raw density (for Beer-Lambert recomposition)
        dab_gt = self._dab_extractor.extract_raw_dab(ihc_b).squeeze(0)  # [1, H, W]

        # DAB from IHC: FOD-transformed (round-trip + FOD, for expression supervision)
        dab_gt_fod = self._dab_extractor(ihc_b).squeeze(0)              # [1, H, W]

        return h_he, dab_gt, dab_gt_fod

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        filename = self.paired_files[idx]

        if self.preloaded is not None:
            he_raw, ihc_raw = self.preloaded[idx]
            he_f  = he_raw.float()  / 255.0
            ihc_f = ihc_raw.float() / 255.0
        else:
            he_f  = read_image(str(self.he_dir  / filename), mode=ImageReadMode.RGB).float() / 255.0
            ihc_f = read_image(str(self.ihc_dir / filename), mode=ImageReadMode.RGB).float() / 255.0

        he_f, ihc_f = self._resize_pair(he_f, ihc_f)

        if self.augment:
            he_f, ihc_f = self._augment(he_f, ihc_f)

        # Stain deconvolution BEFORE normalization (needs [0, 1] input)
        h_he_density, dab_gt, dab_gt_fod = self._deconvolve_pair(he_f, ihc_f)

        # Keep IHC in [0, 1] for recomposition loss
        ihc_01 = ihc_f.clone()

        # Normalize to [-1, 1] for model input
        he_11  = self.normalize(he_f)
        ihc_11 = self.normalize(ihc_f)

        return {
            "he":           he_11,           # [3, H, W] in [-1, 1]
            "ihc":          ihc_11,          # [3, H, W] in [-1, 1]
            "ihc_01":       ihc_01,          # [3, H, W] in [0, 1]
            "h_he_density": h_he_density,    # [1, H, W] PSPStain H density
            "dab_gt":       dab_gt,          # [1, H, W] raw DAB density (recomposition)
            "dab_gt_fod":   dab_gt_fod,      # [1, H, W] FOD DAB (expression supervision)
            "filename":     filename,
        }


# ── DataLoader factories ──────────────────────────────────────────────────────

def create_dataloaders(config):
    """Build train/val DataLoaders from preprocessed 512 dataset."""
    paths = get_dataset_paths(config.dataset_root, config.stains)

    train_datasets = []
    for stain in config.stains:
        he_path  = paths[stain]["train_he"]
        ihc_path = paths[stain]["train_ihc"]
        if he_path.exists() and any(he_path.glob("*.png")):
            train_datasets.append(DABPairedDataset(
                he_path, ihc_path,
                resolution=config.image_size,
                source_resolution=config.source_image_size,
                augment=True,
                preload_to_ram=config.preload_to_ram,
            ))
        else:
            print(f"  WARNING: No data for {stain} at {he_path}")

    if not train_datasets:
        raise ValueError("No training data found. Run download_dataset() first.")

    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else ConcatDataset(train_datasets)
    )

    val_stain = config.stains[0]
    val_dataset = DABPairedDataset(
        paths[val_stain]["val_he"],
        paths[val_stain]["val_ihc"],
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
        prefetch_factor=config.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    return train_loader, val_loader


def create_dataloaders_1024(config):
    """Build train/val DataLoaders from original MIST 1024 dataset."""
    paths = get_dataset_paths_1024(config.dataset_root, config.stains)

    train_datasets = []
    for stain in config.stains:
        he_path  = paths[stain]["train_he"]
        ihc_path = paths[stain]["train_ihc"]
        if he_path.exists() and (
            any(he_path.glob("*.png")) or any(he_path.glob("*.jpg"))
        ):
            train_datasets.append(DABPairedDataset(
                he_path, ihc_path,
                resolution=config.image_size,
                source_resolution=config.source_image_size,
                augment=True,
                preload_to_ram=config.preload_to_ram,
            ))
        else:
            print(f"  WARNING: No 1024 data for {stain} at {he_path}")

    if not train_datasets:
        raise ValueError("No 1024 training data found.")

    train_dataset = (
        train_datasets[0]
        if len(train_datasets) == 1
        else ConcatDataset(train_datasets)
    )

    val_stain = config.stains[0]
    val_dataset = DABPairedDataset(
        paths[val_stain]["val_he"],
        paths[val_stain]["val_ihc"],
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
        prefetch_factor=config.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, config.batch_size // 2),
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )

    print(f"Train (1024): {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val   (1024): {len(val_dataset)} samples, {len(val_loader)} batches")
    return train_loader, val_loader

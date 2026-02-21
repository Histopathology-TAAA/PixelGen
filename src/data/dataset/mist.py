import os
import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize


def save_fn(image, metadata, root_path):
    filename = metadata.get('filename', 'sample')
    image_path = os.path.join(root_path, f"{filename}.png")
    Image.fromarray(image).save(image_path)


class MISTTrainDataset(Dataset):
    """
    Paired H&E (domain A) -> IHC (domain B) dataset for MIST virtual staining.
    Images are 1024x1024. We take random 512x512 crops and resize to target resolution.
    
    Directory structure:
        root/
            trainA/  (H&E images)
            trainB/  (IHC images)
    
    Files in trainA and trainB must have matching filenames.
    """
    def __init__(self, root, resolution=256, crop_size=512, random_flip=True, augment_he=False):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.augment_he = augment_he

        # H&E-only augmentations (applied AFTER geometric transforms, BEFORE tensor conversion)
        # IHC target is NOT augmented to preserve DAB stain signal for DAB loss.
        if augment_he:
            self.color_jitter = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02
            )
            self.gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

        self.dir_A = os.path.join(root, 'trainA')
        self.dir_B = os.path.join(root, 'trainB')

        # Get sorted file lists and ensure pairing
        self.filenames = sorted([
            f for f in os.listdir(self.dir_A)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        ])
        # Verify all A files have matching B files
        for fname in self.filenames:
            assert os.path.exists(os.path.join(self.dir_B, fname)), \
                f"Missing paired file in trainB: {fname}"

        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load paired images
        img_A = Image.open(os.path.join(self.dir_A, fname)).convert('RGB')  # H&E
        img_B = Image.open(os.path.join(self.dir_B, fname)).convert('RGB')  # IHC

        # Random crop (same location for both)
        w, h = img_A.size
        if w >= self.crop_size and h >= self.crop_size:
            i = random.randint(0, h - self.crop_size)
            j = random.randint(0, w - self.crop_size)
            img_A = TF.crop(img_A, i, j, self.crop_size, self.crop_size)
            img_B = TF.crop(img_B, i, j, self.crop_size, self.crop_size)
        else:
            # If image is smaller than crop_size, resize to crop_size
            img_A = TF.resize(img_A, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BICUBIC)
            img_B = TF.resize(img_B, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BICUBIC)

        # Resize to target resolution
        img_A = TF.resize(img_A, (self.resolution, self.resolution), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        img_B = TF.resize(img_B, (self.resolution, self.resolution), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # Random horizontal flip (same for both)
        if self.random_flip and random.random() > 0.5:
            img_A = TF.hflip(img_A)
            img_B = TF.hflip(img_B)

        # Random vertical flip (same for both)
        if self.random_flip and random.random() > 0.5:
            img_A = TF.vflip(img_A)
            img_B = TF.vflip(img_B)

        # Random 90-degree rotation (histopathology patches have no canonical orientation)
        if self.random_flip:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img_A = TF.rotate(img_A, angle)
                img_B = TF.rotate(img_B, angle)

        # H&E-only color augmentations (after geometry, before tensor conversion)
        # Not applied to IHC target to preserve DAB stain signal for DAB loss
        if self.augment_he:
            img_A = self.color_jitter(img_A)
            if random.random() < 0.3:
                img_A = self.gaussian_blur(img_A)

        # Convert to tensor [0, 1]
        raw_A = to_tensor(img_A)  # H&E raw [0,1]
        raw_B = to_tensor(img_B)  # IHC raw [0,1]

        # Normalize to [-1, 1]
        norm_A = self.normalize(raw_A)  # H&E normalized
        norm_B = self.normalize(raw_B)  # IHC normalized

        metadata = {
            'raw_image': raw_B,           # IHC raw [0,1] for DINO/LPIPS
            'condition_image': norm_A,     # H&E normalized [-1,1] for conditioning
            'condition_image_raw': raw_A,  # H&E raw [0,1] for DINO
            'filename': os.path.splitext(fname)[0],
        }

        # x = IHC target (normalized), y = dummy label
        return norm_B, 0, metadata


class MISTValDataset(Dataset):
    """
    Validation paired H&E -> IHC dataset for MIST virtual staining.
    Uses center crop for deterministic evaluation.
    Returns noise as x (for diffusion sampling) and condition/GT in metadata.
    
    Directory structure:
        root/
            valA/  (H&E images)
            valB/  (IHC images)
    """
    def __init__(self, root, resolution=256, crop_size=512, noise_scale=1.0, seed=42):
        super().__init__()
        self.root = root
        self.resolution = resolution
        self.crop_size = crop_size
        self.noise_scale = noise_scale
        self.seed = seed

        self.dir_A = os.path.join(root, 'valA')
        self.dir_B = os.path.join(root, 'valB')

        self.filenames = sorted([
            f for f in os.listdir(self.dir_A)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        ])
        for fname in self.filenames:
            assert os.path.exists(os.path.join(self.dir_B, fname)), \
                f"Missing paired file in valB: {fname}"

        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # Load paired images
        img_A = Image.open(os.path.join(self.dir_A, fname)).convert('RGB')  # H&E
        img_B = Image.open(os.path.join(self.dir_B, fname)).convert('RGB')  # IHC

        # Center crop
        w, h = img_A.size
        if w >= self.crop_size and h >= self.crop_size:
            i = (h - self.crop_size) // 2
            j = (w - self.crop_size) // 2
            img_A = TF.crop(img_A, i, j, self.crop_size, self.crop_size)
            img_B = TF.crop(img_B, i, j, self.crop_size, self.crop_size)
        else:
            img_A = TF.resize(img_A, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BICUBIC)
            img_B = TF.resize(img_B, (self.crop_size, self.crop_size), interpolation=TF.InterpolationMode.BICUBIC)

        # Resize to target resolution
        img_A = TF.resize(img_A, (self.resolution, self.resolution), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        img_B = TF.resize(img_B, (self.resolution, self.resolution), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)

        # Convert to tensor [0, 1]
        raw_A = to_tensor(img_A)
        raw_B = to_tensor(img_B)

        # Normalize to [-1, 1]
        norm_A = self.normalize(raw_A)
        norm_B = self.normalize(raw_B)

        # Generate deterministic noise per sample
        generator = torch.Generator().manual_seed(self.seed + idx)
        noise = self.noise_scale * torch.randn(3, self.resolution, self.resolution, generator=generator)

        metadata = {
            'condition_image': norm_A,         # H&E normalized [-1,1]
            'condition_image_raw': raw_A,      # H&E raw [0,1]
            'gt_image': norm_B,                # IHC normalized [-1,1]
            'gt_image_raw': raw_B,             # IHC raw [0,1]
            'filename': os.path.splitext(fname)[0],
            'save_fn': save_fn,
        }

        # x = noise (starting point for sampling), y = dummy label
        return noise, 0, metadata


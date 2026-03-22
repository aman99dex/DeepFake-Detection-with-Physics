"""
Dataset classes for deepfake detection.

Supports:
- FaceForensics++ (FF++)
- CelebDF-v2
- DFDC
- Custom/synthetic datasets
"""

import os
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as T


class DeepfakeDataset(Dataset):
    """Unified dataset for deepfake detection across multiple benchmarks.

    Expected directory structure for each dataset:
        dataset_root/
            real/
                img_0001.png
                img_0002.png
                ...
            fake/
                img_0001.png
                ...
            (or)
            metadata.json  (with paths and labels)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        max_samples: int = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")

        # Build sample list
        self.samples = self._load_samples(max_samples)

        # Transforms
        self.transform = self._build_transforms()

    def _load_samples(self, max_samples: Optional[int]) -> list:
        """Load image paths and labels."""
        samples = []

        # Check for metadata.json
        meta_path = self.root_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            for entry in metadata:
                p = self.root_dir / entry["path"]
                if p.exists():
                    samples.append((str(p.resolve()), entry["label"]))
                else:
                    samples.append((str(p), entry["label"]))
        else:
            # Default structure: real/ and fake/ subdirectories
            real_dir = self.root_dir / "real"
            fake_dir = self.root_dir / "fake"

            if real_dir.exists():
                for img_path in sorted(real_dir.glob("*.*")):
                    if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        samples.append((str(img_path.resolve()), 0))

            if fake_dir.exists():
                for img_path in sorted(fake_dir.glob("*.*")):
                    if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                        samples.append((str(img_path.resolve()), 1))

        # Split
        random.seed(42)
        random.shuffle(samples)
        n = len(samples)
        if self.split == "train":
            samples = samples[:int(0.8 * n)]
        elif self.split == "val":
            samples = samples[int(0.8 * n):int(0.9 * n)]
        else:
            samples = samples[int(0.9 * n):]

        if max_samples:
            samples = samples[:max_samples]

        return samples

    def _build_transforms(self):
        transforms = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if self.augment:
            transforms = [
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        return T.Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Return a random valid sample on error
            return self.__getitem__(random.randint(0, len(self) - 1))

        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "path": img_path,
        }


class SyntheticPBRDataset(Dataset):
    """Synthetic dataset with known ground-truth PBR properties.

    Used for validating the physics scoring module.
    Generates faces with controlled BRDF + lighting,
    then adds "fake" artifacts to test detection.
    """

    def __init__(self, num_samples: int = 1000, image_size: int = 256):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Generate a synthetic sample with known physics properties."""
        is_fake = idx % 2 == 0  # Alternate real/fake

        # Generate random but physically plausible BRDF for "real"
        albedo = torch.rand(3) * 0.6 + 0.2  # Skin-like albedo [0.2, 0.8]
        roughness = torch.rand(1) * 0.5 + 0.3  # Skin roughness [0.3, 0.8]
        metallic = torch.rand(1) * 0.05  # Skin is dielectric [0, 0.05]

        if is_fake:
            # Inject physics violations:
            # 1. Impossible albedo (too bright)
            albedo = albedo * 1.5
            albedo = torch.clamp(albedo, 0, 1)
            # 2. Inconsistent metallic (skin shouldn't be metallic)
            metallic = torch.rand(1) * 0.5 + 0.3
            # 3. Energy conservation violation
            roughness = torch.rand(1) * 0.1  # Unnaturally smooth for skin

        # Render a simple face-like image using the BRDF
        image = self._render_sphere(albedo, roughness, metallic)

        return {
            "image": image,
            "label": torch.tensor(1 if is_fake else 0, dtype=torch.long),
            "gt_albedo": albedo,
            "gt_roughness": roughness,
            "gt_metallic": metallic,
            "path": f"synthetic_{idx}",
        }

    def _render_sphere(
        self, albedo: torch.Tensor, roughness: torch.Tensor, metallic: torch.Tensor
    ) -> torch.Tensor:
        """Render a sphere with given BRDF (used as proxy for face)."""
        size = self.image_size

        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing="ij",
        )
        r2 = x ** 2 + y ** 2
        mask = r2 < 1.0

        # Normal from sphere
        z = torch.sqrt(torch.clamp(1.0 - r2, 0, 1))
        normal = torch.stack([x, y, z], dim=-1)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-8)

        # Simple directional light
        light_dir = torch.tensor([0.3, 0.5, 0.8])
        light_dir = light_dir / light_dir.norm()

        n_dot_l = torch.clamp((normal * light_dir).sum(dim=-1), 0, 1)

        # Lambertian diffuse
        diffuse = albedo.view(1, 1, 3) * n_dot_l.unsqueeze(-1) * (1.0 - metallic)

        # Simple specular
        view_dir = torch.tensor([0.0, 0.0, 1.0])
        h = (light_dir + view_dir)
        h = h / h.norm()
        n_dot_h = torch.clamp((normal * h).sum(dim=-1), 0, 1)
        spec = n_dot_h ** (2.0 / (roughness + 0.01))
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic
        specular = f0.view(1, 1, 3) * spec.unsqueeze(-1)

        image = (diffuse + specular) * mask.unsqueeze(-1).float()
        image = torch.clamp(image, 0, 1)
        image = image.permute(2, 0, 1)  # (3, H, W)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image


def get_dataloaders(
    data_config: dict,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict:
    """Create train/val/test dataloaders.

    Args:
        data_config: dict with dataset paths and settings
        batch_size: batch size
        num_workers: number of data loading workers

    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    image_size = data_config.get("image_size", 256)

    for split in ["train", "val", "test"]:
        datasets = []
        for ds_config in data_config.get("datasets", []):
            ds_path = ds_config["path"]
            if os.path.exists(ds_path):
                ds = DeepfakeDataset(
                    root_dir=ds_path,
                    split=split,
                    image_size=image_size,
                    augment=(split == "train"),
                )
                if len(ds) > 0:
                    datasets.append(ds)

        # Fallback to synthetic if no real data available
        if not datasets:
            ds = SyntheticPBRDataset(
                num_samples=1000 if split == "train" else 200,
                image_size=image_size,
            )
            datasets.append(ds)

        combined = torch.utils.data.ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        loaders[split] = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    return loaders

"""
Dataset Download and Preparation Scripts.

Downloads publicly available deepfake detection datasets:
1. FaceForensics++ (requires Google Form approval)
2. CelebDF-v2 (GitHub)
3. DFDC (Kaggle)
4. WildDeepfake (GitHub)

Also creates synthetic PBR test data.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import urllib.request


BASE_DIR = Path(__file__).parent.parent / "data" / "raw"
BASE_DIR.mkdir(parents=True, exist_ok=True)


def download_celeb_df_v2():
    """Download CelebDF-v2 dataset.

    Official: https://github.com/yuezunli/celeb-deepfakeforensics
    The dataset itself requires requesting access.
    This sets up the directory and provides instructions.
    """
    output_dir = BASE_DIR / "celeb-df-v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    readme = """# CelebDF-v2 Dataset

## How to Download
1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics
2. Fill out the Google Form linked in the README
3. You will receive download links via email
4. Download and extract to this directory

## Expected Structure
celeb-df-v2/
    Celeb-real/        # 590 real celebrity videos
    Celeb-synthesis/   # 5639 synthesized deepfake videos
    YouTube-real/      # 300 additional real videos from YouTube
    List_of_testing_videos.txt

## Citation
@inproceedings{li2020celeb,
    title={Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
    author={Li, Yuezun and Yang, Xin and Sun, Pu and Qi, Honggang and Lyu, Siwei},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020}
}
"""
    with open(output_dir / "DOWNLOAD_INSTRUCTIONS.md", "w") as f:
        f.write(readme)

    # Create directory structure
    (output_dir / "real").mkdir(exist_ok=True)
    (output_dir / "fake").mkdir(exist_ok=True)

    print(f"[CelebDF-v2] Directory created at {output_dir}")
    print("[CelebDF-v2] Please follow instructions in DOWNLOAD_INSTRUCTIONS.md")


def download_faceforensics():
    """Set up FaceForensics++ dataset download.

    Official: https://github.com/ondyari/FaceForensics
    Requires filling a Google Form for download access.
    """
    output_dir = BASE_DIR / "ff++"
    output_dir.mkdir(parents=True, exist_ok=True)

    readme = """# FaceForensics++ Dataset

## How to Download
1. Visit: https://github.com/ondyari/FaceForensics
2. Fill out the Google Form: https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform
3. Once approved, you receive a download script
4. Run: python download_script.py <output_path> -d all -c c23 -t videos

## Expected Structure
ff++/
    original_sequences/
        youtube/           # 1000 original videos
    manipulated_sequences/
        Deepfakes/        # 1000 DeepFake manipulated videos
        Face2Face/        # 1000 Face2Face reenacted videos
        FaceSwap/         # 1000 FaceSwap videos
        NeuralTextures/   # 1000 NeuralTextures videos

## Compression Levels
- c0: Raw (uncompressed)
- c23: Light compression (recommended)
- c40: Heavy compression

## Citation
@inproceedings{roessler2019faceforensicspp,
    author = {Andreas Rossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Niessner},
    title = {FaceForensics++: Learning to Detect Manipulated Facial Images},
    booktitle= {International Conference on Computer Vision (ICCV)},
    year = {2019}
}
"""
    with open(output_dir / "DOWNLOAD_INSTRUCTIONS.md", "w") as f:
        f.write(readme)

    (output_dir / "real").mkdir(exist_ok=True)
    (output_dir / "fake").mkdir(exist_ok=True)

    print(f"[FF++] Directory created at {output_dir}")
    print("[FF++] Please follow instructions in DOWNLOAD_INSTRUCTIONS.md")


def download_dfdc():
    """Set up DFDC dataset download.

    Official: Facebook DeepFake Detection Challenge
    Available on Kaggle.
    """
    output_dir = BASE_DIR / "dfdc"
    output_dir.mkdir(parents=True, exist_ok=True)

    readme = """# DFDC (DeepFake Detection Challenge) Dataset

## How to Download
### Option 1: Kaggle CLI
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials (~/.kaggle/kaggle.json)
3. Run: kaggle competitions download -c deepfake-detection-challenge
4. Extract to this directory

### Option 2: Manual Download
1. Visit: https://www.kaggle.com/c/deepfake-detection-challenge/data
2. Download the dataset files
3. Extract to this directory

## Expected Structure
dfdc/
    train_sample_videos/   # Sample training videos
    test_videos/           # Test videos
    metadata.json          # Labels and metadata

## Dataset Size
- Full dataset: ~470GB
- Sample: ~10GB

## Citation
@article{dolhansky2020deepfake,
    title={The deepfake detection challenge (dfdc) dataset},
    author={Dolhansky, Brian and Bitton, Joanna and Pflaum, Ben and Lu, Jikuo and Howes, Russ and Wang, Menglin and Ferrer, Cristian Canton},
    journal={arXiv preprint arXiv:2006.07397},
    year={2020}
}
"""
    with open(output_dir / "DOWNLOAD_INSTRUCTIONS.md", "w") as f:
        f.write(readme)

    (output_dir / "real").mkdir(exist_ok=True)
    (output_dir / "fake").mkdir(exist_ok=True)

    print(f"[DFDC] Directory created at {output_dir}")
    print("[DFDC] Please follow instructions in DOWNLOAD_INSTRUCTIONS.md")


def create_synthetic_dataset():
    """Create synthetic PBR test dataset using our SyntheticPBRDataset."""
    output_dir = BASE_DIR / "synthetic_pbr"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Synthetic] Creating synthetic PBR dataset...")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.deepfake_dataset import SyntheticPBRDataset

    import torch
    from PIL import Image
    import numpy as np

    dataset = SyntheticPBRDataset(num_samples=500, image_size=256)

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)

    metadata = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"]
        label = sample["label"].item()

        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (image * std + mean).clamp(0, 1)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        if label == 0:
            path = real_dir / f"real_{i:04d}.png"
        else:
            path = fake_dir / f"fake_{i:04d}.png"

        pil_img.save(str(path))
        metadata.append({
            "path": str(path.relative_to(output_dir)),
            "label": label,
            "gt_albedo": sample["gt_albedo"].tolist(),
            "gt_roughness": sample["gt_roughness"].tolist(),
            "gt_metallic": sample["gt_metallic"].tolist(),
        })

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{len(dataset)} samples")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Synthetic] Created {len(dataset)} samples at {output_dir}")
    print(f"  Real: {sum(1 for m in metadata if m['label'] == 0)}")
    print(f"  Fake: {sum(1 for m in metadata if m['label'] == 1)}")


def main():
    print("=" * 60)
    print("PhysForensics Dataset Preparation")
    print("=" * 60)

    print("\n1. Setting up FaceForensics++...")
    download_faceforensics()

    print("\n2. Setting up CelebDF-v2...")
    download_celeb_df_v2()

    print("\n3. Setting up DFDC...")
    download_dfdc()

    print("\n4. Creating synthetic PBR dataset...")
    create_synthetic_dataset()

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("\nFor real datasets, follow the DOWNLOAD_INSTRUCTIONS.md in each folder.")
    print("Synthetic data is ready for immediate use.")
    print(f"\nAll data at: {BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Download real deepfake detection datasets for training and evaluation.

Downloads:
1. OpenRL/DeepFakeFace from HuggingFace (real + diffusion-generated faces)
2. pujanpaudel/deepfake_face_classification from HuggingFace (balanced real/fake)

These are freely available and don't require special access forms.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_deepfake_face_classification():
    """Download pujanpaudel/deepfake_face_classification from HuggingFace.

    ~32K images, balanced real/fake, pre-split into train/val/test.
    """
    from huggingface_hub import snapshot_download

    output_dir = Path("data/raw/hf_deepfake_classify")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[HF Classification] Downloading pujanpaudel/deepfake_face_classification...")
    print("  This dataset has ~32K balanced real/fake face images")

    try:
        path = snapshot_download(
            repo_id="pujanpaudel/deepfake_face_classification",
            repo_type="dataset",
            local_dir=str(output_dir),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print(f"  Downloaded to: {path}")

        # Count files
        image_count = 0
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_count += len(list(output_dir.rglob(ext)))
        print(f"  Total images found: {image_count}")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_openrl_deepfakeface():
    """Download OpenRL/DeepFakeFace from HuggingFace.

    30K real images from IMDB-WIKI + 90K fake images from diffusion models.
    """
    from huggingface_hub import snapshot_download

    output_dir = Path("data/raw/hf_openrl_deepfake")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[HF OpenRL] Downloading OpenRL/DeepFakeFace...")
    print("  This has 30K real + 90K diffusion-generated faces")

    try:
        path = snapshot_download(
            repo_id="OpenRL/DeepFakeFace",
            repo_type="dataset",
            local_dir=str(output_dir),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print(f"  Downloaded to: {path}")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def prepare_unified_dataset():
    """Prepare a unified dataset structure from all downloaded sources.

    Creates:
        data/processed/unified/
            real/   -> all real face images
            fake/   -> all fake face images
            metadata.json -> labels and source info
    """
    print("\n[Prepare] Creating unified dataset...")

    unified_dir = Path("data/processed/unified")
    real_dir = unified_dir / "real"
    fake_dir = unified_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    real_count = 0
    fake_count = 0

    # Source 1: HF Classification dataset
    hf_class_dir = Path("data/raw/hf_deepfake_classify")
    if hf_class_dir.exists():
        print("  Processing hf_deepfake_classify...")
        for split_dir in hf_class_dir.rglob("*"):
            if split_dir.is_dir():
                dirname = split_dir.name.lower()
                if "real" in dirname or "authentic" in dirname:
                    for img in split_dir.glob("*.*"):
                        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                            dst = real_dir / f"hfclass_real_{real_count:06d}{img.suffix}"
                            try:
                                shutil.copy2(str(img), str(dst))
                                metadata.append({"path": str(dst.relative_to(unified_dir)), "label": 0, "source": "hf_classify"})
                                real_count += 1
                            except Exception:
                                pass
                elif "fake" in dirname or "deepfake" in dirname or "synthetic" in dirname:
                    for img in split_dir.glob("*.*"):
                        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                            dst = fake_dir / f"hfclass_fake_{fake_count:06d}{img.suffix}"
                            try:
                                shutil.copy2(str(img), str(dst))
                                metadata.append({"path": str(dst.relative_to(unified_dir)), "label": 1, "source": "hf_classify"})
                                fake_count += 1
                            except Exception:
                                pass

    # Source 2: OpenRL DeepFakeFace
    openrl_dir = Path("data/raw/hf_openrl_deepfake")
    if openrl_dir.exists():
        print("  Processing hf_openrl_deepfake...")
        for item in openrl_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                parent_name = item.parent.name.lower()
                # Detect if real or fake based on directory name
                if "real" in parent_name or "original" in parent_name or "imdb" in parent_name:
                    dst = real_dir / f"openrl_real_{real_count:06d}{item.suffix}"
                    try:
                        shutil.copy2(str(item), str(dst))
                        metadata.append({"path": str(dst.relative_to(unified_dir)), "label": 0, "source": "openrl"})
                        real_count += 1
                    except Exception:
                        pass
                else:
                    dst = fake_dir / f"openrl_fake_{fake_count:06d}{item.suffix}"
                    try:
                        shutil.copy2(str(item), str(dst))
                        metadata.append({"path": str(dst.relative_to(unified_dir)), "label": 1, "source": "openrl"})
                        fake_count += 1
                    except Exception:
                        pass

    # Also check for any images directly in subdirectories with number-based naming
    for src_dir in [hf_class_dir, openrl_dir]:
        if not src_dir.exists():
            continue
        for split in ["train", "test", "val", "validation"]:
            split_path = src_dir / split
            if not split_path.exists():
                continue
            for label_dir in split_path.iterdir():
                if not label_dir.is_dir():
                    continue
                label_name = label_dir.name.lower()
                is_real = label_name in {"real", "0", "authentic", "original"}
                is_fake = label_name in {"fake", "1", "deepfake", "synthetic", "generated"}
                if not is_real and not is_fake:
                    # Try numeric: 0=fake, 1=real (or vice versa depending on dataset)
                    if label_name == "0":
                        is_fake = True
                    elif label_name == "1":
                        is_real = True
                    else:
                        continue

                for img in label_dir.glob("*.*"):
                    if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                        continue
                    if is_real:
                        dst = real_dir / f"split_real_{real_count:06d}{img.suffix}"
                        try:
                            shutil.copy2(str(img), str(dst))
                            metadata.append({"path": str(dst.relative_to(unified_dir)), "label": 0, "source": src_dir.name})
                            real_count += 1
                        except Exception:
                            pass
                    elif is_fake:
                        dst = fake_dir / f"split_fake_{fake_count:06d}{img.suffix}"
                        try:
                            shutil.copy2(str(img), str(dst))
                            metadata.append({"path": str(dst.relative_to(unified_dir)), "label": 1, "source": src_dir.name})
                            fake_count += 1
                        except Exception:
                            pass

    # Save metadata
    with open(unified_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"\n  Unified dataset ready:")
    print(f"    Real images: {real_count}")
    print(f"    Fake images: {fake_count}")
    print(f"    Total: {real_count + fake_count}")
    print(f"    Location: {unified_dir}")

    return real_count + fake_count > 0


def main():
    print("=" * 60)
    print("PhysForensics - Real Dataset Download")
    print("=" * 60)

    os.chdir(Path(__file__).parent.parent)

    success_any = False

    print("\n--- Dataset 1: HuggingFace Classification Dataset ---")
    if download_deepfake_face_classification():
        success_any = True

    print("\n--- Dataset 2: OpenRL DeepFakeFace ---")
    if download_openrl_deepfakeface():
        success_any = True

    if success_any:
        print("\n--- Preparing Unified Dataset ---")
        prepare_unified_dataset()

    print("\n" + "=" * 60)
    if success_any:
        print("Download complete! Ready for training.")
        print("Run: python train.py --epochs 20 --batch_size 8")
    else:
        print("No datasets downloaded. Check your internet connection.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
PhysForensics - Comprehensive Data Preparation Script.

This script handles ALL dataset preparation in one place:

FREE DATASETS (auto-download, no sign-up needed):
  - HuggingFace: pujanpaudel/deepfake_face_classification (~32K images)
  - HuggingFace: OpenRL/DeepFakeFace (~120K images, real + diffusion fakes)
  - HuggingFace: hannesdelbeke/deepfake-detection-dataset
  - HuggingFace: dhaiwat10/deepfake-face-detection

ACADEMIC DATASETS (manual download required):
  - FaceForensics++ — request access at:
      https://github.com/ondyari/FaceForensics
      Place extracted frames at: data/raw/ff++/
        data/raw/ff++/real/  (original_sequences frames)
        data/raw/ff++/fake/  (any manipulation type frames)

  - CelebDF-v2 — download at:
      https://github.com/yuezunli/celeb-deepfakeforensics
      Place at: data/raw/celeb-df/
        data/raw/celeb-df/real/
        data/raw/celeb-df/fake/

  - DFDC (DeepFake Detection Challenge) — download at:
      https://www.kaggle.com/competitions/deepfake-detection-challenge/data
      (requires Kaggle account)
      Place at: data/raw/dfdc/
        data/raw/dfdc/real/
        data/raw/dfdc/fake/

Usage:
    python scripts/prepare_data.py                    # Download free datasets only
    python scripts/prepare_data.py --include-manual   # Also process manually placed datasets
    python scripts/prepare_data.py --max-per-source 5000  # Limit per source
    python scripts/prepare_data.py --check           # Just check what's available
"""

import os
import sys
import json
import shutil
import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare PhysForensics datasets")
    parser.add_argument("--include-manual", action="store_true",
                        help="Also process manually downloaded datasets (FF++, CelebDF, DFDC)")
    parser.add_argument("--max-per-source", type=int, default=None,
                        help="Maximum images to take from each source (for quick testing)")
    parser.add_argument("--check", action="store_true",
                        help="Only check what data is available, don't download")
    parser.add_argument("--output-dir", type=str, default="data/processed/unified",
                        help="Output directory for unified dataset")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─── Free HuggingFace Dataset Downloaders ─────────────────────────────────────

HF_DATASETS = [
    {
        "repo_id": "pujanpaudel/deepfake_face_classification",
        "name": "hf_classify",
        "description": "~32K balanced real/fake face classification dataset",
        "real_dirs": ["real", "Real", "authentic", "0"],
        "fake_dirs": ["fake", "Fake", "deepfake", "1"],
    },
    {
        "repo_id": "OpenRL/DeepFakeFace",
        "name": "hf_openrl",
        "description": "30K real (IMDB-WIKI) + 90K diffusion-generated faces",
        "real_dirs": ["real", "Real", "original", "imdb", "wiki"],
        "fake_dirs": ["fake", "Fake", "deepfake", "diffusion", "generated"],
    },
]


def download_hf_dataset(repo_id: str, local_dir: str) -> bool:
    """Download a dataset from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        print(f"  Downloading {repo_id}...")
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            ignore_patterns=["*.md", "*.txt", ".gitattributes", "*.zip"],
        )
        print(f"  Downloaded to: {path}")
        return True
    except ImportError:
        print("  ERROR: huggingface_hub not installed. Run: pip install huggingface_hub datasets")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"  Try manually: pip install huggingface_hub && huggingface-cli download {repo_id} --repo-type dataset --local-dir {local_dir}")
        return False


def download_free_datasets(data_root: Path) -> list:
    """Download all free HuggingFace datasets."""
    downloaded = []
    for ds in HF_DATASETS:
        local_dir = data_root / "raw" / ds["name"]
        if local_dir.exists() and any(local_dir.rglob("*.jpg")) or any(local_dir.rglob("*.png") if local_dir.exists() else []):
            print(f"  {ds['name']}: already downloaded, skipping")
            downloaded.append((ds, local_dir))
            continue

        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {ds['description']} ---")
        if download_hf_dataset(ds["repo_id"], str(local_dir)):
            downloaded.append((ds, local_dir))

    return downloaded


# ─── Image Collection ──────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images_from_dir(source_dir: Path, label: int, source_name: str,
                             max_count: int = None) -> list:
    """Recursively collect all image files from a directory with a given label."""
    images = []
    for p in source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append({"src": str(p), "label": label, "source": source_name})
            if max_count and len(images) >= max_count:
                break
    return images


def guess_label_from_dirname(dirname: str) -> int:
    """Guess real=0 / fake=1 from directory name."""
    d = dirname.lower()
    if any(k in d for k in ["real", "authentic", "original", "genuine"]):
        return 0
    if any(k in d for k in ["fake", "deepfake", "generated", "synthetic", "manipulated",
                              "diffusion", "gan", "swap", "f2f", "face2face", "neural"]):
        return 1
    return -1  # unknown


def collect_from_hf_source(ds_info: dict, local_dir: Path, max_per_source: int) -> tuple:
    """Collect real and fake images from a downloaded HuggingFace dataset."""
    real_images = []
    fake_images = []

    for p in local_dir.rglob("*"):
        if not p.is_dir():
            continue
        label = guess_label_from_dirname(p.name)
        if label == 0:
            imgs = collect_images_from_dir(p, 0, ds_info["name"],
                                           max_per_source)
            real_images.extend(imgs)
        elif label == 1:
            imgs = collect_images_from_dir(p, 1, ds_info["name"],
                                           max_per_source)
            fake_images.extend(imgs)

    # Deduplicate by path
    seen = set()
    real_images = [x for x in real_images if not (x["src"] in seen or seen.add(x["src"]))]
    seen = set()
    fake_images = [x for x in fake_images if not (x["src"] in seen or seen.add(x["src"]))]

    if max_per_source:
        real_images = real_images[:max_per_source]
        fake_images = fake_images[:max_per_source]

    return real_images, fake_images


def collect_from_manual_dataset(dataset_dir: Path, name: str,
                                 max_per_source: int) -> tuple:
    """Collect from manually placed dataset (expects real/ and fake/ subdirs)."""
    real_dir = dataset_dir / "real"
    fake_dir = dataset_dir / "fake"

    if not real_dir.exists() or not fake_dir.exists():
        return [], []

    real_images = collect_images_from_dir(real_dir, 0, name, max_per_source)
    fake_images = collect_images_from_dir(fake_dir, 1, name, max_per_source)

    print(f"  {name}: {len(real_images)} real, {len(fake_images)} fake")
    return real_images, fake_images


# ─── Unified Dataset Builder ──────────────────────────────────────────────────

def copy_image(args):
    """Copy a single image to destination. Returns (success, src, dst)."""
    src, dst = args
    try:
        shutil.copy2(src, dst)
        return True, src, str(dst)
    except Exception:
        return False, src, None


def build_unified_dataset(
    all_real: list,
    all_fake: list,
    output_dir: Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Build the final unified dataset with train/val/test splits.

    Structure:
        output_dir/
            real/
                img_000001.jpg
                ...
            fake/
                img_000001.jpg
                ...
            metadata.json    (full sample list with labels, sources, splits)
            splits.json      (just paths grouped by split for easy access)
    """
    random.seed(seed)

    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding unified dataset:")
    print(f"  Input: {len(all_real)} real, {len(all_fake)} fake")

    # Balance classes (optional but recommended)
    min_count = min(len(all_real), len(all_fake))
    if len(all_real) > min_count * 2:
        print(f"  Balancing: capping real at {min_count * 2}")
        random.shuffle(all_real)
        all_real = all_real[:min_count * 2]
    if len(all_fake) > min_count * 2:
        print(f"  Balancing: capping fake at {min_count * 2}")
        random.shuffle(all_fake)
        all_fake = all_fake[:min_count * 2]

    # Shuffle
    random.shuffle(all_real)
    random.shuffle(all_fake)

    print(f"  After balancing: {len(all_real)} real, {len(all_fake)} fake")

    # Copy images in parallel
    copy_tasks = []
    metadata = []
    idx_real = 0
    idx_fake = 0

    for item in all_real:
        ext = Path(item["src"]).suffix.lower() or ".jpg"
        dst = real_dir / f"real_{idx_real:07d}{ext}"
        copy_tasks.append((item["src"], dst))
        metadata.append({
            "path": str(dst.relative_to(output_dir)),
            "label": 0,
            "source": item["source"],
            "original_path": item["src"],
        })
        idx_real += 1

    for item in all_fake:
        ext = Path(item["src"]).suffix.lower() or ".jpg"
        dst = fake_dir / f"fake_{idx_fake:07d}{ext}"
        copy_tasks.append((item["src"], dst))
        metadata.append({
            "path": str(dst.relative_to(output_dir)),
            "label": 1,
            "source": item["source"],
            "original_path": item["src"],
        })
        idx_fake += 1

    print(f"  Copying {len(copy_tasks)} images...")
    success_count = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for success, src, dst in pool.map(copy_image, copy_tasks):
            if success:
                success_count += 1

    print(f"  Successfully copied: {success_count}/{len(copy_tasks)}")

    # Create train/val/test splits (stratified)
    random.shuffle(metadata)
    n = len(metadata)
    n_test = max(int(n * test_ratio), 1)
    n_val = max(int(n * val_ratio), 1)
    n_train = n - n_val - n_test

    splits = {
        "train": [m["path"] for m in metadata[:n_train]],
        "val": [m["path"] for m in metadata[n_train:n_train + n_val]],
        "test": [m["path"] for m in metadata[n_train + n_val:]],
    }

    for m in metadata[:n_train]:
        m["split"] = "train"
    for m in metadata[n_train:n_train + n_val]:
        m["split"] = "val"
    for m in metadata[n_train + n_val:]:
        m["split"] = "test"

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "splits.json", "w") as f:
        json.dump({
            "train_count": n_train,
            "val_count": n_val,
            "test_count": n_test,
            "real_count": idx_real,
            "fake_count": idx_fake,
            "splits": splits,
        }, f, indent=2)

    # Summary by source
    source_summary = {}
    for m in metadata:
        s = m["source"]
        if s not in source_summary:
            source_summary[s] = {"real": 0, "fake": 0}
        key = "real" if m["label"] == 0 else "fake"
        source_summary[s][key] += 1

    with open(output_dir / "source_summary.json", "w") as f:
        json.dump(source_summary, f, indent=2)

    print(f"\n  Dataset ready at: {output_dir}")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"  Sources: {list(source_summary.keys())}")

    return {
        "total": len(metadata),
        "real": idx_real,
        "fake": idx_fake,
        "train": n_train,
        "val": n_val,
        "test": n_test,
        "sources": source_summary,
    }


# ─── Availability Check ───────────────────────────────────────────────────────

def check_available_data(data_root: Path) -> dict:
    """Check what data is already available."""
    available = {}

    # Free datasets
    for ds in HF_DATASETS:
        local_dir = data_root / "raw" / ds["name"]
        if local_dir.exists():
            count = sum(1 for _ in local_dir.rglob("*") if _.suffix.lower() in IMAGE_EXTS)
            available[ds["name"]] = count
        else:
            available[ds["name"]] = 0

    # Manual datasets
    for name in ["ff++", "celeb-df", "dfdc"]:
        d = data_root / "raw" / name
        if d.exists():
            real_count = sum(1 for _ in (d / "real").rglob("*") if _.suffix.lower() in IMAGE_EXTS) if (d / "real").exists() else 0
            fake_count = sum(1 for _ in (d / "fake").rglob("*") if _.suffix.lower() in IMAGE_EXTS) if (d / "fake").exists() else 0
            available[name] = {"real": real_count, "fake": fake_count}
        else:
            available[name] = None

    # Unified dataset
    unified = data_root / "processed" / "unified"
    if unified.exists() and (unified / "metadata.json").exists():
        with open(unified / "metadata.json") as f:
            meta = json.load(f)
        available["unified"] = len(meta)
    else:
        available["unified"] = 0

    return available


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_root = Path("data")
    output_dir = Path(args.output_dir)

    print("=" * 65)
    print("PhysForensics - Data Preparation")
    print("=" * 65)

    if args.check:
        print("\nChecking available data...")
        available = check_available_data(data_root)
        print("\n--- Data Availability ---")
        for name, count in available.items():
            if count is None:
                print(f"  {name:20s}: not found (manual download required)")
            elif isinstance(count, dict):
                print(f"  {name:20s}: {count['real']} real, {count['fake']} fake")
            else:
                print(f"  {name:20s}: {count} images")

        print("\n--- How to get more data ---")
        print("\nFree (auto-download):")
        print("  python scripts/prepare_data.py")
        print("\nFF++ (~1000 videos):")
        print("  1. Fill form: https://github.com/ondyari/FaceForensics")
        print("  2. Extract frames to: data/raw/ff++/real/ and data/raw/ff++/fake/")
        print("\nCelebDF-v2:")
        print("  1. Download: https://github.com/yuezunli/celeb-deepfakeforensics")
        print("  2. Extract to: data/raw/celeb-df/real/ and data/raw/celeb-df/fake/")
        print("\nDFDC (100K+ videos, most challenging):")
        print("  1. Kaggle: https://kaggle.com/competitions/deepfake-detection-challenge")
        print("  2. Extract to: data/raw/dfdc/real/ and data/raw/dfdc/fake/")
        return

    all_real = []
    all_fake = []

    # ── Step 1: Download free datasets ──────────────────────────────────────
    print("\n--- Step 1: Free HuggingFace Datasets ---")
    downloaded = download_free_datasets(data_root)

    for ds_info, local_dir in downloaded:
        real_imgs, fake_imgs = collect_from_hf_source(ds_info, local_dir, args.max_per_source)
        print(f"  {ds_info['name']}: {len(real_imgs)} real, {len(fake_imgs)} fake")
        all_real.extend(real_imgs)
        all_fake.extend(fake_imgs)

    # ── Step 2: Process manual datasets if available ─────────────────────────
    if args.include_manual:
        print("\n--- Step 2: Manual Datasets ---")
        manual_datasets = {
            "ff++": data_root / "raw" / "ff++",
            "celeb-df": data_root / "raw" / "celeb-df",
            "dfdc": data_root / "raw" / "dfdc",
        }
        for name, path in manual_datasets.items():
            if path.exists():
                real_imgs, fake_imgs = collect_from_manual_dataset(path, name, args.max_per_source)
                all_real.extend(real_imgs)
                all_fake.extend(fake_imgs)
            else:
                print(f"  {name}: not found at {path}")
                print(f"         See instructions above for download links")
    else:
        print("\n--- Step 2: Manual Datasets (skipped — use --include-manual) ---")
        for name in ["ff++", "celeb-df", "dfdc"]:
            p = data_root / "raw" / name
            if p.exists():
                r = sum(1 for _ in (p / "real").rglob("*") if _.suffix.lower() in IMAGE_EXTS) if (p / "real").exists() else 0
                f = sum(1 for _ in (p / "fake").rglob("*") if _.suffix.lower() in IMAGE_EXTS) if (p / "fake").exists() else 0
                if r + f > 0:
                    print(f"  {name}: found {r} real, {f} fake — run with --include-manual to use")

    # ── Step 3: Build unified dataset ────────────────────────────────────────
    print("\n--- Step 3: Building Unified Dataset ---")
    if not all_real and not all_fake:
        print("ERROR: No images found. Check your download or data directories.")
        print("Tip: Run 'python scripts/prepare_data.py --check' to see what's available.")
        return

    stats = build_unified_dataset(
        all_real, all_fake, output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ── Step 4: Print final summary and next steps ────────────────────────────
    print("\n" + "=" * 65)
    print("DATA PREPARATION COMPLETE")
    print("=" * 65)
    print(f"\nTotal images: {stats['total']}")
    print(f"  Real:  {stats['real']}")
    print(f"  Fake:  {stats['fake']}")
    print(f"\nSplits:")
    print(f"  Train: {stats['train']}")
    print(f"  Val:   {stats['val']}")
    print(f"  Test:  {stats['test']}")
    print(f"\nSource breakdown:")
    for src, counts in stats["sources"].items():
        print(f"  {src:20s}: {counts['real']} real, {counts['fake']} fake")
    print(f"\nNext step — start training:")
    print(f"  python train_v2.py --data {output_dir}")
    print(f"\nFor best results, also add academic datasets and retrain:")
    print(f"  python scripts/prepare_data.py --include-manual")
    print(f"  python train_v2.py --data {output_dir} --epochs 50")
    print("=" * 65)


if __name__ == "__main__":
    main()

"""
Cross-Dataset Generalization Evaluation for PhysForensics.

This is the MOST IMPORTANT evaluation — it tests whether the model can detect
deepfakes it has NEVER seen before. Standard deepfake detectors memorize
generator-specific artifacts and collapse on new generators. PhysForensics
should generalize because physics laws don't change.

Protocol:
    Train on one dataset (e.g., FF++) → Test on unseen datasets (CelebDF, DFDC)
    No fine-tuning between datasets.

Usage:
    # Evaluate trained checkpoint on all available test datasets
    python scripts/cross_dataset_eval.py --checkpoint outputs/real_data/checkpoints/best_model.pt

    # Evaluate v2 model
    python scripts/cross_dataset_eval.py --checkpoint path/to/model.pt --model v2

    # Compare multiple checkpoints
    python scripts/cross_dataset_eval.py --checkpoints model_v1.pt model_v2.pt --names V1 V2

    # Use specific datasets
    python scripts/cross_dataset_eval.py --datasets ff++ celeb-df dfdc
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.deepfake_dataset import DeepfakeDataset
from src.evaluation.evaluator import Evaluator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset generalization evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to single model checkpoint")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="List of checkpoints for comparison")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Names for each checkpoint (used in output table)")
    parser.add_argument("--model", type=str, default="v1", choices=["v1", "v2", "v2_extended"],
                        help="Model architecture version")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory containing all datasets")
    parser.add_argument("--datasets", nargs="+",
                        default=["unified", "ff++", "celeb-df", "dfdc"],
                        help="Datasets to evaluate on")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="outputs/cross_dataset")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_model(checkpoint_path, model_version, device, image_size=256):
    """Load a PhysForensics model from checkpoint."""
    if model_version == "v2" or model_version == "v2_extended":
        if model_version == "v2_extended":
            from src.models.physforensics_v2_extended import PhysForensicsV2Extended as ModelClass
        else:
            from src.models.physforensics_v2 import PhysForensicsV2 as ModelClass
        model = ModelClass(
            image_size=image_size,
            num_sample_points=512,
            nerf_hidden=128,
            nerf_layers=4,
            gtr_samples=32,
        )
    else:
        from src.models.physforensics import PhysForensics as ModelClass
        model = ModelClass(
            image_size=image_size,
            num_sample_points=512,
            nerf_hidden=128,
            nerf_layers=4,
        )

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded: {checkpoint_path}")
        if "best_auc" in ckpt:
            print(f"  Best Val AUC: {ckpt['best_auc']:.4f}")
    else:
        print(f"  WARNING: No checkpoint at {checkpoint_path}, using random weights")

    return model.to(device)


def find_dataset_dir(data_root: Path, name: str) -> Path:
    """Locate dataset directory for a given name."""
    candidates = [
        data_root / "raw" / name,
        data_root / "processed" / name,
        data_root / name,
        Path(name),
    ]
    for c in candidates:
        if c.exists():
            real_exists = (c / "real").exists()
            fake_exists = (c / "fake").exists()
            meta_exists = (c / "metadata.json").exists()
            if real_exists and fake_exists or meta_exists:
                return c
    return None


def count_dataset_images(dataset_dir: Path) -> tuple:
    """Count real and fake images in a dataset directory."""
    real = sum(1 for _ in (dataset_dir / "real").rglob("*")
               if _.suffix.lower() in IMAGE_EXTS) if (dataset_dir / "real").exists() else 0
    fake = sum(1 for _ in (dataset_dir / "fake").rglob("*")
               if _.suffix.lower() in IMAGE_EXTS) if (dataset_dir / "fake").exists() else 0
    return real, fake


@torch.no_grad()
def evaluate_on_dataset(
    model,
    dataset_dir: Path,
    dataset_name: str,
    batch_size: int,
    image_size: int,
    device: torch.device,
    model_version: str,
    num_workers: int = 0,
) -> dict:
    """Evaluate a model on a single dataset."""
    try:
        test_ds = DeepfakeDataset(
            str(dataset_dir),
            split="test",
            image_size=image_size,
            augment=False,
        )
        if len(test_ds) == 0:
            # Fall back to using all samples
            test_ds = DeepfakeDataset(
                str(dataset_dir),
                split="train",
                image_size=image_size,
                augment=False,
            )

        loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model.eval()
        all_preds = []
        all_labels = []
        all_physics = []

        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            if model_version in ("v2", "v2_extended"):
                output = model(images)
            else:
                output = model(images)

            probs = output["probability"].squeeze(-1).cpu().numpy()
            physics = output["physics_scores"].cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            all_physics.extend(physics)

        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        physics_scores = np.array(all_physics)

        if len(np.unique(labels)) < 2:
            return {"error": "Need both real and fake samples for evaluation"}

        evaluator = Evaluator()
        metrics = evaluator.compute_metrics(predictions, labels)

        real_mask = labels == 0
        fake_mask = labels == 1

        physics_analysis = {}
        if real_mask.any() and physics_scores.ndim > 1:
            physics_analysis = {
                "real_mean": physics_scores[real_mask].mean(axis=0).tolist(),
                "fake_mean": physics_scores[fake_mask].mean(axis=0).tolist() if fake_mask.any() else [],
                "score_names": ["ECS", "SCS", "BSS", "ICS", "TPS", "SSS", "FAS", "WAS",
                                "ShCS", "SpSS"][:physics_scores.shape[1]],
            }

        return {
            "dataset": dataset_name,
            "n_samples": len(predictions),
            "n_real": int(real_mask.sum()),
            "n_fake": int(fake_mask.sum()),
            "auc": float(metrics["auc"]),
            "eer": float(metrics["eer"]),
            "accuracy": float(metrics["accuracy"]),
            "f1": float(metrics["f1"]),
            "average_precision": float(metrics["average_precision"]),
            "tpr_at_fpr5": float(np.interp(0.05, metrics["fpr_curve"], metrics["tpr_curve"])),
            "tpr_at_fpr10": float(np.interp(0.10, metrics["fpr_curve"], metrics["tpr_curve"])),
            "physics_analysis": physics_analysis,
        }

    except Exception as e:
        return {"dataset": dataset_name, "error": str(e)}


def print_results_table(all_results: dict):
    """Print cross-dataset results as a formatted table."""
    print("\n" + "=" * 100)
    print("CROSS-DATASET GENERALIZATION RESULTS")
    print("=" * 100)
    print(f"{'Model':<20} {'Dataset':<15} {'AUC':>8} {'EER':>8} {'Acc':>8} {'F1':>8} {'AP':>8} {'TPR@5%FPR':>12}")
    print("-" * 100)

    for model_name, datasets in all_results.items():
        for ds_name, metrics in datasets.items():
            if "error" in metrics:
                print(f"{model_name:<20} {ds_name:<15} ERROR: {metrics['error']}")
                continue
            print(
                f"{model_name:<20} {ds_name:<15} "
                f"{metrics['auc']:>8.4f} {metrics['eer']:>8.4f} "
                f"{metrics['accuracy']:>8.4f} {metrics['f1']:>8.4f} "
                f"{metrics['average_precision']:>8.4f} "
                f"{metrics.get('tpr_at_fpr5', 0):>12.4f}"
            )
        print("-" * 100)

    print("=" * 100)
    print("\nKey metric: AUC > 0.90 on unseen datasets = strong generalization")
    print("Compare with SOTA: XceptionNet drops from 0.997 (FF++) to ~0.65 (CelebDF)")


def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    print("=" * 65)
    print("PhysForensics - Cross-Dataset Evaluation")
    print("=" * 65)
    print(f"Device: {device}")

    # Gather checkpoints
    checkpoints = []
    names = []

    if args.checkpoints:
        checkpoints = args.checkpoints
        names = args.names if args.names else [f"Model_{i+1}" for i in range(len(checkpoints))]
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
        names = ["PhysForensics"]
    else:
        # Find latest checkpoint automatically
        default_paths = [
            "outputs/real_data/checkpoints/best_model.pt",
            "outputs/checkpoints/best_model.pt",
        ]
        for p in default_paths:
            if Path(p).exists():
                checkpoints = [p]
                names = ["PhysForensics"]
                print(f"Auto-detected checkpoint: {p}")
                break

    if not checkpoints:
        print("ERROR: No checkpoint found. Specify with --checkpoint path/to/model.pt")
        print("Or train first: python train_v2.py")
        return

    # Find available datasets
    print(f"\nLooking for datasets in {data_root}...")
    available_datasets = {}
    for ds_name in args.datasets:
        if ds_name == "unified":
            d = data_root / "processed" / "unified"
        else:
            d = find_dataset_dir(data_root, ds_name)

        if d and d.exists():
            real, fake = count_dataset_images(d)
            if real + fake > 0:
                available_datasets[ds_name] = d
                print(f"  {ds_name:15s}: {real} real, {fake} fake — {d}")
            else:
                print(f"  {ds_name:15s}: found directory but no images")
        else:
            print(f"  {ds_name:15s}: not found (skipping)")

    if not available_datasets:
        print("ERROR: No datasets found. Run python scripts/prepare_data.py first.")
        return

    # Evaluate each model on each dataset
    all_results = {}

    for ckpt_path, model_name in zip(checkpoints, names):
        print(f"\n{'='*65}")
        print(f"Evaluating: {model_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*65}")

        model = load_model(ckpt_path, args.model, device, args.image_size)
        model_results = {}

        for ds_name, ds_dir in available_datasets.items():
            print(f"\n  Evaluating on: {ds_name} ({ds_dir})")
            result = evaluate_on_dataset(
                model, ds_dir, ds_name,
                args.batch_size, args.image_size, device,
                args.model, args.num_workers,
            )
            model_results[ds_name] = result

            if "error" not in result:
                print(f"    AUC: {result['auc']:.4f} | EER: {result['eer']:.4f} | "
                      f"Acc: {result['accuracy']:.4f} | "
                      f"Samples: {result['n_real']} real + {result['n_fake']} fake")
            else:
                print(f"    ERROR: {result['error']}")

        all_results[model_name] = model_results

    # Print summary table
    print_results_table(all_results)

    # Save results
    # Convert numpy arrays for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import json

    clean_results = {}
    for model_name, datasets in all_results.items():
        clean_results[model_name] = {}
        for ds, metrics in datasets.items():
            clean_results[model_name][ds] = {
                k: convert(v) for k, v in metrics.items()
                if k not in ("physics_analysis",)
            }
            if "physics_analysis" in metrics:
                clean_results[model_name][ds]["physics_analysis"] = {
                    k: convert(v) for k, v in metrics["physics_analysis"].items()
                }

    results_path = output_dir / "cross_dataset_results.json"
    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Physics score analysis per dataset
    print("\n--- Physics Score Analysis by Dataset ---")
    score_names = ["ECS", "SCS", "BSS", "ICS", "TPS", "SSS", "FAS", "WAS", "ShCS", "SpSS"]
    for model_name, datasets in all_results.items():
        for ds_name, metrics in datasets.items():
            if "physics_analysis" in metrics and metrics["physics_analysis"]:
                pa = metrics["physics_analysis"]
                print(f"\n{model_name} on {ds_name}:")
                names_used = pa.get("score_names", score_names[:len(pa.get("real_mean", []))])
                if pa.get("real_mean") and pa.get("fake_mean"):
                    print(f"  {'Score':<8} {'Real Mean':>12} {'Fake Mean':>12} {'Separation':>12}")
                    print(f"  {'-'*44}")
                    for i, (rm, fm) in enumerate(zip(pa["real_mean"], pa["fake_mean"])):
                        name = names_used[i] if i < len(names_used) else f"S{i}"
                        sep = abs(fm - rm)
                        print(f"  {name:<8} {rm:>12.4f} {fm:>12.4f} {sep:>12.4f}")


if __name__ == "__main__":
    main()

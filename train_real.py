"""
Train PhysForensics on real deepfake datasets.

Usage:
    python train_real.py                          # Use unified dataset
    python train_real.py --data data/raw/...      # Specify dataset path
    python train_real.py --epochs 30 --lr 5e-5    # Custom hyperparams
    python train_real.py --model v2               # Use v2 model
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import numpy as np

from src.data.deepfake_dataset import DeepfakeDataset, get_dataloaders
from src.evaluation.evaluator import Evaluator
from src.utils.visualization import PhysicsVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhysForensics on real data")
    parser.add_argument("--data", type=str, default="data/processed/unified",
                        help="Path to dataset root (with real/ and fake/ subdirs)")
    parser.add_argument("--model", type=str, default="v1", choices=["v1", "v2"],
                        help="Model version")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per split (for quick testing)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/real_data")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def build_model(version, device, image_size=256):
    if version == "v2":
        from src.models.physforensics_v2 import PhysForensicsV2
        model = PhysForensicsV2(
            image_size=image_size,
            num_sample_points=512,
            nerf_hidden=128,
            nerf_layers=4,
            classifier_hidden=256,
            fusion_type="attention",
            dropout=0.3,
            gtr_samples=32,
        )
    else:
        from src.models.physforensics import PhysForensics
        model = PhysForensics(
            image_size=image_size,
            num_sample_points=512,
            nerf_hidden=128,
            nerf_layers=4,
            classifier_hidden=256,
            fusion_type="attention",
            dropout=0.3,
        )
    return model.to(device)


def build_criterion(version, device):
    if version == "v2":
        from src.losses.advanced_losses import PhysicsForensicsLossV2
        return PhysicsForensicsLossV2()
    else:
        from src.losses.physics_losses import PhysicsForensicsLoss
        return PhysicsForensicsLoss()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, model_version="v1"):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if model_version == "v2":
            output = model(images, is_real_for_calibration=(labels == 0))
        else:
            output = model(images)

        losses = criterion(output, labels)
        loss = losses["total"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARNING: NaN/Inf loss at batch {batch_idx}, skipping")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (output["probability"].squeeze(-1) > 0.5).long()
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        batch_count += 1

        if (batch_idx + 1) % 20 == 0:
            acc = total_correct / total_samples if total_samples > 0 else 0
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} Acc: {acc:.4f}"
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device, model_version="v1"):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_physics = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if model_version == "v2":
            output = model(images)
        else:
            output = model(images)

        losses = criterion(output, labels)
        if not torch.isnan(losses["total"]):
            total_loss += losses["total"].item() * images.size(0)

        all_preds.extend(output["probability"].squeeze(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_physics.extend(output["physics_scores"].cpu().numpy())

    avg_loss = total_loss / max(len(all_labels), 1)
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    physics = np.array(all_physics)

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5

    acc = ((preds > 0.5).astype(int) == labels).mean()

    return avg_loss, auc, acc, preds, labels, physics


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Create output dirs
    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    # Check dataset
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Run: python scripts/download_real_datasets.py")
        return

    real_count = len(list((data_path / "real").glob("*.*"))) if (data_path / "real").exists() else 0
    fake_count = len(list((data_path / "fake").glob("*.*"))) if (data_path / "fake").exists() else 0
    print(f"Dataset: {data_path}")
    print(f"  Real images: {real_count}")
    print(f"  Fake images: {fake_count}")
    print(f"  Total: {real_count + fake_count}")

    if real_count + fake_count == 0:
        print("ERROR: No images found!")
        return

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_ds = DeepfakeDataset(str(data_path), split="train", image_size=args.image_size,
                                augment=True, max_samples=args.max_samples)
    val_ds = DeepfakeDataset(str(data_path), split="val", image_size=args.image_size,
                              augment=False, max_samples=args.max_samples)
    test_ds = DeepfakeDataset(str(data_path), split="test", image_size=args.image_size,
                               augment=False, max_samples=args.max_samples)

    from torch.utils.data import DataLoader
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers),
    }

    print(f"  Train: {len(train_ds)} samples ({len(loaders['train'])} batches)")
    print(f"  Val:   {len(val_ds)} samples ({len(loaders['val'])} batches)")
    print(f"  Test:  {len(test_ds)} samples ({len(loaders['test'])} batches)")

    # Build model
    model = build_model(args.model, device, args.image_size)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: PhysForensics {args.model} ({params:,} parameters)")

    # Build loss
    criterion = build_criterion(args.model, device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # Resume
    start_epoch = 0
    best_auc = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    # Train
    training_log = []
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch + 1, args.model
        )
        val_loss, val_auc, val_acc, _, _, _ = validate(
            model, loaders["val"], criterion, device, args.model
        )
        scheduler.step()

        elapsed = time.time() - t0
        log = {
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_auc": val_auc, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "time": elapsed,
        }
        training_log.append(log)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.0f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_auc": best_auc,
            }, output_dir / "checkpoints" / "best_model.pt")
            print(f"  >> New best AUC: {best_auc:.4f}, saved!")

    # Save training log
    with open(output_dir / "logs" / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    # Load best model
    best_ckpt = output_dir / "checkpoints" / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model (AUC: {ckpt['best_auc']:.4f})")

    test_loss, test_auc, test_acc, test_preds, test_labels, test_physics = validate(
        model, loaders["test"], criterion, device, args.model
    )

    evaluator = Evaluator(output_dir=str(output_dir / "evaluation"))
    metrics = evaluator.compute_metrics(test_preds, test_labels)

    print(f"\nTest Results on REAL Data:")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  EER:      {metrics['eer']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  AP:       {metrics['average_precision']:.4f}")

    # Save results
    results = {
        "dataset": str(args.data),
        "model": args.model,
        "epochs": args.epochs,
        "best_val_auc": best_auc,
        "test_auc": metrics["auc"],
        "test_eer": metrics["eer"],
        "test_accuracy": metrics["accuracy"],
        "test_f1": metrics["f1"],
        "test_ap": metrics["average_precision"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }
    with open(output_dir / "logs" / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualizations
    viz = PhysicsVisualizer(output_dir=str(output_dir / "visualizations"))

    train_losses = [l["train_loss"] for l in training_log]
    val_losses = [l["val_loss"] for l in training_log]
    train_aucs = [l["train_acc"] for l in training_log]
    val_aucs = [l["val_auc"] for l in training_log]
    viz.plot_training_curves(train_losses, val_losses, train_aucs, val_aucs,
                            save_name="real_training_curves.png")
    viz.plot_roc_curve(metrics["fpr_curve"], metrics["tpr_curve"], metrics["auc"],
                       dataset_name="Real Deepfake Data", save_name="real_roc_curve.png")

    real_mask = test_labels == 0
    fake_mask = test_labels == 1
    if real_mask.any() and fake_mask.any():
        viz.plot_physics_scores(test_physics[real_mask], test_physics[fake_mask],
                                save_name="real_physics_distribution.png")

    print(f"\nAll results saved to {output_dir}")
    print("Training on real data complete!")


if __name__ == "__main__":
    main()

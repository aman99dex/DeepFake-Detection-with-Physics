"""
PhysForensics Training Script.

Usage:
    python train.py --config configs/default.yaml
    python train.py --synthetic  # Train on synthetic data for testing
"""

import argparse
import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.models.physforensics import PhysForensics
from src.losses.physics_losses import PhysicsForensicsLoss
from src.data.deepfake_dataset import get_dataloaders, SyntheticPBRDataset
from src.evaluation.evaluator import Evaluator
from src.utils.visualization import PhysicsVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhysForensics")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    loss_components = {}

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(images)

        # Compute loss
        losses = criterion(output, labels)
        loss = losses["total"]

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        preds = (output["probability"].squeeze(-1) > 0.5).long()
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        # Track loss components
        for key, val in losses.items():
            if key != "total":
                loss_components[key] = loss_components.get(key, 0) + val.item()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} Acc: {total_correct/total_samples:.4f}"
            )

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, loss_components


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        output = model(images)
        losses = criterion(output, labels)

        total_loss += losses["total"].item() * images.size(0)
        all_preds.extend(output["probability"].squeeze(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5

    acc = ((preds > 0.5).astype(int) == labels).mean()

    return avg_loss, auc, acc


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = PhysForensics(
        image_size=256,
        num_sample_points=1024,
        nerf_hidden=256,
        nerf_layers=8,
        neilf_hidden=128,
        neilf_layers=4,
        classifier_hidden=512,
        fusion_type="attention",
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: PhysForensics")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize loss
    criterion = PhysicsForensicsLoss(
        bce_weight=1.0,
        energy_conservation_weight=0.5,
        rendering_weight=0.3,
        smoothness_weight=0.2,
        anomaly_weight=0.3,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Data
    if args.synthetic:
        print("Using synthetic PBR dataset for training")
        from torch.utils.data import DataLoader
        train_ds = SyntheticPBRDataset(num_samples=2000, image_size=256)
        val_ds = SyntheticPBRDataset(num_samples=400, image_size=256)
        test_ds = SyntheticPBRDataset(num_samples=400, image_size=256)
        loaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0),
            "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
            "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0),
        }
    else:
        data_config = {
            "image_size": 256,
            "datasets": [
                {"name": "FF++", "path": "data/raw/ff++"},
                {"name": "CelebDF", "path": "data/raw/celeb-df-v2"},
            ],
        }
        loaders = get_dataloaders(data_config, batch_size=args.batch_size)

    # Resume from checkpoint
    start_epoch = 0
    best_auc = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_auc = checkpoint.get("best_auc", 0.0)
        print(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    # Training loop
    train_losses, val_losses = [], []
    train_aucs, val_aucs = [], []
    training_log = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, loss_comps = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_loss, val_auc, val_acc = validate(
            model, loaders["val"], criterion, device
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
        }
        training_log.append(log_entry)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_aucs.append(train_acc)  # Using accuracy as proxy
        val_aucs.append(val_auc)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
            }, output_dir / "checkpoints" / "best_model.pt")
            print(f"  >> New best AUC: {best_auc:.4f}, model saved!")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
            }, output_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt")

    # Save training log
    with open(output_dir / "logs" / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    evaluator = Evaluator(output_dir=str(output_dir / "evaluation"))
    test_metrics = evaluator.evaluate_model(model, loaders["test"], str(device))

    print(f"Test AUC:      {test_metrics['auc']:.4f}")
    print(f"Test EER:      {test_metrics['eer']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1:       {test_metrics['f1']:.4f}")
    print(f"Test AP:       {test_metrics['average_precision']:.4f}")

    # Visualizations
    visualizer = PhysicsVisualizer(output_dir=str(output_dir / "visualizations"))

    # Plot training curves
    visualizer.plot_training_curves(train_losses, val_losses, train_aucs, val_aucs)

    # Plot ROC
    visualizer.plot_roc_curve(
        test_metrics["fpr_curve"],
        test_metrics["tpr_curve"],
        test_metrics["auc"],
    )

    # Plot physics score distributions
    real_mask = test_metrics["raw_labels"] == 0
    fake_mask = test_metrics["raw_labels"] == 1
    if real_mask.any() and fake_mask.any():
        visualizer.plot_physics_scores(
            test_metrics["physics_scores"][real_mask],
            test_metrics["physics_scores"][fake_mask],
        )

    print(f"\nAll results saved to {output_dir}")
    print("Training complete!")


if __name__ == "__main__":
    main()

"""
PhysForensics v2 Extended Training Script.

Best-practice training pipeline for the 10-score physics model.

Features:
    - Mixed precision training (FP16 on CUDA for 2x speed)
    - Gradient accumulation for larger effective batch sizes
    - Cosine LR schedule with linear warmup
    - Checkpoint every N epochs + best model tracking
    - Per-epoch physics score monitoring (which signals are learning)
    - Multi-source dataset support (HF + FF++ + CelebDF + DFDC)
    - Automatic class balancing
    - TensorBoard logging

Usage:
    # Quickest start (uses any available data):
    python train_v2.py

    # With specific dataset:
    python train_v2.py --data data/processed/unified

    # Full training run (recommended):
    python train_v2.py --data data/processed/unified --epochs 50 --batch-size 16

    # Resume from checkpoint:
    python train_v2.py --resume outputs/v2_extended/checkpoints/last.pt

    # Quick test (5 epochs, small batches):
    python train_v2.py --epochs 5 --max-samples 500

    # After downloading FF++/CelebDF/DFDC:
    python scripts/prepare_data.py --include-manual
    python train_v2.py --data data/processed/unified --epochs 50
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.deepfake_dataset import DeepfakeDataset, SyntheticPBRDataset
from src.evaluation.evaluator import Evaluator
from src.utils.visualization import PhysicsVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="Train PhysForensics v2 Extended (10-score model)")

    # Data
    p.add_argument("--data", type=str, default="data/processed/unified",
                   help="Path to dataset (real/ + fake/ subdirs or metadata.json)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples per split — useful for quick testing")

    # Model
    p.add_argument("--model", type=str, default="v2_extended",
                   choices=["v1", "v2", "v2_extended"],
                   help="Model version (v2_extended = 10 physics scores, recommended)")
    p.add_argument("--nerf-hidden", type=int, default=128,
                   help="PBR-NeRF hidden dimension (256 for full quality, 128 for speed)")
    p.add_argument("--nerf-layers", type=int, default=4,
                   help="PBR-NeRF layers (8 for full quality, 4 for speed)")
    p.add_argument("--sample-points", type=int, default=512,
                   help="3D face surface sample points (1024 for full quality)")
    p.add_argument("--gtr-samples", type=int, default=32,
                   help="GTR importance sampling points (128 for full quality)")

    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-GPU batch size")
    p.add_argument("--grad-accum", type=int, default=2,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Linear LR warmup epochs")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mixed-precision", action="store_true", default=True,
                   help="Use FP16 mixed precision on CUDA (enabled by default)")
    p.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")

    # Checkpointing
    p.add_argument("--output-dir", type=str, default="outputs/v2_extended")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--num-workers", type=int, default=0)

    # Misc
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-fallback", action="store_true",
                   help="Use synthetic data if real data not found")
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, device):
    if args.model == "v2_extended":
        from src.models.physforensics_v2_extended import PhysForensicsV2Extended
        model = PhysForensicsV2Extended(
            image_size=args.image_size,
            num_sample_points=args.sample_points,
            nerf_hidden=args.nerf_hidden,
            nerf_layers=args.nerf_layers,
            classifier_hidden=256,
            fusion_type="attention",
            dropout=0.3,
            gtr_samples=args.gtr_samples,
        )
    elif args.model == "v2":
        from src.models.physforensics_v2 import PhysForensicsV2
        model = PhysForensicsV2(
            image_size=args.image_size,
            num_sample_points=args.sample_points,
            nerf_hidden=args.nerf_hidden,
            nerf_layers=args.nerf_layers,
            gtr_samples=args.gtr_samples,
        )
    else:
        from src.models.physforensics import PhysForensics
        model = PhysForensics(
            image_size=args.image_size,
            num_sample_points=args.sample_points,
            nerf_hidden=args.nerf_hidden,
            nerf_layers=args.nerf_layers,
        )
    return model.to(device)


def build_criterion(model_version):
    if model_version == "v2_extended":
        from src.losses.extended_losses import PhysicsForensicsLossExtended
        return PhysicsForensicsLossExtended()
    elif model_version == "v2":
        from src.losses.advanced_losses import PhysicsForensicsLossV2
        return PhysicsForensicsLossV2()
    else:
        from src.losses.physics_losses import PhysicsForensicsLoss
        return PhysicsForensicsLoss()


def get_lr_with_warmup(epoch, warmup_epochs, base_lr, min_lr, total_epochs):
    """Linear warmup followed by cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device,
    model_version, grad_accum, max_grad_norm, epoch
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_components = {}
    physics_scores_sum = None
    n_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        use_amp = scaler is not None and images.device.type == "cuda"

        with autocast(enabled=use_amp):
            if model_version == "v2_extended":
                output = model(images, is_real_for_calibration=(labels == 0))
            elif model_version == "v2":
                output = model(images, is_real_for_calibration=(labels == 0))
            else:
                output = model(images)

            losses = criterion(output, labels)
            loss = losses["total"] / grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  WARN: NaN/Inf at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        actual_loss = losses["total"].item()
        total_loss += actual_loss * images.size(0)
        preds = (output["probability"].squeeze(-1) > 0.5).long()
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        n_batches += 1

        # Accumulate loss components for logging
        for k, v in losses.items():
            if k != "total":
                loss_components[k] = loss_components.get(k, 0.0) + v.item()

        # Accumulate physics scores
        with torch.no_grad():
            ps = output["physics_scores"].detach().cpu()
            if physics_scores_sum is None:
                physics_scores_sum = ps.sum(dim=0)
            else:
                physics_scores_sum += ps.sum(dim=0)

        if (batch_idx + 1) % 20 == 0:
            acc = total_correct / max(total_samples, 1)
            print(f"  [{batch_idx+1}/{len(loader)}] loss={actual_loss:.4f} acc={acc:.4f}")

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    avg_physics = (physics_scores_sum / max(total_samples, 1)).numpy() \
                  if physics_scores_sum is not None else None

    return avg_loss, avg_acc, avg_components, avg_physics


@torch.no_grad()
def validate(model, loader, criterion, device, model_version):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_physics = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        output = model(images)
        losses = criterion(output, labels)

        if not torch.isnan(losses["total"]):
            total_loss += losses["total"].item() * images.size(0)

        all_preds.extend(output["probability"].squeeze(-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_physics.extend(output["physics_scores"].cpu().numpy())

    predictions = np.array(all_preds)
    labels_arr = np.array(all_labels)
    physics_arr = np.array(all_physics)

    avg_loss = total_loss / max(len(labels_arr), 1)

    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(labels_arr, predictions)
    except ValueError:
        auc = 0.5

    acc = ((predictions > 0.5).astype(int) == labels_arr).mean()
    return avg_loss, auc, acc, predictions, labels_arr, physics_arr


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    viz_dir = output_dir / "visualizations"
    for d in [ckpt_dir, log_dir, viz_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PhysForensics {args.model.upper()} — Training")
    print("=" * 70)
    print(f"Device:   {device}")
    print(f"Model:    {args.model}")
    print(f"Epochs:   {args.epochs}")
    print(f"Batch:    {args.batch_size} × {args.grad_accum} (accum) = {args.batch_size * args.grad_accum} effective")
    print(f"LR:       {args.lr}")
    use_amp = args.mixed_precision and device.type == "cuda"
    print(f"AMP:      {'enabled' if use_amp else 'disabled'}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        if args.synthetic_fallback:
            print(f"\nDataset not found at {data_path}")
            print("Using synthetic PBR data (for architecture testing only).")
            print("For real results, run: python scripts/prepare_data.py")
            train_ds = SyntheticPBRDataset(num_samples=2000, image_size=args.image_size)
            val_ds = SyntheticPBRDataset(num_samples=400, image_size=args.image_size)
            test_ds = SyntheticPBRDataset(num_samples=400, image_size=args.image_size)
        else:
            print(f"\nERROR: Dataset not found at {data_path}")
            print("Options:")
            print("  1. python scripts/prepare_data.py    (download free data)")
            print("  2. python train_v2.py --synthetic-fallback  (use synthetic data)")
            return
    else:
        real_count = len(list((data_path / "real").glob("*.*"))) \
                     if (data_path / "real").exists() else 0
        fake_count = len(list((data_path / "fake").glob("*.*"))) \
                     if (data_path / "fake").exists() else 0
        print(f"\nDataset: {data_path}")
        print(f"  Real: {real_count:,} | Fake: {fake_count:,} | Total: {real_count + fake_count:,}")

        train_ds = DeepfakeDataset(str(data_path), split="train",
                                    image_size=args.image_size, augment=True,
                                    max_samples=args.max_samples)
        val_ds = DeepfakeDataset(str(data_path), split="val",
                                  image_size=args.image_size, augment=False,
                                  max_samples=args.max_samples)
        test_ds = DeepfakeDataset(str(data_path), split="test",
                                   image_size=args.image_size, augment=False,
                                   max_samples=args.max_samples)

        if len(train_ds) == 0:
            print("ERROR: No training samples found.")
            print("Run: python scripts/prepare_data.py")
            return

    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda")),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers),
    }

    print(f"  Train: {len(train_ds):,} ({len(loaders['train'])} batches) | "
          f"Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ── Model, Loss, Optimizer ──────────────────────────────────────────────
    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model} ({n_params:,} parameters)")

    criterion = build_criterion(args.model)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = GradScaler() if use_amp else None

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auc = 0.0
    training_log = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_auc", 0.0)
        training_log = ckpt.get("training_log", [])
        print(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    # ── Training Loop ────────────────────────────────────────────────────
    score_names = getattr(model, "SCORE_NAMES",
                          ["ECS", "SCS", "BSS", "ICS", "TPS", "SSS", "FAS", "WAS"])
    print(f"\nPhysics scores: {', '.join(score_names)}")
    print(f"Starting training — {args.epochs} epochs\n{'='*70}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Update LR with warmup
        new_lr = get_lr_with_warmup(epoch, args.warmup_epochs, args.lr,
                                    args.lr * 0.01, args.epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        # Train
        train_loss, train_acc, loss_comps, train_physics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler,
            device, args.model, args.grad_accum, args.max_grad_norm, epoch
        )

        # Validate
        val_loss, val_auc, val_acc, _, _, val_physics = validate(
            model, loaders["val"], criterion, device, args.model
        )

        elapsed = time.time() - t0

        # Log entry
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_acc": val_acc,
            "lr": new_lr,
            "time": elapsed,
            "loss_components": loss_comps,
        }
        if train_physics is not None:
            log_entry["physics_scores_train"] = {
                name: float(v)
                for name, v in zip(score_names, train_physics)
            }
        if val_physics is not None and len(val_physics) > 0:
            log_entry["physics_scores_val"] = {
                name: float(v)
                for name, v in zip(score_names, val_physics.mean(axis=0))
            }

        training_log.append(log_entry)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train {train_loss:.4f}/{train_acc:.3f} | "
            f"Val {val_loss:.4f}/AUC:{val_auc:.4f}/{val_acc:.3f} | "
            f"LR:{new_lr:.2e} | {elapsed:.0f}s"
        )

        # Physics score summary every 5 epochs
        if (epoch + 1) % 5 == 0 and train_physics is not None:
            print(f"  Physics scores (train mean): " +
                  " | ".join(f"{n}={v:.3f}" for n, v in zip(score_names, train_physics)))

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "args": vars(args),
                "score_names": score_names,
                "training_log": training_log,
            }, ckpt_dir / "best_model.pt")
            print(f"  ★ New best AUC: {best_auc:.4f} — checkpoint saved")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "training_log": training_log,
            }, ckpt_dir / f"epoch_{epoch+1:03d}.pt")

        # Save last checkpoint (for resuming)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_auc": best_auc,
            "training_log": training_log,
        }, ckpt_dir / "last.pt")

        # Save training log after each epoch
        with open(log_dir / "training_log.json", "w") as f:
            safe_log = [{k: v for k, v in e.items()
                         if isinstance(v, (int, float, str, dict, list))}
                        for e in training_log]
            json.dump(safe_log, f, indent=2)

    # ── Final Evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Final Test Set Evaluation")
    print(f"{'='*70}")

    # Load best model
    best_ckpt = ckpt_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best model (Val AUC: {ckpt['best_auc']:.4f})")

    test_loss, test_auc, test_acc, test_preds, test_labels, test_physics = validate(
        model, loaders["test"], criterion, device, args.model
    )

    evaluator = Evaluator(output_dir=str(output_dir / "evaluation"))
    metrics = evaluator.compute_metrics(test_preds, test_labels)

    print(f"\nTest Results:")
    print(f"  AUC-ROC:    {metrics['auc']:.4f}")
    print(f"  EER:        {metrics['eer']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  Avg Prec:   {metrics['average_precision']:.4f}")

    # Physics score analysis
    real_mask = test_labels == 0
    fake_mask = test_labels == 1
    if real_mask.any() and fake_mask.any() and test_physics.ndim > 1:
        print(f"\nPhysics Score Separability (real vs fake):")
        print(f"  {'Score':<8} {'Real Mean':>12} {'Fake Mean':>12} {'Δ':>10}")
        print(f"  {'-'*44}")
        for i, name in enumerate(score_names):
            if i < test_physics.shape[1]:
                rm = test_physics[real_mask, i].mean()
                fm = test_physics[fake_mask, i].mean()
                print(f"  {name:<8} {rm:>12.4f} {fm:>12.4f} {abs(fm-rm):>10.4f}")

    # Save final results
    results = {
        "model": args.model,
        "dataset": str(args.data),
        "epochs": args.epochs,
        "best_val_auc": float(best_auc),
        "test_auc": float(metrics["auc"]),
        "test_eer": float(metrics["eer"]),
        "test_accuracy": float(metrics["accuracy"]),
        "test_f1": float(metrics["f1"]),
        "test_ap": float(metrics["average_precision"]),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "score_names": score_names,
        "hyperparams": vars(args),
    }
    with open(log_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualizations
    viz = PhysicsVisualizer(output_dir=str(viz_dir))
    train_losses = [e["train_loss"] for e in training_log]
    val_losses = [e["val_loss"] for e in training_log]
    train_accs = [e["train_acc"] for e in training_log]
    val_aucs = [e["val_auc"] for e in training_log]
    viz.plot_training_curves(train_losses, val_losses, train_accs, val_aucs,
                             save_name="training_curves.png")
    viz.plot_roc_curve(metrics["fpr_curve"], metrics["tpr_curve"], metrics["auc"],
                       dataset_name="Test Set", save_name="roc_curve.png")
    if real_mask.any() and fake_mask.any():
        viz.plot_physics_scores(test_physics[real_mask], test_physics[fake_mask],
                                save_name="physics_distribution.png")

    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  Cross-dataset eval: python scripts/cross_dataset_eval.py "
          f"--checkpoint {ckpt_dir / 'best_model.pt'} --model {args.model}")
    print(f"  View dashboard:     python app.py")


if __name__ == "__main__":
    main()

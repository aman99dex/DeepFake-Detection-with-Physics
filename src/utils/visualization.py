"""
Visualization utilities for PhysForensics.

Creates interpretable visualizations of:
- Physics violation heatmaps
- BRDF decomposition maps
- Illumination estimation
- ROC curves and metrics
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class PhysicsVisualizer:
    """Visualization tools for physics-based forensics results."""

    def __init__(self, output_dir: str = "outputs/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_physics_scores(
        self,
        real_scores: np.ndarray,
        fake_scores: np.ndarray,
        score_names: list = None,
        save_name: str = "physics_scores_distribution.png",
    ):
        """Plot distribution of physics scores for real vs fake."""
        if score_names is None:
            score_names = ["ECS", "SCS", "BSS", "ICS", "TPS"]

        num_scores = real_scores.shape[1]
        fig, axes = plt.subplots(1, num_scores, figsize=(4 * num_scores, 4))

        for i, (ax, name) in enumerate(zip(axes, score_names)):
            ax.hist(real_scores[:, i], bins=50, alpha=0.6, label="Real", color="green", density=True)
            ax.hist(fake_scores[:, i], bins=50, alpha=0.6, label="Fake", color="red", density=True)
            ax.set_title(name)
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_decomposition(
        self,
        face_image: torch.Tensor,
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        normals: torch.Tensor,
        depth: torch.Tensor,
        illumination: torch.Tensor,
        save_name: str = "decomposition.png",
    ):
        """Visualize the PBR decomposition of a face."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Denormalize face image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        face_vis = (face_image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        axes[0, 0].imshow(face_vis)
        axes[0, 0].set_title("Input Face")

        if albedo.dim() >= 2:
            side = int(np.sqrt(albedo.shape[0]))
            if side * side == albedo.shape[0]:
                axes[0, 1].imshow(albedo[:, :3].reshape(side, side, 3).detach().cpu().numpy())
            else:
                axes[0, 1].imshow(np.ones((10, 10, 3)) * albedo.mean(0)[:3].detach().cpu().numpy())
        axes[0, 1].set_title("Albedo")

        if roughness.dim() >= 1:
            side = int(np.sqrt(roughness.shape[0]))
            if side * side == roughness.shape[0]:
                axes[0, 2].imshow(roughness[:, 0].reshape(side, side).detach().cpu().numpy(), cmap="viridis")
            else:
                axes[0, 2].imshow(np.ones((10, 10)) * roughness.mean().item(), cmap="viridis")
        axes[0, 2].set_title("Roughness")

        if normals.dim() >= 2:
            side = int(np.sqrt(normals.shape[0]))
            if side * side == normals.shape[0]:
                normal_vis = (normals.reshape(side, side, 3).detach().cpu().numpy() + 1) / 2
                axes[1, 0].imshow(normal_vis)
            else:
                axes[1, 0].imshow(np.ones((10, 10, 3)) * 0.5)
        axes[1, 0].set_title("Normals")

        if depth.dim() >= 2:
            axes[1, 1].imshow(depth.squeeze().detach().cpu().numpy(), cmap="plasma")
        axes[1, 1].set_title("Depth")

        if illumination.dim() >= 2:
            side = int(np.sqrt(illumination.shape[0]))
            if side * side == illumination.shape[0]:
                axes[1, 2].imshow(illumination.reshape(side, side, 3).detach().cpu().numpy())
            else:
                axes[1, 2].imshow(np.ones((10, 10, 3)) * 0.5)
        axes[1, 2].set_title("Illumination")

        for ax in axes.flat:
            ax.axis("off")

        plt.suptitle("PhysForensics: PBR Decomposition", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        dataset_name: str = "Test",
        save_name: str = "roc_curve.png",
    ):
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"PhysForensics (AUC={auc_score:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=14)
        ax.set_ylabel("True Positive Rate", fontsize=14)
        ax.set_title(f"ROC Curve - {dataset_name}", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_training_curves(
        self,
        train_losses: list,
        val_losses: list,
        train_aucs: list,
        val_aucs: list,
        save_name: str = "training_curves.png",
    ):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, "b-", label="Train Loss")
        ax1.plot(epochs, val_losses, "r-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_aucs, "b-", label="Train AUC")
        ax2.plot(epochs, val_aucs, "r-", label="Val AUC")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC")
        ax2.set_title("AUC Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("PhysForensics Training Progress", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()

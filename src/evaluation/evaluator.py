"""
Evaluation and Benchmarking Suite for PhysForensics.

Computes:
- AUC-ROC (primary metric)
- Equal Error Rate (EER)
- Accuracy, F1, Average Precision
- Cross-dataset generalization
- Per-manipulation-type analysis
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from typing import Optional
from pathlib import Path


class Evaluator:
    """Comprehensive evaluation for deepfake detection."""

    def __init__(self, output_dir: str = "outputs/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """
        Compute all evaluation metrics.

        Args:
            predictions: (N,) predicted probabilities [0, 1]
            labels: (N,) ground truth labels {0, 1}
            threshold: classification threshold

        Returns:
            dict with all metrics
        """
        # AUC-ROC
        auc = roc_auc_score(labels, predictions)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(labels, predictions)

        # Equal Error Rate
        eer, eer_threshold = self._compute_eer(fpr, tpr, thresholds)

        # Binary predictions at threshold
        binary_preds = (predictions >= threshold).astype(int)

        # Accuracy
        acc = accuracy_score(labels, binary_preds)

        # F1 Score
        f1 = f1_score(labels, binary_preds)

        # Average Precision
        ap = average_precision_score(labels, predictions)

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()

        return {
            "auc": auc,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "accuracy": acc,
            "f1": f1,
            "average_precision": ap,
            "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "fpr_curve": fpr,
            "tpr_curve": tpr,
            "threshold_curve": thresholds,
        }

    def _compute_eer(self, fpr, tpr, thresholds):
        """Compute Equal Error Rate (where FPR == FNR)."""
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2
        return eer, thresholds[idx]

    @torch.no_grad()
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> dict:
        """
        Run full evaluation on a dataloader.

        Args:
            model: trained PhysForensics model
            dataloader: test dataloader
            device: computation device

        Returns:
            dict with metrics and raw predictions
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_physics_scores = []

        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]

            output = model(images)
            probs = output["probability"].squeeze(-1).cpu().numpy()
            physics = output["physics_scores"].cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels.numpy())
            all_physics_scores.extend(physics)

        predictions = np.array(all_preds)
        labels = np.array(all_labels)
        physics_scores = np.array(all_physics_scores)

        metrics = self.compute_metrics(predictions, labels)

        # Physics score analysis
        real_mask = labels == 0
        fake_mask = labels == 1
        physics_analysis = {
            "real_physics_mean": physics_scores[real_mask].mean(axis=0).tolist() if real_mask.any() else [],
            "fake_physics_mean": physics_scores[fake_mask].mean(axis=0).tolist() if fake_mask.any() else [],
            "real_physics_std": physics_scores[real_mask].std(axis=0).tolist() if real_mask.any() else [],
            "fake_physics_std": physics_scores[fake_mask].std(axis=0).tolist() if fake_mask.any() else [],
        }

        metrics["physics_analysis"] = physics_analysis
        metrics["raw_predictions"] = predictions
        metrics["raw_labels"] = labels
        metrics["physics_scores"] = physics_scores

        return metrics

    def cross_dataset_evaluation(
        self,
        model: torch.nn.Module,
        dataloaders: dict,
        device: str = "cuda",
    ) -> dict:
        """
        Evaluate cross-dataset generalization.

        Args:
            model: trained model
            dataloaders: dict of {dataset_name: dataloader}
            device: computation device

        Returns:
            dict of {dataset_name: metrics}
        """
        results = {}
        for name, loader in dataloaders.items():
            print(f"Evaluating on {name}...")
            metrics = self.evaluate_model(model, loader, device)
            results[name] = {
                "auc": metrics["auc"],
                "eer": metrics["eer"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "ap": metrics["average_precision"],
            }
            print(f"  AUC: {metrics['auc']:.4f} | EER: {metrics['eer']:.4f} | Acc: {metrics['accuracy']:.4f}")

        return results

    def print_results_table(self, results: dict):
        """Print results as a formatted table."""
        print("\n" + "=" * 80)
        print("PhysForensics Evaluation Results")
        print("=" * 80)
        print(f"{'Dataset':<20} {'AUC':>8} {'EER':>8} {'Acc':>8} {'F1':>8} {'AP':>8}")
        print("-" * 80)
        for name, metrics in results.items():
            print(
                f"{name:<20} {metrics['auc']:>8.4f} {metrics['eer']:>8.4f} "
                f"{metrics['accuracy']:>8.4f} {metrics['f1']:>8.4f} {metrics['ap']:>8.4f}"
            )
        print("=" * 80)

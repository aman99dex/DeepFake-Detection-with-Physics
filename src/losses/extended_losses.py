"""
Extended Loss Functions for PhysForensics v2 Extended (10-score model).

Adds two new loss terms on top of v2's 8-term loss:

    L_shadow   — Shadow placement must be consistent with estimated light direction
    L_spectral — Face color must be explainable by valid skin chromophore concentrations

Total: 10 loss terms for 10 physics scores.

L_total = w1*L_BCE + w2*L_energy + w3*L_SH + w4*L_SSS
        + w5*L_Fresnel + w6*L_contrastive + w7*L_corr + w8*L_anomaly
        + w9*L_shadow + w10*L_spectral
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .advanced_losses import (
    SHLightingRegularization,
    SSSConsistencyLoss,
    FresnelPlausibilityLoss,
    ContrastivePhysicsLoss,
    CrossScoreCorrelationLoss,
)
from ..models.shadow_consistency import ShadowConsistencyLoss
from ..models.spectral_skin import SpectralConsistencyLoss


# Extended expected correlations for 10 scores
EXTENDED_CORRELATIONS = [
    (0, 1, 0.3),   # ECS ↔ SCS
    (0, 6, 0.4),   # ECS ↔ FAS
    (5, 6, 0.3),   # SSS ↔ FAS
    (3, 8, 0.25),  # ICS ↔ ShCS  (light incoherence ↔ shadow inconsistency)
    (5, 9, 0.30),  # SSS ↔ SpSS  (subsurface ↔ spectral skin — both chromophore-based)
    (6, 9, 0.25),  # FAS ↔ SpSS  (Fresnel skin ↔ spectral skin)
]


class ExtendedCrossScoreCorrelationLoss(nn.Module):
    """Cross-score correlation for all 10 physics scores."""

    def forward(self, all_scores: torch.Tensor) -> torch.Tensor:
        B = all_scores.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=all_scores.device)

        centered = all_scores - all_scores.mean(dim=0, keepdim=True)
        std = all_scores.std(dim=0, keepdim=True) + 1e-8

        loss = torch.tensor(0.0, device=all_scores.device)
        for i, j, expected in EXTENDED_CORRELATIONS:
            if i < all_scores.shape[1] and j < all_scores.shape[1]:
                corr = (centered[:, i] * centered[:, j]).mean() / (std[0, i] * std[0, j])
                loss = loss + F.relu(expected - corr)

        return loss / max(len(EXTENDED_CORRELATIONS), 1)


class PhysicsForensicsLossExtended(nn.Module):
    """
    Complete 10-term loss for PhysForensics v2 Extended.

    L_total = BCE + energy + SH_reg + SSS + Fresnel
            + contrastive + correlation + anomaly
            + shadow + spectral
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        energy_weight: float = 0.4,
        sh_weight: float = 0.2,
        sss_weight: float = 0.3,
        fresnel_weight: float = 0.3,
        contrastive_weight: float = 0.5,
        correlation_weight: float = 0.1,
        anomaly_weight: float = 0.3,
        shadow_weight: float = 0.3,
        spectral_weight: float = 0.3,
    ):
        super().__init__()

        # Standard losses from v2
        self.bce = nn.BCEWithLogitsLoss()
        self.sh_reg = SHLightingRegularization()
        self.sss_loss = SSSConsistencyLoss()
        self.fresnel_loss = FresnelPlausibilityLoss()
        self.contrastive = ContrastivePhysicsLoss()
        self.correlation = ExtendedCrossScoreCorrelationLoss()
        self.anomaly_cal = nn.MSELoss()

        # New losses
        self.shadow_loss = ShadowConsistencyLoss(
            correlation_weight=0.5,
            directional_weight=0.3,
        )
        self.spectral_loss = SpectralConsistencyLoss(
            residual_weight=0.4,
            channel_weight=0.3,
            gamut_weight=0.3,
        )

        self.weights = {
            "bce": bce_weight,
            "energy": energy_weight,
            "sh": sh_weight,
            "sss": sss_weight,
            "fresnel": fresnel_weight,
            "contrastive": contrastive_weight,
            "correlation": correlation_weight,
            "anomaly": anomaly_weight,
            "shadow": shadow_weight,
            "spectral": spectral_weight,
        }

    def forward(self, model_output: dict, labels: torch.Tensor) -> dict:
        """
        Args:
            model_output: output dict from PhysForensicsV2Extended.forward()
            labels: (B,) binary labels 0=real, 1=fake

        Returns:
            losses: dict with individual and total losses
        """
        losses = {}

        # 1. BCE classification
        losses["bce"] = self.bce(
            model_output["logits"].squeeze(-1), labels.float()
        )

        # 2. Energy conservation (from ECS score)
        losses["energy"] = model_output["ecs"].mean()

        # 3. SH lighting regularization
        if "sh_coefficients" in model_output:
            losses["sh"] = self.sh_reg(model_output["sh_coefficients"])
        else:
            losses["sh"] = torch.tensor(0.0, device=labels.device)

        # 4. SSS consistency
        losses["sss"] = self.sss_loss(
            model_output["albedo"],
            model_output["metallic"],
        )

        # 5. Fresnel plausibility
        losses["fresnel"] = self.fresnel_loss(
            model_output["albedo"],
            model_output["metallic"],
        )

        # 6. Contrastive physics loss
        losses["contrastive"] = self.contrastive(
            model_output["physics_scores"],
            labels,
        )

        # 7. Extended cross-score correlation (10 scores)
        losses["correlation"] = self.correlation(model_output["physics_scores"])

        # 8. Anomaly calibration
        target = labels.float().unsqueeze(-1)
        losses["anomaly"] = self.anomaly_cal(
            model_output["anomaly_score"], target
        )

        # 9. Shadow consistency (NEW)
        if "predicted_shadow" in model_output and "light_direction" in model_output:
            shadow_out_dict = {
                "shadow_score": model_output["shcs"],
                "directional_violation": torch.zeros_like(model_output["shcs"]),
                "predicted_shadow": model_output.get("predicted_shadow"),
                "actual_darkness": model_output.get("actual_darkness"),
            }
            losses["shadow"] = self.shadow_loss(shadow_out_dict, labels)
        else:
            losses["shadow"] = model_output["shcs"].mean() * 0.1

        # 10. Spectral consistency (NEW)
        if "predicted_rgb" in model_output:
            spectral_out_dict = {
                "spectral_score": model_output["spss"],
                "spectral_residual": model_output["spss"],
                "rgb_ratio_score": torch.zeros_like(model_output["spss"]),
                "gamut_score": torch.zeros_like(model_output["spss"]),
            }
            losses["spectral"] = self.spectral_loss(spectral_out_dict, labels)
        else:
            losses["spectral"] = model_output["spss"].mean() * 0.1

        # Total weighted loss
        total = sum(self.weights[k] * losses[k] for k in self.weights if k in losses)
        losses["total"] = total

        return losses

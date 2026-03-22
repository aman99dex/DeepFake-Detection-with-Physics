"""
Advanced Physics Losses for PhysForensics v2.

New loss terms:
1. SH Lighting Regularization — penalizes noisy high-order SH coefficients
2. SSS Consistency Loss — penalizes deviation from skin scattering model
3. Fresnel Plausibility Loss — penalizes non-physical Fresnel parameters
4. Cross-Score Correlation Loss — enforces expected correlations between scores
5. Contrastive Physics Loss — pulls real physics features together,
   pushes fake physics features apart (supervised contrastive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SHLightingRegularization(nn.Module):
    """Regularize SH lighting coefficients.

    For faces under natural lighting:
    - Most energy should be in l=0 (ambient) and l=1 (directional)
    - l=2 (quadratic) should be small relative to l=0+l=1
    - Coefficients should be smooth (no extreme values)

    L_SH = lambda_1 * E_high_order / E_total
         + lambda_2 * max(0, ||L|| - L_max)  [coefficient magnitude bound]
    """

    def __init__(self, high_order_weight: float = 0.5, max_magnitude: float = 5.0):
        super().__init__()
        self.high_order_weight = high_order_weight
        self.max_magnitude = max_magnitude

    def forward(self, sh_coefficients: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sh_coefficients: (B, 9, 3) SH lighting coefficients
        """
        # High-order energy ratio
        l0_energy = (sh_coefficients[:, 0:1, :] ** 2).sum(dim=(1, 2))
        l1_energy = (sh_coefficients[:, 1:4, :] ** 2).sum(dim=(1, 2))
        l2_energy = (sh_coefficients[:, 4:9, :] ** 2).sum(dim=(1, 2))
        total = l0_energy + l1_energy + l2_energy + 1e-8

        high_order_loss = (l2_energy / total).mean()

        # Magnitude bound
        magnitude = sh_coefficients.abs().max()
        magnitude_loss = F.relu(magnitude - self.max_magnitude)

        return self.high_order_weight * high_order_loss + magnitude_loss


class SSSConsistencyLoss(nn.Module):
    """Subsurface Scattering Consistency Loss.

    Real skin albedo should satisfy:
    1. RGB channel ordering: R > G > B (due to hemoglobin absorption)
    2. Albedo range consistent with skin scattering parameters
    3. Smooth spatial variation (skin texture is locally smooth)

    L_SSS = L_channel_order + L_albedo_range + L_spatial_smooth
    """

    def forward(self, albedo: torch.Tensor, metallic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            albedo: (B, N, 3) or (B, 3)
            metallic: (B, N, 1) or (B, 1)
        """
        if albedo.dim() == 3:
            albedo = albedo.mean(dim=1)
            metallic = metallic.mean(dim=1)

        r, g, b = albedo[:, 0], albedo[:, 1], albedo[:, 2]

        # Channel ordering: R > G > B (for non-metallic skin)
        non_metallic = (1.0 - metallic.squeeze(-1))
        channel_loss = non_metallic * (F.relu(g - r) + F.relu(b - g))

        # Albedo range for skin: typically [0.1, 0.9] per channel
        range_loss = F.relu(0.05 - albedo).mean(dim=-1) + F.relu(albedo - 0.95).mean(dim=-1)

        return (channel_loss + range_loss).mean()


class FresnelPlausibilityLoss(nn.Module):
    """Fresnel Plausibility Loss.

    For non-metallic materials (skin):
    F0 = ((n-1)/(n+1))^2

    For skin (n ~ 1.4): F0 ~ 0.028
    Acceptable range: [0.02, 0.05]

    This is a HARD physical constraint that deepfake generators
    systematically violate.

    L_Fresnel = max(0, |F0_estimated - F0_skin| - tolerance)
    """

    def __init__(self, f0_target: float = 0.028, tolerance: float = 0.02):
        super().__init__()
        self.f0_target = f0_target
        self.tolerance = tolerance

    def forward(self, albedo: torch.Tensor, metallic: torch.Tensor) -> torch.Tensor:
        if albedo.dim() == 3:
            albedo = albedo.mean(dim=1)
            metallic = metallic.mean(dim=1)

        f0 = 0.04 * (1.0 - metallic) + albedo * metallic
        f0_mean = f0.mean(dim=-1, keepdim=True)

        deviation = torch.abs(f0_mean - self.f0_target)
        loss = F.relu(deviation - self.tolerance)

        return loss.mean()


class ContrastivePhysicsLoss(nn.Module):
    """Supervised Contrastive Loss on Physics Features.

    Pulls physics features of real faces close together
    and pushes fake physics features apart from real ones.

    L_contrastive = -log(
        sum_{j in P(i)} exp(sim(z_i, z_j) / tau) /
        sum_{k != i} exp(sim(z_i, z_k) / tau)
    )

    where P(i) are positive pairs (same class as i).

    This encourages a tight cluster of "real face physics"
    in feature space, making anomaly detection easier.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, physics_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_features: (B, D) physics score vectors
            labels: (B,) binary labels (0=real, 1=fake)
        """
        B = physics_features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=physics_features.device)

        # Normalize features (add eps to avoid zero-norm NaN gradients)
        z = F.normalize(physics_features + 1e-8, dim=-1)

        # Similarity matrix
        sim = z @ z.T / self.temperature  # (B, B)

        # Mask for positive pairs (same label)
        labels_col = labels.unsqueeze(0).expand(B, B)
        labels_row = labels.unsqueeze(1).expand(B, B)
        positive_mask = (labels_col == labels_row).float()

        # Remove self-similarities
        eye_mask = 1.0 - torch.eye(B, device=physics_features.device)
        positive_mask = positive_mask * eye_mask

        # Log-sum-exp trick for numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        # Denominator: sum over all non-self pairs
        exp_logits = torch.exp(logits) * eye_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Numerator: mean log-prob of positive pairs
        pos_count = positive_mask.sum(dim=1)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

        # Only compute for samples with at least one positive pair
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=physics_features.device)

        loss = -mean_log_prob[valid].mean()
        return loss


class CrossScoreCorrelationLoss(nn.Module):
    """Enforces expected correlations between physics scores.

    Physics scores are not independent — they have known relationships:
    - High ECS (energy violation) should correlate with high SCS (specular inconsistency)
    - High BSS (rough BRDF) should correlate with high ICS (incoherent lighting)
    - SSS violation should correlate with Fresnel anomaly

    If these correlations are violated, something is wrong with the decomposition.

    L_corr = sum |corr(s_i, s_j) - expected_corr(s_i, s_j)|
    for known correlated pairs
    """

    # Expected positive correlations between score pairs (indices)
    EXPECTED_CORRELATIONS = [
        (0, 1, 0.3),   # ECS <-> SCS
        (0, 6, 0.4),   # ECS <-> FAS (Fresnel)
        (5, 6, 0.3),   # SSS <-> FAS
    ]

    def forward(self, all_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_scores: (B, 8) all physics scores
        """
        B = all_scores.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=all_scores.device)

        # Center scores
        centered = all_scores - all_scores.mean(dim=0, keepdim=True)
        std = all_scores.std(dim=0, keepdim=True) + 1e-8

        loss = torch.tensor(0.0, device=all_scores.device)
        for i, j, expected in self.EXPECTED_CORRELATIONS:
            if i < all_scores.shape[1] and j < all_scores.shape[1]:
                corr = (centered[:, i] * centered[:, j]).mean() / (std[0, i] * std[0, j])
                # Penalize if correlation is lower than expected
                loss = loss + F.relu(expected - corr)

        return loss / max(len(self.EXPECTED_CORRELATIONS), 1)


class PhysicsForensicsLossV2(nn.Module):
    """Complete loss function for PhysForensics v2.

    L_total = w1*L_BCE + w2*L_energy + w3*L_SH + w4*L_SSS
            + w5*L_Fresnel + w6*L_contrastive + w7*L_corr + w8*L_anomaly
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
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.sh_reg = SHLightingRegularization()
        self.sss_loss = SSSConsistencyLoss()
        self.fresnel_loss = FresnelPlausibilityLoss()
        self.contrastive = ContrastivePhysicsLoss()
        self.correlation = CrossScoreCorrelationLoss()
        self.anomaly_cal = nn.MSELoss()

        self.weights = {
            "bce": bce_weight,
            "energy": energy_weight,
            "sh": sh_weight,
            "sss": sss_weight,
            "fresnel": fresnel_weight,
            "contrastive": contrastive_weight,
            "correlation": correlation_weight,
            "anomaly": anomaly_weight,
        }

    def forward(self, model_output: dict, labels: torch.Tensor) -> dict:
        losses = {}

        # 1. BCE classification
        losses["bce"] = self.bce(model_output["logits"].squeeze(-1), labels.float())

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

        # 7. Cross-score correlation
        losses["correlation"] = self.correlation(model_output["physics_scores"])

        # 8. Anomaly calibration
        target = labels.float().unsqueeze(-1)
        losses["anomaly"] = self.anomaly_cal(model_output["anomaly_score"], target)

        # Total
        total = sum(self.weights[k] * losses[k] for k in self.weights if k in losses)
        losses["total"] = total

        return losses

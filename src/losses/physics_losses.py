"""
Physics-Informed Loss Functions for PhysForensics.

Combines standard classification loss with physics-based regularization:
1. BCE loss for real/fake classification
2. Energy conservation regularization
3. Rendering consistency loss
4. BRDF smoothness regularization
5. Anomaly score calibration loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnergyConservationLoss(nn.Module):
    """Penalizes BRDF parameters that violate energy conservation.

    From PBR-NeRF: The total reflected energy from a surface point
    must not exceed the incident energy. This acts as a physics prior
    that constrains the material estimation.

    L_CoE = E[max(0, integral(f_r * cos_theta dw) - 1)]
    """

    def forward(self, brdf_params: dict) -> torch.Tensor:
        albedo = brdf_params["albedo"]       # (B, N, 3) or (B, 3)
        roughness = brdf_params["roughness"]  # (B, N, 1) or (B, 1)
        metallic = brdf_params["metallic"]    # (B, N, 1) or (B, 1)

        # Diffuse energy: albedo * (1 - metallic)
        diffuse_energy = albedo * (1.0 - metallic)

        # Specular energy approximation: F0 (Fresnel at normal incidence)
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic

        # Total should not exceed 1.0
        total_energy = diffuse_energy + f0
        violation = F.relu(total_energy - 1.0)

        return violation.mean()


class RenderingConsistencyLoss(nn.Module):
    """Measures how well the PBR rendering matches observed pixel colors.

    L_render = ||I_observed - (f_diffuse + f_specular) * L_incident||^2

    Real faces: low loss (physics explains the observation)
    Deepfakes: higher loss (physics can't fully explain the pixel values)
    """

    def forward(
        self,
        observed: torch.Tensor,
        rendered: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(rendered, observed)


class BRDFSmoothnessLoss(nn.Module):
    """Encourages spatially smooth BRDF parameters.

    Real skin has smooth material properties. This loss acts as
    a regularizer during training and produces higher values
    for deepfakes with spatially inconsistent materials.
    """

    def forward(self, brdf_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            brdf_map: (B, C, H, W) BRDF parameter maps
        """
        if brdf_map.dim() < 4:
            return torch.tensor(0.0, device=brdf_map.device)

        dx = brdf_map[:, :, :, 1:] - brdf_map[:, :, :, :-1]
        dy = brdf_map[:, :, 1:, :] - brdf_map[:, :, :-1, :]

        return (dx.abs().mean() + dy.abs().mean()) / 2.0


class AnomalyCalibrationLoss(nn.Module):
    """Calibrates the anomaly score to be meaningful.

    Ensures that:
    - Real faces get anomaly scores near 0
    - Fake faces get anomaly scores near 1
    - The score is well-calibrated (not just a shifted sigmoid)
    """

    def forward(
        self,
        anomaly_score: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # Target: real -> 0, fake -> 1
        target = labels.float().unsqueeze(-1) if labels.dim() == 1 else labels.float()
        return F.mse_loss(anomaly_score, target)


class NDFWeightedSpecularLoss(nn.Module):
    """NDF-Weighted Specular Loss from PBR-NeRF.

    Correctly separates specular from diffuse reflections using the
    GGX Normal Distribution Function as a weighting term. Prevents
    specular highlights from being "baked into" the diffuse albedo.

    L_spec = E[D(h, alpha) * ||c_observed - c_diffuse - c_specular||^2]
    """

    def forward(
        self,
        observed: torch.Tensor,
        diffuse: torch.Tensor,
        specular: torch.Tensor,
        normals: torch.Tensor,
        view_dirs: torch.Tensor,
        light_dirs: torch.Tensor,
        roughness: torch.Tensor,
    ) -> torch.Tensor:
        # Compute half-vector
        h = F.normalize(view_dirs + light_dirs, dim=-1)
        n_dot_h = torch.clamp(torch.sum(normals * h, dim=-1, keepdim=True), 0.0, 1.0)

        # GGX NDF
        alpha = roughness ** 2
        a2 = alpha ** 2
        denom = n_dot_h ** 2 * (a2 - 1.0) + 1.0
        D = a2 / (math.pi * denom ** 2 + 1e-7)

        # Weighted reconstruction error
        residual = (observed - diffuse - specular) ** 2
        weighted = D * residual

        return weighted.mean()


class PhysicsForensicsLoss(nn.Module):
    """Complete loss function for PhysForensics training.

    L_total = w_bce * L_bce
            + w_coe * L_energy_conservation
            + w_render * L_rendering_consistency
            + w_smooth * L_brdf_smoothness
            + w_anomaly * L_anomaly_calibration
            + w_spec * L_ndf_specular
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        energy_conservation_weight: float = 0.5,
        rendering_weight: float = 0.3,
        smoothness_weight: float = 0.2,
        anomaly_weight: float = 0.3,
        specular_weight: float = 0.2,
    ):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.energy_loss = EnergyConservationLoss()
        self.render_loss = RenderingConsistencyLoss()
        self.smooth_loss = BRDFSmoothnessLoss()
        self.anomaly_loss = AnomalyCalibrationLoss()
        self.specular_loss = NDFWeightedSpecularLoss()

        self.w_bce = bce_weight
        self.w_coe = energy_conservation_weight
        self.w_render = rendering_weight
        self.w_smooth = smoothness_weight
        self.w_anomaly = anomaly_weight
        self.w_spec = specular_weight

    def forward(self, model_output: dict, labels: torch.Tensor, observed: torch.Tensor = None) -> dict:
        """
        Args:
            model_output: dict from PhysForensics.forward()
            labels: (B,) binary labels (0=real, 1=fake)
            observed: (B, N, 3) observed pixel colors

        Returns:
            dict with total loss and individual components
        """
        losses = {}

        # 1. Classification loss
        losses["bce"] = self.bce_loss(model_output["logits"].squeeze(-1), labels.float())

        # 2. Energy conservation
        brdf_params = {
            "albedo": model_output["albedo"],
            "roughness": model_output["roughness"],
            "metallic": model_output["metallic"],
        }
        losses["energy_conservation"] = self.energy_loss(brdf_params)

        # 3. Rendering consistency
        if observed is not None:
            losses["rendering"] = self.render_loss(observed, model_output["rendered"])
        else:
            losses["rendering"] = torch.tensor(0.0, device=labels.device)

        # 4. Anomaly calibration
        losses["anomaly"] = self.anomaly_loss(model_output["anomaly_score"], labels)

        # 5. Total weighted loss
        total = (
            self.w_bce * losses["bce"]
            + self.w_coe * losses["energy_conservation"]
            + self.w_render * losses["rendering"]
            + self.w_anomaly * losses["anomaly"]
        )
        losses["total"] = total

        return losses

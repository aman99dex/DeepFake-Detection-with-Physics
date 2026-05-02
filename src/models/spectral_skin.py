"""
Spectral Skin Model for PhysForensics.

Physical Principle:
    Human skin color is determined by the concentrations of two primary
    chromophores: melanin and hemoglobin. Both have well-characterized
    spectral absorption spectra measured in vivo. The relationship between
    R, G, B channel responses and these chromophore concentrations follows
    a specific Beer-Lambert law model.

    A real face's color MUST be explainable by physically valid melanin
    and hemoglobin concentrations. Deepfakes generate colors by learning
    pixel distributions — they do not enforce this spectral constraint,
    leading to systematic spectral violations.

Spectral Skin Score (SpSS):
    1. Model skin color via Beer-Lambert: I = I0 * exp(-mu_a * t)
       where mu_a = c_mel * mu_mel + c_hemo * mu_hemo
    2. Find best-fit (c_mel, c_hemo) for observed face color
    3. Compute spectral residual = ||I_observed - I_predicted||
    4. Residual is low for real faces, high for fakes

Measured absorption coefficients:
    Melanin:    mu_mel  = [6.6, 3.3, 2.0] cm^-1 at [R, G, B] wavelengths
    Oxyhemoglobin: mu_hemo = [0.5, 2.3, 0.5] cm^-1 at [R, G, B]
    (Based on Bashkatov et al., 2011; Jacques, 2013)

References:
    - Jacques, "Optical properties of biological tissues: a review," 2013
    - Bashkatov et al., "Optical properties of skin, subcutaneous, and
      muscle tissue: a review," J. Innovative Optical Health Sciences, 2011
    - Tsumura et al., "Image-based skin color decomposition," SIGGRAPH 2003
    - Nunez et al., "Skin melanin and hemoglobin spectral model for face
      forgery detection," CVPRW 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── Measured Skin Optical Properties ─────────────────────────────────────────

# Spectral absorption coefficients for skin chromophores
# Wavelengths: Red (~670nm), Green (~540nm), Blue (~450nm)
# Units: cm^-1 (absorption per unit depth)

# Melanin absorption (increases strongly toward blue)
MELANIN_ABS = torch.tensor([6.6, 3.3, 2.0], dtype=torch.float32)

# Oxyhemoglobin absorption (two peaks at ~415nm and ~540nm)
OXYHEMO_ABS = torch.tensor([0.5, 2.3, 0.5], dtype=torch.float32)

# Reduced scattering coefficient for skin (wavelength-dependent)
# mu_s' = a * (lambda/500)^(-b) where a~10, b~1.5 for dermis
SCATTER_COEFF = torch.tensor([7.4, 9.2, 12.8], dtype=torch.float32)

# IOR of skin layers (for Fresnel at air-skin interface)
SKIN_IOR = 1.4

# Expected albedo range for healthy skin (all ethnicities)
SKIN_ALBEDO_MIN = torch.tensor([0.12, 0.09, 0.06], dtype=torch.float32)
SKIN_ALBEDO_MAX = torch.tensor([0.87, 0.78, 0.65], dtype=torch.float32)


class SpectralSkinModel(nn.Module):
    """
    Physics-based spectral model for human skin color.

    Fits melanin and hemoglobin concentrations to observed face colors
    using a simplified Beer-Lambert diffuse reflectance model, then
    measures how well the spectral model explains the observed colors.

    A face that can be well-explained by the model is likely real.
    A face with unexplainable spectral properties is likely a deepfake.
    """

    def __init__(
        self,
        num_iterations: int = 5,
        skin_depth: float = 0.15,
    ):
        """
        Args:
            num_iterations: iterations for fitting chromophore concentrations
            skin_depth: effective optical depth for reflectance model (cm)
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.skin_depth = skin_depth

        # Register spectral constants as buffers (move to device automatically)
        self.register_buffer("melanin_abs", MELANIN_ABS.clone())
        self.register_buffer("oxyhemo_abs", OXYHEMO_ABS.clone())
        self.register_buffer("scatter_coeff", SCATTER_COEFF.clone())
        self.register_buffer("skin_albedo_min", SKIN_ALBEDO_MIN.clone())
        self.register_buffer("skin_albedo_max", SKIN_ALBEDO_MAX.clone())

        # Learnable correction factors for model-image domain gap
        self.spectral_bias = nn.Parameter(torch.zeros(3))
        self.spectral_scale = nn.Parameter(torch.ones(3))

        # Learned spectral feature extractor
        self.spectral_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        # Chromophore concentration predictor
        self.conc_predictor = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   # [c_melanin, c_hemoglobin]
            nn.Softplus(),       # Concentrations must be non-negative
        )

    def beer_lambert_reflectance(
        self,
        c_melanin: torch.Tensor,
        c_hemoglobin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expected diffuse reflectance via Beer-Lambert law.

        R(lambda) ≈ exp(-2 * mu_eff * d)
        where:
            mu_eff = sqrt(3 * mu_a * (mu_a + mu_s'))
            mu_a = c_mel * mu_mel + c_hemo * mu_hemo

        Args:
            c_melanin: (B,) melanin concentration
            c_hemoglobin: (B,) hemoglobin concentration

        Returns:
            reflectance: (B, 3) predicted RGB reflectance in [0, 1]
        """
        B = c_melanin.shape[0]

        # Total absorption: (B, 3)
        mu_a = (
            c_melanin.unsqueeze(-1) * self.melanin_abs.unsqueeze(0) +
            c_hemoglobin.unsqueeze(-1) * self.oxyhemo_abs.unsqueeze(0)
        )

        # Effective transport coefficient: mu_eff = sqrt(3 * mu_a * (mu_a + mu_s'))
        mu_s = self.scatter_coeff.unsqueeze(0).expand(B, -1)
        mu_eff = torch.sqrt(3.0 * mu_a * (mu_a + mu_s) + 1e-8)

        # Reflectance: R = exp(-2 * mu_eff * d)
        # Factor of 2 for round-trip (in + out)
        reflectance = torch.exp(-2.0 * mu_eff * self.skin_depth)

        return reflectance.clamp(0, 1)

    def fit_chromophores(self, observed_rgb: torch.Tensor) -> tuple:
        """
        Fit melanin and hemoglobin concentrations to explain observed colors.

        Uses the learned predictor network for efficient fitting.

        Args:
            observed_rgb: (B, 3) mean observed face color (normalized 0-1)

        Returns:
            c_mel: (B,) melanin concentrations
            c_hemo: (B,) hemoglobin concentrations
            predicted_rgb: (B, 3) model-predicted RGB
        """
        # Encode color features
        features = self.spectral_encoder(observed_rgb)

        # Predict chromophore concentrations
        concentrations = self.conc_predictor(features)
        c_mel = concentrations[:, 0] * 2.0    # Scale: typical range 0-2
        c_hemo = concentrations[:, 1] * 0.5   # Scale: typical range 0-0.5

        # Compute predicted reflectance
        predicted = self.beer_lambert_reflectance(c_mel, c_hemo)

        # Apply learnable domain correction
        scale = self.spectral_scale.unsqueeze(0).clamp(0.5, 2.0)
        bias = self.spectral_bias.unsqueeze(0).clamp(-0.2, 0.2)
        predicted = (predicted * scale + bias).clamp(0, 1)

        return c_mel, c_hemo, predicted

    def compute_spectral_anomalies(
        self,
        observed_rgb: torch.Tensor,
        c_mel: torch.Tensor,
        c_hemo: torch.Tensor,
    ) -> dict:
        """
        Compute specific spectral anomalies that deepfakes tend to exhibit.

        1. RGB ratio anomaly: real skin has R > G > B (melanin + hemo absorption)
        2. Spectral slope anomaly: blue channel should always be lowest
        3. Concentration plausibility: c_mel and c_hemo in valid ranges
        4. Skin gamut violation: color must lie within the skin color gamut

        Args:
            observed_rgb: (B, 3) observed mean face color
            c_mel: (B,) fitted melanin concentration
            c_hemo: (B,) fitted hemoglobin concentration

        Returns:
            dict of individual anomaly scores
        """
        R, G, B = observed_rgb[:, 0], observed_rgb[:, 1], observed_rgb[:, 2]

        # 1. Channel ordering: real skin has R > G (melanin + hemo absorb more at G/B)
        #    Exception: very pale/metallic-looking fakes often have R ≈ G ≈ B or B > R
        rg_violation = F.relu(G - R)      # G should not exceed R for skin
        gb_violation = F.relu(B - G)      # B should not exceed G for skin
        channel_score = (rg_violation + gb_violation) * 0.5

        # 2. Spectral contrast: real skin has meaningful R-B difference
        #    Very uniform spectrally flat colors are unphysical for faces
        spectral_contrast = (R - B).abs()
        low_contrast_penalty = F.relu(0.05 - spectral_contrast)  # too flat

        # 3. Concentration plausibility
        #    Melanin: [0.1, 4.0] covers albino to dark skin
        #    Hemoglobin: [0.01, 1.5] covers pale to very flushed
        mel_violation = (
            F.relu(0.05 - c_mel) +         # Too little melanin
            F.relu(c_mel - 5.0)             # Too much melanin
        )
        hemo_violation = (
            F.relu(0.005 - c_hemo) +        # Too little hemoglobin
            F.relu(c_hemo - 2.0)            # Too much hemoglobin
        )
        concentration_score = mel_violation * 0.3 + hemo_violation * 0.3

        # 4. Absolute skin gamut: R and G must be in [0.1, 0.9], B in [0.05, 0.75]
        gamut_violation = (
            F.relu(self.skin_albedo_min[0] - R) + F.relu(R - self.skin_albedo_max[0]) +
            F.relu(self.skin_albedo_min[1] - G) + F.relu(G - self.skin_albedo_max[1]) +
            F.relu(self.skin_albedo_min[2] - B) + F.relu(B - self.skin_albedo_max[2])
        )

        return {
            "channel_ordering": channel_score,
            "low_contrast": low_contrast_penalty,
            "concentration": concentration_score,
            "gamut_violation": gamut_violation,
        }

    def forward(
        self,
        face_image: torch.Tensor,
        albedo: torch.Tensor = None,
    ) -> dict:
        """
        Compute the Spectral Skin Score.

        Args:
            face_image: (B, 3, H, W) face image
            albedo: (B, N, 3) or (B, 3) optional PBR-estimated albedo
                    If not provided, uses face_image colors directly

        Returns:
            dict with:
                spectral_score: (B, 1) total spectral anomaly (0=real, 1=fake)
                c_melanin: (B,) fitted melanin concentration
                c_hemoglobin: (B,) fitted hemoglobin concentration
                spectral_residual: (B, 1) fitting residual
                rgb_ratio_score: (B, 1) RGB channel ordering anomaly
                gamut_score: (B, 1) skin gamut violation
        """
        B = face_image.shape[0]

        # Use albedo if available (PBR-NeRF output), else use raw image
        if albedo is not None:
            if albedo.dim() == 3:
                rgb_mean = albedo.mean(dim=1).clamp(0, 1)   # (B, 3)
            else:
                rgb_mean = albedo.clamp(0, 1)
        else:
            # Normalize image to [0, 1]
            img = face_image
            if img.min() < -0.1:
                img = (img + 1.0) / 2.0
            rgb_mean = img.mean(dim=(2, 3)).clamp(0, 1)     # (B, 3)

        # Fit chromophore concentrations
        c_mel, c_hemo, predicted_rgb = self.fit_chromophores(rgb_mean)

        # Spectral fitting residual
        residual = (rgb_mean - predicted_rgb).abs().mean(dim=-1, keepdim=True)  # (B, 1)

        # Compute specific anomalies
        anomalies = self.compute_spectral_anomalies(rgb_mean, c_mel, c_hemo)

        # Combine into total spectral score
        channel_score = anomalies["channel_ordering"].unsqueeze(-1)
        gamut_score = anomalies["gamut_violation"].unsqueeze(-1).clamp(0, 1)
        conc_score = anomalies["concentration"].unsqueeze(-1)
        contrast_score = anomalies["low_contrast"].unsqueeze(-1)

        # Weighted combination
        spectral_score = (
            0.35 * residual +
            0.30 * channel_score +
            0.20 * gamut_score +
            0.10 * conc_score +
            0.05 * contrast_score
        ).clamp(0, 2.0)

        return {
            "spectral_score": spectral_score,
            "c_melanin": c_mel,
            "c_hemoglobin": c_hemo,
            "spectral_residual": residual,
            "rgb_ratio_score": channel_score,
            "gamut_score": gamut_score,
            "concentration_score": conc_score,
            "predicted_rgb": predicted_rgb,
            "observed_rgb": rgb_mean,
        }


class SpectralConsistencyLoss(nn.Module):
    """
    Loss function enforcing spectral skin consistency.

    For real faces:
        - Spectral residual should be small (skin is well-modeled)
        - RGB ordering should satisfy R > G > B
        - Chromophore concentrations should be in valid ranges

    L_spectral = L_residual + L_channel_order + L_gamut
    """

    def __init__(
        self,
        residual_weight: float = 0.4,
        channel_weight: float = 0.3,
        gamut_weight: float = 0.3,
    ):
        super().__init__()
        self.residual_weight = residual_weight
        self.channel_weight = channel_weight
        self.gamut_weight = gamut_weight

    def forward(
        self,
        spectral_output: dict,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spectral_output: output from SpectralSkinModel.forward()
            labels: (B,) binary labels, 0=real, 1=fake
        """
        real_mask = (labels == 0).float()

        # Penalize real faces that have large spectral residuals
        residual_loss = (real_mask * spectral_output["spectral_residual"].squeeze(-1)).sum() / (
            real_mask.sum() + 1e-8
        )

        # Penalize real faces that violate RGB ordering
        channel_loss = (real_mask * spectral_output["rgb_ratio_score"].squeeze(-1)).sum() / (
            real_mask.sum() + 1e-8
        )

        # Penalize all faces that fall outside skin gamut
        # (even fakes should look like faces, so gamut violations are errors)
        gamut_loss = spectral_output["gamut_score"].mean()

        return (
            self.residual_weight * residual_loss +
            self.channel_weight * channel_loss +
            self.gamut_weight * gamut_loss
        )

"""
PhysForensics v2: Enhanced Model with Advanced Physics.

Major upgrades over v1:
1. Spherical Harmonics lighting (SfSNet-inspired) instead of NeILF
2. Subsurface Scattering consistency (Jensen dipole model)
3. Fresnel reflectance anomaly detection
4. GTR importance-sampled energy conservation (reduced MC variance)
5. Wasserstein anomaly scoring for distribution-level detection
6. 8 physics scores instead of 5

The physics scoring now uses 8 complementary signals:
- ECS: Energy Conservation Score (GTR importance-sampled)
- SCS: Specular Consistency Score
- BSS: BRDF Smoothness Score
- ICS: Illumination Coherence Score (SH-based)
- TPS: Temporal Physics Stability
- SSS: Subsurface Scattering Consistency
- FAS: Fresnel Anomaly Score
- WAS: Wasserstein Anomaly Score (distribution-level)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .pbr_nerf_backbone import PBRNeRFBackbone, DisneyBRDF
from .physics_scorer import (
    SpecularConsistencyScorer,
    BRDFSmoothnessScorer,
    TemporalPhysicsStabilityScorer,
)
from .advanced_physics import (
    SphericalHarmonicsLighting,
    AdaptiveSHLightingEstimator,
    SubsurfaceScatteringModel,
    GTRImportanceSampler,
    WassersteinAnomalyDetector,
    FresnelAnomalyDetector,
)
from .forensic_classifier import ForensicClassifier


class SHIlluminationCoherenceScorer(nn.Module):
    """SH-based Illumination Coherence Scoring.

    More principled than v1's grid-based approach.
    Uses SH coefficient analysis: real faces have most
    energy in low-order (l=0,1) bands. Deepfakes leak
    into higher-order (l=2) bands.
    """

    def __init__(self):
        super().__init__()

    def forward(self, sh_coefficients: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sh_coefficients: (B, 9, 3) SH lighting coefficients

        Returns:
            ics: (B, 1) illumination coherence score
        """
        # Energy per SH band
        l0_energy = (sh_coefficients[:, 0:1, :] ** 2).sum(dim=(1, 2))  # DC
        l1_energy = (sh_coefficients[:, 1:4, :] ** 2).sum(dim=(1, 2))  # Linear
        l2_energy = (sh_coefficients[:, 4:9, :] ** 2).sum(dim=(1, 2))  # Quadratic

        total = l0_energy + l1_energy + l2_energy + 1e-8

        # Incoherence = ratio of high-order energy
        # Real faces: most energy in l0+l1 (smooth lighting)
        # Deepfakes: spurious energy in l2 (noisy lighting)
        incoherence = l2_energy / total

        return incoherence.unsqueeze(-1)


class EnhancedEnergyConservationScorer(nn.Module):
    """GTR importance-sampled Energy Conservation Score.

    Uses VNDF sampling for dramatically reduced MC variance
    compared to v1's uniform hemisphere sampling.
    """

    def __init__(self, num_samples: int = 128):
        super().__init__()
        self.gtr_sampler = GTRImportanceSampler(num_samples=num_samples)

    def forward(self, brdf_params: dict) -> torch.Tensor:
        albedo = brdf_params["albedo"]
        roughness = brdf_params["roughness"]
        metallic = brdf_params["metallic"]

        total_energy = self.gtr_sampler.compute_energy_integral(
            albedo, roughness, metallic
        )

        # Violation: energy exceeds 1.0
        violation = F.relu(total_energy - 1.0).mean(dim=-1, keepdim=True)
        return violation


class PhysForensicsV2(nn.Module):
    """PhysForensics v2: Enhanced Physics-Informed Deepfake Detection.

    End-to-end model with 8 physics consistency scores and
    Wasserstein anomaly detection.
    """

    NUM_PHYSICS_SCORES = 8

    def __init__(
        self,
        image_size: int = 256,
        num_sample_points: int = 1024,
        nerf_hidden: int = 256,
        nerf_layers: int = 8,
        classifier_hidden: int = 512,
        fusion_type: str = "attention",
        dropout: float = 0.3,
        gtr_samples: int = 128,
        skin_ior: float = 1.4,
    ):
        super().__init__()

        # ---- Stage 1: Face to 3D ----
        from .physforensics import FacePointSampler
        self.point_sampler = FacePointSampler(
            image_size=image_size,
            num_points=num_sample_points,
        )

        # ---- Stage 2: PBR-NeRF Backbone ----
        self.pbr_nerf = PBRNeRFBackbone(
            hidden_dim=nerf_hidden,
            num_layers=nerf_layers,
        )

        # ---- Stage 2b: SH Lighting ----
        self.sh_estimator = AdaptiveSHLightingEstimator()
        self.sh_lighting = SphericalHarmonicsLighting()

        # ---- Stage 3: Physics Scoring (8 scores) ----
        # Score 1: Energy Conservation (GTR importance-sampled)
        self.ecs_scorer = EnhancedEnergyConservationScorer(num_samples=gtr_samples)
        # Score 2: Specular Consistency
        self.scs_scorer = SpecularConsistencyScorer()
        # Score 3: BRDF Smoothness
        self.bss_scorer = BRDFSmoothnessScorer()
        # Score 4: Illumination Coherence (SH-based)
        self.ics_scorer = SHIlluminationCoherenceScorer()
        # Score 5: Temporal Physics Stability
        self.tps_scorer = TemporalPhysicsStabilityScorer()
        # Score 6: Subsurface Scattering Consistency
        self.sss_model = SubsurfaceScatteringModel(eta=skin_ior)
        # Score 7: Fresnel Anomaly
        self.fresnel_detector = FresnelAnomalyDetector()
        # Score 8: Wasserstein Anomaly
        self.wasserstein_detector = WassersteinAnomalyDetector(
            feature_dim=self.NUM_PHYSICS_SCORES,
        )

        # Score normalization
        self.score_norm = nn.BatchNorm1d(self.NUM_PHYSICS_SCORES)

        # Learnable score weights
        self.score_weights = nn.Parameter(torch.ones(self.NUM_PHYSICS_SCORES))

        # ---- Stage 4: Forensic Classifier ----
        self.classifier = ForensicClassifier(
            visual_feature_dim=256,
            physics_feature_dim=self.NUM_PHYSICS_SCORES,
            hidden_dim=classifier_hidden,
            fusion_type=fusion_type,
            dropout=dropout,
        )

    def forward(
        self,
        face_image: torch.Tensor,
        temporal_scores: torch.Tensor = None,
        is_real_for_calibration: torch.Tensor = None,
    ) -> dict:
        """
        Full forward pass with 8 physics scores.

        Args:
            face_image: (B, 3, H, W) aligned face crop
            temporal_scores: (B, T, 8) previous frame scores (video mode)
            is_real_for_calibration: (B,) bool mask for Wasserstein calibration

        Returns:
            Complete prediction dictionary.
        """
        B = face_image.shape[0]
        device = face_image.device

        # ---- Stage 1: Sample 3D points ----
        sample_output = self.point_sampler(face_image)
        points = sample_output["points"]
        normals = sample_output["normals"]
        view_dirs = sample_output["view_dirs"]

        # ---- Stage 2: PBR-NeRF inverse rendering ----
        rendering = self.pbr_nerf(
            points=points,
            view_dirs=view_dirs,
            normals=normals,
        )

        # ---- Stage 2b: SH Lighting estimation ----
        sh_coeffs = self.sh_estimator(face_image)  # (B, 9, 3)
        sh_irradiance = SphericalHarmonicsLighting.evaluate_sh_basis(
            normals.reshape(B, -1, 3)
        )  # (B, N, 9)
        sh_light = torch.matmul(sh_irradiance, sh_coeffs)  # (B, N, 3)
        sh_light = F.softplus(sh_light, beta=5)

        # ---- Stage 3: Compute 8 physics scores ----
        # Aggregate spatial features for scoring
        albedo_mean = rendering["albedo"].reshape(B, -1, 3).mean(dim=1)
        roughness_mean = rendering["roughness"].reshape(B, -1, 1).mean(dim=1)
        metallic_mean = rendering["metallic"].reshape(B, -1, 1).mean(dim=1)
        normals_mean = F.normalize(rendering["normals"].reshape(B, -1, 3).mean(dim=1), dim=-1)
        diffuse_mean = rendering["diffuse"].reshape(B, -1, 3).mean(dim=1)
        specular_mean = rendering["specular"].reshape(B, -1, 3).mean(dim=1)

        brdf_params = {
            "albedo": albedo_mean,
            "roughness": roughness_mean,
            "metallic": metallic_mean,
        }

        # Score 1: ECS (GTR importance-sampled)
        ecs = self.ecs_scorer(brdf_params)

        # Score 2: SCS
        # Use observed pixel colors
        pixel_coords = sample_output["pixel_coords"].unsqueeze(2)
        observed = F.grid_sample(
            face_image, pixel_coords, mode="bilinear", align_corners=True
        ).squeeze(-1).permute(0, 2, 1)
        obs_mean = observed.mean(dim=1)
        scs = self.scs_scorer(obs_mean, diffuse_mean, specular_mean, sh_light.mean(dim=1))

        # Score 3: BSS
        N = rendering["albedo"].shape[1]
        side = int(math.sqrt(N))
        if side * side == N and side > 1:
            brdf_map = torch.cat([
                rendering["albedo"],
                rendering["roughness"],
                rendering["metallic"],
            ], dim=-1).reshape(B, side, side, -1).permute(0, 3, 1, 2)
            bss = self.bss_scorer(brdf_map)
        else:
            bss = torch.zeros(B, 1, device=device)

        # Score 4: ICS (SH-based)
        ics = self.ics_scorer(sh_coeffs)

        # Score 5: TPS
        if temporal_scores is not None:
            tps = self.tps_scorer(temporal_scores)
        else:
            tps = torch.zeros(B, 1, device=device)

        # Score 6: SSS
        sss_output = self.sss_model(rendering["albedo"].reshape(B, -1, 3))
        sss = sss_output["sss_albedo_score"].unsqueeze(-1) if sss_output["sss_albedo_score"].dim() == 1 else sss_output["sss_albedo_score"]
        sss_ratio = sss_output["sss_ratio_score"].unsqueeze(-1) if sss_output["sss_ratio_score"].dim() == 1 else sss_output["sss_ratio_score"]
        # Combined SSS score
        sss_combined = sss + sss_ratio

        # Score 7: Fresnel Anomaly
        fresnel_output = self.fresnel_detector(
            rendering["albedo"].reshape(B, -1, 3),
            rendering["metallic"].reshape(B, -1, 1),
            rendering["roughness"].reshape(B, -1, 1),
        )
        fas = (
            fresnel_output["f0_violation"]
            + fresnel_output["metallic_violation"]
            + fresnel_output["roughness_violation"]
            + fresnel_output["monotonicity_violation"]
        )

        # Aggregate first 7 scores
        scores_7 = torch.cat([ecs, scs, bss, ics, tps, sss_combined, fas], dim=-1)  # (B, 7)

        # Score 8: Wasserstein Anomaly (uses the other 7 scores as features)
        # Pad to 8 features for the Wasserstein detector
        scores_padded = F.pad(scores_7, (0, 1), value=0)  # (B, 8)

        # Update real statistics during training
        if is_real_for_calibration is not None and self.training:
            real_mask = is_real_for_calibration.bool()
            if real_mask.any():
                self.wasserstein_detector.update_real_statistics(
                    scores_padded[real_mask].detach()
                )

        wasserstein_output = self.wasserstein_detector(scores_padded)
        was = wasserstein_output["mahalanobis_distance"]

        # All 8 scores
        all_scores = torch.cat([ecs, scs, bss, ics, tps, sss_combined, fas, was], dim=-1)

        # Normalize and weight
        normalized = self.score_norm(all_scores)
        weights = torch.softmax(self.score_weights, dim=0)
        aggregated = (normalized * weights).sum(dim=-1, keepdim=True)

        # ---- Stage 4: Classification ----
        classification = self.classifier(
            face_image=face_image,
            physics_scores=all_scores,
        )

        return {
            # Final output
            "logits": classification["logits"],
            "probability": classification["probability"],
            "anomaly_score": classification["anomaly_score"],

            # 8 Physics scores
            "physics_scores": all_scores,
            "ecs": ecs, "scs": scs, "bss": bss, "ics": ics,
            "tps": tps, "sss": sss_combined, "fas": fas, "was": was,
            "aggregated_physics": aggregated,

            # Decomposition
            "albedo": rendering["albedo"],
            "roughness": rendering["roughness"],
            "metallic": rendering["metallic"],
            "normals": rendering["normals"],
            "depth_map": sample_output["depth_map"],
            "sh_coefficients": sh_coeffs,
            "sh_irradiance": sh_light,
            "rendered": rendering["rendered"],

            # Fresnel details
            "estimated_f0": fresnel_output["estimated_f0"],

            # Wasserstein details
            "mahalanobis_distance": wasserstein_output["mahalanobis_distance"],
        }

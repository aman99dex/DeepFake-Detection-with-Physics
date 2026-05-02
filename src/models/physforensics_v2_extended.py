"""
PhysForensics v2 Extended: 10-Score Physics-Informed Deepfake Detection.

Adds two new physics signals on top of v2's 8 scores:

Score 9:  ShCS — Shadow Consistency Score
    Cast shadows must geometrically align with the dominant light direction
    estimated from Spherical Harmonics. Deepfakes frequently violate this
    because the face is generated under one lighting context and composited
    into another.

Score 10: SpSS — Spectral Skin Score
    Human skin color is determined by melanin and hemoglobin concentrations
    following Beer-Lambert spectral absorption. Colors that cannot be
    explained by physically valid chromophore concentrations are fake.

All 10 scores:
    1.  ECS  — Energy Conservation Score        (GTR importance-sampled)
    2.  SCS  — Specular Consistency Score        (GGX NDF residual)
    3.  BSS  — BRDF Smoothness Score             (spatial gradient TV)
    4.  ICS  — Illumination Coherence Score      (SH band energy ratio)
    5.  TPS  — Temporal Physics Stability        (video-mode inter-frame)
    6.  SSS  — Subsurface Scattering Consistency (Jensen dipole R>G>B)
    7.  FAS  — Fresnel Anomaly Score             (F0 ≈ 0.028 for skin)
    8.  WAS  — Wasserstein Anomaly Score         (distribution-level)
    9.  ShCS — Shadow Consistency Score          (NEW)
    10. SpSS — Spectral Skin Score               (NEW)
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
from .shadow_consistency import ShadowConsistencyScorer
from .spectral_skin import SpectralSkinModel

# Re-use helpers from v2
from .physforensics_v2 import SHIlluminationCoherenceScorer, EnhancedEnergyConservationScorer


class PhysForensicsV2Extended(nn.Module):
    """
    PhysForensics v2 Extended: 10 complementary physics signals.

    Each signal is independently motivated by a different physical law,
    making the ensemble highly robust to any single signal being noisy
    or domain-shifted.
    """

    NUM_PHYSICS_SCORES = 10
    SCORE_NAMES = ["ECS", "SCS", "BSS", "ICS", "TPS", "SSS", "FAS", "WAS", "ShCS", "SpSS"]

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
        self.image_size = image_size

        # ── Stage 1: Face → 3D ─────────────────────────────────────────────
        from .physforensics import FacePointSampler
        self.point_sampler = FacePointSampler(
            image_size=image_size,
            num_points=num_sample_points,
        )

        # ── Stage 2: PBR-NeRF Inverse Rendering ────────────────────────────
        self.pbr_nerf = PBRNeRFBackbone(
            hidden_dim=nerf_hidden,
            num_layers=nerf_layers,
        )
        self.sh_estimator = AdaptiveSHLightingEstimator()
        self.sh_lighting = SphericalHarmonicsLighting()

        # ── Stage 3: Physics Scorers (10 scores) ───────────────────────────
        self.ecs_scorer = EnhancedEnergyConservationScorer(num_samples=gtr_samples)
        self.scs_scorer = SpecularConsistencyScorer()
        self.bss_scorer = BRDFSmoothnessScorer()
        self.ics_scorer = SHIlluminationCoherenceScorer()
        self.tps_scorer = TemporalPhysicsStabilityScorer()
        self.sss_model = SubsurfaceScatteringModel(eta=skin_ior)
        self.fresnel_detector = FresnelAnomalyDetector()
        self.wasserstein_detector = WassersteinAnomalyDetector(
            feature_dim=self.NUM_PHYSICS_SCORES,
        )
        # New scores
        self.shadow_scorer = ShadowConsistencyScorer(image_size=image_size)
        self.spectral_scorer = SpectralSkinModel()

        # Score normalization & weighting
        self.score_norm = nn.BatchNorm1d(self.NUM_PHYSICS_SCORES)
        self.score_weights = nn.Parameter(torch.ones(self.NUM_PHYSICS_SCORES))

        # ── Stage 4: Forensic Classifier ───────────────────────────────────
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
        Full 10-score forward pass.

        Args:
            face_image: (B, 3, H, W) aligned face crop
            temporal_scores: (B, T, 10) previous frame scores (video mode)
            is_real_for_calibration: (B,) bool mask for Wasserstein calibration

        Returns:
            Complete prediction dict with all 10 physics scores.
        """
        B = face_image.shape[0]
        device = face_image.device
        H, W = face_image.shape[2], face_image.shape[3]

        # ── Stage 1: 3D point sampling ──────────────────────────────────────
        sample_output = self.point_sampler(face_image)
        points = sample_output["points"]
        normals = sample_output["normals"]       # (B, N, 3)
        view_dirs = sample_output["view_dirs"]
        depth_map = sample_output["depth_map"]   # (B, 1, H, W)

        # ── Stage 2: PBR-NeRF rendering ─────────────────────────────────────
        rendering = self.pbr_nerf(points=points, view_dirs=view_dirs, normals=normals)

        # SH lighting
        sh_coeffs = self.sh_estimator(face_image)                        # (B, 9, 3)
        sh_irradiance = SphericalHarmonicsLighting.evaluate_sh_basis(
            normals.reshape(B, -1, 3)
        )                                                                  # (B, N, 9)
        sh_light = F.softplus(torch.matmul(sh_irradiance, sh_coeffs), beta=5)  # (B, N, 3)

        # Aggregate spatial features
        albedo_pts = rendering["albedo"]            # (B, N, 3)
        roughness_pts = rendering["roughness"]      # (B, N, 1)
        metallic_pts = rendering["metallic"]        # (B, N, 1)
        normals_pts = rendering["normals"]          # (B, N, 3)
        diffuse_mean = rendering["diffuse"].reshape(B, -1, 3).mean(dim=1)
        specular_mean = rendering["specular"].reshape(B, -1, 3).mean(dim=1)

        albedo_mean = albedo_pts.reshape(B, -1, 3).mean(dim=1)
        roughness_mean = roughness_pts.reshape(B, -1, 1).mean(dim=1)
        metallic_mean = metallic_pts.reshape(B, -1, 1).mean(dim=1)

        brdf_params = {
            "albedo": albedo_mean,
            "roughness": roughness_mean,
            "metallic": metallic_mean,
        }

        # ── Stage 3: Compute all 10 physics scores ───────────────────────────

        # 1. ECS — Energy Conservation (GTR importance-sampled)
        ecs = self.ecs_scorer(brdf_params)                               # (B, 1)

        # 2. SCS — Specular Consistency
        pixel_coords = sample_output["pixel_coords"].unsqueeze(2)
        observed = F.grid_sample(
            face_image, pixel_coords, mode="bilinear", align_corners=True
        ).squeeze(-1).permute(0, 2, 1)
        obs_mean = observed.mean(dim=1)
        scs = self.scs_scorer(obs_mean, diffuse_mean, specular_mean, sh_light.mean(dim=1))

        # 3. BSS — BRDF Smoothness
        N = albedo_pts.shape[1]
        side = int(math.sqrt(N))
        if side * side == N and side > 1:
            brdf_map = torch.cat([albedo_pts, roughness_pts, metallic_pts], dim=-1
                                  ).reshape(B, side, side, -1).permute(0, 3, 1, 2)
            bss = self.bss_scorer(brdf_map)
        else:
            bss = torch.zeros(B, 1, device=device)

        # 4. ICS — Illumination Coherence (SH-based)
        ics = self.ics_scorer(sh_coeffs)                                 # (B, 1)

        # 5. TPS — Temporal Physics Stability
        tps = self.tps_scorer(temporal_scores) if temporal_scores is not None \
              else torch.zeros(B, 1, device=device)

        # 6. SSS — Subsurface Scattering Consistency
        sss_out = self.sss_model(albedo_pts.reshape(B, -1, 3))
        sss = sss_out["sss_albedo_score"].unsqueeze(-1) \
              + sss_out["sss_ratio_score"].unsqueeze(-1)                 # (B, 1)

        # 7. FAS — Fresnel Anomaly Score
        fresnel_out = self.fresnel_detector(albedo_pts.reshape(B, -1, 3),
                                            metallic_pts.reshape(B, -1, 1),
                                            roughness_pts.reshape(B, -1, 1))
        fas = (fresnel_out["f0_violation"] + fresnel_out["metallic_violation"] +
               fresnel_out["roughness_violation"] + fresnel_out["monotonicity_violation"])

        # 8. WAS — Wasserstein Anomaly (fit on 7 existing scores first)
        scores_7 = torch.cat([ecs, scs, bss, ics, tps, sss, fas], dim=-1)   # (B, 7)
        scores_pad = F.pad(scores_7, (0, 3), value=0)                        # (B, 10)

        if is_real_for_calibration is not None and self.training:
            real_mask = is_real_for_calibration.bool()
            if real_mask.any():
                self.wasserstein_detector.update_real_statistics(
                    scores_pad[real_mask].detach()
                )
        was_out = self.wasserstein_detector(scores_pad)
        was = was_out["mahalanobis_distance"]                                # (B, 1)

        # 9. ShCS — Shadow Consistency Score (NEW)
        shadow_out = self.shadow_scorer(
            face_image=face_image,
            normal_map=normals_pts,
            sh_coefficients=sh_coeffs,
            depth_map=depth_map,
        )
        shcs = shadow_out["shadow_score"]                                    # (B, 1)

        # 10. SpSS — Spectral Skin Score (NEW)
        spectral_out = self.spectral_scorer(
            face_image=face_image,
            albedo=albedo_pts.reshape(B, -1, 3),
        )
        spss = spectral_out["spectral_score"]                                # (B, 1)

        # ── Aggregate all 10 scores ───────────────────────────────────────────
        all_scores = torch.cat([ecs, scs, bss, ics, tps, sss, fas, was, shcs, spss], dim=-1)
        # shape: (B, 10)

        # Normalize and weight
        normalized = self.score_norm(all_scores)
        weights = torch.softmax(self.score_weights, dim=0)
        aggregated = (normalized * weights).sum(dim=-1, keepdim=True)

        # ── Stage 4: Classify ─────────────────────────────────────────────────
        classification = self.classifier(
            face_image=face_image,
            physics_scores=all_scores,
        )

        return {
            # Final predictions
            "logits": classification["logits"],
            "probability": classification["probability"],
            "anomaly_score": classification["anomaly_score"],

            # All 10 physics scores
            "physics_scores": all_scores,
            "score_names": self.SCORE_NAMES,
            "ecs": ecs, "scs": scs, "bss": bss, "ics": ics,
            "tps": tps, "sss": sss, "fas": fas, "was": was,
            "shcs": shcs, "spss": spss,
            "aggregated_physics": aggregated,

            # Material decomposition
            "albedo": albedo_pts,
            "roughness": roughness_pts,
            "metallic": metallic_pts,
            "normals": normals_pts,
            "depth_map": depth_map,
            "sh_coefficients": sh_coeffs,
            "sh_irradiance": sh_light,
            "rendered": rendering["rendered"],

            # Score sub-components
            "estimated_f0": fresnel_out["estimated_f0"],
            "mahalanobis_distance": was_out["mahalanobis_distance"],
            "light_direction": shadow_out["light_direction"],
            "predicted_shadow": shadow_out["predicted_shadow"],
            "c_melanin": spectral_out["c_melanin"],
            "c_hemoglobin": spectral_out["c_hemoglobin"],
            "predicted_rgb": spectral_out["predicted_rgb"],
        }

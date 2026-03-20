"""
Physics Consistency Scoring Module.

Computes forensic scores measuring how well a face image obeys
the laws of physically-based rendering. Real faces produce low scores;
deepfakes produce elevated scores.

Five complementary physics scores:
1. ECS - Energy Conservation Score
2. SCS - Specular Consistency Score
3. BSS - BRDF Smoothness Score
4. ICS - Illumination Coherence Score
5. TPS - Temporal Physics Stability (video only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EnergyConservationScorer(nn.Module):
    """Score 1: Energy Conservation Score (ECS).

    Physics Law: Total reflected energy must not exceed incident energy.
    For a BRDF f_r: integral[f_r * cos(theta) dw] <= 1

    Real faces satisfy this; deepfakes often violate it because neural
    generators don't explicitly enforce energy conservation.
    """

    def __init__(self, num_samples: int = 64):
        super().__init__()
        self.num_samples = num_samples

    def forward(self, brdf_params: dict, normals: torch.Tensor) -> torch.Tensor:
        """
        Estimate energy conservation violation via Monte Carlo integration.

        Args:
            brdf_params: dict with albedo, roughness, metallic
            normals: (B, 3) surface normals

        Returns:
            ecs: (B, 1) energy conservation violation score (0 = perfect, >0 = violation)
        """
        B = normals.shape[0]
        device = normals.device

        # Sample random directions on the hemisphere
        # Use cosine-weighted importance sampling
        u1 = torch.rand(B, self.num_samples, 1, device=device)
        u2 = torch.rand(B, self.num_samples, 1, device=device)

        cos_theta = torch.sqrt(u1)
        sin_theta = torch.sqrt(1.0 - u1)
        phi = 2.0 * math.pi * u2

        # Local frame directions
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = cos_theta
        local_dirs = torch.cat([x, y, z], dim=-1)  # (B, num_samples, 3)

        # Transform to world space using normal as Z-axis
        tangent, bitangent = self._build_frame(normals)
        world_dirs = (
            local_dirs[..., 0:1] * tangent.unsqueeze(1)
            + local_dirs[..., 1:2] * bitangent.unsqueeze(1)
            + local_dirs[..., 2:3] * normals.unsqueeze(1)
        )

        # Evaluate BRDF * cos(theta) for all sampled directions
        albedo = brdf_params["albedo"].unsqueeze(1).expand(-1, self.num_samples, -1)
        roughness = brdf_params["roughness"].unsqueeze(1).expand(-1, self.num_samples, -1)
        metallic = brdf_params["metallic"].unsqueeze(1).expand(-1, self.num_samples, -1)

        # Diffuse contribution (Lambertian)
        diffuse_energy = albedo / math.pi * (1.0 - metallic)

        # Approximate total energy: diffuse integral + specular estimate
        # Diffuse integral of (albedo/pi * (1-m)) * cos(theta) over hemisphere = albedo * (1-m)
        diffuse_total = albedo * (1.0 - metallic)

        # Specular energy upper bound via roughness
        # Smoother surfaces (low roughness) concentrate more energy in specular peak
        # F0 for dielectrics ~ 0.04, for metals ~ albedo
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic
        specular_total = f0  # Approximate specular integral (energy preserving estimate)

        # Total reflected energy estimate
        total_energy = diffuse_total + specular_total  # (B, num_samples, 3)

        # ECS: how much energy exceeds 1.0 (violation)
        violation = F.relu(total_energy - 1.0)
        ecs = violation.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)  # (B, 1)

        return ecs

    def _build_frame(self, normal: torch.Tensor) -> tuple:
        """Build orthonormal frame from normal vector."""
        # Pick a vector not parallel to normal
        up = torch.zeros_like(normal)
        up[..., 1] = 1.0
        mask = (torch.abs(normal[..., 1]) > 0.99).unsqueeze(-1)
        up_alt = torch.zeros_like(normal)
        up_alt[..., 0] = 1.0
        up = torch.where(mask, up_alt, up)

        tangent = F.normalize(torch.cross(up, normal, dim=-1), dim=-1)
        bitangent = F.normalize(torch.cross(normal, tangent, dim=-1), dim=-1)
        return tangent, bitangent


class SpecularConsistencyScorer(nn.Module):
    """Score 2: Specular Consistency Score (SCS).

    Measures how well the observed color decomposes into
    physically plausible diffuse + specular components.

    Real faces have consistent decomposition; deepfakes show residual errors.
    """

    def forward(
        self,
        observed: torch.Tensor,
        diffuse: torch.Tensor,
        specular: torch.Tensor,
        incident_light: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            observed: (B, 3) observed pixel color
            diffuse: (B, 3) estimated diffuse component
            specular: (B, 3) estimated specular component
            incident_light: (B, 3) estimated illumination

        Returns:
            scs: (B, 1) specular consistency violation score
        """
        reconstructed = (diffuse + specular) * incident_light
        residual = (observed - reconstructed) ** 2
        scs = residual.mean(dim=-1, keepdim=True)
        return scs


class BRDFSmoothnessScorer(nn.Module):
    """Score 3: BRDF Smoothness Score (BSS).

    Real skin has spatially smooth material properties.
    Deepfakes often have discontinuous or noisy BRDF parameters
    due to per-pixel generation without physical constraints.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, brdf_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            brdf_map: (B, C, H, W) spatial BRDF parameter maps
                      (albedo, roughness, metallic stacked)

        Returns:
            bss: (B, 1) BRDF smoothness violation score
        """
        # Compute spatial gradients
        dx = brdf_map[:, :, :, 1:] - brdf_map[:, :, :, :-1]
        dy = brdf_map[:, :, 1:, :] - brdf_map[:, :, :-1, :]

        # Total variation as smoothness measure
        tv = torch.abs(dx).mean(dim=(1, 2, 3)) + torch.abs(dy).mean(dim=(1, 2, 3))
        return tv.unsqueeze(-1)


class IlluminationCoherenceScorer(nn.Module):
    """Score 4: Illumination Coherence Score (ICS).

    Real faces are lit by coherent light sources.
    Deepfakes may have spatially inconsistent illumination
    because the generator models appearance, not physics.
    """

    def __init__(self, num_regions: int = 9):
        super().__init__()
        self.num_regions = num_regions  # 3x3 face grid

    def forward(self, illumination_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            illumination_map: (B, 3, H, W) estimated illumination map

        Returns:
            ics: (B, 1) illumination coherence violation score
        """
        B, C, H, W = illumination_map.shape
        grid = int(math.sqrt(self.num_regions))
        rh, rw = H // grid, W // grid

        region_means = []
        for i in range(grid):
            for j in range(grid):
                region = illumination_map[:, :, i * rh:(i + 1) * rh, j * rw:(j + 1) * rw]
                region_means.append(region.mean(dim=(2, 3)))

        region_means = torch.stack(region_means, dim=1)  # (B, num_regions, C)

        # Variance across regions (should be low for coherent lighting)
        ics = region_means.var(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)
        return ics


class TemporalPhysicsStabilityScorer(nn.Module):
    """Score 5: Temporal Physics Stability (TPS). [Video only]

    Physics properties of a real face should be temporally stable
    (BRDF doesn't suddenly change between frames).
    Deepfakes show temporal flickering in physics-decomposed components.
    """

    def forward(self, physics_scores_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_scores_sequence: (B, T, num_scores) physics scores over T frames

        Returns:
            tps: (B, 1) temporal instability score
        """
        if physics_scores_sequence.shape[1] < 2:
            return torch.zeros(physics_scores_sequence.shape[0], 1, device=physics_scores_sequence.device)

        # Temporal differences
        diffs = physics_scores_sequence[:, 1:, :] - physics_scores_sequence[:, :-1, :]
        tps = (diffs ** 2).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        return tps


class PhysicsConsistencyScorer(nn.Module):
    """Aggregates all five physics consistency scores.

    This is the core forensic module that produces the physics-based
    feature vector for deepfake classification.
    """

    def __init__(
        self,
        ecs_weight: float = 1.0,
        scs_weight: float = 1.0,
        bss_weight: float = 0.5,
        ics_weight: float = 0.5,
        tps_weight: float = 0.3,
        num_mc_samples: int = 64,
    ):
        super().__init__()
        self.ecs_scorer = EnergyConservationScorer(num_samples=num_mc_samples)
        self.scs_scorer = SpecularConsistencyScorer()
        self.bss_scorer = BRDFSmoothnessScorer()
        self.ics_scorer = IlluminationCoherenceScorer()
        self.tps_scorer = TemporalPhysicsStabilityScorer()

        self.weights = nn.Parameter(
            torch.tensor([ecs_weight, scs_weight, bss_weight, ics_weight, tps_weight])
        )

        # Learnable normalization for each score
        self.score_norms = nn.BatchNorm1d(5)

    def forward(self, rendering_output: dict, observed: torch.Tensor = None,
                temporal_scores: torch.Tensor = None) -> dict:
        """
        Compute all physics consistency scores.

        Args:
            rendering_output: dict from PBRNeRFBackbone
            observed: (B, N, 3) observed pixel colors (for SCS)
            temporal_scores: (B, T, 4) previous frame scores (for TPS)

        Returns:
            dict with individual scores and aggregated physics feature vector
        """
        B = rendering_output["albedo"].shape[0]
        device = rendering_output["albedo"].device

        # Flatten spatial dims for scoring
        albedo = rendering_output["albedo"].reshape(B, -1, 3).mean(dim=1)
        roughness = rendering_output["roughness"].reshape(B, -1, 1).mean(dim=1)
        metallic = rendering_output["metallic"].reshape(B, -1, 1).mean(dim=1)
        normals = rendering_output["normals"].reshape(B, -1, 3).mean(dim=1)
        normals = F.normalize(normals, dim=-1)
        diffuse = rendering_output["diffuse"].reshape(B, -1, 3).mean(dim=1)
        specular = rendering_output["specular"].reshape(B, -1, 3).mean(dim=1)
        incident_light = rendering_output["incident_light"].reshape(B, -1, 3).mean(dim=1)

        brdf_params = {
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
        }

        # Score 1: Energy Conservation
        ecs = self.ecs_scorer(brdf_params, normals)

        # Score 2: Specular Consistency
        if observed is not None:
            obs_mean = observed.reshape(B, -1, 3).mean(dim=1)
            scs = self.scs_scorer(obs_mean, diffuse, specular, incident_light)
        else:
            scs = torch.zeros(B, 1, device=device)

        # Score 3: BRDF Smoothness (needs spatial maps)
        N = rendering_output["albedo"].shape[1]
        side = int(math.sqrt(N)) if N > 1 else 1
        if side * side == N and side > 1:
            brdf_map = torch.cat([
                rendering_output["albedo"],
                rendering_output["roughness"],
                rendering_output["metallic"],
            ], dim=-1).reshape(B, side, side, -1).permute(0, 3, 1, 2)
            bss = self.bss_scorer(brdf_map)
        else:
            bss = torch.zeros(B, 1, device=device)

        # Score 4: Illumination Coherence
        if side * side == N and side > 2:
            illum_map = rendering_output["incident_light"].reshape(B, side, side, 3).permute(0, 3, 1, 2)
            ics = self.ics_scorer(illum_map)
        else:
            ics = torch.zeros(B, 1, device=device)

        # Score 5: Temporal Stability
        if temporal_scores is not None:
            tps = self.tps_scorer(temporal_scores)
        else:
            tps = torch.zeros(B, 1, device=device)

        # Aggregate
        all_scores = torch.cat([ecs, scs, bss, ics, tps], dim=-1)  # (B, 5)
        normalized_scores = self.score_norms(all_scores)
        weighted_score = (normalized_scores * torch.softmax(self.weights, dim=0)).sum(dim=-1, keepdim=True)

        return {
            "ecs": ecs,
            "scs": scs,
            "bss": bss,
            "ics": ics,
            "tps": tps,
            "all_scores": all_scores,
            "normalized_scores": normalized_scores,
            "aggregated_score": weighted_score,
        }

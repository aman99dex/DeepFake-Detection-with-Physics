"""
PhysForensics: The Complete Model.

End-to-end pipeline combining:
1. Face detection and 3D point sampling
2. PBR-NeRF inverse rendering decomposition
3. Physics consistency scoring
4. Forensic classification

This is the main model class that ties everything together.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .pbr_nerf_backbone import PBRNeRFBackbone
from .physics_scorer import PhysicsConsistencyScorer
from .forensic_classifier import ForensicClassifier


class FacePointSampler(nn.Module):
    """Generates 3D point samples on the face surface from 2D face crops.

    Uses a learnable depth estimator to lift 2D face pixels to 3D,
    creating the input for the PBR-NeRF backbone.
    """

    def __init__(self, image_size: int = 256, num_points: int = 1024):
        super().__init__()
        self.image_size = image_size
        self.num_points = num_points

        # Lightweight depth estimator
        self.depth_net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        # Normal estimator from depth
        self.normal_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, face_image: torch.Tensor) -> dict:
        """
        Args:
            face_image: (B, 3, H, W) face crop

        Returns:
            points: (B, N, 3) 3D surface points
            normals: (B, N, 3) estimated surface normals
            view_dirs: (B, N, 3) viewing directions
            pixel_coords: (B, N, 2) corresponding pixel coordinates
        """
        B, _, H, W = face_image.shape

        # Estimate depth
        depth = self.depth_net(face_image)  # (B, 1, H', W')
        depth_h, depth_w = depth.shape[2], depth.shape[3]

        # Estimate normals from depth
        normals = self.normal_net(depth)  # (B, 3, H', W')
        normals = F.normalize(normals, dim=1)

        # Create pixel grid
        y_coords = torch.linspace(-1, 1, depth_h, device=face_image.device)
        x_coords = torch.linspace(-1, 1, depth_w, device=face_image.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        pixel_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Lift to 3D: (x, y, depth)
        points_3d = torch.cat([
            pixel_grid,
            depth.permute(0, 2, 3, 1),
        ], dim=-1)  # (B, H', W', 3)

        # Flatten and sample
        points_flat = points_3d.reshape(B, -1, 3)
        normals_flat = normals.permute(0, 2, 3, 1).reshape(B, -1, 3)
        pixels_flat = pixel_grid.reshape(B, -1, 2)

        total_points = points_flat.shape[1]
        if total_points > self.num_points:
            indices = torch.randperm(total_points, device=face_image.device)[:self.num_points]
            points_flat = points_flat[:, indices]
            normals_flat = normals_flat[:, indices]
            pixels_flat = pixels_flat[:, indices]

        # View directions (camera at origin looking at face)
        view_dirs = F.normalize(-points_flat, dim=-1)

        return {
            "points": points_flat,
            "normals": normals_flat,
            "view_dirs": view_dirs,
            "pixel_coords": pixels_flat,
            "depth_map": depth,
        }


class PhysForensics(nn.Module):
    """PhysForensics: Physics-Informed NeRF Forensics.

    End-to-end deepfake detection via physically-based inverse rendering.
    """

    def __init__(
        self,
        image_size: int = 256,
        num_sample_points: int = 1024,
        nerf_hidden: int = 256,
        nerf_layers: int = 8,
        neilf_hidden: int = 128,
        neilf_layers: int = 4,
        classifier_hidden: int = 512,
        fusion_type: str = "attention",
        dropout: float = 0.3,
    ):
        super().__init__()

        # Stage 1: Face to 3D points
        self.point_sampler = FacePointSampler(
            image_size=image_size,
            num_points=num_sample_points,
        )

        # Stage 2: PBR-NeRF inverse rendering
        self.pbr_nerf = PBRNeRFBackbone(
            hidden_dim=nerf_hidden,
            num_layers=nerf_layers,
            neilf_hidden=neilf_hidden,
            neilf_layers=neilf_layers,
        )

        # Stage 3: Physics consistency scoring
        self.physics_scorer = PhysicsConsistencyScorer()

        # Stage 4: Forensic classification
        self.classifier = ForensicClassifier(
            visual_feature_dim=256,
            physics_feature_dim=5,
            hidden_dim=classifier_hidden,
            fusion_type=fusion_type,
            dropout=dropout,
        )

    def forward(
        self,
        face_image: torch.Tensor,
        temporal_scores: torch.Tensor = None,
    ) -> dict:
        """
        Full forward pass.

        Args:
            face_image: (B, 3, H, W) aligned face crop
            temporal_scores: (B, T, 5) previous frame physics scores (video mode)

        Returns:
            Complete prediction dictionary with all intermediate results.
        """
        B = face_image.shape[0]

        # Stage 1: Sample 3D points from face
        sample_output = self.point_sampler(face_image)
        points = sample_output["points"]
        normals = sample_output["normals"]
        view_dirs = sample_output["view_dirs"]

        # Stage 2: PBR-NeRF inverse rendering
        rendering_output = self.pbr_nerf(
            points=points,
            view_dirs=view_dirs,
            normals=normals,
        )

        # Sample observed colors from face image at corresponding pixels
        pixel_coords = sample_output["pixel_coords"]  # (B, N, 2)
        grid = pixel_coords.unsqueeze(2)  # (B, N, 1, 2) for grid_sample
        observed_colors = F.grid_sample(
            face_image, grid, mode="bilinear", align_corners=True
        ).squeeze(-1).permute(0, 2, 1)  # (B, N, 3)

        # Stage 3: Physics consistency scoring
        physics_output = self.physics_scorer(
            rendering_output=rendering_output,
            observed=observed_colors,
            temporal_scores=temporal_scores,
        )

        # Stage 4: Forensic classification
        classification = self.classifier(
            face_image=face_image,
            physics_scores=physics_output["all_scores"],
        )

        return {
            # Final output
            "logits": classification["logits"],
            "probability": classification["probability"],
            "anomaly_score": classification["anomaly_score"],

            # Physics scores (for interpretability)
            "physics_scores": physics_output["all_scores"],
            "ecs": physics_output["ecs"],
            "scs": physics_output["scs"],
            "bss": physics_output["bss"],
            "ics": physics_output["ics"],
            "tps": physics_output["tps"],

            # Decomposition (for visualization)
            "albedo": rendering_output["albedo"],
            "roughness": rendering_output["roughness"],
            "metallic": rendering_output["metallic"],
            "normals": rendering_output["normals"],
            "depth_map": sample_output["depth_map"],
            "incident_light": rendering_output["incident_light"],
            "rendered": rendering_output["rendered"],
        }

    def get_physics_map(self, face_image: torch.Tensor) -> torch.Tensor:
        """Generate a spatial physics violation heatmap for visualization.

        Shows WHERE on the face physics violations occur.
        Useful for interpretability and paper figures.
        """
        output = self.forward(face_image)
        # Use aggregated anomaly score as per-pixel map proxy
        return output["anomaly_score"], output["depth_map"]

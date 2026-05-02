"""
Shadow Consistency Module for PhysForensics.

Physical Principle:
    Cast shadows in a real photograph must be geometrically consistent with
    the dominant light source direction. The light direction can be estimated
    from Spherical Harmonics coefficients (l=1 band encodes directionality).
    A face region lit from the left MUST be darker on the right — deepfakes
    frequently violate this because the face is generated with one lighting
    context and composited into another.

Shadow Consistency Score (ShCS):
    1. Estimate dominant light direction from SH l=1 coefficients
    2. Predict shadow likelihood per pixel: P_shadow(x) = max(0, -dot(N(x), L))
    3. Measure actual darkness: D(x) = 1 - luminance(I(x))
    4. Score = 1 - correlation(P_shadow, D) over the face region

    A real face has ShCS ≈ 0 (high correlation).
    A deepfake has ShCS > 0 (poor correlation between predicted and actual shadows).

References:
    - Ramamoorthi & Hanrahan, "On the relationship between radiance and irradiance," 2001
    - Johnson & Farid, "Exposing digital forgeries by detecting inconsistencies
      in lighting," ACM MM 2005
    - Kee et al., "Exposing photo manipulation with inconsistent shadows," TOIT 2014
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ShadowConsistencyScorer(nn.Module):
    """
    Computes a Shadow Consistency Score (ShCS) by comparing predicted shadow
    regions (from SH light direction + normal map) against actual dark regions
    in the input image.

    A low score means shadow placement is physically consistent (real face).
    A high score means shadow placement is physically inconsistent (deepfake).
    """

    def __init__(self, image_size: int = 256, num_normal_bins: int = 8):
        super().__init__()
        self.image_size = image_size
        self.num_normal_bins = num_normal_bins

        # Learnable weight for blending direct + ambient shadow prediction
        self.shadow_mix = nn.Parameter(torch.tensor(0.7))

        # Small CNN to refine shadow prediction from image features
        self.shadow_refiner = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),  # 3 normal channels + 1 depth
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def extract_light_direction(sh_coefficients: torch.Tensor) -> torch.Tensor:
        """
        Extract dominant light direction from SH l=1 band coefficients.

        The l=1 SH band (coefficients 1-3) encodes the first-order
        directional component of lighting:
            L_{1,-1} ∝ y  (up-down)
            L_{1, 0} ∝ z  (front-back)
            L_{1, 1} ∝ x  (left-right)

        The dominant light direction is approximately:
            L = normalize([L_{1,1}, L_{1,-1}, L_{1,0}])

        Args:
            sh_coefficients: (B, 9, 3) SH coefficients

        Returns:
            light_dir: (B, 3) normalized dominant light direction per image
        """
        # Average over color channels for a single direction
        sh_avg = sh_coefficients.mean(dim=-1)  # (B, 9)

        # l=1 band: indices 1, 2, 3
        # Y_{1,1} ∝ x, Y_{1,-1} ∝ y, Y_{1,0} ∝ z
        lx = sh_avg[:, 3]  # Y_{1,1}
        ly = sh_avg[:, 1]  # Y_{1,-1}
        lz = sh_avg[:, 2]  # Y_{1,0}

        light_dir = torch.stack([lx, ly, lz], dim=-1)  # (B, 3)
        light_dir = F.normalize(light_dir + 1e-8, dim=-1)
        return light_dir

    @staticmethod
    def predict_shadow_map(
        normal_map: torch.Tensor,
        light_dir: torch.Tensor,
        image_size: int = 256,
    ) -> torch.Tensor:
        """
        Predict which pixels are in shadow based on normals and light direction.

        Shadow probability: P(x) = max(0, -dot(N(x), L))
        - When normal faces toward light: dot > 0 → lit (no shadow)
        - When normal faces away from light: dot < 0 → shadowed

        Args:
            normal_map: (B, N, 3) surface normals (unit vectors)
            light_dir: (B, 3) dominant light direction
            image_size: spatial resolution for output map

        Returns:
            shadow_map: (B, 1, H, W) predicted shadow probability map
        """
        B, N, _ = normal_map.shape

        # Dot product between normals and light direction
        # Shape: (B, N)
        light_expanded = light_dir.unsqueeze(1).expand_as(normal_map)  # (B, N, 3)
        dot = (normal_map * light_expanded).sum(dim=-1)  # (B, N)

        # Shadow where dot < 0 (facing away from light)
        shadow = F.relu(-dot)  # (B, N)

        # Reshape to spatial map
        side = int(math.sqrt(N))
        if side * side == N and side > 1:
            shadow_map = shadow.reshape(B, 1, side, side)
            if side != image_size:
                shadow_map = F.interpolate(
                    shadow_map,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            # Fallback: create uniform map
            shadow_map = shadow.mean(dim=-1, keepdim=True).unsqueeze(-1).expand(
                B, 1, image_size, image_size
            )

        return shadow_map

    @staticmethod
    def compute_image_darkness(image: torch.Tensor) -> torch.Tensor:
        """
        Compute darkness map from image (1 - luminance).

        Uses perceptual luminance weights:
            Y = 0.299R + 0.587G + 0.114B

        Args:
            image: (B, 3, H, W) face image, values in [-1, 1] or [0, 1]

        Returns:
            darkness: (B, 1, H, W) darkness map in [0, 1]
        """
        # Normalize to [0, 1] if needed
        img = image
        if img.min() < -0.1:
            img = (img + 1.0) / 2.0

        # Perceptual luminance
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        darkness = 1.0 - luminance
        return darkness.clamp(0, 1)

    def compute_shadow_correlation(
        self,
        predicted_shadow: torch.Tensor,
        actual_darkness: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Pearson correlation between predicted shadow map and actual darkness.

        A high positive correlation means shadows are in the right places (real face).
        A low or negative correlation means shadow placement is inconsistent (deepfake).

        Returns:
            inconsistency: (B, 1) score where 0=consistent, 1=inconsistent
        """
        B = predicted_shadow.shape[0]

        # Flatten spatial dimensions
        pred = predicted_shadow.reshape(B, -1)  # (B, H*W)
        actual = actual_darkness.reshape(B, -1)  # (B, H*W)

        # Pearson correlation per sample
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        actual_centered = actual - actual.mean(dim=-1, keepdim=True)

        numerator = (pred_centered * actual_centered).sum(dim=-1)
        denominator = (
            torch.sqrt((pred_centered ** 2).sum(dim=-1) + 1e-8) *
            torch.sqrt((actual_centered ** 2).sum(dim=-1) + 1e-8)
        )
        correlation = numerator / denominator  # (B,) in [-1, 1]

        # Convert to inconsistency score: 0=perfect match, 1=completely inconsistent
        inconsistency = (1.0 - correlation.clamp(-1, 1)) / 2.0
        return inconsistency.unsqueeze(-1)

    def forward(
        self,
        face_image: torch.Tensor,
        normal_map: torch.Tensor,
        sh_coefficients: torch.Tensor,
        depth_map: torch.Tensor = None,
    ) -> dict:
        """
        Compute Shadow Consistency Score.

        Args:
            face_image: (B, 3, H, W) aligned face crop
            normal_map: (B, N, 3) surface normals from PBR-NeRF
            sh_coefficients: (B, 9, 3) SH lighting coefficients
            depth_map: (B, 1, H, W) optional depth map for refinement

        Returns:
            dict with:
                shadow_score: (B, 1) — inconsistency score (higher = more fake)
                predicted_shadow: (B, 1, H, W) — predicted shadow map
                actual_darkness: (B, 1, H, W) — actual image darkness
                light_direction: (B, 3) — estimated dominant light direction
                shadow_correlation: (B, 1) — Pearson correlation (higher = more real)
        """
        B = face_image.shape[0]
        H, W = face_image.shape[2], face_image.shape[3]

        # 1. Extract light direction from SH
        light_dir = self.extract_light_direction(sh_coefficients)  # (B, 3)

        # 2. Predict shadow map from normals + light direction
        predicted_shadow = self.predict_shadow_map(normal_map, light_dir, H)

        # 3. Optionally refine with learned network
        if depth_map is not None:
            norm_spatial = F.interpolate(
                F.normalize(normal_map.mean(dim=1, keepdim=True).expand(B, 3, 1, 1), dim=1),
                size=(H, W), mode="bilinear", align_corners=False,
            ) if normal_map.dim() == 3 else torch.zeros(B, 3, H, W, device=face_image.device)

            # Use depth + normal-like features for refinement
            if depth_map.shape[2] != H or depth_map.shape[3] != W:
                depth_resized = F.interpolate(depth_map, size=(H, W), mode="bilinear", align_corners=False)
            else:
                depth_resized = depth_map

            # Approximate normal spatial map from point cloud mean
            n_mean = F.normalize(normal_map.reshape(B, -1, 3).mean(dim=1), dim=-1)  # (B, 3)
            n_spatial = n_mean.view(B, 3, 1, 1).expand(B, 3, H, W)

            refiner_input = torch.cat([n_spatial, depth_resized], dim=1)  # (B, 4, H, W)
            refined = self.shadow_refiner(refiner_input)  # (B, 1, H, W)

            # Blend geometry-based and learned predictions
            alpha = self.shadow_mix.clamp(0, 1)
            predicted_shadow = alpha * predicted_shadow + (1 - alpha) * refined

        # 4. Compute actual image darkness
        actual_darkness = self.compute_image_darkness(face_image)

        # 5. Compute correlation-based inconsistency
        shadow_score = self.compute_shadow_correlation(predicted_shadow, actual_darkness)

        # 6. Additional directional consistency check:
        #    Light from left → right half of face should be darker
        light_x = light_dir[:, 0]  # positive = light from right
        left_half = actual_darkness[:, :, :, :W // 2].mean(dim=(1, 2, 3))
        right_half = actual_darkness[:, :, :, W // 2:].mean(dim=(1, 2, 3))

        # Expected: if light_x > 0 (from right), left half should be darker
        # Actual sign of darkness difference
        actual_sign = left_half - right_half  # positive if left is darker
        expected_sign = light_x  # positive if light from right (left should be darker)

        directional_violation = F.relu(-actual_sign * expected_sign).unsqueeze(-1)

        # Combined score
        combined = shadow_score * 0.7 + directional_violation.clamp(0, 1) * 0.3

        return {
            "shadow_score": combined,
            "base_shadow_score": shadow_score,
            "directional_violation": directional_violation,
            "predicted_shadow": predicted_shadow,
            "actual_darkness": actual_darkness,
            "light_direction": light_dir,
        }


class ShadowConsistencyLoss(nn.Module):
    """
    Loss function for shadow consistency.

    Enforces that:
    1. Real faces have high shadow correlation (predicted matches actual)
    2. The loss is higher when shadow placement is clearly wrong

    L_shadow = w1 * L_correlation + w2 * L_directional
    """

    def __init__(self, correlation_weight: float = 0.5, directional_weight: float = 0.3):
        super().__init__()
        self.correlation_weight = correlation_weight
        self.directional_weight = directional_weight

    def forward(
        self,
        shadow_output: dict,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            shadow_output: output from ShadowConsistencyScorer.forward()
            labels: (B,) binary labels, 0=real, 1=fake

        Returns:
            loss: scalar loss
        """
        shadow_score = shadow_output["shadow_score"].squeeze(-1)
        directional = shadow_output["directional_violation"].squeeze(-1)

        real_mask = (labels == 0).float()
        fake_mask = (labels == 1).float()

        # Real faces should have LOW shadow inconsistency
        real_shadow_loss = (real_mask * shadow_score).sum() / (real_mask.sum() + 1e-8)

        # Fake faces should ideally have HIGHER shadow inconsistency
        # Soft margin: penalize fakes that sneak below threshold
        fake_shadow_loss = (fake_mask * F.relu(0.3 - shadow_score)).sum() / (fake_mask.sum() + 1e-8)

        correlation_loss = real_shadow_loss + 0.5 * fake_shadow_loss
        directional_loss = (real_mask * directional).sum() / (real_mask.sum() + 1e-8)

        return self.correlation_weight * correlation_loss + self.directional_weight * directional_loss

"""
CLIP Visual Feature Backbone for PhysForensics v2.

Uses a frozen CLIP visual encoder to extract rich semantic features
that complement our physics-based features. This follows the trend
of CVPR 2025 methods (M2F2-Det, D3) that leverage foundation models.

Key insight: CLIP features capture high-level semantic inconsistencies
(e.g., "this face looks AI-generated") while our physics features
capture low-level physical violations. Together they are complementary.

We use CLIP features as an OPTIONAL additional signal, not as the
primary backbone — our physics scores remain the core contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPFeatureExtractor(nn.Module):
    """Extracts CLIP visual features from face crops.

    Uses a lightweight proxy network that mimics CLIP features,
    since the full CLIP model is large. In deployment, this can
    be replaced with actual CLIP if available.
    """

    def __init__(self, output_dim: int = 512, use_pretrained: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.use_pretrained = use_pretrained

        if use_pretrained:
            self._init_clip()
        else:
            self._init_proxy()

    def _init_proxy(self):
        """Lightweight proxy for CLIP features (trainable)."""
        self.encoder = nn.Sequential(
            # Patch embedding (like ViT)
            nn.Conv2d(3, 64, kernel_size=16, stride=16),  # 256->16 patches
            nn.GELU(),
            nn.Flatten(2),  # (B, 64, 256) for 256x256 input
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=256,
                batch_first=True,
                dropout=0.1,
            ),
            num_layers=2,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, self.output_dim),
        )

    def _init_clip(self):
        """Initialize actual CLIP model (requires clip package)."""
        try:
            import clip
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cpu")
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.proj = nn.Linear(512, self.output_dim)
        except ImportError:
            print("CLIP not available, falling back to proxy")
            self._init_proxy()
            self.use_pretrained = False

    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_image: (B, 3, H, W) normalized face crop

        Returns:
            features: (B, output_dim) CLIP-like features
        """
        if self.use_pretrained:
            with torch.no_grad():
                clip_features = self.clip_model.encode_image(face_image).float()
            return self.proj(clip_features)
        else:
            x = self.encoder(face_image)  # (B, 64, num_patches)
            x = x.permute(0, 2, 1)       # (B, num_patches, 64)
            x = self.transformer(x)       # (B, num_patches, 64)
            x = x.mean(dim=1)            # (B, 64) global average
            return self.head(x)           # (B, output_dim)


class MultiScalePhysicsVisualFusion(nn.Module):
    """Fuses physics scores, CLIP features, and PBR decomposition features.

    Three-stream fusion:
    1. Physics stream: 8 physics consistency scores
    2. Visual stream: CLIP-like semantic features
    3. PBR stream: Geometry features from inverse rendering

    Uses gated fusion to learn the optimal combination.
    """

    def __init__(
        self,
        physics_dim: int = 8,
        visual_dim: int = 512,
        pbr_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()

        # Stream projections
        self.physics_proj = nn.Sequential(
            nn.Linear(physics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, output_dim),
            nn.ReLU(),
        )
        self.pbr_proj = nn.Sequential(
            nn.Linear(pbr_dim, output_dim),
            nn.ReLU(),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(3 * output_dim, 3),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.output = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        physics_features: torch.Tensor,
        visual_features: torch.Tensor,
        pbr_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            physics_features: (B, 8) physics scores
            visual_features: (B, visual_dim) CLIP features
            pbr_features: (B, pbr_dim) PBR geometry features

        Returns:
            fused: (B, output_dim)
        """
        p = self.physics_proj(physics_features)
        v = self.visual_proj(visual_features)
        r = self.pbr_proj(pbr_features)

        # Compute gates
        concat = torch.cat([p, v, r], dim=-1)
        gates = self.gate(concat)  # (B, 3)

        # Gated sum
        fused = (
            gates[:, 0:1] * p +
            gates[:, 1:2] * v +
            gates[:, 2:3] * r
        )

        return self.output(fused)

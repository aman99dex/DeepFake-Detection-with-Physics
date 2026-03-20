"""
Forensic Feature Aggregation and Classification Network.

Combines physics consistency scores with learned visual features
to produce the final real/fake classification.

Key design: Attention-based fusion that learns to weight physics
signals differently based on face region and image content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """Cross-attention between physics scores and visual features.

    Allows the model to learn which physics signals are most
    informative for different types of faces/manipulations.
    """

    def __init__(self, physics_dim: int = 5, visual_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.embed_dim = 64
        self.num_heads = num_heads

        self.physics_proj = nn.Linear(physics_dim, self.embed_dim)
        self.visual_proj = nn.Linear(visual_dim, self.embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, physics_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            physics_features: (B, 5) physics consistency scores
            visual_features: (B, N, visual_dim) spatial visual features

        Returns:
            fused: (B, embed_dim) fused feature vector
        """
        # Project physics to query
        physics_q = self.physics_proj(physics_features).unsqueeze(1)  # (B, 1, embed_dim)

        # Project visual features to key/value
        visual_kv = self.visual_proj(visual_features)  # (B, N, embed_dim)

        # Cross attention: physics attends to visual
        attn_out, _ = self.cross_attn(physics_q, visual_kv, visual_kv)
        fused = self.norm(attn_out.squeeze(1) + physics_q.squeeze(1))
        fused = self.output_proj(fused)

        return fused


class ForensicClassifier(nn.Module):
    """Final classification head combining physics and visual features.

    Architecture:
    1. Visual feature extractor (lightweight CNN on face crops)
    2. Cross-attention fusion with physics scores
    3. MLP classification head -> real/fake probability + anomaly score
    """

    def __init__(
        self,
        visual_feature_dim: int = 256,
        physics_feature_dim: int = 5,
        hidden_dim: int = 512,
        fusion_type: str = "attention",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fusion_type = fusion_type

        # Visual feature extractor (lightweight)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, visual_feature_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(visual_feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        if fusion_type == "attention":
            self.fusion = CrossAttentionFusion(
                physics_dim=physics_feature_dim,
                visual_dim=visual_feature_dim,
            )
            classifier_input_dim = 64 + physics_feature_dim
        elif fusion_type == "concat":
            classifier_input_dim = visual_feature_dim + physics_feature_dim
        else:  # mlp
            self.fusion_mlp = nn.Sequential(
                nn.Linear(visual_feature_dim + physics_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_input_dim = hidden_dim

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Anomaly score head (continuous score for interpretability)
        self.anomaly_head = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        face_image: torch.Tensor,
        physics_scores: torch.Tensor,
    ) -> dict:
        """
        Args:
            face_image: (B, 3, H, W) face crop
            physics_scores: (B, 5) physics consistency scores

        Returns:
            logits: (B, 1) classification logits
            probability: (B, 1) real/fake probability
            anomaly_score: (B, 1) continuous anomaly score
        """
        # Extract visual features
        visual_feat = self.visual_encoder(face_image)  # (B, C, 4, 4)
        B, C, H, W = visual_feat.shape

        if self.fusion_type == "attention":
            visual_flat = visual_feat.reshape(B, C, H * W).permute(0, 2, 1)  # (B, 16, C)
            fused = self.fusion(physics_scores, visual_flat)
            combined = torch.cat([fused, physics_scores], dim=-1)
        elif self.fusion_type == "concat":
            visual_pooled = visual_feat.reshape(B, C, -1).mean(dim=-1)
            combined = torch.cat([visual_pooled, physics_scores], dim=-1)
        else:
            visual_pooled = visual_feat.reshape(B, C, -1).mean(dim=-1)
            combined = self.fusion_mlp(torch.cat([visual_pooled, physics_scores], dim=-1))

        logits = self.classifier(combined)
        probability = torch.sigmoid(logits)
        anomaly_score = self.anomaly_head(combined)

        return {
            "logits": logits,
            "probability": probability,
            "anomaly_score": anomaly_score,
        }

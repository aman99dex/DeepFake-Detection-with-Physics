"""
PBR-NeRF Backbone for Inverse Rendering Decomposition.

Decomposes facial images into physically-based components:
- Geometry (SDF-based surface)
- Materials (Disney BRDF: albedo, roughness, metallic)
- Illumination (Neural Incident Light Field)

Based on: Wu et al., "PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields," CVPR 2025
Adapted for facial forensics application.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Fourier positional encoding for NeRF inputs."""

    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        if self.include_input:
            encoded.append(x)
        for freq in self.freqs:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        return torch.cat(encoded, dim=-1)

    def output_dim(self, input_dim: int) -> int:
        d = 2 * self.num_freqs * input_dim
        if self.include_input:
            d += input_dim
        return d


class SDFNetwork(nn.Module):
    """Signed Distance Function network for geometry estimation.

    Outputs SDF values and geometric features for each 3D point.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: list = None,
        pe_freqs: int = 10,
        geometric_init: bool = True,
        init_variance: float = 0.3,
    ):
        super().__init__()
        self.skip_connections = skip_connections or [4]
        self.pe = PositionalEncoding(pe_freqs)
        pe_dim = self.pe.output_dim(input_dim)

        layers = []
        for i in range(num_layers):
            if i == 0:
                in_d = pe_dim
            elif i in self.skip_connections:
                in_d = hidden_dim + pe_dim
            else:
                in_d = hidden_dim

            out_d = hidden_dim if i < num_layers - 1 else hidden_dim + 1  # +1 for SDF

            layer = nn.Linear(in_d, out_d)
            if geometric_init:
                self._geometric_init(layer, i, num_layers, init_variance, pe_dim, hidden_dim)
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.activation = nn.Softplus(beta=100)

    def _geometric_init(self, layer, idx, total, variance, pe_dim, hidden_dim):
        """Initialize weights to approximate a sphere SDF."""
        if idx == total - 1:
            nn.init.normal_(layer.weight[:1], mean=np.sqrt(np.pi) / np.sqrt(hidden_dim), std=0.0001)
            nn.init.constant_(layer.bias[:1], -variance)
        elif idx == 0:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(hidden_dim))
            nn.init.constant_(layer.weight[:, 3:], 0.0)
        else:
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(hidden_dim))

    def forward(self, points: torch.Tensor) -> tuple:
        """
        Args:
            points: (B, 3) 3D coordinates
        Returns:
            sdf: (B, 1) signed distance values
            features: (B, hidden_dim) geometric features
        """
        pe = self.pe(points)
        h = pe
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                h = torch.cat([h, pe], dim=-1)
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)

        sdf = h[:, :1]
        features = h[:, 1:]
        return sdf, features


class DisneyBRDFNetwork(nn.Module):
    """Estimates Disney BRDF parameters from geometric features.

    Outputs:
    - albedo (base color): RGB in [0, 1]
    - roughness: scalar in [0.05, 1.0]
    - metallic: scalar in [0, 1]
    - specular: scalar in [0, 1]
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),  # 3 albedo + roughness + metallic + specular
        )

    def forward(self, features: torch.Tensor) -> dict:
        raw = self.net(features)
        albedo = torch.sigmoid(raw[:, :3])
        roughness = 0.05 + 0.95 * torch.sigmoid(raw[:, 3:4])
        metallic = torch.sigmoid(raw[:, 4:5])
        specular = torch.sigmoid(raw[:, 5:6])
        return {
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "specular": specular,
        }


class NeILF(nn.Module):
    """Neural Incident Light Field.

    Models spatially-varying illumination as a function of position and direction.
    """

    def __init__(
        self,
        pos_dim: int = 3,
        dir_dim: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        pe_freqs_pos: int = 6,
        pe_freqs_dir: int = 4,
    ):
        super().__init__()
        self.pe_pos = PositionalEncoding(pe_freqs_pos)
        self.pe_dir = PositionalEncoding(pe_freqs_dir)
        in_dim = self.pe_pos.output_dim(pos_dim) + self.pe_dir.output_dim(dir_dim)

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 3))  # RGB illumination
        layers.append(nn.Softplus(beta=5))  # Non-negative illumination
        self.net = nn.Sequential(*layers)

    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, 3) surface points
            directions: (B, 3) incident light directions
        Returns:
            illumination: (B, 3) RGB incident radiance
        """
        pe_pos = self.pe_pos(positions)
        pe_dir = self.pe_dir(directions)
        x = torch.cat([pe_pos, pe_dir], dim=-1)
        return self.net(x)


class DisneyBRDF(nn.Module):
    """Evaluates the Disney BRDF for given parameters and directions.

    Implements the simplified Disney principled BRDF with:
    - Lambertian diffuse
    - GGX specular (Cook-Torrance microfacet)
    """

    def forward(
        self,
        brdf_params: dict,
        normal: torch.Tensor,
        view_dir: torch.Tensor,
        light_dir: torch.Tensor,
    ) -> tuple:
        """
        Returns:
            diffuse: (B, 3) diffuse component
            specular: (B, 3) specular component
            total: (B, 3) total BRDF * cos(theta)
        """
        albedo = brdf_params["albedo"]
        roughness = brdf_params["roughness"]
        metallic = brdf_params["metallic"]

        # Half vector
        h = F.normalize(view_dir + light_dir, dim=-1)
        n_dot_l = torch.clamp(torch.sum(normal * light_dir, dim=-1, keepdim=True), 0.0, 1.0)
        n_dot_v = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), 0.0, 1.0)
        n_dot_h = torch.clamp(torch.sum(normal * h, dim=-1, keepdim=True), 0.0, 1.0)
        v_dot_h = torch.clamp(torch.sum(view_dir * h, dim=-1, keepdim=True), 0.0, 1.0)

        # Diffuse (Lambertian)
        f_diffuse = albedo / math.pi * (1.0 - metallic)

        # Specular (Cook-Torrance)
        alpha = roughness ** 2
        D = self._ggx_ndf(n_dot_h, alpha)
        F_val = self._fresnel_schlick(v_dot_h, albedo, metallic)
        G = self._smith_ggx(n_dot_l, n_dot_v, alpha)

        denom = 4.0 * n_dot_l * n_dot_v + 1e-7
        f_specular = D * F_val * G / denom

        diffuse_out = f_diffuse * n_dot_l
        specular_out = f_specular * n_dot_l
        total = diffuse_out + specular_out

        return diffuse_out, specular_out, total

    def _ggx_ndf(self, n_dot_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """GGX/Trowbridge-Reitz Normal Distribution Function."""
        a2 = alpha ** 2
        denom = n_dot_h ** 2 * (a2 - 1.0) + 1.0
        return a2 / (math.pi * denom ** 2 + 1e-7)

    def _fresnel_schlick(
        self, cos_theta: torch.Tensor, albedo: torch.Tensor, metallic: torch.Tensor
    ) -> torch.Tensor:
        """Fresnel-Schlick approximation."""
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic
        return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5

    def _smith_ggx(
        self, n_dot_l: torch.Tensor, n_dot_v: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Smith GGX geometry/masking-shadowing function."""
        a2 = alpha ** 2

        def g1(n_dot_x):
            return 2.0 * n_dot_x / (n_dot_x + torch.sqrt(a2 + (1.0 - a2) * n_dot_x ** 2) + 1e-7)

        return g1(n_dot_l) * g1(n_dot_v)


class PBRNeRFBackbone(nn.Module):
    """Full PBR-NeRF inverse rendering backbone.

    Decomposes facial images into geometry, materials, and illumination,
    then evaluates physics consistency.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_connections: list = None,
        pe_freqs: int = 10,
        dir_pe_freqs: int = 4,
        neilf_hidden: int = 128,
        neilf_layers: int = 4,
    ):
        super().__init__()
        self.sdf_net = SDFNetwork(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            skip_connections=skip_connections or [4],
            pe_freqs=pe_freqs,
        )
        self.brdf_net = DisneyBRDFNetwork(feature_dim=hidden_dim, hidden_dim=hidden_dim // 2)
        self.neilf = NeILF(hidden_dim=neilf_hidden, num_layers=neilf_layers)
        self.disney_brdf = DisneyBRDF()

    def forward(
        self,
        points: torch.Tensor,
        view_dirs: torch.Tensor,
        light_dirs: torch.Tensor = None,
        normals: torch.Tensor = None,
    ) -> dict:
        """
        Full forward pass: geometry -> materials -> illumination -> rendering.

        Args:
            points: (B, N, 3) 3D surface points
            view_dirs: (B, N, 3) viewing directions
            light_dirs: (B, N, 3) light directions (estimated if None)
            normals: (B, N, 3) surface normals (computed from SDF if None)

        Returns:
            Dictionary with all decomposed components and rendered output.
        """
        B, N, _ = points.shape
        pts_flat = points.reshape(-1, 3)
        vd_flat = view_dirs.reshape(-1, 3)

        # 1. Geometry: SDF + features
        sdf, geo_features = self.sdf_net(pts_flat)

        # 2. Compute normals from SDF gradient
        if normals is None:
            normals_flat = self._compute_normals(pts_flat)
        else:
            normals_flat = normals.reshape(-1, 3)

        # 3. Materials: Disney BRDF parameters
        brdf_params = self.brdf_net(geo_features)

        # 4. Illumination
        if light_dirs is None:
            light_dirs_flat = self._estimate_light_dirs(pts_flat, normals_flat)
        else:
            light_dirs_flat = light_dirs.reshape(-1, 3)

        incident_light = self.neilf(pts_flat, light_dirs_flat)

        # 5. PBR Rendering
        diffuse, specular, brdf_total = self.disney_brdf(
            brdf_params, normals_flat, vd_flat, light_dirs_flat
        )
        rendered = brdf_total * incident_light

        return {
            "sdf": sdf.reshape(B, N, 1),
            "normals": normals_flat.reshape(B, N, 3),
            "albedo": brdf_params["albedo"].reshape(B, N, 3),
            "roughness": brdf_params["roughness"].reshape(B, N, 1),
            "metallic": brdf_params["metallic"].reshape(B, N, 1),
            "specular_param": brdf_params["specular"].reshape(B, N, 1),
            "incident_light": incident_light.reshape(B, N, 3),
            "diffuse": diffuse.reshape(B, N, 3),
            "specular": specular.reshape(B, N, 3),
            "rendered": rendered.reshape(B, N, 3),
            "brdf_total": brdf_total.reshape(B, N, 3),
            "geo_features": geo_features.reshape(B, N, -1),
        }

    def _compute_normals(self, points: torch.Tensor) -> torch.Tensor:
        """Compute surface normals via SDF gradient."""
        points.requires_grad_(True)
        sdf, _ = self.sdf_net(points)
        grad = torch.autograd.grad(
            sdf.sum(), points, create_graph=True, retain_graph=True
        )[0]
        return F.normalize(grad, dim=-1)

    def _estimate_light_dirs(
        self, points: torch.Tensor, normals: torch.Tensor
    ) -> torch.Tensor:
        """Estimate dominant light direction (simplified: use normal + perturbation)."""
        # In practice, this would use the NeILF to find the dominant light direction
        # Simplified: reflect view around normal with random perturbation
        noise = torch.randn_like(normals) * 0.1
        light_dirs = F.normalize(normals + noise, dim=-1)
        return light_dirs

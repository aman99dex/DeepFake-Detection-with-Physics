"""
Advanced Physics Modules for PhysForensics v2.

Major mathematical improvements:
1. Subsurface Scattering (BSSRDF) Consistency — Jensen dipole model
2. Spherical Harmonics Lighting Decomposition — 2nd-order SH (9 coefficients)
3. Generalized Trowbridge-Reitz (GTR) with importance sampling
4. Skin-specific BRDF model with measured scattering parameters
5. Fresnel-based specular reflectance anomaly detection

References:
- Jensen et al., "A Practical Model for Subsurface Light Transport," SIGGRAPH 2001
- Ramamoorthi & Hanrahan, "On the relationship between radiance and irradiance," SIGGRAPH 2001
- Burley, "Physically-Based Shading at Disney," SIGGRAPH 2012
- Sengupta et al., "SfSNet: Learning Shape, Reflectance and Illuminance," TPAMI 2020
- Zhu et al., "Face Forgery Detection by 3D Decomposition," CVPR 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 1. SPHERICAL HARMONICS LIGHTING MODEL
# ============================================================================

class SphericalHarmonicsLighting(nn.Module):
    """Second-order Spherical Harmonics lighting model for faces.

    Irradiance from a convex Lambertian object can be approximated
    using SH up to order 2 (9 coefficients per color channel = 27 total).

    E(n) = sum_{l=0}^{2} sum_{m=-l}^{l} L_{lm} * Y_{lm}(n)

    where:
    - L_{lm} are the SH lighting coefficients (what we estimate)
    - Y_{lm}(n) are the SH basis functions evaluated at normal n
    - E(n) is the irradiance at surface normal n

    This is mathematically more principled than NeILF for face lighting
    because faces are approximately convex Lambertian surfaces where
    the SH approximation is provably accurate.
    """

    def __init__(self):
        super().__init__()
        # Learnable SH coefficients: 9 basis * 3 channels = 27 params
        # Initialized to approximate frontal lighting
        self.sh_coefficients = nn.Parameter(torch.zeros(9, 3))
        nn.init.normal_(self.sh_coefficients, mean=0.0, std=0.1)
        # Set DC component (ambient) to reasonable default
        self.sh_coefficients.data[0] = torch.tensor([0.5, 0.5, 0.5])

    @staticmethod
    def evaluate_sh_basis(normals: torch.Tensor) -> torch.Tensor:
        """Evaluate 2nd-order SH basis functions at given normals.

        The 9 real spherical harmonic basis functions Y_{lm} for l=0,1,2:

        Y_00 = 1/(2*sqrt(pi))                                  [constant]
        Y_1m1 = sqrt(3/(4*pi)) * y                             [linear]
        Y_10  = sqrt(3/(4*pi)) * z
        Y_11  = sqrt(3/(4*pi)) * x
        Y_2m2 = sqrt(15/(4*pi)) * x*y                          [quadratic]
        Y_2m1 = sqrt(15/(4*pi)) * y*z
        Y_20  = sqrt(5/(16*pi)) * (3*z^2 - 1)
        Y_21  = sqrt(15/(4*pi)) * x*z
        Y_22  = sqrt(15/(16*pi)) * (x^2 - y^2)

        Args:
            normals: (..., 3) surface normals (must be unit length)

        Returns:
            sh_basis: (..., 9) SH basis values
        """
        x = normals[..., 0:1]
        y = normals[..., 1:2]
        z = normals[..., 2:3]

        # Constants
        c0 = 1.0 / (2.0 * math.sqrt(math.pi))           # 0.2821
        c1 = math.sqrt(3.0 / (4.0 * math.pi))            # 0.4886
        c2 = math.sqrt(15.0 / (4.0 * math.pi))           # 1.0925
        c3 = math.sqrt(5.0 / (16.0 * math.pi))           # 0.3154
        c4 = math.sqrt(15.0 / (16.0 * math.pi))          # 0.5463

        # Band 0 (l=0): 1 basis function
        y00 = torch.ones_like(x) * c0

        # Band 1 (l=1): 3 basis functions
        y1m1 = c1 * y
        y10 = c1 * z
        y11 = c1 * x

        # Band 2 (l=2): 5 basis functions
        y2m2 = c2 * x * y
        y2m1 = c2 * y * z
        y20 = c3 * (3.0 * z * z - 1.0)
        y21 = c2 * x * z
        y22 = c4 * (x * x - y * y)

        return torch.cat([y00, y1m1, y10, y11, y2m2, y2m1, y20, y21, y22], dim=-1)

    def forward(self, normals: torch.Tensor) -> torch.Tensor:
        """Compute irradiance from SH lighting at given normals.

        Args:
            normals: (B, N, 3) surface normals

        Returns:
            irradiance: (B, N, 3) RGB irradiance values
        """
        sh_basis = self.evaluate_sh_basis(normals)  # (B, N, 9)
        # irradiance = SH_basis @ SH_coefficients -> (B, N, 3)
        irradiance = torch.matmul(sh_basis, self.sh_coefficients)
        # Ensure non-negative irradiance
        irradiance = F.softplus(irradiance, beta=5)
        return irradiance

    def get_lighting_consistency(self, normals: torch.Tensor) -> torch.Tensor:
        """Measure lighting consistency via SH coefficient analysis.

        Real faces have smooth, low-frequency lighting well-captured by SH.
        Returns a consistency score (lower = more consistent).
        """
        # Ratio of higher-order (l=2) energy to total energy
        # Real faces: most energy in l=0,1. Deepfakes: noisy high-order.
        l0_energy = (self.sh_coefficients[0:1] ** 2).sum()
        l1_energy = (self.sh_coefficients[1:4] ** 2).sum()
        l2_energy = (self.sh_coefficients[4:9] ** 2).sum()

        total_energy = l0_energy + l1_energy + l2_energy + 1e-8
        high_order_ratio = l2_energy / total_energy

        return high_order_ratio


class AdaptiveSHLightingEstimator(nn.Module):
    """Estimates per-image SH lighting coefficients from face images.

    Uses a CNN to regress 27 SH coefficients (9 basis * 3 channels)
    from a face crop. Based on SfSNet architecture.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 27),  # 9 SH basis * 3 channels
        )

    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_image: (B, 3, H, W)

        Returns:
            sh_coefficients: (B, 9, 3) per-image SH lighting coefficients
        """
        coeffs = self.encoder(face_image)  # (B, 27)
        return coeffs.reshape(-1, 9, 3)


# ============================================================================
# 2. SUBSURFACE SCATTERING (BSSRDF) CONSISTENCY MODEL
# ============================================================================

class SubsurfaceScatteringModel(nn.Module):
    """Jensen Dipole Approximation for skin subsurface scattering.

    Human skin exhibits strong subsurface scattering (SSS):
    light enters at one point and exits at a nearby point.
    This creates the characteristic soft, translucent look of skin.

    The BSSRDF is: S(x_i, w_i; x_o, w_o) = S_d(x_i, x_o) * F_t(x_i) * F_t(x_o) / pi

    Where S_d is the dipole diffusion profile:
    S_d(r) = (alpha' / (4*pi)) * [z_r * (sigma_tr + 1/d_r) * exp(-sigma_tr * d_r) / d_r^2
                                  + z_v * (sigma_tr + 1/d_v) * exp(-sigma_tr * d_v) / d_v^2]

    With:
    - r: distance between entry and exit points
    - sigma_tr = sqrt(3 * sigma_a * sigma_t'): effective transport coefficient
    - sigma_a: absorption coefficient
    - sigma_s': reduced scattering coefficient
    - sigma_t' = sigma_a + sigma_s': reduced extinction coefficient
    - z_r = 1/sigma_t': depth of real source
    - z_v = z_r + 4*A*D: depth of virtual source (D = 1/(3*sigma_t'))
    - A: internal reflection parameter from Fresnel

    Deepfakes do NOT model SSS -> they violate the expected
    diffusion profile of real skin.
    """

    # Measured skin optical parameters (from Jensen et al. 2001)
    # Units: mm^-1, for RGB wavelengths (680nm, 550nm, 450nm)
    SKIN_SIGMA_A = torch.tensor([0.032, 0.17, 0.48])    # Absorption
    SKIN_SIGMA_S_PRIME = torch.tensor([0.74, 0.88, 1.01])  # Reduced scattering

    def __init__(self, eta: float = 1.3):
        """
        Args:
            eta: Index of refraction for skin (~1.3-1.5)
        """
        super().__init__()
        self.eta = eta

        # Register measured parameters as buffers
        self.register_buffer("sigma_a", self.SKIN_SIGMA_A)
        self.register_buffer("sigma_s_prime", self.SKIN_SIGMA_S_PRIME)

        # Derived quantities
        sigma_t_prime = self.sigma_a + self.sigma_s_prime
        self.register_buffer("sigma_t_prime", sigma_t_prime)

        sigma_tr = torch.sqrt(3.0 * self.sigma_a * sigma_t_prime)
        self.register_buffer("sigma_tr", sigma_tr)

        # Diffusion coefficient: D = 1 / (3 * sigma_t')
        D = 1.0 / (3.0 * sigma_t_prime)
        self.register_buffer("D", D)

        # Internal reflection parameter A
        A = self._compute_A(eta)
        self.register_buffer("A_param", torch.tensor(A))

        # Source depths
        z_r = 1.0 / sigma_t_prime
        z_v = z_r + 4.0 * A * D
        self.register_buffer("z_r", z_r)
        self.register_buffer("z_v", z_v)

        # Reduced albedo
        alpha_prime = self.sigma_s_prime / sigma_t_prime
        self.register_buffer("alpha_prime", alpha_prime)

    def _compute_A(self, eta: float) -> float:
        """Compute internal reflection parameter A from Fresnel.

        A = (1 + F_dr) / (1 - F_dr)

        where F_dr is the average diffuse Fresnel reflectance.
        Using the approximate formula from Egan & Hilgeman 1973.
        """
        F_dr = -1.440 / (eta ** 2) + 0.710 / eta + 0.668 + 0.0636 * eta
        return (1.0 + F_dr) / (1.0 - F_dr)

    def dipole_profile(self, r: torch.Tensor) -> torch.Tensor:
        """Evaluate the dipole diffusion profile S_d(r).

        Args:
            r: (...) distance between entry and exit points (mm)

        Returns:
            S_d: (..., 3) RGB diffusion profile values
        """
        r = r.unsqueeze(-1).expand(*r.shape, 3)  # (..., 3)
        r = torch.clamp(r, min=1e-6)  # Avoid division by zero

        # Distances to real and virtual sources
        d_r = torch.sqrt(r ** 2 + self.z_r ** 2)
        d_v = torch.sqrt(r ** 2 + self.z_v ** 2)

        # Dipole profile
        term_r = self.z_r * (self.sigma_tr + 1.0 / d_r) * torch.exp(-self.sigma_tr * d_r) / (d_r ** 2)
        term_v = self.z_v * (self.sigma_tr + 1.0 / d_v) * torch.exp(-self.sigma_tr * d_v) / (d_v ** 2)

        S_d = (self.alpha_prime / (4.0 * math.pi)) * (term_r + term_v)
        return S_d

    def expected_diffusion_width(self) -> torch.Tensor:
        """Compute expected diffusion radius for skin.

        The mean free path: l_d = 1/sigma_tr gives the characteristic
        scattering distance. Real skin should show scattering within
        this radius; deepfakes won't match this profile.
        """
        return 1.0 / self.sigma_tr  # RGB mean free paths

    def forward(self, albedo: torch.Tensor, pixel_distances: torch.Tensor = None) -> dict:
        """Compute SSS consistency features.

        Args:
            albedo: (B, N, 3) estimated surface albedo
            pixel_distances: (B, N, N) pairwise distances between surface points (mm)

        Returns:
            dict with SSS consistency features
        """
        B, N, _ = albedo.shape

        # Feature 1: Albedo vs expected skin scattering ratio
        # Real skin: albedo correlates with sigma_s' (more scattering = lighter skin)
        # sigma_s' / sigma_t' gives the reduced albedo
        expected_albedo_ratio = self.alpha_prime.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        albedo_deviation = torch.abs(albedo - expected_albedo_ratio)
        sss_albedo_score = albedo_deviation.mean(dim=(1, 2), keepdim=False)  # (B,)

        # Feature 2: Color channel ratio consistency
        # Skin SSS creates specific RGB ratios due to wavelength-dependent scattering
        # Red scatters more than blue (longer mean free path)
        if albedo.shape[-1] == 3:
            r, g, b = albedo[..., 0], albedo[..., 1], albedo[..., 2]
            # Expected ratio: R > G > B for skin (due to blood absorption)
            rg_ratio = r / (g + 1e-6)
            gb_ratio = g / (b + 1e-6)
            # Real skin: rg_ratio > 1, gb_ratio > 1
            ratio_violation = F.relu(1.0 - rg_ratio) + F.relu(1.0 - gb_ratio)
            sss_ratio_score = ratio_violation.mean(dim=-1)  # (B,)
        else:
            sss_ratio_score = torch.zeros(B, device=albedo.device)

        # Feature 3: Translucency indicator
        # High albedo + low roughness = translucent appearance (expected for skin)
        # Very dark or very metallic regions violate skin assumptions
        translucency = albedo.mean(dim=-1).mean(dim=-1)  # (B,)

        return {
            "sss_albedo_score": sss_albedo_score.unsqueeze(-1),
            "sss_ratio_score": sss_ratio_score.unsqueeze(-1),
            "sss_translucency": translucency.unsqueeze(-1),
            "expected_mfp": self.expected_diffusion_width(),
        }


# ============================================================================
# 3. ADVANCED GTR IMPORTANCE SAMPLING FOR ENERGY CONSERVATION
# ============================================================================

class GTRImportanceSampler(nn.Module):
    """Generalized Trowbridge-Reitz importance sampling.

    D_GTR(theta_h, alpha, gamma) = c / (alpha^2 * cos^2(theta_h) + sin^2(theta_h))^gamma

    For gamma=2: standard GGX/Trowbridge-Reitz
    For gamma=1: Berry distribution (wider tails)

    We use gamma=2 (GGX) with VNDF sampling for reduced variance
    in the energy conservation estimation.

    VNDF sampling (Heitz 2018):
    Instead of sampling D(h), sample D_v(h) = G1(v) * max(v.h, 0) * D(h) / (n.v)
    This reduces variance significantly for grazing angles.
    """

    def __init__(self, num_samples: int = 128):
        super().__init__()
        self.num_samples = num_samples

    def sample_ggx_vndf(
        self,
        view_dir: torch.Tensor,
        roughness: torch.Tensor,
        num_samples: int = None,
    ) -> tuple:
        """Sample the GGX distribution of visible normals (VNDF).

        Implementation of Heitz 2018 "Sampling the GGX Distribution
        of Visible Normals".

        Args:
            view_dir: (B, 3) viewing direction (in tangent space, z=up)
            roughness: (B, 1) surface roughness

        Returns:
            sampled_h: (B, num_samples, 3) sampled half-vectors
            pdf: (B, num_samples) probability density of each sample
        """
        if num_samples is None:
            num_samples = self.num_samples

        B = view_dir.shape[0]
        device = view_dir.device
        alpha = roughness.squeeze(-1)  # (B,)

        # Step 1: Stretch the view vector
        v_stretched = F.normalize(
            torch.stack([alpha * view_dir[:, 0], alpha * view_dir[:, 1], view_dir[:, 2]], dim=-1),
            dim=-1,
        )

        # Step 2: Build orthonormal basis (differentiable, no in-place ops)
        up_z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, -1)
        up_x = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, -1)
        mask = (v_stretched[:, 2].abs() < 0.999).unsqueeze(-1)  # (B, 1)
        up = torch.where(mask, up_z, up_x)

        T1_raw = torch.cross(up, v_stretched, dim=-1)
        T1 = F.normalize(T1_raw + 1e-8, dim=-1)
        T2 = torch.cross(v_stretched, T1, dim=-1)

        # Step 3: Sample points on unit disk
        u1 = torch.rand(B, num_samples, device=device)
        u2 = torch.rand(B, num_samples, device=device)

        r = torch.sqrt(u1)
        phi = 2.0 * math.pi * u2
        t1 = r * torch.cos(phi)
        t2 = r * torch.sin(phi)

        # Step 4: Project onto hemisphere
        s = 0.5 * (1.0 + v_stretched[:, 2:3])  # (B, 1)
        t2 = (1.0 - s) * torch.sqrt(torch.clamp(1.0 - t1 ** 2, 0, 1)) + s * t2

        # Step 5: Compute normal in tangent space
        n_h = (
            t1.unsqueeze(-1) * T1.unsqueeze(1)
            + t2.unsqueeze(-1) * T2.unsqueeze(1)
            + torch.sqrt(torch.clamp(1.0 - t1 ** 2 - t2 ** 2, 0, 1)).unsqueeze(-1) * v_stretched.unsqueeze(1)
        )

        # Step 6: Unstretch
        sampled_h = F.normalize(
            torch.stack([
                alpha.unsqueeze(1) * n_h[..., 0],
                alpha.unsqueeze(1) * n_h[..., 1],
                torch.clamp(n_h[..., 2], min=0.0),
            ], dim=-1),
            dim=-1,
        )

        # Compute PDF
        n_dot_h = torch.clamp(sampled_h[..., 2], 0.0, 1.0)
        v_dot_h = torch.clamp((view_dir.unsqueeze(1) * sampled_h).sum(dim=-1), 0.0, 1.0)

        a2 = (alpha ** 2).unsqueeze(1)
        denom = n_dot_h ** 2 * (a2 - 1.0) + 1.0
        D = a2 / (math.pi * denom ** 2 + 1e-10)

        # Smith G1
        n_dot_v = torch.clamp(view_dir[:, 2:3], 0.001, 1.0)
        G1 = 2.0 * n_dot_v / (n_dot_v + torch.sqrt(a2 + (1.0 - a2) * n_dot_v ** 2) + 1e-7)

        pdf = D * G1 / (4.0 * n_dot_v + 1e-7)

        return sampled_h, pdf

    def compute_energy_integral(
        self,
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        view_dir: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute the energy conservation integral using VNDF importance sampling.

        integral[f_r(w_i, w_o) * cos(theta_i) dw_i] should be <= 1

        With importance sampling via VNDF:
        E = (1/N) * sum_k [ f_r(w_k) * cos(theta_k) / pdf(w_k) ]

        Args:
            albedo: (B, 3)
            roughness: (B, 1)
            metallic: (B, 1)
            view_dir: (B, 3) viewing direction (default: normal direction)

        Returns:
            energy: (B, 3) total reflected energy per channel
        """
        B = albedo.shape[0]
        device = albedo.device

        if view_dir is None:
            view_dir = torch.tensor([0.0, 0.0, 1.0], device=device).expand(B, -1)

        # Sample half-vectors via VNDF
        sampled_h, pdf = self.sample_ggx_vndf(view_dir, roughness, self.num_samples)

        # Compute light directions from half-vectors (reflection of view around h)
        v = view_dir.unsqueeze(1)  # (B, 1, 3)
        v_dot_h = (v * sampled_h).sum(dim=-1, keepdim=True)
        light_dirs = 2.0 * v_dot_h * sampled_h - v  # (B, N, 3)

        cos_theta_i = torch.clamp(light_dirs[..., 2], 0.0, 1.0)

        # Evaluate BRDF
        # Fresnel
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic  # (B, 3)
        F_val = f0.unsqueeze(1) + (1.0 - f0.unsqueeze(1)) * (1.0 - v_dot_h) ** 5

        # Diffuse energy
        diffuse = albedo.unsqueeze(1) / math.pi * (1.0 - metallic.unsqueeze(1))

        # For specular: using importance sampling, the MC estimate simplifies
        # to F * G2 / G1 (the split-sum approximation)
        n_dot_v = torch.clamp(view_dir[:, 2:3], 0.001, 1.0).unsqueeze(1)
        n_dot_l = cos_theta_i.unsqueeze(-1)
        alpha = roughness.unsqueeze(1) ** 2

        G1_v = 2.0 * n_dot_v / (n_dot_v + torch.sqrt(alpha + (1.0 - alpha) * n_dot_v ** 2) + 1e-7)
        G1_l = 2.0 * n_dot_l / (n_dot_l + torch.sqrt(alpha + (1.0 - alpha) * n_dot_l ** 2) + 1e-7)
        G2 = G1_v * G1_l

        specular_contribution = F_val * G2 / (G1_v + 1e-7)  # (B, N, 3)

        # Total energy = diffuse integral + specular MC estimate
        diffuse_energy = albedo * (1.0 - metallic)  # Analytical diffuse integral
        specular_energy = specular_contribution.mean(dim=1)  # MC average

        total_energy = diffuse_energy + specular_energy
        return total_energy


# ============================================================================
# 4. WASSERSTEIN ANOMALY DETECTOR
# ============================================================================

class WassersteinAnomalyDetector(nn.Module):
    """Detects physics anomalies using Wasserstein distance.

    Instead of binary classification, we measure how far the
    physics feature distribution of a test sample is from the
    distribution of known real faces.

    The 1-Wasserstein distance between two 1D distributions:
    W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx

    For multivariate features, we use the sliced Wasserstein distance:
    SW_p(P, Q) = (E_theta [W_p^p(theta#P, theta#Q)])^(1/p)

    where theta#P is the 1D projection of P onto direction theta.
    """

    def __init__(self, feature_dim: int = 8, num_projections: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_projections = num_projections

        # Running statistics of real face physics features
        self.register_buffer("real_mean", torch.zeros(feature_dim))
        self.register_buffer("real_cov", torch.eye(feature_dim))
        self.register_buffer("real_count", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

        # Random projection directions for sliced Wasserstein
        projections = torch.randn(num_projections, feature_dim)
        projections = F.normalize(projections, dim=-1)
        self.register_buffer("projections", projections)

    @torch.no_grad()
    def update_real_statistics(self, physics_features: torch.Tensor):
        """Update running statistics from real face physics features.

        Args:
            physics_features: (N, feature_dim) physics features from known real faces
        """
        N = physics_features.shape[0]
        batch_mean = physics_features.mean(dim=0)
        batch_cov = (physics_features - batch_mean).T @ (physics_features - batch_mean) / max(N - 1, 1)

        if not self.initialized:
            self.real_mean.copy_(batch_mean)
            self.real_cov.copy_(batch_cov)
            self.real_count.fill_(N)
            self.initialized.fill_(True)
        else:
            # Welford's online algorithm for mean and covariance
            old_count = self.real_count.item()
            new_count = old_count + N
            delta = batch_mean - self.real_mean
            self.real_mean += delta * N / new_count
            self.real_cov = (
                (old_count - 1) * self.real_cov + (N - 1) * batch_cov
                + (old_count * N / new_count) * torch.outer(delta, delta)
            ) / (new_count - 1)
            self.real_count.fill_(new_count)

    def mahalanobis_distance(self, physics_features: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance from real face distribution.

        D_M(x) = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))

        This is the natural anomaly score under a Gaussian assumption.

        Args:
            physics_features: (B, feature_dim)

        Returns:
            distance: (B, 1) Mahalanobis distance
        """
        if not self.initialized:
            return torch.zeros(physics_features.shape[0], 1, device=physics_features.device)

        delta = physics_features - self.real_mean.unsqueeze(0)
        # Regularized inverse covariance
        cov_reg = self.real_cov + 1e-4 * torch.eye(self.feature_dim, device=physics_features.device)
        cov_inv = torch.linalg.inv(cov_reg)
        mahal = torch.sqrt(torch.clamp(
            (delta @ cov_inv * delta).sum(dim=-1),
            min=0.0,
        ))
        return mahal.unsqueeze(-1)

    def sliced_wasserstein_distance(
        self, features_a: torch.Tensor, features_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute sliced Wasserstein-1 distance between two feature sets.

        Args:
            features_a: (N, feature_dim)
            features_b: (M, feature_dim)

        Returns:
            distance: scalar
        """
        # Project onto random directions
        proj_a = features_a @ self.projections.T  # (N, num_proj)
        proj_b = features_b @ self.projections.T  # (M, num_proj)

        # Sort projections
        proj_a_sorted, _ = torch.sort(proj_a, dim=0)
        proj_b_sorted, _ = torch.sort(proj_b, dim=0)

        # Interpolate to same size if needed
        if proj_a_sorted.shape[0] != proj_b_sorted.shape[0]:
            n = max(proj_a_sorted.shape[0], proj_b_sorted.shape[0])
            proj_a_sorted = F.interpolate(
                proj_a_sorted.T.unsqueeze(0), size=n, mode="linear", align_corners=True
            ).squeeze(0).T
            proj_b_sorted = F.interpolate(
                proj_b_sorted.T.unsqueeze(0), size=n, mode="linear", align_corners=True
            ).squeeze(0).T

        # W1 distance = mean absolute difference of sorted projections
        return (proj_a_sorted - proj_b_sorted).abs().mean()

    def forward(self, physics_features: torch.Tensor) -> dict:
        """Compute anomaly scores.

        Args:
            physics_features: (B, feature_dim)

        Returns:
            dict with anomaly scores
        """
        mahal = self.mahalanobis_distance(physics_features)

        # Convert to probability via sigmoid-like transform
        anomaly_prob = 1.0 - torch.exp(-0.5 * mahal ** 2)

        return {
            "mahalanobis_distance": mahal,
            "anomaly_probability": anomaly_prob,
        }


# ============================================================================
# 5. FRESNEL ANOMALY DETECTOR
# ============================================================================

class FresnelAnomalyDetector(nn.Module):
    """Detects Fresnel reflectance violations.

    The Fresnel equations describe how light is reflected at
    material boundaries. For dielectric materials (like skin):

    F_0 = ((n1 - n2) / (n1 + n2))^2

    For skin: n_skin ~ 1.4, n_air = 1.0
    F_0_skin ~ ((1.4 - 1.0) / (1.4 + 1.0))^2 ~ 0.028

    This is a HARD PHYSICAL CONSTRAINT. If the estimated F0
    deviates significantly from this range, the face is likely fake.

    The Fresnel-Schlick approximation:
    F(theta) = F_0 + (1 - F_0) * (1 - cos(theta))^5

    Real skin follows this curve precisely. Deepfakes don't.
    """

    # Physical constants for human skin
    SKIN_IOR_RANGE = (1.35, 1.55)  # Index of refraction range for skin
    SKIN_F0_RANGE = (0.020, 0.048)  # Corresponding F0 range

    def __init__(self):
        super().__init__()
        self.f0_min = self.SKIN_F0_RANGE[0]
        self.f0_max = self.SKIN_F0_RANGE[1]

    def estimate_f0(self, albedo: torch.Tensor, metallic: torch.Tensor) -> torch.Tensor:
        """Estimate F0 from material parameters.

        F0 = 0.04 * (1 - metallic) + albedo * metallic

        For non-metallic skin: F0 ~ 0.04 (dielectric)
        """
        return 0.04 * (1.0 - metallic) + albedo * metallic

    def fresnel_schlick(self, cos_theta: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """Evaluate Fresnel-Schlick approximation."""
        return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5

    def forward(
        self,
        albedo: torch.Tensor,
        metallic: torch.Tensor,
        roughness: torch.Tensor,
    ) -> dict:
        """Compute Fresnel anomaly features.

        Args:
            albedo: (B, 3) or (B, N, 3)
            metallic: (B, 1) or (B, N, 1)
            roughness: (B, 1) or (B, N, 1)

        Returns:
            dict with Fresnel anomaly scores
        """
        # Flatten to (B, 3) and (B, 1) if spatial
        if albedo.dim() == 3:
            albedo = albedo.mean(dim=1)
            metallic = metallic.mean(dim=1)
            roughness = roughness.mean(dim=1)

        f0 = self.estimate_f0(albedo, metallic)

        # Score 1: F0 range violation
        # Real skin F0 should be in [0.02, 0.048] for non-metallic parts
        f0_mean = f0.mean(dim=-1, keepdim=True)
        f0_violation = (
            F.relu(self.f0_min - f0_mean) +
            F.relu(f0_mean - self.f0_max)
        )

        # Score 2: Metallic violation
        # Skin should have very low metallic value
        metallic_violation = F.relu(metallic - 0.1)

        # Score 3: Roughness plausibility
        # Real skin roughness ~ 0.3-0.7
        roughness_violation = (
            F.relu(0.2 - roughness) +
            F.relu(roughness - 0.8)
        )

        # Score 4: Fresnel curve consistency at multiple angles
        angles = torch.tensor([0.0, 30.0, 60.0, 80.0], device=albedo.device) * math.pi / 180.0
        cos_thetas = torch.cos(angles)
        fresnel_values = []
        for ct in cos_thetas:
            fv = self.fresnel_schlick(ct, f0_mean)
            fresnel_values.append(fv)
        fresnel_stack = torch.stack(fresnel_values, dim=1)  # (B, 4, 1)

        # Check monotonicity: F should increase as angle increases (cos decreases)
        diffs = fresnel_stack[:, 1:] - fresnel_stack[:, :-1]
        # Should be negative (F increases as cos_theta decreases)
        monotonicity_violation = F.relu(diffs).mean(dim=(1, 2), keepdim=False).unsqueeze(-1)

        return {
            "f0_violation": f0_violation,
            "metallic_violation": metallic_violation,
            "roughness_violation": roughness_violation,
            "monotonicity_violation": monotonicity_violation,
            "estimated_f0": f0_mean,
        }

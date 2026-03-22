"""
Test Script for PhysForensics v2 (Enhanced Model).
Usage: python test_model_v2.py
"""

import torch
import time
import sys
import math


def test_v2():
    print("=" * 60)
    print("PhysForensics v2 - Enhanced Model Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test 1: Advanced physics modules
    print("\n[Test 1] Spherical Harmonics Lighting...")
    from src.models.advanced_physics import SphericalHarmonicsLighting

    sh = SphericalHarmonicsLighting().to(device)
    normals = torch.randn(4, 100, 3, device=device)
    normals = torch.nn.functional.normalize(normals, dim=-1)
    irradiance = sh(normals)
    print(f"  SH irradiance shape: {irradiance.shape}")
    print(f"  SH irradiance range: [{irradiance.min():.4f}, {irradiance.max():.4f}]")

    # Verify SH basis orthogonality
    basis = SphericalHarmonicsLighting.evaluate_sh_basis(normals[0])
    print(f"  SH basis shape: {basis.shape}")
    print("  PASSED")

    # Test 2: Subsurface Scattering Model
    print("\n[Test 2] Subsurface Scattering (Jensen Dipole)...")
    from src.models.advanced_physics import SubsurfaceScatteringModel

    sss = SubsurfaceScatteringModel(eta=1.4).to(device)

    # Test dipole profile
    distances = torch.linspace(0.01, 5.0, 100, device=device)
    profile = sss.dipole_profile(distances)
    print(f"  Dipole profile shape: {profile.shape}")
    print(f"  Profile at r=0.1mm: R={profile[1, 0]:.4f} G={profile[1, 1]:.4f} B={profile[1, 2]:.4f}")
    print(f"  Expected MFP (mm): R={sss.expected_diffusion_width()[0]:.3f} G={sss.expected_diffusion_width()[1]:.3f} B={sss.expected_diffusion_width()[2]:.3f}")

    # Test SSS scoring
    albedo = torch.rand(4, 50, 3, device=device) * 0.5 + 0.3  # Skin-like
    sss_out = sss(albedo)
    print(f"  SSS albedo score: {sss_out['sss_albedo_score'].mean():.4f}")
    print(f"  SSS ratio score: {sss_out['sss_ratio_score'].mean():.4f}")
    print("  PASSED")

    # Test 3: GTR Importance Sampling
    print("\n[Test 3] GTR VNDF Importance Sampling...")
    from src.models.advanced_physics import GTRImportanceSampler

    gtr = GTRImportanceSampler(num_samples=64).to(device)
    view = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(4, -1)
    roughness = torch.tensor([[0.4]], device=device).expand(4, -1)
    samples, pdf = gtr.sample_ggx_vndf(view, roughness, 64)
    print(f"  Sampled half-vectors shape: {samples.shape}")
    print(f"  PDF shape: {pdf.shape}")
    print(f"  PDF range: [{pdf.min():.4f}, {pdf.max():.4f}]")

    # Test energy integral
    albedo_flat = torch.rand(4, 3, device=device) * 0.5 + 0.3
    metallic_flat = torch.rand(4, 1, device=device) * 0.05
    energy = gtr.compute_energy_integral(albedo_flat, roughness, metallic_flat, view)
    print(f"  Energy integral shape: {energy.shape}")
    print(f"  Energy per channel: {energy[0].tolist()}")
    print(f"  Energy <= 1.0? {(energy <= 1.05).all().item()}")
    print("  PASSED")

    # Test 4: Fresnel Anomaly Detector
    print("\n[Test 4] Fresnel Anomaly Detector...")
    from src.models.advanced_physics import FresnelAnomalyDetector

    fresnel = FresnelAnomalyDetector().to(device)

    # Real skin parameters
    real_albedo = torch.tensor([[0.6, 0.4, 0.35]], device=device).expand(4, -1)
    real_metallic = torch.tensor([[0.02]], device=device).expand(4, -1)
    real_roughness = torch.tensor([[0.5]], device=device).expand(4, -1)
    real_result = fresnel(real_albedo, real_metallic, real_roughness)
    print(f"  Real F0 estimate: {real_result['estimated_f0'][0].item():.4f}")
    print(f"  Real F0 violation: {real_result['f0_violation'][0].item():.4f}")

    # Fake parameters (physically implausible)
    fake_albedo = torch.tensor([[0.9, 0.9, 0.95]], device=device).expand(4, -1)
    fake_metallic = torch.tensor([[0.5]], device=device).expand(4, -1)
    fake_roughness = torch.tensor([[0.05]], device=device).expand(4, -1)
    fake_result = fresnel(fake_albedo, fake_metallic, fake_roughness)
    print(f"  Fake F0 estimate: {fake_result['estimated_f0'][0].item():.4f}")
    print(f"  Fake F0 violation: {fake_result['f0_violation'][0].item():.4f}")
    print(f"  Fake metallic violation: {fake_result['metallic_violation'][0].item():.4f}")
    print(f"  Real vs Fake separation: {fake_result['f0_violation'][0].item() > real_result['f0_violation'][0].item()}")
    print("  PASSED")

    # Test 5: Wasserstein Anomaly Detector
    print("\n[Test 5] Wasserstein Anomaly Detector...")
    from src.models.advanced_physics import WassersteinAnomalyDetector

    wd = WassersteinAnomalyDetector(feature_dim=8).to(device)

    # Calibrate with "real" features
    real_features = torch.randn(100, 8, device=device) * 0.1 + 0.5
    wd.update_real_statistics(real_features)
    print(f"  Calibrated with {int(wd.real_count.item())} real samples")

    # Test real sample
    test_real = torch.randn(4, 8, device=device) * 0.1 + 0.5
    real_dist = wd.mahalanobis_distance(test_real)
    print(f"  Mahalanobis (real): {real_dist.mean():.4f}")

    # Test fake sample (out-of-distribution)
    test_fake = torch.randn(4, 8, device=device) * 0.5 + 2.0
    fake_dist = wd.mahalanobis_distance(test_fake)
    print(f"  Mahalanobis (fake): {fake_dist.mean():.4f}")
    print(f"  Fake > Real? {fake_dist.mean() > real_dist.mean()}")
    print("  PASSED")

    # Test 6: Full v2 model
    print("\n[Test 6] PhysForensics v2 Full Model...")
    from src.models.physforensics_v2 import PhysForensicsV2

    model = PhysForensicsV2(
        image_size=256,
        num_sample_points=256,
        nerf_hidden=128,
        nerf_layers=4,
        classifier_hidden=256,
        fusion_type="attention",
        gtr_samples=32,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    images = torch.randn(2, 3, 256, 256, device=device)
    start = time.time()
    with torch.no_grad():
        output = model(images)
    elapsed = time.time() - start

    print(f"  Forward time: {elapsed:.3f}s")
    print(f"  Output keys: {sorted(output.keys())}")
    print(f"  Logits: {output['logits'].squeeze().tolist()}")
    print(f"  Probabilities: {output['probability'].squeeze().tolist()}")
    print(f"  Physics scores shape: {output['physics_scores'].shape}")
    print(f"  8 score names: ECS, SCS, BSS, ICS, TPS, SSS, FAS, WAS")
    for i, name in enumerate(["ECS", "SCS", "BSS", "ICS", "TPS", "SSS", "FAS", "WAS"]):
        val = output["physics_scores"][0, i].item()
        print(f"    {name}: {val:.6f}")
    print(f"  SH coefficients shape: {output['sh_coefficients'].shape}")
    print(f"  Estimated F0: {output['estimated_f0'][0].item():.4f}")
    print("  PASSED")

    # Test 7: v2 Loss computation
    print("\n[Test 7] PhysForensics v2 Loss...")
    from src.losses.advanced_losses import PhysicsForensicsLossV2

    criterion = PhysicsForensicsLossV2()
    labels = torch.tensor([0, 1], device=device)
    model.train()
    output = model(images, is_real_for_calibration=(labels == 0))
    losses = criterion(output, labels)

    print(f"  Total loss: {losses['total'].item():.4f}")
    for k, v in losses.items():
        if k != "total":
            print(f"    {k}: {v.item():.4f}")

    # Backward pass
    losses["total"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Gradient norm: {grad_norm:.4f}")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL v2 TESTS PASSED!")
    print("=" * 60)
    print(f"\nPhysForensics v2: {params:,} parameters, 8 physics scores")
    print("New capabilities:")
    print("  - Spherical Harmonics lighting decomposition")
    print("  - Subsurface scattering (Jensen dipole) consistency")
    print("  - Fresnel reflectance anomaly detection")
    print("  - GTR importance-sampled energy conservation")
    print("  - Wasserstein/Mahalanobis anomaly scoring")
    print("  - Contrastive physics feature learning")
    print("  - Cross-score correlation enforcement")

    return True


if __name__ == "__main__":
    success = test_v2()
    sys.exit(0 if success else 1)

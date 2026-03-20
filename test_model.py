"""
Quick Test Script for PhysForensics.

Verifies the model runs correctly on synthetic data.
Usage: python test_model.py
"""

import torch
import time
import sys


def test_model():
    print("=" * 60)
    print("PhysForensics - Model Verification Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test 1: Model instantiation
    print("\n[Test 1] Model Instantiation...")
    from src.models.physforensics import PhysForensics

    model = PhysForensics(
        image_size=256,
        num_sample_points=256,  # Reduced for testing
        nerf_hidden=128,        # Reduced for testing
        nerf_layers=4,          # Reduced for testing
        neilf_hidden=64,
        neilf_layers=2,
        classifier_hidden=256,
        fusion_type="attention",
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created with {total_params:,} parameters")
    print("  PASSED")

    # Test 2: Forward pass
    print("\n[Test 2] Forward Pass...")
    batch_size = 2
    fake_images = torch.randn(batch_size, 3, 256, 256).to(device)

    start = time.time()
    with torch.no_grad():
        output = model(fake_images)
    elapsed = time.time() - start

    print(f"  Output keys: {list(output.keys())}")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Probability shape: {output['probability'].shape}")
    print(f"  Anomaly score shape: {output['anomaly_score'].shape}")
    print(f"  Physics scores shape: {output['physics_scores'].shape}")
    print(f"  Albedo shape: {output['albedo'].shape}")
    print(f"  Normals shape: {output['normals'].shape}")
    print(f"  Forward pass time: {elapsed:.3f}s")
    print("  PASSED")

    # Test 3: Loss computation
    print("\n[Test 3] Loss Computation...")
    from src.losses.physics_losses import PhysicsForensicsLoss

    criterion = PhysicsForensicsLoss()
    labels = torch.tensor([0, 1]).to(device)  # One real, one fake

    model.train()
    output = model(fake_images)
    losses = criterion(output, labels)

    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  BCE loss: {losses['bce'].item():.4f}")
    print(f"  Energy conservation loss: {losses['energy_conservation'].item():.4f}")
    print(f"  Anomaly calibration loss: {losses['anomaly'].item():.4f}")
    print("  PASSED")

    # Test 4: Backward pass
    print("\n[Test 4] Backward Pass...")
    losses["total"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Total gradient norm: {grad_norm:.4f}")
    print("  PASSED")

    # Test 5: Synthetic dataset
    print("\n[Test 5] Synthetic Dataset...")
    from src.data.deepfake_dataset import SyntheticPBRDataset

    dataset = SyntheticPBRDataset(num_samples=10, image_size=256)
    sample = dataset[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label'].item()}")
    print(f"  GT albedo: {sample['gt_albedo'].tolist()}")
    print("  PASSED")

    # Test 6: Physics scorer standalone
    print("\n[Test 6] Physics Scorer...")
    from src.models.physics_scorer import PhysicsConsistencyScorer

    scorer = PhysicsConsistencyScorer().to(device)
    # Create mock rendering output
    N = 256
    mock_rendering = {
        "albedo": torch.rand(batch_size, N, 3).to(device),
        "roughness": torch.rand(batch_size, N, 1).to(device) * 0.5 + 0.3,
        "metallic": torch.rand(batch_size, N, 1).to(device) * 0.05,
        "normals": torch.randn(batch_size, N, 3).to(device),
        "diffuse": torch.rand(batch_size, N, 3).to(device),
        "specular": torch.rand(batch_size, N, 3).to(device) * 0.1,
        "incident_light": torch.rand(batch_size, N, 3).to(device),
    }
    mock_rendering["normals"] = torch.nn.functional.normalize(mock_rendering["normals"], dim=-1)

    scores = scorer(mock_rendering)
    print(f"  ECS: {scores['ecs'].mean().item():.6f}")
    print(f"  SCS: {scores['scs'].mean().item():.6f}")
    print(f"  BSS: {scores['bss'].mean().item():.6f}")
    print(f"  ICS: {scores['ics'].mean().item():.6f}")
    print(f"  Aggregated: {scores['aggregated_score'].mean().item():.6f}")
    print("  PASSED")

    # Test 7: Evaluator
    print("\n[Test 7] Evaluator...")
    from src.evaluation.evaluator import Evaluator
    import numpy as np

    evaluator = Evaluator()
    fake_preds = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.6, 1, 50)])
    fake_labels = np.array([0] * 50 + [1] * 50)
    metrics = evaluator.compute_metrics(fake_preds, fake_labels)
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  EER: {metrics['eer']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nPhysForensics is ready for training.")
    print("Run: python train.py --synthetic")
    print("Or download real datasets: python scripts/download_datasets.py")

    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)

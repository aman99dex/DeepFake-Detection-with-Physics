# PhysForensics: Deepfake Detection via Physics-Based Neural Inverse Rendering

> **Detecting deepfakes by checking if faces obey the laws of physics.**

## Key Idea

Real faces follow physical laws of light transport. Deepfakes don't.

We decompose facial images into geometry, materials, and illumination using PBR-NeRF inverse rendering, then measure physics violations to detect fakes. This approach generalizes across unseen generators because **physics is universal**.

## Architecture

```
Face Image -> 3D Point Sampling -> PBR-NeRF Inverse Rendering
                                          |
                          +---------------+---------------+
                          |               |               |
                       Geometry       Materials      Illumination
                       (SDF)        (Disney BRDF)     (NeILF)
                          |               |               |
                          +-------+-------+-------+-------+
                                  |
                    Physics Consistency Scoring
                    (5 scores: ECS, SCS, BSS, ICS, TPS)
                                  |
                    Cross-Attention Fusion + Classifier
                                  |
                          Real / Fake + Anomaly Score
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run model tests
python test_model.py

# Train on synthetic data (for testing)
python train.py --synthetic --epochs 10

# Train on real datasets (after downloading)
python train.py --config configs/default.yaml

# Inference
python inference.py --image path/to/face.jpg
python inference.py --dir path/to/images/
```

## Dataset Setup

```bash
# Set up dataset directories and download instructions
python scripts/download_datasets.py
```

Supported datasets:
- **FaceForensics++** (1000 real + 4000 manipulated videos)
- **CelebDF-v2** (590 real + 5639 fake celebrity videos)
- **DFDC** (119,197 video clips)
- **Synthetic PBR** (controlled ground-truth, auto-generated)

## Physics Consistency Scores

| Score | Full Name | What It Measures |
|-------|-----------|-----------------|
| **ECS** | Energy Conservation Score | Does reflected energy exceed incident energy? |
| **SCS** | Specular Consistency Score | Can observed color decompose into plausible diffuse + specular? |
| **BSS** | BRDF Smoothness Score | Are material properties spatially smooth (like real skin)? |
| **ICS** | Illumination Coherence Score | Is the face lit by coherent illumination? |
| **TPS** | Temporal Physics Stability | Do physics properties flicker between frames? (video) |

## Project Structure

```
PBR/
├── src/
│   ├── models/
│   │   ├── physforensics.py       # Main model
│   │   ├── pbr_nerf_backbone.py   # PBR-NeRF inverse rendering
│   │   ├── physics_scorer.py      # 5 physics consistency scores
│   │   └── forensic_classifier.py # Cross-attention fusion classifier
│   ├── losses/
│   │   └── physics_losses.py      # Physics-informed loss functions
│   ├── data/
│   │   ├── deepfake_dataset.py    # Dataset classes
│   │   └── face_processor.py      # Face detection and alignment
│   ├── evaluation/
│   │   └── evaluator.py           # Metrics and benchmarking
│   └── utils/
│       └── visualization.py       # Physics map visualizations
├── configs/
│   └── default.yaml               # Training configuration
├── scripts/
│   └── download_datasets.py       # Dataset download helper
├── train.py                       # Training script
├── test_model.py                  # Model verification tests
├── inference.py                   # Run detection on images/video
├── RESEARCH.md                    # Detailed research documentation
├── REPORT.md                      # Publication-ready research report
└── requirements.txt               # Python dependencies
```

## Novel Contributions

1. **First work** applying full PBR-NeRF inverse rendering to deepfake detection
2. **Five physics consistency scores** as forensic features
3. **Cross-attention fusion** of physics and visual signals
4. **Generator-agnostic** detection via universal physics laws
5. **Interpretable** — shows WHERE physics violations occur

## Key References

- Wu et al., "PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields," CVPR 2025
- "Light2Lie: Detecting Deepfake Images Using Physical Reflectance Laws," NDSS 2026
- Rossler et al., "FaceForensics++," ICCV 2019
- Mildenhall et al., "NeRF," ECCV 2020

## License

Research use only. See individual dataset licenses for data usage terms.

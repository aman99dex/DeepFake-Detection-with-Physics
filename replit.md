# PhysForensics

## Overview
PhysForensics is a deepfake detection research project that uses physics-based neural inverse rendering to detect AI-generated fake faces. Real human faces obey strict physical laws (energy conservation, Fresnel reflectance, microfacet BRDFs, subsurface scattering); deepfakes systematically violate these laws in ways invisible to humans but detectable by the model.

## Architecture
- **PBR-NeRF Backbone** (`src/models/pbr_nerf_backbone.py`) — SDF + Disney BRDF + Neural Incident Light Field (NeILF) inverse rendering
- **Physics Consistency Scorer** (`src/models/physics_scorer.py`) — 5 physics scores: ECS, SCS, BSS, ICS, TPS
- **Forensic Classifier** (`src/models/forensic_classifier.py`) — Cross-attention fusion of visual + physics features
- **Advanced Physics v2** (`src/models/advanced_physics.py`) — SH lighting, SSS, GTR BRDF, Wasserstein loss
- **CLIP Backbone** (`src/models/clip_backbone.py`) — Visual feature extraction with gated fusion

## Results
- **AUC-ROC**: 0.9662 (real deepfake data)
- **Accuracy**: 90.67%
- **F1 Score**: 0.9104
- **EER**: 9.95%

## Web Dashboard
A Flask-based dashboard (`app.py`) serves at port 5000. It displays:
- Key performance metrics
- Visualizations (ROC curves, training curves, physics score distributions)
- Training log table
- Architecture & usage information

## Entry Points
- `app.py` — Flask web dashboard (port 5000)
- `train.py` — Training script (`python train.py --config configs/default.yaml`)
- `train_real.py` — Training on real deepfake data
- `inference.py` — Run inference on images/videos/directories
- `test_model.py` — Model unit tests (v1)
- `test_model_v2.py` — Model unit tests (v2)

## Key Directories
- `src/` — Core model, data, losses, evaluation, utils
- `configs/` — YAML training configurations
- `outputs/` — Checkpoints, logs, visualizations
- `outputs/checkpoints/best_model.pt` — Trained model checkpoint
- `outputs/real_data/` — Results from training on real deepfake data
- `scripts/` — Dataset download helpers
- `docs/` — Documentation and images
- `templates/` — Flask HTML templates

## Dependencies
- Python 3.12
- Flask (web dashboard)
- PyTorch, torchvision (deep learning)
- OpenCV, Pillow (image processing)
- scikit-learn, scipy (evaluation)
- matplotlib (visualization)
- einops, kornia, timm (model utilities)

## Workflow
- **Start application** — `python app.py` on port 5000 (webview)

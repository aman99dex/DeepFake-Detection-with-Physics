# PhysForensics

## Overview
PhysForensics is a deepfake detection research project using physics-based neural inverse rendering. Real faces obey strict physical laws (energy conservation, Fresnel reflectance, microfacet BRDFs, subsurface scattering); deepfakes systematically violate these laws. The model detects this with up to 10 complementary physics consistency scores.

## Architecture — 10-Score Model (v2 Extended)

| Score | Name | Physical Law |
|-------|------|-------------|
| ECS | Energy Conservation | Reflected energy ≤ incident (GTR VNDF sampling) |
| SCS | Specular Consistency | GGX NDF predicts specular highlight shape |
| BSS | BRDF Smoothness | Real skin materials vary smoothly in space |
| ICS | Illumination Coherence | Lighting energy in SH l=0,1 bands, not l=2 |
| TPS | Temporal Physics Stability | Physics stable across video frames |
| SSS | Subsurface Scattering | Skin R > G > B (Jensen dipole model) |
| FAS | Fresnel Anomaly | Skin F₀ ≈ 0.028 (IOR n=1.4) |
| WAS | Wasserstein Anomaly | Mahalanobis distance from real physics distribution |
| ShCS | Shadow Consistency | Cast shadows align with SH light direction (**NEW**) |
| SpSS | Spectral Skin | Color explainable by melanin + hemoglobin concentrations (**NEW**) |

## Key Files

### Models
- `src/models/physforensics.py` — v1 (5 scores)
- `src/models/physforensics_v2.py` — v2 (8 scores)
- `src/models/physforensics_v2_extended.py` — v2 Extended (10 scores) **← recommended**
- `src/models/pbr_nerf_backbone.py` — PBR-NeRF: SDF + Disney BRDF + NeILF
- `src/models/physics_scorer.py` — ECS, SCS, BSS, ICS, TPS scorers
- `src/models/advanced_physics.py` — SH lighting, SSS, GTR, Fresnel, Wasserstein
- `src/models/shadow_consistency.py` — Shadow Consistency scorer (ShCS) **NEW**
- `src/models/spectral_skin.py` — Spectral Skin model (SpSS) **NEW**
- `src/models/forensic_classifier.py` — Cross-attention physics×visual fusion
- `src/models/clip_backbone.py` — CLIP feature extraction

### Losses
- `src/losses/physics_losses.py` — v1 loss (5 terms)
- `src/losses/advanced_losses.py` — v2 loss (8 terms)
- `src/losses/extended_losses.py` — v2 Extended loss (10 terms) **NEW**

### Training Scripts
- `train_v2.py` — Best-practice training with AMP, grad accumulation, warmup **← use this**
- `train_real.py` — v1/v2 training on real data
- `train.py` — Original training script

### Data & Evaluation
- `src/data/deepfake_dataset.py` — Dataset loader (FF++, CelebDF, DFDC, HuggingFace, synthetic)
- `src/data/face_processor.py` — Face detection, alignment, video processing (MTCNN)
- `src/evaluation/evaluator.py` — AUC, EER, F1, cross-dataset benchmarking
- `src/utils/visualization.py` — ROC curves, physics heatmaps, training plots

### Scripts
- `scripts/prepare_data.py` — Download free HuggingFace data + process any manual datasets **NEW**
- `scripts/cross_dataset_eval.py` — Cross-dataset generalization evaluation **NEW**
- `scripts/download_real_datasets.py` — HuggingFace download helper
- `scripts/download_datasets.py` — FF++/CelebDF/DFDC setup helper

### Web Dashboard
- `app.py` — Flask server (port 5000)
- `templates/index.html` — Results dashboard
- `templates/guide.html` — Training guide with step-by-step instructions **NEW**

### Outputs
- `outputs/checkpoints/best_model.pt` — v1 trained checkpoint
- `outputs/real_data/` — v1 results on real deepfake data (AUC=0.9662)
- `outputs/v2_extended/` — v2 Extended results (populated after training)

## Quick Start

```bash
# 1. Get data (free, auto-download ~120K images)
python scripts/prepare_data.py

# 2. Train v2 Extended (10-score model)
python train_v2.py --data data/processed/unified --epochs 30

# 3. Cross-dataset evaluation
python scripts/cross_dataset_eval.py --model v2_extended

# 4. View dashboard
python app.py   # → http://localhost:5000
```

## Dataset Download Links
- **Free (auto):** HuggingFace `pujanpaudel/deepfake_face_classification`, `OpenRL/DeepFakeFace`
- **FF++:** https://github.com/ondyari/FaceForensics (request form, free for research)
- **CelebDF-v2:** https://github.com/yuezunli/celeb-deepfakeforensics
- **DFDC:** https://kaggle.com/competitions/deepfake-detection-challenge

## Results (v1, trained on real diffusion fakes)
- AUC-ROC: 0.9662
- Accuracy: 90.67%
- F1: 0.9104
- EER: 9.95%

## Web Dashboard
- **Start workflow:** `python app.py` on port 5000 (webview)
- **Routes:** `/` (dashboard), `/guide` (training guide), `/api/results`, `/api/status`

## Dependencies
Python 3.12, Flask, PyTorch, torchvision, scikit-learn, scipy, matplotlib, einops, kornia, timm

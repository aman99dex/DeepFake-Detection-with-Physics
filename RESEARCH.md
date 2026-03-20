# PhysForensics: Physics-Informed Neural Radiance Field Forensics for Deepfake Detection

## Research Overview

**Project Title:** PhysForensics — Detecting Deepfakes via Physically-Based Neural Inverse Rendering Consistency Analysis

**Core Thesis:** Real human faces in video strictly obey the physics of light transport (energy conservation, Fresnel reflectance, microfacet BRDF models). Deepfakes — even state-of-the-art ones — are synthesized by neural networks that do NOT explicitly enforce these physical constraints. By performing physics-based inverse rendering on facial regions and measuring deviations from physical laws, we can detect deepfakes with high accuracy and unprecedented generalizability.

**Target Venues:** CVPR 2027, ICCV 2027, ECCV 2026, NeurIPS 2026, or NDSS 2027

---

## 1. Literature Review

### 1.1 Neural Radiance Fields (NeRF)
- **NeRF (Mildenhall et al., ECCV 2020):** Encodes a scene as a continuous volumetric function mapping 3D coordinates to color and density via an MLP. Achieves photorealistic novel view synthesis.
- **Instant-NGP (Muller et al., SIGGRAPH 2022):** Hash-grid acceleration for real-time NeRF training.
- **3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023):** Point-based rendering alternative achieving real-time performance.

### 1.2 Physics-Based Inverse Rendering with NeRF
- **PBR-NeRF (Wu et al., CVPR 2025):** The foundational work for our approach. Decomposes scenes into:
  - **Geometry:** SDF-based surface representation
  - **Materials:** Disney BRDF (albedo, roughness, metallic, specular)
  - **Illumination:** Neural Incident Light Field (NeILF)
  - **Key Innovation:** Two physics-based losses:
    1. **Conservation of Energy Loss (L_CoE):** Ensures total reflected energy <= incident energy
    2. **NDF-weighted Specular Loss (L_spec):** Correctly separates specular from diffuse reflections using the Normal Distribution Function
  - Code: https://github.com/s3anwu/pbrnerf

- **Neural Microfacet Fields (Maiorca et al., 2024):** Inverse rendering with microfacet BRDF for glossy surfaces.
- **RFBR-IR (2025):** Regularized frequency BRDF reconstruction for stable inverse rendering.

### 1.3 Physics-Based Deepfake Detection
- **Light2Lie (NDSS 2026):** Physics-augmented detection using specular reflectance scores, normal maps, and base reflectance F0. Demonstrates that physics-based features generalize across unseen generators.
- **LIDeepDet (2024):** Image decomposition + lighting analysis for detection.
- **Face X-ray (Li et al., CVPR 2020):** Detects blending boundaries rather than content.

### 1.4 Traditional Deepfake Detection
- **CNN-based:** EfficientNet, XceptionNet, ResNet classifiers on face crops.
- **Frequency-domain:** F3-Net, SRM filters for spectral analysis.
- **Temporal:** FTCN, LipForensics for video-level detection.
- **Limitation:** These methods overfit to specific generation artifacts and fail on unseen generators.

### 1.5 Gap in Literature (Our Contribution)
**No existing work combines full PBR-NeRF inverse rendering (geometry + materials + illumination decomposition) with deepfake detection.** Light2Lie uses simplified reflectance estimation. We propose using the full physically-based rendering pipeline to extract rich forensic features.

---

## 2. Novel Contribution: PhysForensics

### 2.1 Key Insight
When PBR-NeRF fits a real face, the physics-based losses (energy conservation, specular separation) converge to small values because real faces obey physics. When fitting a deepfake face, these losses remain elevated because the deepfake's pixel values encode physically inconsistent light transport.

### 2.2 Architecture Overview

```
Input Video/Image
       |
       v
[Face Detection & Alignment] (MTCNN / RetinaFace)
       |
       v
[Multi-View Face Reconstruction] (or single-view with priors)
       |
       v
[PBR-NeRF Inverse Rendering Engine]
   |          |           |
   v          v           v
Geometry   Materials   Illumination
 (SDF)    (Disney BRDF)  (NeILF)
   |          |           |
   v          v           v
[Physics Consistency Scoring Module]
   - Energy Conservation Score (ECS)
   - Specular Consistency Score (SCS)
   - BRDF Smoothness Score (BSS)
   - Illumination Coherence Score (ICS)
   - Temporal Physics Stability (TPS) [video only]
       |
       v
[Forensic Feature Aggregation Network]
   - Physics scores + learned features
   - Multi-scale fusion
       |
       v
[Binary Classification: Real vs Fake]
   + Anomaly Score (continuous)
```

### 2.3 Mathematical Formulation

#### 2.3.1 PBR Rendering Equation
For surface point **x** with view direction **w_o**, the outgoing radiance:

```
L_o(x, w_o) = integral over hemisphere [f_r(x, w_i, w_o) * L_i(x, w_i) * (n . w_i) dw_i]
```

Where f_r is the Disney BRDF:
```
f_r = f_diffuse + f_specular
f_diffuse = (albedo / pi) * (1 - metallic)
f_specular = D(h) * F(w_o, h) * G(w_i, w_o, h) / (4 * (n . w_i) * (n . w_o))
```

- D(h): GGX Normal Distribution Function
- F: Fresnel-Schlick approximation
- G: Smith masking-shadowing function

#### 2.3.2 Physics Consistency Scores

**Energy Conservation Score (ECS):**
```
ECS(x) = max(0, integral[f_r(x, w_i, w_o) * (n . w_i) dw_i] - 1)
```
For real images: ECS -> 0. For deepfakes: ECS > 0 (violates energy conservation).

**Specular Consistency Score (SCS):**
```
SCS(x) = ||L_observed(x) - (L_diffuse(x) + L_specular(x))||^2
```
Measures how well the observed color decomposes into physically plausible diffuse + specular.

**BRDF Smoothness Score (BSS):**
```
BSS = sum over neighbors ||BRDF(x_i) - BRDF(x_j)||^2 * w(x_i, x_j)
```
Real skin has spatially smooth BRDF; deepfakes show discontinuities.

**Illumination Coherence Score (ICS):**
```
ICS = Var(L_estimated(x)) across face regions
```
Real faces have coherent illumination; deepfakes may have inconsistent lighting per region.

**Temporal Physics Stability (TPS) [Video]:**
```
TPS = (1/T) * sum_t ||PhysicsScores(t) - PhysicsScores(t-1)||^2
```
Real physics properties are temporally stable; deepfake physics flicker.

#### 2.3.3 Final Classification Loss
```
L_total = L_BCE(y, y_hat) + lambda_1 * L_CoE + lambda_2 * L_spec + lambda_3 * L_smooth
```

### 2.4 Why This Will Generalize
- Physics laws are **universal** — they don't depend on which GAN/diffusion model generated the deepfake
- Energy conservation violations are **fundamental** — any neural generator that doesn't explicitly model light transport will violate them
- This is **complementary** to existing methods — can be combined with frequency/spatial detectors

---

## 3. Datasets

| Dataset | Size | Manipulation Methods | Resolution | Access |
|---------|------|---------------------|------------|--------|
| FaceForensics++ | 1000 orig + 4000 manip videos | DeepFakes, Face2Face, FaceSwap, NeuralTextures | Various | Google Form |
| Celeb-DF v2 | 590 real + 5639 fake | Improved DeepFake synthesis | 256x256 | GitHub |
| DFDC | 119,197 clips | Multiple methods | 320x240 to 4K | Kaggle |
| WildDeepfake | 7,314 sequences | In-the-wild deepfakes | Various | GitHub |
| DeeperForensics-1.0 | 60,000 videos | End-to-end pipeline | 1080p | Request |

### Synthetic PBR Test Dataset
We also create a controlled synthetic dataset using Blender with known ground-truth BRDF, lighting, and geometry to validate our physics scoring module.

---

## 4. Experimental Plan

### Phase 1: Validation (Weeks 1-2)
- Implement physics scoring module
- Test on synthetic Blender faces with known ground truth
- Verify that scores correctly distinguish physically consistent vs inconsistent renders

### Phase 2: Single-Image Detection (Weeks 3-4)
- Train on FaceForensics++ (train split)
- Test on FF++ (test split) — within-dataset performance
- Cross-dataset evaluation on CelebDF, DFDC

### Phase 3: Video-Level Detection (Weeks 5-6)
- Add temporal physics stability scoring
- Evaluate on video-level benchmarks
- Compare with FTCN, LipForensics

### Phase 4: Ablation & Analysis (Weeks 7-8)
- Ablate each physics score component
- Visualize physics inconsistency maps
- Test robustness to compression, resizing
- Adversarial robustness evaluation

---

## 5. Expected Results

Based on Light2Lie achieving ~95% AUC with simplified physics, and our richer feature set:
- **Within-dataset AUC:** >98% on FF++
- **Cross-dataset AUC:** >92% on CelebDF (vs ~85% for CNN baselines)
- **Key differentiator:** Generalization to unseen generators

---

## 6. Novelty Claims

1. **First work to apply full PBR-NeRF inverse rendering for deepfake detection**
2. **Novel physics consistency scoring framework** with 5 complementary scores
3. **Temporal physics stability** — new video-level forensic signal
4. **Generator-agnostic detection** via universal physics laws
5. **Interpretable results** — physics violation maps show WHERE the fake fails

---

## References

1. Wu et al., "PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields," CVPR 2025
2. Light2Lie, "Detecting Deepfake Images Using Physical Reflectance Laws," NDSS 2026
3. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields," ECCV 2020
4. Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019
5. Li et al., "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics," CVPR 2020
6. Dolhansky et al., "The DeepFake Detection Challenge (DFDC) Dataset," arXiv 2020
7. Li et al., "Face X-ray for More General Face Forgery Detection," CVPR 2020
8. Disney BRDF model, Burley, SIGGRAPH 2012

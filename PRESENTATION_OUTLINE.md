# PhysForensics - Presentation Outline

## Slide Deck Structure (25-30 slides)

---

### Slide 1: Title
**PhysForensics: Deepfake Detection via Physically-Based Neural Inverse Rendering Consistency**
- Authors, Affiliation, Date

### Slide 2: The Problem
- 8 million deepfakes online (16x growth in 2 years)
- Current detectors fail on unseen generators (~65% AUC cross-dataset)
- Visual: Side-by-side real vs deepfake (indistinguishable to humans)

### Slide 3: Why Current Methods Fail
- They learn generator-specific artifacts (GAN fingerprints, blending boundaries)
- When the generator changes, artifacts change -> detector fails
- Table: Cross-dataset AUC drops for XceptionNet, EfficientNet

### Slide 4: Our Key Insight
- **"Real faces obey physics. Deepfakes don't."**
- Real faces: light transport follows conservation of energy, Fresnel equations, BRDF models
- Deepfakes: neural networks model appearance, not physics
- This insight is GENERATOR-AGNOSTIC

### Slide 5: What is Physically-Based Rendering?
- Rendering equation diagram
- Disney BRDF components: diffuse + specular
- Visual: real photo decomposed into albedo, normals, lighting

### Slide 6: PBR-NeRF Background
- Wu et al., CVPR 2025
- Decomposes scene -> geometry (SDF) + materials (Disney BRDF) + illumination (NeILF)
- Physics-based losses: energy conservation + NDF-weighted specular

### Slide 7: PhysForensics Architecture (Overview)
- Full pipeline diagram: Face -> 3D Points -> PBR-NeRF -> Physics Scores -> Classification
- Highlight the novelty at each stage

### Slide 8: Stage 1 - Face to 3D
- Face detection (MTCNN)
- Learned depth estimation
- Point sampling on face surface

### Slide 9: Stage 2 - PBR-NeRF Inverse Rendering
- SDF geometry estimation
- Disney BRDF material estimation (albedo, roughness, metallic)
- NeILF illumination estimation
- Visual: decomposition examples

### Slide 10: Stage 3 - Physics Consistency Scores
- Five scores overview diagram
- Each score measures a different physical law

### Slide 11: ECS - Energy Conservation Score
- Math: E_reflected <= E_incident
- Visual: real face (ECS~0) vs fake face (ECS>0)
- Why deepfakes violate this

### Slide 12: SCS - Specular Consistency Score
- Math: ||observed - (diffuse + specular) * light||
- Visual: residual maps real vs fake

### Slide 13: BSS, ICS, TPS
- BRDF Smoothness: skin is smooth, fakes have discontinuities
- Illumination Coherence: coherent lighting vs inconsistent
- Temporal Stability: physics doesn't flicker

### Slide 14: Stage 4 - Cross-Attention Fusion
- Physics scores (queries) attend to visual features (keys/values)
- Learns which physics signals matter for different face regions
- Diagram of attention mechanism

### Slide 15: Training Objective
- Multi-component loss: BCE + Energy Conservation + Rendering + Anomaly
- Each term explained briefly

### Slide 16: Experimental Setup
- Datasets: FF++, CelebDF, DFDC, Synthetic PBR
- Metrics: AUC, EER, Accuracy
- Baselines: XceptionNet, EfficientNet, F3-Net, Face X-ray, Light2Lie

### Slide 17: Synthetic Validation
- Physics scores clearly separate real vs fake
- Histogram plots for each score
- Validates the core hypothesis

### Slide 18: Within-Dataset Results
- FF++ results table
- Competitive with specialized detectors

### Slide 19: Cross-Dataset Generalization (KEY RESULT)
- Train on FF++, test on CelebDF and DFDC
- PhysForensics vs baselines table
- Bar chart highlighting generalization gap

### Slide 20: Ablation Study
- Remove each physics score -> performance drops
- Remove all physics (visual only) -> generalization collapses
- Physics only (no visual) -> still strong cross-dataset

### Slide 21: Interpretability
- Physics violation heatmaps on real vs fake faces
- Shows WHERE on the face physics fails
- Decomposition views: albedo, roughness, normals, lighting

### Slide 22: Comparison with Light2Lie
- We use full PBR inverse rendering vs simplified reflectance
- Richer feature set -> better generalization
- Both validate physics-based detection paradigm

### Slide 23: Computational Analysis
- Model size: ~626K parameters (lightweight!)
- Inference time comparison
- Trade-off: accuracy vs speed

### Slide 24: Limitations & Future Work
- Single-view ambiguity
- Computation cost of inverse rendering
- Future: multi-view, sub-surface scattering, real-time distillation
- NeRF-based generators that model physics -> arms race

### Slide 25: Conclusion
- **Physics-based forensics is a fundamental defense**
- Universal laws don't change when generators improve
- Interpretable, generalizable, principled
- Opens new research direction at graphics x forensics intersection

### Slide 26: Thank You / Q&A
- Contact info
- GitHub/project link
- Key references

---

## Talking Points for Q&A

**Q: What if generators start modeling physics?**
A: This would be very computationally expensive and would essentially require solving the inverse rendering problem. Even so, our approach raises the bar significantly. We could evolve into adversarial training against physics-aware generators.

**Q: How does this compare to watermarking/C2PA approaches?**
A: Complementary. C2PA proves provenance of KNOWN content. We detect manipulation of UNKNOWN content. Both are needed.

**Q: Can this detect partial manipulations?**
A: Yes — our physics violation maps can localize WHERE manipulation occurred, not just whether it exists.

**Q: What about heavily compressed images?**
A: Compression destroys some physics signals. We plan to study robustness systematically. Initial augmentation with JPEG compression during training helps.

**Q: Runtime for real-time detection?**
A: Current model: ~44ms per image on CPU. With GPU and model distillation, real-time is feasible.

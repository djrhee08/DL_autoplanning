# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is the `DeepLearning-dev/` subdirectory of the larger `fluence_to_dose_2D` project. It is a focused R&D workspace for a VMAT (Volumetric Modulated Arc Therapy) dose prediction model, developed independently from the IMRT pipeline in `DeepLearning/`. Environment setup (uv, pyproject.toml, CUDA config) lives in the parent directory — see `../CLAUDE.md` for those details.

## Running

```bash
# Train the VMAT dose predictor
python DeepLearning-dev/main.py

# Smoke-test the attention model forward pass
python DeepLearning-dev/DoseCalculator_Attention.py

# Smoke-test the MLC aperture layer
python DeepLearning-dev/MLC2Aperture.py
```

Data directory: `./data/test/` — patient subdirectories containing `*_CT.npy`, `*_dose.npy`, `*_mlc_*.npy`, `*_jaw_*.npy`. The `is_cw` rotation direction is inferred from `_odd_start181` in the MLC filename.

Checkpoints saved to `./checkpoints/best_vmat_attn_model.pth`.

## Architecture

This codebase targets **VMAT** (arc therapy, 180 control points at 2° intervals starting at 181°), whereas `../DeepLearning/` targets static IMRT beams. They are separate and not interchangeable.

### Data shapes (key invariants)

| Tensor | Shape | Description |
|--------|-------|-------------|
| CT | `[B, 1, 192, 192, 192]` | Electron density, 3 mm/voxel |
| MLC raw | `[B, 180, 60, 2]` | Leaf positions (mm), X1/X2 per leaf pair |
| Jaw aperture | `[B, 180, 160, 160]` | Pre-computed 2D jaw mask |
| BEV input to model | `[B, 180, 2, 160, 160]` | Jaw + MLC channels stacked |
| Dose (output) | `[B, 1, 192, 192, 192]` | Predicted 3D dose, non-negative |

### Model pipeline (`DoseCalculator_Attention.py` — active model)

```
MLC positions [B,180,60,2]
    → DifferentiableMLCAperture  → [B,180,1,160,160]  (MLC2Aperture.py)
    + Jaw [B,180,1,160,160]
    → concat BEV [B,180,2,160,160]
        → BEVEncoder2D           → [B,180,32,40,40]
        → PerCPProjectionLayer   → [B,180,32,24,24,24]  ← stacked per CP, not summed

CT [B,1,192³]
    → 3D U-Net encoder           → bottleneck [B,128,24³]
        → SpatialBeamAttention   → fused [B,32,24³]  + attn_weights [B,13824,180]
        → fusion_conv            → [B,128,24³]
    → U-Net decoder              → dose [B,1,192³]
```

The key architectural insight is **SpatialBeamAttention**: each of the 13,824 bottleneck voxels (24³) attends over all 180 CPs, making `attn_weights[b, voxel_idx, cp_idx]` interpretable as per-CP dose contribution.

### `DoseCalculator.py` — baseline (no attention)

Older version: beam features are *summed* at 96³ resolution and fused via skip connection at Level 1 of the decoder. Kept for comparison but not used in `main.py`.

### `MLC2Aperture.py` — `DifferentiableMLCAperture`

Converts raw leaf positions (mm) to a 160×160 aperture map using `torch.sigmoid` for differentiability. The Millennium 120 MLC leaf width mapping: 10 outer leaves × 4 px (1 cm), 40 inner leaves × 2 px (0.5 cm), 10 outer leaves × 4 px (1 cm) = 160 rows total.

### Perspective projection geometry

`build_hfs_perspective_grids()` (in both `DoseCalculator*.py`) builds the `F.grid_sample` sampling grid that maps 3D CT voxels back to BEV pixel coordinates. Key parameters:
- SAD = 1000 mm
- Arc starts at 181°, increments by 2° for 180 CPs (wraps at 360°)
- CT FOV: 576 mm (192 voxels × 3 mm)
- BEV FOV in Attention model: 400 mm (160 px × 2.5 mm); in baseline: 560 mm (560 px × 1.0 mm)

### Loss function — `PhysicsInformedDoseLoss`

`L = α·L1 + β·(∇z + ∇y + ∇x)` where gradients are finite-difference L1 terms penalizing penumbra sharpness mismatch. Default: `alpha=1.0, beta=0.5` in `main.py`.

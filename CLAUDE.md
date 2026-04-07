# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical physics R&D workspace for automated VMAT (Volumetric Modulated Arc Therapy) radiotherapy planning using deep learning. The repository has three independent subdirectories that form a full pipeline: RayStation plan generation → DICOM preprocessing → deep learning dose prediction.

Each subdirectory has its own `CLAUDE.md` with component-specific detail. This file covers the cross-cutting architecture.

## Repository Structure

```
DL_autoplanning/
├── VMAT_autoplan_script/      ← RayStation TPS automation (creates training data)
├── preprocessing-dev/          ← DICOM → NumPy pipeline (prepares ML inputs)
└── DeepLearning-dev/           ← PyTorch VMAT dose predictor (model training)
```

## Running Each Component

```bash
# 1. (In RayStation scripting console) Create randomised VMAT plans and export DICOM
#    Run VMAT_autoplan_script/create_plans_VMAT.py via RayStation's Python console

# 2. Preprocess DICOM exports → NumPy arrays
#    Edit root_path in preprocessing-dev/main_total.py first, then:
python preprocessing-dev/main_total.py

# 3. Train the dose prediction model
python DeepLearning-dev/main.py

# Smoke-test model forward pass without training
python DeepLearning-dev/DoseCalculator_Attention.py

# Visualize preprocessed CT/dose volumes interactively
python DeepLearning-dev/visualize_CT.py

# Visualize MLC/jaw aperture stacks interactively
python preprocessing-dev/visualize_aperture_stack.py
```

## End-to-End Data Flow

```
RayStation TPS (VMAT_autoplan_script/)
  └─ Randomised VMAT plans → DICOM export (CT, RT Plan, RT Dose)

DICOM files at {root_path}/{patient}/CT/ and {root_path}/{patient}/temp*/
  └─ preprocessing-dev/main_total.py
      ├─ CT + RTSTRUCT → {patient}_CT.npy  [z, y, x] electron density, 3 mm/voxel
      ├─ RT Plan → jaw/MLC aperture stacks [180, 560, 560] per beam (2° gantry bins)
      └─ RT Dose → {patient}_{plan}_{beam}_dose.npy  [z, y, x] cGy, on CT grid

NumPy arrays at DeepLearning-dev/data/test/{patient}/
  └─ DeepLearning-dev/main.py
      └─ VMATDosePredictorAttention → predicted dose [B, 1, 192, 192, 192]
```

## Key Architectural Decisions

**Preprocessing (preprocessing-dev/):**  
MLC/jaw apertures are built at 10× supersampling (5600×5600) then block-averaged to 560×560 for anti-aliased sub-pixel edges. The 180 aperture slots correspond to 2° gantry bins; slot assignment uses `((G - canonical_start) % 360) // 2` where `canonical_start` = 181° (odd-parity) or 182° (even-parity). Collimator rotation is applied via `scipy.ndimage.rotate`.

**Model (DeepLearning-dev/):**  
The active model (`DoseCalculator_Attention.py`) uses a 3D U-Net encoder/decoder with a **SpatialBeamAttention** bottleneck. Each of the 24³ bottleneck voxels attends over all 180 control points, making `attn_weights[b, voxel_idx, cp_idx]` interpretable as a per-CP dose contribution map. BEV aperture features are projected into 3D via `F.grid_sample` perspective projection (SAD = 1000 mm) and stacked (not summed) across CPs before attention.

`DoseCalculator.py` is an older baseline (features summed at 96³ via skip connection) — kept for comparison, not used in training.

**RayStation scripts (VMAT_autoplan_script/):**  
Scripts use `from connect import *` (legacy) with a fallback to `import raystation` for RayStation ≥2025. `dose_opt_func.py` contains all optimization objective and constraint management; `create_plans_VMAT.py` is the main entry point for plan generation.

## Data Shape Invariants

| Array | Shape | Notes |
|-------|-------|-------|
| CT input to model | `[B, 1, 192, 192, 192]` | Electron density, 3 mm/voxel |
| MLC raw (from preprocessing) | `[180, 560, 560]` | Aperture mask, 1 mm/pixel |
| MLC raw (model input) | `[B, 180, 60, 2]` | Leaf positions mm, X1/X2 per pair |
| BEV to model | `[B, 180, 2, 160, 160]` | Jaw + MLC channels, 2.5 mm/pixel |
| Dose (output) | `[B, 1, 192, 192, 192]` | Non-negative, Gy |

The preprocessing and model use **different MLC representations** — `preprocessing-dev/` produces 560×560 aperture images while `DeepLearning-dev/` loads raw leaf position `.npy` files `[180, 60, 2]` and converts them on the fly via `DifferentiableMLCAperture`.

## Dependencies

No `requirements.txt` at repo root. Core packages: `torch`, `pydicom`, `SimpleITK`, `numpy`, `scipy`, `matplotlib`, `numba`. Training assumes CUDA (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`). RayStation scripts require the `connect`/`raystation` API and must run inside RayStation's embedded Python console.

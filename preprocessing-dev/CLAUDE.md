# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Standalone DICOM preprocessing pipeline that converts RayStation VMAT DICOM exports into NumPy arrays for ML training. Processes CT (with electron density overrides), per-beam dose, and per-gantry-angle jaw/MLC aperture stacks.

## Running the Pipeline

Edit `root_path` at the top of the script, then:
```bash
python main_total.py
```
Outputs to `./npy_total/{patient}/`.

**Visualize outputs interactively:**
```bash
python visualize_3D_nrrds.py
```

## Key Configuration

**`main_total.py`** top-level variables:
- `root_path` — DICOM source directory
- Expected layout: `{root_path}/{patient}/CT/` and `{root_path}/{patient}/temp*/`

## Output Format

Per patient:
- `{patient}_CT.npy` — CT volume as electron density, shape `(z, y, x)`, resampled to CT grid

Per DYNAMIC (VMAT) beam per plan:
- `{patient}_{plan}_{beam}_jaw_{parity}_start{start}.npy` — jaw aperture stack, shape `(180, 560, 560)`
- `{patient}_{plan}_{beam}_mlc_{parity}_start{start}.npy` — MLC aperture stack, shape `(180, 560, 560)`
- `{patient}_{plan}_{beam}_dose.npy` — beam dose in cGy, shape `(z, y, x)` matching CT grid

## Architecture

### Data Flow
```
{root_path}/{patient}/CT/          → DICOMCTRS_importer.read_dicom_CT()  → {patient}_CT.npy
{root_path}/{patient}/temp*/*.dcm  → DICOMRPRD_importer.rt_dose_info()   → beam↔dose file mapping
                                   → create_vmat_mlc_stack_for_beam()    → jaw/mlc .npy stacks
                                   → DICOMRPRD_importer.convert_dose_to_sitk() + sitk resample → dose .npy
```

### CTRS_import.py — `DICOMCTRS_importer`
`read_dicom_CT(CT_dir)` reads a CT DICOM series + RT struct from the same directory. Converts HU to electron density via `CTtoED.txt` lookup curve. Applies overrides: zeros voxels outside `EXTERNAL` contour, then overrides structures with `ROIPhysicalPropertiesSequence` (e.g. couch). Only supports `HFS` orientation. Returns a `SimpleITK.Image`.

### RP_to_aperture.py
`create_vmat_mlc_stack_for_beam(beam)` takes a pydicom DYNAMIC beam and returns a `(180, 560, 560)` stack for jaw and MLC separately. The 180 slots represent 2° gantry-angle bins covering a full 360°. Slot assignment: `slot = ((G - canonical_start) % 360) // 2`, where `canonical_start` is 181° for odd-parity arcs and 182° for even-parity arcs (determined from the first CP's gantry angle). Masks are built at 10× supersampling (5600×5600) then block-averaged to 560×560 for anti-aliased partial-pixel edges at device boundaries. Collimator rotation is applied via `scipy.ndimage.rotate`.

`recon_fluence(fluence, coll_angle)` reshapes a flat fluence vector to `(560, 560)`, flips vertically, and applies collimator rotation. Used when loading exported fluence `.npy` files.

> **Note:** `RP_to_aperture.py` contains `from config import ...` at the top — this is dead code; those variables are never used in any function.

### RPRD_import_total.py — `DICOMRPRD_importer`
Only two methods are used by the current pipeline:
- `rt_dose_info(rd_file_list)` — returns `{"BeamNumber": [...], "DoseFile": [...]}` (basenames) by reading `ReferencedBeamNumber` from each RT Dose file
- `convert_dose_to_sitk(ds)` — converts a pydicom RT Dose dataset to a `SimpleITK.Image` in cGy (normalises from Gy/mGy/cGy as needed using `DoseGridScaling`)

The class also contains `run_RPRD()` and Numba projection kernels (`project_single/two/three_apertures_3d_numba`) used in the parent `dev/` pipeline but not called from `main_total.py` in this branch.

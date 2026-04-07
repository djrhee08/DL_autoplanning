"""
preprocess.py  –  Convert DICOM data to .npy arrays for VMAT dose prediction training.

Input structure:
  data/
    {patient}/
      CT/          ← CT DICOM series (CT*.dcm) + optional RS*.dcm (RTSTRUCT)
      {plan}/      ← RT Plan (RP*.dcm) + RT Dose (RD*.dcm)

Output (default: ./data/processed/):
  {patient}_{plan}_CT.npy         [192, 192, 192]  float32, electron density (relative to water)
  {patient}_{plan}_dose.npy       [192, 192, 192]  float32, physical dose (Gy)
  {patient}_{plan}_mlc_cw_start181.npy  [180, 60, 2]  float32, leaf positions (mm)
  {patient}_{plan}_jaw_cw_start181.npy  [180, 2, 2]   float32, jaw boundaries (mm): [:,0,:]=[X1,X2], [:,1,:]=[Y1,Y2]

CT processing:
  1. HU values are converted to electron density via CTtoED.txt lookup curve.
  2. If an RTSTRUCT (RS*.dcm) is found in CT/:
       - Voxels outside the EXTERNAL contour are set to 0.
       - Structures with ROIPhysicalPropertiesSequence (REL_ELEC_DENSITY) are overridden.
  3. If no RTSTRUCT is found, only the HU→ED conversion is applied.

Usage:
  python DeepLearning-dev/preprocess.py
  python DeepLearning-dev/preprocess.py --data_dir ./data --out_dir ./data/processed
  python DeepLearning-dev/preprocess.py --ctted ./DeepLearning-dev/CTtoED.txt
"""

import os
import glob
import argparse
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path
from matplotlib.path import Path as MplPath


# ─────────────────────────────────────────────────────────────────────────────
# Constants matching the model's expected geometry
# ─────────────────────────────────────────────────────────────────────────────
NUM_CPS       = 180     # control points expected by VMATDosePredictorAttention
NUM_LEAVES    = 60      # Millennium 120 leaf pairs per bank
TARGET_START  = 181.0   # model assumes arc starts at 181° CW
TARGET_STEP   = 2.0     # 2° per CP
GRID_SIZE     = 160     # BEV aperture grid (160 × 160 pixels)
PIXEL_SIZE_MM = 2.5     # mm per BEV pixel
VOL_SIZE      = 192     # CT / dose volume side length (voxels)
VOL_SPACING   = 3.0     # mm per voxel

_DEFAULT_CTTED = str(Path(__file__).resolve().parent / 'CTtoED.txt')


# ─────────────────────────────────────────────────────────────────────────────
# 1. CT  ─ Load DICOM series, convert HU→ED, apply structure overrides
# ─────────────────────────────────────────────────────────────────────────────

def load_ct_sitk(ct_dir: str) -> sitk.Image:
    """Read CT DICOM series as a SimpleITK image (values in HU)."""
    all_dcm = glob.glob(os.path.join(ct_dir, '*.dcm'))
    ct_files = sorted(
        [f for f in all_dcm
         if pydicom.dcmread(f, stop_before_pixels=True).Modality == 'CT'],
        key=lambda f: float(pydicom.dcmread(f, stop_before_pixels=True)
                            .ImagePositionPatient[2])
    )
    if not ct_files:
        raise RuntimeError(f"No CT DICOM files in {ct_dir}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ct_files)
    return reader.Execute()


def _load_ctted_curve(ctted_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read CTtoED.txt and return (ct_values, ed_values) arrays."""
    ct_vals, ed_vals = [], []
    with open(ctted_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                ct_vals.append(float(parts[0]))
                ed_vals.append(float(parts[1]))
    return np.array(ct_vals, dtype=np.float64), np.array(ed_vals, dtype=np.float64)


def _parse_rtstruct(rs_path: str) -> tuple[dict, dict]:
    """
    Extract EXTERNAL contour and density-overridden structures from an RTSTRUCT file.

    Returns:
        contour_data  – {roi_name: [array(n,3), ...]}  contour points per slice
        edensity_data – {roi_name: float}  electron density override value
                        EXTERNAL is always present with value 0 (used as a flag).
    """
    ds = pydicom.dcmread(rs_path)

    roi_data      = {}   # roi_number → roi_name
    edensity_data = {}
    roi_name_list = []

    for obs in ds.RTROIObservationsSequence:
        if obs.RTROIInterpretedType.upper() == 'EXTERNAL':
            roi_number = obs.ReferencedROINumber
            roi_data[roi_number] = 'EXTERNAL'
            roi_name_list.append('EXTERNAL')
            edensity_data['EXTERNAL'] = 0

        if hasattr(obs, 'ROIPhysicalPropertiesSequence'):
            for prop in obs.ROIPhysicalPropertiesSequence:
                if prop.ROIPhysicalProperty == 'REL_ELEC_DENSITY':
                    roi_number = obs.ReferencedROINumber
                    roi_name   = obs.ROIObservationLabel
                    roi_data[roi_number] = roi_name
                    roi_name_list.append(roi_name)
                    edensity_data[roi_name] = float(prop.ROIPhysicalPropertyValue)

    contour_data = {}
    for contour_seq in ds.ROIContourSequence:
        roi_number = contour_seq.ReferencedROINumber
        roi_name   = roi_data.get(roi_number)
        if roi_name in roi_name_list:
            contour_data[roi_name] = []
            for item in contour_seq.ContourSequence:
                pts = np.array(item.ContourData, dtype=np.float64).reshape(-1, 3)
                contour_data[roi_name].append(pts)

    return contour_data, edensity_data


def _build_ct_info(ct_sitk: sitk.Image) -> dict:
    """
    Build a CT_info dict from a SimpleITK image.

    The dict uses the same conventions as CTRS_import.py so that
    _contour_to_mask can be used unchanged:
      image_shape = [rows, cols, slices]   (labeled x_dim, y_dim, z_dim in the rasteriser)
      spacing     = [row_spacing, col_spacing, slice_spacing]
      origin      = [x_origin, y_origin, z_origin]   (physical, DICOM LPS mm)
    """
    arr   = sitk.GetArrayFromImage(ct_sitk)   # [z, y, x]
    sp    = ct_sitk.GetSpacing()              # (col_sp, row_sp, slice_sp) in SimpleITK
    orig  = ct_sitk.GetOrigin()              # (x, y, z)
    return {
        'image_shape': [arr.shape[1], arr.shape[2], arr.shape[0]],  # [rows, cols, slices]
        'spacing':     [sp[1], sp[0], sp[2]],   # [row_sp, col_sp, slice_sp]
        'origin':      list(orig),
    }


def _contour_to_mask(contour_data: dict, ct_info: dict) -> dict[str, np.ndarray]:
    """
    Rasterise DICOM contour coordinates into 3D binary masks.

    Ported from CTRS_import.py: contour_to_binary_mask.
    Uses the even–odd rule so donut-shaped contours are handled correctly.

    Returns:
        masks – {roi_name: np.ndarray [z, y, x] uint8}
    """
    x_dim, y_dim, z_dim = ct_info['image_shape']   # rows, cols, slices
    x_sp, y_sp, z_sp    = ct_info['spacing']        # row_sp, col_sp, slice_sp
    ox, oy, oz          = ct_info['origin']

    masks = {}
    for roi_name, contours in contour_data.items():
        mask_3d = np.zeros((z_dim, x_dim, y_dim), dtype=np.uint8)  # [z, rows, cols]

        for contour in contours:
            x_world = contour[:, 0]
            y_world = contour[:, 1]
            z_world = contour[:, 2]

            z_idx = int(np.round(np.median((z_world - oz) / z_sp)))
            if z_idx < 0 or z_idx >= z_dim:
                continue

            x_pix = (x_world - ox) / x_sp   # → row-axis pixel index (note: same convention as CTRS_import)
            y_pix = (y_world - oy) / y_sp   # → col-axis pixel index

            poly  = np.vstack((x_pix, y_pix)).T
            path  = MplPath(poly, closed=True)

            xg, yg = np.mgrid[0:x_dim, 0:y_dim]
            pts    = np.vstack((xg.ravel(), yg.ravel())).T
            inside = path.contains_points(pts, radius=1e-9).reshape(x_dim, y_dim)

            mask_3d[z_idx] ^= inside.astype(np.uint8)

        # transpose [z, rows, cols] → [z, cols, rows]; then matches sitk [z, y, x]
        masks[roi_name] = mask_3d.transpose(0, 2, 1)

    return masks


def _apply_structure_overrides(
        ct_arr: np.ndarray,
        masks: dict[str, np.ndarray],
        edensity_data: dict
) -> np.ndarray:
    """
    Apply EXTERNAL masking and density overrides to a CT array [z, y, x].

    Order matters:
      1. Zero voxels outside EXTERNAL.
      2. Override each non-EXTERNAL structure with its specified ED value.
         (Running EXTERNAL last would erase everything, so it must go first.)
    """
    if 'EXTERNAL' not in masks:
        print("    [warn] EXTERNAL contour not found in RTSTRUCT – skipping masking")
        return ct_arr

    ct_arr[masks['EXTERNAL'] == 0] = 0.0

    for roi_name, mask in masks.items():
        if roi_name == 'EXTERNAL':
            continue
        ed_val = edensity_data.get(roi_name, None)
        if ed_val is not None:
            ct_arr[mask == 1] = float(ed_val)

    return ct_arr


def apply_ct_corrections(ct_sitk: sitk.Image, ct_dir: str, ctted_path: str) -> sitk.Image:
    """
    Convert HU → electron density and apply structure-based overrides.

    Steps:
      1. Apply CTtoED curve to all voxels.
      2. If an RTSTRUCT (RS*.dcm) is in ct_dir:
           a. Zero voxels outside EXTERNAL.
           b. Override structures that carry a REL_ELEC_DENSITY property.
      3. If no RTSTRUCT, only the HU→ED conversion is applied (with a warning).

    Returns a SimpleITK image with electron density values.
    """
    # Locate RTSTRUCT
    rs_path = None
    for f in glob.glob(os.path.join(ct_dir, '*.dcm')):
        try:
            if pydicom.dcmread(f, stop_before_pixels=True).Modality == 'RTSTRUCT':
                rs_path = f
                break
        except Exception:
            continue

    if rs_path is None:
        print(f"    [warn] No RTSTRUCT in {ct_dir} – applying HU→ED only (no masking/overrides)")

    # 1. HU → electron density
    ct_vals, ed_vals = _load_ctted_curve(ctted_path)
    ct_arr = sitk.GetArrayFromImage(ct_sitk).astype(np.float32)   # [z, y, x], HU
    ct_arr = np.interp(ct_arr, ct_vals, ed_vals).astype(np.float32)

    # 2 & 3. Structure-based overrides
    if rs_path is not None:
        try:
            contour_data, edensity_data = _parse_rtstruct(rs_path)
            ct_info = _build_ct_info(ct_sitk)
            masks   = _contour_to_mask(contour_data, ct_info)
            ct_arr  = _apply_structure_overrides(ct_arr, masks, edensity_data)
        except Exception as e:
            print(f"    [warn] RTSTRUCT processing error: {e} – using ED curve only")

    ct_ed = sitk.GetImageFromArray(ct_arr)
    ct_ed.CopyInformation(ct_sitk)
    return ct_ed


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dose  ─ Load RTDOSE → physical Gy
# ─────────────────────────────────────────────────────────────────────────────

def load_dose_sitk(rd_path: str) -> sitk.Image:
    """
    Load RTDOSE → physical Gy.

    All spatial metadata is extracted directly from pydicom rather than
    sitk.ReadImage, which can misread RTDOSE multi-frame geometry (wrong
    z-spacing or direction).  Follows the same approach as
    RPRD_import_total.py::convert_dose_to_sitk.

    sitk spacing convention: (x_sp, y_sp, z_sp) = (col_sp, row_sp, slice_sp).
    DICOM PixelSpacing: [0]=row spacing, [1]=col spacing  → must be swapped.
    """
    ds = pydicom.dcmread(rd_path)

    # Pixel data → Gy
    scaling = float(ds.DoseGridScaling) if hasattr(ds, 'DoseGridScaling') else 1.0
    dose_arr = ds.pixel_array.astype(np.float32) * scaling   # [frames, rows, cols]

    # Spacing
    col_sp = float(ds.PixelSpacing[1])   # x direction (columns)
    row_sp = float(ds.PixelSpacing[0])   # y direction (rows)
    offsets = [float(v) for v in ds.GridFrameOffsetVector]
    z_sp = abs(offsets[1] - offsets[0]) if len(offsets) > 1 else 1.0

    # Origin
    origin = [float(v) for v in ds.ImagePositionPatient]

    # Direction from ImageOrientationPatient
    # sitk direction matrix: columns are the physical-axis unit vectors
    # col 0 = row_dir (DICOM column direction = sitk x-axis)
    # col 1 = col_dir (DICOM row direction    = sitk y-axis)
    # col 2 = normal  (slice-normal direction  = sitk z-axis)
    if hasattr(ds, 'ImageOrientationPatient'):
        iop     = [float(v) for v in ds.ImageOrientationPatient]
        row_dir = np.array(iop[:3])
        col_dir = np.array(iop[3:])
        normal  = np.cross(row_dir, col_dir)
        direction = [row_dir[0], col_dir[0], normal[0],
                     row_dir[1], col_dir[1], normal[1],
                     row_dir[2], col_dir[2], normal[2]]
    else:
        direction = [1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0]

    img = sitk.GetImageFromArray(dose_arr)
    img.SetSpacing([col_sp, row_sp, z_sp])
    img.SetOrigin(origin)
    img.SetDirection(direction)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 3. Resampling  ─ Resample any volume to isocenter-centred 192³ @ 3 mm
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_isocenter(
        img: sitk.Image,
        isocenter_mm: np.ndarray,
        interpolator=sitk.sitkLinear,
        default_value: float = 0.0,
        output_direction=None,
) -> np.ndarray:
    """
    Resample img to a VOL_SIZE³ isotropic grid centred at isocenter_mm.
    Returns a float32 numpy array [z, y, x].

    output_direction: explicit 9-element direction tuple/list for the output
        reference grid.  Pass the CT direction so that CT and dose are always
        resampled into the same coordinate frame.  Defaults to the input
        image's own direction.
    """
    half       = (VOL_SIZE * VOL_SPACING) / 2.0
    new_origin = [float(isocenter_mm[0]) - half,
                  float(isocenter_mm[1]) - half,
                  float(isocenter_mm[2]) - half]

    ref = sitk.Image([VOL_SIZE, VOL_SIZE, VOL_SIZE], sitk.sitkFloat32)
    ref.SetSpacing([VOL_SPACING, VOL_SPACING, VOL_SPACING])
    ref.SetOrigin(new_origin)
    ref.SetDirection(output_direction if output_direction is not None
                     else img.GetDirection())

    resampled = sitk.Resample(img, ref, sitk.Transform(),
                              interpolator, default_value)
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RT Plan  ─ Extract MLC positions and jaw aperture
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_angles_cw(angles_deg: list[float]) -> np.ndarray:
    """
    Convert a list of CW gantry angles to a strictly increasing sequence
    (adds 360 each time the angle wraps past 0°).
    """
    unwrapped = [float(angles_deg[0])]
    for a in angles_deg[1:]:
        diff = (a - unwrapped[-1]) % 360.0
        if diff > 350.0:
            diff -= 360.0
        unwrapped.append(unwrapped[-1] + diff)
    return np.array(unwrapped, dtype=np.float64)


def _pack_jaw_params(jaw_x: list[float], jaw_y: list[float]) -> np.ndarray:
    """
    Pack jaw boundary positions into a [2, 2] array.

    Layout:
        result[0, :] = [X1, X2]  –  left / right boundary (mm)
        result[1, :] = [Y1, Y2]  –  inferior / superior boundary (mm)
    """
    return np.array([[jaw_x[0], jaw_x[1]],
                     [jaw_y[0], jaw_y[1]]], dtype=np.float32)


def extract_mlc_jaw(
        rp_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a VMAT RT Plan and return:
        mlc   [180, 60, 2]   – MLC leaf positions (mm), axis-2: [X1, X2]
        jaw   [180, 2, 2]    – jaw boundaries (mm): [:,0,:]=[X1,X2], [:,1,:]=[Y1,Y2]
        iso   [3]            – isocenter (x, y, z) in DICOM / LPS mm
    """
    ds   = pydicom.dcmread(rp_path)
    beam = ds.BeamSequence[0]
    cps  = beam.ControlPointSequence
    n    = len(cps)

    iso = np.array([float(v) for v in cps[0].IsocenterPosition], dtype=np.float64)

    jaw_x = jaw_y = None
    for dev in cps[0].BeamLimitingDevicePositionSequence:
        t = dev.RTBeamLimitingDeviceType
        if t == 'ASYMX':
            jaw_x = [float(v) for v in dev.LeafJawPositions]
        elif t == 'ASYMY':
            jaw_y = [float(v) for v in dev.LeafJawPositions]
    if jaw_x is None or jaw_y is None:
        raise RuntimeError("ASYMX / ASYMY jaw positions not found in CP0")

    jaw_params = _pack_jaw_params(jaw_x, jaw_y)
    jaw_stack  = np.tile(jaw_params, (NUM_CPS, 1, 1))

    mlc_raw    = np.zeros((n, NUM_LEAVES, 2), dtype=np.float32)
    angles_raw = np.zeros(n, dtype=np.float64)

    for i, cp in enumerate(cps):
        angles_raw[i] = float(cp.GantryAngle)
        for dev in cp.BeamLimitingDevicePositionSequence:
            if dev.RTBeamLimitingDeviceType == 'MLCX':
                vals = [float(v) for v in dev.LeafJawPositions]
                mlc_raw[i, :, 0] = vals[:NUM_LEAVES]
                mlc_raw[i, :, 1] = vals[NUM_LEAVES:]

    angles_uw  = _unwrap_angles_cw(list(angles_raw))
    target_raw = TARGET_START + TARGET_STEP * np.arange(NUM_CPS)
    shift      = np.round((angles_uw[0] - target_raw[0]) / 360.0) * 360.0
    target_uw  = target_raw + shift

    mlc_180 = np.zeros((NUM_CPS, NUM_LEAVES, 2), dtype=np.float32)
    for leaf in range(NUM_LEAVES):
        for bank in range(2):
            mlc_180[:, leaf, bank] = np.interp(
                target_uw,
                angles_uw,
                mlc_raw[:, leaf, bank],
                left=mlc_raw[0, leaf, bank],
                right=mlc_raw[-1, leaf, bank],
            )

    return mlc_180, jaw_stack, iso


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-patient processing
# ─────────────────────────────────────────────────────────────────────────────

def _find_dicom(directory: str, modality: str) -> str | None:
    """Return the first DICOM file with the given Modality in directory."""
    for f in glob.glob(os.path.join(directory, '*.dcm')):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if ds.Modality == modality:
                return f
        except Exception:
            continue
    return None


def process_patient(patient_dir: str, out_dir: str, ctted_path: str) -> None:
    patient = os.path.basename(patient_dir)
    ct_dir  = os.path.join(patient_dir, 'CT')

    if not os.path.isdir(ct_dir):
        print(f"  [skip] No CT/ directory – {patient_dir}")
        return

    plan_dirs = sorted([
        d for d in glob.glob(os.path.join(patient_dir, '*'))
        if os.path.isdir(d) and os.path.basename(d) != 'CT'
    ])
    if not plan_dirs:
        print(f"  [skip] No plan directories – {patient_dir}")
        return

    print(f"  Loading CT …")
    ct_sitk_hu = load_ct_sitk(ct_dir)

    print(f"  Applying HU→ED conversion and structure overrides …")
    ct_sitk_ed = apply_ct_corrections(ct_sitk_hu, ct_dir, ctted_path)

    for plan_dir in plan_dirs:
        plan   = os.path.basename(plan_dir)
        prefix = f"{patient}_{plan}"
        print(f"  Plan: {plan}")

        rp_path = _find_dicom(plan_dir, 'RTPLAN')
        rd_path = _find_dicom(plan_dir, 'RTDOSE')

        if rp_path is None:
            print(f"    [skip] No RTPLAN")
            continue
        if rd_path is None:
            print(f"    [skip] No RTDOSE")
            continue

        try:
            mlc, jaw, iso = extract_mlc_jaw(rp_path)
        except Exception as e:
            print(f"    [skip] RT Plan error: {e}")
            continue

        # CT: resample to 192³ @ 3 mm centred at isocenter (electron density values)
        # Capture direction so dose is resampled into the same coordinate frame.
        ct_direction = ct_sitk_ed.GetDirection()
        ct_arr = resample_to_isocenter(ct_sitk_ed, iso,
                                       default_value=0.0,
                                       output_direction=ct_direction)

        # Dose: resample to same grid, same orientation as CT
        try:
            dose_sitk = load_dose_sitk(rd_path)
            dose_arr  = resample_to_isocenter(dose_sitk, iso,
                                              default_value=0.0,
                                              output_direction=ct_direction)
        except Exception as e:
            print(f"    [skip] Dose error: {e}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        tag = 'cw_start181'
        np.save(os.path.join(out_dir, f"{prefix}_CT.npy"),        ct_arr)
        np.save(os.path.join(out_dir, f"{prefix}_dose.npy"),      dose_arr)
        np.save(os.path.join(out_dir, f"{prefix}_mlc_{tag}.npy"), mlc)
        np.save(os.path.join(out_dir, f"{prefix}_jaw_{tag}.npy"), jaw)

        print(f"    CT   {ct_arr.shape}  ED range [{ct_arr.min():.4f}, {ct_arr.max():.4f}]")
        print(f"    Dose {dose_arr.shape}  max {dose_arr.max():.4f} Gy")
        print(f"    MLC  {mlc.shape}  X1 [{mlc[:,:,0].min():.1f}, {mlc[:,:,0].max():.1f}] mm")
        print(f"    Jaw  {jaw.shape}  X=[{jaw[0,0,0]:.1f},{jaw[0,0,1]:.1f}] Y=[{jaw[0,1,0]:.1f},{jaw[0,1,1]:.1f}] mm")
        print(f"    → {out_dir}/{prefix}_{{CT,dose,mlc_{tag},jaw_{tag}}}.npy")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess DICOM VMAT data for VMATDosePredictorAttention"
    )
    parser.add_argument('--data_dir', default='./data',
                        help='Root directory containing patient subdirectories')
    parser.add_argument('--out_dir',  default='./data/processed',
                        help='Output directory for .npy files')
    parser.add_argument('--ctted',    default=_DEFAULT_CTTED,
                        help='Path to CTtoED.txt lookup curve')
    args = parser.parse_args()

    if not os.path.exists(args.ctted):
        raise FileNotFoundError(f"CTtoED curve not found: {args.ctted}")

    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(args.data_dir, '*'))
        if os.path.isdir(d) and os.path.basename(d) != 'processed'
    ])

    if not patient_dirs:
        print(f"No patient directories found in {args.data_dir}")
        return

    print(f"Found {len(patient_dirs)} patient(s): "
          f"{[os.path.basename(d) for d in patient_dirs]}")
    print(f"CTtoED curve: {args.ctted}")

    for pd in patient_dirs:
        print(f"\nPatient: {os.path.basename(pd)}")
        process_patient(pd, args.out_dir, args.ctted)

    print("\nDone.")


if __name__ == '__main__':
    main()

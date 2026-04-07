import os
import glob
import pathlib
import numpy as np
import pydicom as dicom
import SimpleITK as sitk
from CTRS_import import DICOMCTRS_importer
from RPRD_import_total import DICOMRPRD_importer
from RP_to_aperture import create_vmat_mlc_stack_for_beam

current_dir = pathlib.Path(__file__).parent.resolve()
root_path      = "Y:\\Research\\DL_autoplanning\\preprocessing-dev\\data\\total"
npy_output_dir = os.path.join(current_dir, "npy_total")

os.makedirs(npy_output_dir, exist_ok=True)

img_importer  = DICOMCTRS_importer()
RPRD_importer = DICOMRPRD_importer()

for patient in os.listdir(root_path):
    print("="*100)
    print(root_path, patient)
    npy_pt_dir = os.path.join(npy_output_dir, patient)
    os.makedirs(npy_pt_dir, exist_ok=True)

    # --- CT (with electron density overrides from RT struct) ---
    CT_path = os.path.join(root_path, patient, 'CT')
    CT_sitk = img_importer.read_dicom_CT(CT_path)

    for plan_dir in glob.glob(os.path.join(root_path, patient, "*")):
        if not os.path.isdir(plan_dir) or os.path.basename(plan_dir) == 'CT':
            continue
        plan_name = os.path.basename(plan_dir)

        rt_plan = None
        RD_list = []
        for dcm_path in glob.glob(os.path.join(plan_dir, "*.dcm")):
            ds = dicom.dcmread(dcm_path)
            if ds.Modality == 'RTPLAN':
                rt_plan = ds
            elif ds.Modality == 'RTDOSE':
                RD_list.append(dcm_path)
        if rt_plan is None:
            print(f"  No RT Plan found in {plan_dir}. Skipping.")
            continue

        # Build beam-number → dose-file lookup
        dose_info = RPRD_importer.rt_dose_info(RD_list)

        for arc_beam in rt_plan.BeamSequence:
            print(arc_beam.BeamType)
            if getattr(arc_beam, 'BeamType', '') != 'DYNAMIC':
                continue

            beam_name = getattr(arc_beam, 'BeamName', str(arc_beam.BeamNumber))
            npy_prefix = os.path.join(npy_pt_dir, f'{patient}_{plan_name}_{beam_name}')
            
            """
            print(f"  Creating VMAT MLC stack for beam '{beam_name}'...")
            stack = create_vmat_mlc_stack_for_beam(arc_beam)
            parity = stack['parity']
            start  = stack['actual_start']
            np.save(f'{npy_prefix}_jaw_{parity}_start{start}.npy', stack['jaw_stack'])
            np.save(f'{npy_prefix}_mlc_{parity}_start{start}.npy', stack['mlc_stack'])
            print(f"  Saved: parity={parity}, canonical_start={stack['canonical_start']}°, actual_start={start}°")
            """

            # --- Per-CP MLC positions [180,2,N], jaw positions [180,2,2], MU [180,1,1] ---

            # Get total beam MU from FractionGroupSequence
            total_mu = 0.0
            if hasattr(rt_plan, 'FractionGroupSequence') and len(rt_plan.FractionGroupSequence) > 0:
                for ref_beam in getattr(rt_plan.FractionGroupSequence[0], 'ReferencedBeamSequence', []):
                    if getattr(ref_beam, 'ReferencedBeamNumber', None) == arc_beam.BeamNumber:
                        total_mu = float(ref_beam.BeamMeterset)
                        break

            # Get MLC leaf-pair count from beam's device definition
            num_leaf_pairs = None
            for dev in getattr(arc_beam, 'BeamLimitingDeviceSequence', []):
                if dev.RTBeamLimitingDeviceType in ('MLCX', 'MLCY'):
                    num_leaf_pairs = int(dev.NumberOfLeafJawPairs)
                    break
            if num_leaf_pairs is None:
                print(f"  No MLC definition for beam '{beam_name}'. Skipping MLC/jaw/MU save.")
            else:
                # Check that all gantry angles are odd
                gantry_angles = [int(round(float(cp.GantryAngle))) % 360
                                 for cp in arc_beam.ControlPointSequence
                                 if hasattr(cp, 'GantryAngle')]
                non_odd = [g for g in gantry_angles if g % 2 == 0]
                if non_odd:
                    print(f"  Warning: {len(non_odd)} even gantry angle(s) in beam '{beam_name}': {non_odd[:5]}")
                else:
                    print(f"  All {len(gantry_angles)} gantry angles are odd for beam '{beam_name}'.")

                # Each slot = one control point recorded directly (no averaging).
                #   - MLC/jaw: positions at CP[i]
                #   - MU:      delta CumulativeMetersetWeight × total beam MU (CP[i]→CP[i+1];
                #              last CP gets 0)
                #   - angle:   CP gantry angle (odd: 181°, 183°, ..., 179°)
                #              slot = ((gantry - 181) % 360) // 2
                # 180 CPs → 180 slots
                # mlc: [180, num_leaf_pairs, 2] — axis2: 0=X1 (bankA), 1=X2 (bankB)
                # jaw: [180, 2, 2]             — axis1: 0=X jaw, 1=Y jaw; axis2: [pos1, pos2]
                # mu:  [180, 1, 1]              — MU from this CP to next
                mlc_array = np.zeros((180, num_leaf_pairs, 2), dtype=np.float32)
                jaw_array = np.zeros((180, 2, 2),             dtype=np.float32)
                mu_array  = np.zeros((180, 1, 1),             dtype=np.float32)

                cps = arc_beam.ControlPointSequence
                cum_weights = [
                    float(cp.CumulativeMetersetWeight) if hasattr(cp, 'CumulativeMetersetWeight') else None
                    for cp in cps
                ]

                # First pass: collect carried-forward device positions for every CP
                cp_mlc    = [None] * len(cps)
                cp_jaw_X  = [None] * len(cps)
                cp_jaw_Y  = [None] * len(cps)
                cp_gantry = [None] * len(cps)

                cur_mlc   = None
                cur_jaw_X = None
                cur_jaw_Y = None

                for i, cp in enumerate(cps):
                    # RayStation only writes devices that changed — carry forward the rest
                    for dev in getattr(cp, 'BeamLimitingDevicePositionSequence', []):
                        t = dev.RTBeamLimitingDeviceType
                        if t == 'ASYMX':
                            cur_jaw_X = list(map(float, dev.LeafJawPositions))
                        elif t == 'ASYMY':
                            cur_jaw_Y = list(map(float, dev.LeafJawPositions))
                        elif t in ('MLCX', 'MLCY'):
                            cur_mlc   = list(map(float, dev.LeafJawPositions))
                    cp_mlc[i]    = list(cur_mlc)   if cur_mlc   is not None else None
                    cp_jaw_X[i]  = list(cur_jaw_X) if cur_jaw_X is not None else None
                    cp_jaw_Y[i]  = list(cur_jaw_Y) if cur_jaw_Y is not None else None
                    cp_gantry[i] = float(cp.GantryAngle) % 360 if hasattr(cp, 'GantryAngle') else None

                # Second pass: one slot per CP — record positions directly, no averaging
                canonical_start = 181  # odd-parity arcs: CPs at 181°, 183°, ..., 179°
                for i in range(len(cps)):
                    g = cp_gantry[i]
                    if g is None:
                        continue

                    gantry_int = int(round(g)) % 360
                    slot = ((gantry_int - canonical_start) % 360) // 2
                    if not (0 <= slot < 180):
                        print(f"  Warning: CP {i} gantry {g:.0f}° → slot {slot} out of [0,179]. Skipped.")
                        continue

                    if cp_mlc[i] is not None:
                        mlc_array[slot, :, 0] = cp_mlc[i][:num_leaf_pairs]  # X1 (bank A)
                        mlc_array[slot, :, 1] = cp_mlc[i][num_leaf_pairs:]  # X2 (bank B)

                    if cp_jaw_X[i] is not None:
                        jaw_array[slot, 0, :] = cp_jaw_X[i]  # [X1, X2]
                    if cp_jaw_Y[i] is not None:
                        jaw_array[slot, 1, :] = cp_jaw_Y[i]  # [Y1, Y2]

                    # MU: delta from this CP to next; last CP gets 0
                    cw_this = cum_weights[i]
                    cw_next = cum_weights[i + 1] if i + 1 < len(cps) else None
                    if cw_this is not None and cw_next is not None:
                        mu_array[slot, 0, 0] = (cw_next - cw_this) * total_mu

                np.save(f'{npy_prefix}_mlc.npy', mlc_array)  # [180, num_leaf_pairs, 2] — X1/X2 per leaf pair
                np.save(f'{npy_prefix}_jaw.npy', jaw_array)  # [180, 2, 2]
                np.save(f'{npy_prefix}_mu.npy',  mu_array)   # [180, 1, 1]
                print(f"  Saved MLC [180,{num_leaf_pairs},2] / jaw [180,2,2] / MU [180,1,1] for beam '{beam_name}'.")

            # --- Dose (resampled to CT grid so dimensions match CT) ---
            beam_number = arc_beam.BeamNumber
            if beam_number not in dose_info["BeamNumber"]:
                print(f"  No dose file found for beam '{beam_name}'. Skipping dose.")
                continue
            rd_idx  = dose_info["BeamNumber"].index(beam_number)
            rd_file = os.path.join(plan_dir, dose_info["DoseFile"][rd_idx])

            dose_ds   = dicom.dcmread(rd_file)
            dose_sitk = RPRD_importer.convert_dose_to_sitk(dose_ds)

            # Isocenter from first control point (mm, DICOM coords: x, y, z)
            iso = [float(v) for v in arc_beam.ControlPointSequence[0].IsocenterPosition]

            # Resample CT and dose to 192³ @ 3 mm centered at isocenter
            DOSE_GRID = 3.0
            RESIZE_DIM = 192
            crop_origin = [iso[i] - 0.5 * RESIZE_DIM * DOSE_GRID for i in range(3)]
            crop_size   = [RESIZE_DIM, RESIZE_DIM, RESIZE_DIM]
            crop_spacing = [DOSE_GRID, DOSE_GRID, DOSE_GRID]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputOrigin(crop_origin)
            resampler.SetSize(crop_size)
            resampler.SetOutputSpacing(crop_spacing)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0.0)

            CT_crop   = resampler.Execute(CT_sitk)
            dose_crop = resampler.Execute(dose_sitk)

            np.save(f'{npy_prefix}_CT.npy',   sitk.GetArrayFromImage(CT_crop))    # (z, y, x)
            np.save(f'{npy_prefix}_dose.npy', sitk.GetArrayFromImage(dose_crop))  # (z, y, x)

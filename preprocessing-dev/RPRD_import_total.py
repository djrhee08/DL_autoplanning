import pydicom as dicom
import SimpleITK as sitk
import numpy as np
import cv2
import os
import sys
import torch
from numba import njit, prange

@njit(parallel=True)
def project_two_apertures_3d_numba(
    aperture_2d_a:    np.ndarray,  # first aperture  (e.g. jaw)
    aperture_2d_b:    np.ndarray,  # second aperture (e.g. mlc)
    H_ap: int, W_ap: int,
    aperture_pixel_size: float,
    nx: int, ny: int, nz: int,
    sx: float, sy: float, sz: float,
    ox: float, oy: float, oz: float,
    iso_x: float, iso_y: float, iso_z: float,
    d: np.ndarray,    # beam direction [3]
    u: np.ndarray,    # in-plane axis   [3]
    v: np.ndarray,    # in-plane axis   [3]
    source: np.ndarray,  # [3] = isocenter - SAD*d
    SAD: float,
    volume_out_a: np.ndarray,  # shape (nz, ny, nx) — output for aperture_a
    volume_out_b: np.ndarray   # shape (nz, ny, nx) — output for aperture_b
) -> None:
    """
    Numba-compiled projection for two apertures in a single voxel-loop pass.
    Ray geometry (t, col_f, row_f) is computed once and applied to both apertures,
    halving the cost vs calling the single-aperture version twice.
    For each voxel:
      t = SAD / ((X-S) dot d)
      P_int = S + t*(X-S)
      delta = P_int - iso
      (x_ap, y_ap) = (delta dot u, delta dot v)
    Then bilinear interpolate both aperture_2d_a and aperture_2d_b at (row, col).
    """
    half_w = W_ap / 2.0
    half_h = H_ap / 2.0

    for iz in prange(nz):
        zc = oz + iz * sz
        for iy in range(ny):
            yc = oy + iy * sy
            for ix in range(nx):
                xc = ox + ix * sx

                # Ray: S -> voxel center (xc, yc, zc)
                Xp = xc - source[0]
                Yp = yc - source[1]
                Zp = zc - source[2]

                denom = Xp*d[0] + Yp*d[1] + Zp*d[2]
                eps = 1e-8
                if denom < eps:
                    volume_out_a[iz, iy, ix] = 0.0
                    volume_out_b[iz, iy, ix] = 0.0
                    continue

                # Intersection param — computed once, reused for both apertures
                t = SAD / denom

                # Intersection point
                Pint_x = source[0] + t * Xp
                Pint_y = source[1] + t * Yp
                Pint_z = source[2] + t * Zp

                # Delta from isocenter
                Dx = Pint_x - iso_x
                Dy = Pint_y - iso_y
                Dz = Pint_z - iso_z

                # Aperture-plane coords
                u_val = Dx*u[0] + Dy*u[1] + Dz*u[2]
                v_val = Dx*v[0] + Dy*v[1] + Dz*v[2]

                # Convert to pixel coords — same for both apertures
                col_f = half_w + (u_val / aperture_pixel_size)
                row_f = half_h + (v_val / aperture_pixel_size)

                col0 = int(np.floor(col_f))
                row0 = int(np.floor(row_f))
                dc = col_f - col0
                dr = row_f - row0

                if col0 < 0 or col0 >= (W_ap - 1) or row0 < 0 or row0 >= (H_ap - 1):
                    volume_out_a[iz, iy, ix] = 0.0
                    volume_out_b[iz, iy, ix] = 0.0
                    continue

                # Bilinear interpolation weights — same for both apertures
                w00 = (1-dc) * (1-dr)
                w01 =    dc  * (1-dr)
                w10 = (1-dc) *    dr
                w11 =    dc  *    dr

                volume_out_a[iz, iy, ix] = (aperture_2d_a[row0,   col0  ] * w00 +
                                            aperture_2d_a[row0,   col0+1] * w01 +
                                            aperture_2d_a[row0+1, col0  ] * w10 +
                                            aperture_2d_a[row0+1, col0+1] * w11)

                volume_out_b[iz, iy, ix] = (aperture_2d_b[row0,   col0  ] * w00 +
                                            aperture_2d_b[row0,   col0+1] * w01 +
                                            aperture_2d_b[row0+1, col0  ] * w10 +
                                            aperture_2d_b[row0+1, col0+1] * w11)


@njit(parallel=True)
def project_single_aperture_3d_numba(
    ap: np.ndarray,
    H_ap: int, W_ap: int,
    aperture_pixel_size: float,
    nx: int, ny: int, nz: int,
    sx: float, sy: float, sz: float,
    ox: float, oy: float, oz: float,
    iso_x: float, iso_y: float, iso_z: float,
    d: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    source: np.ndarray,
    SAD: float,
    vol_out: np.ndarray
) -> None:
    """Numba-compiled single-aperture projection (used for fluence-only mode)."""
    half_w = W_ap / 2.0
    half_h = H_ap / 2.0
    for iz in prange(nz):
        zc = oz + iz * sz
        for iy in range(ny):
            yc = oy + iy * sy
            for ix in range(nx):
                xc = ox + ix * sx
                Xp = xc - source[0];  Yp = yc - source[1];  Zp = zc - source[2]
                denom = Xp*d[0] + Yp*d[1] + Zp*d[2]
                if denom < 1e-8:
                    vol_out[iz, iy, ix] = 0.0
                    continue
                t = SAD / denom
                Dx = source[0] + t*Xp - iso_x
                Dy = source[1] + t*Yp - iso_y
                Dz = source[2] + t*Zp - iso_z
                col_f = half_w + (Dx*u[0] + Dy*u[1] + Dz*u[2]) / aperture_pixel_size
                row_f = half_h + (Dx*v[0] + Dy*v[1] + Dz*v[2]) / aperture_pixel_size
                col0 = int(np.floor(col_f));  row0 = int(np.floor(row_f))
                dc = col_f - col0;            dr = row_f - row0
                if col0 < 0 or col0 >= (W_ap - 1) or row0 < 0 or row0 >= (H_ap - 1):
                    vol_out[iz, iy, ix] = 0.0
                    continue
                vol_out[iz, iy, ix] = (ap[row0,   col0  ] * (1-dc)*(1-dr) +
                                       ap[row0,   col0+1] *    dc *(1-dr) +
                                       ap[row0+1, col0  ] * (1-dc)*   dr  +
                                       ap[row0+1, col0+1] *    dc *   dr)


@njit(parallel=True)
def project_three_apertures_3d_numba(
    ap_a: np.ndarray,   # jaw  \
    ap_b: np.ndarray,   # mlc   } share the same pixel size and image dimensions
    H_ab: int, W_ab: int,
    pixel_size_ab: float,
    ap_c: np.ndarray,   # fluence — may differ in pixel size and image dimensions
    H_c: int, W_c: int,
    pixel_size_c: float,
    nx: int, ny: int, nz: int,
    sx: float, sy: float, sz: float,
    ox: float, oy: float, oz: float,
    iso_x: float, iso_y: float, iso_z: float,
    d: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    source: np.ndarray,
    SAD: float,
    vol_a: np.ndarray,
    vol_b: np.ndarray,
    vol_c: np.ndarray
) -> None:
    """
    Projects jaw, MLC, and fluence into 3D in one voxel-loop pass onto the CT grid.
    Ray geometry (t, u_val, v_val) is computed once per voxel and reused for all three.
    Jaw/MLC share pixel_size_ab; fluence uses pixel_size_c (may differ).
    """
    half_w_ab = W_ab / 2.0;  half_h_ab = H_ab / 2.0
    half_w_c  = W_c  / 2.0;  half_h_c  = H_c  / 2.0

    for iz in prange(nz):
        zc = oz + iz * sz
        for iy in range(ny):
            yc = oy + iy * sy
            for ix in range(nx):
                xc = ox + ix * sx
                Xp = xc - source[0];  Yp = yc - source[1];  Zp = zc - source[2]
                denom = Xp*d[0] + Yp*d[1] + Zp*d[2]
                if denom < 1e-8:
                    vol_a[iz, iy, ix] = 0.0
                    vol_b[iz, iy, ix] = 0.0
                    vol_c[iz, iy, ix] = 0.0
                    continue
                t = SAD / denom
                Dx = source[0] + t*Xp - iso_x
                Dy = source[1] + t*Yp - iso_y
                Dz = source[2] + t*Zp - iso_z
                # aperture-plane coords — computed once, used for all three
                u_val = Dx*u[0] + Dy*u[1] + Dz*u[2]
                v_val = Dx*v[0] + Dy*v[1] + Dz*v[2]

                # --- jaw / MLC ---
                col_f = half_w_ab + u_val / pixel_size_ab
                row_f = half_h_ab + v_val / pixel_size_ab
                col0 = int(np.floor(col_f));  row0 = int(np.floor(row_f))
                dc = col_f - col0;            dr = row_f - row0
                if col0 < 0 or col0 >= (W_ab - 1) or row0 < 0 or row0 >= (H_ab - 1):
                    vol_a[iz, iy, ix] = 0.0
                    vol_b[iz, iy, ix] = 0.0
                else:
                    w00=(1-dc)*(1-dr); w01=dc*(1-dr); w10=(1-dc)*dr; w11=dc*dr
                    vol_a[iz, iy, ix] = (ap_a[row0,col0]*w00 + ap_a[row0,col0+1]*w01 +
                                         ap_a[row0+1,col0]*w10 + ap_a[row0+1,col0+1]*w11)
                    vol_b[iz, iy, ix] = (ap_b[row0,col0]*w00 + ap_b[row0,col0+1]*w01 +
                                         ap_b[row0+1,col0]*w10 + ap_b[row0+1,col0+1]*w11)

                # --- fluence ---
                col_f = half_w_c + u_val / pixel_size_c
                row_f = half_h_c + v_val / pixel_size_c
                col0 = int(np.floor(col_f));  row0 = int(np.floor(row_f))
                dc = col_f - col0;            dr = row_f - row0
                if col0 < 0 or col0 >= (W_c - 1) or row0 < 0 or row0 >= (H_c - 1):
                    vol_c[iz, iy, ix] = 0.0
                else:
                    w00=(1-dc)*(1-dr); w01=dc*(1-dr); w10=(1-dc)*dr; w11=dc*dr
                    vol_c[iz, iy, ix] = (ap_c[row0,col0]*w00 + ap_c[row0,col0+1]*w01 +
                                         ap_c[row0+1,col0]*w10 + ap_c[row0+1,col0+1]*w11)


class DICOMRPRD_importer():
    def __init__(self):
        print("DICOMRPRD_importer (Total Version) is created")
        self.aperture_pixel_size = 1.25 # pixel size of aperture 2D image in mm
    
    def run_RPRD(self, rd_file, rt_plan, dose_info, sitk_CT_info, pt_info=None,
                 mode='aperture', fluence_dir=None, fluence_pixel_size=1.0,
                 ct_sitk=None, terma_voxel_size=2.0, terma_crop_size=192):
        """
        mode options:
          'aperture' — per-beam jaw+MLC 3D projection (original behaviour)
          'fluence'  — per-beam fluence 3D projection
          'all'      — jaw + MLC + fluence 3D projections
          'terma'    — accumulated TERMA via DifferentiableTotalTERMA across all CPs.
                       Requires fluence_dir and ct_sitk (SimpleITK CT in electron-density units).
                       IMRT: returns (beam_info, terma_sitk, norm_dose)
                       VMAT: saves one accumulated terma NRRD and returns True
        """
        if mode not in ('aperture', 'fluence', 'all', 'terma'):
            raise ValueError(f"mode must be 'aperture', 'fluence', 'all', or 'terma'; got '{mode}'")
        if mode in ('fluence', 'all', 'terma') and fluence_dir is None:
            raise ValueError("fluence_dir must be provided when mode is 'fluence', 'all', or 'terma'")
        if mode == 'terma' and ct_sitk is None:
            raise ValueError("ct_sitk must be provided when mode is 'terma'")

        self.sitk_CT_info = sitk_CT_info
        self.pt_info = pt_info

        rd_file_base = os.path.basename(rd_file)
        try:
            idx = dose_info["DoseFile"].index(rd_file_base)
        except:
            print("This RD file does not exist", rd_file_base)
            return None

        BeamNumber = dose_info["BeamNumber"][idx]

        final_beam = None
        for beam in rt_plan.BeamSequence:
            if beam.BeamNumber == BeamNumber:
                final_beam = beam
                break

        if final_beam is None:
            print(f"Beam {BeamNumber} not found in RT Plan")
            return None

        seq = rt_plan.FractionGroupSequence[0]
        num_fx = float(getattr(seq, 'NumberOfFractionsPlanned', 1.0))
        print("# fx : ", num_fx)

        total_beam_mu = self.get_MU(seq, BeamNumber)
        cp_info_list = self.get_generalized_beam_info(final_beam, total_beam_mu)

        dose_ds = dicom.dcmread(rd_file)
        dose_sitk = self.convert_dose_to_sitk(dose_ds)

        # Load fluence once per beam (shared across all CPs; rotated per CP below)
        fluence_raw = None
        if mode in ('fluence', 'all', 'terma'):
            beam_name = getattr(final_beam, 'BeamName', str(BeamNumber))
            fluence_raw = self.load_fluence(fluence_dir, beam_name)

        SAD = 1000.0

        if pt_info is None:
            # IMRT path — normalize by MU*fx, return first CP geometry
            norm_dose = dose_sitk / (num_fx * total_beam_mu)
            cp_info = cp_info_list[0]

            jaw_3D = mlc_3D = fluence_3D = terma_3D = None

            if mode in ('aperture', 'all'):
                jaw_2D, mlc_2D = self.create_2D_apertures(
                    cp_info['jaw_X'], cp_info['jaw_Y'], cp_info['mlc_X'], cp_info['mlc_Y'],
                    cp_info['collimator_angle'], self.aperture_pixel_size)
                jaw_3D, mlc_3D = self.create_3D_from_2D_apertures_pair_numba(
                    jaw_2D, mlc_2D, cp_info['isocenter'], cp_info['gantry_angle'],
                    cp_info['collimator_angle'], cp_info['couch_angle'], self.sitk_CT_info, SAD)

            if mode in ('fluence', 'all'):
                fluence_2D = self.rotate_aperture(fluence_raw, cp_info['collimator_angle'])
                fluence_3D = self.create_3D_from_fluence_numba(
                    fluence_2D, fluence_pixel_size, cp_info['isocenter'], cp_info['gantry_angle'],
                    cp_info['collimator_angle'], cp_info['couch_angle'], self.sitk_CT_info, SAD)

            if mode == 'terma':
                terma_3D = self._compute_terma(
                    cp_info_list, fluence_raw, ct_sitk, fluence_pixel_size,
                    terma_voxel_size, terma_crop_size, SAD)

            beam_info_ret = {
                'gantry_angle': cp_info['gantry_angle'],
                'collimator_angle': cp_info['collimator_angle'],
                'couch_angle': cp_info['couch_angle'],
                'isocenter': cp_info['isocenter'],
                'MU': total_beam_mu,
            }

            if mode == 'aperture':
                return beam_info_ret, jaw_3D, mlc_3D, norm_dose
            elif mode == 'fluence':
                return beam_info_ret, fluence_3D, norm_dose
            elif mode == 'terma':
                return beam_info_ret, terma_3D, norm_dose
            else:  # 'all'
                return beam_info_ret, jaw_3D, mlc_3D, fluence_3D, norm_dose

        else:
            # VMAT/3D path — normalize by fx, save NRRDs per CP
            dose_sitk = dose_sitk / num_fx

            if mode == 'terma':
                # Accumulate all CPs in one forward pass, save a single NRRD
                print(f"Computing TERMA for {len(cp_info_list)} CPs...")
                terma_sitk = self._compute_terma(
                    cp_info_list, fluence_raw, ct_sitk, fluence_pixel_size,
                    terma_voxel_size, terma_crop_size, SAD)
                self.save_nrrd(cp_info_list[0], 0, dose_sitk,
                               jaw_sitk=None, mlc_sitk=None, fluence_sitk=terma_sitk)
                return True

            for cp_idx, cp_info in enumerate(cp_info_list):
                print(f"CP {cp_idx}: isocenter {cp_info['isocenter']}, gantry {cp_info['gantry_angle']}, coll {cp_info['collimator_angle']}")

                jaw_3D = mlc_3D = fluence_3D = None

                if mode in ('aperture', 'all'):
                    jaw_2D, mlc_2D = self.create_2D_apertures(
                        cp_info['jaw_X'], cp_info['jaw_Y'], cp_info['mlc_X'], cp_info['mlc_Y'],
                        cp_info['collimator_angle'], self.aperture_pixel_size)
                    jaw_3D, mlc_3D = self.create_3D_from_2D_apertures_pair_numba(
                        jaw_2D, mlc_2D, cp_info['isocenter'], cp_info['gantry_angle'],
                        cp_info['collimator_angle'], cp_info['couch_angle'], self.sitk_CT_info, SAD)

                if mode in ('fluence', 'all'):
                    fluence_2D = self.rotate_aperture(fluence_raw, cp_info['collimator_angle'])
                    fluence_3D = self.create_3D_from_fluence_numba(
                        fluence_2D, fluence_pixel_size, cp_info['isocenter'], cp_info['gantry_angle'],
                        cp_info['collimator_angle'], cp_info['couch_angle'], self.sitk_CT_info, SAD)

                self.save_nrrd(cp_info, cp_idx, dose_sitk, jaw_3D, mlc_3D, fluence_3D)

            return True

    def rt_dose_info(self, rt_dose_list):
        dose_info = {"BeamNumber":[], "DoseFile":[]}
        for rt_dose_file in rt_dose_list:
            rt_dose = dicom.dcmread(rt_dose_file)
            summation_type = getattr(rt_dose, 'DoseSummationType', '').upper()
            if summation_type != 'BEAM':
                print(f"  Skipping dose file '{os.path.basename(rt_dose_file)}': "
                      f"DoseSummationType='{summation_type}' (expected 'BEAM'). "
                      f"Re-export per-beam doses from RayStation.")
                continue
            if hasattr(rt_dose, 'ReferencedRTPlanSequence'):
                for ref in rt_dose.ReferencedRTPlanSequence:
                    if not hasattr(ref, 'ReferencedFractionGroupSequence'):
                        print(f"  Skipping dose file '{os.path.basename(rt_dose_file)}': "
                              f"missing ReferencedFractionGroupSequence.")
                        continue
                    referenced_beam_number = ref.ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber
                    dose_info["BeamNumber"].append(referenced_beam_number)
                    dose_info["DoseFile"].append(os.path.basename(rt_dose_file))
        return dose_info
    
    def get_MU(self, FractionGroupSequence, BeamNumber):
        for referenced_beam in FractionGroupSequence.ReferencedBeamSequence:
            if hasattr(referenced_beam, 'BeamMeterset'):
                if referenced_beam.ReferencedBeamNumber == BeamNumber:
                    return referenced_beam.BeamMeterset
        return None

    def load_fluence(self, fluence_dir, beam_name):
        """
        Load fluence_{beam_name}.npy from fluence_dir.
        Expected file structure (object array):
          data[1] — isocenter position in CT coordinates (not used here; already in DICOM RP)
          data[2] — actual fluence values, shape (H, W), pixel size = 1.0 mm
          data[3] — jaw positions dict {'X1', 'X2', 'Y1', 'Y2'} in cm
        Returns fluence_array as float32.
        """
        path = os.path.join(fluence_dir, f'fluence_{beam_name}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fluence file not found: {path}")

        data = np.load(path, allow_pickle=True)
        return np.array(data[2], dtype=np.float32)

    def get_generalized_beam_info(self, beam, total_beam_mu):
        """
        Extract geometry for all control points, handling both Static/IMRT and VMAT.
        """
        cp_info_list = []
        if 'ControlPointSequence' not in beam:
            print("No ControlPointSequence found in this beam.")
            return cp_info_list
        
        mlc_boundaries = None
        if 'BeamLimitingDeviceSequence' in beam:
            for device in beam.BeamLimitingDeviceSequence:
                if device.RTBeamLimitingDeviceType == 'MLCX':
                    mlc_boundaries = device.LeafPositionBoundaries
                    break
        
        previous_cumulative_weight = 0.0
        current_gantry = 0.0
        current_collimator = 0.0
        current_couch = 0.0
        current_isocenter = [0.0, 0.0, 0.0]
        current_jaw_X = None
        current_jaw_Y = None
        current_mlc_X = None
        
        for i, cp in enumerate(beam.ControlPointSequence):
            # Geometry can be defined at each CP. If missing, it persists from previous CP.
            if hasattr(cp, 'GantryAngle'):
                current_gantry = float(cp.GantryAngle)
            
            if hasattr(cp, 'BeamLimitingDeviceAngle'):
                current_collimator = float(cp.BeamLimitingDeviceAngle)
                
            if hasattr(cp, 'PatientSupportAngle'):
                current_couch = float(cp.PatientSupportAngle)

            if hasattr(cp, 'IsocenterPosition'):
                current_isocenter = cp.IsocenterPosition
                
            # Update jaws/MLC if present in this CP
            if 'BeamLimitingDevicePositionSequence' in cp:
                for device in cp.BeamLimitingDevicePositionSequence:
                    if device.RTBeamLimitingDeviceType == 'ASYMX':
                        current_jaw_X = device.LeafJawPositions
                    elif device.RTBeamLimitingDeviceType == 'ASYMY':
                        current_jaw_Y = device.LeafJawPositions
                    elif device.RTBeamLimitingDeviceType == 'MLCX':
                        current_mlc_X = device.LeafJawPositions
            
            cp_dict = {
                'cp_index': i,
                'gantry_angle': current_gantry,
                'collimator_angle': current_collimator,
                'couch_angle': current_couch,
                'isocenter': current_isocenter,
                'jaw_X': current_jaw_X,
                'jaw_Y': current_jaw_Y,
                'mlc_X': current_mlc_X,
                'mlc_Y': mlc_boundaries
            }

            # MU logic
            w_i = getattr(cp, 'CumulativeMetersetWeight', None)
            if w_i is not None:
                delta_w = w_i - previous_cumulative_weight
                cp_dict['segment_mu'] = delta_w * total_beam_mu
                cp_dict['cumulative_mu'] = w_i * total_beam_mu
                previous_cumulative_weight = w_i
            else:
                # If no CumulativeMetersetWeight (e.g. static beam with 1 CP)
                cp_dict['segment_mu'] = total_beam_mu
                cp_dict['cumulative_mu'] = total_beam_mu

            # Check if this CP has enough info to be reconstructed (needs Jaws)
            # MLC can be None (e.g. 3D plans)
            if current_jaw_X is not None and current_jaw_Y is not None:
                cp_info_list.append(cp_dict)
        
        
        return cp_info_list

    def create_2D_apertures(self, jaw_X, jaw_Y, mlc_X, mlc_Y, collimator_angle, pixel_size=1.25, field_size=570):
        SUPERSAMPLE_FACTOR = 10

        # Output grid dimensions (e.g. 457x457 for default 570mm / 1.25mm + 1)
        field_dim = round(field_size / pixel_size) + 1

        # High-resolution grid (SUPERSAMPLE_FACTOR x finer in each axis)
        hires_dim = field_dim * SUPERSAMPLE_FACTOR
        hires_res = pixel_size / SUPERSAMPLE_FACTOR   # mm per sub-pixel
        # Pixel-edge convention: isocenter is at the left edge of the central pixel block.
        # Sub-pixel i spans [(i - hires_iso)*hires_res, (i+1 - hires_iso)*hires_res] mm.
        hires_iso = hires_dim / 2.0

        hires_jaw = np.zeros((hires_dim, hires_dim), dtype=np.float32)
        hires_mlc = np.zeros((hires_dim, hires_dim), dtype=np.float32)

        # --- JAW MASK ---
        x1_jaw, x2_jaw = sorted(jaw_X)
        y1_jaw, y2_jaw = sorted(jaw_Y)

        x1_pix = int(np.round(x1_jaw / hires_res + hires_iso))
        x2_pix = int(np.round(x2_jaw / hires_res + hires_iso))
        y1_pix = int(np.round(y1_jaw / hires_res + hires_iso))
        y2_pix = int(np.round(y2_jaw / hires_res + hires_iso))

        x1_c = max(0, x1_pix);  x2_c = min(hires_dim, x2_pix)
        y1_c = max(0, y1_pix);  y2_c = min(hires_dim, y2_pix)
        hires_jaw[y1_c:y2_c, x1_c:x2_c] = 1.0

        # --- MLC MASK ---
        if mlc_X is None or mlc_Y is None:
            sys.exit("MLC coordinates are missing in this control point, but they are required for aperture creation.")

        mlc_X = list(map(float, mlc_X))
        mlc_Y = list(map(float, mlc_Y))
        num_leaves = len(mlc_Y) - 1
        left_MLC  = mlc_X[0:num_leaves]
        right_MLC = mlc_X[num_leaves:]

        for leaf_index in range(num_leaves):
            x_a_mm   = left_MLC[leaf_index]    # Bank A (negative-X side)
            x_b_mm   = right_MLC[leaf_index]   # Bank B (positive-X side)
            y_inf_mm = mlc_Y[leaf_index]
            y_sup_mm = mlc_Y[leaf_index + 1]

            x_a_pix   = int(np.round(x_a_mm   / hires_res + hires_iso))
            x_b_pix   = int(np.round(x_b_mm   / hires_res + hires_iso))
            y_inf_pix = int(np.round(y_inf_mm / hires_res + hires_iso))
            y_sup_pix = int(np.round(y_sup_mm / hires_res + hires_iso))

            y_inf_c = max(0, y_inf_pix);  y_sup_c = min(hires_dim, y_sup_pix)
            x_a_c   = max(0, x_a_pix);   x_b_c   = min(hires_dim, x_b_pix)

            # Skip closed leaves (bank A past bank B) and out-of-bounds leaves
            if y_inf_c >= y_sup_c or x_a_c >= x_b_c:
                continue

            hires_mlc[y_inf_c:y_sup_c, x_a_c:x_b_c] = 1.0

        # --- DOWNSAMPLE by block averaging to get sub-pixel partial coverage ---
        block_shape = (field_dim, SUPERSAMPLE_FACTOR, field_dim, SUPERSAMPLE_FACTOR)
        jaw_apertures = np.flipud(hires_jaw.reshape(block_shape).mean(axis=(1, 3)))
        mlc_apertures = np.flipud(hires_mlc.reshape(block_shape).mean(axis=(1, 3)))

        jaw_apertures = self.rotate_aperture(jaw_apertures, collimator_angle)
        mlc_apertures = self.rotate_aperture(mlc_apertures, collimator_angle)

        return jaw_apertures, mlc_apertures

    def rotate_aperture(self, aperture, angle):
        center = tuple(np.array(aperture.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_aperture = cv2.warpAffine(aperture, rot_mat, aperture.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_aperture 
    
    def create_3D_from_2D_apertures_pair_numba(self, jaw_2d, mlc_2d, isocenter, gantry_angle, collimator_angle, couch_angle, sitk_info, SAD=1000.0):
        """Projects jaw and MLC apertures into 3D in a single voxel-loop pass."""
        volume_spacing, volume_origin, volume_size = sitk_info
        sx, sy, sz = volume_spacing
        ox, oy, oz = volume_origin
        nx, ny, nz = volume_size

        # Beam direction
        d0 = np.array([0., 1., 0.], dtype=np.float64)
        u0 = np.array([1., 0., 0.], dtype=np.float64)

        theta = np.deg2rad(gantry_angle)
        R_gantry = np.array([
            [ np.cos(theta), -np.sin(theta), 0.],
            [ np.sin(theta),  np.cos(theta), 0.],
            [            0.,             0., 1.]
        ], dtype=np.float64)

        R_total = R_gantry  # Could add couch rotation here if needed

        d = R_total @ d0
        u_ = R_total @ u0
        u_ /= np.linalg.norm(u_)
        v_ = np.cross(d, u_)
        v_ /= np.linalg.norm(v_)

        iso_arr = np.array(isocenter, dtype=np.float64)
        source = iso_arr - d * SAD

        jaw_out = np.zeros((nz, ny, nx), dtype=np.float32)
        mlc_out = np.zeros((nz, ny, nx), dtype=np.float32)

        H_ap, W_ap = jaw_2d.shape
        project_two_apertures_3d_numba(
            jaw_2d.astype(np.float32),
            mlc_2d.astype(np.float32),
            H_ap, W_ap,
            self.aperture_pixel_size,
            nx, ny, nz,
            sx, sy, sz,
            ox, oy, oz,
            iso_arr[0], iso_arr[1], iso_arr[2],
            d, u_, v_,
            source, SAD,
            jaw_out,
            mlc_out
        )

        def to_sitk(arr):
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((sx, sy, sz))
            img.SetOrigin((ox, oy, oz))
            img.SetDirection([1.,0.,0.,0.,1.,0.,0.,0.,1.])
            return img

        return to_sitk(jaw_out), to_sitk(mlc_out)

    def create_3D_from_fluence_numba(self, fluence_2d, fluence_pixel_size, isocenter, gantry_angle,
                                      collimator_angle, couch_angle, sitk_info, SAD=1000.0):
        """Projects a 2D fluence map onto the CT grid using the single-aperture Numba kernel."""
        volume_spacing, volume_origin, volume_size = sitk_info
        sx, sy, sz = volume_spacing
        ox, oy, oz = volume_origin
        nx, ny, nz = volume_size

        d0 = np.array([0., 1., 0.], dtype=np.float64)
        u0 = np.array([1., 0., 0.], dtype=np.float64)
        theta = np.deg2rad(gantry_angle)
        R_gantry = np.array([
            [ np.cos(theta), -np.sin(theta), 0.],
            [ np.sin(theta),  np.cos(theta), 0.],
            [            0.,             0., 1.]
        ], dtype=np.float64)
        d = R_gantry @ d0
        u_ = R_gantry @ u0;  u_ /= np.linalg.norm(u_)
        v_ = np.cross(d, u_);  v_ /= np.linalg.norm(v_)

        iso_arr = np.array(isocenter, dtype=np.float64)
        source  = iso_arr - d * SAD

        fluence_out = np.zeros((nz, ny, nx), dtype=np.float32)
        H_f, W_f = fluence_2d.shape
        project_single_aperture_3d_numba(
            fluence_2d.astype(np.float32), H_f, W_f, fluence_pixel_size,
            nx, ny, nz, sx, sy, sz, ox, oy, oz,
            iso_arr[0], iso_arr[1], iso_arr[2],
            d, u_, v_, source, SAD, fluence_out
        )

        img = sitk.GetImageFromArray(fluence_out)
        img.SetSpacing((sx, sy, sz))
        img.SetOrigin((ox, oy, oz))
        img.SetDirection([1.,0.,0.,0.,1.,0.,0.,0.,1.])
        return img

    def _compute_terma(self, cp_info_list, fluence_raw, ct_sitk,
                       fluence_pixel_size=1.0, voxel_size=2.0, crop_size=192, SAD=1000.0):
        """Compute accumulated TERMA for all CPs of one beam using DifferentiableTotalTERMA.

        The CT is resampled to a cube (crop_size³ at voxel_size mm) centered at the
        beam isocenter — the same convention as preprocessing.py — so that
        DifferentiableTotalTERMA's isocenter-centered coordinate assumption holds.

        ct_sitk must contain electron-density values (as produced by DICOMCTRS_importer).
        Returns the TERMA volume as a SimpleITK image in the resampled CT space.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Resample CT to a cube centered at isocenter (isocenter = centre of volume)
        isocenter = [float(v) for v in cp_info_list[0]['isocenter']]
        origin = [iso - 0.5 * crop_size * voxel_size for iso in isocenter]

        resample = sitk.ResampleImageFilter()
        resample.SetTransform(sitk.CompositeTransform(3))
        resample.SetOutputOrigin(origin)
        resample.SetSize([crop_size, crop_size, crop_size])
        resample.SetOutputSpacing([voxel_size, voxel_size, voxel_size])
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(0)
        ct_crop = resample.Execute(ct_sitk)

        ct_array = sitk.GetArrayFromImage(ct_crop)  # (z, y, x) — electron density
        D, H, W = ct_array.shape

        # Build per-CP fluence maps (rotate by collimator angle) and gantry angles
        mlc_list, gantry_list = [], []
        for cp_info in cp_info_list:
            fluence_2d = self.rotate_aperture(fluence_raw, cp_info['collimator_angle'])
            mlc_list.append(fluence_2d)
            gantry_list.append(np.deg2rad(float(cp_info['gantry_angle'])))

        H_f, W_f = mlc_list[0].shape
        mlc_physical_size = W_f * fluence_pixel_size  # total field width in mm

        mlc_tensor    = torch.tensor(np.stack(mlc_list), dtype=torch.float32).unsqueeze(0)  # (1, N_cp, H_f, W_f)
        gantry_tensor = torch.tensor(gantry_list, dtype=torch.float32)                       # (N_cp,)
        ct_tensor     = torch.tensor(ct_array[np.newaxis, np.newaxis], dtype=torch.float32)  # (1, 1, D, H, W)

        model = DifferentiableTotalTERMA(
            ct_shape=(D, H, W),
            voxel_size=voxel_size,
            sad=SAD,
            mlc_size=mlc_physical_size,
            ct_is_ed=True,  # CT from DICOMCTRS_importer is already in electron density
        ).to(device)

        with torch.no_grad():
            terma = model(mlc_tensor.to(device), gantry_tensor.to(device), ct_tensor.to(device))

        terma_np = terma.squeeze().cpu().numpy()  # (z, y, x)

        terma_sitk = sitk.GetImageFromArray(terma_np)
        terma_sitk.CopyInformation(ct_crop)
        return terma_sitk

    def convert_dose_to_sitk(self, ds):
        if ds.DoseUnits.lower() == 'gy':
            factor = 100
        elif ds.DoseUnits.lower() == 'mgy':
            factor = 0.1
        elif ds.DoseUnits.lower() == 'cgy':
            factor = 1
        else:
            print("DoseUnits unknown", ds.DoseUnits)
            factor = 1.0
        
        dose_grid_scaling = float(ds.DoseGridScaling) if hasattr(ds, 'DoseGridScaling') else 1.0
        dose_array = ds.pixel_array.astype(np.float32) * dose_grid_scaling * factor
        
        dose_spacing = [float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.GridFrameOffsetVector[1] - ds.GridFrameOffsetVector[0])]
        dose_origin = [float(ds.ImagePositionPatient[0]), float(ds.ImagePositionPatient[1]), float(ds.ImagePositionPatient[2])]

        if ds.ImageOrientationPatient != [1, 0, 0, 0, 1, 0]:
            print("Warning: ImageOrientationPatient is not aligned with the axes")

        sitk_dose = sitk.GetImageFromArray(dose_array)
        sitk_dose.SetSpacing(dose_spacing)
        sitk_dose.SetOrigin(dose_origin)

        return sitk_dose
    
    def save_nrrd(self, cp_info, cp_idx, dose_sitk, jaw_sitk, mlc_sitk, fluence_sitk=None):
        pt_dir = self.pt_info['pt_dir']
        patient = self.pt_info['patient']
        plan_name = self.pt_info['plan_name']
        dose_idx = self.pt_info['dose_index']

        isocenter = cp_info['isocenter']
        gantry_angle = cp_info['gantry_angle']
        MU = cp_info.get('segment_mu', 0.0)

        def set_meta(img):
            img.SetMetaData("gantry_angle", str(gantry_angle))
            img.SetMetaData("isocenter", str(isocenter))
            img.SetMetaData("MU", str(MU))

        prefix = os.path.join(pt_dir, f'{patient}_{plan_name}_{str(dose_idx)}_{str(cp_idx)}')

        if jaw_sitk is not None:
            set_meta(jaw_sitk)
            sitk.WriteImage(jaw_sitk, f'{prefix}_jaw.nrrd', useCompression=True)
        if mlc_sitk is not None:
            set_meta(mlc_sitk)
            sitk.WriteImage(mlc_sitk, f'{prefix}_mlc.nrrd', useCompression=True)
        if fluence_sitk is not None:
            set_meta(fluence_sitk)
            sitk.WriteImage(fluence_sitk, f'{prefix}_fluence.nrrd', useCompression=True)

        if cp_idx == 0:
            dose_sitk.SetMetaData("gantry_angle", str(gantry_angle))
            dose_sitk.SetMetaData("isocenter", str(isocenter))
            sitk.WriteImage(dose_sitk, os.path.join(pt_dir, f'{patient}_{plan_name}_{str(dose_idx)}_dose.nrrd'), useCompression=True)
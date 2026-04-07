import numpy as np
import pydicom
import sys
from scipy.ndimage import rotate

# --- CONSTANTS: Define the fluence grid shape ---
GRID_SHAPE = (560, 560)  # (rows, cols)
RESOLUTION_MM = 1.0

# --- NEW CONSTANT ---
# We will create a high-res mask (10x) and downsample it
# to get partial pixel values. 10x means 0.1mm sub-pixels.
SUPERSAMPLE_FACTOR = 10
# --- END OF NEW CONSTANT ---

def collimator_rotation(image, angle_degrees):
    """
    Rotates the given image by the specified angle in degrees around the center.
    
    Parameters:
        image (np.ndarray): The input 2D image to rotate.
        angle_degrees (float): The angle by which to rotate the image, in degrees.
        
    Returns:
        np.ndarray: The rotated image.
    """
    # reshape=False keeps the output shape the same as the input shape (560x560).
    # order=1 is bilinear interpolation, suitable for both masks and fluence maps.
    return rotate(image, angle_degrees, reshape=False, order=1)


def create_jaw_mlc_list(rt_file):
    """
    Reads a DICOM RT Plan file and creates 2D masks for the
    jaws and MLCs for every beam.
    
    This version uses SUPERSAMPLING to create masks with
    partial values (anti-aliasing) at the device edges.
    """
    dcm = pydicom.dcmread(rt_file)
    
    # --- DEFINE HIGH-RESOLUTION CONSTANTS ---
    HIRES_SHAPE = (GRID_SHAPE[0] * SUPERSAMPLE_FACTOR, GRID_SHAPE[1] * SUPERSAMPLE_FACTOR)
    HIRES_RES_MM = RESOLUTION_MM / SUPERSAMPLE_FACTOR
    #HIRES_ISO_X = (HIRES_SHAPE[1] - 1) / 2.0
    #HIRES_ISO_Y = (HIRES_SHAPE[0] - 1) / 2.0
    # should be simply half of the shape, since we are naively calculating the center pixel position, which is (0,0) in MLC dimension
    # e.g. for 3x3 grid, central pixel value is (1.5, 1.5), which is 3/2=1.5
    # Pixel edge definition is necessary here
    HIRES_ISO_X = HIRES_SHAPE[1] / 2.0
    HIRES_ISO_Y = HIRES_SHAPE[0] / 2.0
    # ---

    beam_list = []
    jaw_list = []
    mlc_list = []
    MU_list = []
    coll_angle_list = []

    # Create MU Lookup Dictionary ---
    # The most reliable place for MU is in the FractionGroupSequence
    mu_lookup = {}
    if hasattr(dcm, 'FractionGroupSequence') and len(dcm.FractionGroupSequence) > 0:
        # Assuming we only care about the first fraction group
        if hasattr(dcm.FractionGroupSequence[0], 'ReferencedBeamSequence'):
            ref_beam_seq = dcm.FractionGroupSequence[0].ReferencedBeamSequence
            for ref_beam in ref_beam_seq:
                if hasattr(ref_beam, 'ReferencedBeamNumber') and hasattr(ref_beam, 'BeamMeterset'):
                    beam_num = ref_beam.ReferencedBeamNumber
                    mu = ref_beam.BeamMeterset
                    mu_lookup[beam_num] = float(mu)
        
    if not mu_lookup:
        print("Warning: Could not find 'ReferencedBeamSequence' in 'FractionGroupSequence[0]'. "
              "MU values will be 0.0.")

    # Find Device Definition (No Change)
    device_definition_sequence = getattr(dcm, 'BeamLimitingDeviceSequence', None)
    
    if device_definition_sequence is None:
        if hasattr(dcm, 'BeamSequence') and len(dcm.BeamSequence) > 0:
            device_definition_sequence = getattr(dcm.BeamSequence[0], 'BeamLimitingDeviceSequence', None)

    if device_definition_sequence is None:
        raise AttributeError("Fatal: Could not find 'BeamLimitingDeviceSequence' at the root "
                             "or within dcm.BeamSequence[0]. Cannot determine MLC properties.")

    mlc_def = None
    for device_def in device_definition_sequence:
        if device_def.RTBeamLimitingDeviceType in ['MLCX', 'MLCY']:
            mlc_def = device_def
            break
            
    if mlc_def is None:
        raise ValueError("Fatal: Could not find MLC definition ('MLCX' or 'MLCY') "
                         "in the located BeamLimitingDeviceSequence")
    
    leaf_y_boundaries_mm = mlc_def.LeafPositionBoundaries
    num_leaf_pairs = mlc_def.NumberOfLeafJawPairs

    for beam in dcm.BeamSequence:
        beam_list.append(beam.BeamName)
        
        # Extract collimator angle (in degree) and rotation direction
        coll_angle = float(beam.ControlPointSequence[0].BeamLimitingDeviceAngle)
        coll_angle_rotation_direction = beam.ControlPointSequence[0].BeamLimitingDeviceRotationDirection
        
        if coll_angle_rotation_direction != 'NONE':
            print("Warning: Collimator rotation direction is not 'NONE'. ")
            print(f"  Beam {beam.BeamName} has Collimator Rotation Direction: {coll_angle_rotation_direction}")
            sys.exit()
        
        # Extract MU 
        mu_for_this_beam = mu_lookup.get(beam.BeamNumber, None) # Get MU from lookup, default to 0.0
        MU_list.append(mu_for_this_beam)
        
        # Initialize masks at HIGH resolution
        hires_jaw_mask = np.zeros(HIRES_SHAPE, dtype=np.float32)
        hires_mlc_mask = np.zeros(HIRES_SHAPE, dtype=np.float32)
        
        if not hasattr(beam, 'ControlPointSequence') or len(beam.ControlPointSequence) == 0:
            print(f"  Warning: Beam {beam.BeamName} has no Control Points. Appending empty masks.")
            jaw_list.append(np.zeros(GRID_SHAPE, dtype=np.float32))
            mlc_list.append(np.zeros(GRID_SHAPE, dtype=np.float32))
            continue
            
        cp0 = beam.ControlPointSequence[0]
        
        jaw_x_mm = None
        jaw_y_mm = None
        mlc_positions_mm = None
        
        if not hasattr(cp0, 'BeamLimitingDevicePositionSequence'):
            print(f"  Warning: Beam {beam.BeamName}, CP 0 has no 'BeamLimitingDevicePositionSequence'. Appending empty masks.")
            jaw_list.append(np.zeros(GRID_SHAPE, dtype=np.float32))
            mlc_list.append(np.zeros(GRID_SHAPE, dtype=np.float32))
            continue

        for device in cp0.BeamLimitingDevicePositionSequence:
            if device.RTBeamLimitingDeviceType == 'ASYMX':
                jaw_x_mm = device.LeafJawPositions
            elif device.RTBeamLimitingDeviceType == 'ASYMY':
                jaw_y_mm = device.LeafJawPositions
            elif device.RTBeamLimitingDeviceType in ['MLCX', 'MLCY']:
                mlc_positions_mm = device.LeafJawPositions

        # 2. Create the HI-RES Jaw Mask
        if jaw_x_mm and jaw_y_mm:
            # Convert mm to HI-RES pixel indices
            x1_pix = int(np.round((jaw_x_mm[0] / HIRES_RES_MM) + HIRES_ISO_X))
            x2_pix = int(np.round((jaw_x_mm[1] / HIRES_RES_MM) + HIRES_ISO_X))
            y_inf_pix = int(np.round((jaw_y_mm[0] / HIRES_RES_MM) + HIRES_ISO_Y))
            y_sup_pix = int(np.round((jaw_y_mm[1] / HIRES_RES_MM) + HIRES_ISO_Y))
            
            x1_clip = max(0, x1_pix)
            x2_clip = min(HIRES_SHAPE[1], x2_pix)
            y_inf_pix = max(0, y_inf_pix)
            y_sup_pix = min(HIRES_SHAPE[0], y_sup_pix)
            
            hires_jaw_mask[y_inf_pix:y_sup_pix, x1_clip:x2_clip] = 1.0
        else:
            print(f"  Warning: Beam {beam.BeamName} is missing jaw positions. Jaw mask will be empty.")
            sys.exit()
        
        # 3. Create the HI-RES MLC Mask
        if mlc_positions_mm:
            mlc_bank_a_mm = mlc_positions_mm[:num_leaf_pairs]
            mlc_bank_b_mm = mlc_positions_mm[num_leaf_pairs:]
            
            for i in range(num_leaf_pairs):
                y_inf_mm = float(leaf_y_boundaries_mm[i])
                y_sup_mm = float(leaf_y_boundaries_mm[i+1])
                
                # Convert mm to HI-RES pixel indices
                y_inf_pix = int(np.round(HIRES_ISO_Y + (y_inf_mm / HIRES_RES_MM)))
                y_sup_pix = int(np.round(HIRES_ISO_Y + (y_sup_mm / HIRES_RES_MM)))
                
                x_a_mm = mlc_bank_a_mm[i]
                x_b_mm = mlc_bank_b_mm[i]
                x_a_pix = int(np.round(x_a_mm / HIRES_RES_MM + HIRES_ISO_X))
                x_b_pix = int(np.round(x_b_mm / HIRES_RES_MM + HIRES_ISO_X))

                y_inf_pix = max(0, y_inf_pix)
                y_sup_pix = min(HIRES_SHAPE[0], y_sup_pix)
                x_a_clip = max(0, x_a_pix)
                x_b_clip = min(HIRES_SHAPE[1], x_b_pix)
                
                if y_inf_pix >= y_sup_pix or x_a_clip >= x_b_clip:
                    continue
                                
                hires_mlc_mask[y_inf_pix:y_sup_pix, x_a_clip:x_b_clip] = 1.0

        # --- DOWNSAMPLE to final (560, 560) grid ---
        # This uses "block averaging"
        # 1. Reshape (560, 10, 560, 10) -> (560, 560)
        shape = (
            GRID_SHAPE[0], SUPERSAMPLE_FACTOR,
            GRID_SHAPE[1], SUPERSAMPLE_FACTOR
        )
        
        # 2. Take the mean over the (10, 10) sub-blocks
        #    mean(axis=3) -> (560, 10, 560)
        #    mean(axis=1) -> (560, 560)
        lores_jaw_mask = np.flipud(hires_jaw_mask.reshape(shape).mean(axis=(1, 3)))
        lores_mlc_mask = np.flipud(hires_mlc_mask.reshape(shape).mean(axis=(1, 3)))
        
        # Apply collimator rotation for the apertures
        lores_jaw_mask = collimator_rotation(lores_jaw_mask, coll_angle)
        lores_mlc_mask = collimator_rotation(lores_mlc_mask, coll_angle)

        jaw_list.append(lores_jaw_mask)
        mlc_list.append(lores_mlc_mask)
        coll_angle_list.append(coll_angle)

    return beam_list, jaw_list, mlc_list, MU_list, coll_angle_list

def recon_fluence(fluence, coll_angle):
    """
    Reconstructs the fluence map.
    
    ASSUMPTION: The loaded 'fluence' is a 1D vector (e.g., shape 313600,)
    that needs to be reshaped into the 2D grid (560x560).
    
    If your 'fluence' is already 2D, you might just want to
    return it directly or normalize it:
    # return fluence / fluence.max()
    """
    
    fluence[fluence < 0] = 0.0  # Set negative values to zero
    
    try:
        final_fluence = fluence.reshape(GRID_SHAPE)
        final_fluence = np.flipud(final_fluence)  # Flip vertically to match it with RS fluence/aperture orientation after reshaping
        # Apply collimator rotation for the fluence map
        final_fluence = collimator_rotation(final_fluence, coll_angle)
    except ValueError as e:
        print(f"Error: Cannot reshape fluence of shape {fluence.shape} to {GRID_SHAPE}.")
        print("Check the GRID_SHAPE constant at the top of the script.")
        raise e
        
    return final_fluence

def find_beam_index(beamname, beam_list):
    """
    Finds the integer index of a beam name within the beam list from the DICOM file.
    """
    for i, name_in_list in enumerate(beam_list):
        if name_in_list == beamname:
            return i

    # If we get here, the beam was not found.
    raise ValueError(f"Beam name '{beamname}' from file was not found in DICOM list: {beam_list}")


def _create_2d_masks_from_cp(jaw_x_mm, jaw_y_mm, mlc_positions_mm,
                              leaf_y_boundaries_mm, num_leaf_pairs, coll_angle):
    """
    Creates jaw and MLC 2D masks (560×560, 1.0 mm/pixel) from device positions,
    using 10× supersampling for sub-pixel accuracy at device edges.
    Applies collimator rotation before returning.

    Parameters:
        jaw_x_mm            : [X1, X2] jaw positions in mm, or None
        jaw_y_mm            : [Y1, Y2] jaw positions in mm, or None
        mlc_positions_mm    : flat list [bankA_0..bankA_N, bankB_0..bankB_N] in mm, or None
        leaf_y_boundaries_mm: leaf boundary list (length num_leaf_pairs + 1)
        num_leaf_pairs      : int
        coll_angle          : collimator angle in degrees

    Returns:
        (jaw_mask, mlc_mask) — each np.ndarray shape (560, 560), float32
    """
    HIRES_SHAPE = (GRID_SHAPE[0] * SUPERSAMPLE_FACTOR, GRID_SHAPE[1] * SUPERSAMPLE_FACTOR)
    HIRES_RES_MM = RESOLUTION_MM / SUPERSAMPLE_FACTOR
    HIRES_ISO_X = HIRES_SHAPE[1] / 2.0
    HIRES_ISO_Y = HIRES_SHAPE[0] / 2.0

    hires_jaw = np.zeros(HIRES_SHAPE, dtype=np.float32)
    hires_mlc = np.zeros(HIRES_SHAPE, dtype=np.float32)

    # --- JAW MASK ---
    if jaw_x_mm is not None and jaw_y_mm is not None:
        x1_pix = int(np.round(jaw_x_mm[0] / HIRES_RES_MM + HIRES_ISO_X))
        x2_pix = int(np.round(jaw_x_mm[1] / HIRES_RES_MM + HIRES_ISO_X))
        y1_pix = int(np.round(jaw_y_mm[0] / HIRES_RES_MM + HIRES_ISO_Y))
        y2_pix = int(np.round(jaw_y_mm[1] / HIRES_RES_MM + HIRES_ISO_Y))

        x1_c = max(0, x1_pix);  x2_c = min(HIRES_SHAPE[1], x2_pix)
        y1_c = max(0, y1_pix);  y2_c = min(HIRES_SHAPE[0], y2_pix)
        hires_jaw[y1_c:y2_c, x1_c:x2_c] = 1.0

    # --- MLC MASK ---
    if mlc_positions_mm is not None:
        mlc_bank_a = mlc_positions_mm[:num_leaf_pairs]
        mlc_bank_b = mlc_positions_mm[num_leaf_pairs:]

        for i in range(num_leaf_pairs):
            y_inf_mm = float(leaf_y_boundaries_mm[i])
            y_sup_mm = float(leaf_y_boundaries_mm[i + 1])

            y_inf_pix = int(np.round(HIRES_ISO_Y + y_inf_mm / HIRES_RES_MM))
            y_sup_pix = int(np.round(HIRES_ISO_Y + y_sup_mm / HIRES_RES_MM))
            x_a_pix   = int(np.round(mlc_bank_a[i] / HIRES_RES_MM + HIRES_ISO_X))
            x_b_pix   = int(np.round(mlc_bank_b[i] / HIRES_RES_MM + HIRES_ISO_X))

            y_inf_c = max(0, y_inf_pix);  y_sup_c = min(HIRES_SHAPE[0], y_sup_pix)
            x_a_c   = max(0, x_a_pix);   x_b_c   = min(HIRES_SHAPE[1], x_b_pix)

            if y_inf_c >= y_sup_c or x_a_c >= x_b_c:
                continue
            hires_mlc[y_inf_c:y_sup_c, x_a_c:x_b_c] = 1.0

    # --- DOWNSAMPLE by block averaging ---
    shape = (GRID_SHAPE[0], SUPERSAMPLE_FACTOR, GRID_SHAPE[1], SUPERSAMPLE_FACTOR)
    lores_jaw = np.flipud(hires_jaw.reshape(shape).mean(axis=(1, 3)))
    lores_mlc = np.flipud(hires_mlc.reshape(shape).mean(axis=(1, 3)))

    lores_jaw = collimator_rotation(lores_jaw, coll_angle)
    lores_mlc = collimator_rotation(lores_mlc, coll_angle)

    return lores_jaw.astype(np.float32), lores_mlc.astype(np.float32)


def create_vmat_mlc_stack_for_beam(beam):
    """
    Creates VMAT jaw and MLC stacks (180×560×560) from a single DYNAMIC beam.

    The 180 slots represent gantry angles in clockwise order starting from the
    canonical start angle for the plan's parity:
      - Odd  plans: slot 0 = 181°, slot 1 = 183°, ..., slot 179 = 179°
      - Even plans: slot 0 = 182°, slot 1 = 184°, ..., slot 179 = 180° (always zero,
                    as even plans have 179 control points and no CP reaches 180°)

    Slot mapping for a gantry angle G (integer degrees):
        slot = ((G - canonical_start) % 360) // 2

    Control points with no data for their slot (e.g. late arc start) are left as
    all-zero arrays. Counter-clockwise arcs use the same clockwise slot ordering.
    Collimator rotation from CP0 is applied to every control point's mask.

    Parameters:
        beam: pydicom Beam dataset with BeamType == 'DYNAMIC'

    Returns dict:
        'beam_name'      : str
        'jaw_stack'      : np.ndarray (180, 560, 560), float32
        'mlc_stack'      : np.ndarray (180, 560, 560), float32
        'parity'         : 'odd' or 'even'
        'canonical_start': 181 (odd) or 182 (even)
        'actual_start'   : int — first CP's gantry angle (rounded to nearest degree)

    Raises:
        ValueError if BeamType != 'DYNAMIC'
    """
    if getattr(beam, 'BeamType', '') != 'DYNAMIC':
        raise ValueError(
            f"Beam '{getattr(beam, 'BeamName', beam.BeamNumber)}' has BeamType="
            f"'{getattr(beam, 'BeamType', 'UNKNOWN')}'. Only VMAT (DYNAMIC) beams are supported."
        )

    # --- MLC leaf definition ---
    mlc_def = None
    if hasattr(beam, 'BeamLimitingDeviceSequence'):
        for dev in beam.BeamLimitingDeviceSequence:
            if dev.RTBeamLimitingDeviceType in ('MLCX', 'MLCY'):
                mlc_def = dev
                break
    if mlc_def is None:
        raise ValueError(f"No MLC definition found in beam '{beam.BeamName}'.")

    leaf_y_boundaries_mm = list(map(float, mlc_def.LeafPositionBoundaries))
    num_leaf_pairs = int(mlc_def.NumberOfLeafJawPairs)

    # --- Collimator angle from CP0 (fixed for the whole arc) ---
    cp0 = beam.ControlPointSequence[0]
    coll_angle = float(cp0.BeamLimitingDeviceAngle) if hasattr(cp0, 'BeamLimitingDeviceAngle') else 0.0

    # --- Parity and canonical start from first CP's gantry angle ---
    actual_start = int(round(float(cp0.GantryAngle))) % 360
    parity = 'odd' if actual_start % 2 == 1 else 'even'
    canonical_start = 181 if parity == 'odd' else 182

    # --- Initialise stacks (zero = missing / no CP at that angle) ---
    jaw_stack = np.zeros((180, *GRID_SHAPE), dtype=np.float32)
    mlc_stack = np.zeros((180, *GRID_SHAPE), dtype=np.float32)

    # --- Iterate control points ---
    current_gantry      = None
    current_jaw_X       = None
    current_jaw_Y       = None
    current_mlc_pos     = None

    for cp in beam.ControlPointSequence:
        if hasattr(cp, 'GantryAngle'):
            current_gantry = float(cp.GantryAngle)

        if hasattr(cp, 'BeamLimitingDevicePositionSequence'):
            for dev in cp.BeamLimitingDevicePositionSequence:
                t = dev.RTBeamLimitingDeviceType
                if t == 'ASYMX':
                    current_jaw_X = list(map(float, dev.LeafJawPositions))
                elif t == 'ASYMY':
                    current_jaw_Y = list(map(float, dev.LeafJawPositions))
                elif t in ('MLCX', 'MLCY'):
                    current_mlc_pos = list(map(float, dev.LeafJawPositions))

        if current_gantry is None or current_mlc_pos is None:
            continue

        gantry_int = int(round(current_gantry)) % 360
        slot = ((gantry_int - canonical_start) % 360) // 2

        if slot < 0 or slot >= 180:
            print(f"  Warning: gantry {current_gantry}° → slot {slot} out of [0,179]. Skipped.")
            continue

        jaw_2d, mlc_2d = _create_2d_masks_from_cp(
            current_jaw_X, current_jaw_Y, current_mlc_pos,
            leaf_y_boundaries_mm, num_leaf_pairs, coll_angle
        )
        jaw_stack[slot] = jaw_2d
        mlc_stack[slot] = mlc_2d

    return {
        'beam_name':       getattr(beam, 'BeamName', str(beam.BeamNumber)),
        'jaw_stack':       jaw_stack,
        'mlc_stack':       mlc_stack,
        'parity':          parity,
        'canonical_start': canonical_start,
        'actual_start':    actual_start,
    }


"""
rt_plan_file_list = glob.glob(os.path.join(plan_path, "RP*.dcm"))
rt_plan_file = rt_plan_file_list[0]
beam_list, jaw_list, mlc_list, MU_list, coll_angle_list = create_jaw_mlc_list(rt_plan_file)
"""
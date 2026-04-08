import numpy as np
import torch
import torch.nn as nn


def vmat_gantry_angles(average: bool = False) -> np.ndarray:
    """
    Return the gantry angle (degrees) for each aperture slot in the standard
    180-slot VMAT representation (slot 0 = 181°, CCW step = 2°).

    average=False  →  180 values: [181, 183, 185, ..., 357, 359, 1, 3, ..., 177, 179]
                      Each value is the gantry angle of the corresponding CP.

    average=True   →  179 values: [182, 184, 186, ..., 358, 0, 2, ..., 176, 178]
                      Each value is the midpoint angle between adjacent CPs
                      (matches RayStation's segment-midpoint dose calculation).
                      Note: the 0° entry is the midpoint of 359° and 1°.
    """
    if not average:
        return (181.0 + 2.0 * np.arange(180)) % 360.0
    else:
        # Midpoint of slot[i] and slot[i+1]: (181 + 2i + 183 + 2i) / 2 = 182 + 2i
        return (182.0 + 2.0 * np.arange(179)) % 360.0


class DifferentiableMLCAperture(nn.Module):
    """
    TrueBeam Millennium 120 MLC leaf positions → 160×160 aperture (fluence map).

    Pixel size : 2.5 mm
    Field size : 400 mm × 400 mm (160 px × 160 px)

    Leaf-row mapping (Millennium 120):
        10 outer leaves × 4 px/leaf  (1.0 cm)
        40 inner leaves × 2 px/leaf  (0.5 cm)
        10 outer leaves × 4 px/leaf  (1.0 cm)
        Total: 160 rows

    Averaging mode (average=True):
        Requires input with an N_cp dimension [B, N_cp, 60, 2].
        Adjacent CP positions are averaged before aperture creation, giving
        N_cp−1 aperture slices that correspond to the midpoint gantry angles
        returned by vmat_gantry_angles(average=True).
        This matches RayStation's VMAT dose-calculation convention.
    """

    def __init__(self, pixel_size=2.5, grid_size=160, tau=0.1):
        super().__init__()
        self.pixel_size = pixel_size
        self.grid_size  = grid_size
        self.tau        = tau

        indices = []
        for i in range(10):     indices.extend([i] * 4)   # top outer leaves (1 cm)
        for i in range(10, 50): indices.extend([i] * 2)   # inner leaves     (0.5 cm)
        for i in range(50, 60): indices.extend([i] * 4)   # bottom outer     (1 cm)
        self.register_buffer('row_to_leaf', torch.tensor(indices, dtype=torch.long))

        x_coords = (torch.arange(grid_size) - (grid_size - 1) / 2.0) * pixel_size
        self.register_buffer('x_grid', x_coords.view(1, 1, grid_size))

    def forward(self, mlc_positions, average: bool = False):
        """
        Args:
            mlc_positions : [B, 60, 2] or [B, N_cp, 60, 2]  –  leaf positions (mm)
                            axis -1 index 0 = X1 (left bank), index 1 = X2 (right bank)
            average       : if True, average adjacent CP pairs before creating apertures.
                            Requires the N_cp dimension; output has N_cp−1 slices.
                            Use vmat_gantry_angles(average=True) for the matching angles.

        Returns:
            aperture : [B, 1, 160, 160]          when input is [B, 60, 2]
                       [B, N_cp,   1, 160, 160]  when input is [B, N_cp, 60, 2]
                       [B, N_cp-1, 1, 160, 160]  when input is [B, N_cp, 60, 2] + average=True
        """
        if average:
            if mlc_positions.dim() != 4:
                raise ValueError(
                    "average=True requires [B, N_cp, 60, 2] input "
                    f"(got {tuple(mlc_positions.shape)})"
                )
            mlc_positions = (mlc_positions[:, :-1] + mlc_positions[:, 1:]) / 2.0

        extra_dims = mlc_positions.shape[:-2]          # (B,) or (B, N_cp)
        mlc_flat   = mlc_positions.reshape(-1, 60, 2)  # [B*N_cp, 60, 2]

        row_mlc     = mlc_flat[:, self.row_to_leaf, :]  # [B*N_cp, 160, 2]
        left_leaf   = row_mlc[:, :, 0:1]               # [B*N_cp, 160, 1]
        right_leaf  = row_mlc[:, :, 1:2]               # [B*N_cp, 160, 1]

        open_left  = torch.sigmoid((self.x_grid - left_leaf)  / self.tau)
        open_right = torch.sigmoid((right_leaf  - self.x_grid) / self.tau)
        aperture   = (open_left * open_right).unsqueeze(1)  # [B*N_cp, 1, 160, 160]

        return aperture.reshape(*extra_dims, 1, self.grid_size, self.grid_size)


class DifferentiableJawAperture(nn.Module):
    """
    Jaw positions → 160×160 aperture [B, 1, 160, 160].

    Input layout (last two dims always [2, 2]):
        [..., 0, 0] = X1  (left jaw,      mm)
        [..., 0, 1] = X2  (right jaw,     mm)
        [..., 1, 0] = Y1  (inferior jaw,  mm)
        [..., 1, 1] = Y2  (superior jaw,  mm)

    Supports [B, 2, 2] and [B, N_cp, 2, 2] inputs (mirrors DifferentiableMLCAperture).

    Averaging mode (average=True):
        Requires [B, N_cp, 2, 2] input; averages adjacent CP pairs → N_cp−1 slices.
        Matches RayStation's VMAT dose-calculation convention.
        Use vmat_gantry_angles(average=True) for the matching angles.
    """

    def __init__(self, pixel_size=2.5, grid_size=160, tau=0.1):
        super().__init__()
        self.pixel_size = pixel_size
        self.grid_size  = grid_size
        self.tau        = tau

        coords = (torch.arange(grid_size) - (grid_size - 1) / 2.0) * pixel_size
        self.register_buffer('x_grid', coords.view(1, 1, grid_size))  # broadcast over rows
        self.register_buffer('y_grid', coords.view(1, grid_size, 1))  # broadcast over cols

    def forward(self, jaw_positions, average: bool = False):
        """
        Args:
            jaw_positions : [B, 2, 2] or [B, N_cp, 2, 2]  –  jaw positions (mm)
            average       : if True, average adjacent CP pairs before creating apertures.
                            Requires the N_cp dimension; output has N_cp−1 slices.
                            Use vmat_gantry_angles(average=True) for the matching angles.

        Returns:
            aperture : [B, 1, 160, 160]          when input is [B, 2, 2]
                       [B, N_cp,   1, 160, 160]  when input is [B, N_cp, 2, 2]
                       [B, N_cp-1, 1, 160, 160]  when input is [B, N_cp, 2, 2] + average=True
        """
        if average:
            if jaw_positions.dim() != 4:
                raise ValueError(
                    "average=True requires [B, N_cp, 2, 2] input "
                    f"(got {tuple(jaw_positions.shape)})"
                )
            jaw_positions = (jaw_positions[:, :-1] + jaw_positions[:, 1:]) / 2.0

        extra_dims = jaw_positions.shape[:-2]           # (B,) or (B, N_cp)
        jaw_flat   = jaw_positions.reshape(-1, 2, 2)    # [B*N_cp, 2, 2]

        # X jaw gates columns
        x1 = jaw_flat[:, 0, 0:1].unsqueeze(-1)          # [B*N_cp, 1, 1]
        x2 = jaw_flat[:, 0, 1:2].unsqueeze(-1)
        open_x = (torch.sigmoid((self.x_grid - x1) / self.tau) *
                  torch.sigmoid((x2 - self.x_grid) / self.tau))  # [B*N_cp, 1, 160]

        # Y jaw gates rows
        y1 = jaw_flat[:, 1, 0:1].unsqueeze(-1)          # [B*N_cp, 1, 1]
        y2 = jaw_flat[:, 1, 1:2].unsqueeze(-1)
        open_y = (torch.sigmoid((self.y_grid - y1) / self.tau) *
                  torch.sigmoid((y2 - self.y_grid) / self.tau))  # [B*N_cp, 160, 1]

        aperture = (open_y * open_x).unsqueeze(1)        # [B*N_cp, 1, 160, 160]

        return aperture.reshape(*extra_dims, 1, self.grid_size, self.grid_size)

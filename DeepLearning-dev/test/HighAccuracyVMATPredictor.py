"""
High-Accuracy VMAT Dose Predictor (v2)
=======================================
Updated to match the existing pipeline conventions:
  - 560×560 1mm BEV aperture (560mm FOV)
  - 179 averaged control points (RayStation VMAT convention)
  - Uses existing DifferentiableMLCAperture / DifferentiableJawAperture
  - jaw and MLC apertures combined by element-wise multiplication
    (physically: AND of jaw_open and MLC_open)
  - End-to-end from raw MLC/Jaw/MU inputs (no pre-built BEV)
  - 48³ projection volume for memory efficiency
  - 4D spatio-angular convolution with circular padding
  - CT-conditioned TERMA + learned multi-scale dose kernel
  - Refinement decoder upsamples 48³ → 192³ with CT skip connections

Pipeline:
    MLC[B,180,60,2], Jaw[B,180,2,2], MU[B,179]
        → DifferentiableMLCAperture(average=True) → mlc_ap [B,179,1,560,560]
        → DifferentiableJawAperture(average=True) → jaw_ap [B,179,1,560,560]
        → aperture = mlc_ap * jaw_ap                       [B,179,1,560,560]
        → MU weighting (inside model)
        → grid_sample with 48³ perspective grid             [B,179,1,48,48,48]
        → fluence embedding [B,C,179,48,48,48]
        → SpatioAngular conv blocks ×2
        → angular sum → [B,C,48,48,48]
        → CT-conditioned TERMA (concat with CT features at 48³)
        → Learned multi-scale dose kernel
        → Refinement decoder (48³ → 96³ → 192³ with CT skips)
        → dose [B,1,192,192,192]
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from MLC2Aperture import (vmat_gantry_angles,
                          DifferentiableMLCAperture,
                          DifferentiableJawAperture)


# =====================================================================
# 1. Perspective projection grid (560×560 1mm BEV convention)
# =====================================================================
def build_hfs_perspective_grids(gantry_angles_deg,
                                feat_vol_size=(48, 48, 48),
                                raw_ct_size=(192, 192, 192),
                                ct_spacing=(3.0, 3.0, 3.0),
                                raw_bev_size=(560, 560),
                                bev_spacing=(1.0, 1.0),
                                sad=1000.0):
    """Build perspective projection grids from a 3D volume to a 2D BEV plane.

    Args:
        gantry_angles_deg: array/tensor of gantry angles in degrees.
            Use vmat_gantry_angles(average=True) for 179 mid-segment angles.
        feat_vol_size: 3D volume resolution at which projection happens.
            48³ here for memory efficiency.
        raw_bev_size: BEV plane resolution. Must match the aperture
            resolution from DifferentiableMLCAperture (560×560 at 1mm).

    Returns:
        grid: [N, D, H, W, 2] normalized coords for F.grid_sample.
    """
    if not isinstance(gantry_angles_deg, torch.Tensor):
        gantry_angles_deg = torch.tensor(gantry_angles_deg, dtype=torch.float32)

    angles_rad = gantry_angles_deg * (math.pi / 180.0)
    num_cps = gantry_angles_deg.shape[0]

    D, H, W = feat_vol_size

    # Physical FOVs (mm)
    fov_z = raw_ct_size[0] * ct_spacing[0]    # 576 mm
    fov_y = raw_ct_size[1] * ct_spacing[1]    # 576 mm
    fov_x = raw_ct_size[2] * ct_spacing[2]    # 576 mm
    fov_v = raw_bev_size[0] * bev_spacing[0]  # 560 mm (SI)
    fov_u = raw_bev_size[1] * bev_spacing[1]  # 560 mm (LR)

    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, D),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )

    phys_x = (grid_x * (fov_x / 2.0)).unsqueeze(0)
    phys_y = (grid_y * (fov_y / 2.0)).unsqueeze(0)
    phys_z = (grid_z * (fov_z / 2.0)).unsqueeze(0)

    cos_t = torch.cos(angles_rad).view(-1, 1, 1, 1)
    sin_t = torch.sin(angles_rad).view(-1, 1, 1, 1)

    # Rotate around z-axis by gantry angle
    rot_x = phys_x * cos_t + phys_y * sin_t
    rot_y = -phys_x * sin_t + phys_y * cos_t
    rot_z = phys_z.expand(num_cps, -1, -1, -1)

    # Perspective projection (point source at SAD)
    magnification = sad / (sad + rot_y)
    u_phys = rot_x * magnification
    v_phys = rot_z * magnification

    # Normalize to [-1, 1] for grid_sample
    u_norm = u_phys / (fov_u / 2.0)
    v_norm = v_phys / (fov_v / 2.0)

    return torch.stack((u_norm, v_norm), dim=-1).float()  # [N, D, H, W, 2]


# =====================================================================
# 2. 4D Spatio-Angular Convolution Block
# =====================================================================
class SpatioAngularConvBlock(nn.Module):
    """Separable 4D convolution treating angle as an explicit dimension.

    Input/Output: [B, C, A, D, H, W]
        A = 179 angles, DHW = 48³ spatial

    Decomposes a 4D conv into:
        1. Spatial 3D conv  (each angle independent)
        2. Angular 1D conv  (each spatial location independent, circular padding)
        3. Spatial 3D conv  (refine after angular mixing)

    Why circular padding on angular axis:
        Gantry angles wrap around (181° → ... → 179° → 181°). Circular
        padding lets the network see this continuity at the boundary.
    """

    def __init__(self, channels, angular_kernel=5):
        super().__init__()
        assert angular_kernel % 2 == 1, "angular_kernel must be odd"

        self.spatial1 = nn.Conv3d(channels, channels, kernel_size=3,
                                  padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)

        self.angular = nn.Conv1d(channels, channels,
                                 kernel_size=angular_kernel,
                                 padding=angular_kernel // 2,
                                 padding_mode='circular',
                                 bias=False)

        self.spatial2 = nn.Conv3d(channels, channels, kernel_size=3,
                                  padding=1, bias=False)
        self.norm3 = nn.GroupNorm(8, channels)

        self.act = nn.GELU()

    def forward(self, x):
        """x: [B, C, A, D, H, W]"""
        B, C, A, D, H, W = x.shape

        # === Spatial conv 1 (each angle independent) ===
        x_sp = x.permute(0, 2, 1, 3, 4, 5).reshape(B * A, C, D, H, W)
        x_sp = self.act(self.norm1(self.spatial1(x_sp)))
        x = x_sp.reshape(B, A, C, D, H, W).permute(0, 2, 1, 3, 4, 5)

        # === Angular conv (each spatial position independent) ===
        x_ang = x.permute(0, 3, 4, 5, 1, 2).reshape(B * D * H * W, C, A)
        x_ang = self.angular(x_ang)
        x_ang = self.act(F.group_norm(x_ang, num_groups=8))
        x = x_ang.view(B, D, H, W, C, A).permute(0, 4, 5, 1, 2, 3)

        # === Spatial conv 2 ===
        x_sp = x.permute(0, 2, 1, 3, 4, 5).reshape(B * A, C, D, H, W)
        x_sp = self.act(self.norm3(self.spatial2(x_sp)))
        x = x_sp.reshape(B, A, C, D, H, W).permute(0, 2, 1, 3, 4, 5)

        return x


# =====================================================================
# 3. CT Encoder (multi-scale features for skip connections + TERMA)
# =====================================================================
class CTEncoder(nn.Module):
    """CT → multi-scale features at 192³, 96³, 48³."""

    def __init__(self):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.GroupNorm(8, 16), nn.GELU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.GroupNorm(8, 16), nn.GELU(),
        )  # [B, 16, 192³]

        self.enc1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, stride=2),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
        )  # [B, 32, 96³]

        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, stride=2),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
        )  # [B, 64, 48³]

    def forward(self, ct):
        f0 = self.enc0(ct)  # 192³
        f1 = self.enc1(f0)  # 96³
        f2 = self.enc2(f1)  # 48³
        return f0, f1, f2


# =====================================================================
# 4. CT-Conditioned TERMA Module
# =====================================================================
class TERMAModule(nn.Module):
    """Computes TERMA-like features from fluence and CT context.

    Physical analog: TERMA(r) = fluence(r) × μ(r)
    where μ depends on local CT density. The relationship is nonlinear
    for polyenergetic photon beams, so we learn it.
    """

    def __init__(self, fluence_channels=16, ct_channels=64, terma_channels=16):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Conv3d(fluence_channels + ct_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, terma_channels, 3, padding=1),
            nn.GroupNorm(8, terma_channels), nn.GELU(),
        )

    def forward(self, fluence_feat, ct_feat):
        return self.combine(torch.cat([fluence_feat, ct_feat], dim=1))


# =====================================================================
# 5. Multi-Scale Learned Dose Kernel
# =====================================================================
class LearnedDoseKernel(nn.Module):
    """Convolves TERMA with a learned dose deposition kernel.

    Inspired by collapsed-cone convolution superposition (CCCS).
    Multiple kernel sizes capture primary, first-scatter, multiple-scatter.
    Uses separable 1D convs in 3 axes to keep cost manageable.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.channels = channels

        # Small (primary)
        self.small_d = nn.Conv3d(channels, channels, (5, 1, 1), padding=(2, 0, 0))
        self.small_h = nn.Conv3d(channels, channels, (1, 5, 1), padding=(0, 2, 0))
        self.small_w = nn.Conv3d(channels, channels, (1, 1, 5), padding=(0, 0, 2))

        # Medium (first scatter)
        self.med_d = nn.Conv3d(channels, channels, (11, 1, 1), padding=(5, 0, 0))
        self.med_h = nn.Conv3d(channels, channels, (1, 11, 1), padding=(0, 5, 0))
        self.med_w = nn.Conv3d(channels, channels, (1, 1, 11), padding=(0, 0, 5))

        # Large (multiple scatter)
        self.large_d = nn.Conv3d(channels, channels, (21, 1, 1), padding=(10, 0, 0))
        self.large_h = nn.Conv3d(channels, channels, (1, 21, 1), padding=(0, 10, 0))
        self.large_w = nn.Conv3d(channels, channels, (1, 1, 21), padding=(0, 0, 10))

        self.combine = nn.Sequential(
            nn.Conv3d(channels * 3, channels, 1),
            nn.GroupNorm(8, channels), nn.GELU(),
        )

    def _separable(self, x, conv_d, conv_h, conv_w):
        return conv_w(conv_h(conv_d(x)))

    def forward(self, terma):
        small = self._separable(terma, self.small_d, self.small_h, self.small_w)
        med = self._separable(terma, self.med_d, self.med_h, self.med_w)
        large = self._separable(terma, self.large_d, self.large_h, self.large_w)
        return self.combine(torch.cat([small, med, large], dim=1))


# =====================================================================
# 6. Refinement Decoder (48³ → 192³)
# =====================================================================
class RefinementDecoder(nn.Module):
    """Upsamples dose features from 48³ to 192³ using CT skip connections."""

    def __init__(self, dose_channels=16):
        super().__init__()
        # 48³ → 96³
        self.up1 = nn.ConvTranspose3d(dose_channels, dose_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(dose_channels + 32, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
        )

        # 96³ → 192³
        self.up0 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec0 = nn.Sequential(
            nn.Conv3d(16 + 16, 16, 3, padding=1),
            nn.GroupNorm(8, 16), nn.GELU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.GroupNorm(8, 16), nn.GELU(),
        )

        self.dose_out = nn.Sequential(
            nn.Conv3d(16, 1, 1),
            nn.ReLU()
        )

    def forward(self, dose_feat_48, ct_f1, ct_f0):
        x = self.up1(dose_feat_48)
        x = self.dec1(torch.cat([x, ct_f1], dim=1))

        x = self.up0(x)
        x = self.dec0(torch.cat([x, ct_f0], dim=1))

        return self.dose_out(x)


# =====================================================================
# 7. Main Model: HighAccuracyVMATPredictor
# =====================================================================
class HighAccuracyVMATPredictor(nn.Module):
    """End-to-end VMAT dose predictor.

    Input:  CT [B,1,192,192,192]
            MLC [B,180,60,2]    raw leaf positions in mm
            Jaw [B,180,2,2]     raw jaw positions in mm
            MU  [B,179]         monitor units per averaged CP

    Output: dose [B,1,192,192,192]

    Hard-coded (no learnable weights):
        - MLC/Jaw → 2D aperture
        - 2D → 3D perspective projection
        - MU weighting

    Learned:
        - Per-CP fluence embedding
        - 4D spatio-angular conv
        - CT encoder
        - CT-conditioned TERMA
        - Multi-scale dose kernel
        - Refinement decoder
    """

    def __init__(self,
                 base_channels=16,
                 mlc_pixel_size=1.0,
                 mlc_grid_size=560,
                 mlc_tau=0.1,
                 jaw_tau=0.1,
                 vol_size=(48, 48, 48),
                 projection_chunk_size=20,
                 embed_chunk_size=40):
        super().__init__()
        self.vol_size = vol_size
        self.base_channels = base_channels
        self.projection_chunk_size = projection_chunk_size
        self.embed_chunk_size = embed_chunk_size

        # --- Hard-coded aperture generation ---
        self.mlc_layer = DifferentiableMLCAperture(
            pixel_size=mlc_pixel_size, grid_size=mlc_grid_size, tau=mlc_tau
        )
        self.jaw_layer = DifferentiableJawAperture(
            pixel_size=mlc_pixel_size, grid_size=mlc_grid_size, tau=jaw_tau
        )

        # --- Hard-coded perspective projection grid (179 averaged angles) ---
        gantry_angles = vmat_gantry_angles(average=True)  # [179]
        proj_grid = build_hfs_perspective_grids(
            gantry_angles_deg=gantry_angles,
            feat_vol_size=vol_size,
            raw_ct_size=(192, 192, 192),
            ct_spacing=(3.0, 3.0, 3.0),
            raw_bev_size=(mlc_grid_size, mlc_grid_size),
            bev_spacing=(mlc_pixel_size, mlc_pixel_size),
            sad=1000.0,
        )
        self.register_buffer('proj_grid', proj_grid)  # [179, D, H, W, 2]
        self.num_cps = proj_grid.shape[0]  # 179

        # --- Per-CP fluence feature embedding (after projection to 48³) ---
        self.fluence_embed = nn.Sequential(
            nn.Conv3d(1, base_channels // 2, 3, padding=1),
            nn.GroupNorm(4, base_channels // 2), nn.GELU(),
            nn.Conv3d(base_channels // 2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels), nn.GELU(),
        )

        # --- 4D spatio-angular conv stack ---
        self.sa_block1 = SpatioAngularConvBlock(base_channels)
        self.sa_block2 = SpatioAngularConvBlock(base_channels)

        # --- CT encoder ---
        self.ct_encoder = CTEncoder()

        # --- TERMA module (CT-conditioned at 48³) ---
        self.terma = TERMAModule(
            fluence_channels=base_channels,
            ct_channels=64,
            terma_channels=base_channels
        )

        # --- Learned dose kernel ---
        self.kernel = LearnedDoseKernel(channels=base_channels)

        # --- Refinement decoder ---
        self.refine = RefinementDecoder(dose_channels=base_channels)

    def _build_aperture(self, mlc_pos, jaw_pos):
        """Build combined aperture by element-wise multiplication.

        Args:
            mlc_pos: [B, 180, 60, 2]
            jaw_pos: [B, 180, 2, 2]
        Returns:
            aperture: [B, 179, 1, 560, 560]  in [0, 1]
        """
        mlc_ap = self.mlc_layer(mlc_pos, average=True)  # [B, 179, 1, 560, 560]
        jaw_ap = self.jaw_layer(jaw_pos, average=True)  # [B, 179, 1, 560, 560]
        return mlc_ap * jaw_ap

    def _project_to_3d(self, aperture, mu):
        """Project [B, 179, 1, 560, 560] aperture to [B, 179, 1, 48, 48, 48]
        applying MU weighting per CP. Processed in chunks of CPs.
        """
        B, N, _, H_bev, W_bev = aperture.shape
        D, H, W = self.vol_size
        assert N == self.num_cps, f"Expected {self.num_cps} CPs, got {N}"

        out_chunks = []

        for i in range(0, N, self.projection_chunk_size):
            cs = min(self.projection_chunk_size, N - i)

            # Aperture chunk: [B, cs, 1, 560, 560] → [B*cs, 1, 560, 560]
            ap_chunk = aperture[:, i:i+cs].reshape(B * cs, 1, H_bev, W_bev)

            # Grid chunk: [cs, D, H, W, 2] → [B*cs, D*H, W, 2]
            grid_chunk = self.proj_grid[i:i+cs]
            grid_chunk = grid_chunk.view(cs, D * H, W, 2)
            grid_chunk = grid_chunk.unsqueeze(0).expand(B, -1, -1, -1, -1)
            grid_chunk = grid_chunk.reshape(B * cs, D * H, W, 2)

            # grid_sample → [B*cs, 1, D*H, W]
            fluence = F.grid_sample(
                ap_chunk, grid_chunk,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )
            fluence = fluence.view(B, cs, 1, D, H, W)

            # MU weighting
            mu_chunk = mu[:, i:i+cs].view(B, cs, 1, 1, 1, 1)
            fluence = fluence * mu_chunk

            out_chunks.append(fluence)

        return torch.cat(out_chunks, dim=1)  # [B, 179, 1, 48, 48, 48]

    def _embed_fluence(self, fluence_3d):
        """Embed [B, 179, 1, 48³] → [B, C, 179, 48³] in chunks of CPs."""
        B, N, _, D, H, W = fluence_3d.shape
        C = self.base_channels

        embedded_chunks = []
        for i in range(0, N, self.embed_chunk_size):
            cs = min(self.embed_chunk_size, N - i)
            chunk = fluence_3d[:, i:i+cs].reshape(B * cs, 1, D, H, W)
            embedded = self.fluence_embed(chunk)  # [B*cs, C, D, H, W]
            embedded = embedded.view(B, cs, C, D, H, W)
            embedded_chunks.append(embedded)

        # [B, 179, C, D, H, W] → [B, C, 179, D, H, W]
        all_embedded = torch.cat(embedded_chunks, dim=1)
        return all_embedded.permute(0, 2, 1, 3, 4, 5).contiguous()

    def forward(self, ct, mlc_pos, jaw_pos, mu):
        """
        Args:
            ct:      [B, 1, 192, 192, 192]
            mlc_pos: [B, 180, 60, 2]
            jaw_pos: [B, 180, 2, 2]
            mu:      [B, 179]
        Returns:
            dose: [B, 1, 192, 192, 192]
        """
        # === 1. Build 2D aperture (hard-coded) ===
        aperture = self._build_aperture(mlc_pos, jaw_pos)
        # [B, 179, 1, 560, 560]

        # === 2. Project to 3D 48³ + MU weighting (hard-coded) ===
        fluence_3d = self._project_to_3d(aperture, mu)
        # [B, 179, 1, 48, 48, 48]

        # === 3. Per-CP feature embedding (learned) ===
        fluence_4d = self._embed_fluence(fluence_3d)
        # [B, C, 179, 48, 48, 48]

        # === 4. CT encoder (multi-scale features) ===
        ct_f0, ct_f1, ct_f2 = self.ct_encoder(ct)
        # [B,16,192³], [B,32,96³], [B,64,48³]

        # === 5. 4D spatio-angular conv ===
        x = self.sa_block1(fluence_4d)
        x = self.sa_block2(x) + x  # residual

        # === 6. Angular sum (collapse 179 angles) ===
        fluence_summed = x.sum(dim=2)  # [B, C, 48, 48, 48]

        # === 7. CT-conditioned TERMA ===
        terma = self.terma(fluence_summed, ct_f2)

        # === 8. Learned dose kernel ===
        dose_features = self.kernel(terma)

        # === 9. Refinement decoder ===
        dose = self.refine(dose_features, ct_f1, ct_f0)

        return dose


# =====================================================================
# 8. Physics-Informed Loss
# =====================================================================
class PhysicsInformedDoseLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        gz = self.l1(
            torch.abs(pred[:, :, 1:] - pred[:, :, :-1]),
            torch.abs(target[:, :, 1:] - target[:, :, :-1])
        )
        gy = self.l1(
            torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]),
            torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        )
        gx = self.l1(
            torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]),
            torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])
        )
        return self.alpha * loss_l1 + self.beta * (gz + gy + gx)


# =====================================================================
# 9. Plan-Optimisation Epoch Timer (frozen dose engine)
# =====================================================================
def time_plan_optimization_epoch(
    model: 'HighAccuracyVMATPredictor',
    ct: torch.Tensor,
    target_dose: torch.Tensor,
    n_warmup: int = 3,
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> dict:
    """Benchmark one inverse-planning epoch with the dose engine frozen.

    Plan parameters (MLC leaf positions, jaw positions, MU weights) are
    differentiable; gradients flow through the frozen dose engine back to
    them.  The model handles MLC/Jaw → aperture conversion internally via
    DifferentiableMLCAperture / DifferentiableJawAperture (average=True).
    """
    device   = next(model.parameters()).device
    use_cuda = (device.type == 'cuda')
    B        = ct.shape[0]

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # Raw plan parameters: always 180 CPs (averaging to 179 segments happens inside the model)
    mlc_init = torch.zeros(B, 180, 60, 2, device=device)
    mlc_init[..., 0] = -100.0   # X1 leaves: open 100 mm left
    mlc_init[..., 1] =  100.0   # X2 leaves: open 100 mm right
    mlc_positions = nn.Parameter(mlc_init)

    jaw_init = torch.zeros(B, 180, 2, 2, device=device)
    jaw_init[:, :, 0, 0] = -100.0   # X1
    jaw_init[:, :, 0, 1] =  100.0   # X2
    jaw_init[:, :, 1, 0] = -100.0   # Y1
    jaw_init[:, :, 1, 1] =  100.0   # Y2
    jaw_positions = nn.Parameter(jaw_init)

    # 179 segment MU values (one per averaged aperture slot)
    mu_logits = nn.Parameter(torch.zeros(B, 179, device=device))

    optimizer = torch.optim.Adam([mlc_positions, jaw_positions, mu_logits], lr=lr)
    loss_fn   = PhysicsInformedDoseLoss(alpha=1.0, beta=0.5)

    def _run_epoch():
        optimizer.zero_grad(set_to_none=True)

        # MU per segment: positive via softplus
        mu = F.softplus(mu_logits)  # [B, 179]

        # Model internally builds apertures and projects to 3D
        pred_dose = model(ct, mlc_positions, jaw_positions, mu)
        loss = loss_fn(pred_dose, target_dose)
        loss.backward()
        optimizer.step()
        return loss.item()

    for _ in range(n_warmup):
        _run_epoch()
        if use_cuda:
            torch.cuda.synchronize(device)

    times      = []
    final_loss = None
    for _ in range(n_epochs):
        if use_cuda:
            torch.cuda.synchronize(device)
        t0         = time.perf_counter()
        final_loss = _run_epoch()
        if use_cuda:
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
        print("time:", time.perf_counter() - t0)

    mean_t   = sum(times) / len(times)
    variance = sum((t - mean_t) ** 2 for t in times) / max(len(times) - 1, 1)
    std_t    = variance ** 0.5

    print(f"[HighAccuracyVMATPredictor] Plan-optimisation epoch timing  ({device})")
    print(f"  epochs    : {n_epochs}  (+{n_warmup} warmup)")
    print(f"  mean/std  : {mean_t:.3f} ± {std_t:.3f} s")
    print(f"  total     : {sum(times):.2f} s")
    print(f"  final loss: {final_loss:.6f}")

    return {'times_s': times, 'mean_s': mean_t, 'std_s': std_t, 'final_loss': final_loss}


# =====================================================================
# 10. Smoke test
# =====================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    B = 1
    print("Initializing model...")
    model = HighAccuracyVMATPredictor().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffer = sum(b.numel() for b in model.buffers())
    print(f"Trainable parameters: {total_params:,}")
    print(f"Buffer elements:      {total_buffer:,} "
          f"(~{total_buffer * 4 / 1e9:.2f} GB at fp32)")

    print("\nGenerating dummy inputs...")
    ct = torch.rand(B, 1, 192, 192, 192, device=device)
    mlc_pos = torch.zeros(B, 180, 60, 2, device=device)
    mlc_pos[..., 0] = -50.0  # left bank
    mlc_pos[..., 1] = 50.0   # right bank
    jaw_pos = torch.zeros(B, 180, 2, 2, device=device)
    jaw_pos[..., 0, 0] = -100.0
    jaw_pos[..., 0, 1] = 100.0
    jaw_pos[..., 1, 0] = -100.0
    jaw_pos[..., 1, 1] = 100.0
    mu = torch.ones(B, 179, device=device) * 5.0

    print("Running forward pass...")
    with torch.no_grad():
        dose = model(ct, mlc_pos, jaw_pos, mu)
    print(f"Dose output shape: {dose.shape}")
    print(f"Dose range: [{dose.min():.4f}, {dose.max():.4f}]")
    print("Forward pass successful!")

    print("\nGenerating dummy target dose for plan-optimisation benchmark...")
    target_dose = torch.rand(B, 1, 192, 192, 192, device=device)

    print("Starting plan-optimisation epoch benchmark...\n")
    results = time_plan_optimization_epoch(
        model       = model,
        ct          = ct,
        target_dose = target_dose,
        n_warmup    = 2,
        n_epochs    = 10,
        lr          = 1e-3,
    )

    print("\nPer-epoch times (s):", [f"{t:.3f}" for t in results['times_s']])

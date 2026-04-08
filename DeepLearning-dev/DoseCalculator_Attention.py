import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from MLC2Aperture import (vmat_gantry_angles,
                          DifferentiableMLCAperture,
                          DifferentiableJawAperture)


# =====================================================================
# 1. Geometric 3D Projection Grid Builder
# =====================================================================
def build_hfs_perspective_grids(gantry_angles_deg,
                                feat_vol_size=(96, 96, 96),
                                raw_ct_size=(192, 192, 192), ct_spacing=(3.0, 3.0, 3.0),
                                raw_bev_size=(160, 160), bev_spacing=(2.5, 2.5),
                                sad=1000.0, is_parallel=False):
    """
    Builds a perspective projection grid from BEV space into a 3D feature volume.

    Args:
        gantry_angles_deg : 1-D array or tensor of gantry angles (degrees).
                            Obtain via vmat_gantry_angles(average=False) for 180
                            per-CP angles, or vmat_gantry_angles(average=True) for
                            179 mid-segment angles (RayStation VMAT convention).

    Returns:
        grid : [N, D, H, W, 2]  – normalised coords for F.grid_sample,
               where N = len(gantry_angles_deg).
    """
    if not isinstance(gantry_angles_deg, torch.Tensor):
        gantry_angles_deg = torch.tensor(gantry_angles_deg, dtype=torch.float32)

    angles_rad = gantry_angles_deg * (math.pi / 180.0)
    num_cps    = gantry_angles_deg.shape[0]

    D, H, W = feat_vol_size

    fov_z = raw_ct_size[0] * ct_spacing[0]    # 576 mm
    fov_y = raw_ct_size[1] * ct_spacing[1]    # 576 mm
    fov_x = raw_ct_size[2] * ct_spacing[2]    # 576 mm
    fov_v = raw_bev_size[0] * bev_spacing[0]  # 400 mm (SI)
    fov_u = raw_bev_size[1] * bev_spacing[1]  # 400 mm (LR)

    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, D),
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )

    phys_x = (grid_x * (fov_x / 2.0)).unsqueeze(0)   # [1, D, H, W]
    phys_y = (grid_y * (fov_y / 2.0)).unsqueeze(0)
    phys_z = (grid_z * (fov_z / 2.0)).unsqueeze(0)

    cos_t = torch.cos(angles_rad).view(-1, 1, 1, 1)   # [N, 1, 1, 1]
    sin_t = torch.sin(angles_rad).view(-1, 1, 1, 1)

    rot_x = phys_x * cos_t + phys_y * sin_t            # [N, D, H, W]
    rot_y = -phys_x * sin_t + phys_y * cos_t
    rot_z = phys_z.expand(num_cps, -1, -1, -1)

    if is_parallel:
        u_phys, v_phys = rot_x, rot_z
    else:
        magnification = sad / (sad + rot_y)
        u_phys = rot_x * magnification
        v_phys = rot_z * magnification

    u_norm = u_phys / (fov_u / 2.0)
    v_norm = v_phys / (fov_v / 2.0)

    return torch.stack((u_norm, v_norm), dim=-1).float()  # [N, D, H, W, 2]


# =====================================================================
# 2. 2D BEV Feature Encoder
# =====================================================================
class BEVEncoder2D(nn.Module):
    """Encodes each CP's 2-channel aperture (jaw + MLC) into a 32-channel feature map.

    Input:  [B, N_cp, 2, 160, 160]   (N_cp = 179 with averaging, 180 without)
    Output: [B, N_cp, 32,  40,  40]
    """
    def __init__(self, in_channels=2):
        super().__init__()
        self.block1 = nn.Sequential(           # 160 → 80
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(           # 80 → 40
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.block2(self.block1(x))
        _, C_out, H_out, W_out = x.shape
        return x.view(B, N, C_out, H_out, W_out)


# =====================================================================
# 3. Per-CP Projection Layer
# =====================================================================
class PerCPProjectionLayer(nn.Module):
    """Projects each CP's BEV feature map into the 3D bottleneck volume (24³)
    using a perspective grid built from vmat_gantry_angles.

    Results are *stacked* (not summed) for the downstream attention layer.

    Args:
        average  : if True, builds the grid for 179 mid-segment gantry angles
                   (vmat_gantry_angles(average=True)); otherwise for 180 CP angles.
        vol_size : spatial resolution of the output volume.

    Input:  bev_features  [B, N_cp, 32,  40,  40]
    Output: beam_stack    [B, N_cp, 32,  24,  24, 24]
    """
    def __init__(self, average=False, vol_size=(24, 24, 24)):
        super().__init__()
        self.vol_size = vol_size

        angles_deg     = torch.tensor(vmat_gantry_angles(average=average), dtype=torch.float32)
        self.num_cps   = int(angles_deg.shape[0])   # 179 or 180

        grid = build_hfs_perspective_grids(
            gantry_angles_deg=angles_deg,
            feat_vol_size=vol_size,
            raw_ct_size=(192, 192, 192), ct_spacing=(3.0, 3.0, 3.0),
            raw_bev_size=(160, 160),     bev_spacing=(2.5, 2.5)
        )
        self.register_buffer('sampling_grid', grid)  # [N_cp, D, H, W, 2]

    def forward(self, bev_features):
        B, N, C, H_bev, W_bev = bev_features.shape
        D, H_vol, W_vol = self.vol_size
        assert N == self.num_cps, \
            f"PerCPProjectionLayer expects {self.num_cps} CPs, got {N}"

        bev_flat = bev_features.reshape(B * N, C, H_bev, W_bev)

        grid = self.sampling_grid.view(N, D * H_vol, W_vol, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        grid = grid.reshape(B * N, D * H_vol, W_vol, 2)

        projected = F.grid_sample(
            bev_flat, grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        return projected.view(B, N, C, D, H_vol, W_vol)     # [B, N_cp, C, 24, 24, 24]


# =====================================================================
# 4. Spatially-Resolved Beam Attention
# =====================================================================
class SpatialBeamAttention(nn.Module):
    """Cross-attention between CT bottleneck voxels and the per-CP projected beam stack.

    For each voxel in the 24³ bottleneck, computes a soft attention weight
    over all N_cp CPs, then produces a weighted sum of their projected features.

    attn_weights[b, voxel_idx, cp_idx] is interpretable as the contribution
    of CP cp_idx to that 3D voxel's dose.

    Args:
        ct_channels   : channels of the CT bottleneck (128)
        beam_channels : channels of the per-CP projected features (32)
        num_cps       : number of control points (179 with averaging, 180 without)
    """
    def __init__(self, ct_channels=128, beam_channels=32, num_cps=179):
        super().__init__()
        self.scale      = beam_channels ** -0.5
        self.query_proj = nn.Conv3d(ct_channels, beam_channels, kernel_size=1)
        self.norm       = nn.LayerNorm(beam_channels)

    def forward(self, ct_feat, beam_stack):
        """
        Args:
            ct_feat    : [B, C_ct,    D, H, W]
            beam_stack : [B, N_cp, C_b, D, H, W]
        Returns:
            fused      : [B, C_b, D, H, W]
            attn_w     : [B, D*H*W, N_cp]
        """
        B, C_ct, D, H, W = ct_feat.shape
        B, N, C_b, _, _, _ = beam_stack.shape
        V = D * H * W

        q  = self.query_proj(ct_feat).view(B, C_b, V).permute(0, 2, 1)  # [B, V, C_b]
        kv = beam_stack.view(B, N, C_b, V)                               # [B, N, C_b, V]

        attn_logits = torch.einsum('bvc,bncv->bvn', q, kv) * self.scale
        attn_w      = torch.softmax(attn_logits, dim=-1)                 # [B, V, N]

        fused = torch.einsum('bvn,bncv->bvc', attn_w, kv)
        fused = self.norm(fused).permute(0, 2, 1).view(B, C_b, D, H, W)

        return fused, attn_w


# =====================================================================
# 5. Shared 3D Conv Block
# =====================================================================
class Conv3DBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_c), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_c), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# =====================================================================
# 6. Full Model: VMATDosePredictorAttention
# =====================================================================
class VMATDosePredictorAttention(nn.Module):
    """3D dose predictor for VMAT with spatially-resolved beam attention.

    Args:
        average : if True (recommended), uses 179 mid-segment gantry angles
                  (RayStation VMAT convention: dose computed from the average of
                  adjacent CP apertures).  If False, uses 180 per-CP angles.
                  Must match the average= flag passed to DifferentiableMLCAperture
                  and DifferentiableJawAperture when building bev_apertures.

    Data flow (average=True, N_cp=179):
        BEV apertures [B, 179, 2, 160, 160]   MU [B, 179]
            → BEVEncoder2D           → [B, 179, 32,  40,  40]
            → PerCPProjectionLayer   → [B, 179, 32,  24,  24, 24]
            × mu[:, :, None, None, None, None]   ← MU scales 3D volumes
                                                      ↓
        CT [B,1,192³] → U-Net encoder → bottleneck [B,128,24³]
                                                      ↓
                          SpatialBeamAttention  →  fused [B, 32, 24³]
                                                      ↓
                     concat + fusion_conv       →  [B,128,24³]
                                                      ↓
                          U-Net decoder         →  dose [B,1,192³]

    For autoplanning (frozen model):
        CT is fixed.  MLC/jaw positions, jaw positions, and MU are all
        differentiable plan parameters. Gradients flow:
            loss → dose → decoder → fused → (beam_stack × MU) → MU
                                                               → beam_stack → BEV apertures
                                                                              → MLC/jaw positions
    """
    def __init__(self, average=True):
        super().__init__()
        num_cps = 179 if average else 180

        # --- Beam pathway ---
        self.bev_encoder = BEVEncoder2D(in_channels=2)
        self.per_cp_proj = PerCPProjectionLayer(average=average, vol_size=(24, 24, 24))
        self.beam_attn   = SpatialBeamAttention(ct_channels=128, beam_channels=32, num_cps=num_cps)

        # Merge attended beam features into bottleneck channels (128 + 32 = 160)
        self.fusion_conv = Conv3DBlock(160, 128)

        # --- CT encoder (U-Net) ---
        self.enc0       = Conv3DBlock(1, 16)                       # [B,  16, 192³]
        self.down1      = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.enc1       = Conv3DBlock(32, 32)                      # [B,  32,  96³]
        self.down2      = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.enc2       = Conv3DBlock(64, 64)                      # [B,  64,  48³]
        self.down3      = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        self.bottleneck = Conv3DBlock(128, 128)                    # [B, 128,  24³]

        # --- Decoder ---
        self.up2  = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = Conv3DBlock(128, 64)    # 64 up + 64 skip2

        self.up1  = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = Conv3DBlock(64, 32)     # 32 up + 32 skip1

        self.up0  = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec0 = Conv3DBlock(32, 16)     # 16 up + 16 skip0

        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, ct, bev_apertures, mu):
        """
        Args:
            ct            : [B, 1, 192, 192, 192]
            bev_apertures : [B, N_cp, 2, 160, 160]   normalised jaw & MLC apertures (0–1)
                            N_cp = 179 with average=True, 180 with average=False
            mu            : [B, N_cp]                 MU per segment (cGy or monitor units).
                            With average=True: pass mu_array[:, :179] — the last (0-valued)
                            element of the 180-slot array is dropped by the caller.
                            Applied after 3D projection so the aperture encoder always
                            sees normalised geometry; MU is a separate amplitude pathway.
                            Fully differentiable — autoplanning optimises MLC/jaw/MU
                            jointly by backpropagating through the frozen model.
        Returns:
            dose         : [B, 1, 192, 192, 192]
            attn_weights : [B, D*H*W, N_cp]
        """
        B, N_cp = mu.shape

        # --- 1. Beam pathway ---
        bev_feat   = self.bev_encoder(bev_apertures)            # [B, N_cp, 32, 40, 40]
        beam_stack = self.per_cp_proj(bev_feat)                 # [B, N_cp, 32, 24, 24, 24]
        # Scale each CP's projected 3D volume by its segment MU.
        # Separates geometry (aperture encoder) from magnitude (MU), giving the
        # autoplanning optimiser independent, well-conditioned gradients for both.
        beam_stack = beam_stack * mu.view(B, N_cp, 1, 1, 1, 1)  # [B, N_cp, 32, 24, 24, 24]

        # --- 2. CT encoder ---
        skip0  = self.enc0(ct)                         # [B,  16, 192, 192, 192]
        skip1  = self.enc1(self.down1(skip0))          # [B,  32,  96,  96,  96]
        skip2  = self.enc2(self.down2(skip1))          # [B,  64,  48,  48,  48]
        bottle = self.bottleneck(self.down3(skip2))    # [B, 128,  24,  24,  24]

        # --- 3. Attention fusion at bottleneck ---
        beam_fused, attn_weights = self.beam_attn(bottle, beam_stack)
        fused = self.fusion_conv(
            torch.cat([bottle, beam_fused], dim=1)     # [B, 160, 24³] → [B, 128, 24³]
        )

        # --- 4. Decoder ---
        up2  = self.up2(fused)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))

        up1  = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))

        up0  = self.up0(dec1)
        dec0 = self.dec0(torch.cat([up0, skip0], dim=1))

        dose = self.final_conv(dec0)
        return dose, attn_weights


# =====================================================================
# 7. Physics-Informed Loss
# =====================================================================
class PhysicsInformedDoseLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha   = alpha
        self.beta    = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_dose, true_dose):
        loss_l1 = self.l1_loss(pred_dose, true_dose)

        grad_z = self.l1_loss(
            torch.abs(pred_dose[:, :, 1:, :, :] - pred_dose[:, :, :-1, :, :]),
            torch.abs(true_dose[:, :, 1:, :, :] - true_dose[:, :, :-1, :, :])
        )
        grad_y = self.l1_loss(
            torch.abs(pred_dose[:, :, :, 1:, :] - pred_dose[:, :, :, :-1, :]),
            torch.abs(true_dose[:, :, :, 1:, :] - true_dose[:, :, :, :-1, :])
        )
        grad_x = self.l1_loss(
            torch.abs(pred_dose[:, :, :, :, 1:] - pred_dose[:, :, :, :, :-1]),
            torch.abs(true_dose[:, :, :, :, 1:] - true_dose[:, :, :, :, :-1])
        )
        return self.alpha * loss_l1 + self.beta * (grad_z + grad_y + grad_x)


# =====================================================================
# 8. Plan-Optimisation Epoch Timer (frozen dose engine)
# =====================================================================
def time_plan_optimization_epoch(
    model: 'VMATDosePredictorAttention',
    ct: torch.Tensor,
    target_dose: torch.Tensor,
    n_warmup: int = 3,
    n_epochs: int = 100,
    lr: float = 1e-3,
) -> dict:
    """Benchmark one inverse-planning epoch with the dose engine frozen.

    Plan parameters (MLC leaf positions, jaw positions, MU weights) are
    differentiable; gradients flow through the frozen dose engine back to
    them.  Apertures are computed via DifferentiableMLCAperture /
    DifferentiableJawAperture with average=True (RayStation convention).
    """
    device   = next(model.parameters()).device
    use_cuda = (device.type == 'cuda')
    B        = ct.shape[0]

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    mlc_to_aperture = DifferentiableMLCAperture().to(device)
    jaw_to_aperture = DifferentiableJawAperture().to(device)

    # Raw plan parameters: always 180 CPs (averaging inside aperture layers → 179 segments)
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

        # Apertures: average adjacent CP pairs → 179 segments
        mlc_aperture = mlc_to_aperture(mlc_positions, average=True)  # [B, 179, 1, 160, 160]
        jaw_aperture = jaw_to_aperture(jaw_positions, average=True)  # [B, 179, 1, 160, 160]
        bev_apertures = torch.cat([jaw_aperture, mlc_aperture], dim=2)  # [B, 179, 2, 160, 160]

        # MU per segment: positive via softplus
        mu = F.softplus(mu_logits)  # [B, 179]

        pred_dose, _ = model(ct, bev_apertures, mu)
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

    mean_t   = sum(times) / len(times)
    variance = sum((t - mean_t) ** 2 for t in times) / max(len(times) - 1, 1)
    std_t    = variance ** 0.5

    print(f"[DoseEngine] Plan-optimisation epoch timing  ({device})")
    print(f"  epochs    : {n_epochs}  (+{n_warmup} warmup)")
    print(f"  mean/std  : {mean_t:.3f} ± {std_t:.3f} s")
    print(f"  total     : {sum(times):.2f} s")
    print(f"  final loss: {final_loss:.6f}")

    return {'times_s': times, 'mean_s': mean_t, 'std_s': std_t, 'final_loss': final_loss}


# =====================================================================
# 9. Smoke-test / benchmark (run this file directly)
# =====================================================================
if __name__ == "__main__":
    print("DoseCalculation_Attention")
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 1

    print(f"Device : {device}")
    print("Building model...")
    model = VMATDosePredictorAttention(average=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}")

    print("Generating dummy CT and target dose...")
    dummy_ct          = torch.rand(B, 1, 192, 192, 192, device=device)
    dummy_target_dose = torch.rand(B, 1, 192, 192, 192, device=device)

    print("Starting plan-optimisation epoch benchmark...\n")
    results = time_plan_optimization_epoch(
        model        = model,
        ct           = dummy_ct,
        target_dose  = dummy_target_dose,
        n_warmup     = 2,
        n_epochs     = 100,
        lr           = 1e-3,
    )

    print("\nPer-epoch times (s):", [f"{t:.3f}" for t in results['times_s']])
    """

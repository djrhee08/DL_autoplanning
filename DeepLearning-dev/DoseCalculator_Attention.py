import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =====================================================================
# 1. Geometric 3D Projection Grid Builder (unchanged)
# =====================================================================
def build_hfs_perspective_grids(num_cps=180,
                                feat_vol_size=(96, 96, 96),
                                raw_ct_size=(192, 192, 192), ct_spacing=(3.0, 3.0, 3.0),
                                raw_bev_size=(160, 160), bev_spacing=(2.5, 2.5),
                                sad=1000.0, is_parallel=False):
    """
    Builds a perspective projection grid from BEV space to a 3D feature volume.
    Grid shape: [num_cps, D, H, W, 2]  (normalized coords for F.grid_sample)
    """
    D, H, W = feat_vol_size

    fov_z = raw_ct_size[0] * ct_spacing[0]    # 576 mm
    fov_y = raw_ct_size[1] * ct_spacing[1]    # 576 mm
    fov_x = raw_ct_size[2] * ct_spacing[2]    # 576 mm
    fov_v = raw_bev_size[0] * bev_spacing[0]  # 400 mm (SI)
    fov_u = raw_bev_size[1] * bev_spacing[1]  # 400 mm (LR)

    angles_deg = (181.0 + 2.0 * torch.arange(num_cps)) % 360.0
    angles_rad = angles_deg * (math.pi / 180.0)

    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
    )

    phys_x = (grid_x * (fov_x / 2.0)).unsqueeze(0)
    phys_y = (grid_y * (fov_y / 2.0)).unsqueeze(0)
    phys_z = (grid_z * (fov_z / 2.0)).unsqueeze(0)

    cos_t = torch.cos(angles_rad).view(-1, 1, 1, 1)
    sin_t = torch.sin(angles_rad).view(-1, 1, 1, 1)

    rot_x = phys_x * cos_t + phys_y * sin_t
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
# 2. 2D BEV Feature Encoder (unchanged from DoseCalculator.py)
# =====================================================================
class BEVEncoder2D(nn.Module):
    """Encodes each CP's 2-channel aperture (jaw + MLC) into a 32-channel feature map.
    Input:  [B, 180, 2, 160, 160]
    Output: [B, 180, 32, 40, 40]
    """
    def __init__(self, in_channels=2):
        super().__init__()
        self.block1 = nn.Sequential(           # 160 -> 80
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = nn.Sequential(           # 80 -> 40
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
# 3. Per-CP Projection Layer  (NEW: stacks instead of sums)
# =====================================================================
class PerCPProjectionLayer(nn.Module):
    """Projects each CP's BEV feature map into a 3D volume at bottleneck resolution
    (24³) using the same perspective grid_sample trick as DoseCalculator.py.

    Difference from DoseCalculator.py: results are *stacked*, not summed.

    Input:  bev_features  [B, N, C, 140, 140]
    Output: beam_stack    [B, N, C, 24, 24, 24]
    """
    def __init__(self, num_cps=180, vol_size=(24, 24, 24)):
        super().__init__()
        self.num_cps = num_cps
        self.vol_size = vol_size

        grid = build_hfs_perspective_grids(
            num_cps=num_cps,
            feat_vol_size=vol_size,
            raw_ct_size=(192, 192, 192), ct_spacing=(3.0, 3.0, 3.0),
            raw_bev_size=(160, 160), bev_spacing=(2.5, 2.5)
        )
        self.register_buffer('sampling_grid', grid)  # [N, D, H, W, 2]

    def forward(self, bev_features):
        B, N, C, H_bev, W_bev = bev_features.shape
        D, H_vol, W_vol = self.vol_size

        # Fold N into batch: [B, N, C, H, W] → [B*N, C, H, W]
        bev_flat = bev_features.reshape(B * N, C, H_bev, W_bev)

        # Build batched grid: [N, D, H, W, 2] → [N, D*H, W, 2] → [B*N, D*H, W, 2]
        # expand copies N across B (stride-0 view), reshape forces contiguous copy
        grid = self.sampling_grid.view(N, D * H_vol, W_vol, 2)            # [N, D*H, W, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)               # [B, N, D*H, W, 2]
        grid = grid.reshape(B * N, D * H_vol, W_vol, 2)                  # [B*N, D*H, W, 2]

        projected = F.grid_sample(                                         # [B*N, C, D*H, W]
            bev_flat, grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        return projected.view(B, N, C, D, H_vol, W_vol)                   # [B, N, C, 24, 24, 24]


# =====================================================================
# 4. Spatially-Resolved Beam Attention  (NEW)
# =====================================================================
class SpatialBeamAttention(nn.Module):
    """Cross-attention between CT bottleneck voxels and the per-CP projected beam stack.

    For each voxel in the 24³ bottleneck volume, compute a soft attention weight
    over all 180 CPs, then produce a weighted sum of their projected features.

    This replaces the naive summation in DoseCalculator.py:
      - Old: fused = Σ_i  projected_i                  (commutative, attribution lost)
      - New: fused = Σ_i  attn(voxel, CP_i) * proj_i  (each voxel attends selectively)

    The learned attention weight is interpretable: attn[b, voxel, i] indicates how
    much CP_i (at its gantry angle) contributes to that 3D voxel's dose.

    Args:
        ct_channels  : channels of CT bottleneck features (128)
        beam_channels: channels of per-CP projected features (32)
        num_cps      : number of control points (180)
    """
    def __init__(self, ct_channels=128, beam_channels=32, num_cps=180):
        super().__init__()
        self.scale = beam_channels ** -0.5

        # Project CT features into the same space as beam features for dot-product attention
        self.query_proj = nn.Conv3d(ct_channels, beam_channels, kernel_size=1)
        self.norm = nn.LayerNorm(beam_channels)

    def forward(self, ct_feat, beam_stack):
        """
        Args:
            ct_feat   : [B, C_ct,   D, H, W]        CT bottleneck (24³)
            beam_stack: [B, N, C_b, D, H, W]        per-CP projected volumes (24³)
        Returns:
            fused     : [B, C_b, D, H, W]           attention-weighted beam feature
            attn_w    : [B, D*H*W, N]               attention weights (interpretable)
        """
        B, C_ct, D, H, W = ct_feat.shape
        B, N, C_b, _, _, _ = beam_stack.shape
        V = D * H * W  # 13,824 voxels at 24³

        # Query from CT: [B, C_b, D, H, W] → [B, V, C_b]
        q = self.query_proj(ct_feat).view(B, C_b, V).permute(0, 2, 1)

        # Key & Value = beam_stack: [B, N, C_b, V]
        kv = beam_stack.view(B, N, C_b, V)

        # Attention logits: dot product query [B, V, C_b] with keys [B, N, C_b, V]
        # → [B, V, N]  (each voxel scores each CP)
        attn_logits = torch.einsum('bvc,bncv->bvn', q, kv) * self.scale
        attn_w = torch.softmax(attn_logits, dim=-1)  # [B, V, N]

        # Weighted aggregation: [B, V, N] × [B, N, C_b, V] → [B, V, C_b]
        fused = torch.einsum('bvn,bncv->bvc', attn_w, kv)

        # LayerNorm then reshape back to 3D
        fused = self.norm(fused).permute(0, 2, 1).view(B, C_b, D, H, W)

        return fused, attn_w  # attn_w is kept for visualization/analysis


# =====================================================================
# 5. Shared 3D Conv Block (unchanged)
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

    Key difference from VMATDosePredictor (DoseCalculator.py):
      - Beam features are projected to 24³ (bottleneck resolution) individually,
        kept separate as [B, 180, 32, 24³], then fused into the CT bottleneck
        via cross-attention — not summed blindly at 96³.
      - Attention weights [B, 13824, 180] tell you which CP contributes to
        each 3D voxel, enabling per-angle dose attribution.

    Data flow:
        BEV apertures [B,180,2,160,160]
            → BEVEncoder2D           → [B, 180, 32, 40, 40]
            → PerCPProjectionLayer   → [B, 180, 32,  24,  24, 24]   ← stacked, not summed
                                                         ↓
        CT [B,1,192³] → U-Net encoder → bottleneck [B,128,24³]
                                                         ↓
                              SpatialBeamAttention  →  fused [B, 32, 24³]
                                                         ↓
                         concat + fusion_conv       →  [B,128,24³]
                                                         ↓
                              U-Net decoder         →  dose [B,1,192³]
    """
    def __init__(self):
        super().__init__()

        # --- Beam pathway ---
        self.bev_encoder = BEVEncoder2D(in_channels=2)
        self.per_cp_proj = PerCPProjectionLayer(num_cps=180, vol_size=(24, 24, 24))
        self.beam_attn   = SpatialBeamAttention(ct_channels=128, beam_channels=32, num_cps=180)

        # Merge attended beam features into bottleneck channels
        # Input: 128 (CT bottleneck) + 32 (attended beam) = 160
        self.fusion_conv = Conv3DBlock(160, 128)

        # --- CT encoder (U-Net) ---
        self.enc0      = Conv3DBlock(1, 16)                       # [B,  16, 192³]
        self.down1     = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.enc1      = Conv3DBlock(32, 32)                      # [B,  32,  96³]
        self.down2     = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        self.enc2      = Conv3DBlock(64, 64)                      # [B,  64,  48³]
        self.down3     = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        self.bottleneck = Conv3DBlock(128, 128)                   # [B, 128,  24³]

        # --- Decoder ---
        self.up2  = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = Conv3DBlock(128, 64)   # 64 up + 64 skip2

        self.up1  = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = Conv3DBlock(64, 32)    # 32 up + 32 skip1

        self.up0  = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec0 = Conv3DBlock(32, 16)    # 16 up + 16 skip0

        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # dose must be non-negative
        )

    def forward(self, ct, bev_apertures):
        """
        Args:
            ct           : [B, 1, 192, 192, 192]   patient CT (electron density)
            bev_apertures: [B, 180, 2, 160, 160]   jaw & MLC aperture sequence
        Returns:
            dose         : [B, 1, 192, 192, 192]   predicted 3D dose
            attn_weights : [B, 13824, 180]          per-voxel per-CP attention (24³ = 13824)
        """
        # --- 1. Beam pathway ---
        bev_feat   = self.bev_encoder(bev_apertures)   # [B, 180, 32, 140, 140]
        beam_stack = self.per_cp_proj(bev_feat)        # [B, 180, 32,  24,  24, 24]

        # --- 2. CT encoder ---
        skip0      = self.enc0(ct)                     # [B,  16, 192, 192, 192]
        skip1      = self.enc1(self.down1(skip0))      # [B,  32,  96,  96,  96]
        skip2      = self.enc2(self.down2(skip1))      # [B,  64,  48,  48,  48]
        bottle     = self.bottleneck(self.down3(skip2))# [B, 128,  24,  24,  24]

        # --- 3. Attention fusion at bottleneck ---
        beam_fused, attn_weights = self.beam_attn(bottle, beam_stack)
        # beam_fused: [B, 32, 24³]  attn_weights: [B, 13824, 180]

        fused = self.fusion_conv(
            torch.cat([bottle, beam_fused], dim=1)     # [B, 160, 24³] → [B, 128, 24³]
        )

        # --- 4. Decoder ---
        up2   = self.up2(fused)                                  # [B,  64, 48³]
        dec2  = self.dec2(torch.cat([up2, skip2], dim=1))        # [B,  64, 48³]

        up1   = self.up1(dec2)                                   # [B,  32, 96³]
        dec1  = self.dec1(torch.cat([up1, skip1], dim=1))        # [B,  32, 96³]

        up0   = self.up0(dec1)                                   # [B,  16, 192³]
        dec0  = self.dec0(torch.cat([up0, skip0], dim=1))        # [B,  16, 192³]

        dose  = self.final_conv(dec0)                            # [B,   1, 192³]
        return dose, attn_weights


# =====================================================================
# 7. Physics-Informed Loss (unchanged from DoseCalculator.py)
# =====================================================================
class PhysicsInformedDoseLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
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
    """Benchmark one inverse-planning epoch: the dose engine is frozen and
    MLC leaf positions, jaw BEV apertures, and MU weights are the
    differentiable plan parameters being optimised via gradient descent.

    The frozen model acts as a differentiable dose engine — gradients flow
    through it back to the plan parameters, but its weights are not updated.

    Args:
        model       : Trained VMATDosePredictorAttention. Frozen in-place
                      for the duration of this call; call
                      ``model.train()`` and re-enable grad afterwards if needed.
        ct          : [B, 1, 192, 192, 192]  patient CT on the target device.
        target_dose : [B, 1, 192, 192, 192]  clinical goal dose.
        n_warmup    : Warmup epochs (GPU cache warm-up, not timed).
        n_epochs    : Timed epochs.
        lr          : Adam learning rate for plan parameters.

    Returns:
        dict with keys:
            ``times_s``    – list[float]  per-epoch wall-clock time in seconds
            ``mean_s``     – float        mean epoch time
            ``std_s``      – float        std of epoch times
            ``final_loss`` – float        PhysicsInformedDoseLoss after last epoch
    """
    device = next(model.parameters()).device
    use_cuda = (device.type == 'cuda')
    B = ct.shape[0]

    # --- Freeze the dose engine ------------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # --- Try to use DifferentiableMLCAperture for leaf → aperture image ---
    try:
        from MLC2Aperture import DifferentiableMLCAperture
        mlc_to_aperture = DifferentiableMLCAperture().to(device)
        _use_mlc_module = True
    except ImportError:
        mlc_to_aperture = None
        _use_mlc_module = False

    # --- Initialise learnable plan parameters ----------------------------
    # MLC leaf positions (mm): X1 (negative/left bank) and X2 (positive/right bank)
    mlc_init = torch.zeros(B, 180, 60, 2, device=device)
    mlc_init[..., 0] = -100.0   # X1 leaves: open 100 mm left of centreline
    mlc_init[..., 1] =  100.0   # X2 leaves: open 100 mm right of centreline
    mlc_positions = nn.Parameter(mlc_init)

    # Jaw BEV aperture: one channel per CP, initialised nearly fully open
    jaw_bev = nn.Parameter(torch.ones(B, 180, 1, 160, 160, device=device) * 2.0)
    # (sigmoid(2.0) ≈ 0.88 — nearly open; will be squashed to [0,1] during forward)

    # MU weights per CP: unconstrained logits, mapped to positive values via softplus
    mu_logits = nn.Parameter(torch.zeros(B, 180, device=device))

    optimizer = torch.optim.Adam([mlc_positions, jaw_bev, mu_logits], lr=lr)
    loss_fn = PhysicsInformedDoseLoss(alpha=1.0, beta=0.5)

    def _run_epoch():
        optimizer.zero_grad(set_to_none=True)

        # MU per CP: [B, 180] → broadcast over aperture spatial dims
        mu = F.softplus(mu_logits).view(B, 180, 1, 1, 1)  # [B, 180, 1, 1, 1]

        # MLC → 160×160 aperture image
        if _use_mlc_module:
            mlc_aperture = mlc_to_aperture(mlc_positions)         # [B, 180, 1, 160, 160]
        else:
            # Fallback: sigmoid ramp between X1 and X2 columns averaged over leaves
            # (gradient-compatible proxy when MLC2Aperture is unavailable)
            x1 = mlc_positions[..., 0:1]  # [B, 180, 60, 1]
            x2 = mlc_positions[..., 1:2]  # [B, 180, 60, 1]
            opening = torch.sigmoid(x2 - x1)  # [B, 180, 60, 1] ∈ (0,1)
            mlc_aperture = opening.mean(dim=2, keepdim=True)       # [B, 180, 1, 1]
            mlc_aperture = mlc_aperture.unsqueeze(-1).expand(B, 180, 1, 160, 160)

        jaw_aperture = torch.sigmoid(jaw_bev)                      # [B, 180, 1, 160, 160]

        # Stack channels and scale by MU: [B, 180, 2, 160, 160]
        bev_apertures = torch.cat([jaw_aperture, mlc_aperture], dim=2) * mu

        pred_dose, _ = model(ct, bev_apertures)
        loss = loss_fn(pred_dose, target_dose)
        loss.backward()
        optimizer.step()
        return loss.item()

    # --- Warmup (not timed) ----------------------------------------------
    for _ in range(n_warmup):
        _run_epoch()
        if use_cuda:
            torch.cuda.synchronize(device)

    # --- Timed epochs ----------------------------------------------------
    times = []
    final_loss = None
    for _ in range(n_epochs):
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        final_loss = _run_epoch()
        if use_cuda:
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    mean_t = sum(times) / len(times)
    variance = sum((t - mean_t) ** 2 for t in times) / max(len(times) - 1, 1)
    std_t = variance ** 0.5

    print(f"[DoseEngine] Plan-optimisation epoch timing  ({device})")
    print(f"  epochs    : {n_epochs}  (+{n_warmup} warmup)")
    print(f"  mean/std  : {mean_t:.3f} ± {std_t:.3f} s")
    print(f"  total     : {sum(times):.2f} s")
    print(f"  final loss: {final_loss:.6f}")
    if not _use_mlc_module:
        print("  [WARN] MLC2Aperture not found — using sigmoid proxy for MLC aperture")

    return {
        'times_s':     times,
        'mean_s':      mean_t,
        'std_s':       std_t,
        'final_loss':  final_loss,
    }


# =====================================================================
# 9. Plan-optimisation timing benchmark (run this file directly)
# =====================================================================
if __name__ == "__main__":
    print("DoseCalculation_Attention")
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 1

    print(f"Device : {device}")
    print("Building model...")
    model = VMATDosePredictorAttention().to(device)
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
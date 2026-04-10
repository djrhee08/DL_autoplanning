"""
High-Accuracy VMAT Dose Predictor (v2-48 with gradient checkpointing)
======================================================================
Same architecture as v2-48 (A + B improvements at 48³) but with gradient
checkpointing on the two memory-heaviest components:

  - SpatioAngularConvBlock: 4D state permute/reshape chain saves ~12GB
  - PerAngleTERMAModule: per-chunk activations saves ~10GB

Training peak memory drops from ~37GB to ~18-22GB on A100 40GB.
Speed cost: ~20-30% slower per iteration due to recomputation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from MLC2Aperture import (vmat_gantry_angles,
                          DifferentiableMLCAperture,
                          DifferentiableJawAperture)


# =====================================================================
# Helper: dynamic GroupNorm
# =====================================================================
def make_gn(channels, max_groups=8):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return nn.GroupNorm(g, channels)
    return nn.GroupNorm(1, channels)


def gn_groups(channels, max_groups=8):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# =====================================================================
# 1. Perspective projection grid
# =====================================================================
def build_hfs_perspective_grids(gantry_angles_deg,
                                feat_vol_size=(48, 48, 48),
                                raw_ct_size=(192, 192, 192),
                                ct_spacing=(3.0, 3.0, 3.0),
                                raw_bev_size=(140, 140),
                                bev_spacing=(4.0, 4.0),
                                sad=1000.0):
    if not isinstance(gantry_angles_deg, torch.Tensor):
        gantry_angles_deg = torch.tensor(gantry_angles_deg, dtype=torch.float32)

    angles_rad = gantry_angles_deg * (math.pi / 180.0)
    num_cps = gantry_angles_deg.shape[0]

    D, H, W = feat_vol_size

    fov_z = raw_ct_size[0] * ct_spacing[0]
    fov_y = raw_ct_size[1] * ct_spacing[1]
    fov_x = raw_ct_size[2] * ct_spacing[2]
    fov_v = raw_bev_size[0] * bev_spacing[0]
    fov_u = raw_bev_size[1] * bev_spacing[1]

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

    rot_x = phys_x * cos_t + phys_y * sin_t
    rot_y = -phys_x * sin_t + phys_y * cos_t
    rot_z = phys_z.expand(num_cps, -1, -1, -1)

    magnification = sad / (sad + rot_y)
    u_phys = rot_x * magnification
    v_phys = rot_z * magnification

    u_norm = u_phys / (fov_u / 2.0)
    v_norm = v_phys / (fov_v / 2.0)

    return torch.stack((u_norm, v_norm), dim=-1).float()


# =====================================================================
# 2. 4D Spatio-Angular Convolution Block
# =====================================================================
class SpatioAngularConvBlock(nn.Module):
    """Separable 4D conv: spatial → angular (circular) → spatial."""
    def __init__(self, channels, angular_kernel=5):
        super().__init__()
        assert angular_kernel % 2 == 1
        self.num_groups = gn_groups(channels)

        self.spatial1 = nn.Conv3d(channels, channels, kernel_size=3,
                                  padding=1, bias=False)
        self.norm1 = make_gn(channels)

        self.angular = nn.Conv1d(channels, channels,
                                 kernel_size=angular_kernel,
                                 padding=angular_kernel // 2,
                                 padding_mode='circular',
                                 bias=False)

        self.spatial2 = nn.Conv3d(channels, channels, kernel_size=3,
                                  padding=1, bias=False)
        self.norm3 = make_gn(channels)

        self.act = nn.GELU()

    def forward(self, x):
        """x: [B, C, A, D, H, W]"""
        B, C, A, D, H, W = x.shape

        x_sp = x.permute(0, 2, 1, 3, 4, 5).reshape(B * A, C, D, H, W)
        x_sp = self.act(self.norm1(self.spatial1(x_sp)))
        x = x_sp.reshape(B, A, C, D, H, W).permute(0, 2, 1, 3, 4, 5)

        x_ang = x.permute(0, 3, 4, 5, 1, 2).reshape(B * D * H * W, C, A)
        x_ang = self.angular(x_ang)
        x_ang = self.act(F.group_norm(x_ang, num_groups=self.num_groups))
        x = x_ang.view(B, D, H, W, C, A).permute(0, 4, 5, 1, 2, 3)

        x_sp = x.permute(0, 2, 1, 3, 4, 5).reshape(B * A, C, D, H, W)
        x_sp = self.act(self.norm3(self.spatial2(x_sp)))
        x = x_sp.reshape(B, A, C, D, H, W).permute(0, 2, 1, 3, 4, 5)

        return x


# =====================================================================
# 3. CT Encoder
# =====================================================================
class CTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            make_gn(16), nn.GELU(),
            nn.Conv3d(16, 16, 3, padding=1),
            make_gn(16), nn.GELU(),
        )
        self.enc1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, stride=2),
            make_gn(32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
            make_gn(32), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, stride=2),
            make_gn(64), nn.GELU(),
            nn.Conv3d(64, 64, 3, padding=1),
            make_gn(64), nn.GELU(),
        )

    def forward(self, ct):
        f0 = self.enc0(ct)
        f1 = self.enc1(f0)
        f2 = self.enc2(f1)
        return f0, f1, f2


# =====================================================================
# 4. Per-Angle TERMA Module (with checkpointing inside the loop)
# =====================================================================
class PerAngleTERMAModule(nn.Module):
    """Per-angle TERMA with CT broadcast, streaming summation.

    Uses gradient checkpointing on each chunk's conv operation:
    the activations inside terma_conv are recomputed during backward,
    saving significant memory per chunk.
    """
    def __init__(self, fluence_channels=16, ct_channels=64, terma_channels=16):
        super().__init__()
        self.terma_conv = nn.Sequential(
            nn.Conv3d(fluence_channels + ct_channels, 32, 3, padding=1),
            make_gn(32), nn.GELU(),
            nn.Conv3d(32, terma_channels, 3, padding=1),
            make_gn(terma_channels), nn.GELU(),
        )

    def _process_chunk(self, fl_chunk, ct_exp):
        """Helper for gradient checkpointing: conv on concatenated input."""
        combined = torch.cat([fl_chunk, ct_exp], dim=1)
        return self.terma_conv(combined)

    def forward(self, fluence_4d, ct_feat, angle_chunk_size=20,
                use_checkpoint=True):
        """
        fluence_4d: [B, C_f, A, D, H, W]
        ct_feat:    [B, C_ct, D, H, W]
        """
        B, C_f, A, D, H, W = fluence_4d.shape
        C_ct = ct_feat.shape[1]

        terma_sum = None

        for a_start in range(0, A, angle_chunk_size):
            a_end = min(a_start + angle_chunk_size, A)
            cs = a_end - a_start

            fl_chunk = fluence_4d[:, :, a_start:a_end]
            fl_chunk = fl_chunk.permute(0, 2, 1, 3, 4, 5).reshape(B * cs, C_f, D, H, W)

            ct_exp = ct_feat.unsqueeze(1).expand(B, cs, C_ct, D, H, W)
            ct_exp = ct_exp.reshape(B * cs, C_ct, D, H, W)

            if use_checkpoint and self.training:
                terma_chunk = checkpoint(
                    self._process_chunk, fl_chunk, ct_exp,
                    use_reentrant=False
                )
            else:
                terma_chunk = self._process_chunk(fl_chunk, ct_exp)

            C_t = terma_chunk.shape[1]
            terma_chunk = terma_chunk.view(B, cs, C_t, D, H, W).sum(dim=1)

            if terma_sum is None:
                terma_sum = terma_chunk
            else:
                terma_sum = terma_sum + terma_chunk

        return terma_sum


# =====================================================================
# 5. Multi-Scale Learned Dose Kernel at 48³
# =====================================================================
class LearnedDoseKernel(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.channels = channels

        self.small_d = nn.Conv3d(channels, channels, (5, 1, 1), padding=(2, 0, 0))
        self.small_h = nn.Conv3d(channels, channels, (1, 5, 1), padding=(0, 2, 0))
        self.small_w = nn.Conv3d(channels, channels, (1, 1, 5), padding=(0, 0, 2))

        self.med_d = nn.Conv3d(channels, channels, (11, 1, 1), padding=(5, 0, 0))
        self.med_h = nn.Conv3d(channels, channels, (1, 11, 1), padding=(0, 5, 0))
        self.med_w = nn.Conv3d(channels, channels, (1, 1, 11), padding=(0, 0, 5))

        self.large_d = nn.Conv3d(channels, channels, (21, 1, 1), padding=(10, 0, 0))
        self.large_h = nn.Conv3d(channels, channels, (1, 21, 1), padding=(0, 10, 0))
        self.large_w = nn.Conv3d(channels, channels, (1, 1, 21), padding=(0, 0, 10))

        self.combine = nn.Sequential(
            nn.Conv3d(channels * 3, channels, 1),
            make_gn(channels), nn.GELU(),
        )

    def _separable(self, x, conv_d, conv_h, conv_w):
        return conv_w(conv_h(conv_d(x)))

    def forward(self, terma):
        small = self._separable(terma, self.small_d, self.small_h, self.small_w)
        med = self._separable(terma, self.med_d, self.med_h, self.med_w)
        large = self._separable(terma, self.large_d, self.large_h, self.large_w)
        return self.combine(torch.cat([small, med, large], dim=1))


# =====================================================================
# 6. Refinement Decoder
# =====================================================================
class RefinementDecoder(nn.Module):
    def __init__(self, dose_channels=16):
        super().__init__()
        self.up1 = nn.ConvTranspose3d(dose_channels, dose_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(dose_channels + 32, 32, 3, padding=1),
            make_gn(32), nn.GELU(),
            nn.Conv3d(32, 32, 3, padding=1),
            make_gn(32), nn.GELU(),
        )
        self.up0 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec0 = nn.Sequential(
            nn.Conv3d(16 + 16, 16, 3, padding=1),
            make_gn(16), nn.GELU(),
            nn.Conv3d(16, 16, 3, padding=1),
            make_gn(16), nn.GELU(),
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
# 7. Main Model with gradient checkpointing on SpatioAngularConvBlock
# =====================================================================
class HighAccuracyVMATPredictorV2(nn.Module):
    def __init__(self,
                 base_channels=16,
                 mlc_pixel_size=1.0,
                 mlc_grid_size=560,
                 mlc_tau=0.1,
                 jaw_tau=0.1,
                 vol_size=(48, 48, 48),
                 projection_chunk_size=20,
                 embed_chunk_size=40,
                 terma_angle_chunk=20,
                 use_checkpoint=True):
        super().__init__()
        self.vol_size = vol_size
        self.base_channels = base_channels
        self.projection_chunk_size = projection_chunk_size
        self.embed_chunk_size = embed_chunk_size
        self.terma_angle_chunk = terma_angle_chunk
        self.use_checkpoint = use_checkpoint

        self.mlc_layer = DifferentiableMLCAperture(
            pixel_size=mlc_pixel_size, grid_size=mlc_grid_size, tau=mlc_tau
        )
        self.jaw_layer = DifferentiableJawAperture(
            pixel_size=mlc_pixel_size, grid_size=mlc_grid_size, tau=jaw_tau
        )

        self.aperture_downsample = nn.AvgPool2d(kernel_size=4, stride=4)

        bev_down_size = mlc_grid_size // 4
        bev_down_spacing = mlc_pixel_size * 4

        gantry_angles = vmat_gantry_angles(average=True)
        proj_grid = build_hfs_perspective_grids(
            gantry_angles_deg=gantry_angles,
            feat_vol_size=vol_size,
            raw_ct_size=(192, 192, 192),
            ct_spacing=(3.0, 3.0, 3.0),
            raw_bev_size=(bev_down_size, bev_down_size),
            bev_spacing=(bev_down_spacing, bev_down_spacing),
            sad=1000.0,
        )
        self.register_buffer('proj_grid', proj_grid)
        self.num_cps = proj_grid.shape[0]

        half_c = max(base_channels // 2, 1)
        self.fluence_embed = nn.Sequential(
            nn.Conv3d(1, half_c, 3, padding=1),
            make_gn(half_c), nn.GELU(),
            nn.Conv3d(half_c, base_channels, 3, padding=1),
            make_gn(base_channels), nn.GELU(),
        )

        self.sa_block1 = SpatioAngularConvBlock(base_channels)
        self.sa_block2 = SpatioAngularConvBlock(base_channels)

        self.ct_encoder = CTEncoder()

        self.terma = PerAngleTERMAModule(
            fluence_channels=base_channels,
            ct_channels=64,
            terma_channels=base_channels
        )

        self.kernel = LearnedDoseKernel(channels=base_channels)
        self.refine = RefinementDecoder(dose_channels=base_channels)

    def _build_aperture(self, mlc_pos, jaw_pos):
        mlc_ap = self.mlc_layer(mlc_pos, average=True)
        jaw_ap = self.jaw_layer(jaw_pos, average=True)
        return mlc_ap * jaw_ap

    def _downsample_aperture(self, aperture):
        B, N, _, H, W = aperture.shape
        aperture = aperture.reshape(B * N, 1, H, W)
        downsampled = self.aperture_downsample(aperture)
        _, _, H_new, W_new = downsampled.shape
        return downsampled.view(B, N, 1, H_new, W_new)

    def _project_to_3d(self, aperture, mu):
        B, N, _, H_bev, W_bev = aperture.shape
        D, H, W = self.vol_size

        out_chunks = []
        for i in range(0, N, self.projection_chunk_size):
            cs = min(self.projection_chunk_size, N - i)

            ap_chunk = aperture[:, i:i+cs].reshape(B * cs, 1, H_bev, W_bev)

            grid_chunk = self.proj_grid[i:i+cs]
            grid_chunk = grid_chunk.view(cs, D * H, W, 2)
            grid_chunk = grid_chunk.unsqueeze(0).expand(B, -1, -1, -1, -1)
            grid_chunk = grid_chunk.reshape(B * cs, D * H, W, 2)

            fluence = F.grid_sample(
                ap_chunk, grid_chunk,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )
            fluence = fluence.view(B, cs, 1, D, H, W)

            mu_chunk = mu[:, i:i+cs].view(B, cs, 1, 1, 1, 1)
            fluence = fluence * mu_chunk

            out_chunks.append(fluence)

        return torch.cat(out_chunks, dim=1)

    def _embed_fluence(self, fluence_3d):
        B, N, _, D, H, W = fluence_3d.shape
        C = self.base_channels

        embedded_chunks = []
        for i in range(0, N, self.embed_chunk_size):
            cs = min(self.embed_chunk_size, N - i)
            chunk = fluence_3d[:, i:i+cs].reshape(B * cs, 1, D, H, W)
            embedded = self.fluence_embed(chunk)
            embedded = embedded.view(B, cs, C, D, H, W)
            embedded_chunks.append(embedded)

        all_embedded = torch.cat(embedded_chunks, dim=1)
        return all_embedded.permute(0, 2, 1, 3, 4, 5).contiguous()

    def forward(self, ct, mlc_pos, jaw_pos, mu):
        aperture = self._build_aperture(mlc_pos, jaw_pos)
        aperture = self._downsample_aperture(aperture)
        fluence_3d = self._project_to_3d(aperture, mu)
        fluence_4d = self._embed_fluence(fluence_3d)

        ct_f0, ct_f1, ct_f2 = self.ct_encoder(ct)

        # === Checkpointed spatio-angular conv blocks ===
        if self.use_checkpoint and self.training:
            x = checkpoint(self.sa_block1, fluence_4d, use_reentrant=False)
            x2 = checkpoint(self.sa_block2, x, use_reentrant=False)
            x = x2 + x
        else:
            x = self.sa_block1(fluence_4d)
            x = self.sa_block2(x) + x

        # === Per-angle TERMA (has its own checkpointing) ===
        terma = self.terma(x, ct_f2,
                           angle_chunk_size=self.terma_angle_chunk,
                           use_checkpoint=self.use_checkpoint)

        dose_features = self.kernel(terma)
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
# 9. Smoke test
# =====================================================================
if __name__ == "__main__":
    from torch.amp import autocast

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    B = 1
    print("Initializing v2-48 model with gradient checkpointing...")
    model = HighAccuracyVMATPredictorV2(base_channels=16).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffer = sum(b.numel() for b in model.buffers())
    print(f"Trainable parameters: {total_params:,}")
    print(f"Buffer elements:      {total_buffer:,} "
          f"(~{total_buffer * 4 / 1e9:.2f} GB at fp32)")

    print("\nGenerating dummy inputs...")
    ct = torch.rand(B, 1, 192, 192, 192, device=device)
    mlc_pos = torch.zeros(B, 180, 60, 2, device=device)
    mlc_pos[..., 0] = -50.0
    mlc_pos[..., 1] = 50.0
    jaw_pos = torch.zeros(B, 180, 2, 2, device=device)
    jaw_pos[..., 0, 0] = -100.0
    jaw_pos[..., 0, 1] = 100.0
    jaw_pos[..., 1, 0] = -100.0
    jaw_pos[..., 1, 1] = 100.0
    mu = torch.ones(B, 179, device=device) * 5.0

    print("\n[1] Inference forward pass with bfloat16 autocast...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            dose = model(ct, mlc_pos, jaw_pos, mu)

    print(f"  Dose output shape: {dose.shape}")
    print(f"  Dose output dtype: {dose.dtype}")
    print(f"  Dose range: [{dose.float().min():.4f}, {dose.float().max():.4f}]")
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM (inference): {peak:.2f} GB")

    print("\n[2] Training step (forward + backward) with bfloat16 autocast...")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = PhysicsInformedDoseLoss(alpha=1.0, beta=0.5).to(device)

    dose_target = torch.rand(B, 1, 192, 192, 192, device=device)

    optimizer.zero_grad()
    with autocast('cuda', dtype=torch.bfloat16):
        dose_pred = model(ct, mlc_pos, jaw_pos, mu)
        loss = criterion(dose_pred, dose_target)

    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM (training): {peak:.2f} GB")

    print("\n[3] Timing 5 training iterations...")
    if device.type == 'cuda':
        torch.cuda.synchronize()

    import time
    times = []
    for i in range(5):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()

        optimizer.zero_grad()
        with autocast('cuda', dtype=torch.bfloat16):
            dose_pred = model(ct, mlc_pos, jaw_pos, mu)
            loss = criterion(dose_pred, dose_target)
        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"  Iter {i+1}: {times[-1]*1000:.0f} ms, loss={loss.item():.4f}")

    avg_time = sum(times[1:]) / len(times[1:])
    print(f"  Average (excl. first): {avg_time*1000:.0f} ms/iter")

    print("\nSmoke test complete!")
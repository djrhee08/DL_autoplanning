import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =====================================================================
# 1. Geometric 3D Projection Grid Builder (with spacing & FOV)
# =====================================================================
def build_hfs_perspective_grids(num_cps=180,
                                feat_vol_size=(96, 96, 96),
                                raw_ct_size=(192, 192, 192), ct_spacing=(3.0, 3.0, 3.0),
                                raw_bev_size=(560, 560), bev_spacing=(1.0, 1.0),
                                sad=1000.0, is_parallel=False):
    """
    Builds a projection grid based on CT (3mm) and BEV (1mm) spacing, incorporating FOV.
    """
    D, H, W = feat_vol_size
    
    # 1. Compute physical field of view (FOV)
    fov_z = raw_ct_size[0] * ct_spacing[0]   # 192 * 3 = 576 mm
    fov_y = raw_ct_size[1] * ct_spacing[1]   # 192 * 3 = 576 mm
    fov_x = raw_ct_size[2] * ct_spacing[2]   # 192 * 3 = 576 mm

    fov_v = raw_bev_size[0] * bev_spacing[0] # 560 * 1 = 560 mm (Superior-Inferior)
    fov_u = raw_bev_size[1] * bev_spacing[1] # 560 * 1 = 560 mm (Left-Right, MLC travel direction)

    # 1. Start at 181 degrees, increment by 2 degrees for num_cps steps
    angles_deg = 181.0 + 2.0 * torch.arange(num_cps)

    # 2. Wrap angles exceeding 360 degrees back to 0-359 range (e.g. 361 -> 1)
    angles_deg = angles_deg % 360.0
    angles_rad = angles_deg * (math.pi / 180.0)

    # 3. Generate normalized coordinates [-1, 1] for the 3D feature volume
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, D), torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij'
    )
    
    # 4. Convert normalized coordinates to physical coordinates (mm), isocenter = (0, 0, 0)
    phys_x = (grid_x * (fov_x / 2.0)).unsqueeze(0)
    phys_y = (grid_y * (fov_y / 2.0)).unsqueeze(0)
    phys_z = (grid_z * (fov_z / 2.0)).unsqueeze(0)

    cos_t = torch.cos(angles_rad).view(-1, 1, 1, 1)
    sin_t = torch.sin(angles_rad).view(-1, 1, 1, 1)

    # 5. Gantry rotation (about Z-axis)
    rot_x = phys_x * cos_t + phys_y * sin_t
    rot_y = -phys_x * sin_t + phys_y * cos_t
    rot_z = phys_z.expand(num_cps, -1, -1, -1)

    # 6. BEV plane projection (source at Y = -SAD, HFS & AP convention)
    if is_parallel:
        u_phys, v_phys = rot_x, rot_z
    else:
        # Voxels closer to the source (-SAD) are magnified more
        magnification = sad / (sad + rot_y)
        u_phys = rot_x * magnification
        v_phys = rot_z * magnification

    # 7. Convert projected physical coordinates to normalized BEV feature map coordinates [-1, 1]
    u_norm = u_phys / (fov_u / 2.0)
    v_norm = v_phys / (fov_v / 2.0)

    # Grid shape: [180, D, H, W, 2]
    grid = torch.stack((u_norm, v_norm), dim=-1)
    return grid.float()


# =====================================================================
# 2. 2D BEV Feature Encoder (MLC/Jaw -> Feature)
# =====================================================================
class BEVEncoder2D(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        # 560x560 -> 280x280
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, inplace=True)
        )
        # 280x280 -> 140x140
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.block1(x)
        x = self.block2(x)
        _, C_out, H_out, W_out = x.shape
        return x.view(B, N, C_out, H_out, W_out)


# =====================================================================
# 3. Differentiable 3D Projection Layer (dimension error fix)
# =====================================================================
class DifferentiableProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_cps = 180
        self.feat_vol_size = (96, 96, 96)

        # Build grid once (incorporating 3mm CT and 1mm BEV spacing)
        grid = build_hfs_perspective_grids(
            num_cps=self.num_cps, 
            feat_vol_size=self.feat_vol_size,
            raw_ct_size=(192, 192, 192), ct_spacing=(3.0, 3.0, 3.0),
            raw_bev_size=(560, 560), bev_spacing=(1.0, 1.0)
        )
        self.register_buffer('sampling_grid', grid)

    def forward(self, bev_features):
        B, N, C, H_bev, W_bev = bev_features.shape
        D, H_vol, W_vol = self.feat_vol_size
        device = bev_features.device
        
        volume = torch.zeros((B, C, D, H_vol, W_vol), device=device)
        
        for i in range(self.num_cps):
            # 1. 4D Input: [B, C, H_bev, W_bev]
            current_bev = bev_features[:, i, ...]  
            
            # 2. 5D Grid: [B, D, H_vol, W_vol, 2]
            current_grid = self.sampling_grid[i].unsqueeze(0).expand(B, -1, -1, -1, -1) 
            
            # Reshape 5D grid to 4D grid temporarily (merge Depth and Height)
            # Shape after reshape: [B, D * H_vol, W_vol, 2]
            reshaped_grid = current_grid.view(B, D * H_vol, W_vol, 2)

            # 3. Run grid_sample (4D input <-> 4D grid)
            # Output shape: [B, C, D * H_vol, W_vol]
            projected = F.grid_sample(
                current_bev, reshaped_grid, mode='bilinear', padding_mode='zeros', align_corners=True
            )

            # 4. Restore result back to 3D volume shape
            # Shape after reshape: [B, C, D, H_vol, W_vol]
            projected = projected.view(B, C, D, H_vol, W_vol)
            
            # 5. Accumulation
            volume += projected
            
        return volume


# =====================================================================
# 4. 3D CT Encoder & Dose Decoder (Naming Convention Fixed)
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


class VMATDosePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        # ---------------------------------------------------------
        # 1. Beam Pathway (Encoder & Differentiable Projection)
        # ---------------------------------------------------------
        self.bev_encoder = BEVEncoder2D(in_channels=2)
        self.projector = DifferentiableProjectionLayer()
        
        # ---------------------------------------------------------
        # 2. CT Pathway (3D U-Net Encoder)
        # ---------------------------------------------------------
        # Level 0: 192^3 Resolution
        self.enc0 = Conv3DBlock(1, 16)                     
        self.down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        
        # Level 1: 96^3 Resolution
        self.enc1 = Conv3DBlock(32, 32)                    
        self.down2 = nn.Conv3d(32, 64, kernel_size=2, stride=2)
        
        # Level 2: 48^3 Resolution
        self.enc2 = Conv3DBlock(64, 64)                    
        self.down3 = nn.Conv3d(64, 128, kernel_size=2, stride=2)
        
        # Bottleneck: 24^3 Resolution
        self.bottleneck = Conv3DBlock(128, 128)            
        
        # ---------------------------------------------------------
        # 3. Fusion Pathway (3D U-Net Decoder)
        # * Naming convention: up(N) and dec(N) both refer to Level(N) resolution
        # ---------------------------------------------------------

        # Level 2 Decoder (restore 48^3 resolution)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        # Input: 64 (upsampled) + 64 (skip2) = 128
        self.dec2 = Conv3DBlock(128, 64)

        # Level 1 Decoder (restore 96^3 resolution + beam fusion)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        # Input: 32 (upsampled) + 32 (skip1) + 32 (beam feature) = 96
        self.dec1 = Conv3DBlock(96, 32)

        # Level 0 Decoder (restore final 192^3 resolution)
        self.up0 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        # Input: 16 (upsampled) + 16 (skip0) = 32
        self.dec0 = Conv3DBlock(32, 16)

        # ---------------------------------------------------------
        # 4. Final Output Head
        # ---------------------------------------------------------
        self.final_conv = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1),
            nn.ReLU(inplace=True)  # radiation dose must be non-negative
        )

    def forward(self, ct, bev_apertures):
        """
        Args:
            ct: [B, 1, 192, 192, 192] (Patient Anatomy)
            bev_apertures: [B, 180, 2, 560, 560] (Jaw & MLC sequences)
        Returns:
            out_dose: [B, 1, 192, 192, 192] (Predicted 3D Dose Distribution)
        """
        # --- 1. Beam Pathway ---
        bev_feat = self.bev_encoder(bev_apertures)         # [B, 180, 32, 140, 140]
        beam_vol = self.projector(bev_feat)                # [B, 32, 96, 96, 96]
        
        # --- 2. CT Pathway (Encoder) ---
        skip0 = self.enc0(ct)                              # Level 0: [B, 16, 192, 192, 192]
        skip1 = self.enc1(self.down1(skip0))               # Level 1: [B, 32, 96, 96, 96]
        skip2 = self.enc2(self.down2(skip1))               # Level 2: [B, 64, 48, 48, 48]
        bottle = self.bottleneck(self.down3(skip2))        # Bottleneck: [B, 128, 24, 24, 24]
        
        # --- 3. Fusion Pathway (Decoder) ---
        
        # Level 2 (48^3)
        up_feat2 = self.up2(bottle)                                # [B, 64, 48, 48, 48]
        concat2 = torch.cat([up_feat2, skip2], dim=1)              # [B, 128, 48, 48, 48]
        dec_feat2 = self.dec2(concat2)                             # [B, 64, 48, 48, 48]
        
        # Level 1 (96^3) - 3-way fusion point
        up_feat1 = self.up1(dec_feat2)                             # [B, 32, 96, 96, 96]
        concat1 = torch.cat([up_feat1, skip1, beam_vol], dim=1)    # [B, 96, 96, 96, 96]
        dec_feat1 = self.dec1(concat1)                             # [B, 32, 96, 96, 96]
        
        # Level 0 (192^3)
        up_feat0 = self.up0(dec_feat1)                             # [B, 16, 192, 192, 192]
        concat0 = torch.cat([up_feat0, skip0], dim=1)              # [B, 32, 192, 192, 192]
        dec_feat0 = self.dec0(concat0)                             # [B, 16, 192, 192, 192]
        
        # --- 4. Final Output ---
        out_dose = self.final_conv(dec_feat0)                      # [B, 1, 192, 192, 192]
        
        return out_dose


class PhysicsInformedDoseLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Args:
            alpha: weight for the base L1 loss
            beta: weight for the 3D gradient penalty (controls penumbra sharpness)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_dose, true_dose):
        """
        pred_dose, true_dose shape: [B, 1, D, H, W]
        """
        # 1. Base Voxel-wise L1 Loss
        loss_l1 = self.l1_loss(pred_dose, true_dose)
        
        # 2. 3D Spatial Gradient Computation (Finite Difference Method)
        # Compute the difference between adjacent voxels along each axis (Z, Y, X)
        
        # Z-axis (Depth) gradient
        grad_z_pred = torch.abs(pred_dose[:, :, 1:, :, :] - pred_dose[:, :, :-1, :, :])
        grad_z_true = torch.abs(true_dose[:, :, 1:, :, :] - true_dose[:, :, :-1, :, :])
        loss_grad_z = self.l1_loss(grad_z_pred, grad_z_true)
        
        # Y-axis (Height/AP) gradient
        grad_y_pred = torch.abs(pred_dose[:, :, :, 1:, :] - pred_dose[:, :, :, :-1, :])
        grad_y_true = torch.abs(true_dose[:, :, :, 1:, :] - true_dose[:, :, :, :-1, :])
        loss_grad_y = self.l1_loss(grad_y_pred, grad_y_true)
        
        # X-axis (Width/LR) gradient
        grad_x_pred = torch.abs(pred_dose[:, :, :, :, 1:] - pred_dose[:, :, :, :, :-1])
        grad_x_true = torch.abs(true_dose[:, :, :, :, 1:] - true_dose[:, :, :, :, :-1])
        loss_grad_x = self.l1_loss(grad_x_pred, grad_x_true)
        
        # Total Gradient Penalty
        loss_grad = loss_grad_z + loss_grad_y + loss_grad_x
        
        # 3. Final Weighted Loss
        total_loss = (self.alpha * loss_l1) + (self.beta * loss_grad)
        
        return total_loss


# =====================================================================
# 5. Run & Test (Dummy Forward Pass)
# =====================================================================
if __name__ == "__main__":
    print("DoseCalculator")
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')""
    print(f"Using device: {device}")
    
    B = 1
    print("Initializing Model...")
    model = VMATDosePredictor().to(device)
    
    print("Generating Dummy Data...")
    dummy_ct = torch.rand((B, 1, 192, 192, 192), device=device)
    dummy_bev = torch.rand((B, 180, 2, 560, 560), device=device)
    
    print("Running Forward Pass...")
    with torch.no_grad(): 
        predicted_dose = model(dummy_ct, dummy_bev)
        
    print(f"Success! Output Dose Shape: {predicted_dose.shape}")
    """
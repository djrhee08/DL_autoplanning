import torch
import torch.nn as nn
import math

# ==========================================
# 1. 물리 좌표계 생성 함수 (순수 텐서만 반환하도록 수정)
# ==========================================
def create_base_ct_coords(depth, height, width, voxel_size_mm):
    """Batch 차원이 1인 기본 CT 물리 좌표계 텐서 생성"""
    cz, cy, cx = (depth - 1) / 2.0, (height - 1) / 2.0, (width - 1) / 2.0
    
    z = (torch.arange(depth) - cz) * voxel_size_mm[0]
    y = (torch.arange(height) - cy) * voxel_size_mm[1]
    x = (torch.arange(width) - cx) * voxel_size_mm[2]
    
    grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y, grid_z], dim=0) # [3, D, H, W]
    
    max_dist = max(depth * voxel_size_mm[0], height * voxel_size_mm[1], width * voxel_size_mm[2]) / 2.0
    coords_normalized = coords / max_dist
    
    return coords_normalized.unsqueeze(0) # [1, 3, D, H, W] 반환

def create_base_aperture_coords(num_cp, height, width, pixel_size_mm):
    """각 Control Point에 해당하는 기본 Aperture 물리 좌표계 텐서 생성"""
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    
    y = (torch.arange(height) - cy) * pixel_size_mm[0]
    x = (torch.arange(width) - cx) * pixel_size_mm[1]
    
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=0) # [2, H, W]
    
    max_dist = max(height * pixel_size_mm[0], width * pixel_size_mm[1]) / 2.0
    coords_normalized = coords / max_dist
    
    # [num_cp, 2, H, W] 형태로 복사하여 반환
    return coords_normalized.unsqueeze(0).expand(num_cp, -1, -1, -1) 


# ==========================================
# 2. 모델 아키텍처 (Encoder, Attention, Decoder는 이전과 동일)
# ==========================================
class GantryPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, angles):
        angles_rad = angles * (math.pi / 180.0)
        pe = torch.zeros(angles.size(0), angles.size(1), self.d_model, device=angles.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=angles.device).float() * (-math.log(10000.0) / self.d_model))
        pe[:, :, 0::2] = torch.sin(angles_rad.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(angles_rad.unsqueeze(-1) * div_term)
        return pe

class ApertureEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, d_model)
        )

    def forward(self, fluence, coords):
        x = torch.cat([fluence, coords], dim=1) # [B*180, 3, 160, 160]
        return self.net(x)

class CTEncoder3D(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv3d(64, d_model, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )

    def forward(self, ct, coords):
        x = torch.cat([ct, coords], dim=1) # [B, 4, 192, 192, 192]
        return self.net(x)

class VMATCrossAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, ct_latent, aperture_features):
        B, C, D, H, W = ct_latent.shape
        query = ct_latent.view(B, C, -1).permute(0, 2, 1)
        key = value = aperture_features 
        
        attn_out, attn_weights = self.attention(query, key, value)
        out = self.norm(query + attn_out)
        return out.permute(0, 2, 1).view(B, C, D, H, W), attn_weights

class CTDecoder3D(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(d_model, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=4)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 3. 최적화된 통합 모델 파이프라인 (수정됨)
# ==========================================
class VMATCoordinateDoseModel(nn.Module):
    def __init__(self, d_model=256, num_cp=180):
        super().__init__()
        self.d_model = d_model
        self.num_cp = num_cp
        
        # [최적화 파트]: 모델 초기화 시 좌표계를 딱 한 번만 생성
        base_ct_coords = create_base_ct_coords(192, 192, 192, (3.0, 3.0, 3.0))
        base_ap_coords = create_base_aperture_coords(num_cp, 160, 160, (2.5, 2.5))
        
        # register_buffer를 통해 모델의 상태(State)로 등록
        # - persistent=False: Checkpoint 저장 시 이 거대한 텐서를 모델 가중치 파일에 굳이 저장하지 않도록 함
        # - 이렇게 등록해두면 model.to(device) 호출 시 알아서 CPU/GPU로 이동함
        self.register_buffer('base_ct_coords', base_ct_coords, persistent=False)
        self.register_buffer('base_ap_coords', base_ap_coords, persistent=False)

        # 모델 서브모듈 선언
        self.gantry_pe = GantryPositionalEncoding(d_model)
        self.aperture_encoder = ApertureEncoder(d_model)
        self.ct_encoder = CTEncoder3D(d_model)
        self.cross_attn = VMATCrossAttention(d_model)
        self.decoder = CTDecoder3D(d_model)

    def forward(self, ct, apertures, gantry_angles):
        B = ct.size(0)
        
        # [최적화 파트]: Iteration마다 새로 만들지 않고, 고정된 Buffer를 현재 Batch Size에 맞게 '확장'만 함
        # .expand()는 새로운 메모리를 할당하지 않고 참조 포인터만 복제하므로 연산 시간이 0에 수렴함
        batch_ct_coords = self.base_ct_coords.expand(B, -1, -1, -1, -1) 
        
        # Aperture 좌표도 Batch와 CP 차원에 맞게 형태 변환
        # [num_cp, 2, H, W] -> [1, num_cp, 2, H, W] -> [B, num_cp, 2, H, W] -> [B * num_cp, 2, H, W]
        batch_ap_coords = self.base_ap_coords.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * self.num_cp, 2, 160, 160)
        
        # 이하 연산은 동일
        ct_feat = self.ct_encoder(ct, batch_ct_coords) 
        
        ap_flat = apertures.view(B * self.num_cp, 1, 160, 160)
        ap_feat_flat = self.aperture_encoder(ap_flat, batch_ap_coords)
        ap_feat = ap_feat_flat.view(B, self.num_cp, self.d_model) 
        
        pe = self.gantry_pe(gantry_angles)
        ap_feat = ap_feat + pe
        
        fused_feat, attn_map = self.cross_attn(ct_feat, ap_feat)
        dose = self.decoder(fused_feat)
        
        return dose, attn_map
    
# ==========================================
# 4. 실행 및 형상 확인 (Test)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = VMATCoordinateDoseModel(d_model=256, num_cp=180).to(device)
    
    # 데이터 형상 정의 (Batch=1)
    dummy_ct = torch.randn(1, 1, 192, 192, 192).to(device)       # CT: 192x192x192
    dummy_apertures = torch.randn(1, 180, 160, 160).to(device)   # Aperture: 180x160x160
    
    # Gantry Angles: 181, 183, ... 179 (180개)
    angles = list(range(181, 360, 2)) + list(range(1, 180, 2))
    dummy_angles = torch.tensor([angles], dtype=torch.float32).to(device)
    
    print("모델 연산 시작...")
    pred_dose, attention_weights = model(dummy_ct, dummy_apertures, dummy_angles)
    
    print(f"최종 선량 출력 Shape: {pred_dose.shape}")      # [1, 1, 192, 192, 192]
    print(f"어텐션 맵 Shape: {attention_weights.shape}") # [1, 1728, 180]
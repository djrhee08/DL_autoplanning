import torch
import torch.nn as nn

class DifferentiableMLCAperture(nn.Module):
    """
    TrueBeam Millennium 120 MLC 포지션을 160x160 2D Aperture (Fluence map)로 변환.
    - Pixel size: 2.5 mm
    - Field size: 400 mm x 400 mm
    """
    def __init__(self, pixel_size=2.5, grid_size=160, tau=0.1):
        super().__init__()
        self.pixel_size = pixel_size
        self.grid_size = grid_size
        self.tau = tau  # Differentiable sharpness (낮을수록 sharp)

        # 1. 160개의 이미지 행(Row)이 60쌍의 MLC 중 어디에 해당하는지 매핑
        # 1.0cm (4픽셀) x 10쌍 / 0.5cm (2픽셀) x 40쌍 / 1.0cm (4픽셀) x 10쌍
        indices = []
        for i in range(10):      indices.extend([i] * 4)     # Top outer leaves
        for i in range(10, 50):  indices.extend([i] * 2)     # Inner leaves
        for i in range(50, 60):  indices.extend([i] * 4)     # Bottom outer leaves
        self.register_buffer('row_to_leaf', torch.tensor(indices, dtype=torch.long))

        # 2. X축 물리 좌표계 생성 (Isocenter 기준 mm 단위)
        # 160개 픽셀의 중심 좌표: -198.75 부터 +198.75 까지
        x_coords = (torch.arange(grid_size) - (grid_size - 1) / 2.0) * pixel_size
        self.register_buffer('x_grid', x_coords.view(1, 1, grid_size))

    def forward(self, mlc_positions):
        """
        Args:
            mlc_positions: [B, 60, 2] or [B, N_cp, 60, 2] tensor of leaf positions (mm).
                           index 0은 왼쪽(X1) Leaf 위치 (mm),
                           index 1은 오른쪽(X2) Leaf 위치 (mm).
                           (Isocenter 기준 좌표계라고 가정)
        Returns:
            aperture: [B, 1, 160, 160] or [B, N_cp, 1, 160, 160]
        """
        extra_dims = mlc_positions.shape[:-2]  # (B,) or (B, N_cp)
        mlc_flat = mlc_positions.reshape(-1, 60, 2)  # [B*N_cp, 60, 2]

        # [Batch, 160, 2] 형태로 확장 (160개의 각 행에 해당하는 MLC 위치 할당)
        row_mlc = mlc_flat[:, self.row_to_leaf, :]

        # [Batch, 160, 1] 형태로 분리
        left_leaf = row_mlc[:, :, 0:1]
        right_leaf = row_mlc[:, :, 1:2]

        # Broadcasting 연산을 통한 160x160 Grid 상의 투과도(Transmission) 계산
        # x_grid는 [1, 1, 160], left_leaf/right_leaf는 [B, 160, 1] 이므로 결과는 [B, 160, 160]
        open_left = torch.sigmoid((self.x_grid - left_leaf) / self.tau)
        open_right = torch.sigmoid((right_leaf - self.x_grid) / self.tau)

        # 양쪽 조건(왼쪽 잎보다 오른쪽, 오른쪽 잎보다 왼쪽)을 모두 만족하는 영역
        aperture = open_left * open_right  # [B*N_cp, 160, 160]

        # Channel 차원 추가 후 원래 shape으로 복원
        aperture = aperture.unsqueeze(1)   # [B*N_cp, 1, 160, 160]
        return aperture.reshape(*extra_dims, 1, self.grid_size, self.grid_size)


class DifferentiableJawAperture(nn.Module):
    """
    Jaw positions [B, 2, 2] → 160×160 aperture [B, 1, 160, 160].

    Input layout:
        jaw_positions[:, 0, :] = [X1, X2]  –  left / right jaw boundary (mm)
        jaw_positions[:, 1, :] = [Y1, Y2]  –  inferior / superior jaw boundary (mm)

    Mirrors DifferentiableMLCAperture: sigmoid gating in both axes, fully differentiable.
    """
    def __init__(self, pixel_size=2.5, grid_size=160, tau=0.1):
        super().__init__()
        self.pixel_size = pixel_size
        self.grid_size  = grid_size
        self.tau        = tau

        # Physical coordinates of each pixel centre (mm, isocenter = 0)
        coords = (torch.arange(grid_size) - (grid_size - 1) / 2.0) * pixel_size

        # x_grid: [1, 1, 160]  (broadcast over rows)
        # y_grid: [1, 160, 1]  (broadcast over cols)
        self.register_buffer('x_grid', coords.view(1, 1, grid_size))
        self.register_buffer('y_grid', coords.view(1, grid_size, 1))

    def forward(self, jaw_positions):
        """
        Args:
            jaw_positions: [B, 2, 2]
                           [:, 0, 0] = X1 (left jaw, mm)
                           [:, 0, 1] = X2 (right jaw, mm)
                           [:, 1, 0] = Y1 (inferior jaw, mm)
                           [:, 1, 1] = Y2 (superior jaw, mm)
        Returns:
            aperture: [B, 1, 160, 160]
        """
        # X jaw: gates columns  →  [B, 1, 160]
        x1 = jaw_positions[:, 0, 0:1].unsqueeze(-1)   # [B, 1, 1]
        x2 = jaw_positions[:, 0, 1:2].unsqueeze(-1)   # [B, 1, 1]
        open_x = torch.sigmoid((self.x_grid - x1) / self.tau) * \
                 torch.sigmoid((x2 - self.x_grid) / self.tau)  # [B, 1, 160]

        # Y jaw: gates rows  →  [B, 160, 1]
        y1 = jaw_positions[:, 1, 0:1].unsqueeze(-1)   # [B, 1, 1]
        y2 = jaw_positions[:, 1, 1:2].unsqueeze(-1)   # [B, 1, 1]
        open_y = torch.sigmoid((self.y_grid - y1) / self.tau) * \
                 torch.sigmoid((y2 - self.y_grid) / self.tau)  # [B, 160, 1]

        # Outer product: [B, 160, 160]
        aperture = open_y * open_x

        return aperture.unsqueeze(1)   # [B, 1, 160, 160]
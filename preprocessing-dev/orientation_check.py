import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

current_dir = pathlib.Path(__file__).parent.resolve()
print(current_dir)

ct = np.load(os.path.join(current_dir, 'npy_total/xxxxxx', 'xxxxxx_VMAT_1arcs_00_A_CT.npy'))  # [192, 192, 192]   

# 1. z축 가장 양 끝의 axial slice 확인
"""
HFS (Head First Supine)에서 +z 방향은 superior(머리 쪽)입니다. 따라서:
CT[0, :, :]: physically 가장 inferior (발 쪽)
CT[191, :, :]: physically 가장 superior (머리 쪽)
"""
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(ct[30,   :, :], cmap='gray');   axes[0].set_title('CT[30]   (z=0)')
axes[1].imshow(ct[63,  :, :], cmap='gray');   axes[1].set_title('CT[63]')
axes[2].imshow(ct[127, :, :], cmap='gray');   axes[2].set_title('CT[127]')
axes[3].imshow(ct[170, :, :], cmap='gray');   axes[3].set_title('CT[170]  (z=191)')
plt.savefig(os.path.join(current_dir, 'ct_axial.png'))

# Sagittal: y-z plane, x 고정
"""
origin='lower'로 그리면 row index가 위로 갈수록 증가합니다. 
만약 row가 z축이고 +z가 superior라면, 머리가 이미지 위쪽에 보여야 정상입니다. 
머리가 아래쪽에 보이면 z가 뒤집혀 있는 것입니다.
"""
sagittal = ct[:, :, 96]  # [192, 192]  z is rows, y is cols
plt.figure(figsize=(6, 6))
plt.imshow(sagittal, cmap='gray', origin='lower')  # origin='lower' 중요!
plt.title('Sagittal — superior should be UP')
plt.savefig(os.path.join(current_dir, 'ct_sagittal.png'))



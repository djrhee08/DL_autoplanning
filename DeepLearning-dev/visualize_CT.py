import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

current_dir = Path(__file__).resolve().parent
data_dir = os.path.join(current_dir, '../preprocessing-dev/npy_total/test')

CT_file   = 'test_plan0_A_CT.npy'
dose_file = 'test_plan0_A_dose.npy'

CT   = np.load(os.path.join(data_dir, CT_file))    # [192, 192, 192]
dose = np.load(os.path.join(data_dir, dose_file))  # [192, 192, 192]

print(f"CT   shape: {CT.shape}   range: [{CT.min():.3f}, {CT.max():.3f}]")
print(f"Dose shape: {dose.shape}  range: [{dose.min():.4f}, {dose.max():.4f}] Gy")

num_slices = CT.shape[0]
dose_max = dose.max()

# Overlay alpha slider value (mutable via widget)
overlay_alpha = [0.4]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#1a1a1a')
plt.subplots_adjust(bottom=0.22, left=0.04, right=0.96, top=0.90, wspace=0.12)

# CT panel
axes[0].set_facecolor('black')
axes[0].set_title('CT', color='white', fontsize=12)
axes[0].axis('off')
ct_img = axes[0].imshow(CT[0], cmap='gray', vmin=0, vmax=1,
                         origin='upper', interpolation='nearest')

# Dose panel
axes[1].set_facecolor('black')
axes[1].set_title('Dose', color='white', fontsize=12)
axes[1].axis('off')
dose_img = axes[1].imshow(dose[0], cmap='jet', vmin=0, vmax=dose_max,
                           origin='upper', interpolation='nearest')
fig.colorbar(dose_img, ax=axes[1], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

# Overlay panel
axes[2].set_facecolor('black')
axes[2].set_title('CT + Dose overlay', color='white', fontsize=12)
axes[2].axis('off')
overlay_ct   = axes[2].imshow(CT[0], cmap='gray', vmin=0, vmax=1,
                               origin='upper', interpolation='nearest')
overlay_dose = axes[2].imshow(dose[0], cmap='jet', vmin=0, vmax=dose_max,
                               origin='upper', interpolation='nearest',
                               alpha=overlay_alpha[0])
fig.colorbar(overlay_dose, ax=axes[2], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')

suptitle = fig.suptitle('', color='white', fontsize=13, y=0.97)

# Slice slider
ax_slider = fig.add_axes([0.12, 0.09, 0.76, 0.03], facecolor='#333333')
slider = Slider(ax_slider, 'Slice (z)', 0, num_slices - 1, valinit=0, valstep=1,
                color='#5599ff')
slider.label.set_color('white')
slider.valtext.set_color('white')

# Alpha slider
ax_alpha = fig.add_axes([0.12, 0.04, 0.76, 0.03], facecolor='#333333')
alpha_slider = Slider(ax_alpha, 'Overlay α', 0.0, 1.0, valinit=overlay_alpha[0],
                      color='#ff9955')
alpha_slider.label.set_color('white')
alpha_slider.valtext.set_color('white')

def draw(z):
    z = int(z)
    ct_img.set_data(CT[z])
    dose_img.set_data(dose[z])
    overlay_ct.set_data(CT[z])
    overlay_dose.set_data(dose[z])
    suptitle.set_text(f"Slice {z} / {num_slices - 1}")
    fig.canvas.draw_idle()

def update_alpha(val):
    overlay_dose.set_alpha(val)
    fig.canvas.draw_idle()

alpha_slider.on_changed(update_alpha)

slider.on_changed(draw)
draw(0)

def on_key(event):
    cur = int(slider.val)
    if event.key == 'right' or event.key == 'up':
        slider.set_val(min(cur + 1, num_slices - 1))
    elif event.key == 'left' or event.key == 'down':
        slider.set_val(max(cur - 1, 0))
    elif event.key == 'ctrl+right' or event.key == 'ctrl+up':
        slider.set_val(min(cur + 10, num_slices - 1))
    elif event.key == 'ctrl+left' or event.key == 'ctrl+down':
        slider.set_val(max(cur - 10, 0))

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

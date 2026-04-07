"""
visualize_npy.py  –  Interactive viewer for jaw [180,160,160] and MLC [180,160,160] apertures.

Shows a 1×3 panel (Jaw | MLC | Overlay) for each of the 180 control points.
Use the slider or ← / → arrow keys to scrub through control points.

Usage:
    python DeepLearning-dev/visualize_npy.py
    python DeepLearning-dev/visualize_npy.py --jaw path/to/jaw.npy --mlc path/to/mlc.npy
"""

import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Patch
from pathlib import Path


def build_overlay(jaw_2d: np.ndarray, mlc_2d: np.ndarray) -> np.ndarray:
    """
    RGB overlay:
      both open   → white
      jaw only    → blue
      MLC only    → red
      both closed → black
    """
    rgb = np.zeros((jaw_2d.shape[0], jaw_2d.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = mlc_2d               # red   = MLC
    rgb[:, :, 2] = jaw_2d               # blue  = jaw
    rgb[:, :, 1] = jaw_2d * mlc_2d      # green = overlap → white where both open
    return rgb


def main():
    parser = argparse.ArgumentParser()
    current_dir = Path(__file__).resolve().parent

    def _find(pattern):
        hits = glob.glob(os.path.join(current_dir, pattern))
        return hits[0] if hits else None

    parser.add_argument('--jaw', default=None)
    parser.add_argument('--mlc', default=None)
    args = parser.parse_args()

    jaw_path = args.jaw or _find('*_jaw_*.npy') or os.path.join(current_dir, 'jaw.npy')
    mlc_path = args.mlc or _find('*_mlc_*.npy') or os.path.join(current_dir, 'mlc.npy')

    if not os.path.exists(jaw_path):
        raise FileNotFoundError(f"Jaw file not found: {jaw_path}")
    if not os.path.exists(mlc_path):
        raise FileNotFoundError(f"MLC file not found: {mlc_path}")

    jaw_all = np.load(jaw_path)   # [180, 160, 160]
    mlc_all = np.load(mlc_path)   # [180, 160, 160]

    print(f"Loaded jaw: {jaw_all.shape}  from {os.path.basename(jaw_path)}")
    print(f"Loaded MLC: {mlc_all.shape}  from {os.path.basename(mlc_path)}")

    num_cps = jaw_all.shape[0]
    angles  = (181.0 + 2.0 * np.arange(num_cps)) % 360.0

    grid_size    = jaw_all.shape[1]
    pixel_size   = 2.5
    half         = (grid_size - 1) / 2.0 * pixel_size
    extents      = [-half - pixel_size/2, half + pixel_size/2,
                    -half - pixel_size/2, half + pixel_size/2]

    # Pre-build overlays
    ovl_all = np.stack([build_overlay(jaw_all[i], mlc_all[i]) for i in range(num_cps)])

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('#1a1a1a')
    plt.subplots_adjust(bottom=0.18, left=0.04, right=0.98, top=0.88, wspace=0.12)

    titles = ['Jaw', 'MLC', 'Overlay']
    cmaps  = ['Blues', 'Oranges', None]

    img_handles = []
    for ax, title, cmap in zip(axes, titles, cmaps):
        ax.set_facecolor('black')
        ax.set_title(title, color='white', fontsize=12)
        ax.set_xlabel('X (mm)', color='#aaaaaa', fontsize=9)
        ax.set_ylabel('Y (mm)', color='#aaaaaa', fontsize=9)
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

        if cmap is not None:
            h = ax.imshow(np.zeros((grid_size, grid_size)), cmap=cmap,
                          vmin=0, vmax=1, origin='lower', extent=extents,
                          interpolation='nearest')
        else:
            h = ax.imshow(np.zeros((grid_size, grid_size, 3), dtype=np.float32),
                          origin='lower', extent=extents, interpolation='nearest')

        ax.axhline(0, color='#ffffff44', linewidth=0.6, linestyle='--')
        ax.axvline(0, color='#ffffff44', linewidth=0.6, linestyle='--')
        img_handles.append(h)

    legend_elems = [
        Patch(facecolor=(0, 0, 1), label='Jaw open'),
        Patch(facecolor=(1, 0, 0), label='MLC open'),
        Patch(facecolor=(1, 1, 1), label='Both open'),
        Patch(facecolor='black',   label='Both closed', edgecolor='#555'),
    ]
    axes[2].legend(handles=legend_elems, loc='lower right',
                   fontsize=7, framealpha=0.5, labelcolor='white',
                   facecolor='#333333', edgecolor='#555555')

    suptitle = fig.suptitle('', color='white', fontsize=13, y=0.97)

    # ── Slider ──
    ax_slider = fig.add_axes([0.12, 0.06, 0.76, 0.03], facecolor='#333333')
    slider = Slider(ax_slider, 'CP', 0, num_cps - 1, valinit=0, valstep=1,
                    color='#5599ff')
    slider.label.set_color('white')
    slider.valtext.set_color('white')

    def draw(cp_idx):
        cp_idx = int(cp_idx)
        img_handles[0].set_data(jaw_all[cp_idx])
        img_handles[1].set_data(mlc_all[cp_idx])
        img_handles[2].set_data(ovl_all[cp_idx])
        suptitle.set_text(
            f"CP {cp_idx:3d} / {num_cps - 1}   |   Gantry {angles[cp_idx]:.1f}°"
        )
        fig.canvas.draw_idle()

    slider.on_changed(draw)
    draw(0)

    def on_key(event):
        cur = int(slider.val)
        if event.key == 'right':
            slider.set_val(min(cur + 1, num_cps - 1))
        elif event.key == 'left':
            slider.set_val(max(cur - 1, 0))
        elif event.key == 'ctrl+right':
            slider.set_val(min(cur + 10, num_cps - 1))
        elif event.key == 'ctrl+left':
            slider.set_val(max(cur - 10, 0))

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


if __name__ == '__main__':
    main()

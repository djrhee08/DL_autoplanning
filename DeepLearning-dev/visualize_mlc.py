"""
visualize_mlc.py  –  Interactive viewer for jaw [180,2,2] and MLC [180,60,2] position arrays.

Converts raw leaf/jaw positions to 560×560 aperture maps on the fly via
DifferentiableJawAperture and DifferentiableMLCAperture (MLC2Aperture.py),
then shows a 1×3 panel (Jaw | MLC | Overlay) for each of the 180 control points.

Usage:
    python DeepLearning-dev/visualize_mlc.py
    python DeepLearning-dev/visualize_mlc.py --jaw path/to/jaw.npy --mlc path/to/mlc.npy
"""

import argparse
import glob
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from MLC2Aperture import DifferentiableMLCAperture, DifferentiableJawAperture, vmat_gantry_angles


def positions_to_apertures(jaw_raw: np.ndarray, mlc_raw: np.ndarray, average: bool = False):
    """
    Convert raw position arrays to 2D aperture maps.

    Args:
        jaw_raw : [180, 2, 2]  – jaw positions (mm)
        mlc_raw : [180, 60, 2] – MLC leaf positions (mm)
        average : if True, average adjacent CP pairs (RayStation VMAT convention).
                  Output has 179 slices instead of 180.

    Returns:
        jaw_2d   : [180, 560, 560] or [179, 560, 560]
        mlc_2d   : [180, 560, 560] or [179, 560, 560]
        angles   : [180] or [179]  gantry angles (degrees)
    """
    jaw_module = DifferentiableJawAperture(tau=0.1)
    mlc_module = DifferentiableMLCAperture(tau=0.1)

    if average:
        # Reshape to [1, 180, *, *] for the N_cp averaging path
        jaw_t = torch.from_numpy(jaw_raw).float().unsqueeze(0)   # [1, 180, 2, 2]
        mlc_t = torch.from_numpy(mlc_raw).float().unsqueeze(0)   # [1, 180, 60, 2]
        with torch.no_grad():
            jaw_2d = jaw_module(jaw_t, average=True).squeeze(0).squeeze(1).numpy()  # [179, 560, 560]
            mlc_2d = mlc_module(mlc_t, average=True).squeeze(0).squeeze(1).numpy()  # [179, 560, 560]
    else:
        jaw_t = torch.from_numpy(jaw_raw).float()   # [180, 2, 2]
        mlc_t = torch.from_numpy(mlc_raw).float()   # [180, 60, 2]
        with torch.no_grad():
            jaw_2d = jaw_module(jaw_t).squeeze(1).numpy()   # [180, 560, 560]
            mlc_2d = mlc_module(mlc_t).squeeze(1).numpy()   # [180, 560, 560]

    angles = vmat_gantry_angles(average=average)
    return jaw_2d, mlc_2d, angles


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
    data_dir = current_dir.parent / 'preprocessing-dev' / 'npy_total'

    def _find(pattern):
        for root in [str(data_dir), str(current_dir)]:
            hits = glob.glob(os.path.join(root, '**', pattern), recursive=True)
            if hits:
                return hits[0]
        return None

    parser.add_argument('--jaw', default=None)
    parser.add_argument('--mlc', default=None)
    parser.add_argument('--average', default=False, action='store_true',
                        help='Average adjacent CP positions (RayStation VMAT convention). '
                             'Produces 179 aperture slices at midpoint gantry angles.')
    args = parser.parse_args()

    jaw_path = args.jaw or _find('*_jaw.npy') or os.path.join(current_dir, 'jaw.npy')
    mlc_path = args.mlc or _find('*_mlc.npy') or os.path.join(current_dir, 'mlc.npy')

    if not os.path.exists(jaw_path):
        raise FileNotFoundError(f"Jaw file not found: {jaw_path}")
    if not os.path.exists(mlc_path):
        raise FileNotFoundError(f"MLC file not found: {mlc_path}")

    jaw_raw = np.load(jaw_path)   # [180, 2, 2]
    mlc_raw = np.load(mlc_path)   # [180, 60, 2]

    print(f"Loaded jaw: {jaw_raw.shape}  from {os.path.basename(jaw_path)}")
    print(f"Loaded MLC: {mlc_raw.shape}  from {os.path.basename(mlc_path)}")
    mode_str = "averaged (RayStation)" if args.average else "per-CP"
    print(f"Converting positions to 2D apertures [{mode_str}]...")

    jaw_all, mlc_all, angles = positions_to_apertures(jaw_raw, mlc_raw, average=args.average)

    print(f"Jaw aperture: {jaw_all.shape}  range [{jaw_all.min():.3f}, {jaw_all.max():.3f}]")
    print(f"MLC aperture: {mlc_all.shape}  range [{mlc_all.min():.3f}, {mlc_all.max():.3f}]")

    num_cps = jaw_all.shape[0]

    grid_size  = 560
    pixel_size = 1.0
    half       = (grid_size - 1) / 2.0 * pixel_size
    extents    = [-half - pixel_size / 2, half + pixel_size / 2,
                  -half - pixel_size / 2, half + pixel_size / 2]

    # Pre-build overlays for all CPs
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

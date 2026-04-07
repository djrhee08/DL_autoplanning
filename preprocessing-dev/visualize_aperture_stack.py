"""
Visualize jaw/MLC aperture stacks slice by slice.

Usage:
    python visualize_aperture_stack.py <jaw_npy> <mlc_npy>

Example:
    python visualize_aperture_stack.py "npy_total/test/test_vmat_3arc_A_jaw_odd_start181.npy" \
                                       "npy_total/test/test_vmat_3arc_A_mlc_odd_start181.npy"

Controls:
    Left/Right arrow  — previous/next slice (gantry angle)
    Home / End        — jump to first / last slice
    Q or Escape       — quit
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import re
import os


def parse_parity_start(filepath):
    """Extract parity and actual_start from filename."""
    m = re.search(r'_(odd|even)_start(\d+)', os.path.basename(filepath))
    if m:
        parity = m.group(1)
        actual_start = int(m.group(2))
        canonical_start = 181 if parity == 'odd' else 182
        return parity, canonical_start, actual_start
    return None, None, None


def slot_to_gantry(slot, canonical_start):
    """Convert slot index [0..179] to gantry angle in degrees."""
    return (canonical_start + slot * 2) % 360


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    jaw_path = sys.argv[1]
    mlc_path = sys.argv[2]

    jaw_stack = np.load(jaw_path)   # (180, 560, 560)
    mlc_stack = np.load(mlc_path)   # (180, 560, 560)

    assert jaw_stack.shape == mlc_stack.shape, "jaw and mlc stacks must have the same shape"
    n_slots = jaw_stack.shape[0]

    parity, canonical_start, actual_start = parse_parity_start(jaw_path)
    label_prefix = f"parity={parity}, canonical_start={canonical_start}°, actual_start={actual_start}°" if parity else ""

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(os.path.basename(jaw_path).replace('_jaw', ''), fontsize=10)
    gs = gridspec.GridSpec(2, 2, height_ratios=[20, 1])

    ax_jaw = fig.add_subplot(gs[0, 0])
    ax_mlc = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, :])

    ax_jaw.set_title("Jaw")
    ax_mlc.set_title("MLC")
    for ax in (ax_jaw, ax_mlc):
        ax.axis('off')

    im_jaw = ax_jaw.imshow(jaw_stack[0], cmap='gray', vmin=0, vmax=1, origin='upper')
    im_mlc = ax_mlc.imshow(mlc_stack[0], cmap='gray', vmin=0, vmax=1, origin='upper')

    def gantry_label(slot):
        if parity is not None:
            g = slot_to_gantry(slot, canonical_start)
            return f"Slot {slot}  |  Gantry ≈ {g}°  |  {label_prefix}"
        return f"Slot {slot}"

    title = fig.text(0.5, 0.93, gantry_label(0), ha='center', va='top', fontsize=9)

    slider = Slider(ax_slider, 'Slot', 0, n_slots - 1, valinit=0, valstep=1)

    def update(val):
        slot = int(slider.val)
        im_jaw.set_data(jaw_stack[slot])
        im_mlc.set_data(mlc_stack[slot])
        title.set_text(gantry_label(slot))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        slot = int(slider.val)
        if event.key == 'right':
            slider.set_val(min(slot + 1, n_slots - 1))
        elif event.key == 'left':
            slider.set_val(max(slot - 1, 0))
        elif event.key == 'home':
            slider.set_val(0)
        elif event.key == 'end':
            slider.set_val(n_slots - 1)
        elif event.key in ('q', 'escape'):
            plt.close('all')

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


if __name__ == '__main__':
    main()
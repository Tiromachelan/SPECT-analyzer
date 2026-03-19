"""
SPECT Scan Reconstruction Viewer
Displays two .raw SPECT reconstructions side by side with a slider
to navigate through the 64 slices (128x128 pixels, 32-bit float).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from tkinter import Tk, filedialog


def load_raw(path: str) -> np.ndarray:
    """Load a .raw file as a stack of 128x128 float32 images."""
    data = np.fromfile(path, dtype=np.float32)
    expected = 128 * 128 * 64
    if data.size != expected:
        raise ValueError(
            f"Expected {expected} floats in '{path}', got {data.size}"
        )
    return data.reshape((64, 128, 128))


def pick_files() -> tuple[str, str]:
    """Open two file-picker dialogs to choose the .raw files."""
    root = Tk()
    root.withdraw()

    path1 = filedialog.askopenfilename(
        title="Select FIRST .raw file",
        filetypes=[("Raw files", "*.raw"), ("All files", "*.*")],
    )
    if not path1:
        sys.exit("No first file selected.")

    path2 = filedialog.askopenfilename(
        title="Select SECOND .raw file",
        filetypes=[("Raw files", "*.raw"), ("All files", "*.*")],
    )
    if not path2:
        sys.exit("No second file selected.")

    root.destroy()
    return path1, path2


def main() -> None:
    # --- load data -----------------------------------------------------------
    if len(sys.argv) == 3:
        path1, path2 = sys.argv[1], sys.argv[2]
    else:
        path1, path2 = pick_files()

    vol1 = load_raw(path1)
    vol2 = load_raw(path2)

    # --- normalise each volume to [0, 1] for display ------------------------
    def norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi != lo else v - lo

    vol1 = norm(vol1)
    vol2 = norm(vol2)

    # --- orientation helpers -------------------------------------------------
    ORIENTATIONS = ("Axial", "Coronal", "Sagittal")
    orientation: list[str] = ["Axial"]

    def get_slice(vol: np.ndarray, idx: int, orient: str) -> np.ndarray:
        """Return a 2-D cross-section for the given orientation and 0-based index."""
        if orient == "Axial":
            return vol[idx]          # (128, 128)
        elif orient == "Coronal":
            return vol[:, idx, :]    # (64, 128)
        else:                        # Sagittal
            return vol[:, :, idx]    # (64, 128)

    def n_slices(orient: str) -> int:
        return 64 if orient == "Axial" else 128

    def slice_shape(orient: str) -> tuple[int, int]:
        """Return (rows, cols) of one cross-section in the given orientation."""
        if orient == "Axial":
            return (128, 128)
        elif orient == "Coronal":
            return (64, 128)   # z rows, x cols
        else:                  # Sagittal
            return (64, 128)   # z rows, y cols

    def make_extent(orient: str) -> list[float]:
        rows, cols = slice_shape(orient)
        return [-0.5, cols - 0.5, rows - 0.5, -0.5]

    # --- set up figure -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.97)

    init_slice = 1
    im1 = ax1.imshow(
        get_slice(vol1, init_slice - 1, orientation[0]),
        cmap="gray", vmin=0, vmax=1, aspect="equal",
        extent=make_extent(orientation[0]),
    )
    im2 = ax2.imshow(
        get_slice(vol2, init_slice - 1, orientation[0]),
        cmap="gray", vmin=0, vmax=1, aspect="equal",
        extent=make_extent(orientation[0]),
    )

    label1 = path1.rsplit("/", 1)[-1]
    label2 = path2.rsplit("/", 1)[-1]
    ax1.set_title(label1)
    ax2.set_title(label2)
    for ax in (ax1, ax2):
        ax.set_axis_off()

    # --- orientation radio buttons ------------------------------------------
    ax_radio = fig.add_axes([0.01, 0.35, 0.11, 0.22])
    ax_radio.set_title("View", fontsize=9)
    radio = RadioButtons(ax_radio, ORIENTATIONS, active=0)
    for lbl in radio.labels:
        lbl.set_fontsize(9)

    # --- slider --------------------------------------------------------------
    ax_slider = fig.add_axes([0.15, 0.06, 0.70, 0.04])
    slider = Slider(
        ax_slider, "Slice", 1, 64, valinit=init_slice, valstep=1, valfmt="%d"
    )

    def update(val: float) -> None:
        idx = int(val) - 1
        orient = orientation[0]
        im1.set_data(get_slice(vol1, idx, orient))
        im2.set_data(get_slice(vol2, idx, orient))
        fig.canvas.draw_idle()

    def on_orient(label: str) -> None:
        orientation[0] = label
        new_max = n_slices(label)
        # Update slider range to match the new orientation's depth
        slider.valmax = new_max
        slider.ax.set_xlim(slider.valmin, new_max)
        new_val = min(int(slider.val), new_max)
        # Update image extent so axes resize to match the new slice shape
        extent = make_extent(label)
        for im, ax in ((im1, ax1), (im2, ax2)):
            im.set_extent(extent)
            ax.set_aspect("equal")
        slider.set_val(new_val)
        # set_val fires on_changed; if val didn't change, force a redraw
        if new_val == int(slider.val):
            update(new_val)

    radio.on_clicked(on_orient)
    slider.on_changed(update)

    fig.suptitle("SPECT Reconstruction Comparison", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()

"""
SPECT Scan Reconstruction Viewer
Displays two .raw SPECT reconstructions side by side with a slider
to navigate through the 64 slices (128x128 pixels, 32-bit float).
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

    # --- set up figure -------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    plt.subplots_adjust(bottom=0.18)

    init_slice = 1
    im1 = ax1.imshow(vol1[init_slice - 1], cmap="gray", vmin=0, vmax=1)
    im2 = ax2.imshow(vol2[init_slice - 1], cmap="gray", vmin=0, vmax=1)

    label1 = path1.rsplit("/", 1)[-1]
    label2 = path2.rsplit("/", 1)[-1]
    ax1.set_title(label1)
    ax2.set_title(label2)
    for ax in (ax1, ax2):
        ax.set_axis_off()

    # --- slider --------------------------------------------------------------
    ax_slider = fig.add_axes([0.15, 0.06, 0.70, 0.04])
    slider = Slider(
        ax_slider, "Slice", 1, 64, valinit=init_slice, valstep=1, valfmt="%d"
    )

    def update(val: float) -> None:
        idx = int(val) - 1
        im1.set_data(vol1[idx])
        im2.set_data(vol2[idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    fig.suptitle("SPECT Reconstruction Comparison", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()

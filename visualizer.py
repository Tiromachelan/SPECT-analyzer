"""
SPECT Scan Reconstruction Viewer
Displays two .raw SPECT reconstructions side by side with a slider
to navigate through the 64 slices (128x128 pixels, 32-bit float).
A third panel shows the per-slice SSIM heatmap; MSE, MAE, and SSIM
are displayed both per-slice and as whole-volume aggregates.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from skimage.metrics import structural_similarity
from tkinter import Tk, filedialog


def compute_slice_metrics(
    s1: np.ndarray, s2: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    """Return (MSE, MAE, SSIM_scalar, SSIM_map) for two 2-D slices in [0, 1]."""
    diff = s1.astype(np.float64) - s2.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    ssim_val, ssim_map = structural_similarity(
        s1.astype(np.float64),
        s2.astype(np.float64),
        full=True,
        data_range=1.0,
    )
    return mse, mae, float(ssim_val), ssim_map


def compute_volume_metrics(v1: np.ndarray, v2: np.ndarray) -> tuple[float, float, float]:
    """Return (MSE, MAE, SSIM) aggregated over the entire 3-D volume."""
    diff = v1.astype(np.float64) - v2.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    # Average per-axial-slice SSIM scalars for a consistent volume SSIM
    ssim_sum = 0.0
    for i in range(v1.shape[0]):
        ssim_sum += structural_similarity(
            v1[i].astype(np.float64),
            v2[i].astype(np.float64),
            data_range=1.0,
        )
    ssim_vol = ssim_sum / v1.shape[0]
    return mse, mae, ssim_vol


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

    mip_mode: list[bool] = [False]

    def get_slice(vol: np.ndarray, idx: int, orient: str) -> np.ndarray:
        """Return a 2-D cross-section for the given orientation and 0-based index."""
        if orient == "Axial":
            return vol[idx]          # (128, 128)
        elif orient == "Coronal":
            return vol[:, idx, :]    # (64, 128)
        else:                        # Sagittal
            return vol[:, :, idx]    # (64, 128)

    def get_mip(vol: np.ndarray, orient: str) -> np.ndarray:
        """Return a maximum-intensity projection along the depth axis."""
        if orient == "Axial":
            return vol.max(axis=0)   # collapse 64 slices → (128, 128)
        elif orient == "Coronal":
            return vol.max(axis=1)   # collapse 128 rows  → (64, 128)
        else:                        # Sagittal
            return vol.max(axis=2)   # collapse 128 cols  → (64, 128)

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

    # --- precompute whole-volume metrics ------------------------------------
    vol_mse, vol_mae, vol_ssim = compute_volume_metrics(vol1, vol2)
    vol_metrics_str = (
        f"Volume — MSE: {vol_mse:.4f} | MAE: {vol_mae:.4f} | SSIM: {vol_ssim:.4f}"
    )

    # --- set up figure -------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(left=0.09, bottom=0.18, right=0.97, top=0.82, wspace=0.32)

    init_slice = 1
    init_s1 = get_slice(vol1, init_slice - 1, orientation[0])
    init_s2 = get_slice(vol2, init_slice - 1, orientation[0])
    init_mse, init_mae, init_ssim_val, init_ssim_map = compute_slice_metrics(init_s1, init_s2)

    im1 = ax1.imshow(
        init_s1,
        cmap="gray", vmin=0, vmax=1, aspect="equal",
        extent=make_extent(orientation[0]),
    )
    im2 = ax2.imshow(
        init_s2,
        cmap="gray", vmin=0, vmax=1, aspect="equal",
        extent=make_extent(orientation[0]),
    )
    im3 = ax3.imshow(
        init_ssim_map,
        cmap="RdYlGn", vmin=0, vmax=1, aspect="equal",
        extent=make_extent(orientation[0]),
    )
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    label1 = path1.rsplit("/", 1)[-1]
    label2 = path2.rsplit("/", 1)[-1]
    ax1.set_title(label1)
    ax2.set_title(label2)
    ax3.set_title("SSIM Heatmap")
    for ax in (ax1, ax2, ax3):
        ax.set_axis_off()

    # --- metrics text --------------------------------------------------------
    # Whole-volume metrics: fixed, shown below the suptitle
    fig.text(
        0.5, 0.88, vol_metrics_str,
        ha="center", va="top", fontsize=9, color="dimgray",
    )
    # Per-slice metrics: dynamic, updated in redraw()
    init_metrics_str = (
        f"Slice {init_slice} — MSE: {init_mse:.4f} | MAE: {init_mae:.4f}"
        f" | SSIM: {init_ssim_val:.4f}"
    )
    metrics_text = fig.text(
        0.5, 0.94, init_metrics_str,
        ha="center", va="top", fontsize=10, fontweight="bold",
    )

    # --- mode radio buttons (Slice / MIP) ------------------------------------
    ax_mode = fig.add_axes([0.01, 0.62, 0.07, 0.14])
    ax_mode.set_title("Mode", fontsize=9)
    radio_mode = RadioButtons(ax_mode, ("Slice", "MIP"), active=0)
    for lbl in radio_mode.labels:
        lbl.set_fontsize(9)

    # --- orientation radio buttons ------------------------------------------
    ax_radio = fig.add_axes([0.01, 0.33, 0.07, 0.22])
    ax_radio.set_title("View", fontsize=9)
    radio = RadioButtons(ax_radio, ORIENTATIONS, active=0)
    for lbl in radio.labels:
        lbl.set_fontsize(9)

    # --- slider --------------------------------------------------------------
    ax_slider = fig.add_axes([0.15, 0.06, 0.70, 0.04])
    slider = Slider(
        ax_slider, "Slice", 1, 64, valinit=init_slice, valstep=1,
    )
    slider.valtext.set_text(f"{init_slice}/{n_slices(orientation[0])}")

    def redraw() -> None:
        orient = orientation[0]
        extent = make_extent(orient)
        for im, ax in ((im1, ax1), (im2, ax2), (im3, ax3)):
            im.set_extent(extent)
            ax.set_aspect("equal")
        if mip_mode[0]:
            s1 = get_mip(vol1, orient)
            s2 = get_mip(vol2, orient)
            slider.valtext.set_text("MIP")
            slice_label = "MIP"
        else:
            idx = int(slider.val) - 1
            s1 = get_slice(vol1, idx, orient)
            s2 = get_slice(vol2, idx, orient)
            slider.valtext.set_text(f"{int(slider.val)}/{n_slices(orient)}")
            slice_label = f"Slice {int(slider.val)}"
        im1.set_data(s1)
        im2.set_data(s2)
        mse, mae, ssim_val, ssim_map = compute_slice_metrics(s1, s2)
        im3.set_data(ssim_map)
        metrics_text.set_text(
            f"{slice_label} — MSE: {mse:.4f} | MAE: {mae:.4f} | SSIM: {ssim_val:.4f}"
        )
        fig.canvas.draw_idle()

    def update(val: float) -> None:
        if not mip_mode[0]:
            redraw()

    def on_orient(label: str) -> None:
        orientation[0] = label
        new_max = n_slices(label)
        slider.valmax = new_max
        slider.ax.set_xlim(slider.valmin, new_max)
        new_val = min(int(slider.val), new_max)
        slider.set_val(new_val)
        redraw()

    def on_mode(label: str) -> None:
        mip_mode[0] = label == "MIP"
        slider.ax.set_visible(not mip_mode[0])
        redraw()

    radio_mode.on_clicked(on_mode)
    radio.on_clicked(on_orient)
    slider.on_changed(update)

    fig.suptitle("SPECT Reconstruction Comparison", fontsize=14, y=0.99)
    plt.show()


if __name__ == "__main__":
    main()

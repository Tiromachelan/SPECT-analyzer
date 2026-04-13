"""
SPECT Scan Reconstruction Viewer
Displays two .raw SPECT reconstructions side by side with a slider
to navigate through the 64 slices (128x128 pixels, 32-bit float).
A third panel shows the per-slice SSIM heatmap; MSE, MAE, and SSIM
are displayed both per-slice and as whole-volume aggregates.
File selection is done via in-window buttons; CLI args bypass the dialogs.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
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


def main() -> None:
    # --- normalise helper ----------------------------------------------------
    def norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi != lo else v - lo

    # --- mutable data state --------------------------------------------------
    vols: list[np.ndarray | None] = [None, None]
    paths: list[str | None] = [None, None]

    # --- initialise tkinter root before matplotlib ---------------------------
    # A single persistent hidden root avoids event-loop conflicts when opening
    # file dialogs from inside a matplotlib callback (macOS in particular).
    _tk_root = Tk()
    _tk_root.withdraw()

    # --- CLI fast path: pre-load before building UI --------------------------
    if len(sys.argv) == 3:
        for i, p in enumerate(sys.argv[1:3]):
            try:
                vols[i] = norm(load_raw(p))
                paths[i] = p
            except (ValueError, FileNotFoundError) as e:
                sys.exit(f"Error loading '{p}': {e}")

    # --- orientation helpers -------------------------------------------------
    ORIENTATIONS = ("Axial", "Coronal", "Sagittal")
    orientation: list[str] = ["Axial"]
    mip_mode: list[bool] = [False]

    def get_slice(vol: np.ndarray, idx: int, orient: str) -> np.ndarray:
        """Return a 2-D cross-section for the given orientation and 0-based index."""
        if orient == "Axial":
            return vol[idx]
        elif orient == "Coronal":
            return vol[:, idx, :]
        else:
            return vol[:, :, idx]

    def get_mip(vol: np.ndarray, orient: str) -> np.ndarray:
        """Return a maximum-intensity projection along the depth axis."""
        if orient == "Axial":
            return vol.max(axis=0)
        elif orient == "Coronal":
            return vol.max(axis=1)
        else:
            return vol.max(axis=2)

    def n_slices(orient: str) -> int:
        return 64 if orient == "Axial" else 128

    def slice_shape(orient: str) -> tuple[int, int]:
        """Return (rows, cols) of one cross-section in the given orientation."""
        if orient == "Axial":
            return (128, 128)
        elif orient == "Coronal":
            return (64, 128)
        else:
            return (64, 128)

    def make_extent(orient: str) -> list[float]:
        rows, cols = slice_shape(orient)
        return [-0.5, cols - 0.5, rows - 0.5, -0.5]

    # --- set up figure -------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(left=0.09, bottom=0.18, right=0.97, top=0.82, wspace=0.32)

    blank = np.zeros(slice_shape("Axial"))
    im1 = ax1.imshow(
        blank, cmap="gray", vmin=0, vmax=1, aspect="equal",
        extent=make_extent("Axial"),
    )
    im2 = ax2.imshow(
        blank, cmap="gray", vmin=0, vmax=1, aspect="equal",
        extent=make_extent("Axial"),
    )
    im3 = ax3.imshow(
        blank, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal",
        extent=make_extent("Axial"),
    )
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    def _label(idx: int, fallback: str) -> str:
        return paths[idx].rsplit("/", 1)[-1] if paths[idx] else fallback

    ax1.set_title(_label(0, "File 1"))
    ax2.set_title(_label(1, "File 2"))
    ax3.set_title("SSIM Heatmap")
    for ax in (ax1, ax2, ax3):
        ax.set_axis_off()

    # --- placeholder text ----------------------------------------------------
    _ph_kw = dict(ha="center", va="center", fontsize=11, color="gray")
    ph1 = ax1.text(0.5, 0.5, "No file loaded\nClick 'Choose File' below",
                   transform=ax1.transAxes, **_ph_kw)
    ph2 = ax2.text(0.5, 0.5, "No file loaded\nClick 'Choose File' below",
                   transform=ax2.transAxes, **_ph_kw)
    ph3 = ax3.text(0.5, 0.5, "Load both files\nto compare",
                   transform=ax3.transAxes, **_ph_kw)
    placeholders = [ph1, ph2, ph3]
    for i in range(2):
        if vols[i] is not None:
            placeholders[i].set_visible(False)

    # --- metrics text --------------------------------------------------------
    vol_text = fig.text(
        0.5, 0.88, "",
        ha="center", va="top", fontsize=9, color="dimgray",
    )
    metrics_text = fig.text(
        0.5, 0.94, "",
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

    # --- colormap radio buttons ----------------------------------------------
    CMAPS = ("gray", "hot", "viridis", "plasma", "bone")
    colormap: list[str] = ["gray"]
    ax_cmap = fig.add_axes([0.01, 0.18, 0.07, 0.13])
    ax_cmap.set_title("Cmap", fontsize=9)
    radio_cmap = RadioButtons(ax_cmap, CMAPS, active=0)
    for lbl in radio_cmap.labels:
        lbl.set_fontsize(9)

    # --- slider --------------------------------------------------------------
    ax_slider = fig.add_axes([0.15, 0.06, 0.70, 0.04])
    slider = Slider(
        ax_slider, "Slice", 1, 64, valinit=1, valstep=1,
    )
    slider.valtext.set_text(f"1/{n_slices('Axial')}")

    # --- Choose / Change File buttons ----------------------------------------
    ax_btn1 = fig.add_axes([0.13, 0.12, 0.16, 0.04])
    ax_btn2 = fig.add_axes([0.45, 0.12, 0.16, 0.04])
    btn1 = Button(ax_btn1, "Change File" if vols[0] is not None else "Choose File")
    btn2 = Button(ax_btn2, "Change File" if vols[1] is not None else "Choose File")
    btns = [btn1, btn2]
    axes_pair = [ax1, ax2]

    # --- inner callbacks -----------------------------------------------------
    def open_file(idx: int) -> None:
        titles = ("Select FIRST .raw file", "Select SECOND .raw file")
        _tk_root.lift()
        path = filedialog.askopenfilename(
            parent=_tk_root,
            title=titles[idx],
            filetypes=[("Raw files", "*.raw"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            vol = norm(load_raw(path))
        except ValueError as e:
            print(f"Error loading file: {e}", file=sys.stderr)
            return
        vols[idx] = vol
        paths[idx] = path
        axes_pair[idx].set_title(path.rsplit("/", 1)[-1])
        placeholders[idx].set_visible(False)
        btns[idx].label.set_text("Change File")
        refresh_state()

    def refresh_state() -> None:
        both = vols[0] is not None and vols[1] is not None
        if both:
            v_mse, v_mae, v_ssim = compute_volume_metrics(vols[0], vols[1])
            vol_text.set_text(
                f"Volume — MSE: {v_mse:.4f} | MAE: {v_mae:.4f} | SSIM: {v_ssim:.4f}"
            )
            placeholders[2].set_visible(False)
        else:
            vol_text.set_text("")
            metrics_text.set_text("")
            placeholders[2].set_visible(True)
        redraw()

    def redraw() -> None:
        orient = orientation[0]
        extent = make_extent(orient)
        for im, ax in ((im1, ax1), (im2, ax2), (im3, ax3)):
            im.set_extent(extent)
            ax.set_aspect("equal")
        if mip_mode[0]:
            s = [get_mip(v, orient) if v is not None else None for v in vols]
            slider.valtext.set_text("MIP")
            slice_label = "MIP"
        else:
            sidx = int(slider.val) - 1
            s = [get_slice(v, sidx, orient) if v is not None else None for v in vols]
            slider.valtext.set_text(f"{int(slider.val)}/{n_slices(orient)}")
            slice_label = f"Slice {int(slider.val)}"
        if s[0] is not None:
            im1.set_data(s[0])
        if s[1] is not None:
            im2.set_data(s[1])
        if s[0] is not None and s[1] is not None:
            mse, mae, ssim_val, ssim_map = compute_slice_metrics(s[0], s[1])
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

    def on_cmap(label: str) -> None:
        colormap[0] = label
        im1.set_cmap(label)
        im2.set_cmap(label)
        fig.canvas.draw_idle()

    btn1.on_clicked(lambda _: open_file(0))
    btn2.on_clicked(lambda _: open_file(1))
    radio_mode.on_clicked(on_mode)
    radio.on_clicked(on_orient)
    radio_cmap.on_clicked(on_cmap)
    slider.on_changed(update)

    # Populate display if CLI pre-loaded files
    if any(v is not None for v in vols):
        refresh_state()

    fig.suptitle("SPECT Reconstruction Comparison", fontsize=14, y=0.99)
    plt.show()


if __name__ == "__main__":
    main()

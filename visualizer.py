"""
SPECT Scan Reconstruction Viewer
Displays two .raw SPECT reconstructions side by side.  A third panel shows
the per-slice SSIM heatmap; MSE, MAE, and SSIM are displayed both per-slice
and as whole-volume aggregates.  Built with PySide6 and an embedded matplotlib
canvas.
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from skimage.metrics import structural_similarity
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QComboBox,
    QButtonGroup,
    QFileDialog,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# --- pure metric functions ---------------------------------------------------

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
    ssim_sum = 0.0
    for i in range(v1.shape[0]):
        ssim_sum += structural_similarity(
            v1[i].astype(np.float64),
            v2[i].astype(np.float64),
            data_range=1.0,
        )
    return mse, mae, ssim_sum / v1.shape[0]


def load_raw(path: str) -> np.ndarray:
    """Load a .raw file as a (64, 128, 128) float32 array."""
    data = np.fromfile(path, dtype=np.float32)
    expected = 128 * 128 * 64
    if data.size != expected:
        raise ValueError(f"Expected {expected} floats in '{path}', got {data.size}")
    return data.reshape((64, 128, 128))


# --- orientation helpers (module-level so MainWindow can use them) -----------

def _norm(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi != lo else v - lo


def _get_slice(vol: np.ndarray, idx: int, orient: str) -> np.ndarray:
    if orient == "Axial":
        return vol[idx]
    elif orient == "Coronal":
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def _get_mip(vol: np.ndarray, orient: str) -> np.ndarray:
    if orient == "Axial":
        return vol.max(axis=0)
    elif orient == "Coronal":
        return vol.max(axis=1)
    else:
        return vol.max(axis=2)


def _n_slices(orient: str) -> int:
    return 64 if orient == "Axial" else 128


def _slice_shape(orient: str) -> tuple[int, int]:
    return (128, 128) if orient == "Axial" else (64, 128)


def _make_extent(orient: str) -> list[float]:
    rows, cols = _slice_shape(orient)
    return [-0.5, cols - 0.5, rows - 0.5, -0.5]


# --- main window -------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self, preload_paths: list[str | None]) -> None:
        super().__init__()
        self.setWindowTitle("SPECT Reconstruction Comparison")

        self._vols: list[np.ndarray | None] = [None, None]
        self._paths: list[str | None] = [None, None]
        self._orientation: str = "Axial"
        self._mip_mode: bool = False
        self._colormap: str = "gray"

        self._build_ui()

        for i, p in enumerate(preload_paths):
            if p:
                self._load_file(i, p)

    # --- UI construction -----------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        # per-slice metrics (bold, updated on each redraw)
        self._metrics_label = QLabel("")
        bold = QFont()
        bold.setBold(True)
        self._metrics_label.setFont(bold)
        self._metrics_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._metrics_label)

        # volume metrics (dimgray, set once when both files are loaded)
        self._vol_label = QLabel("")
        small = QFont()
        small.setPointSize(9)
        self._vol_label.setFont(small)
        self._vol_label.setStyleSheet("color: dimgray;")
        self._vol_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._vol_label)

        body = QHBoxLayout()
        body.setSpacing(8)
        root.addLayout(body)

        body.addWidget(self._build_controls())
        body.addWidget(self._build_canvas(), stretch=1)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(175)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # file selection buttons
        self._file_btns: list[QPushButton] = []
        self._fname_labels: list[QLabel] = []
        for i in range(2):
            grp = QGroupBox(f"File {i + 1}")
            glay = QVBoxLayout(grp)
            btn = QPushButton("Choose File")
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: gray; font-size: 8pt;")
            glay.addWidget(btn)
            glay.addWidget(lbl)
            layout.addWidget(grp)
            self._file_btns.append(btn)
            self._fname_labels.append(lbl)
        self._file_btns[0].clicked.connect(lambda: self._pick_file(0))
        self._file_btns[1].clicked.connect(lambda: self._pick_file(1))

        # mode (Slice / MIP)
        grp_mode = QGroupBox("Mode")
        mode_lay = QVBoxLayout(grp_mode)
        self._rb_slice = QRadioButton("Slice")
        self._rb_mip = QRadioButton("MIP")
        self._rb_slice.setChecked(True)
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self._rb_slice)
        self._mode_grp.addButton(self._rb_mip)
        mode_lay.addWidget(self._rb_slice)
        mode_lay.addWidget(self._rb_mip)
        layout.addWidget(grp_mode)
        self._rb_slice.toggled.connect(
            lambda checked: self._on_mode_changed() if checked else None
        )
        self._rb_mip.toggled.connect(
            lambda checked: self._on_mode_changed() if checked else None
        )

        # view orientation
        grp_view = QGroupBox("View")
        view_lay = QVBoxLayout(grp_view)
        self._view_btns: dict[str, QRadioButton] = {}
        self._view_grp = QButtonGroup(self)
        for label in ("Axial", "Coronal", "Sagittal"):
            rb = QRadioButton(label)
            if label == "Axial":
                rb.setChecked(True)
            self._view_grp.addButton(rb)
            view_lay.addWidget(rb)
            self._view_btns[label] = rb
            rb.toggled.connect(
                lambda checked, l=label: self._on_orient_changed(l) if checked else None
            )
        layout.addWidget(grp_view)

        # colormap
        grp_cmap = QGroupBox("Colormap")
        cmap_lay = QVBoxLayout(grp_cmap)
        self._cmap_combo = QComboBox()
        for cm in ("gray", "hot", "viridis", "plasma", "bone"):
            self._cmap_combo.addItem(cm)
        cmap_lay.addWidget(self._cmap_combo)
        layout.addWidget(grp_cmap)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)

        # slice slider
        grp_slice = QGroupBox("Slice")
        slice_lay = QVBoxLayout(grp_slice)
        self._slice_slider = QSlider(Qt.Orientation.Horizontal)
        self._slice_slider.setMinimum(1)
        self._slice_slider.setMaximum(64)
        self._slice_slider.setValue(1)
        self._slice_val_label = QLabel("1 / 64")
        self._slice_val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slice_lay.addWidget(self._slice_slider)
        slice_lay.addWidget(self._slice_val_label)
        layout.addWidget(grp_slice)
        self._slice_slider.valueChanged.connect(self._on_slice_changed)

        layout.addStretch()
        return panel

    def _build_canvas(self) -> FigureCanvasQTAgg:
        blank = np.zeros(_slice_shape("Axial"))
        extent = _make_extent("Axial")

        self._fig, (self._ax1, self._ax2, self._ax3) = plt.subplots(
            1, 3, figsize=(12, 4)
        )
        self._fig.subplots_adjust(
            left=0.02, right=0.97, top=0.88, bottom=0.04, wspace=0.28
        )

        self._im1 = self._ax1.imshow(
            blank, cmap="gray", vmin=0, vmax=1, aspect="equal", extent=extent
        )
        self._im2 = self._ax2.imshow(
            blank, cmap="gray", vmin=0, vmax=1, aspect="equal", extent=extent
        )
        self._im3 = self._ax3.imshow(
            blank, cmap="RdYlGn", vmin=0, vmax=1, aspect="equal", extent=extent
        )
        self._fig.colorbar(self._im3, ax=self._ax3, fraction=0.046, pad=0.04)

        self._ax1.set_title("File 1")
        self._ax2.set_title("File 2")
        self._ax3.set_title("SSIM Heatmap")
        for ax in (self._ax1, self._ax2, self._ax3):
            ax.set_axis_off()

        _ph_kw = dict(ha="center", va="center", fontsize=11, color="gray")
        self._placeholders = [
            self._ax1.text(
                0.5, 0.5, "No file loaded\nClick 'Choose File'",
                transform=self._ax1.transAxes, **_ph_kw,
            ),
            self._ax2.text(
                0.5, 0.5, "No file loaded\nClick 'Choose File'",
                transform=self._ax2.transAxes, **_ph_kw,
            ),
            self._ax3.text(
                0.5, 0.5, "Load both files\nto compare",
                transform=self._ax3.transAxes, **_ph_kw,
            ),
        ]

        canvas = FigureCanvasQTAgg(self._fig)
        canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        return canvas

    # --- file loading --------------------------------------------------------

    def _pick_file(self, idx: int) -> None:
        title = f"Select {'first' if idx == 0 else 'second'} .raw file"
        path, _ = QFileDialog.getOpenFileName(
            self, title, "", "Raw files (*.raw);;All files (*.*)"
        )
        if path:
            self._load_file(idx, path)

    def _load_file(self, idx: int, path: str) -> None:
        try:
            vol = _norm(load_raw(path))
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading '{path}': {e}", file=sys.stderr)
            return
        self._vols[idx] = vol
        self._paths[idx] = path
        fname = path.rsplit("/", 1)[-1]
        [self._ax1, self._ax2][idx].set_title(fname)
        self._fname_labels[idx].setText(fname)
        self._file_btns[idx].setText("Change File")
        self._placeholders[idx].set_visible(False)
        self._refresh_state()

    # --- state / redraw ------------------------------------------------------

    def _refresh_state(self) -> None:
        both = self._vols[0] is not None and self._vols[1] is not None
        if both:
            v_mse, v_mae, v_ssim = compute_volume_metrics(
                self._vols[0], self._vols[1]
            )
            self._vol_label.setText(
                f"Volume — MSE: {v_mse:.4f} | MAE: {v_mae:.4f} | SSIM: {v_ssim:.4f}"
            )
            self._placeholders[2].set_visible(False)
        else:
            self._vol_label.setText("")
            self._metrics_label.setText("")
            self._placeholders[2].set_visible(True)
        self._redraw()

    def _redraw(self) -> None:
        orient = self._orientation
        extent = _make_extent(orient)
        for im, ax in (
            (self._im1, self._ax1),
            (self._im2, self._ax2),
            (self._im3, self._ax3),
        ):
            im.set_extent(extent)
            ax.set_aspect("equal")

        if self._mip_mode:
            s = [
                _get_mip(v, orient) if v is not None else None
                for v in self._vols
            ]
            slice_label = "MIP"
        else:
            sidx = self._slice_slider.value() - 1
            s = [
                _get_slice(v, sidx, orient) if v is not None else None
                for v in self._vols
            ]
            slice_label = f"Slice {self._slice_slider.value()}"

        if s[0] is not None:
            self._im1.set_data(s[0])
        if s[1] is not None:
            self._im2.set_data(s[1])
        if s[0] is not None and s[1] is not None:
            mse, mae, ssim_val, ssim_map = compute_slice_metrics(s[0], s[1])
            self._im3.set_data(ssim_map)
            self._metrics_label.setText(
                f"{slice_label} — MSE: {mse:.4f} | MAE: {mae:.4f} | SSIM: {ssim_val:.4f}"
            )
        self._fig.canvas.draw_idle()

    # --- Qt signal handlers --------------------------------------------------

    def _on_mode_changed(self) -> None:
        self._mip_mode = self._rb_mip.isChecked()
        self._slice_slider.setEnabled(not self._mip_mode)
        if self._mip_mode:
            self._slice_val_label.setText("MIP")
        else:
            v = self._slice_slider.value()
            self._slice_val_label.setText(f"{v} / {_n_slices(self._orientation)}")
        self._redraw()

    def _on_orient_changed(self, label: str) -> None:
        self._orientation = label
        new_max = _n_slices(label)
        self._slice_slider.setMaximum(new_max)
        if self._slice_slider.value() > new_max:
            self._slice_slider.setValue(new_max)
        self._slice_val_label.setText(
            f"{self._slice_slider.value()} / {new_max}"
        )
        self._redraw()

    def _on_slice_changed(self, val: int) -> None:
        self._slice_val_label.setText(f"{val} / {_n_slices(self._orientation)}")
        if not self._mip_mode:
            self._redraw()

    def _on_cmap_changed(self, name: str) -> None:
        self._colormap = name
        self._im1.set_cmap(name)
        self._im2.set_cmap(name)
        self._fig.canvas.draw_idle()


# --- entry point -------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)

    preload: list[str | None] = [None, None]
    if len(sys.argv) == 3:
        preload = [sys.argv[1], sys.argv[2]]

    window = MainWindow(preload)
    window.resize(1200, 580)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

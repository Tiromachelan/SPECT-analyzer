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
from matplotlib.colors import LinearSegmentedColormap
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


# --- custom colormaps --------------------------------------------------------

_hot_metal_blue = LinearSegmentedColormap.from_list(
    "hot_metal_blue",
    [
    (0.000000, 0.000000, 0.000000), (0.000000, 0.000000, 0.007843), (0.000000, 0.000000, 0.015686), (0.000000, 0.000000, 0.023529),
    (0.000000, 0.000000, 0.031373), (0.000000, 0.000000, 0.039216), (0.000000, 0.000000, 0.047059), (0.000000, 0.000000, 0.054902),
    (0.000000, 0.000000, 0.062745), (0.000000, 0.000000, 0.066667), (0.000000, 0.000000, 0.074510), (0.000000, 0.000000, 0.082353),
    (0.000000, 0.000000, 0.090196), (0.000000, 0.000000, 0.098039), (0.000000, 0.000000, 0.105882), (0.000000, 0.000000, 0.113725),
    (0.000000, 0.000000, 0.121569), (0.000000, 0.000000, 0.129412), (0.000000, 0.000000, 0.137255), (0.000000, 0.000000, 0.145098),
    (0.000000, 0.000000, 0.152941), (0.000000, 0.000000, 0.160784), (0.000000, 0.000000, 0.168627), (0.000000, 0.000000, 0.176471),
    (0.000000, 0.000000, 0.184314), (0.000000, 0.000000, 0.192157), (0.000000, 0.000000, 0.200000), (0.000000, 0.000000, 0.207843),
    (0.000000, 0.000000, 0.215686), (0.000000, 0.000000, 0.223529), (0.000000, 0.000000, 0.231373), (0.000000, 0.000000, 0.239216),
    (0.000000, 0.000000, 0.247059), (0.000000, 0.000000, 0.254902), (0.000000, 0.000000, 0.262745), (0.000000, 0.000000, 0.270588),
    (0.000000, 0.000000, 0.278431), (0.000000, 0.000000, 0.286275), (0.000000, 0.000000, 0.294118), (0.000000, 0.000000, 0.301961),
    (0.000000, 0.000000, 0.309804), (0.000000, 0.000000, 0.317647), (0.000000, 0.000000, 0.325490), (0.000000, 0.000000, 0.329412),
    (0.000000, 0.000000, 0.337255), (0.000000, 0.000000, 0.345098), (0.000000, 0.000000, 0.352941), (0.000000, 0.000000, 0.360784),
    (0.000000, 0.000000, 0.368627), (0.000000, 0.000000, 0.376471), (0.000000, 0.000000, 0.384314), (0.000000, 0.000000, 0.392157),
    (0.000000, 0.000000, 0.400000), (0.000000, 0.000000, 0.407843), (0.000000, 0.000000, 0.415686), (0.000000, 0.000000, 0.423529),
    (0.000000, 0.000000, 0.431373), (0.000000, 0.000000, 0.439216), (0.000000, 0.000000, 0.447059), (0.000000, 0.000000, 0.454902),
    (0.000000, 0.000000, 0.458824), (0.000000, 0.000000, 0.466667), (0.000000, 0.000000, 0.474510), (0.000000, 0.000000, 0.482353),
    (0.000000, 0.000000, 0.490196), (0.000000, 0.000000, 0.498039), (0.000000, 0.000000, 0.505882), (0.000000, 0.000000, 0.513725),
    (0.000000, 0.000000, 0.521569), (0.000000, 0.000000, 0.529412), (0.000000, 0.000000, 0.537255), (0.000000, 0.000000, 0.545098),
    (0.000000, 0.000000, 0.552941), (0.000000, 0.000000, 0.560784), (0.000000, 0.000000, 0.568627), (0.000000, 0.000000, 0.576471),
    (0.000000, 0.000000, 0.584314), (0.000000, 0.000000, 0.592157), (0.000000, 0.000000, 0.600000), (0.000000, 0.000000, 0.607843),
    (0.000000, 0.000000, 0.615686), (0.000000, 0.000000, 0.623529), (0.000000, 0.000000, 0.631373), (0.000000, 0.000000, 0.639216),
    (0.000000, 0.000000, 0.647059), (0.000000, 0.000000, 0.654902), (0.011765, 0.000000, 0.662745), (0.023529, 0.000000, 0.670588),
    (0.035294, 0.000000, 0.678431), (0.047059, 0.000000, 0.686275), (0.058824, 0.000000, 0.694118), (0.070588, 0.000000, 0.701961),
    (0.082353, 0.000000, 0.709804), (0.094118, 0.000000, 0.717647), (0.101961, 0.000000, 0.721569), (0.113725, 0.000000, 0.729412),
    (0.125490, 0.000000, 0.737255), (0.137255, 0.000000, 0.745098), (0.149020, 0.000000, 0.752941), (0.160784, 0.000000, 0.760784),
    (0.172549, 0.000000, 0.768627), (0.184314, 0.000000, 0.776471), (0.196078, 0.000000, 0.784314), (0.203922, 0.000000, 0.772549),
    (0.215686, 0.000000, 0.760784), (0.223529, 0.000000, 0.749020), (0.231373, 0.000000, 0.737255), (0.243137, 0.000000, 0.725490),
    (0.250980, 0.000000, 0.713725), (0.258824, 0.000000, 0.701961), (0.270588, 0.000000, 0.690196), (0.278431, 0.000000, 0.682353),
    (0.290196, 0.000000, 0.670588), (0.298039, 0.000000, 0.658824), (0.305882, 0.000000, 0.647059), (0.317647, 0.000000, 0.635294),
    (0.325490, 0.000000, 0.623529), (0.333333, 0.000000, 0.611765), (0.345098, 0.000000, 0.600000), (0.352941, 0.000000, 0.588235),
    (0.364706, 0.007843, 0.564706), (0.376471, 0.015686, 0.541176), (0.388235, 0.023529, 0.517647), (0.400000, 0.031373, 0.494118),
    (0.411765, 0.035294, 0.474510), (0.423529, 0.043137, 0.450980), (0.435294, 0.050980, 0.427451), (0.447059, 0.058824, 0.403922),
    (0.454902, 0.066667, 0.380392), (0.466667, 0.074510, 0.356863), (0.478431, 0.082353, 0.333333), (0.490196, 0.090196, 0.309804),
    (0.501961, 0.094118, 0.290196), (0.513725, 0.101961, 0.266667), (0.525490, 0.109804, 0.243137), (0.537255, 0.117647, 0.219608),
    (0.549020, 0.125490, 0.196078), (0.560784, 0.133333, 0.184314), (0.572549, 0.141176, 0.172549), (0.584314, 0.149020, 0.160784),
    (0.596078, 0.156863, 0.149020), (0.607843, 0.160784, 0.137255), (0.619608, 0.168627, 0.125490), (0.631373, 0.176471, 0.113725),
    (0.643137, 0.184314, 0.101961), (0.650980, 0.192157, 0.094118), (0.662745, 0.200000, 0.082353), (0.674510, 0.207843, 0.070588),
    (0.686275, 0.215686, 0.058824), (0.698039, 0.219608, 0.047059), (0.709804, 0.227451, 0.035294), (0.721569, 0.235294, 0.023529),
    (0.733333, 0.243137, 0.011765), (0.745098, 0.250980, 0.000000), (0.760784, 0.258824, 0.000000), (0.776471, 0.266667, 0.000000),
    (0.788235, 0.274510, 0.000000), (0.803922, 0.282353, 0.000000), (0.819608, 0.286275, 0.000000), (0.835294, 0.294118, 0.000000),
    (0.850980, 0.301961, 0.000000), (0.866667, 0.309804, 0.000000), (0.878431, 0.317647, 0.000000), (0.894118, 0.325490, 0.000000),
    (0.909804, 0.333333, 0.000000), (0.925490, 0.341176, 0.000000), (0.941176, 0.345098, 0.000000), (0.956863, 0.352941, 0.000000),
    (0.968627, 0.360784, 0.000000), (0.984314, 0.368627, 0.000000), (1.000000, 0.376471, 0.000000), (1.000000, 0.384314, 0.011765),
    (1.000000, 0.392157, 0.023529), (1.000000, 0.400000, 0.035294), (1.000000, 0.407843, 0.047059), (1.000000, 0.411765, 0.058824),
    (1.000000, 0.419608, 0.070588), (1.000000, 0.427451, 0.082353), (1.000000, 0.435294, 0.094118), (1.000000, 0.443137, 0.101961),
    (1.000000, 0.450980, 0.113725), (1.000000, 0.458824, 0.125490), (1.000000, 0.466667, 0.137255), (1.000000, 0.470588, 0.149020),
    (1.000000, 0.478431, 0.160784), (1.000000, 0.486275, 0.172549), (1.000000, 0.494118, 0.184314), (1.000000, 0.501961, 0.196078),
    (1.000000, 0.509804, 0.207843), (1.000000, 0.517647, 0.219608), (1.000000, 0.525490, 0.231373), (1.000000, 0.533333, 0.243137),
    (1.000000, 0.537255, 0.254902), (1.000000, 0.545098, 0.266667), (1.000000, 0.552941, 0.278431), (1.000000, 0.560784, 0.290196),
    (1.000000, 0.568627, 0.298039), (1.000000, 0.576471, 0.309804), (1.000000, 0.584314, 0.321569), (1.000000, 0.592157, 0.333333),
    (1.000000, 0.596078, 0.345098), (1.000000, 0.603922, 0.356863), (1.000000, 0.611765, 0.368627), (1.000000, 0.619608, 0.380392),
    (1.000000, 0.627451, 0.392157), (1.000000, 0.635294, 0.403922), (1.000000, 0.643137, 0.415686), (1.000000, 0.650980, 0.427451),
    (1.000000, 0.658824, 0.439216), (1.000000, 0.662745, 0.450980), (1.000000, 0.670588, 0.462745), (1.000000, 0.678431, 0.474510),
    (1.000000, 0.686275, 0.486275), (1.000000, 0.694118, 0.494118), (1.000000, 0.701961, 0.505882), (1.000000, 0.709804, 0.517647),
    (1.000000, 0.717647, 0.529412), (1.000000, 0.721569, 0.541176), (1.000000, 0.729412, 0.552941), (1.000000, 0.737255, 0.564706),
    (1.000000, 0.745098, 0.576471), (1.000000, 0.752941, 0.588235), (1.000000, 0.760784, 0.600000), (1.000000, 0.768627, 0.611765),
    (1.000000, 0.776471, 0.623529), (1.000000, 0.784314, 0.635294), (1.000000, 0.788235, 0.647059), (1.000000, 0.796078, 0.658824),
    (1.000000, 0.803922, 0.670588), (1.000000, 0.811765, 0.682353), (1.000000, 0.819608, 0.690196), (1.000000, 0.827451, 0.701961),
    (1.000000, 0.835294, 0.713725), (1.000000, 0.843137, 0.725490), (1.000000, 0.847059, 0.737255), (1.000000, 0.854902, 0.749020),
    (1.000000, 0.862745, 0.760784), (1.000000, 0.870588, 0.772549), (1.000000, 0.878431, 0.784314), (1.000000, 0.886275, 0.796078),
    (1.000000, 0.894118, 0.807843), (1.000000, 0.898039, 0.823529), (1.000000, 0.905882, 0.835294), (1.000000, 0.913725, 0.847059),
    (1.000000, 0.921569, 0.858824), (1.000000, 0.929412, 0.874510), (1.000000, 0.937255, 0.886275), (1.000000, 0.941176, 0.898039),
    (1.000000, 0.949020, 0.909804), (1.000000, 0.956863, 0.925490), (1.000000, 0.964706, 0.937255), (1.000000, 0.972549, 0.949020),
    (1.000000, 0.980392, 0.960784), (1.000000, 0.984314, 0.976471), (1.000000, 0.992157, 0.988235), (1.000000, 1.000000, 1.000000),
    ],
)
plt.colormaps.register(_hot_metal_blue)

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
        self._colormap: str = "hot_metal_blue"

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
        for cm in ("hot_metal_blue", "gray", "hot", "afmhot", "viridis", "plasma"):
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
            blank, cmap="hot_metal_blue", vmin=0, vmax=1, aspect="equal", extent=extent
        )
        self._im2 = self._ax2.imshow(
            blank, cmap="hot_metal_blue", vmin=0, vmax=1, aspect="equal", extent=extent
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

"""
SPECT Scan Reconstruction Viewer
Displays two .raw SPECT reconstructions side by side.  A third panel shows
the per-slice SSIM heatmap; MSE, MAE, and SSIM are displayed both per-slice
and as whole-volume aggregates.  Built with PySide6 and an embedded matplotlib
canvas.
"""

import sys
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import LinearSegmentedColormap
from skimage.metrics import structural_similarity
from skimage.filters import threshold_otsu
from skimage.morphology import binary_erosion, square
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
    QCheckBox,
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

def compute_mask_threshold(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return an Otsu threshold on max(v1, v2), clipped to [0.02, 0.50]."""
    joint = np.maximum(v1, v2).ravel()
    try:
        t = float(threshold_otsu(joint))
    except Exception:
        t = 0.10
    return float(np.clip(t, 0.02, 0.50))


def compute_slice_metrics(
    s1: np.ndarray,
    s2: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float, np.ndarray]:
    """Return (MSE, MAE, SSIM_scalar, SSIM_map) for two 2-D slices in [0, 1].

    If *mask* is provided (True = background pixel to exclude), MSE and MAE are
    computed over foreground pixels only.  The SSIM scalar uses the eroded
    foreground (interior pixels whose full 7×7 window lies in the foreground) to
    avoid window-boundary contamination.  The returned SSIM map is a masked array
    with background pixels masked out.  Returns NaN scalars when no valid pixels
    exist.
    """
    s1f = s1.astype(np.float64)
    s2f = s2.astype(np.float64)

    if mask is not None:
        fg = ~mask
        if not fg.any():
            blank = np.ma.masked_all(s1.shape)
            return float("nan"), float("nan"), float("nan"), blank

        diff = (s1f - s2f)[fg]
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))

        _, ssim_map_full = structural_similarity(s1f, s2f, full=True, data_range=1.0)
        # erode foreground by SSIM window radius to get clean interior centers
        valid_centers = binary_erosion(fg, square(7))
        if valid_centers.any():
            ssim_val = float(ssim_map_full[valid_centers].mean())
        else:
            ssim_val = float("nan")

        ssim_map = np.ma.masked_where(mask, ssim_map_full)
        return mse, mae, ssim_val, ssim_map

    diff = s1f - s2f
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    ssim_val, ssim_map = structural_similarity(s1f, s2f, full=True, data_range=1.0)
    return mse, mae, float(ssim_val), ssim_map


def compute_volume_metrics(
    v1: np.ndarray,
    v2: np.ndarray,
    threshold: float | None = None,
) -> tuple[float, float, float]:
    """Return (MSE, MAE, SSIM) aggregated over the entire 3-D volume.

    When *threshold* is given, MSE/MAE use only foreground voxels and SSIM is
    weighted by each axial slice's valid-center count.
    """
    if threshold is not None:
        mask3d = np.maximum(v1, v2) < threshold
        fg3d = ~mask3d
        if not fg3d.any():
            return float("nan"), float("nan"), float("nan")
        diff = (v1.astype(np.float64) - v2.astype(np.float64))[fg3d]
        mse = float(np.mean(diff ** 2))
        mae = float(np.mean(np.abs(diff)))
        ssim_weighted = 0.0
        total_weight = 0
        for i in range(v1.shape[0]):
            fg2d = fg3d[i]
            if not fg2d.any():
                continue
            valid = binary_erosion(fg2d, square(7))
            n = int(valid.sum())
            if n == 0:
                continue
            _, sm = structural_similarity(
                v1[i].astype(np.float64), v2[i].astype(np.float64),
                data_range=1.0, full=True,
            )
            ssim_weighted += sm[valid].mean() * n
            total_weight += n
        ssim = ssim_weighted / total_weight if total_weight > 0 else float("nan")
        return mse, mae, ssim

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
        self._overlay_enabled: list[bool] = [False, False]
        self._mask_enabled: bool = False
        self._mask_threshold: float = 0.0
        self._auto_threshold: float = 0.0
        self._recon_model = None
        self._recon_model_path: str | None = None

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
        self._overlay_checks: list[QCheckBox] = []
        self._recon_btns: list[QPushButton] = []
        for i in range(2):
            grp = QGroupBox(f"File {i + 1}")
            glay = QVBoxLayout(grp)
            btn = QPushButton("Choose File")
            lbl = QLabel("—")
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: gray; font-size: 8pt;")
            overlay_cb = QCheckBox("SSIM Overlay")
            overlay_cb.setEnabled(False)
            recon_btn = QPushButton("Reconstruct…")
            glay.addWidget(lbl)
            glay.addWidget(btn)
            glay.addWidget(recon_btn)
            glay.addWidget(overlay_cb)
            layout.addWidget(grp)
            self._file_btns.append(btn)
            self._fname_labels.append(lbl)
            self._overlay_checks.append(overlay_cb)
            self._recon_btns.append(recon_btn)
        self._file_btns[0].clicked.connect(lambda: self._pick_file(0))
        self._file_btns[1].clicked.connect(lambda: self._pick_file(1))
        self._overlay_checks[0].toggled.connect(
            lambda checked: self._on_overlay_toggled(0, checked)
        )
        self._overlay_checks[1].toggled.connect(
            lambda checked: self._on_overlay_toggled(1, checked)
        )
        self._recon_btns[0].clicked.connect(lambda: self._on_reconstruct_clicked(0))
        self._recon_btns[1].clicked.connect(lambda: self._on_reconstruct_clicked(1))

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
        for cm in ("hot_metal_blue", "gray", "afmhot", "viridis", "plasma"):
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

        # mask controls
        grp_mask = QGroupBox("Mask")
        mask_lay = QVBoxLayout(grp_mask)
        self._mask_check = QCheckBox("Enable")
        self._mask_check.setEnabled(False)
        self._mask_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_slider.setMinimum(0)
        self._mask_slider.setMaximum(1000)
        self._mask_slider.setValue(0)
        self._mask_slider.setEnabled(False)
        self._mask_val_label = QLabel("0.000")
        self._mask_val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._mask_auto_btn = QPushButton("Auto")
        self._mask_auto_btn.setEnabled(False)
        mask_lay.addWidget(self._mask_check)
        mask_lay.addWidget(self._mask_slider)
        mask_lay.addWidget(self._mask_val_label)
        mask_lay.addWidget(self._mask_auto_btn)
        layout.addWidget(grp_mask)
        self._mask_check.toggled.connect(self._on_mask_toggled)
        self._mask_slider.valueChanged.connect(self._on_mask_slider_changed)
        self._mask_slider.sliderReleased.connect(self._on_mask_slider_released)
        self._mask_auto_btn.clicked.connect(self._on_mask_auto_clicked)

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
        self._im1_overlay = self._ax1.imshow(
            blank, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.5, aspect="equal",
            extent=extent, visible=False,
        )
        self._im2 = self._ax2.imshow(
            blank, cmap="hot_metal_blue", vmin=0, vmax=1, aspect="equal", extent=extent
        )
        self._im2_overlay = self._ax2.imshow(
            blank, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.5, aspect="equal",
            extent=extent, visible=False,
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
        label = Path(path).name
        self._paths[idx] = path
        self._load_vol(idx, vol, label)

    def _load_vol(self, idx: int, vol: np.ndarray, label: str) -> None:
        """Load a pre-computed normalised volume into panel *idx*."""
        self._vols[idx] = vol
        self._auto_threshold = 0.0
        [self._ax1, self._ax2][idx].set_title(label)
        self._fname_labels[idx].setText(label)
        self._file_btns[idx].setText("Change File")
        self._placeholders[idx].set_visible(False)
        self._refresh_state()

    # --- reconstruction ------------------------------------------------------

    @staticmethod
    def _find_default_pth() -> str | None:
        """Return the first .pth file next to visualizer.py, preferring model_*.pth."""
        here = Path(__file__).parent
        preferred = sorted(here.glob("model_*.pth"))
        if preferred:
            return str(preferred[0])
        fallback = sorted(here.glob("*.pth"))
        return str(fallback[0]) if fallback else None

    def _on_reconstruct_clicked(self, idx: int) -> None:
        import torch
        import model as _model

        # resolve model .pth
        pth_path = self._recon_model_path or self._find_default_pth()
        if not pth_path:
            pth_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select model checkpoint (.pth)",
                "",
                "PyTorch checkpoints (*.pth);;All files (*.*)",
            )
        if not pth_path:
            return

        # pick sinogram
        sino_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select sinogram for File {idx + 1}",
            "",
            "Raw files (*.raw);;All files (*.*)",
        )
        if not sino_path:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        all_btns = self._file_btns + self._recon_btns
        for btn in all_btns:
            btn.setEnabled(False)
        try:
            if self._recon_model is None or self._recon_model_path != pth_path:
                self._recon_model = _model.load_model(pth_path, device)
                self._recon_model_path = pth_path
            vol = _model.reconstruct_sinogram(self._recon_model, sino_path, device)
            vol = _norm(vol)
        except Exception as e:
            print(f"Reconstruction error: {e}", file=sys.stderr)
            return
        finally:
            for btn in all_btns:
                btn.setEnabled(True)

        label = f"[recon] {Path(sino_path).name}"
        self._paths[idx] = sino_path
        self._load_vol(idx, vol, label)

    # --- state / redraw ------------------------------------------------------

    def _refresh_state(self) -> None:
        both = self._vols[0] is not None and self._vols[1] is not None
        if both:
            if self._auto_threshold == 0.0:
                self._auto_threshold = compute_mask_threshold(
                    self._vols[0], self._vols[1]
                )
                self._mask_threshold = self._auto_threshold
                self._mask_slider.blockSignals(True)
                self._mask_slider.setValue(round(self._auto_threshold * 1000))
                self._mask_slider.blockSignals(False)
                self._mask_val_label.setText(f"{self._auto_threshold:.3f}")
            threshold = self._mask_threshold if self._mask_enabled else None
            v_mse, v_mae, v_ssim = compute_volume_metrics(
                self._vols[0], self._vols[1], threshold=threshold
            )
            ssim_str = f"{v_ssim:.4f}" if not math.isnan(v_ssim) else "N/A"
            mse_str = f"{v_mse:.4f}" if not math.isnan(v_mse) else "N/A"
            mae_str = f"{v_mae:.4f}" if not math.isnan(v_mae) else "N/A"
            self._vol_label.setText(
                f"Volume — MSE: {mse_str} | MAE: {mae_str} | SSIM: {ssim_str}"
            )
            self._placeholders[2].set_visible(False)
        else:
            self._vol_label.setText("")
            self._metrics_label.setText("")
            self._placeholders[2].set_visible(True)
        for cb in self._overlay_checks:
            cb.setEnabled(both)
        self._mask_check.setEnabled(both)
        self._mask_slider.setEnabled(both)
        self._mask_auto_btn.setEnabled(both)
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
        for ov in (self._im1_overlay, self._im2_overlay):
            ov.set_extent(extent)

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

        ssim_map: np.ndarray | None = None
        if s[0] is not None and s[1] is not None:
            mask_2d: np.ndarray | None = None
            if self._mask_enabled and self._mask_threshold > 0:
                mask_2d = np.maximum(s[0], s[1]) < self._mask_threshold
            mse, mae, ssim_val, ssim_map = compute_slice_metrics(s[0], s[1], mask=mask_2d)
            self._im3.set_data(ssim_map)
            ssim_str = f"{ssim_val:.4f}" if not math.isnan(ssim_val) else "N/A"
            mse_str = f"{mse:.4f}" if not math.isnan(mse) else "N/A"
            mae_str = f"{mae:.4f}" if not math.isnan(mae) else "N/A"
            self._metrics_label.setText(
                f"{slice_label} — MSE: {mse_str} | MAE: {mae_str} | SSIM: {ssim_str}"
            )

        for i, ov in enumerate((self._im1_overlay, self._im2_overlay)):
            if self._overlay_enabled[i] and ssim_map is not None:
                ov.set_data(ssim_map)
                ov.set_visible(True)
            else:
                ov.set_visible(False)

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

    def _on_overlay_toggled(self, idx: int, checked: bool) -> None:
        self._overlay_enabled[idx] = checked
        self._redraw()

    def _on_mask_toggled(self, checked: bool) -> None:
        self._mask_enabled = checked
        self._refresh_state()

    def _on_mask_slider_changed(self, val: int) -> None:
        self._mask_threshold = val / 1000.0
        self._mask_val_label.setText(f"{self._mask_threshold:.3f}")
        self._redraw()

    def _on_mask_slider_released(self) -> None:
        self._refresh_state()

    def _on_mask_auto_clicked(self) -> None:
        self._mask_threshold = self._auto_threshold
        self._mask_slider.blockSignals(True)
        self._mask_slider.setValue(round(self._auto_threshold * 1000))
        self._mask_slider.blockSignals(False)
        self._mask_val_label.setText(f"{self._auto_threshold:.3f}")
        self._refresh_state()


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

# Project Guidelines

## Overview

SPECT-analyzer is a single-script Python tool for side-by-side comparison of SPECT scan reconstructions. The main entry point is `visualizer.py`.

## Data Format

- Input files are `.raw` — flat binary arrays of **32-bit floats** (`np.float32`).
- Each file contains **64 slices** of **128×128 pixels** (total: 1,048,576 floats).
- Data is reshaped as `(64, 128, 128)` — slice-first ordering: `vol[z, y, x]`.
- Slices are presented to the user as **1-indexed** (1–64); convert to 0-indexed for array access.
- `.raw` files are gitignored; sample data lives in the repo root but is not committed.

## Architecture

The UI is built with **PySide6** (`QMainWindow`) with an embedded **matplotlib** canvas (`FigureCanvasQTAgg`). All Qt controls (buttons, sliders, combos, radio buttons) live in a fixed-width left panel; the 3-panel matplotlib figure fills the remaining space.

```
QMainWindow
└── central widget
    └── QVBoxLayout
        ├── QLabel  — per-slice metrics (bold)
        ├── QLabel  — volume metrics (dimgray)
        └── QHBoxLayout
            ├── controls panel (QWidget, 175 px)
            │   ├── QGroupBox "File 1/2"  → QPushButton + QLabel
            │   ├── QGroupBox "Mode"      → QRadioButton × 2
            │   ├── QGroupBox "View"      → QRadioButton × 3
            │   ├── QGroupBox "Colormap"  → QComboBox
            │   └── QGroupBox "Slice"     → QSlider + QLabel
            └── FigureCanvasQTAgg (3 matplotlib axes)
```

## Code Style

- Python 3.14, single-file architecture (`visualizer.py`).
- Type hints on all function signatures.
- Docstrings on public functions; `# --- label ---` section dividers.
- Module-level pure functions: `load_raw`, `compute_slice_metrics`, `compute_volume_metrics`, and `_`-prefixed orientation helpers.
- Application logic lives in `MainWindow(QMainWindow)`. Instance state uses normal attributes (no single-element list pattern — that was needed for closure-based callbacks only).

## Build and Run

The project uses **uv** for dependency management (`uv.lock` is committed).

```bash
# Install dependencies
uv sync

# Run — window opens immediately; use the in-window buttons to load files
uv run visualizer.py

# CLI fast path — loads both files immediately
uv run visualizer.py path/to/first.raw path/to/second.raw

# Or activate the venv manually
source .venv/bin/activate
python visualizer.py path/to/first.raw path/to/second.raw
```

## Project Conventions

- Volumes are independently normalised to `[0, 1]` before display — do not assume shared intensity scales across files.
- The slice slider controls both images simultaneously; left/right views stay locked to the same slice index.
- Default colormap is `"gray"`; the SSIM heatmap always uses `"RdYlGn"`.
- Filenames (extracted from paths) are used as subplot titles.

### Orientations and slice counts

| Orientation | Axis sliced      | Slice shape | Slider range |
|-------------|------------------|-------------|--------------|
| Axial       | `vol[idx]`       | 128×128     | 1–64         |
| Coronal     | `vol[:, idx, :]` | 64×128      | 1–128        |
| Sagittal    | `vol[:, :, idx]` | 64×128      | 1–128        |

When switching orientation, update `QSlider.setMaximum()` and the image `extent` via `_make_extent()` so non-square slices display with correct aspect ratio.

### MIP mode

When MIP is active, call `_get_mip(vol, orient)` instead of `_get_slice`; the slice slider is disabled (`setEnabled(False)`). Metrics and the SSIM heatmap are computed on the MIP projections.

### Metrics display

Two `QLabel` rows above the canvas:
- **Per-slice** (bold): `"Slice N — MSE: X | MAE: X | SSIM: X"` — updated on every `_redraw()` call. Shows `"MIP — …"` in MIP mode.
- **Whole-volume** (dimgray): computed once by `compute_volume_metrics()` (averages per-axial-slice SSIM) when both files are loaded; set via `_refresh_state()`.

### SSIM heatmap (3rd panel)

- `cmap="RdYlGn"`, `vmin=0, vmax=1` — red = low similarity, green = high.
- Computed per-slice/MIP via `compute_slice_metrics(s1, s2)` → `structural_similarity(..., full=True, data_range=1.0)`.
- Extent and aspect ratio update with orientation changes just like the grayscale panels.

### File loading flow

- `_pick_file(idx)` — opens `QFileDialog.getOpenFileName()`; calls `_load_file(idx, path)` on success.
- `_load_file(idx, path)` — loads + normalises the volume, updates panel title, filename label, button text, placeholder visibility, then calls `_refresh_state()`.
- `_refresh_state()` — if both vols are loaded, computes volume metrics and shows/hides the SSIM placeholder; always calls `_redraw()`.
- `_redraw()` — guards against `None` vols; updates `im1`/`im2`/`im3` data and extents, then calls `canvas.draw_idle()`.

## Key Dependencies

| Package      | Purpose                                           |
|--------------|---------------------------------------------------|
| numpy        | Binary file I/O and array ops                    |
| matplotlib   | Image display via embedded `FigureCanvasQTAgg`   |
| scikit-image | SSIM scalar and heatmap (`structural_similarity`) |
| PySide6      | Qt main window, widgets, file dialog             |


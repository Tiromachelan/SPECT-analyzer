# Project Guidelines

## Overview

SPECT-analyzer is a single-script Python tool for side-by-side comparison of SPECT scan reconstructions. The main entry point is `visualizer.py`.

## Data Format

- Input files are `.raw` — flat binary arrays of **32-bit floats** (`np.float32`).
- Each file contains **64 slices** of **128×128 pixels** (total: 1,048,576 floats).
- Data is reshaped as `(64, 128, 128)` — slice-first ordering: `vol[z, y, x]`.
- Slices are presented to the user as **1-indexed** (1–64); convert to 0-indexed for array access.
- `.raw` files are gitignored; sample data lives in the repo root but is not committed.

## Code Style

- Python 3.14, single-file architecture (`visualizer.py`).
- Type hints on function signatures (`-> np.ndarray`, `-> tuple[str, str]`).
- Docstrings on public functions; section comments with `# --- label ---` dividers.
- Visualization uses **matplotlib** with `matplotlib.widgets.Slider`, `RadioButtons`, and `Button` for interactivity.
- File selection via in-window `Button` widgets that open a **tkinter** `filedialog` on click.

## Build and Run

The project uses **uv** for dependency management (`uv.lock` is committed).

```bash
# Install dependencies
uv sync

# Run — window opens immediately; use the in-window buttons to load files
uv run visualizer.py

# CLI fast path — loads both files immediately, bypasses the dialogs
uv run visualizer.py path/to/first.raw path/to/second.raw

# Or activate the venv manually
source .venv/bin/activate
python visualizer.py path/to/first.raw path/to/second.raw
```

## Project Conventions

- Volumes are independently normalised to `[0, 1]` before display — do not assume shared intensity scales across files.
- The slider controls both images simultaneously; keep left/right views locked to the same slice index.
- Use grayscale colormap (`cmap="gray"`) for medical imaging display.
- Filenames (extracted from paths) are used as subplot titles.

### Orientations and slice counts

| Orientation | Axis sliced    | Slice shape | Slider range |
|-------------|----------------|-------------|--------------|
| Axial       | `vol[idx]`     | 128×128     | 1–64         |
| Coronal     | `vol[:, idx, :]` | 64×128    | 1–128        |
| Sagittal    | `vol[:, :, idx]` | 64×128    | 1–128        |

When switching orientation, update `slider.valmax` and the image `extent` via `make_extent()` so the non-square coronal/sagittal slices display with correct aspect ratio.

### MIP mode

When MIP is active (`mip_mode[0] == True`), call `get_mip(vol, orient)` instead of `get_slice` and hide the slider. Each orientation collapses a different axis (`axis=0/1/2` for Axial/Coronal/Sagittal). Metrics and the SSIM heatmap are computed on the MIP projections.

### Metrics display

Two tiers of metrics are shown:
- **Per-slice** (bold, line 1): `"Slice N — MSE: X | MAE: X | SSIM: X"` — updates on every slider/orientation/mode change via `metrics_text.set_text(...)`. Shows `"MIP — …"` in MIP mode.
- **Whole-volume** (gray, line 2): computed once at startup by `compute_volume_metrics()` (averages per-axial-slice SSIM); rendered as a fixed `fig.text()`.

### SSIM heatmap (3rd panel)

- `cmap="RdYlGn"`, `vmin=0, vmax=1` — red = low similarity, green = high.
- Computed per-slice/MIP via `compute_slice_metrics(s1, s2)` which calls `structural_similarity(..., full=True, data_range=1.0)`.
- The heatmap extent and aspect ratio update with orientation changes exactly like the gray panels.

### Shared mutable state in closures

Widget callbacks share state through single-element lists (`orientation: list[str] = ["Axial"]`, `mip_mode: list[bool] = [False]`). This avoids `nonlocal` and is the established pattern — continue using it for any new shared state.

Volume data is stored as `vols: list[np.ndarray | None] = [None, None]` and file paths as `paths: list[str | None] = [None, None]`. Either slot can be `None` before the user selects a file.

### File loading flow

The window opens immediately with blank placeholder panels. Each reconstruction panel has a `Button` widget below it labeled "Choose File" (changes to "Change File" after loading).

- `open_file(idx)` — opens tkinter dialog, loads+normalizes the volume, updates the panel title and placeholder visibility, changes the button label, then calls `refresh_state()`.
- `refresh_state()` — checks which vols are loaded; if both are present, recomputes volume metrics and calls `redraw()`; otherwise clears metrics text and shows the SSIM placeholder.
- `redraw()` — guards against `None` entries in `vols`; only updates panels whose data is available.

`Button` widget refs (`btn1`, `btn2`) must be kept as variables to prevent garbage collection from disconnecting callbacks.

## Key Dependencies

| Package       | Purpose                                      |
|---------------|----------------------------------------------|
| numpy         | Binary file I/O and array ops               |
| matplotlib    | Image display and slider/radio widgets       |
| scikit-image  | SSIM scalar and heatmap (`structural_similarity`) |
| tkinter       | Native file picker dialogs (stdlib)          |

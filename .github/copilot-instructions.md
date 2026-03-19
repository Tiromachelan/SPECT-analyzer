# Project Guidelines

## Overview

SPECT-analyzer is a single-script Python tool for side-by-side comparison of SPECT scan reconstructions. The main entry point is `visualizer.py`.

## Data Format

- Input files are `.raw` — flat binary arrays of **32-bit floats** (`np.float32`).
- Each file contains **64 slices** of **128×128 pixels** (total: 1,048,576 floats).
- Data is reshaped as `(64, 128, 128)` — slice-first ordering.
- Slices are presented to the user as **1-indexed** (1–64); convert to 0-indexed for array access.
- `.raw` files are gitignored; sample data lives in the repo root but is not committed.

## Code Style

- Python 3.14, single-file architecture (`visualizer.py`).
- Type hints on function signatures (`-> np.ndarray`, `-> tuple[str, str]`).
- Docstrings on public functions; section comments with `# --- label ---` dividers.
- Visualization uses **matplotlib** with `matplotlib.widgets.Slider` for interactivity.
- File selection via **tkinter** `filedialog` as fallback when CLI args aren't provided.

## Build and Test

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install numpy matplotlib

# Run with file picker dialogs
python visualizer.py

# Run with explicit file paths
python visualizer.py path/to/first.raw path/to/second.raw
```

## Project Conventions

- Volumes are independently normalised to `[0, 1]` before display — do not assume shared intensity scales across files.
- The slider controls both images simultaneously; keep left/right views locked to the same slice index.
- Use grayscale colormap (`cmap="gray"`) for medical imaging display.
- Filenames (extracted from paths) are used as subplot titles.

## Key Dependencies

| Package    | Purpose                            |
|------------|------------------------------------|
| numpy      | Binary file I/O and array ops     |
| matplotlib | Image display and slider widget    |
| tkinter    | Native file picker dialogs (stdlib)|

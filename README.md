# SPECT-analyzer

A desktop tool for side-by-side comparison of two reconstructed SPECT scans. Load two `.raw` files and interactively inspect per-slice and whole-volume similarity metrics, with controls for orientation, colormap, MIP projection, and background masking.

## Requirements

- Python 3.14+
- uv

## Installation

```bash
git clone https://github.com/Tiromachelan/SPECT-analyzer.git
cd SPECT-analyzer
uv sync
```

## Running

```bash
uv run visualizer.py
```

## Data Format

Input files must be flat binary arrays of 32-bit floats (`.raw`). 
- **Pre-Reconstructed Volumes**: Must be exactly 64 slices of 128x128 pixels (4,194,304 bytes total).  Data is interpreted in slice-first order: `vol[z, y, x]`.

- **Sinograms**: Exactly 120 128x128 images (7,864,320 bytes total)

## Interface Overview

The window is divided into a left control panel and a three-panel display area. Two metric rows appear above the display:

- **Per-slice metrics** (bold): MSE, MAE, and SSIM for the currently displayed slice or MIP projection. Updated every time the view changes.
- **Volume metrics** (gray): MSE, MAE, and SSIM averaged across all 64 axial slices. Computed once when both files are loaded.

### Display panels

| Panel | Content |
|---|---|
| Left | File 1 |
| Center | File 2 |
| Right | SSIM heatmap (red = low similarity, green = high similarity) |

Each volume is independently normalized to [0, 1] before display.

### Controls

#### File 1 / File 2

- **Choose File** — opens a file dialog to load a `.raw` file. The button label changes to "Change File" after a file is loaded. The filename appears below the button.
- **SSIM Overlay** — when checked, overlays the SSIM heatmap semi-transparently on top of that panel's scan image. Only available when both files are loaded.

#### Mode

- **Slice** — displays a single cross-sectional slice. The slice slider is active.
- **MIP** — displays the Maximum Intensity Projection across all slices for the selected orientation. The slice slider is disabled.

#### View

Selects the anatomical orientation of the slice plane:

| Option | Slice axis | Slider range |
|---|---|---|
| Axial | Top-down (z-axis) | 1–64 |
| Coronal | Front-back (y-axis) | 1–128 |
| Sagittal | Left-right (x-axis) | 1–128 |

Switching orientation updates the slider range and redraws the display.

#### Slice

A horizontal slider that selects which slice to display. The current position is shown as `N / total`. Disabled in MIP mode.

#### Colormap

Selects the colormap applied to both scan panels. The SSIM heatmap always uses RdYlGn regardless of this setting.

| Option | Description |
|---|---|
| Hot Metal Blue | Custom nuclear medicine colormap (default) |
| Gray | Standard grayscale |
| Afmhot | Sequential warm colormap |
| Viridis | Perceptually uniform, blue to yellow |
| Plasma | Perceptually uniform, blue to orange |

#### Mask

Background masking excludes low-intensity voxels from metric calculations. Only active when both files are loaded.

- **Enable** checkbox — turns masking on or off. When enabled, voxels where both volumes are below the threshold are excluded from MSE, MAE, and SSIM computations.
- **Threshold slider** — sets the intensity cutoff (0.000–1.000). Voxels with `max(file1, file2) < threshold` are masked out.
- **Threshold value label** — shows the current threshold value.
- **Auto** button — resets the threshold to the automatically computed value (Otsu's method applied to the combined foreground of both volumes).

## Metrics

All metrics are computed on normalized [0, 1] intensity data.

- **MSE** (Mean Squared Error) — average squared difference between corresponding voxels. Lower is better; 0 is identical.
- **MAE** (Mean Absolute Error) — average absolute difference between corresponding voxels. Lower is better; 0 is identical.
- **SSIM** (Structural Similarity Index) — perceptual similarity measure ranging from 0 to 1. Higher is better; 1 is identical.

When masking is enabled, metrics are computed only over foreground voxels. SSIM is weighted by the number of valid interior pixels per slice to avoid boundary artifacts from the SSIM sliding window.


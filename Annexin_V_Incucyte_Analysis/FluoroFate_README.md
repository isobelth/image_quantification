# FluoroFate

A **napari**-based GUI for quantifying cell death and cell fate from multi-channel fluorescence timelapse TIFF images.

## Overview

The pipeline segments cells, tracks them across frames, thresholds fluorescence channels, and classifies each cell's fate. Results are exported as CSVs, PDFs, and visualised interactively in napari.

![Pipeline overview](images/image1.png)

## Pipeline

1. **Cellpose segmentation** — Brightfield frames are segmented into individual cell masks using a built-in or user-supplied Cellpose model.
2. **TrackMate tracking** — Cell identities are linked across frames using TrackMate's Advanced Kalman Tracker (run headlessly via PyImageJ). When splitting is enabled, division events are detected and a hierarchical lineage tree is built.
3. **Fluorescence thresholding** — Up to 3 fluorescence channels are blurred (Gaussian) and thresholded (per-channel configurable method) to produce binary positive-signal masks.
4. **Fate assignment** — Each tracked cell is classified in one of two modes:
   - **Persistent** — the first fluorophore to appear determines the cell's permanent fate (e.g. Annexin V = apoptosis, PI = necroptosis).
   - **Snapshot** — per-frame classification by currently active fluorophores (e.g. FUCCI cell-cycle analysis).
5. **Outputs** — Percentage trajectories, spatial trajectory plots, cell timeline swimlane plots, and cell-level CSVs are generated at multiple frame-presence cutoffs (30/40/50/60%).

## Requirements

- Python 3.9+
- [napari](https://napari.org/)
- [Cellpose](https://www.cellpose.org/)
- [PyImageJ](https://pyimagej.readthedocs.io/) (for TrackMate)
- Standard scientific Python: numpy, pandas, matplotlib, seaborn, scikit-image, tifffile

Install with:

```bash
pip install napari[all] cellpose pyimagej tifffile scikit-image pandas matplotlib seaborn
```

A compatible Java installation is required for TrackMate. The GUI will attempt to locate one automatically via `jdk4py` or the conda environment.

## Quick Start

### From a notebook

```python
from fluorofate import launch

app = launch()
```

### From the command line

```bash
python fluorofate.py
```

## Input Format

- **4-D TIFF** with shape `(T, C, Y, X)` — time, channels, height, width.
- At least one brightfield channel and one fluorescence channel.
- Supported extensions: `.tif`, `.tiff`.

## GUI Panels

### Files

| Parameter | Description |
|---|---|
| Single TIFF | Path to one multi-channel timelapse TIFF |
| Batch folder | Folder of TIFFs (optional; only needed for batch mode) |
| Output directory | Where results are saved (defaults to a subfolder next to the image) |

### Channels

Each fluorophore has its own name, channel index, and thresholding method.

| Parameter | Default | Description |
|---|---|---|
| Brightfield channel | -1 (last) | Channel index for brightfield |
| Fluorophore 1 name | Annexin_V | Name used in output columns/files |
| Fluorophore 1 channel | 0 | Channel index |
| Fluorophore 1 threshold | mean | Thresholding method for this channel |
| Fluorophore 2 name | PI | Name used in output columns/files |
| Fluorophore 2 channel | 1 | Channel index |
| Fluorophore 2 threshold | mean | Thresholding method for this channel |
| Fluorophore 3 name | *(blank)* | Leave blank to skip |
| Fluorophore 3 channel | 2 | Channel index |
| Fluorophore 3 threshold | mean | Thresholding method for this channel |

### Analysis

| Parameter | Default | Description |
|---|---|---|
| Analysis mode | persistent | `persistent` or `snapshot` |
| Blur sigma | 1.0 | Gaussian blur sigma applied before thresholding |
| Custom model file | *(empty)* | Optional `.pt`/`.pth` Cellpose model; overrides built-in model |

### Cellpose

| Parameter | Default | Description |
|---|---|---|
| Model | cyto3 | Built-in model (auto-disabled when a custom model is selected) |
| Min cell size | 15 | Minimum object size in pixels |
| Use GPU | True | Enable GPU acceleration |

### TrackMate

| Parameter | Default | Description |
|---|---|---|
| Init search radius | 30.0 | First-frame linking distance |
| Search radius | 150.0 | Kalman filter search radius |
| Max frame gap | 3 | Frames a cell can disappear before the track is closed |
| Allow splitting | False | Permit cell division events |
| Splitting max distance | 15.0 | Maximum distance for splitting links (only used when splitting is enabled) |
| Allow merging | False | Permit cell fusion events |

## Workflow Buttons

Stages can be run individually or all at once:

| Button | Description |
|---|---|
| **1) Run Segmentation** | Cellpose segmentation only; saves `masks_stack.tiff` |
| **2) Run Tracking** | TrackMate tracking from saved masks; saves `linked_labels_trackmate.tiff` and `trackmate_tracks.csv` |
| **3a) Run Persistent Analysis** | Persistent fate assignment + plots from saved tracking |
| **3b) Run Snapshot Analysis** | Snapshot fate assignment + plots from saved tracking |
| **Run All (Single Image)** | All stages sequentially on one image (runs both persistent and snapshot) |
| **Analyse All in Folder** | Batch: segment + track + analyse every TIFF in a folder |
| **Save Results** | Export accumulated result rows to a user-chosen CSV |

## Output Files

All outputs are saved in a per-image subfolder under the output directory.

### Segmentation & Tracking

| File | Description |
|---|---|
| `masks_stack.tiff` | Cellpose segmentation masks (T, Y, X) |
| `linked_labels_trackmate.tiff` | Tracked labels with consistent cell IDs |
| `trackmate_tracks.csv` | Track positions: `track_id`, `t`, `y`, `x`, `quality`, plus `lineage_id`, `parent_track_id`, `generation` columns |
| `trackmate_tracks_by_cell.csv` | Same data with additional `cell_id` (= track_id + 1) and `frame` columns |
| `lineage_summary.csv` | One row per track: lineage ID, parent track, generation, first/last frame, number of frames |

### Persistent Mode

| File | Description |
|---|---|
| `assignments_persistent.csv` | Per-cell fate, first-positive frame, and positive area per fluorophore |
| `persistent_by_cell.csv` | Same with `cell_id`, `track_id`, and lineage columns |
| `percentages_persistent.csv` | Cumulative % positive cells per frame |
| `percentages_persistent.pdf` | Line plot of cumulative percentages |
| `percentages_persistent_{N}pct.csv/pdf` | Same filtered to cells present in ≥N% of frames (N = 30, 40, 50, 60) |

### Snapshot Mode

| File | Description |
|---|---|
| `snapshot.csv` | Per-cell per-frame category, boolean flags, and positive area per fluorophore |
| `snapshot_by_cell_long.csv` | Long format with `cell_id`, `track_id`, `frame`, `category`, and lineage columns |
| `snapshot_by_cell_wide.csv` | Wide format: one row per cell, one column per frame's category, plus lineage columns |
| `percentages_snapshot.csv` | Per-frame category percentages |
| `percentages_snapshot.pdf` | Line plot of category percentages |
| `percentages_snapshot_{N}pct.csv/pdf` | Filtered to cells present in ≥N% of frames |
| `snapshot_trajectories_{N}pct.pdf` | Spatial trajectory plots colored by category |
| `snapshot_timelines_{N}pct.pdf` | Swimlane timeline plots showing each cell's category over time |

### Important Notes on Outputs

- **All CSVs include every tracked cell** regardless of how many frames it appears in. Frame-presence filtering (30/40/50/60%) is applied only to the cutoff plots.
- **Positive area columns** (`{fluorophore}_positive_area`) report the number of thresholded positive pixels overlapping each cell mask — per fluorophore, per frame (snapshot) or summed across all frames (persistent).
- **Lineage columns** (`lineage_id`, `parent_track_id`, `generation`) are included when TrackMate splitting is enabled. Lineage IDs use dotted notation (e.g. `1`, `1.1`, `1.2`) so that mother and daughter cells can be grouped together.

## Custom Cellpose Model

To use a custom-trained model:

1. Train a model using the [Cellpose GUI](https://www.cellpose.org/) or the training API.
2. In the **Analysis** panel, click the file picker next to **Select file (optional custom model)** and choose your `.pt` or `.pth` file.
3. The built-in model dropdown will auto-disable.

An example custom model is included in the `Example_Custom_Cellpose_Model` folder.

## Analysis Modes Explained

### Persistent

Each cell is assigned the fate of whichever fluorophore's threshold is exceeded **first** across the timelapse. Once assigned, the fate is permanent. This suits experiments where fluorescence appearance order determines the biological outcome (e.g. Annexin V before PI = apoptosis).

### Snapshot

Each cell is classified **independently at every frame** based on which fluorophores are currently active. Categories are formed by combining active fluorophore names (e.g. `Annexin_V`, `PI`, `Annexin_V+PI`, `negative`). This suits experiments with reversible or cyclical states (e.g. FUCCI cell-cycle reporters).

## Thresholding Methods

Each fluorophore channel can use a different thresholding method, configured in the Channels panel.

| Method | Description |
|---|---|
| mean | Pixels above the image mean intensity |
| minimum | Histogram-based minimum method (bimodal assumption) |
| yen | Yen's method (maximises correlation of original vs thresholded) |
| otsu | Otsu's method (minimises intra-class variance) |
| triangle | Triangle algorithm (suitable for unimodal histograms with a tail) |

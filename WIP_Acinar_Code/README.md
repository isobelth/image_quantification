# Acinar Analysis & GUI

## Overview
A unified, modular pipeline for 3D acinar image quantification with a standalone GUI for batch analysis. All analyses share a common acinus segmentation step, then branch into specific measurements.

---

## Quick Start

**From a notebook:**
```python
%gui qt
from acinar_gui import launch_and_run
results = launch_and_run()
```

**From the command line:**
```bash
python acinar_analysis.py --image-dir ./images --analyses acinus_shape --nuclear-channel 0 --membrane-channel 2
```

---

## Shared Step: Acinus Segmentation

Every analysis begins by identifying the primary acinus in the image. The pipeline:

1. **Builds an acinus approximation** by summing the nuclear + membrane channels (plus any active stain channels like C3, EdU, mito).
2. **Downscales** to 0.25× isotropic resolution for speed.
3. **Clips intensity** to the 10th–85th percentile range, then applies Gaussian smoothing (σ = 3).
4. **Thresholds** with Li's method, removes small holes/objects, keeps the largest connected component.
5. **Per-slice hole filling** — each Z-slice is independently filled and only the largest component retained to avoid capturing neighbouring acini.
6. **Sphericity check** — if the shape is too elongated (eigenvalue ratio < 0.55, suggesting merged acini), the pipeline re-segments with a triangle threshold + erosion to split them.
7. **Final erosion** (ball radius 5) offsets any expansion from smoothing/filling, then a final largest-component filter.

The result is a binary 3D mask used by all downstream analyses.

---

## Analysis Modules

### 1. Acinus Shape

| | |
|---|---|
| **What it measures** | Acinus volume (µm³) and roundness (sphericity) |
| **GUI checkbox** | Acinus Shape |
| **Required folders** | Image folder only |
| **Required channels** | None |

**How it works:** Uses the shared acinus segmentation, then computes volume from voxel count × voxel size³ and roundness as the ratio of the smallest to largest inertia tensor eigenvalue (1.0 = perfect sphere).

**Output columns:**
| Column | Description |
|---|---|
| `acinus_volume_um3` | Total acinus volume in µm³ |
| `acinus_roundness` | Eigenvalue ratio (closer to 1 = more spherical) |
| `flag` | `None`, `hole` (internal cavity detected), or `multiple_acini_split` |

**QC plot:** Mid-Z slice showing the acinus mask boundary overlaid on the raw image.

---

### 2. Cell & Nuclear Shape

| | |
|---|---|
| **What it measures** | Per-cell and per-nucleus volume, roundness, and cell–cell neighbour relationships |
| **GUI checkbox** | Cell & Nuclear Shape |
| **Required folders** | Nuclear Mask Folder, Membrane Mask Folder |
| **Required channels** | None |

**How it works:**
1. Rescales nuclear and membrane binary masks to acinus resolution and restricts them to the acinus boundary.
2. **Nuclei:** Otsu threshold → hole fill → distance-transform watershed (4 µm min separation, 2 µm min radius filter).
3. **Cells:** Membrane mask is watershed-segmented using nuclear centroids as seeds, then expanded (12 px) within the acinus to fill gaps.
4. Each nucleus is matched to its enclosing cell by centroid lookup.
5. Cell–cell neighbour counts are computed via voxel adjacency along all three axes.

**Output columns:**
| Column | Description |
|---|---|
| `nucleus_label`, `cell_label` | Integer IDs |
| `nucleus_volume_um3`, `cell_volume_um3` | Volumes in µm³ |
| `nucleus_roundness`, `cell_roundness` | Eigenvalue ratios |
| `nucleus_cell_volume_ratio` | Nucleus vol / cell vol |
| `sum` | Total neighbour count |
| `external` | Neighbours touching the acinus boundary |
| `internal` | Neighbours entirely within the acinus |

**QC plot:** Mid-Z nuclear labels + cell labels colour-coded on the raw image.

---

### 3. Protein Polarisation

| | |
|---|---|
| **What it measures** | How protein intensity varies with radial distance from the acinus centre to its edge |
| **GUI checkbox** | Protein Polarisation |
| **Required folders** | None |
| **Required channels** | Protein Channel (≥ 0) |

**How it works:**
1. Rescales the protein channel and masks it to a slightly expanded acinus region (5 px dilation).
2. Computes a distance transform from the acinus boundary, normalised by the equivalent sphere radius of the acinus.
3. Bins voxels by their normalised distance (rounded to 2 decimals) and averages the protein intensity per bin.

**Output columns:**
| Column | Description |
|---|---|
| `rounded_distance` | Normalised radial distance (0 = edge, larger = deeper inside) |
| `protein_intensity` | Mean protein intensity in that distance bin |

**QC plot:** Acinus mask overlay with the protein channel shown in red.

---

### 4. Apoptosis (C3)

| | |
|---|---|
| **What it measures** | Number and spatial distribution of C3-positive (apoptotic) cells, plus total nuclear count |
| **GUI checkbox** | Apoptosis (C3) |
| **Required folders** | C3 Mask Folder, Nuclear Mask Folder |
| **Required channels** | C3 Channel (≥ 0) |

**How it works:**
1. Watershed-segments C3 mask objects within the acinus (7 µm separation, 1.3 µm min radius).
2. Watershed-segments nuclear mask objects (6 µm separation, 2 µm min radius).
3. Computes a distance map normalised 0–1 from the acinus boundary.
4. Records each C3 object's volume and normalised distance from the acinus edge.

**Output columns:**
| Column | Description |
|---|---|
| `label` | C3 object ID (or `no_c3` if none found) |
| `c3_volume_um3` | Volume of each C3+ object |
| `normalised_distance` | Position within acinus (0 = boundary, 1 = centre) |
| `acinus_volume_um3`, `acinus_roundness` | Acinus-level metrics |
| `number_of_nuclei` | Total nuclei detected in the acinus |

**QC plot:** Acinus mask + colour-coded C3 labels + colour-coded nuclear labels.

---

### 5. Protein Proximity

| | |
|---|---|
| **What it measures** | Intensity of a chosen protein near dying (C3+) vs non-dying cells |
| **GUI checkbox** | Protein Proximity |
| **Required folders** | C3 Mask Folder, Nuclear Mask Folder |
| **Required channels** | C3 Channel (≥ 0), Proximity Protein Channel (≥ 0) |

**How it works:**
1. Classifies cells as **dying** (C3 mask objects) or **non-dying** (nuclear mask minus dilated C3 mask).
2. Watershed-segments both populations within the acinus.
3. Builds estimated cell territories (expand labels × 20), then expands each territory by a search radius (default 5 µm).
4. Measures the proximity-protein intensity both inside each cell and in its surrounding neighbourhood.

**Output columns:**
| Column | Description |
|---|---|
| `dying` | `Y` (C3+) or `N` |
| `proximity_intensity_in_cell` | Total protein signal within the cell |
| `proximity_intensity_around_cell` | Signal in neighbourhood minus in-cell |
| `proximity_mean_intensity_in_cell` | Mean intensity within cell |
| `proximity_mean_intensity_neighborhood` | Mean intensity in full neighbourhood |
| `estimated_cell_territory_volume_um3` | Cell territory volume |
| `proximity_neighborhood_volume_um3` | Neighbourhood volume |
| `number_dying`, `number_not_dying` | Cell counts per acinus |

**QC plot:** Acinus mask + colour-coded dying cells + colour-coded non-dying cells.

---

### 6. Proliferation (EdU)

| | |
|---|---|
| **What it measures** | Number and spatial distribution of EdU-positive (dividing) vs non-dividing cells |
| **GUI checkbox** | Proliferation (EdU) |
| **Required folders** | EdU Mask Folder, Nuclear Mask Folder |
| **Required channels** | EdU Channel (≥ 0) |

**How it works:**
1. **Dividing cells** = EdU mask objects, watershed-segmented (4 µm separation, 2 µm min radius).
2. **Non-dividing cells** = nuclear mask minus dilated EdU mask, watershed-segmented.
3. Computes normalised distance of each cell from the acinus boundary.

**Output columns:**
| Column | Description |
|---|---|
| `dividing` | `Y` (EdU+) or `N` |
| `cell_volume_um3` | Individual cell volume |
| `normalised_distance` | Position within acinus (0–1) |
| `acinus_volume_um3`, `acinus_roundness` | Acinus-level metrics |
| `number_dividing`, `number_not_dividing` | Cell counts per acinus |

**QC plot:** Acinus mask + colour-coded dividing cells + colour-coded non-dividing cells.

---

### 7. Mitochondria

| | |
|---|---|
| **What it measures** | Per-cell mitochondrial count, total mito volume, mito-to-cell volume ratio, and mito distance distribution from the nucleus |
| **GUI checkbox** | Mitochondria |
| **Required folders** | Nuclear Mask Folder, Membrane Mask Folder, Mito Mask Folder |
| **Required channels** | None |

**How it works:**
1. Labels and size-filters the mito mask (min 10 voxels), rescales to acinus resolution.
2. Segments nuclei and cells (same as Cell & Nuclear Shape).
3. For each nucleus, finds mito objects overlapping its watershed territory.
4. Distance-bins mito pixels from the nucleus surface within the cell boundary.
5. Filters implausible cells (mito/cell vol ratio > 0.5, nucleus/cell vol ratio > 0.9, or zero mito volume).

**Output columns:**
| Column | Description |
|---|---|
| `nucleus_label`, `cell_label` | Integer IDs |
| `nucleus_volume_um3`, `cell_volume_um3` | Volumes |
| `number_of_mito` | Mito objects per cell |
| `mito_volume_um3` | Total mito volume per cell |
| `mito_cell_vol_ratio` | Mito vol / cell vol |
| `mean_mito_distance_ratio` | Mean mito pixel density per distance bin from nucleus |
| `acinus_volume_um3`, `total_mito_volume_um3`, `number_of_cells` | Acinus-level summaries |

**QC plot:** Acinus mask + cell labels + mito labels.

---

### 8. Membrane Upregulation

| | |
|---|---|
| **What it measures** | Whether membrane-channel signal is enriched at the acinus periphery compared to deeper inside |
| **GUI checkbox** | Membrane Upregulation |
| **Required folders** | None |
| **Required channels** | Membrane Channel (≥ 0) |

**How it works:**
1. Computes a distance transform from the acinus boundary inward.
2. Defines an **edge shell** (0–3 µm from boundary) and an **inner shell** (8–11 µm from boundary; i.e. 3 µm shell + 5 µm gap + 3 µm shell).
3. Measures the **median** membrane-channel intensity in each shell.
4. Computes the ratio (edge median / inner median). Values > 1 indicate peripheral enrichment.

Shell width (3 µm) and offset (5 µm) are fixed and not user-configurable.

**Output columns:**
| Column | Description |
|---|---|
| `acinus_volume_um3`, `acinus_roundness` | Acinus-level metrics |
| `membrane_edge_shell_median` | Median membrane intensity in the outer shell |
| `membrane_inner_shell_median` | Median membrane intensity in the inner shell |
| `membrane_edge_to_inner_ratio` | Edge / inner (NaN if inner = 0) |
| `edge_shell_volume_um3`, `inner_shell_volume_um3` | Shell volumes |

**QC plot:** Acinus mask + edge shell region + inner shell region overlaid on raw image (red = membrane channel). Title shows the ratio value.

---

## Requirements Summary Table

| Analysis | Mask Folders Needed | Channels Needed (≥ 0) |
|---|---|---|
| Acinus Shape | — | — |
| Cell & Nuclear Shape | Nuclear, Membrane | — |
| Protein Polarisation | — | Protein |
| Apoptosis (C3) | C3, Nuclear | C3 |
| Protein Proximity | C3, Nuclear | C3, Proximity Protein |
| Proliferation (EdU) | EdU, Nuclear | EdU |
| Mitochondria | Nuclear, Membrane, Mito | — |
| Membrane Upregulation | — | Membrane |

---

## Imaging Record (Optional)

An `imaging_record.yml` file can be provided to control how experimental metadata (well, day, cell type, condition, treatment) is extracted from filenames. If no file is provided, built-in rules are used. The YAML uses simple substring-matching rules evaluated top-to-bottom per field. See the included [imaging_record.yml](imaging_record.yml) for the format and default rules.

---

## Output

- One CSV per analysis type is saved in the output directory (e.g. `acinar_results_acinus_shape.csv`).
- Every output row includes parsed filename metadata: `filename`, `well`, `day`, `cell_type`, `condition`, `treatment`, `image_type`, `flag`.
- If **Save QC Plots** is enabled, overlay PNGs are saved to `output_dir/qc_plots/`.

---

## Background: Mask Preparation

Signal attenuation in deeper z-slices is a major challenge in 3D acinar imaging. The recommended workflow:

1. Split z-stacks into top (bright) and bottom (dim) halves.
2. Train separate [Labkit](https://imagej.net/plugins/labkit/) classifiers in FIJI/ImageJ for each half.
3. Segment, then concatenate top + bottom masks into full-stack masks.
4. Save masks in separate folders per channel (one mask TIFF per image TIFF, matched alphabetically).

---

## Requirements
- Python 3.8+
- numpy, pandas, scikit-image, scipy, tifffile, joblib, tqdm, magicgui, qtpy, matplotlib, pyyaml
- FIJI/ImageJ with [Labkit](https://imagej.net/plugins/labkit/) for mask generation

---

## Troubleshooting
- Mask folders must contain the **same number** of files as the image folder (matched alphabetically).
- Channel indices are **zero-based** (e.g. nuclear channel = 0).
- Set unused channels to **-1** in the GUI.
- If acinus segmentation fails or looks wrong, check the QC plots — a `multiple_acini_split` flag means merged acini were detected and re-segmented.

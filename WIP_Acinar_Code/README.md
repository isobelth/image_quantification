# Acinar Analysis & GUI

## Overview
This project provides a unified, modular pipeline for 3D acinar image quantification, including a standalone graphical user interface (GUI) for batch analysis. It supports:
- Acinus shape (volume & roundness)
- Cell & nuclear shape (requires segmentation masks)
- Protein polarisation (BM intensity vs radial distance)
- Apoptosis quantification (C3+ cell counting)
- Protein proximity analysis (intensity near dying vs non-dying cells)
- Proliferation analysis (EdU+ dividing vs non-dividing cells)
- Mitochondria analysis (per-cell mito count, volume, distance from nucleus)

The GUI allows users to configure analyses, select input/output folders, and run batch processing with progress feedback.

---

## Background
A major challenge in 3D acinar imaging is signal attenuation in deeper z-slices. To address this, image stacks are split into top (bright) and bottom (dim) halves, and segmented separately using [Labkit](https://imagej.net/plugins/labkit/) in FIJI/ImageJ. Classifiers are trained for each substack, and segmentations are recombined for downstream analysis. This approach ensures robust segmentation across varying fluorescence intensities.

---

## Workflow
### 1. Prepare Segmentation Masks
- Use Labkit in FIJI/ImageJ to train classifiers for nuclear and membrane channels on both top and bottom halves of your z-stacks.
- Segment each half, then concatenate results to form full-stack masks.
- Save masks in separate folders for each channel (nuclear, membrane, C3, EdU, mito, etc.).

### 2. Launch the GUI
- From a notebook:
  ```python
  %gui qt
  from acinar_gui import launch
  app = launch()
  ```
- Or standalone:
  ```bash
  python acinar_gui.py
  ```

### 3. Configure Analysis
- Select input image folder and mask folders as required.
- Set channel indices for each marker (nuclear, membrane, protein, etc.).
- Choose analyses to run (tick boxes).
- Set output directory.
- Click **Run Analysis**. The GUI will close and processing will begin with progress shown in the terminal.

### 4. Output
- Results are saved as CSV files in the output directory, one per analysis type.
- Optionally, QC plots are generated for each image.

---

## Analysis Modules
- **Acinus Shape:** Segments the acinus, computes volume and roundness.
- **Cell & Nuclear Shape:** Segments cells/nuclei, computes volume, roundness, and neighbour info (requires nuclear & membrane masks).
- **Protein Polarisation:** Quantifies protein intensity as a function of radial distance from the acinus exterior (requires protein channel).
- **Apoptosis:** Counts C3+ (apoptotic) cells and total nuclei per acinus (requires C3 & nuclear masks).
- **Protein Proximity:** Compares protein intensity near dying vs non-dying cells (requires C3, nuclear masks, and proximity protein channel).
- **Proliferation:** Counts EdU+ (dividing) vs non-dividing cells (requires EdU & nuclear masks).
- **Mitochondria:** Per-cell mitochondria count, volume, and distance from nucleus (requires nuclear, membrane, and mito masks).

---

## Example: Batch Analysis via CLI
```bash
python acinar_analysis.py --image-dir ./images --analyses acinus_shape protein_polarisation --protein-channel 1 --output results.csv
```

---

## Experimental Metadata
The pipeline parses filenames to infer experimental details (well, day, cell type, condition, treatment) and includes these in the output CSVs.

---

## Requirements
- Python 3.8+
- numpy, pandas, scikit-image, tifffile, joblib, tqdm, magicgui, matplotlib
- FIJI/ImageJ with Labkit plugin for mask generation

---

## References
- [Labkit plugin](https://imagej.net/plugins/labkit/)
- [magicgui](https://github.com/napari/magicgui)

---

## Troubleshooting
- Ensure mask folders contain the same number of files as the image folder.
- Channel indices are zero-based (e.g., nuclear channel = 0).
- For best results, use isotropic voxel sizes or rescale as needed.


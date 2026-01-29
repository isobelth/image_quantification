# Placenta Perfusion Analysis

## Overview
This notebook processes Leica .lif perfusion datasets to segment maternal/fetal regions, compute permeability metrics across timepoints, and aggregate results for each image into a table.

## What it does
- Loads each image in every .lif file within a target folder.
- Extracts a boundary surface per z-plane using edge/ridge detection.
- Builds a smoothed interface surface and segments maternal vs fetal regions.
- Calculates volumes, interface area, intensities, bleaching correction, and permeability values.
- Collects all metrics into a single pandas DataFrame.

## Inputs
- A folder containing one or more .lif files.
- Images with at least 4 timepoints and a dextran channel.

## Key parameters
Update these in the analysis cell:
- `folder`: path to the dataset directory
- `DEXTRAN_CHANNEL`: index of the dextran channel
- `params`: ridge/edge detection parameters

## Output
The analysis cell returns a DataFrame with one row per image. It includes:
- image identifiers (file, index, name)
- z-plane counts and kept z-range
- maternal/fetal volumes (voxels and µm³)
- interface area (µm²)
- permeability values and percent changes
- voxel size and failure diagnostics

To save results to CSV, add:

```python
results_df.to_csv("perfusion_results.csv", index=False)
```

## Notes
- The segmentation step can fail for some images. Those rows will include an `error` or geometry diagnostics.
- Adjust `params` if boundary detection is unstable for a dataset.

## Dependencies
- numpy, pandas, scipy
- scikit-image
- liffile
- napari (only for visualization cells)
- matplotlib

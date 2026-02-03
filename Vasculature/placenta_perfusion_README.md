# Placenta Perfusion Analysis

## Overview
This workflow quantifies dextran permeability across a placental barrier from Leica `.lif` perfusion datasets. The pipeline segments maternal and fetal compartments, reconstructs the perfusable interface in 3D, and computes permeability over time using fluorescence intensity dynamics.

Experimental layout:
![README_images/image2.png](README_images/image2.png)

Key objectives:
1. Reconstruct the barrier surface across all $z$-slices.
2. Exclude out-of-focus lower planes and upper planes where maternal and fetal regions are separated by an impermeable channel.
3. Quantify permeability by measuring dextran signal in maternal and fetal compartments over time.

## Methods (Summary)
**Assumptions:** the barrier is approximately horizontal in the field of view and its position is stable throughout the time series.

1. **Boundary detection (at $t_0$):** A boundary is extracted per $z$-plane using edge/ridge detection.
   - The pipeline computes an intensity ratio between the upper and lower halves of each image. If the ratio is below `ratio_cutoff`, the barrier appears as a ridge; Sato filtering is used to detect it, followed by skeletonization and shortest-path tracing across the ridge. If the ratio is above `ratio_cutoff`, a Li threshold separates bright/dark regions, and the boundary is defined at their interface.
2. **Perfusable plane selection:** Slices without a valid path are discarded, as are slices with low similarity to the bulk of the stack (indicative of paths around the impermeable channel).
3. **Interface reconstruction:** Selected planes are interpolated into a smooth surface; isolated missing slices are filled by interpolation.
4. **Compartment segmentation:** Maternal and fetal regions are segmented, and volumes, interface area, and total dextran intensities are computed. Photobleaching correction and permeability metrics are derived.
5. **Time-series quantification:** The reconstructed compartments are treated as fixed over time; dextran intensities are measured per timepoint to estimate permeability across the perfusable interface.

Illustrative reconstruction:
![README_images/image3.png](README_images/image3.png)

## Inputs
- Leica `.lif` perfusion image stacks containing a dextran tracer channel.
- Voxel size metadata (µm) embedded in the image or provided by the user.
- A representative $t_0$ frame for boundary detection.

## Outputs
The analysis returns a per-image table with:
- image identifiers (file, index, name)
- retained $z$-range and number of planes
- maternal/fetal volumes (µm³)
- interface area (µm²)
- permeability values and percent change over time
- voxel size and quality-control diagnostics

## Quality Control and Troubleshooting
- Boundary detection can fail in low-contrast or atypical morphologies; failures are reported via an `error` field or geometry diagnostics.
- If boundary detection is unstable, tune `params` (notably `ratio_cutoff` and thresholding settings) and re-run.
- Verify reconstructed interfaces visually before downstream analysis.

## Limitations
- The method assumes minimal barrier motion and a roughly horizontal interface. Significant tissue drift or oblique barriers may require reorientation or customized preprocessing.
- Permeability is inferred from fluorescence intensity and depends on consistent imaging settings and bleaching correction.

## Citation
If you use this pipeline in a publication, please cite the associated study and this repository.


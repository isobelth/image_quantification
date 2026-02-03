# Placenta Perfusion Analysis

## Overview
This notebook processes .lif perfusion datasets to segment maternal/fetal regions and compute permeability across a perfusable region.

The experiment is set up as follows:
![images/image2.png](images/image2.png)

Code aims to:
1. Reconstruct the barrier across all z slices
2. Remove lower (out of focus) slices and upper slices, where the maternal and fetal regions are separated by a thick, impermeable channel
3. Calculate permeability across the perfusable barrier by quantifying the intensity of a dextran tracer in the maternal and fetal regions at multiple timepoints

## Technical Details
*Assumptions*: Barrier runs horizontally across the image, the barrier position is unchanged over time

1. Uses the t_0 image to extract a boundary surface per z-plane using edge/ridge detection.
    - Due to heterogeneity across conditions, different thresholding methods are useful in different conditons. The code calculates the intensity ratio of the top and lower half of the image. If this ratio < ratio_cutoff, the barrier appears as a ridge, and sato thresholding is used to identify the boundary. The boundary is skeletonised and a route through it calculated that minimises route length and time off binary skeleton. If the intensity ratio is > ratio_cutoff, a Li threshold is used to identify the light and dark regio of the image, then define the barrier as their meeting point.
2. Identifies planes from (1) that contain the perfusable barrier
    - Z slices where no path was found are removed, as well as those that score low on similarity matching compared to the bulk of the stack (suggests that this path travels around the impermeable barrier rather than the perfusable barrier)
3. Planes of interest are separated and...
4. ...reconstructed into a smooth interface. Missing single-slices are interpolated.
5. The fetal and maternal regions of interest are segmented and their area and total dextran signal intensity is summed. Calculates volumes, interface area, intensities, bleaching correction, and permeability values.
6. The calculated barrier and fetal and maternal regions are assumed constant over the imaging times. The dextran intensities are calculated in each region at each timepoint to calculate permeability through the perfusable region


![images/image3.png](images/image3.png)


## Output
The analysis cell returns a DataFrame with one row per image. It includes:
- image identifiers (file, index, name)
- z-plane counts and kept z-range
- maternal/fetal volumes (µm³)
- interface area (µm²)
- permeability values and percent changes
- voxel size and failure diagnostics


## Notes
- The segmentation step can fail for some images. Those rows will include an `error` or geometry diagnostics.
- Adjust `params` if boundary detection is unstable for a dataset.


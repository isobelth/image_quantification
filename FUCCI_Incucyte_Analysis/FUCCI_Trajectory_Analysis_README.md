# Background

This pipeline extracts, tracks, and classifies fluorescently labelled nuclei from 4D time-lapse microscopy images. 

Nuclei are coloured red and/or green due to FUCCI (Fluorescent Ubiquitination-based Cell Cycle Indicator), a live-cell reporter system that marks nuclei according to their cell cycle phase. Red fluorescence indicates G1 phase, green marks S/G2/M phases, and yellow (red- and green-positive) represents cells transitioning from G1 to S phase.

The code supports:

- Preprocessing and scaling of raw image data
- Watershed-based segmentation of nuclei
- Tracking of nuclei across time using [TrackPy](https://soft-matter.github.io/trackpy/v0.6.4/)
- Classification of nuclei based on dominant colour signal
- Visualisation of nuclear tracks with colour-coded identities

## Image Segmentation

Nuclei are extracted and segmented using intensity thresholding, morphological operations, and watershed segmentation (the principles of which are set out below)

![images/image.png](images/image.png)

Red and green channels are combined to create a combined nuclear signal. This combined image is segmented to create “all_combined_nuclei” (no colour dependence)

Individual red and green channels are also segmented independently to create “all_green_nuclei” and “all_red_nuclei”: binary masks denoting green/red-positive nuclei

![images/image1.png](images/image1.png)

## Nuclear Tracking and Plotting

Nuclear trajectories are tracked using centroids and [TrackPy](https://soft-matter.github.io/trackpy/v0.6.4/). Filters for long-lived tracks are applied, and the dominant colour of each nucleus at each timepoint is calculated.

Cells are classified as:

- ≥15% yellow → 'yellow'
- Else → 'red' or 'green' depending on which is dominant.

Trajectories of long-lived nuclei are plotted and coloured by their assigned identity

Output dataframes can be used for cell cycle analysis

![images/image2.png](images/image2.png)
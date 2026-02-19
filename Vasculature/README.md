# Vasculature Perfusion Analysis

This notebook (`perfusion.ipynb`) provides a **Napari GUI** for quantifying dextran permeability from `.lif` timelapse volumes.

Vasculature segmentation uses a Trainable Weka classifier workflow based on the approach from [Hajal et al.](https://www.nature.com/articles/s41596-021-00635-w#Sec44), with preprocessing in Fiji/ImageJ.

## What the notebook does

For each selected image in a `.lif` file:

1. Loads 3D timelapse data (`T,Z,Y,X` or `T,C,Z,Y,X`).
2. Segments vasculature from the **first selected dextran channel at t0** using:
     - 8-bit conversion
     - Otsu auto-threshold (stack)
     - 3D erosion
     - Weka classifier inference
3. Cleans segmentation by removing connected components with area < 20 voxels.
4. Uses the cleaned vasculature mask to quantify intensities:
     - inside vasculature
     - outside vasculature (gel)
     - at `t0` and `t2` (`t2` is time index 2 in code)
5. Computes geometric terms (surface area and volumes) after z-to-x rescaling.
6. Calculates permeability:

$$
p = \frac{1}{360} \cdot \frac{V_{gel}}{A_{vasc}} \cdot \frac{(b\,I_{gel,t2} - I_{gel,t0})}{(I_{vasc,t0} - I_{gel,t0})}
$$

where $b = I_{vasc,t0}/I_{vasc,t2}$ is the bleaching coefficient.

## GUI workflow

The app is created by instantiating `PerfusionNapariGUIApp()` and adds dock widgets in Napari:

- **Load images**
    - Select `.lif`
    - Select Weka classifier file
    - Click **Load images**
- **Dextran channel config**
    - Choose up to 3 dextran channels
    - Assign custom names per channel
    - Channel options are auto-populated from the `.lif` channel count
- **Analyse Single Image**
    - Select one image from dropdown
    - Runs quantification
    - Adds preview layers to Napari:
        - `Dextran <name> t0`
        - `Dextran <name> tfinal`
        - `Segmented Vasculature`
- **Analyse All Images**
    - Runs the same pipeline across all images in the loaded `.lif`
    - Saves results to CSV
- **Save This Result**
    - Saves currently accumulated results table to CSV

## Inputs

- A valid `.lif` file
- A valid Trainable Weka classifier file
- At least one selected dextran channel
- Images with at least 3 timepoints for permeability calculation (`t0` and `t2` used)

## Output table

The results dataframe/CSV can include:

- `lif_name`
- `image_name`
- `dextran_channel`
- `dextran_channel_index`
- `image_shape`
- `final_gel_intensity`
- `final_vascular_intensity`
- `initial_gel_intensity`
- `initial_vascular_intensity`
- `vascular_volume_um3`
- `gel_volume_um3`
- `vasculature_surface_area_um2`
- `bleaching_coefficient`
- `p_um/s`
- `p_cm/s`

If an image/channel cannot be processed, a `flag` column is written with an explanatory message (for example: missing voxel metadata, insufficient timepoints, or processing failure).

## Notes and assumptions

- Segmentation is always generated from the **first selected dextran channel**, then reused for all selected channels in that image.
- The code supports either single-channel (`T,Z,Y,X`) or multi-channel (`T,C,Z,Y,X`) image layouts.
- When channel counts vary across images in a `.lif`, unavailable channels are skipped per-image.
- Permeability uses `t0` and `t2` indices for calculation; Napari preview shows `t0` and final timepoint.

## Quick start

1. Open `perfusion.ipynb`.
2. Run the import cell and GUI cell.
3. In Napari, select `.lif` and classifier, then click **Load images**.
4. Set dextran channels/names.
5. Run **Analyse Single Image** (QC) or **Analyse All Images** (batch).
6. Save/inspect `perfusion_all_images.csv` and/or `perfusion_results.csv`.

## Example images

![Segmentation example](README_images/image1.png)

![Permeability example](README_images/image2.png)
# Vasculature Perfusion Analysis

This notebook (`perfusion.ipynb`) provides a **Napari GUI** for quantifying dextran permeability from `.lif` timelapse volumes **or a folder of `.tif`/`.tiff` files**.

Vasculature segmentation uses a Trainable Weka classifier workflow based on the approach from [Hajal et al.](https://www.nature.com/articles/s41596-021-00635-w#Sec44), with preprocessing in Fiji/ImageJ.

## What the notebook does

For each selected image (from a `.lif` file or a folder of TIFs):

1. Loads 3D timelapse data (`T,Z,Y,X` or `T,C,Z,Y,X`).
2. **Extracts pixel size metadata** (X, Y, Z in µm):
   - For `.lif` files: read from the embedded Leica metadata.
   - For `.tif`/`.tiff` files: tries OME-XML metadata, then ImageJ-style metadata, then standard TIFF resolution tags (`XResolution`, `YResolution`, `ResolutionUnit`).
3. Segments vasculature from the **first selected dextran channel at t0** using a Weka classifier from [Hajal et al.](https://www.nature.com/articles/s41596-021-00635-w#Sec44)
4. Cleans segmentation by removing connected components with area < 20 voxels.
5. Uses the cleaned vasculature mask to quantify intensities:
     - inside vasculature
     - outside vasculature (gel)
     - at `t0` and `tfinal`
6. Computes geometric terms (surface area and volumes) after z-to-x rescaling.
7. Calculates permeability:

$$
p = \frac{1}{t} \cdot \frac{V_{gel}}{A_{vasc}} \cdot \frac{(b\,I_{gel,t_{final}} - I_{gel,t0})}{(I_{vasc,t0} - I_{gel,t0})}
$$

where $b = I_{vasc,t0}/I_{vasc,t_{final}}$ is the bleaching coefficient and t is the time (seconds) between the first and last frame.

## GUI workflow

The app is created by instantiating `PerfusionNapariGUIApp()` and adds dock widgets in Napari:

- **Load images** — fill **one** of the two input fields (not both):
    - **Select .lif file** — pick a `.lif` file; the app lists all images inside it.
    - **Select folder of TIFs** — pick a folder; the app lists every `.tif`/`.tiff` in it and extracts pixel-size metadata for each file.
    - Select Weka classifier file
    - Click **Load images**
- **Dextran channel config**
    - Choose up to 3 dextran channels
    - Assign custom names per channel
    - Channel options are auto-populated from the image channel count
- **Analyse Single Image**
    - Select one image from dropdown
    - Runs quantification
    - Adds preview layers to Napari:
        - `Dextran <name> t0`
        - `Dextran <name> tfinal`
        - `Segmented Vasculature`
- **Analyse All Images**
    - Runs the same pipeline across all images in the loaded `.lif` or all TIFs in the folder
    - Saves results to CSV
- **Save This Result**
    - Saves currently accumulated results table to CSV

![Permeability example](README_images/image2.png)

## Inputs

- **Either** a valid `.lif` file **or** a folder containing `.tif`/`.tiff` files (only one should be filled)
- A valid Trainable Weka classifier file
- At least one selected dextran channel
- Images with at least 2 timepoints for permeability calculation

## TIF pixel-size extraction

When loading TIF files, the app automatically extracts X, Y, and Z pixel sizes (in µm) from metadata. It tries the following sources in order:

1. **OME-XML** — reads `PhysicalSizeX`, `PhysicalSizeY`, `PhysicalSizeZ` and their unit attributes.
2. **ImageJ metadata** — reads `spacing` (Z interval) and `unit` from the ImageJ description tag.
3. **Standard TIFF tags** — reads `XResolution`, `YResolution`, and `ResolutionUnit`, converting from inch or centimeter to µm as needed.

Extracted pixel sizes are logged in the GUI and used for surface-area / volume calculations. If pixel sizes cannot be determined, the permeability calculation is skipped for that image and a warning is logged.

## Notes and assumptions

- Segmentation is always generated from the **first selected dextran channel**, then reused for all selected channels in that image.
- When channel counts vary across images in a `.lif`, unavailable channels are skipped per-image.
- When using TIF input, selecting a `.lif` path clears the TIF folder field and vice versa to prevent conflicts.
- You need to download VS code, miniconda (add to PATH!), and the environment.yml file from this directory. When you first open VS code, open a terminal (command prompt) and run conda env create -f environment.yml       . Once it has loaded, run  conda activate nap-ij      (the name of the new environment).


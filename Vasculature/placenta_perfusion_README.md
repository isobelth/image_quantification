# Placenta Perfusion Analysis (Interactive Viewer)

This workflow quantifies dextran permeability across a placental barrier from Leica `.lif` perfusion datasets.

Unlike a “pure batch” script, the primary entry point is an interactive napari app embedded in the notebook:

- `placenta_perfusion_individal_viewing.ipynb`

It lets you load a `.lif`, choose an image + dextran channel, run segmentation, visually inspect the maternal/fetal masks + detected interface paths, optionally override the $z$ range, and export results to CSV.

Experimental layout:
![README_images/image2.png](README_images/image2.png)

Illustrative reconstruction:
![README_images/image3.png](README_images/image3.png)

## Before you start

Create the conda environment shipped with this repo:

1. Install conda/miniconda.
2. From the repo root, run:
    - `conda env create -f environment.yml`
3. Launch Jupyter in that environment and open `placenta_perfusion_individal_viewing.ipynb`.

## How to use the notebook (what you actually click)

1. Run notebook Cells 1 and 2 (imports + processor definitions).
2. Run Cell 3 to create the app (`app = PlacentaPerfusionApp()`). A napari window opens.
3. In the napari dock widgets (right side):

    - **Load images**
       - Choose a `.lif` file and click **Load images**.
       - The “Image” dropdown populates with all images inside the `.lif`.

    - **Segment + View**
       - Select an **Image** from the dropdown.
       - Select the **Dextran channel** (0-based channel index).
       - Click **Segment + View**.
       - The viewer shows:
          - `dextran_t0` (the $t_0$ 3D stack)
          - `maternal` and `fetal` label layers (if segmentation succeeds)
          - `paths_all` (detected interface paths per $z$)
       - A results table is also displayed in the notebook output.

    - **Run again with chosen z range**
       - After a segmentation, the min/max $z$ controls become enabled.
       - Use this if the automatic slice selection includes/excludes planes you disagree with.
       - Clicking reruns the measurements using only paths within the chosen range.

    - **Save results**
       - Exports the current in-notebook results table to a CSV you choose.

    - **Run all images to CSV**
       - Processes every image in the loaded `.lif` with the currently selected dextran channel.
       - Writes one CSV with one row per image (errors are recorded per-image).

## What the code computes (matches the notebook implementation)

### 1) Boundary detection at $t_0$

For each $z$ plane, the processor detects a left-to-right path representing the interface.

- It gates between “ridge mode” and “edge mode” using a vertical brightness ratio:
   - $\left|\log\left(\frac{\mathrm{median}(\mathrm{top})}{\mathrm{median}(\mathrm{bottom})}\right)\right|$
- In the notebook’s app, `RidgeParams(ratio_cutoff=0.9)` is used.

### 2) Slice selection

- Planes with no valid path are discarded.
- Additional filtering keeps a consistent band of similar planes (intended to remove out-of-focus planes and planes where paths route around an “impermeable channel”).
- If filtering removes everything but some paths exist, the app falls back to “use all found slices”.

### 3) Surface reconstruction + compartment segmentation

- The selected per-plane paths are interpolated into a smoothed surface.
- Maternal/fetal compartments are segmented by that surface.
- Volumes and interface area are computed, using voxel sizes read from the `.lif` metadata when available.

### 4) Timepoints used

The notebook expects 5D Leica data shaped like `(t, c, z, y, x)`.

- $t_0$ is always time index 0.
- $t_{final}$ is the last time index.
- If there are more than 3 timepoints, the app also computes a halfway timepoint:
   - $t_{1/2} = \lceil \tfrac{T}{2} \rceil$

The segmentation is computed from the $t_0$ image and then reused to measure intensities at later times.

### 5) Permeability calculation

The app assumes **180 seconds between frames** and converts the computed permeability from µm/s to cm/s.

For each time interval, it computes a bleaching coefficient from maternal intensity and then:

$$
P\;[\mathrm{cm/s}] = 10^{-4}\;\cdot\;\frac{1}{\Delta t}\;\cdot\;\frac{V_{fetal}}{A_{interface}}\;\cdot\;\frac{(B\cdot F_{final}) - F_{initial}}{M_{initial} - F_{initial}}
$$

Where:

- $B$ = maternal bleaching coefficient (e.g. $M_{t0}/M_{tfinal}$ for the $t0\to tfinal$ interval)
- $F$ = fetal total intensity
- $M$ = maternal total intensity
- $V_{fetal}$ = fetal volume
- $A_{interface}$ = interface area

## Inputs

- Leica `.lif` perfusion image stacks.
- A valid dextran tracer channel index (selected in the GUI).

Important: the current notebook implementation expects 5D time-series data `(t,c,z,y,x)` for segmentation + timepoint selection.

## Outputs

The GUI accumulates results in a per-image table and can export it to CSV. Columns include (at minimum):

- image identifiers: `lif_name`, `lif_path`, `image_index`, `image_name`
- intensities: maternal/fetal at $t_0$, $t_{1/2}$ (if present), and $t_{final}$
- permeability: `t0_tfinal_p`, `t0_t1_2_p`, `t1_2_tfinal_p` plus percent-change checks
- geometry: `maternal_um3`, `fetal_um3`, `interface_um2`, `n_planes_found`, `n_planes_kept`, kept $z$ min/max
- QC flags: `failed_geometry`, `failed_hump`, `hump_zs`, plus per-image `error` when something fails

## Quality control / troubleshooting

- Always visually check `paths_all`, `maternal`, and `fetal` layers after clicking **Segment + View**.
- If the automatic $z$ selection looks wrong, use **Run again with chosen z range**.
- If an image returns `error: no_kept_slices`, it means no usable boundary paths survived filtering.

## Limitations

- Assumes a roughly horizontal interface and minimal motion over time.
- Permeability is inferred from intensity dynamics; it depends on consistent imaging settings and bleaching behavior.



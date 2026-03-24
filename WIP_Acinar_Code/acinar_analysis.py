"""
Unified Acinar Analysis Module
==============================
Isobel Taylor-Hearn, 2023-2024

A single modular codebase for all 3D acinar image quantification:
  - Acinus shape (volume & roundness)
  - Cell & nuclear shape (requires segmentation masks)
  - Protein polarisation (BM intensity vs radial distance)
  - Apoptosis quantification (C3+ cell counting)
  - Protein proximity analysis (intensity near dying vs non-dying cells)
  - Proliferation analysis (EdU+ dividing vs non-dividing cells)
  - Mitochondria analysis (per-cell mito count, volume, distance from nucleus)

Usage
-----
Single image::

    from acinar_analysis import AcinarImage

    img = AcinarImage(
        "path/to/image.tif",
        nuclear_channel=0,
        membrane_channel=2,
        protein_channel=1,
    )

    # Run individual analyses
    df = img.acinus_shape()
    df = img.protein_polarisation()

    # Or run several at once
    results = img.run(["acinus_shape", "protein_polarisation"])

Batch::

    from acinar_analysis import batch_analyse
    df = batch_analyse(
        image_dir="path/to/images",
        analyses=["acinus_shape"],
        nuclear_channel=0,
        membrane_channel=2,
        output_csv="results.csv",
    )

CLI::

    python acinar_analysis.py --image-dir path/to/images --analyses acinus_shape --nuclear-channel 0 --membrane-channel 2
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import os
import pathlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from skimage import util
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_li, threshold_otsu, threshold_triangle
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import (
    ball,
    dilation,
    erosion,
    opening,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import clear_border, expand_labels, watershed
from skimage.transform import rescale
from tifffile import imread
from tqdm import tqdm

import tifffile as _tifffile

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
VALID_ANALYSES = {
    "acinus_shape",
    "cell_nuclear_shape",
    "protein_polarisation",
    "apoptosis",
    "protein_proximity",
    "proliferation",
    "mitochondria",
}

_UNSET = object()  # sentinel for "use instance default"

# ---------------------------------------------------------------------------
#  Utility helpers (stateless)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object, progress_callback=None):
    """Context manager so joblib Parallel updates a tqdm bar."""

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            if progress_callback is not None:
                progress_callback(tqdm_object.n, tqdm_object.total)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


def rescale_intensity(img: np.ndarray, target_min: float = 0, target_max: float = 255,
                      target_dtype=np.uint8) -> np.ndarray:
    """Linearly rescale *img* intensity to [target_min, target_max] and cast."""
    imin, imax = float(img.min()), float(img.max())
    if imax == imin:
        return np.full_like(img, target_min, dtype=target_dtype)
    a = (target_max - target_min) / (imax - imin)
    b = target_max - a * imax
    return (a * img + b).astype(target_dtype)


def read_pixel_size(tif_path: str) -> List[float]:
    """Return [x, y, z] pixel spacing in Âµm from TIFF metadata."""
    with _tifffile.TiffFile(str(tif_path)) as tif:
        tags: Dict[str, Any] = {}
        for tag in tif.pages[0].tags.values():
            tags[tag.name] = tag.value

        x = 1.0 / (tags["XResolution"][0] / tags["XResolution"][1])
        y = 1.0 / (tags["YResolution"][0] / tags["YResolution"][1])
        try:
            z = float(
                str(tags["IJMetadata"])
                .split("nscales=")[1]
                .split(",")[2]
                .split("\\nunit")[0]
            )
        except Exception:
            z = float(
                str(tags["ImageDescription"]).split("spacing=")[1].split("loop")[0]
            )
    return [x, y, z]


def add_image_details(df: pd.DataFrame, filename: str, flag: str) -> pd.DataFrame:
    """
    Add experimental details extracted from the filename to a DataFrame.

    Parses the filename to infer experimental details such as well number,
    imaging day, mechanical stiffness condition, and treatment type.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to which metadata will be added.
    filename : str
        The filename of the image (used to extract experimental details).
    flag : str
        A flag indicating any segmentation issues detected during processing.

    Returns
    -------
    pd.DataFrame
        The updated DataFrame with added metadata columns.
    """
    fn = filename.lower().replace("_", "")
    df["filename"] = filename
    df["flag"] = flag

    # Well number
    if "well1" in fn:
        df["well"] = 1
    elif "well2" in fn:
        df["well"] = 2
    else:
        df["well"] = np.nan

    # Day
    if "_d0" in filename.lower() or "d0" in fn:
        df["day"] = 0
    elif "d1" in fn:
        df["day"] = 1
    elif "_d3" in filename.lower() or "d3" in fn:
        df["day"] = 3
    else:
        df["day"] = 7

    # Cell type
    if "wt" in fn:
        df["cell_type"] = "WT"
    elif "bad" in fn:
        df["cell_type"] = "BADER"
    else:
        df["cell_type"] = "unknown"

    # Stiffness
    if "soft" in fn:
        df["condition"] = "soft"
    elif "stiff" in fn:
        df["condition"] = "stiff"
    else:
        df["condition"] = "blank"

    # Treatment
    if "bleb" in fn:
        df["treatment"] = "blebbistatin"
    elif "4oht" in fn:
        df["treatment"] = "4OHT"
    elif "rock" in fn or "y27632" in fn:
        df["treatment"] = "ROCKi"
    elif "batimastat" in fn or "mmpi" in fn:
        df["treatment"] = "batimastat"
    elif "abt" in fn:
        df["treatment"] = "ABT737"
    else:
        df["treatment"] = "untreated"

    df["image_type"] = df["condition"].astype(str) + ", d" + df["day"].astype(str)
    return df


# ---------------------------------------------------------------------------
#  Core acinus segmentation (shared by every analysis)
# ---------------------------------------------------------------------------

def segment_acinus(
    acinus_image: np.ndarray,
    spacing: List[float],
    scale: float = 0.25,
    smoothing_sigma: float = 3.0,
    clip_low_quantile: float = 0.1,
    clip_high_quantile: float = 0.85,
    min_sphericity: float = 0.55,
    qc_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> Tuple[np.ndarray, float, str]:
    """
    Segment the primary acinus from a combined intensity image.

    Returns
    -------
    acinus_mask : 3-D labelled array (background=0, acinus=1)
    new_pixel_size : isotropic voxel size in Âµm
    flag : string flag describing any issues
    """
    scale_z = spacing[2] / spacing[0]
    rescaled = rescale(acinus_image, scale=(scale * scale_z, scale, scale), anti_aliasing=False)
    new_pixel_size = spacing[0] / scale

    clipped = rescaled.clip(
        min=np.quantile(rescaled, clip_low_quantile),
        max=np.quantile(rescaled, clip_high_quantile),
    )
    smoothed = gaussian(clipped, sigma=smoothing_sigma)

    # --- Step 1: Li threshold, keep largest component ---
    thresh = threshold_li(smoothed)
    binary = smoothed > thresh
    binary = remove_small_holes(binary, area_threshold=100000)
    binary = remove_small_objects(binary, min_size=10000)

    labelled = label(binary)
    props = regionprops_table(labelled, properties=("label", "area"))
    keep = props["label"][np.argmax(props["area"])]
    acinus_mask = np.where(labelled == keep, 1, 0).astype(np.int32)
    threshold_method = "li"

    # --- Step 2: Per-slice 2D hole fill + keep largest per slice ---
    # Each slice is filled independently, then only the single largest
    # component is kept to avoid capturing proximal acini.
    for z in range(acinus_mask.shape[0]):
        filled = ndi.binary_fill_holes(acinus_mask[z]).astype(np.int32)
        lbl = label(filled)
        if lbl.max() > 1:
            rps = regionprops_table(lbl, properties=("label", "area"))
            keep_lbl = rps["label"][np.argmax(rps["area"])]
            filled = np.where(lbl == keep_lbl, 1, 0).astype(np.int32)
        acinus_mask[z] = filled

    # --- Step 3: Sphericity check  split merged acini if needed ---
    flag = "None"
    eigvals = regionprops_table(
        acinus_mask, properties=("label", "inertia_tensor_eigvals")
    )
    if len(eigvals["label"]) > 0:
        sphericity = eigvals["inertia_tensor_eigvals-2"][0] / eigvals["inertia_tensor_eigvals-0"][0]
        if sphericity < min_sphericity:
            flag = "multiple_acini_split"
            smoothed2 = gaussian(clipped, sigma=3)
            thresh2 = threshold_triangle(smoothed2)
            binary2 = smoothed2 > thresh2
            binary2 = remove_small_holes(binary2, area_threshold=100000)
            binary2 = remove_small_objects(binary2, min_size=10000)
            binary2 = erosion(binary2, ball(8))
            labelled2 = label(binary2)
            props2 = regionprops_table(labelled2, properties=("label", "area"))
            if len(props2["label"]) > 0:
                keep2 = props2["label"][np.argmax(props2["area"])]
                acinus_mask = np.where(labelled2 == keep2, 1, 0).astype(np.int32)
                acinus_mask = expand_labels(acinus_mask, distance=8)
                # Re-fill per slice after the split, keep largest only
                for z in range(acinus_mask.shape[0]):
                    filled = ndi.binary_fill_holes(acinus_mask[z]).astype(np.int32)
                    lbl = label(filled)
                    if lbl.max() > 1:
                        rps = regionprops_table(lbl, properties=("label", "area"))
                        keep_lbl = rps["label"][np.argmax(rps["area"])]
                        filled = np.where(lbl == keep_lbl, 1, 0).astype(np.int32)
                    acinus_mask[z] = filled

    # --- Step 4: Final erosion to offset any expansion from smoothing/filling ---
    acinus_mask = erosion(acinus_mask > 0, ball(5)).astype(np.int32)

    # Keep only the largest 3-D component (erosion can fragment thin bridges)
    lbl_post = label(acinus_mask)
    if lbl_post.max() > 1:
        rps_post = regionprops_table(lbl_post, properties=("label", "area"))
        keep_post = rps_post["label"][np.argmax(rps_post["area"])]
        acinus_mask = np.where(lbl_post == keep_post, 1, 0).astype(np.int32)

    # --- Compute final solidity for QC ---
    def _slice_solidity(mask):
        mid = mask.shape[0] // 2
        sols = []
        for off in [0, -2, 2, -4, 4]:
            z = mid + off
            if 0 <= z < mask.shape[0]:
                sl = mask[z] > 0
                if sl.any():
                    rps = regionprops(label(sl.astype(np.int32)))
                    if rps:
                        largest = max(rps, key=lambda r: r.area)
                        sols.append(largest.solidity)
        return float(np.median(sols)) if sols else 1.0

    final_solidity = _slice_solidity(acinus_mask)

    return acinus_mask, new_pixel_size, flag, threshold_method, final_solidity


# ---------------------------------------------------------------------------
#  Watershed helper (used by apoptosis + cell/nuclear shape)
# ---------------------------------------------------------------------------

def watershed_segment(
    binary_mask: np.ndarray,
    acinus_mask: np.ndarray,
    new_pixel_size: float,
    separation_um: float = 4.0,
    min_radius_um: float = 2.0,
    smooth_sigma: float = 0.8,
    erosion_radius: int = 3,
    opening_radius: int = 5,
    min_object_size: int = 1000,
    min_hole_size: int = 500,
) -> np.ndarray:
    """Clean a binary mask, restrict to acinus, and watershed-separate objects."""
    cleaned = binary_mask * acinus_mask
    cleaned = gaussian(cleaned, smooth_sigma)
    thresh = threshold_otsu(cleaned)
    cleaned = cleaned > thresh
    cleaned = remove_small_objects(cleaned, min_size=min_object_size)
    cleaned = remove_small_holes(cleaned, area_threshold=min_hole_size)
    cleaned = opening(cleaned, ball(opening_radius))

    distances = ndi.distance_transform_edt(erosion(cleaned, ball(erosion_radius)))
    coords = peak_local_max(distances, min_distance=max(1, int(separation_um / new_pixel_size)))
    markers = np.zeros(cleaned.shape, dtype=np.uint32)
    idx = tuple(np.round(coords).astype(int).T)
    markers[idx] = np.arange(len(coords)) + 1
    markers = dilation(markers, ball(2))
    seg = watershed(-distances, markers, mask=cleaned)
    seg = clear_border(seg)

    # Filter small objects
    props = regionprops_table(seg, properties=("label", "area"))
    vol_thresh = (4 / 3) * np.pi * (min_radius_um / new_pixel_size) ** 3
    keep = props["area"] >= vol_thresh
    return util.map_array(seg, props["label"], props["label"] * keep)


def _watershed_from_seeds(
    membrane_mask: np.ndarray,
    seed_labels: np.ndarray,
    acinus_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Watershed membrane based on nuclear seed positions.

    Returns (unexpanded_labels, expanded_labels_within_acinus).
    """
    props = regionprops_table(seed_labels, properties=("label", "centroid"))
    coords = np.stack(
        [props["centroid-0"], props["centroid-1"], props["centroid-2"]], axis=1
    ).astype(int)

    distances = ndi.distance_transform_edt(util.invert(membrane_mask))
    markers = np.zeros(membrane_mask.shape, dtype=np.uint32)
    idx = tuple(coords.T)
    markers[idx] = np.arange(len(coords)) + 1
    markers = dilation(markers, ball(2))
    seg = watershed(-distances, markers, mask=util.invert(membrane_mask))
    seg = clear_border(seg)
    expanded = expand_labels(seg, distance=12) * acinus_mask
    return seg, expanded


# ---------------------------------------------------------------------------
#  Neighbour analysis (cell/nuclear shape)
# ---------------------------------------------------------------------------

def find_neighbours(label_matrix: np.ndarray) -> pd.DataFrame:
    """
    Identify neighbouring labelled regions in a 3-D label matrix.

    Uses vectorised numpy shifts instead of a per-voxel Python loop,
    giving orders of magnitude speedup on typical volumes.

    Returns a DataFrame indexed by cell_label with columns
    ``sum`` (neighbour count), ``external``, ``internal``.
    """
    max_label = int(label_matrix.max())
    neighbours = np.zeros((max_label + 1, max_label + 1), dtype=np.uint8)

    # For each axis, compare each voxel with its forward neighbour.
    # Where they differ we record a pair of adjacent labels.
    for axis in range(3):
        a = np.take(label_matrix, range(0, label_matrix.shape[axis] - 1), axis=axis)
        b = np.take(label_matrix, range(1, label_matrix.shape[axis]), axis=axis)
        diff = a != b
        pairs_a = a[diff]
        pairs_b = b[diff]
        neighbours[pairs_a, pairs_b] = 1
        neighbours[pairs_b, pairs_a] = 1

    df = pd.DataFrame(neighbours)
    df = df.drop(labels=0, axis=1)
    df["sum"] = df.sum(axis=1)
    df = df[df["sum"] != 0]
    external_count = int(df["sum"].iloc[0]) if 0 in df.index else 0
    df["external"] = external_count
    df["internal"] = (len(np.unique(label_matrix)) - 1) - external_count
    df = df[["sum", "external", "internal"]]
    df.index.name = "cell_label"
    return df


def _match_nuclei_to_cells(
    labelled_nucleus: np.ndarray,
    labelled_cell: np.ndarray,
    new_pixel_size: float,
) -> pd.DataFrame:
    """Match each nucleus to its enclosing cell and compute morphological properties."""
    rows = []
    for region in regionprops(labelled_nucleus):
        z, x, y = int(region.centroid[0]), int(region.centroid[1]), int(region.centroid[2])
        rows.append([region.label, labelled_cell[z, x, y]])
    match_df = pd.DataFrame(rows, columns=["nucleus_label", "cell_label"])

    cell_p = pd.DataFrame(
        regionprops_table(labelled_cell, properties=("label", "area", "inertia_tensor_eigvals"))
    ).rename(columns={
        "label": "cell_label", "area": "cell_volume",
        "inertia_tensor_eigvals-0": "cell_eig0",
        "inertia_tensor_eigvals-1": "cell_eig1",
        "inertia_tensor_eigvals-2": "cell_eig2",
    })
    nuc_p = pd.DataFrame(
        regionprops_table(labelled_nucleus, properties=("label", "area", "inertia_tensor_eigvals"))
    ).rename(columns={
        "label": "nucleus_label", "area": "nucleus_volume",
        "inertia_tensor_eigvals-0": "nuc_eig0",
        "inertia_tensor_eigvals-1": "nuc_eig1",
        "inertia_tensor_eigvals-2": "nuc_eig2",
    })

    match_df = match_df.merge(cell_p).merge(nuc_p)
    vx3 = new_pixel_size ** 3
    match_df["nucleus_cell_volume_ratio"] = match_df["nucleus_volume"] / match_df["cell_volume"]
    match_df["nucleus_volume_um3"] = match_df["nucleus_volume"] * vx3
    match_df["cell_volume_um3"] = match_df["cell_volume"] * vx3
    match_df["cell_roundness"] = match_df["cell_eig2"] / match_df["cell_eig0"]
    match_df["nucleus_roundness"] = match_df["nuc_eig2"] / match_df["nuc_eig0"]
    match_df.drop(
        columns=["cell_volume", "nucleus_volume", "cell_eig0", "cell_eig1", "cell_eig2",
                 "nuc_eig0", "nuc_eig1", "nuc_eig2"],
        inplace=True,
    )
    return match_df


# ===========================================================================
#  AcinarImage, main class
# ===========================================================================

class AcinarImage:
    """
    Represents a single 3D acinar microscopy image.

    Loads image data and metadata once, then exposes analysis methods
    that share the cached state.

    Parameters
    ----------
    image_path : str or Path
        Path to the multi-channel TIFF stack.
    nuclear_channel : int
        Nuclear channel index.
    membrane_channel : int or None
        Membrane channel index (None = no membrane channel).
    protein_channel : int or None
        Channel for protein polarisation analysis.
    c3_channel : int
        Channel for C3 / apoptosis analyses.
    edu_channel : int or None
        Channel for EdU / proliferation analysis.
    proximity_protein_channel : int or None
        Channel for the proximity-protein analysis.
    nuclear_mask_path, membrane_mask_path : str or None
        Paths to binary segmentation masks (for cell_nuclear_shape).
    c3_mask_path : str or None
        Path to binary C3 segmentation mask (for apoptosis / protein_proximity).
    edu_mask_path : str or None
        Path to binary EdU segmentation mask (for proliferation).
    mito_channel : int or None
        Channel for mitochondria analysis.
    mito_mask_path : str or None
        Path to binary mitochondria segmentation mask.
    extra_acinus_channels : list of int or None
        Additional channels to include in acinus approximation.
    """

    def __init__(
        self,
        image_path,
        *,
        nuclear_channel=0,
        membrane_channel=2,
        protein_channel=None,
        c3_channel=3,
        edu_channel=None,
        mito_channel=None,
        proximity_protein_channel=None,
        nuclear_mask_path=None,
        membrane_mask_path=None,
        c3_mask_path=None,
        edu_mask_path=None,
        mito_mask_path=None,
        extra_acinus_channels=None,
    ):
        self.image_path = str(image_path)
        self.filename = os.path.basename(self.image_path).lower()

        # Channel configuration
        self.nuclear_channel = nuclear_channel
        self.membrane_channel = membrane_channel
        self.protein_channel = protein_channel
        self.c3_channel = c3_channel
        self.edu_channel = edu_channel
        self.mito_channel = mito_channel
        self.proximity_protein_channel = proximity_protein_channel
        self.extra_acinus_channels = extra_acinus_channels
        self.qc_dir = None  # set externally or via batch_analyse
        self.return_volumes = False  # when True, populate self.volumes
        self.volumes: Dict[str, np.ndarray] = {}  # name -> 3D array

        # Mask paths
        self.nuclear_mask_path = nuclear_mask_path
        self.membrane_mask_path = membrane_mask_path
        self.c3_mask_path = c3_mask_path
        self.edu_mask_path = edu_mask_path
        self.mito_mask_path = mito_mask_path

        # Lazy-loaded shared state
        self._image = None
        self._spacing = None
        self._segment_cache: Dict[tuple, tuple] = {}  # (extra_chs, sigma) -> (mask, px, flag)
        self._mask_raw_cache: Dict[str, np.ndarray] = {}  # path -> raw array
        self._mask_rescaled_cache: Dict[str, np.ndarray] = {}  # path -> rescaled array
        self._acinus_approx_cache: Dict[tuple, np.ndarray] = {}  # extra_chs_key -> array

    # -- Lazy properties --------------------------------------------------

    @property
    def image(self):
        """Intensity-rescaled multi-channel image (loaded once)."""
        if self._image is None:
            self._image = rescale_intensity(imread(self.image_path))
        return self._image

    @property
    def spacing(self):
        """[x, y, z] pixel spacing in Âµm (read once from TIFF metadata)."""
        if self._spacing is None:
            self._spacing = read_pixel_size(self.image_path)
        return self._spacing

    @property
    def scale_z(self):
        """Z-to-XY scale ratio derived from spacing."""
        return self.spacing[2] / self.spacing[0]

    # -- Internal helpers -------------------------------------------------

    def _rescale_volume(self, volume, scale=0.25):
        """Rescale a 3-D volume to isotropic voxels at *scale*."""
        return rescale(volume, (scale * self.scale_z, scale, scale), anti_aliasing=False)

    # -- QC plotting -------------------------------------------------------

    def _rgb_mid_z(self, red_channel: Optional[int] = None) -> np.ndarray:
        """Build an RGB composite of the mid-Z slice at rescaled resolution.

        B = nuclear channel, G = membrane channel,
        R = *red_channel* (a single analysis-specific channel).

        The image is rescaled to the same 0.25-scale isotropic resolution
        used by the acinus mask so that overlays align correctly.
        """
        nuc_ch = self.nuclear_channel
        mem_ch = self.membrane_channel

        def _rescale_ch(ch_idx):
            return self._rescale_volume(self.image[:, ch_idx, :, :])

        def _norm(arr):
            mn, mx = float(arr.min()), float(arr.max())
            if mx == mn:
                return np.zeros_like(arr, dtype=np.float64)
            return (arr.astype(np.float64) - mn) / (mx - mn)

        nuc_vol = _rescale_ch(nuc_ch)
        mid = nuc_vol.shape[0] // 2

        blue = _norm(nuc_vol[mid])
        if mem_ch is not None:
            green = _norm(_rescale_ch(mem_ch)[mid])
        else:
            green = np.zeros_like(blue)

        if red_channel is not None:
            red = _norm(_rescale_ch(red_channel)[mid])
        else:
            red = np.zeros_like(blue)

        return np.stack([red, green, blue], axis=-1).clip(0, 1)

    def _save_qc(self, analysis_name: str,
                 overlays: List[Tuple[np.ndarray, str]], title_extra: str = "",
                 red_channel: Optional[int] = None):
        """Save a QC figure with RGB composite + label overlays at the mid-Z slice.

        Parameters
        ----------
        analysis_name : str
            Used in the output filename.
        overlays : list of (2-D array, name[, is_label])
            Each overlay is drawn as a coloured region (label) or cyan contour (mask).
        title_extra : str
            Extra text appended to the figure title.
        red_channel : int or None
            Channel index for the red layer of the RGB background.
        """
        if self.qc_dir is None:
            return

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        qc_path = pathlib.Path(self.qc_dir)
        qc_path.mkdir(parents=True, exist_ok=True)

        n_panels = 1 + len(overlays)
        fig = Figure(figsize=(5 * n_panels, 5))
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, n_panels, squeeze=False)[0]

        stem = pathlib.Path(self.image_path).stem

        # Use RGB composite if available, otherwise fall back to grayscale
        rgb = self._rgb_mid_z(red_channel=red_channel)

        # Raw image panel
        axes[0].imshow(rgb)
        r_label = f"ch{red_channel}" if red_channel is not None else "--"
        axes[0].set_title(f"Raw (R={r_label}, G=mem, B=nuc)")
        axes[0].axis("off")

        # Overlay panels
        for ax, item in zip(axes[1:], overlays):
            overlay, name = item[0], item[1]
            is_label = item[2] if len(item) > 2 else (overlay.max() > 1)
            ax.imshow(rgb)
            if is_label:  # label image
                from matplotlib.colors import ListedColormap
                _qc_colours = ["red", "darkorange", "yellow", "limegreen",
                               "dodgerblue", "darkviolet", "deeppink"]
                _label_cmap = ListedColormap(_qc_colours)
                masked = np.ma.masked_where(overlay == 0, overlay)
                ax.imshow(masked, cmap=_label_cmap, alpha=0.5, interpolation="nearest")
            else:  # binary mask
                ax.contour(overlay, levels=[0.5], colors="darkorange", linewidths=0.8)
            ax.set_title(name)
            ax.axis("off")

        fig.suptitle(f"{stem} \u2014 {analysis_name} {title_extra}", fontsize=10)
        fig.tight_layout()
        out = qc_path / f"{stem}_{analysis_name}_qc.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")

    def _mid_z(self, vol: np.ndarray) -> np.ndarray:
        """Return the middle Z-slice of a 3-D volume."""
        return vol[vol.shape[0] // 2]

    def _build_acinus_approx(self, nuclear_ch=_UNSET, membrane_ch=_UNSET,
                             extra_channels=_UNSET):
        """Sum selected channels to approximate acinus extent (cached).

        Automatically includes ``c3_channel`` and ``edu_channel`` when they
        are set on the instance so that fluorescent signal from those stains
        contributes to the acinus segmentation.
        """
        if nuclear_ch is _UNSET:
            nuclear_ch = self.nuclear_channel
        if membrane_ch is _UNSET:
            membrane_ch = self.membrane_channel
        if extra_channels is _UNSET:
            extra_channels = list(self.extra_acinus_channels) if self.extra_acinus_channels else []
        else:
            extra_channels = list(extra_channels) if extra_channels else []

        # Always include C3 / EdU / mito channels when available
        for ch in (self.c3_channel, self.edu_channel, self.mito_channel):
            if ch is not None and ch != nuclear_ch and ch != membrane_ch and ch not in extra_channels:
                extra_channels.append(ch)

        key = (nuclear_ch, membrane_ch,
               tuple(sorted(extra_channels)) if extra_channels else ())
        if key in self._acinus_approx_cache:
            return self._acinus_approx_cache[key]

        combined = rescale_intensity(self.image[:, nuclear_ch, :, :]).astype(np.float64)
        if membrane_ch is not None:
            combined += rescale_intensity(self.image[:, membrane_ch, :, :]).astype(np.float64)
        if extra_channels:
            for ch in extra_channels:
                combined += rescale_intensity(self.image[:, ch, :, :]).astype(np.float64)
        self._acinus_approx_cache[key] = combined
        return combined

    def _segment(self, acinus_approx, smoothing_sigma=3.0):
        """Segment acinus and return (mask, pixel_size, flag, threshold_method, solidity)."""
        return segment_acinus(acinus_approx, self.spacing,
                              smoothing_sigma=smoothing_sigma,
                              qc_dir=self.qc_dir,
                              filename=self.filename)

    def _get_acinus_mask(self):
        """Cached acinus segmentation — single unified mask for all analyses."""
        if not self._segment_cache:
            approx = self._build_acinus_approx()
            self._segment_cache["unified"] = self._segment(approx, 3.0)
        return self._segment_cache["unified"]

    def _load_mask_raw(self, path):
        """Load a mask image from disk, cached by path."""
        path = str(path)
        if path not in self._mask_raw_cache:
            self._mask_raw_cache[path] = imread(path)
        return self._mask_raw_cache[path]

    def _load_mask_rescaled(self, path):
        """Load and rescale a mask image, cached by path."""
        path = str(path)
        if path not in self._mask_rescaled_cache:
            raw = self._load_mask_raw(path)
            self._mask_rescaled_cache[path] = self._rescale_volume(raw)
        return self._mask_rescaled_cache[path]

    # =====================================================================
    #  Analysis methods
    # =====================================================================

    def acinus_shape(self):
        """Calculate acinus volume (ÂµmÂ³) and roundness."""
        acinus_mask, px, flag, threshold_method, solidity = self._get_acinus_mask()

        regions = regionprops(acinus_mask)
        if not regions:
            return pd.DataFrame({"acinus_volume_um3": [np.nan], "acinus_roundness": [np.nan],
                                 "flag": ["no_acinus"]})

        r = regions[0]
        vol = r.area * px ** 3
        roundness = r.inertia_tensor_eigvals[2] / r.inertia_tensor_eigvals[0]

        hole_sizes = [rr.area for rr in regionprops(label(util.invert(acinus_mask > 0)))]
        if len(hole_sizes) > 1 and flag == "None":
            flag = "hole"

        # QC plot
        self._save_qc("acinus_shape",
                      [(self._mid_z(acinus_mask), "Acinus mask")],
                      title_extra=f"vol={vol:.0f} ÂµmÂ³, round={roundness:.2f}, thresh={threshold_method}, solidity={solidity:.2f}",
                      red_channel=self.membrane_channel)

        if self.return_volumes:
            self.volumes["acinus_mask"] = acinus_mask

        return pd.DataFrame({"acinus_volume_um3": [vol], "acinus_roundness": [roundness], "flag": [flag]})

    def cell_nuclear_shape(self):
        """Segment cells/nuclei and compute volume, roundness, and neighbour info.

        Requires ``nuclear_mask_path`` and ``membrane_mask_path`` to be set.
        """
        if self.nuclear_mask_path is None or self.membrane_mask_path is None:
            raise ValueError(
                "cell_nuclear_shape requires both 'nuclear_mask_path' and "
                "'membrane_mask_path'. Set them on the AcinarImage instance."
            )

        acinus_mask, px, flag, _thresh, _sol = self._get_acinus_mask()

        rescaled_nuc = self._load_mask_rescaled(self.nuclear_mask_path)
        rescaled_mem = self._load_mask_rescaled(self.membrane_mask_path)

        # Restrict to acinus
        rescaled_nuc = rescaled_nuc * acinus_mask
        rescaled_mem = rescaled_mem * acinus_mask

        # --- Segment nuclei via watershed ---
        cleaned_nuc = gaussian(rescaled_nuc, 1)
        thresh = threshold_otsu(cleaned_nuc)
        cleaned_nuc = cleaned_nuc > thresh
        cleaned_nuc = remove_small_holes(cleaned_nuc, area_threshold=1000)

        distances = ndi.distance_transform_edt(erosion(cleaned_nuc, ball(3)))
        coords = peak_local_max(distances, min_distance=max(1, int(4 / px)))
        markers = np.zeros(cleaned_nuc.shape, dtype=np.uint32)
        idx = tuple(np.round(coords).astype(int).T)
        markers[idx] = np.arange(len(coords)) + 1
        markers = dilation(markers, ball(2))
        seg_nuc = watershed(-distances, markers, mask=cleaned_nuc)
        seg_nuc = clear_border(seg_nuc)

        # Filter small nuclei
        props = regionprops_table(seg_nuc, properties=("label", "area"))
        vol_thresh = (4 / 3) * np.pi * (2 / px) ** 3
        keep = props["area"] >= vol_thresh
        seg_nuc = util.map_array(seg_nuc, props["label"], props["label"] * keep)

        # --- Segment membranes via watershed seeded by nuclei ---
        cleaned_mem = gaussian(rescaled_mem, sigma=1)
        thresh_m = threshold_otsu(cleaned_mem)
        cleaned_mem = cleaned_mem > thresh_m
        _, seg_mem_exp = _watershed_from_seeds(cleaned_mem, seg_nuc, acinus_mask)

        # --- Match nuclei to cells and compute properties ---
        matching = _match_nuclei_to_cells(seg_nuc, seg_mem_exp, px)
        matching = matching.merge(find_neighbours(seg_mem_exp).reset_index())
        matching["flag"] = flag

        if self.spacing == [1, 1, 1]:
            matching["flag"] = "wrong_metadata"

        # QC plot
        self._save_qc("cell_nuclear_shape", [
            (self._mid_z(seg_nuc), "Nuclei labels"),
            (self._mid_z(seg_mem_exp), "Cell labels"),
        ], red_channel=self.membrane_channel)

        if self.return_volumes:
            self.volumes["acinus_mask"] = acinus_mask
            self.volumes["nuclei_labels"] = seg_nuc
            self.volumes["cell_labels"] = seg_mem_exp

        return matching

    def protein_polarisation(self):
        """Quantify protein intensity as a function of normalised radial distance.

        Requires ``protein_channel`` to be set.
        """
        if self.protein_channel is None:
            raise ValueError(
                "protein_polarisation requires 'protein_channel' to be set."
            )

        acinus_mask, px, flag, _thresh, _sol = self._get_acinus_mask()

        protein_rescaled = self._rescale_volume(
            self.image[:, self.protein_channel, :, :]
        )
        expanded_acinus_mask = expand_labels(acinus_mask, distance=5)
        protein_rescaled = protein_rescaled * expanded_acinus_mask

        # Distance map normalised by equivalent sphere radius
        distance = ndi.distance_transform_edt(acinus_mask)
        regions = regionprops(acinus_mask)
        if not regions:
            return pd.DataFrame()
        r = np.cbrt((3 * regions[0].area) / (4 * np.pi))
        distance = distance / r

        df = pd.DataFrame({
            "distance_over_radius": np.ravel(distance),
            "protein_intensity": np.ravel(protein_rescaled),
        })
        df["rounded_distance"] = df["distance_over_radius"].round(2)
        df = df.groupby("rounded_distance", as_index=False)["protein_intensity"].mean()
        df["flag"] = flag

        # QC plot
        self._save_qc("protein_polarisation", [
            (self._mid_z(acinus_mask), "Acinus mask"),
        ], red_channel=self.protein_channel)

        if self.return_volumes:
            self.volumes["acinus_mask"] = acinus_mask
            self.volumes["protein"] = protein_rescaled

        return df

    def apoptosis(self, c3_separation_um=7.0, c3_min_radius_um=1.3,
                  nuclear_separation_um=6.0, nuclear_min_radius_um=2.0):
        """Count C3-positive (apoptotic) cells and total nuclei per acinus.

        Requires ``c3_mask_path`` and ``nuclear_mask_path`` to be set.
        """
        if self.c3_mask_path is None or self.nuclear_mask_path is None:
            raise ValueError(
                "apoptosis requires both 'c3_mask_path' and 'nuclear_mask_path'."
            )

        acinus_mask, px, flag, _thresh, _sol = self._get_acinus_mask()

        c3_mask = self._load_mask_rescaled(self.c3_mask_path)
        nuclear_mask = self._load_mask_rescaled(self.nuclear_mask_path)

        c3_labels = watershed_segment(c3_mask, acinus_mask, px,
                                      c3_separation_um, c3_min_radius_um)
        nuclear_labels = watershed_segment(nuclear_mask, acinus_mask, px,
                                        nuclear_separation_um, nuclear_min_radius_um)

        # Distance map (normalised 0-1)
        dist = ndi.distance_transform_edt(acinus_mask > 0) * px
        dist_scaled = np.interp(dist, (dist.min(), dist.max()), (0, 1))

        # Acinus-level measurements
        acinus_regions = regionprops(acinus_mask)
        acinus_vol = acinus_regions[0].area * px ** 3 if acinus_regions else np.nan
        acinus_round = (
            acinus_regions[0].inertia_tensor_eigvals[2]
            / acinus_regions[0].inertia_tensor_eigvals[0]
            if acinus_regions else np.nan
        )

        c3_props = pd.DataFrame(
            regionprops_table(c3_labels, properties=("label", "area", "centroid"))
        )
        if c3_props.empty:
            c3_props = pd.DataFrame({
                "label": ["no_c3"], "centroid-0": [np.nan], "centroid-1": [np.nan],
                "centroid-2": [np.nan], "c3_volume_um3": [np.nan],
                "normalised_distance": [np.nan],
            })
        else:
            c3_props["c3_volume_um3"] = c3_props["area"] * px ** 3
            c3_props["normalised_distance"] = c3_props.apply(
                lambda row: dist_scaled[
                    int(round(row["centroid-0"])),
                    int(round(row["centroid-1"])),
                    int(round(row["centroid-2"])),
                ],
                axis=1,
            )
            c3_props.drop("area", axis=1, inplace=True)

        c3_props["acinus_volume_um3"] = acinus_vol
        c3_props["acinus_roundness"] = acinus_round
        c3_props["number_of_nuclei"] = pd.DataFrame(
            regionprops_table(nuclear_labels, properties=("label",))
        ).shape[0]
        c3_props["flag"] = flag

        # QC plot
        self._save_qc("apoptosis", [
            (self._mid_z(acinus_mask), "Acinus mask"),
            (self._mid_z(c3_labels), "C3 labels", True),
            (self._mid_z(nuclear_labels), "Nuclear labels", True),
        ], red_channel=self.c3_channel)

        if self.return_volumes:
            self.volumes["acinus_mask"] = acinus_mask
            self.volumes["c3_labels"] = c3_labels
            self.volumes["nuclear_labels"] = nuclear_labels

        return c3_props

    def protein_proximity(self, search_radius_um=5.0,
                          c3_separation_um=7.0, c3_min_radius_um=3.0,
                          nuclear_separation_um=6.0, nuclear_min_radius_um=3.0):
        """Compare protein intensity near dying vs non-dying cells.

        Requires ``c3_mask_path``, ``nuclear_mask_path``, and
        ``proximity_protein_channel`` to be set.
        """
        if self.c3_mask_path is None or self.nuclear_mask_path is None:
            raise ValueError(
                "protein_proximity requires both 'c3_mask_path' and 'nuclear_mask_path'."
            )
        if self.proximity_protein_channel is None:
            raise ValueError(
                "protein_proximity requires 'proximity_protein_channel' to be set."
            )

        acinus_mask, px, flag, _thresh, _sol = self._get_acinus_mask()

        c3_raw = self._load_mask_raw(self.c3_mask_path)
        nuclear_raw = self._load_mask_raw(self.nuclear_mask_path)
        c3_mask = self._load_mask_rescaled(self.c3_mask_path)
        # Live cells = nuclear mask minus dilated C3
        live_cells = self._rescale_volume(nuclear_raw - dilation(c3_raw, ball(2)))
        live_cells = np.where(live_cells > 1, 0, live_cells)

        c3_labels = watershed_segment(c3_mask, acinus_mask, px,
                                      c3_separation_um, c3_min_radius_um)
        live_labels = watershed_segment(live_cells, acinus_mask, px,
                                        nuclear_separation_um, nuclear_min_radius_um)

        # Distance map (normalised 0-1)
        dist = ndi.distance_transform_edt(acinus_mask > 0)
        dist_scaled = np.interp(dist, (dist.min(), dist.max()), (0, 1))

        acinus_regions = regionprops(acinus_mask)
        acinus_vol = acinus_regions[0].area * px ** 3 if acinus_regions else np.nan
        acinus_round = (
            acinus_regions[0].inertia_tensor_eigvals[2]
            / acinus_regions[0].inertia_tensor_eigvals[0]
            if acinus_regions else np.nan
        )

        # Rescale proximity protein
        prox_img = self._rescale_volume(
            self.image[:, self.proximity_protein_channel, :, :]
        )

        # Build dying / non-dying info
        dying_info = pd.DataFrame(
            regionprops_table(c3_labels, properties=("label", "area", "centroid"))
        )
        dying_info["nuclear_or_c3_volume_um3"] = dying_info["area"] * px ** 3
        dying_info.drop("area", axis=1, inplace=True)
        dying_info["dying"] = "Y"

        live_info = pd.DataFrame(
            regionprops_table(live_labels, properties=("label", "area", "centroid"))
        )
        live_info["nuclear_or_c3_volume_um3"] = live_info["area"] * px ** 3
        live_info.drop("area", axis=1, inplace=True)
        live_info["dying"] = "N"

        all_cells = pd.concat([dying_info, live_info], ignore_index=True)
        all_cells["acinus_volume_um3"] = acinus_vol
        all_cells["acinus_roundness"] = acinus_round
        all_cells["number_dying"] = dying_info.shape[0]
        all_cells["number_not_dying"] = live_info.shape[0]

        if not all_cells.empty and "centroid-0" in all_cells.columns:
            all_cells["normalised_distance"] = all_cells.apply(
                lambda row: dist_scaled[
                    int(round(row["centroid-0"])),
                    int(round(row["centroid-1"])),
                    int(round(row["centroid-2"])),
                ]
                if pd.notna(row["centroid-0"])
                else np.nan,
                axis=1,
            )

        # --- Combine labels and measure proximity protein ---
        combined = live_labels.copy()
        if c3_labels.max() > 0:
            offset = live_labels.max()
            c3_offset = np.where(c3_labels > 0, c3_labels + offset, 0)
            combined = np.where(combined == 0, c3_offset, combined)
        expanded_acinus_mask = expand_labels(acinus_mask, distance=5)
        estimated_territories = expanded_acinus_mask * expand_labels(combined, distance=20)

        search_px = max(1, int(search_radius_um / px))
        prox_rows = []
        for region in regionprops(estimated_territories):
            cell_mask = estimated_territories == region.label
            expanded = expand_labels(cell_mask.astype(np.uint8), distance=search_px) > 0
            in_cell = float(prox_img[cell_mask].sum())
            in_nbhd = float(prox_img[expanded].sum())
            cell_vol = int(cell_mask.sum())
            nbhd_vol = int(expanded.sum())
            prox_rows.append({
                "label": region.label,
                "proximity_intensity_in_cell": in_cell,
                "proximity_intensity_around_cell": in_nbhd - in_cell,
                "proximity_intensity_total_neighborhood": in_nbhd,
                "proximity_mean_intensity_in_cell": in_cell / cell_vol if cell_vol else 0,
                "proximity_mean_intensity_neighborhood": in_nbhd / nbhd_vol if nbhd_vol else 0,
                "estimated_cell_territory_volume_um3": cell_vol * px ** 3,
                "proximity_neighborhood_volume_um3": nbhd_vol * px ** 3,
            })

        if prox_rows:
            all_cells = all_cells.merge(pd.DataFrame(prox_rows), on="label", how="left")

        all_cells["flag"] = flag

        # QC plot
        self._save_qc("protein_proximity", [
            (self._mid_z(acinus_mask), "Acinus mask"),
            (self._mid_z(c3_labels), "Dying (C3)", True),
            (self._mid_z(live_labels), "Non-dying", True),
        ], red_channel=self.proximity_protein_channel)

        if self.return_volumes:
            self.volumes["acinus_mask"] = acinus_mask
            self.volumes["c3_labels"] = c3_labels
            self.volumes["live_labels"] = live_labels

        return all_cells

    def proliferation(self, edu_separation_um=4.0, edu_min_radius_um=2.0,
                      nuclear_separation_um=4.0, nuclear_min_radius_um=2.0):
        """Count EdU-positive (dividing) vs non-dividing cells per acinus.

        Dividing cells are identified from the EdU mask.  Non-dividing cells
        are nuclear-positive but EdU-negative (nuclear mask minus dilated EdU mask).

        Requires ``edu_mask_path`` and ``nuclear_mask_path`` to be set.
        The ``edu_channel`` is used (along with nuclear channel) to build the acinus
        approximation.
        """
        if self.edu_mask_path is None or self.nuclear_mask_path is None:
            raise ValueError(
                "proliferation requires both 'edu_mask_path' and 'nuclear_mask_path'."
            )
        if self.edu_channel is None:
            raise ValueError(
                "proliferation requires 'edu_channel' to be set."
            )

        acinus_mask, px, flag, _thresh, _sol = self._get_acinus_mask()

        edu_mask = self._load_mask_rescaled(self.edu_mask_path)
        nuclear_raw = self._load_mask_raw(self.nuclear_mask_path)
        edu_raw = self._load_mask_raw(self.edu_mask_path)

        # Non-dividing = nuclear mask minus dilated EdU
        non_dividing_mask = self._rescale_volume(
            nuclear_raw - dilation(edu_raw, ball(2))
        )
        non_dividing_mask = np.where(non_dividing_mask > 1, 0, non_dividing_mask)

        dividing_labels = watershed_segment(
            edu_mask, acinus_mask, px, edu_separation_um, edu_min_radius_um
        )
        non_dividing_labels = watershed_segment(
            non_dividing_mask, acinus_mask, px, nuclear_separation_um, nuclear_min_radius_um
        )

        # Distance map (normalised 0-1)
        dist = ndi.distance_transform_edt(acinus_mask > 0)
        dist_scaled = np.interp(dist, (dist.min(), dist.max()), (0, 1))

        # Acinus-level measurements
        acinus_regions = regionprops(acinus_mask)
        acinus_vol = acinus_regions[0].area * px ** 3 if acinus_regions else np.nan
        acinus_round = (
            acinus_regions[0].inertia_tensor_eigvals[2]
            / acinus_regions[0].inertia_tensor_eigvals[0]
            if acinus_regions else np.nan
        )

        dividing_info = pd.DataFrame(
            regionprops_table(dividing_labels, properties=("label", "area", "centroid"))
        )
        dividing_info["dividing"] = "Y"
        number_dividing = dividing_info.shape[0]

        non_dividing_info = pd.DataFrame(
            regionprops_table(non_dividing_labels, properties=("label", "area", "centroid"))
        )
        non_dividing_info["dividing"] = "N"
        number_not_dividing = non_dividing_info.shape[0]

        all_cells = pd.concat([dividing_info, non_dividing_info], ignore_index=True)
        all_cells["acinus_volume_um3"] = acinus_vol
        all_cells["acinus_roundness"] = acinus_round
        all_cells["number_dividing"] = number_dividing
        all_cells["number_not_dividing"] = number_not_dividing

        if all_cells.empty:
            all_cells = pd.DataFrame({
                "label": [np.nan], "centroid-0": [np.nan],
                "centroid-1": [np.nan], "centroid-2": [np.nan],
                "normalised_distance": [np.nan],
                "dividing": [np.nan],
            })
        else:
            all_cells["cell_volume_um3"] = all_cells["area"] * px ** 3
            all_cells.drop("area", axis=1, inplace=True)
            all_cells["normalised_distance"] = all_cells.apply(
                lambda row: dist_scaled[
                    int(round(row["centroid-0"])),
                    int(round(row["centroid-1"])),
                    int(round(row["centroid-2"])),
                ]
                if pd.notna(row["centroid-0"])
                else np.nan,
                axis=1,
            )

        all_cells["flag"] = flag

        # QC plot
        self._save_qc("proliferation", [
            (self._mid_z(acinus_mask), "Acinus mask"),
            (self._mid_z(dividing_labels), "Dividing (EdU+)", True),
            (self._mid_z(non_dividing_labels), "Non-dividing", True),
        ], red_channel=self.edu_channel)

        if self.return_volumes:
            self.volumes["acinus_mask"] = acinus_mask
            self.volumes["dividing_labels"] = dividing_labels
            self.volumes["non_dividing_labels"] = non_dividing_labels

        return all_cells

    def mitochondria(self, mito_min_object_size=10):
        """Per-cell mitochondria count, volume, and distance from nucleus.

        Requires ``nuclear_mask_path``, ``membrane_mask_path``, and
        ``mito_mask_path`` to be set.
        """
        if self.nuclear_mask_path is None or self.membrane_mask_path is None:
            raise ValueError(
                "mitochondria requires 'nuclear_mask_path' and 'membrane_mask_path'."
            )
        if self.mito_mask_path is None:
            raise ValueError(
                "mitochondria requires 'mito_mask_path' to be set."
            )

        acinus_mask, px, flag, _thresh, _sol = self._get_acinus_mask()

        # Rescale masks
        rescaled_nuc = self._load_mask_rescaled(self.nuclear_mask_path)
        rescaled_mem = self._load_mask_rescaled(self.membrane_mask_path)
        rescaled_nuc = rescaled_nuc * acinus_mask
        rescaled_mem = rescaled_mem * acinus_mask

        # Rescale and label the mito mask
        mito_raw = self._load_mask_raw(self.mito_mask_path)
        mito_labelled = label(mito_raw)
        mito_labelled = remove_small_objects(mito_labelled, min_size=mito_min_object_size)
        mito_labelled = self._rescale_volume(
            mito_labelled.astype(np.uint16), scale=0.25
        ).astype(np.int32)

        # --- Segment nuclei via watershed (same as cell_nuclear_shape) ---
        cleaned_nuc = gaussian(rescaled_nuc, 1)
        thresh = threshold_otsu(cleaned_nuc)
        cleaned_nuc = cleaned_nuc > thresh
        cleaned_nuc = remove_small_holes(cleaned_nuc, area_threshold=1000)

        distances_nuc = ndi.distance_transform_edt(erosion(cleaned_nuc, ball(3)))
        coords = peak_local_max(distances_nuc, min_distance=max(1, int(4 / px)))
        markers = np.zeros(cleaned_nuc.shape, dtype=np.uint32)
        idx = tuple(np.round(coords).astype(int).T)
        markers[idx] = np.arange(len(coords)) + 1
        markers = dilation(markers, ball(2))
        seg_nuc = watershed(-distances_nuc, markers, mask=cleaned_nuc)
        seg_nuc = clear_border(seg_nuc)

        props = regionprops_table(seg_nuc, properties=("label", "area"))
        vol_thresh = (4 / 3) * np.pi * (2 / px) ** 3
        keep = props["area"] >= vol_thresh
        seg_nuc = util.map_array(seg_nuc, props["label"], props["label"] * keep)

        # --- Segment cells via membrane watershed seeded by nuclei ---
        cleaned_mem = gaussian(rescaled_mem, sigma=1)
        thresh_m = threshold_otsu(cleaned_mem)
        cleaned_mem = cleaned_mem > thresh_m
        _, seg_mem_exp = _watershed_from_seeds(cleaned_mem, seg_nuc, acinus_mask)

        # --- Match nuclei to cells ---
        matching = _match_nuclei_to_cells(seg_nuc, seg_mem_exp, px)

        # --- Mito properties ---
        mito_props = pd.DataFrame(
            regionprops_table(mito_labelled, properties=("label", "area"))
        )
        vx3 = px ** 3

        mito_rows = []
        for _, row in matching.iterrows():
            nuc_idx = int(row["nucleus_label"])
            cell_idx = int(row["cell_label"])

            # Mito that overlap with this nucleus's watershed region
            mito_in_nuc = np.unique(mito_labelled * (seg_nuc == nuc_idx))
            mito_in_nuc = mito_in_nuc[mito_in_nuc > 0]
            n_mito = len(mito_in_nuc)
            total_mito_vol = float(
                mito_props[mito_props["label"].isin(mito_in_nuc)]["area"].sum() * vx3
            )

            # Mito distribution: distance from nucleus surface within cell
            nuc_binary = (seg_nuc == nuc_idx).astype(np.uint8)
            cell_binary = (seg_mem_exp == cell_idx).astype(np.uint8)
            dist_from_nuc = ndi.distance_transform_edt(1 - nuc_binary) * cell_binary * (px ** 2)
            mito_in_cell = cell_binary * (mito_labelled > 0)

            # Bin mito pixel counts by distance
            flat_dist = np.ravel(dist_from_nuc)
            flat_mito = np.ravel(mito_in_cell)
            mask_nonzero = flat_dist > 0
            if mask_nonzero.any():
                binned_dist = np.round(flat_dist[mask_nonzero], 2)
                binned_mito = flat_mito[mask_nonzero]
                dist_df = pd.DataFrame({"distance": binned_dist, "mito_pixels": binned_mito})
                agg = dist_df.groupby("distance")["mito_pixels"].agg(["sum", "count"]).reset_index()
                max_d = agg["distance"].max()
                if max_d > 0:
                    agg["normalised_distance"] = agg["distance"] / max_d
                else:
                    agg["normalised_distance"] = 0.0
                agg["mito_ratio"] = agg["sum"] / agg["count"]
                mean_mito_ratio = float(agg["mito_ratio"].mean())
            else:
                mean_mito_ratio = np.nan

            mito_rows.append({
                "cell_label": cell_idx,
                "number_of_mito": n_mito,
                "mito_volume_um3": total_mito_vol,
                "mito_cell_vol_ratio": (
                    total_mito_vol / row["cell_volume_um3"]
                    if row["cell_volume_um3"] > 0 else np.nan
                ),
                "mean_mito_distance_ratio": mean_mito_ratio,
            })

        mito_df = pd.DataFrame(mito_rows)
        result = matching.merge(mito_df, on="cell_label", how="left")

        # Acinus-level stats
        acinus_regions = regionprops(acinus_mask)
        if acinus_regions:
            result["acinus_volume_um3"] = acinus_regions[0].area * vx3
            result["total_mito_volume_um3"] = mito_df["mito_volume_um3"].sum()
            result["number_of_cells"] = len(matching)
        else:
            result["acinus_volume_um3"] = np.nan
            result["total_mito_volume_um3"] = np.nan
            result["number_of_cells"] = 0

        # Filter out biologically implausible cells
        result = result[
            (result["mito_cell_vol_ratio"] < 0.5)
            & (result["nucleus_cell_volume_ratio"] < 0.9)
            & (result["mito_volume_um3"] > 0)
        ]

        result["flag"] = flag

        # QC plot
        self._save_qc("mitochondria", [
            (self._mid_z(acinus_mask), "Acinus mask"),
            (self._mid_z(seg_mem_exp), "Cell labels"),
            (self._mid_z(mito_labelled), "Mito labels"),
        ], red_channel=self.mito_channel)

        if self.return_volumes:
            self.volumes["acinus_mask"] = acinus_mask
            self.volumes["nuclei_labels"] = seg_nuc
            self.volumes["cell_labels"] = seg_mem_exp
            self.volumes["mito_labels"] = mito_labelled

        return result

    # =====================================================================
    #  Run multiple analyses at once
    # =====================================================================

    def run(self, analyses, **kwargs):
        """
        Run one or more analyses and return a dict of DataFrames.

        Parameters
        ----------
        analyses : list of str
            Choose from: ``"acinus_shape"``, ``"cell_nuclear_shape"``,
            ``"protein_polarisation"``, ``"apoptosis"``, ``"protein_proximity"``,
            ``"proliferation"``, ``"mitochondria"``.
        **kwargs
            Analysis-specific overrides (e.g. ``search_radius_um``,
            ``c3_separation_um``).  Only kwargs matching each method's
            signature are forwarded.
        """
        unknown = set(analyses) - VALID_ANALYSES
        if unknown:
            raise ValueError(f"Unknown analyses: {unknown}. Choose from {VALID_ANALYSES}")

        results: Dict[str, pd.DataFrame] = {}
        for name in analyses:
            try:
                method = getattr(self, name)
                sig = inspect.signature(method)
                filtered = {k: v for k, v in kwargs.items()
                            if k in sig.parameters and v is not None}
                df = method(**filtered)
                # Add filename and parsed experimental metadata
                flag = df["flag"].iloc[0] if "flag" in df.columns and len(df) > 0 else "None"
                df = add_image_details(df, self.filename, flag)
                results[name] = df
            except Exception as e:
                err_df = pd.DataFrame(
                    {"filename": [self.filename], "flag": [f"FAILED: {e}"]}
                )
                results[name] = err_df

        return results


# ---------------------------------------------------------------------------
#  Batch processing
# ---------------------------------------------------------------------------

_CONSTRUCTOR_KEYS = frozenset({
    "nuclear_channel", "membrane_channel", "protein_channel",
    "c3_channel", "edu_channel", "mito_channel",
    "proximity_protein_channel", "extra_acinus_channels",
})


def batch_analyse(
    image_dir: str,
    analyses: List[str],
    *,
    file_extension: str = "tif",
    n_jobs: int = 3,
    output_csv: Optional[str] = None,
    nuclear_mask_dir: Optional[str] = None,
    membrane_mask_dir: Optional[str] = None,
    c3_mask_dir: Optional[str] = None,
    edu_mask_dir: Optional[str] = None,
    mito_mask_dir: Optional[str] = None,
    qc_dir: Optional[str] = None,
    progress_callback=None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Run analyses on every image in a directory.

    Mask directories, if provided, must contain one mask TIFF per image TIFF
    (matched alphabetically).

    Returns a dict mapping analysis name -> concatenated DataFrame for all images.
    """
    image_paths = sorted(pathlib.Path(image_dir).rglob(f"*.{file_extension}"))
    if not image_paths:
        raise FileNotFoundError(f"No .{file_extension} files found in {image_dir}")

    def _sorted_masks(d):
        if d is None:
            return [None] * len(image_paths)
        masks = sorted(pathlib.Path(d).rglob(f"*.{file_extension}"))
        if len(masks) != len(image_paths):
            raise ValueError(
                f"Mask count mismatch: {len(image_paths)} images vs {len(masks)} masks in {d}"
            )
        return [str(m) for m in masks]

    nuc_masks = _sorted_masks(nuclear_mask_dir)
    mem_masks = _sorted_masks(membrane_mask_dir)
    c3_masks = _sorted_masks(c3_mask_dir)
    edu_masks = _sorted_masks(edu_mask_dir)
    mito_masks = _sorted_masks(mito_mask_dir)

    # Split kwargs into constructor args vs analysis args
    ctor_kwargs = {k: v for k, v in kwargs.items()
                   if k in _CONSTRUCTOR_KEYS and v is not None}
    analysis_kwargs = {k: v for k, v in kwargs.items()
                       if k not in _CONSTRUCTOR_KEYS}

    def _process(i):
        img = AcinarImage(
            image_paths[i],
            nuclear_mask_path=nuc_masks[i],
            membrane_mask_path=mem_masks[i],
            c3_mask_path=c3_masks[i],
            edu_mask_path=edu_masks[i],
            mito_mask_path=mito_masks[i],
            **ctor_kwargs,
        )
        img.qc_dir = qc_dir
        return img.run(analyses, **analysis_kwargs)

    print(f"Found {len(image_paths)} images. Running analyses: {analyses}")
    with _tqdm_joblib(tqdm(desc="Acinar Analysis", total=len(image_paths)), progress_callback=progress_callback):
        all_results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process)(i) for i in range(len(image_paths))
        )

    # Merge per-analysis DataFrames across images
    merged: Dict[str, pd.DataFrame] = {}
    for name in analyses:
        frames = [r[name] for r in all_results if name in r]
        merged[name] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if output_csv:
        for name, df in merged.items():
            out = output_csv.replace(".csv", f"_{name}.csv")
            df.to_csv(out, index=False)
            print(f"Saved {out}")

    return merged


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified acinar analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Acinus shape only
  python acinar_analysis.py --image-dir ./images --analyses acinus_shape

  # Protein polarisation
  python acinar_analysis.py --image-dir ./images --analyses protein_polarisation --protein-channel 1

  # Apoptosis counting (needs masks)
  python acinar_analysis.py --image-dir ./images --analyses apoptosis \\
      --c3-mask-dir ./c3_masks --nuclear-mask-dir ./nuclear_masks --c3-channel 3

  # Proliferation (EdU, needs masks)
  python acinar_analysis.py --image-dir ./images --analyses proliferation \\
      --edu-mask-dir ./edu_masks --nuclear-mask-dir ./nuclear_masks --edu-channel 1

  # Mitochondria analysis (needs nuclear, membrane, and mito masks)
  python acinar_analysis.py --image-dir ./images --analyses mitochondria \\
      --nuclear-mask-dir ./nuc_masks --membrane-mask-dir ./mem_masks --mito-mask-dir ./mito_masks

  # Multiple analyses at once
  python acinar_analysis.py --image-dir ./images --analyses acinus_shape protein_polarisation \\
      --protein-channel 1 --output results.csv
""",
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing image TIFFs")
    parser.add_argument(
        "--analyses", nargs="+", required=True,
        choices=sorted(VALID_ANALYSES),
        help="Which analyses to run",
    )
    parser.add_argument("--file-extension", default="tif")
    parser.add_argument("--nuclear-channel", type=int, default=0)
    parser.add_argument("--membrane-channel", type=int, default=None)
    parser.add_argument("--protein-channel", type=int, default=None)
    parser.add_argument("--c3-channel", type=int, default=None)
    parser.add_argument("--edu-channel", type=int, default=None)
    parser.add_argument("--mito-channel", type=int, default=None)
    parser.add_argument("--proximity-protein-channel", type=int, default=None)
    parser.add_argument("--nuclear-mask-dir", default=None)
    parser.add_argument("--membrane-mask-dir", default=None)
    parser.add_argument("--c3-mask-dir", default=None)
    parser.add_argument("--edu-mask-dir", default=None)
    parser.add_argument("--mito-mask-dir", default=None)
    parser.add_argument("--search-radius-um", type=float, default=5.0)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--output", default="acinar_results.csv", help="Output CSV path")

    args = parser.parse_args()

    batch_analyse(
        image_dir=args.image_dir,
        analyses=args.analyses,
        file_extension=args.file_extension,
        n_jobs=args.n_jobs,
        output_csv=args.output,
        nuclear_mask_dir=args.nuclear_mask_dir,
        membrane_mask_dir=args.membrane_mask_dir,
        c3_mask_dir=args.c3_mask_dir,
        edu_mask_dir=args.edu_mask_dir,
        mito_mask_dir=args.mito_mask_dir,
        nuclear_channel=args.nuclear_channel,
        membrane_channel=args.membrane_channel,
        protein_channel=args.protein_channel,
        c3_channel=args.c3_channel,
        edu_channel=args.edu_channel,
        mito_channel=args.mito_channel,
        proximity_protein_channel=args.proximity_protein_channel,
        search_radius_um=args.search_radius_um,
    )


if __name__ == "__main__":
    main()

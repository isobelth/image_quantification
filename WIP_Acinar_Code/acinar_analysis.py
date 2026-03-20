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

Usage
-----
Single image::

    from acinar_analysis import analyse_image
    results = analyse_image(
        image_path="path/to/image.tif",
        analyses=["acinus_shape", "protein_polarisation"],
        dapi_channel=0,
        membrane_channel=2,
        protein_channel=1,
    )

Batch::

    from acinar_analysis import batch_analyse
    df = batch_analyse(
        image_dir="path/to/images",
        analyses=["acinus_shape"],
        dapi_channel=0,
        membrane_channel=2,
        output_csv="results.csv",
    )

CLI::

    python acinar_analysis.py --image-dir path/to/images --analyses acinus_shape --dapi-channel 0 --membrane-channel 2
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import pathlib
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from skimage import util
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import (
    ball,
    clear_border,
    dilation,
    erosion,
    opening,
    remove_small_holes,
    remove_small_objects,
)
from skimage.segmentation import expand_labels, watershed
from skimage.transform import rescale
from tifffile import imread
from tqdm import tqdm

import tifffile as _tifffile

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
#  Valid analysis names
# ---------------------------------------------------------------------------
VALID_ANALYSES = {
    "acinus_shape",
    "cell_nuclear_shape",
    "protein_polarisation",
    "apoptosis",
    "protein_proximity",
}

# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """Context manager so joblib Parallel updates a tqdm bar."""

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
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
    """Return [x, y, z] pixel spacing in µm from TIFF metadata."""
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


def _rescale_image(image: np.ndarray, spacing: List[float], scale: float = 0.25
                   ) -> Tuple[np.ndarray, float]:
    """Rescale a 3-D volume to isotropic voxels. Returns (rescaled, new_pixel_size)."""
    scale_z = spacing[2] / spacing[0]
    rescaled = rescale(image, scale=(scale * scale_z, scale, scale), anti_aliasing=False)
    new_pixel_size = spacing[0] / scale  # e.g. 4 * original if scale=0.25
    return rescaled, new_pixel_size


# ---------------------------------------------------------------------------
#  Core acinus segmentation (shared by every analysis)
# ---------------------------------------------------------------------------

def segment_acinus(
    acinus_image: np.ndarray,
    spacing: List[float],
    scale: float = 0.25,
    smoothing_sigma: float = 4.0,
    clip_low_quantile: float = 0.1,
    clip_high_quantile: float = 0.85,
    min_sphericity: float = 0.55,
) -> Tuple[np.ndarray, float, str]:
    """
    Segment the primary acinus from a combined intensity image.

    Returns
    -------
    acinus_mask : 3-D labelled array (background=0, acinus=1)
    new_pixel_size : isotropic voxel size in µm
    flag : string flag describing any issues
    """
    rescaled, new_pixel_size = _rescale_image(acinus_image, spacing, scale)

    clipped = rescaled.clip(
        min=np.quantile(rescaled, clip_low_quantile),
        max=np.quantile(rescaled, clip_high_quantile),
    )
    smoothed = gaussian(clipped, sigma=smoothing_sigma)
    thresh = threshold_otsu(smoothed)
    binary = smoothed > thresh
    binary = remove_small_holes(binary, area_threshold=100000)
    binary = remove_small_objects(binary, min_size=10000)

    labelled = label(binary)
    props = regionprops_table(labelled, properties=("label", "area"))
    keep = props["label"][np.argmax(props["area"])]
    acinus_mask = np.where(labelled == keep, 1, 0).astype(np.int32)

    # Test sphericity
    flag = "None"
    eigvals = regionprops_table(
        acinus_mask, properties=("label", "inertia_tensor_eigvals")
    )
    if len(eigvals["label"]) > 0:
        sphericity = eigvals["inertia_tensor_eigvals-2"][0] / eigvals["inertia_tensor_eigvals-0"][0]
        if sphericity < min_sphericity:
            flag = "multiple_acini_split"
            smoothed2 = gaussian(clipped, sigma=1)
            thresh2 = threshold_otsu(smoothed2)
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

    return acinus_mask, new_pixel_size, flag


def _build_acinus_approximation(
    image: np.ndarray,
    dapi_channel: int,
    membrane_channel: Optional[int],
    extra_channels: Optional[List[int]] = None,
) -> np.ndarray:
    """Sum selected channels (as uint8) to approximate acinus extent."""
    combined = rescale_intensity(image[:, dapi_channel, :, :]).astype(np.float64)
    if membrane_channel is not None:
        combined = combined + rescale_intensity(image[:, membrane_channel, :, :]).astype(np.float64)
    if extra_channels:
        for ch in extra_channels:
            combined = combined + rescale_intensity(image[:, ch, :, :]).astype(np.float64)
    return combined


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
    out = util.map_array(seg, props["label"], props["label"] * keep)
    return out


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

    Returns a DataFrame indexed by cell_label with columns
    ``sum`` (neighbour count), ``external``, ``internal``.
    """
    max_label = int(label_matrix.max())
    neighbours = np.zeros((max_label + 1, max_label + 1), dtype=np.uint8)
    sz, sy, sx = label_matrix.shape

    for i in range(sz):
        for j in range(sy):
            for k in range(sx):
                v = label_matrix[i, j, k]
                if i > 0 and label_matrix[i - 1, j, k] != v:
                    neighbours[label_matrix[i - 1, j, k], v] = 1
                if i < sz - 1 and label_matrix[i + 1, j, k] != v:
                    neighbours[label_matrix[i + 1, j, k], v] = 1
                if j > 0 and label_matrix[i, j - 1, k] != v:
                    neighbours[label_matrix[i, j - 1, k], v] = 1
                if j < sy - 1 and label_matrix[i, j + 1, k] != v:
                    neighbours[label_matrix[i, j + 1, k], v] = 1
                if k > 0 and label_matrix[i, j, k - 1] != v:
                    neighbours[label_matrix[i, j, k - 1], v] = 1
                if k < sx - 1 and label_matrix[i, j, k + 1] != v:
                    neighbours[label_matrix[i, j, k + 1], v] = 1

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


# ---------------------------------------------------------------------------
#  Analysis 1: Acinus Shape
# ---------------------------------------------------------------------------

def analyse_acinus_shape(
    image_path: str,
    dapi_channel: int = 0,
    membrane_channel: Optional[int] = 2,
    extra_channels: Optional[List[int]] = None,
    smoothing_sigma: float = 4.0,
) -> pd.DataFrame:
    """Calculate acinus volume (µm³) and roundness.

    Requires only the raw multi-channel image.
    """
    image = rescale_intensity(imread(str(image_path)))
    spacing = read_pixel_size(str(image_path))
    acinus_approx = _build_acinus_approximation(image, dapi_channel, membrane_channel, extra_channels)
    acinus_mask, px, flag = segment_acinus(acinus_approx, spacing, smoothing_sigma=smoothing_sigma)

    regions = regionprops(acinus_mask)
    if not regions:
        return pd.DataFrame({"vol_um": [np.nan], "roundness": [np.nan], "flag": ["no_acinus"]})
    r = regions[0]
    vol = r.area * px ** 3
    roundness = r.inertia_tensor_eigvals[2] / r.inertia_tensor_eigvals[0]

    # Check for holes
    hole_sizes = [rr.area for rr in regionprops(label(util.invert(acinus_mask > 0)))]
    if len(hole_sizes) > 1 and flag == "None":
        flag = "hole"

    return pd.DataFrame({"vol_um": [vol], "roundness": [roundness], "flag": [flag]})


# ---------------------------------------------------------------------------
#  Analysis 2: Cell & Nuclear Shape
# ---------------------------------------------------------------------------

def analyse_cell_nuclear_shape(
    image_path: str,
    nuclear_mask_path: str,
    membrane_mask_path: str,
    dapi_channel: int = 0,
    membrane_channel: int = 2,
    nuclear_channel: Optional[int] = None,
    smoothing_sigma: float = 4.0,
) -> pd.DataFrame:
    """Segment cells/nuclei and compute volume, roundness, and neighbour info.

    **Requires** pre-computed binary nuclear and membrane segmentation masks.
    """
    if nuclear_mask_path is None or membrane_mask_path is None:
        raise ValueError(
            "Cell/nuclear shape analysis requires both 'nuclear_mask_path' and "
            "'membrane_mask_path'. Provide paths to binary segmentation TIFFs."
        )

    image = rescale_intensity(imread(str(image_path)))
    spacing = read_pixel_size(str(image_path))
    nuc_ch = nuclear_channel if nuclear_channel is not None else dapi_channel
    acinus_approx = _build_acinus_approximation(image, nuc_ch, membrane_channel)
    acinus_mask, px, flag = segment_acinus(acinus_approx, spacing, smoothing_sigma=smoothing_sigma)

    scale_z = spacing[2] / spacing[0]
    nuclear_mask = imread(str(nuclear_mask_path))
    membrane_mask = imread(str(membrane_mask_path))

    rescaled_nuc = rescale(nuclear_mask, (0.25 * scale_z, 0.25, 0.25), anti_aliasing=False)
    rescaled_mem = rescale(membrane_mask, (0.25 * scale_z, 0.25, 0.25), anti_aliasing=False)

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

    if spacing == [1, 1, 1]:
        matching["flag"] = "wrong_metadata"

    return matching


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
    match_df["nucleus_volume_um"] = match_df["nucleus_volume"] * vx3
    match_df["cell_volume_um"] = match_df["cell_volume"] * vx3
    match_df["cell_roundness"] = match_df["cell_eig2"] / match_df["cell_eig0"]
    match_df["nucleus_roundness"] = match_df["nuc_eig2"] / match_df["nuc_eig0"]
    match_df.drop(
        columns=["cell_volume", "nucleus_volume", "cell_eig0", "cell_eig1", "cell_eig2",
                 "nuc_eig0", "nuc_eig1", "nuc_eig2"],
        inplace=True,
    )
    return match_df


# ---------------------------------------------------------------------------
#  Analysis 3: Protein Polarisation
# ---------------------------------------------------------------------------

def analyse_protein_polarisation(
    image_path: str,
    dapi_channel: int = 0,
    membrane_channel: int = 2,
    protein_channel: int = 1,
    extra_channels: Optional[List[int]] = None,
    smoothing_sigma: float = 11.0,
) -> pd.DataFrame:
    """Quantify protein intensity as a function of normalised radial distance.

    Requires only the raw multi-channel image (protein channel specified).
    """
    if protein_channel is None:
        raise ValueError(
            "Protein polarisation analysis requires 'protein_channel' to be set."
        )

    image = rescale_intensity(imread(str(image_path)))
    spacing = read_pixel_size(str(image_path))

    # Build acinus approximation from DAPI + membrane + protein
    channels_for_acinus = [protein_channel]
    if extra_channels:
        channels_for_acinus.extend(extra_channels)
    acinus_approx = _build_acinus_approximation(
        image, dapi_channel, membrane_channel, channels_for_acinus
    )
    acinus_mask, px, flag = segment_acinus(
        acinus_approx, spacing, smoothing_sigma=smoothing_sigma
    )

    # Rescale protein channel
    scale_z = spacing[2] / spacing[0]
    protein_rescaled = rescale(
        image[:, protein_channel, :, :], (0.25 * scale_z, 0.25, 0.25), anti_aliasing=False
    )
    protein_rescaled = protein_rescaled * acinus_mask

    # Distance map normalised by equivalent sphere radius
    distance = distance_transform_edt(acinus_mask)
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
    return df


# ---------------------------------------------------------------------------
#  Analysis 4: Apoptosis (C3 counting)
# ---------------------------------------------------------------------------

def analyse_apoptosis(
    image_path: str,
    c3_mask_path: str,
    dapi_mask_path: str,
    dapi_channel: int = 0,
    c3_channel: int = 3,
    membrane_channel: Optional[int] = None,
    c3_separation_um: float = 7.0,
    c3_min_radius_um: float = 1.3,
    dapi_separation_um: float = 6.0,
    dapi_min_radius_um: float = 2.0,
) -> pd.DataFrame:
    """Count C3-positive (apoptotic) cells and total nuclei per acinus.

    **Requires** pre-computed binary C3 and DAPI segmentation masks.
    """
    if c3_mask_path is None or dapi_mask_path is None:
        raise ValueError(
            "Apoptosis analysis requires both 'c3_mask_path' and 'dapi_mask_path'. "
            "Provide paths to binary segmentation TIFFs."
        )

    image = rescale_intensity(imread(str(image_path)))
    spacing = read_pixel_size(str(image_path))
    filename = os.path.basename(str(image_path)).lower()

    extra = [c3_channel]
    acinus_approx = _build_acinus_approximation(image, dapi_channel, membrane_channel, extra)
    acinus_mask, px, flag = segment_acinus(acinus_approx, spacing, smoothing_sigma=12.0)

    scale_z = spacing[2] / spacing[0]
    c3_mask = rescale(imread(str(c3_mask_path)), (0.25 * scale_z, 0.25, 0.25), anti_aliasing=False)
    dapi_mask = rescale(imread(str(dapi_mask_path)), (0.25 * scale_z, 0.25, 0.25), anti_aliasing=False)

    c3_labels = watershed_segment(c3_mask, acinus_mask, px, c3_separation_um, c3_min_radius_um)
    dapi_labels = watershed_segment(dapi_mask, acinus_mask, px, dapi_separation_um, dapi_min_radius_um)

    # Distance map (normalised 0-1)
    dist = distance_transform_edt(acinus_mask > 0) * px
    dist_scaled = np.interp(dist, (dist.min(), dist.max()), (0, 1))

    # Acinus-level measurements
    acinus_regions = regionprops(acinus_mask)
    acinus_vol = acinus_regions[0].area * px ** 3 if acinus_regions else np.nan
    acinus_round = (
        acinus_regions[0].inertia_tensor_eigvals[2] / acinus_regions[0].inertia_tensor_eigvals[0]
        if acinus_regions else np.nan
    )

    c3_props = pd.DataFrame(regionprops_table(c3_labels, properties=("label", "area", "centroid")))
    if c3_props.empty:
        c3_props = pd.DataFrame({
            "label": ["no_c3"], "centroid-0": [np.nan], "centroid-1": [np.nan],
            "centroid-2": [np.nan], "c3_volume_um": [np.nan],
            "normalised_distance": [np.nan],
        })
    else:
        c3_props["c3_volume_um"] = c3_props["area"] * px ** 3
        c3_props["normalised_distance"] = c3_props.apply(
            lambda row: dist_scaled[
                int(round(row["centroid-0"])),
                int(round(row["centroid-1"])),
                int(round(row["centroid-2"])),
            ],
            axis=1,
        )
        c3_props.drop("area", axis=1, inplace=True)

    c3_props["acinus_volume_um"] = acinus_vol
    c3_props["acinus_roundness"] = acinus_round
    c3_props["number_of_nuclei"] = pd.DataFrame(
        regionprops_table(dapi_labels, properties=("label",))
    ).shape[0]
    c3_props["flag"] = flag
    return c3_props


# ---------------------------------------------------------------------------
#  Analysis 5: Protein Proximity
# ---------------------------------------------------------------------------

def analyse_protein_proximity(
    image_path: str,
    c3_mask_path: str,
    dapi_mask_path: str,
    dapi_channel: int = 0,
    c3_channel: int = 3,
    membrane_channel: Optional[int] = None,
    proximity_protein_channel: int = 1,
    search_radius_um: float = 5.0,
    c3_separation_um: float = 7.0,
    c3_min_radius_um: float = 3.0,
    dapi_separation_um: float = 6.0,
    dapi_min_radius_um: float = 3.0,
) -> pd.DataFrame:
    """Compare protein intensity near dying vs non-dying cells.

    **Requires** binary C3 and DAPI masks, plus a proximity protein channel.
    """
    if c3_mask_path is None or dapi_mask_path is None:
        raise ValueError(
            "Protein proximity analysis requires both 'c3_mask_path' and 'dapi_mask_path'."
        )
    if proximity_protein_channel is None:
        raise ValueError(
            "Protein proximity analysis requires 'proximity_protein_channel' to be set."
        )

    image = rescale_intensity(imread(str(image_path)))
    spacing = read_pixel_size(str(image_path))
    filename = os.path.basename(str(image_path)).lower()

    extra = [c3_channel]
    acinus_approx = _build_acinus_approximation(image, dapi_channel, membrane_channel, extra)
    acinus_mask, px, flag = segment_acinus(acinus_approx, spacing, smoothing_sigma=12.0)

    scale_z = spacing[2] / spacing[0]
    c3_raw = imread(str(c3_mask_path))
    dapi_raw = imread(str(dapi_mask_path))
    c3_mask = rescale(c3_raw, (0.25 * scale_z, 0.25, 0.25), anti_aliasing=False)
    # Live cells = DAPI minus dilated C3
    live_cells = rescale(
        (dapi_raw - dilation(c3_raw, ball(2))),
        (0.25 * scale_z, 0.25, 0.25),
        anti_aliasing=False,
    )
    live_cells = np.where(live_cells > 1, 0, live_cells)

    c3_labels = watershed_segment(c3_mask, acinus_mask, px, c3_separation_um, c3_min_radius_um)
    live_labels = watershed_segment(live_cells, acinus_mask, px, dapi_separation_um, dapi_min_radius_um)

    # Distance map (normalised 0-1)
    dist = distance_transform_edt(acinus_mask > 0)
    dist_scaled = np.interp(dist, (dist.min(), dist.max()), (0, 1))

    acinus_regions = regionprops(acinus_mask)
    acinus_vol = acinus_regions[0].area * px ** 3 if acinus_regions else np.nan
    acinus_round = (
        acinus_regions[0].inertia_tensor_eigvals[2] / acinus_regions[0].inertia_tensor_eigvals[0]
        if acinus_regions else np.nan
    )

    # Rescale proximity protein
    prox_img = rescale(
        image[:, proximity_protein_channel, :, :],
        (0.25 * scale_z, 0.25, 0.25),
        anti_aliasing=False,
    )

    # Build dying / non-dying info
    dying_info = pd.DataFrame(regionprops_table(c3_labels, properties=("label", "area", "centroid")))
    dying_info["nuclear_or_c3_volume_um3"] = dying_info["area"] * px ** 3
    dying_info.drop("area", axis=1, inplace=True)
    dying_info["dying"] = "Y"

    live_info = pd.DataFrame(regionprops_table(live_labels, properties=("label", "area", "centroid")))
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
    estimated_territories = acinus_mask * expand_labels(combined, distance=20)

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
    return all_cells


# ---------------------------------------------------------------------------
#  Unified single-image entry point
# ---------------------------------------------------------------------------

def analyse_image(
    image_path: str,
    analyses: List[str],
    *,
    dapi_channel: int = 0,
    membrane_channel: Optional[int] = 2,
    protein_channel: Optional[int] = None,
    c3_channel: Optional[int] = None,
    proximity_protein_channel: Optional[int] = None,
    nuclear_mask_path: Optional[str] = None,
    membrane_mask_path: Optional[str] = None,
    c3_mask_path: Optional[str] = None,
    dapi_mask_path: Optional[str] = None,
    extra_acinus_channels: Optional[List[int]] = None,
    smoothing_sigma: Optional[float] = None,
    search_radius_um: float = 5.0,
    c3_separation_um: float = 7.0,
    c3_min_radius_um: float = 1.3,
    dapi_separation_um: float = 6.0,
    dapi_min_radius_um: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    """
    Run one or more analyses on a single 3-D acinar image.

    Parameters
    ----------
    image_path : str
        Path to the multi-channel TIFF stack.
    analyses : list of str
        Which analyses to run. Choose from:
        ``"acinus_shape"``, ``"cell_nuclear_shape"``,
        ``"protein_polarisation"``, ``"apoptosis"``,
        ``"protein_proximity"``.
    dapi_channel, membrane_channel : int
        Channel indices in the image.
    protein_channel : int, optional
        Channel for protein polarisation analysis.
    c3_channel : int, optional
        Channel for C3 / apoptosis analysis.
    proximity_protein_channel : int, optional
        Channel for the proximity-protein analysis.
    nuclear_mask_path, membrane_mask_path : str, optional
        Paths to binary segmentation masks (required for cell_nuclear_shape).
    c3_mask_path, dapi_mask_path : str, optional
        Paths to binary segmentation masks (required for apoptosis / protein_proximity).
    search_radius_um : float
        Search radius for proximity analysis.

    Returns
    -------
    dict mapping analysis name -> DataFrame of results.
    """
    unknown = set(analyses) - VALID_ANALYSES
    if unknown:
        raise ValueError(f"Unknown analyses: {unknown}. Choose from {VALID_ANALYSES}")

    results: Dict[str, pd.DataFrame] = {}
    filename = os.path.basename(str(image_path)).lower()

    for name in analyses:
        try:
            if name == "acinus_shape":
                sigma = smoothing_sigma if smoothing_sigma is not None else 4.0
                df = analyse_acinus_shape(
                    image_path, dapi_channel, membrane_channel,
                    extra_acinus_channels, sigma,
                )

            elif name == "cell_nuclear_shape":
                sigma = smoothing_sigma if smoothing_sigma is not None else 4.0
                df = analyse_cell_nuclear_shape(
                    image_path, nuclear_mask_path, membrane_mask_path,
                    dapi_channel, membrane_channel,
                    smoothing_sigma=sigma,
                )

            elif name == "protein_polarisation":
                sigma = smoothing_sigma if smoothing_sigma is not None else 11.0
                df = analyse_protein_polarisation(
                    image_path, dapi_channel, membrane_channel,
                    protein_channel, extra_acinus_channels, sigma,
                )

            elif name == "apoptosis":
                df = analyse_apoptosis(
                    image_path, c3_mask_path, dapi_mask_path,
                    dapi_channel, c3_channel or 3, membrane_channel,
                    c3_separation_um, c3_min_radius_um,
                    dapi_separation_um, dapi_min_radius_um,
                )

            elif name == "protein_proximity":
                df = analyse_protein_proximity(
                    image_path, c3_mask_path, dapi_mask_path,
                    dapi_channel, c3_channel or 3, membrane_channel,
                    proximity_protein_channel,
                    search_radius_um, c3_separation_um, c3_min_radius_um,
                    dapi_separation_um, dapi_min_radius_um,
                )

            df["filename"] = filename
            results[name] = df

        except Exception as e:
            results[name] = pd.DataFrame({"filename": [filename], "flag": [f"FAILED: {e}"]})

    return results


# ---------------------------------------------------------------------------
#  Batch processing
# ---------------------------------------------------------------------------

def batch_analyse(
    image_dir: str,
    analyses: List[str],
    *,
    file_extension: str = "tif",
    n_jobs: int = 3,
    output_csv: Optional[str] = None,
    # Segmentation mask directories (optional)
    nuclear_mask_dir: Optional[str] = None,
    membrane_mask_dir: Optional[str] = None,
    c3_mask_dir: Optional[str] = None,
    dapi_mask_dir: Optional[str] = None,
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
    dapi_masks = _sorted_masks(dapi_mask_dir)

    def _process(i):
        return analyse_image(
            str(image_paths[i]),
            analyses,
            nuclear_mask_path=nuc_masks[i],
            membrane_mask_path=mem_masks[i],
            c3_mask_path=c3_masks[i],
            dapi_mask_path=dapi_masks[i],
            **kwargs,
        )

    print(f"Found {len(image_paths)} images. Running analyses: {analyses}")
    with _tqdm_joblib(tqdm(desc="Acinar Analysis", total=len(image_paths))):
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(_process)(i) for i in range(len(image_paths))
        )

    # Merge per-analysis DataFrames across images
    merged: Dict[str, pd.DataFrame] = {}
    for name in analyses:
        frames = [r[name] for r in all_results if name in r]
        if frames:
            merged[name] = pd.concat(frames, ignore_index=True)
        else:
            merged[name] = pd.DataFrame()

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
      --c3-mask-dir ./c3_masks --dapi-mask-dir ./dapi_masks --c3-channel 3

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
    parser.add_argument("--dapi-channel", type=int, default=0)
    parser.add_argument("--membrane-channel", type=int, default=None)
    parser.add_argument("--protein-channel", type=int, default=None)
    parser.add_argument("--c3-channel", type=int, default=None)
    parser.add_argument("--proximity-protein-channel", type=int, default=None)
    parser.add_argument("--nuclear-mask-dir", default=None)
    parser.add_argument("--membrane-mask-dir", default=None)
    parser.add_argument("--c3-mask-dir", default=None)
    parser.add_argument("--dapi-mask-dir", default=None)
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
        dapi_mask_dir=args.dapi_mask_dir,
        dapi_channel=args.dapi_channel,
        membrane_channel=args.membrane_channel,
        protein_channel=args.protein_channel,
        c3_channel=args.c3_channel,
        proximity_protein_channel=args.proximity_protein_channel,
        search_radius_um=args.search_radius_um,
    )


if __name__ == "__main__":
    main()

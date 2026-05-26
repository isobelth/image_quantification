"""Shared I/O helpers: pixel-size extraction, 2D slice extraction, image iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import tifffile
from liffile import LifFile


# ---------------------------------------------------------------------------
# Intensity utilities
# ---------------------------------------------------------------------------

def rescale_intensity(img: np.ndarray, target_min: float = 0, target_max: float = 255, target_dtype=np.uint8) -> np.ndarray:
    """Linearly rescale img intensity to [target_min, target_max] and cast."""
    img_min, img_max = float(img.min()), float(img.max())
    if img_max == img_min:
        return np.full_like(img, target_min, dtype=target_dtype)
    scale = (target_max - target_min) / (img_max - img_min)
    offset = target_max - scale * img_max
    return (scale * img + offset).astype(target_dtype)


# ---------------------------------------------------------------------------
# Pixel size extraction
# ---------------------------------------------------------------------------

def get_tiff_pixel_size_um(tif: tifffile.TiffFile) -> Optional[Tuple[float, float]]:
    """Return (y_um_per_px, x_um_per_px) from TIFF metadata, if available."""
    tags = tif.pages[0].tags
    imagej_metadata = getattr(tif, "imagej_metadata", None) or {}
    unit_hint = (imagej_metadata.get("unit") or "").lower()
    if "XResolution" not in tags or "YResolution" not in tags:
        return None
    if not unit_hint and "ResolutionUnit" in tags:
        unit_hint = {2: "inch", 3: "cm"}.get(int(tags["ResolutionUnit"].value), "")

    def resolution_to_um(res, unit):
        num, den = res
        if num == 0:
            return None
        um_per_px = den / num
        if unit == "cm":
            um_per_px *= 1e4
        elif unit == "mm":
            um_per_px *= 1e3
        elif unit in ("inch", "in"):
            um_per_px *= 25_400.0
        return um_per_px

    x_um = resolution_to_um(tags["XResolution"].value, unit_hint)
    y_um = resolution_to_um(tags["YResolution"].value, unit_hint)
    if x_um and y_um:
        return float(y_um), float(x_um)
    return None


def get_lif_pixel_size_um(image) -> Optional[Tuple[float, float]]:
    """Return (y_um_per_px, x_um_per_px) for a liffile image."""
    coords = image.asxarray().coords
    if "X" not in coords or coords["X"].size < 2 or "Y" not in coords or coords["Y"].size < 2:
        return None
    x_um = abs(float(coords["X"][1] - coords["X"][0])) * 1e6
    y_um = abs(float(coords["Y"][1] - coords["Y"][0])) * 1e6
    if x_um <= 0 or y_um <= 0:
        return None
    return y_um, x_um


# ---------------------------------------------------------------------------
# 2D slice extraction
# ---------------------------------------------------------------------------

def extract_2d_slice(arr: np.ndarray, axes: str, channel_no: int) -> np.ndarray:
    """Select a 2D YX slice from an arbitrarily-axed array, picking channel_no along C and max-projecting remaining axes."""
    axes = axes.upper()
    arr = np.asarray(arr)
    if len(axes) != arr.ndim:
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[min(channel_no, arr.shape[0] - 1)]
        sliced = arr[min(channel_no, arr.shape[0] - 1)]
        while sliced.ndim > 2:
            sliced = sliced.max(axis=0)
        return sliced
    if "C" in axes:
        channel_axis_idx = axes.index("C")
        arr = np.take(arr, min(channel_no, arr.shape[channel_axis_idx] - 1), axis=channel_axis_idx)
        axes = axes.replace("C", "")
    while arr.ndim > 2:
        arr = arr.max(axis=0)
    return arr


def get_channel_count(arr: np.ndarray, axes: str) -> int:
    """Return the number of channels (size of C axis), or 1 if no C axis."""
    axes = axes.upper()
    if len(axes) == arr.ndim and "C" in axes:
        return int(arr.shape[axes.index("C")])
    # ambiguous case: 3D without axis info — treat first axis as channel
    if arr.ndim == 3:
        return int(arr.shape[0])
    return 1


# ---------------------------------------------------------------------------
# File enumeration
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredImage:
    source_file: Path
    image_index: int        # index inside LIF; 0 for TIFs
    image_name: str         # LIF inner name, or TIF stem
    n_channels: int
    has_zt: bool            # True if Z or T dim > 1
    axes: str               # detected axes string


def _has_zt_dim(arr: np.ndarray, axes: str) -> bool:
    axes = axes.upper()
    if len(axes) != arr.ndim:
        return False
    return any(ax in ("Z", "T") and arr.shape[i] > 1 for i, ax in enumerate(axes))


def enumerate_images_in_file(file_path: Path) -> Iterator[DiscoveredImage]:
    """Yield one DiscoveredImage per image inside file_path (one for tifs, many for lifs)."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(file_path) as tif:
            arr = tif.asarray()
            axes = (tif.series[0].axes or "").upper() or ("YX" if arr.ndim == 2 else "")
            yield DiscoveredImage(
                source_file=file_path,
                image_index=0,
                image_name=file_path.stem,
                n_channels=get_channel_count(arr, axes),
                has_zt=_has_zt_dim(arr, axes),
                axes=axes,
            )
        return
    if suffix == ".lif":
        with LifFile(str(file_path)) as lif:
            for image_index, image in enumerate(lif.images):
                axes = "".join(image.sizes.keys()).upper()
                # avoid loading full array to detect channels / ZT
                size_map = {k.upper(): v for k, v in image.sizes.items()}
                n_channels = int(size_map.get("C", 1))
                has_zt = (size_map.get("Z", 1) > 1) or (size_map.get("T", 1) > 1)
                yield DiscoveredImage(
                    source_file=file_path,
                    image_index=image_index,
                    image_name=getattr(image, "name", ""),
                    n_channels=n_channels,
                    has_zt=has_zt,
                    axes=axes,
                )
        return
    raise ValueError(f"Unsupported file type: {suffix}")


def load_image_arrays(
    source_file: Path,
    image_index: int,
    bf_channel: int,
    fl_channel: Optional[int],
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float]]:
    """Load BF + optional FL 2D slice + (y_um, x_um) pixel size for one image."""
    source_file = Path(source_file)
    suffix = source_file.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(source_file) as tif:
            arr = tif.asarray()
            axes = (tif.series[0].axes or "").upper() or ("YX" if arr.ndim == 2 else "")
            pixel_size = get_tiff_pixel_size_um(tif) or (float("nan"), float("nan"))
            bf = extract_2d_slice(arr, axes, bf_channel)
            fl = extract_2d_slice(arr, axes, fl_channel) if fl_channel is not None else None
            return bf, fl, pixel_size
    if suffix == ".lif":
        with LifFile(str(source_file)) as lif:
            image = list(lif.images)[image_index]
            axes = "".join(image.sizes.keys()).upper()
            arr = np.asarray(image.asarray())
            pixel_size = get_lif_pixel_size_um(image) or (float("nan"), float("nan"))
            bf = extract_2d_slice(arr, axes, bf_channel)
            fl = extract_2d_slice(arr, axes, fl_channel) if fl_channel is not None else None
            return bf, fl, pixel_size
    raise ValueError(f"Unsupported file type: {suffix}")

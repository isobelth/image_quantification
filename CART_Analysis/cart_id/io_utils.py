"""Shared I/O helpers for the CART analysis pipeline.

Curation TIF channel layout (11 channels, uint8, CYX):
    ch0:  BF raw
    ch1:  B cells raw    (LIF channel 0)
    ch2:  Vasculature raw (LIF channel 1)
    ch3:  CART cells raw  (LIF channel -1)
    ch4:  chip_center mask  [editable in Stage 3]
    ch5:  tumour mask       [editable in Stage 3]
    ch6:  left_device mask  (auto only)
    ch7:  right_device mask (auto only)
    ch8:  vasculature mask  (auto only)
    ch9:  B cells mask      (auto only)
    ch10: CART cells mask   (auto only)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile
import liffile

N_CHANNELS = 11  # total TIF channels


# ---------------------------------------------------------------------------
# Intensity utilities
# ---------------------------------------------------------------------------

def rescale_uint8(img: np.ndarray) -> np.ndarray:
    """Linearly rescale img to [0, 255] uint8. Returns all-zeros on flat input."""
    img = np.asarray(img, dtype=float)
    mn, mx = img.min(), img.max()
    if mx == mn:
        return np.zeros(img.shape, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pixel size
# ---------------------------------------------------------------------------

def get_tiff_pixel_size_um(tif: tifffile.TiffFile) -> Optional[Tuple[float, float]]:
    """Return (y_um_per_px, x_um_per_px) from a TiffFile, or None."""
    tags = tif.pages[0].tags
    imagej_metadata = getattr(tif, "imagej_metadata", None) or {}
    unit_hint = (imagej_metadata.get("unit") or "").lower()
    if "XResolution" not in tags or "YResolution" not in tags:
        return None
    if not unit_hint and "ResolutionUnit" in tags:
        unit_hint = {2: "inch", 3: "cm"}.get(int(tags["ResolutionUnit"].value), "")

    def _res_to_um(res, unit: str) -> Optional[float]:
        num, den = res
        if num == 0:
            return None
        um = den / num
        if unit == "cm":
            um *= 1e4
        elif unit == "mm":
            um *= 1e3
        elif unit in ("inch", "in"):
            um *= 25_400.0
        return float(um)

    x_um = _res_to_um(tags["XResolution"].value, unit_hint)
    y_um = _res_to_um(tags["YResolution"].value, unit_hint)
    if x_um and y_um:
        return float(y_um), float(x_um)
    return None


def get_lif_pixel_size_um(image) -> Optional[Tuple[float, float]]:
    """Return (y_um_per_px, x_um_per_px) for a liffile image."""
    try:
        xarr = image.asxarray()
        coords = xarr.coords
        if "X" not in coords or "Y" not in coords:
            return None
        x_coords = coords["X"]
        y_coords = coords["Y"]
        if x_coords.size < 2 or y_coords.size < 2:
            return None
        x_um = abs(float(x_coords[1] - x_coords[0])) * 1e6
        y_um = abs(float(y_coords[1] - y_coords[0])) * 1e6
        if x_um > 0 and y_um > 0:
            return float(y_um), float(x_um)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# LIF loading
# ---------------------------------------------------------------------------

def load_lif_channels(
    lif_path: Path,
    image_index: int,
) -> Tuple[np.ndarray, float]:
    """Load max-projection (C, Y, X) uint16 from a LIF image.

    Returns (stack_cyx, um_per_px).  ``um_per_px`` is the mean of x/y
    (used as a scalar scale for the analysis helpers).
    """
    with liffile.LifFile(str(lif_path)) as lif:
        image = list(lif.images)[image_index]
        arr = np.asarray(image.asarray())  # shape varies by Z/C/Y/X ordering

        # Identify Z axis and max-project
        sizes = {k.upper(): v for k, v in image.sizes.items()}
        axes_str = "".join(image.sizes.keys()).upper()

        # Build axis index map
        ax_idx = {ax: i for i, ax in enumerate(axes_str)}

        if "Z" in ax_idx and sizes.get("Z", 1) > 1:
            arr = arr.max(axis=ax_idx["Z"])
            axes_str = axes_str.replace("Z", "")
            ax_idx = {ax: i for i, ax in enumerate(axes_str)}

        # After Z projection we expect C, Y, X (possibly in any order)
        # Re-map to CYX
        remaining = axes_str.replace("C", "").replace("Y", "").replace("X", "")
        # Max-project any remaining non-spatial axes
        while remaining:
            ax = remaining[0]
            if ax in ax_idx:
                arr = arr.max(axis=ax_idx[ax])
                axes_str = axes_str.replace(ax, "")
                ax_idx = {a: i for i, a in enumerate(axes_str)}
            remaining = remaining[1:]

        # Ensure CYX order
        if axes_str and axes_str != "CYX":
            perm = [ax_idx.get(a) for a in "CYX" if a in ax_idx]
            # If C is missing treat as 1-channel
            if "C" not in ax_idx:
                arr = arr[np.newaxis]  # add C axis
                axes_str = "C" + axes_str
                ax_idx = {a: i for i, a in enumerate(axes_str)}
                perm = [ax_idx.get(a) for a in "CYX" if a in ax_idx]
            arr = np.transpose(arr, perm)

        pixel_size = get_lif_pixel_size_um(image)
        if pixel_size is not None:
            y_um, x_um = pixel_size
            um_per_px = float(np.mean([x_um, y_um]))
        else:
            um_per_px = 1.0
            warnings.warn(f"Could not read pixel size for image {image_index} in {lif_path.name}; using 1 µm/px")

        return arr, um_per_px


# ---------------------------------------------------------------------------
# Curation TIF read/write
# ---------------------------------------------------------------------------

def write_curation_tif(
    path: Path,
    stack_11ch: np.ndarray,
    pixel_size_um: Tuple[float, float],
) -> None:
    """Write an 11-channel uint8 CYX TIF with pixel size in resolution tags."""
    assert stack_11ch.ndim == 3 and stack_11ch.shape[0] == N_CHANNELS, (
        f"Expected ({N_CHANNELS}, Y, X) but got {stack_11ch.shape}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    y_um, x_um = pixel_size_um
    kwargs: dict = dict(imagej=True, metadata={"axes": "CYX", "unit": "um"})
    if np.isfinite(x_um) and np.isfinite(y_um) and x_um > 0 and y_um > 0:
        kwargs["resolution"] = (1.0 / x_um, 1.0 / y_um)
    tifffile.imwrite(path, stack_11ch.astype(np.uint8), **kwargs)


def read_curation_tif(
    path: Path,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """Return (stack_11ch uint8, pixel_size_um or None)."""
    with tifffile.TiffFile(path) as tif:
        stack = np.asarray(tif.asarray())
        pixel_size = get_tiff_pixel_size_um(tif)
    if stack.ndim != 3 or stack.shape[0] != N_CHANNELS:
        raise ValueError(
            f"Expected ({N_CHANNELS}, Y, X) curation TIF; got {stack.shape} in {path}"
        )
    return stack.astype(np.uint8), pixel_size


# ---------------------------------------------------------------------------
# Filename metadata parsing
# ---------------------------------------------------------------------------

def parse_filename_metadata(name: str) -> dict:
    """Extract patient and day from a filename string.

    Returns dict with keys 'patient' and 'day' (or 'unknown').
    """
    n = name.lower()
    patient = "fl32" if "fl32" in n else "fl12" if "fl12" in n else "unknown"
    day_m = re.search(r"day\s*(\d+)", n)
    day = day_m.group(1) if day_m else "unknown"
    return {"patient": patient, "day": day}

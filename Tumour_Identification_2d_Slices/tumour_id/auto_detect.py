"""Stage 2 - Auto-detect tumour regions and save a per-job curation TIF.

For each manifest row with should_process=True:
- Load BF (+ optional FL) 2D slice from the source file.
- Run `identify_tumour` on each to produce a boolean mask.
- Write a 4-channel curation TIF to <output_root>/stage2/<name>.tif:
    ch0 = BF raw (uint8, rescaled)
    ch1 = FL raw (uint8, rescaled) or zeros if no FL
    ch2 = BF tumour mask (uint8: 0 or 1)
    ch3 = FL tumour mask (uint8: 0 or 1) or zeros if no FL
  Pixel size (µm/px) is embedded in the TIF resolution tags.
  <output_root> is the directory containing the manifest CSV.
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage import util
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion, h_maxima, opening, remove_small_holes
from skimage.segmentation import expand_labels, watershed

from .io_utils import load_image_arrays, rescale_intensity

warnings.filterwarnings("ignore")

STAGE2_SUBDIR = "stage2"


# ---------------------------------------------------------------------------
# Core segmentation (ported from notebook; matplotlib debug stripped)
# ---------------------------------------------------------------------------

def identify_tumour(
    image_2d: np.ndarray,
    pixel_size_um: Tuple[float, float],
    image_type: str = "bf",
    max_eccentricity: float = 0.5,
    min_solidity: float = 0.9,
    erosion_radius_um: float = 60.0,
    verbose: bool = False,
) -> dict:
    """Identify the tumour in a 2D image. Returns {rescaled, mask, area_px2, area_um2, eccentricity, split}."""
    if image_2d.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image_2d.shape}")
    pixel_size_y, pixel_size_x = float(pixel_size_um[0]), float(pixel_size_um[1])
    area_per_pixel = pixel_size_y * pixel_size_x

    rescaled = rescale_intensity(
        util.invert(image_2d) if image_type.lower() == "bf" else image_2d,
        0, 255, np.uint8,
    )
    smoothed = gaussian(rescaled, sigma=2, preserve_range=True)
    binary = opening(smoothed > threshold_otsu(smoothed), disk(3))

    labelled = label(binary)
    if labelled.max() == 0:
        return {
            "rescaled": rescaled,
            "mask": np.zeros_like(binary, dtype=bool),
            "area_px2": 0.0,
            "area_um2": 0.0,
            "eccentricity": 0.0,
            "split": False,
        }

    regions = regionprops(labelled, intensity_image=rescaled)
    regions.sort(key=lambda r: r.area, reverse=True)
    initial_candidate = regions[0]
    initial_candidate_mask = labelled == initial_candidate.label
    eccentricity = float(initial_candidate.eccentricity)
    solidity = float(initial_candidate.solidity)
    if verbose:
        print(f"  eccentricity={eccentricity:.3f}  solidity={solidity:.3f}")

    best_mask = initial_candidate_mask
    split = False

    # Use a reasonable erosion radius even when pixel size is unknown
    erosion_radius_px = int(erosion_radius_um / pixel_size_y) if np.isfinite(pixel_size_y) and pixel_size_y > 0 else 20

    # Step 1: low solidity → erode to separate merged blobs
    if solidity < min_solidity and erosion_radius_px > 0:
        eroded_image = erosion(initial_candidate_mask, disk(erosion_radius_px))
        eroded_labels = label(eroded_image)
        if eroded_labels.max() > 1:
            eroded_regions = regionprops(eroded_labels, intensity_image=rescaled)
            eroded_regions.sort(key=lambda r: r.area, reverse=True)
            expanded = expand_labels(label(eroded_labels == eroded_regions[0].label), distance=erosion_radius_px)
            best_mask = expanded > 0
            eccentricity = float(regionprops(label(best_mask))[0].eccentricity)

    # Step 2: high eccentricity → distance-transform watershed
    if eccentricity > max_eccentricity:
        split = True
        mask = remove_small_holes(best_mask, area_threshold=50000)
        distances = ndi.distance_transform_edt(mask)
        coords = peak_local_max(distances, min_distance=50)
        if len(coords) > 0:
            markers = ndi.label(h_maxima(distances, h=7))[0]
            seg = watershed(-distances, markers, mask=mask)
            watershed_regions = regionprops(seg, intensity_image=rescaled)
            if watershed_regions:
                cy, cx = mask.shape[0] / 2, mask.shape[1] / 2
                best = min(
                    watershed_regions,
                    key=lambda r: (r.centroid[0] - cy) ** 2 + (r.centroid[1] - cx) ** 2,
                )
                best_mask = seg == best.label

    final_region = regionprops(label(best_mask))[0] if best_mask.any() else None
    return {
        "rescaled": rescaled,
        "mask": best_mask,
        "area_px2": float(final_region.area) if final_region is not None else 0.0,
        "area_um2": float(final_region.area * area_per_pixel) if final_region is not None and np.isfinite(area_per_pixel) else float("nan"),
        "eccentricity": float(final_region.eccentricity) if final_region is not None else 0.0,
        "split": split,
    }


# ---------------------------------------------------------------------------
# Stage 2 driver
# ---------------------------------------------------------------------------

def _build_curation_stack(
    bf_rescaled: np.ndarray,
    fl_rescaled: Optional[np.ndarray],
    bf_mask: np.ndarray,
    fl_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Stack the 4 channels in (C, Y, X) uint8 order: [BF_raw, FL_raw|0, BF_label, FL_label|0]."""
    shape = bf_rescaled.shape
    zeros = np.zeros(shape, dtype=np.uint8)
    ch_fl_raw = fl_rescaled.astype(np.uint8) if fl_rescaled is not None else zeros
    ch_bf_label = bf_mask.astype(np.uint8)
    ch_fl_label = fl_mask.astype(np.uint8) if fl_mask is not None else zeros
    return np.stack([bf_rescaled.astype(np.uint8), ch_fl_raw, ch_bf_label, ch_fl_label], axis=0)


def _write_curation_tif(
    path: Path,
    stack_cyx: np.ndarray,
    pixel_size_um: Tuple[float, float],
) -> None:
    """Save a 4-channel (CYX) uint8 stack with pixel size embedded in resolution tags.

    pixel_size_um is (y_um, x_um). When either is NaN/non-positive, no resolution
    tags are written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pixel_size_y, pixel_size_x = float(pixel_size_um[0]), float(pixel_size_um[1])
    write_kwargs: dict = dict(imagej=True, metadata={"axes": "CYX", "unit": "um"})
    if np.isfinite(pixel_size_x) and np.isfinite(pixel_size_y) and pixel_size_x > 0 and pixel_size_y > 0:
        # ImageJ resolution = pixels per unit
        write_kwargs["resolution"] = (1.0 / pixel_size_x, 1.0 / pixel_size_y)
    tifffile.imwrite(path, stack_cyx, **write_kwargs)


def process_job(row: pd.Series, output_root: Path, verbose: bool = False) -> dict:
    """Run Stage 2 on a single manifest row. Returns a status dict for logging."""
    stage2_dir = output_root / STAGE2_SUBDIR
    stage2_dir.mkdir(parents=True, exist_ok=True)
    name = str(row["expected_output_name"])

    fl_channel_raw = row["fl_channel"]
    has_fluorescence = fl_channel_raw not in (None, "", float("nan")) and not (isinstance(fl_channel_raw, float) and np.isnan(fl_channel_raw))
    fl_channel = int(fl_channel_raw) if has_fluorescence else None

    status: dict = {
        "job_id": row["job_id"],
        "expected_output_name": name,
        "has_fluorescence": has_fluorescence,
        "bf_error": "",
        "fl_error": "",
        "skip_reason": "",
    }

    try:
        bf_img2d, fl_img2d, pixel_size_um = load_image_arrays(
            source_file=Path(row["source_file"]),
            image_index=int(row["image_index"]),
            bf_channel=int(row["bf_channel"]),
            fl_channel=fl_channel,
        )
    except Exception as exc:
        status["skip_reason"] = f"load_failed: {type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()
        return status

    bf_res = None
    try:
        bf_res = identify_tumour(bf_img2d, pixel_size_um=pixel_size_um, image_type="bf", verbose=verbose)
    except Exception as exc:
        status["bf_error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()

    fl_res = None
    if has_fluorescence and fl_img2d is not None:
        try:
            fl_res = identify_tumour(fl_img2d, pixel_size_um=pixel_size_um, image_type="fluorescent", verbose=verbose)
        except Exception as exc:
            status["fl_error"] = f"{type(exc).__name__}: {exc}"
            if verbose:
                traceback.print_exc()

    if bf_res is None:
        # Fall back so we still produce a curation TIF the user can fix by hand
        bf_rescaled = rescale_intensity(util.invert(bf_img2d), 0, 255, np.uint8)
        bf_mask = np.zeros(bf_rescaled.shape, dtype=bool)
    else:
        bf_rescaled = bf_res["rescaled"]
        bf_mask = bf_res["mask"]

    fl_rescaled = None
    fl_mask = None
    if has_fluorescence and fl_img2d is not None:
        if fl_res is None:
            fl_rescaled = rescale_intensity(fl_img2d, 0, 255, np.uint8)
            fl_mask = np.zeros(fl_rescaled.shape, dtype=bool)
        else:
            fl_rescaled = fl_res["rescaled"]
            fl_mask = fl_res["mask"]

    stack_cyx = _build_curation_stack(bf_rescaled, fl_rescaled, bf_mask, fl_mask)
    _write_curation_tif(stage2_dir / f"{name}.tif", stack_cyx, pixel_size_um)
    return status


def run(manifest_path: str | Path, verbose: bool = False) -> None:
    """Run Stage 2 for every processable row in the manifest."""
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest_df = pd.read_csv(manifest_path)
    processable = manifest_df[manifest_df["should_process"]]
    print(f"Stage 2: processing {len(processable)} / {len(manifest_df)} jobs from {manifest_path}")

    n_ok = 0
    for _, row in processable.iterrows():
        name = row["expected_output_name"]
        try:
            status = process_job(row, output_root=output_root, verbose=verbose)
            if status["skip_reason"]:
                print(f"  [FAIL] {name}: {status['skip_reason']}")
            else:
                errors = []
                if status["bf_error"]:
                    errors.append(f"bf={status['bf_error']}")
                if status["fl_error"]:
                    errors.append(f"fl={status['fl_error']}")
                tag = "OK  " if not errors else "WARN"
                print(f"  [{tag}] {name}" + (f"  ({'; '.join(errors)})" if errors else ""))
                n_ok += 1
        except Exception as exc:
            print(f"  [CRASH] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if verbose:
                traceback.print_exc()
    print(f"Stage 2 done: {n_ok}/{len(processable)} jobs produced outputs.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 - auto-detect tumour regions.")
    parser.add_argument("manifest_path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    run(args.manifest_path, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

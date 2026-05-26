"""Stage 4 - Quantify tumour areas and write a single summary CSV.

For each processable manifest row:
- Prefer <output_root>/stage3/<name>.tif (curated), else
  <output_root>/stage2/<name>.tif (auto).
- Read BF/FL masks (channels 2 and 3) and pixel size from the TIF's
  resolution tags.
- Compute area in pixels and (when pixel size known) in um^2.
- Write a single CSV at <output_root>/tumour_areas.csv.

<output_root> is the directory containing the manifest CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label as sk_label, regionprops

from .io_utils import get_tiff_pixel_size_um

STAGE2_SUBDIR = "stage2"
STAGE3_SUBDIR = "stage3"

CSV_COLUMNS = [
    "image_name",
    "pixel_size_x_um", "pixel_size_y_um",
    "bf_area_px", "bf_area_um2",
    "bf_major_axis_px", "bf_major_axis_um",
    "bf_minor_axis_px", "bf_minor_axis_um",
    "bf_ellipticity",
    "fl_area_px", "fl_area_um2",
    "fl_major_axis_px", "fl_major_axis_um",
    "fl_minor_axis_px", "fl_minor_axis_um",
    "fl_ellipticity",
    "source", "skip_reason",
]


def _resolve_curation_tif(output_root: Path, name: str) -> tuple[Optional[Path], str]:
    """Return (path, source_label). source is 'curated' / 'auto' / 'missing'."""
    override = output_root / STAGE3_SUBDIR / f"{name}.tif"
    if override.exists():
        return override, "curated"
    auto = output_root / STAGE2_SUBDIR / f"{name}.tif"
    if auto.exists():
        return auto, "auto"
    return None, "missing"


def _mask_shape_props(mask: np.ndarray) -> tuple[float, float, float]:
    """Return (major_axis_px, minor_axis_px, ellipticity) for the largest
    connected component in ``mask``, or (nan, nan, nan) when the mask is empty.

    Ellipticity is defined as ``1 - minor_axis / major_axis`` so that
    0 = circle and values approaching 1 = very elongated.
    """
    if not mask.any():
        return float("nan"), float("nan"), float("nan")
    labelled = sk_label(mask)
    props = regionprops(labelled)
    # Pick the largest region by area so noise blobs don't dominate.
    region = max(props, key=lambda r: r.area)
    major = region.major_axis_length
    minor = region.minor_axis_length
    ellipticity = (1.0 - minor / major) if major > 0 else float("nan")
    return major, minor, ellipticity


def quantify_one(curation_tif_path: Path) -> dict:
    """Read a 4-channel curation TIF and compute BF/FL areas.

    Pixel size is read from the TIF's resolution tags.
    """
    with tifffile.TiffFile(curation_tif_path) as tif:
        stack = np.asarray(tif.asarray())
        pixel_size = get_tiff_pixel_size_um(tif)

    if stack.ndim != 3 or stack.shape[0] != 4:
        raise ValueError(f"Unexpected curation TIF shape {stack.shape} in {curation_tif_path}")

    if pixel_size is None:
        pixel_size_y_um, pixel_size_x_um = float("nan"), float("nan")
    else:
        pixel_size_y_um, pixel_size_x_um = pixel_size

    fl_raw = stack[1].astype(np.uint8)
    bf_mask = stack[2].astype(bool)
    fl_mask = stack[3].astype(bool)
    has_fluorescence = bool(fl_raw.any())

    area_per_px = pixel_size_x_um * pixel_size_y_um
    pixel_size_known = np.isfinite(area_per_px) and area_per_px > 0
    # pixel_size_um is the mean linear scale (used to convert axis lengths)
    linear_scale = float(np.sqrt(area_per_px)) if pixel_size_known else float("nan")

    bf_area_px = int(bf_mask.sum())
    bf_area_um2 = float(bf_area_px * area_per_px) if pixel_size_known else float("nan")
    bf_major_px, bf_minor_px, bf_ellipticity = _mask_shape_props(bf_mask)
    bf_major_um = bf_major_px * linear_scale if np.isfinite(bf_major_px) else float("nan")
    bf_minor_um = bf_minor_px * linear_scale if np.isfinite(bf_minor_px) else float("nan")

    if has_fluorescence:
        fl_area_px: Optional[int] = int(fl_mask.sum())
        fl_area_um2: Optional[float] = float(fl_area_px * area_per_px) if pixel_size_known else float("nan")
        fl_major_px, fl_minor_px, fl_ellipticity = _mask_shape_props(fl_mask)
        fl_major_um = fl_major_px * linear_scale if np.isfinite(fl_major_px) else float("nan")
        fl_minor_um = fl_minor_px * linear_scale if np.isfinite(fl_minor_px) else float("nan")
    else:
        fl_area_px = None
        fl_area_um2 = None
        fl_major_px = fl_minor_px = fl_ellipticity = None
        fl_major_um = fl_minor_um = None

    return {
        "pixel_size_x_um": pixel_size_x_um,
        "pixel_size_y_um": pixel_size_y_um,
        "bf_area_px": bf_area_px,
        "bf_area_um2": bf_area_um2,
        "bf_major_axis_px": bf_major_px,
        "bf_major_axis_um": bf_major_um,
        "bf_minor_axis_px": bf_minor_px,
        "bf_minor_axis_um": bf_minor_um,
        "bf_ellipticity": bf_ellipticity,
        "fl_area_px": fl_area_px,
        "fl_area_um2": fl_area_um2,
        "fl_major_axis_px": fl_major_px,
        "fl_major_axis_um": fl_major_um,
        "fl_minor_axis_px": fl_minor_px,
        "fl_minor_axis_um": fl_minor_um,
        "fl_ellipticity": fl_ellipticity,
    }


def run(manifest_path: str | Path, csv_path: Optional[str | Path] = None, n_workers: int = 1) -> Path:
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest_df = pd.read_csv(manifest_path)

    if csv_path is None:
        csv_path = output_root / "tumour_areas.csv"
    csv_path = Path(csv_path)

    # Build the list of (row, curation_tif, source) up front so the parallel
    # phase only has to do the heavy reads.
    jobs: List[tuple] = []
    rows: List[dict] = []
    for row in manifest_df.itertuples(index=False):
        name = str(row.expected_output_name)
        if not row.should_process:
            rows.append({
                "image_name": name,
                "pixel_size_x_um": float("nan"),
                "pixel_size_y_um": float("nan"),
                "bf_area_px": None, "bf_area_um2": None,
                "bf_major_axis_px": None, "bf_major_axis_um": None,
                "bf_minor_axis_px": None, "bf_minor_axis_um": None,
                "bf_ellipticity": None,
                "fl_area_px": None, "fl_area_um2": None,
                "fl_major_axis_px": None, "fl_major_axis_um": None,
                "fl_minor_axis_px": None, "fl_minor_axis_um": None,
                "fl_ellipticity": None,
                "source": "skipped",
                "skip_reason": row.skip_reason,
            })
            continue

        curation_tif, source = _resolve_curation_tif(output_root, name)
        if curation_tif is None:
            rows.append({
                "image_name": name,
                "pixel_size_x_um": float("nan"),
                "pixel_size_y_um": float("nan"),
                "bf_area_px": None, "bf_area_um2": None,
                "bf_major_axis_px": None, "bf_major_axis_um": None,
                "bf_minor_axis_px": None, "bf_minor_axis_um": None,
                "bf_ellipticity": None,
                "fl_area_px": None, "fl_area_um2": None,
                "fl_major_axis_px": None, "fl_major_axis_um": None,
                "fl_minor_axis_px": None, "fl_minor_axis_um": None,
                "fl_ellipticity": None,
                "source": "missing",
                "skip_reason": "no_stage2_or_stage3_output",
            })
            continue
        jobs.append((name, curation_tif, source))

    def _quantify_one(name: str, curation_tif: Path, source: str) -> dict:
        try:
            quant = quantify_one(curation_tif)
        except Exception as exc:
            return {
                "image_name": name,
                "pixel_size_x_um": float("nan"),
                "pixel_size_y_um": float("nan"),
                "bf_area_px": None, "bf_area_um2": None,
                "bf_major_axis_px": None, "bf_major_axis_um": None,
                "bf_minor_axis_px": None, "bf_minor_axis_um": None,
                "bf_ellipticity": None,
                "fl_area_px": None, "fl_area_um2": None,
                "fl_major_axis_px": None, "fl_major_axis_um": None,
                "fl_minor_axis_px": None, "fl_minor_axis_um": None,
                "fl_ellipticity": None,
                "source": source,
                "skip_reason": f"quantify_failed: {exc}",
            }
        return {
            "image_name": name,
            "pixel_size_x_um": quant["pixel_size_x_um"],
            "pixel_size_y_um": quant["pixel_size_y_um"],
            "bf_area_px": quant["bf_area_px"],
            "bf_area_um2": quant["bf_area_um2"],
            "bf_major_axis_px": quant["bf_major_axis_px"],
            "bf_major_axis_um": quant["bf_major_axis_um"],
            "bf_minor_axis_px": quant["bf_minor_axis_px"],
            "bf_minor_axis_um": quant["bf_minor_axis_um"],
            "bf_ellipticity": quant["bf_ellipticity"],
            "fl_area_px": quant["fl_area_px"],
            "fl_area_um2": quant["fl_area_um2"],
            "fl_major_axis_px": quant["fl_major_axis_px"],
            "fl_major_axis_um": quant["fl_major_axis_um"],
            "fl_minor_axis_px": quant["fl_minor_axis_px"],
            "fl_minor_axis_um": quant["fl_minor_axis_um"],
            "fl_ellipticity": quant["fl_ellipticity"],
            "source": source,
            "skip_reason": "",
        }

    if n_workers and n_workers > 1 and jobs:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, prefer="processes", verbose=0)(
            delayed(_quantify_one)(name, curation_tif, source)
            for name, curation_tif, source in jobs
        )
    else:
        results = [_quantify_one(name, curation_tif, source) for name, curation_tif, source in jobs]

    for result in results:
        if result["skip_reason"].startswith("quantify_failed"):
            print(f"  [FAIL] {result['image_name']}: {result['skip_reason']}", file=sys.stderr)
        rows.append(result)

    results_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Stage 4 done: wrote {csv_path}  ({len(results_df)} rows)")
    return csv_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4 - quantify tumour areas.")
    parser.add_argument("manifest_path")
    parser.add_argument("--csv", default=None, help="output CSV path (default: <manifest_dir>/tumour_areas.csv)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1).")
    args = parser.parse_args(argv)
    run(args.manifest_path, csv_path=args.csv, n_workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


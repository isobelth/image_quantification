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

from .io_utils import get_tiff_pixel_size_um

STAGE2_SUBDIR = "stage2"
STAGE3_SUBDIR = "stage3"

CSV_COLUMNS = [
    "image_name",
    "pixel_size_x_um", "pixel_size_y_um",
    "bf_area_px", "bf_area_um2",
    "fl_area_px", "fl_area_um2",
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

    bf_area_px = int(bf_mask.sum())
    bf_area_um2 = float(bf_area_px * area_per_px) if pixel_size_known else float("nan")

    if has_fluorescence:
        fl_area_px: Optional[int] = int(fl_mask.sum())
        fl_area_um2: Optional[float] = float(fl_area_px * area_per_px) if pixel_size_known else float("nan")
    else:
        fl_area_px = None
        fl_area_um2 = None

    return {
        "pixel_size_x_um": pixel_size_x_um,
        "pixel_size_y_um": pixel_size_y_um,
        "bf_area_px": bf_area_px,
        "bf_area_um2": bf_area_um2,
        "fl_area_px": fl_area_px,
        "fl_area_um2": fl_area_um2,
    }


def run(manifest_path: str | Path, csv_path: Optional[str | Path] = None) -> Path:
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest_df = pd.read_csv(manifest_path)

    if csv_path is None:
        csv_path = output_root / "tumour_areas.csv"
    csv_path = Path(csv_path)

    rows: List[dict] = []
    for row in manifest_df.itertuples(index=False):
        name = str(row.expected_output_name)
        if not row.should_process:
            rows.append({
                "image_name": name,
                "pixel_size_x_um": float("nan"),
                "pixel_size_y_um": float("nan"),
                "bf_area_px": None, "bf_area_um2": None,
                "fl_area_px": None, "fl_area_um2": None,
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
                "fl_area_px": None, "fl_area_um2": None,
                "source": "missing",
                "skip_reason": "no_stage2_or_stage3_output",
            })
            continue

        try:
            quant = quantify_one(curation_tif)
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}", file=sys.stderr)
            rows.append({
                "image_name": name,
                "pixel_size_x_um": float("nan"),
                "pixel_size_y_um": float("nan"),
                "bf_area_px": None, "bf_area_um2": None,
                "fl_area_px": None, "fl_area_um2": None,
                "source": source,
                "skip_reason": f"quantify_failed: {exc}",
            })
            continue

        rows.append({
            "image_name": name,
            "pixel_size_x_um": quant["pixel_size_x_um"],
            "pixel_size_y_um": quant["pixel_size_y_um"],
            "bf_area_px": quant["bf_area_px"],
            "bf_area_um2": quant["bf_area_um2"],
            "fl_area_px": quant["fl_area_px"],
            "fl_area_um2": quant["fl_area_um2"],
            "source": source,
            "skip_reason": "",
        })

    results_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Stage 4 done: wrote {csv_path}  ({len(results_df)} rows)")
    return csv_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4 - quantify tumour areas.")
    parser.add_argument("manifest_path")
    parser.add_argument("--csv", default=None, help="output CSV path (default: <manifest_dir>/tumour_areas.csv)")
    args = parser.parse_args(argv)
    run(args.manifest_path, csv_path=args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


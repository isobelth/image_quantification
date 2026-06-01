"""Stage 4 — Count cells, compute distances and colocalisation, write CSVs.

For each processable manifest row, reads the curation TIF (preferring
stage3/ over stage2/), rebuilds canonical regions, counts CART cells and
B cells per region, computes per-cell distances from the tumour edge, and
computes two colocalisation metrics between CART and B cells.

Outputs (written to ``output_dir/``):
    all_counts.csv          — rows: image × cell_type × region
    all_distances.csv       — rows: individual cells with distance + region flags
    all_colocalisation.csv  — rows: method × region × scale parameter

Usage:
    python -m cart_id.analyse <manifest_path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as _stats
from scipy.ndimage import distance_transform_edt
from skimage.filters import gaussian
from skimage.measure import label, regionprops_table

from .io_utils import read_curation_tif

STAGE2_SUBDIR = "stage2"
STAGE3_SUBDIR = "stage3"

# Channel indices in the 11-channel TIF
_CH_BF = 0
_CH_BCELL_RAW = 1
_CH_VASC_RAW = 2
_CH_CART_RAW = 3
_CH_CHIP = 4
_CH_TUMOUR = 5
_CH_LEFT = 6
_CH_RIGHT = 7
_CH_VASC_MASK = 8
_CH_BCELL_MASK = 9
_CH_CART_MASK = 10

# Colocalisation parameters (mirrored from the original notebook)
# Sigma capped at 50 um: beyond the cellular structure scale the blurred
# channels become near-constant within a region, making Pearson undefined.
GAUSS_SIGMAS_UM = [5, 10, 20, 30, 50]
PROXIMITY_RADII_UM = [5, 10, 20, 30, 40, 50, 100]

COUNTS_COLUMNS = [
    "patient", "day", "lif_file", "image", "pixel_size_um",
    "cell_type", "region", "cell_count",
    "positive_pixels", "positive_area_um2",
    "region_area_px", "region_area_um2", "positive_pixel_pct",
]
DISTANCES_COLUMNS = [
    "patient", "day", "lif_file", "image", "pixel_size_um",
    "cell_type", "distance_from_tumour_um",
    "in_chip", "in_not_chip", "in_tumour", "in_chip_not_tumour",
    "in_chip_vasculature", "in_chip_not_vasculature",
    "in_left", "in_right",
]
COLOC_COLUMNS = [
    "patient", "day", "lif_file", "image", "pixel_size_um",
    "method", "param_um", "region", "score",
]


def _resolve_tif(output_root: Path, name: str):
    override = output_root / STAGE3_SUBDIR / f"{name}.tif"
    if override.exists():
        return override, "curated"
    auto = output_root / STAGE2_SUBDIR / f"{name}.tif"
    if auto.exists():
        return auto, "auto"
    return None, "missing"


def _count_cells(region_mask: np.ndarray, crow: np.ndarray, ccol: np.ndarray) -> int:
    if len(crow) == 0:
        return 0
    return int(region_mask[crow, ccol].sum())


def _build_canonical_regions(
    chip_center: np.ndarray,
    tumour: np.ndarray,
    vasc: np.ndarray,
    left_device: np.ndarray,
    right_device: np.ndarray,
) -> dict:
    chip = chip_center.astype(bool)
    tum = tumour.astype(bool)
    v = vasc.astype(bool)
    left = left_device.astype(bool)
    right = right_device.astype(bool)
    not_chip = left | right
    return {
        "chip": chip,
        "not_chip": not_chip,
        "tumour": chip & tum,
        "chip_not_tumour": chip & ~tum,
        "chip_vasculature": chip & v,
        "chip_not_vasculature": chip & ~v,
        "left": left,
        "right": right,
    }


def analyse_one(
    tif_path: Path,
    meta: dict,
    image_name: str,
    um_per_px: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_counts, df_distances, df_coloc) for a single curation TIF."""
    stack, _pixel_size = read_curation_tif(tif_path)

    # Read masks
    chip_center = stack[_CH_CHIP].astype(bool)
    tumour_mask = stack[_CH_TUMOUR].astype(bool)
    left_mask = stack[_CH_LEFT].astype(bool)
    right_mask = stack[_CH_RIGHT].astype(bool)
    vasc_mask = stack[_CH_VASC_MASK].astype(bool)
    bcell_mask = stack[_CH_BCELL_MASK].astype(bool)
    cart_mask = stack[_CH_CART_MASK].astype(bool)
    bcell_raw = stack[_CH_BCELL_RAW].astype(float)
    cart_raw = stack[_CH_CART_RAW].astype(float)

    dist_from_tumour = distance_transform_edt(~tumour_mask)
    regions = _build_canonical_regions(chip_center, tumour_mask, vasc_mask, left_mask, right_mask)
    h, w = chip_center.shape
    um2_per_px = float(um_per_px) ** 2

    count_records: List[dict] = []
    dist_records: List[dict] = []

    for cell_type, cell_mask in [("CART cells", cart_mask), ("B cells", bcell_mask)]:
        cell_mask = cell_mask.astype(bool)
        labelled_cells = label(cell_mask)
        total_cells = int(labelled_cells.max())
        total_pixels = int(cell_mask.sum())

        if total_cells > 0:
            cprops = pd.DataFrame(regionprops_table(labelled_cells, properties=("centroid",)))
            crow = np.clip(np.round(cprops["centroid-0"].values).astype(int), 0, h - 1)
            ccol = np.clip(np.round(cprops["centroid-1"].values).astype(int), 0, w - 1)
        else:
            crow = np.array([], dtype=int)
            ccol = np.array([], dtype=int)

        # Overall count
        overall_area = int(h * w)
        count_records.append(dict(
            **meta, image=image_name, cell_type=cell_type,
            region="overall", cell_count=total_cells,
            positive_pixels=total_pixels,
            positive_area_um2=total_pixels * um2_per_px,
            region_area_px=overall_area,
            region_area_um2=overall_area * um2_per_px,
            positive_pixel_pct=(100.0 * total_pixels / overall_area) if overall_area else np.nan,
        ))

        # Per-region counts
        for region_name, region_mask in regions.items():
            n_cells = _count_cells(region_mask, crow, ccol)
            n_pixels = int(np.sum(cell_mask & region_mask))
            area_px = int(region_mask.sum())
            count_records.append(dict(
                **meta, image=image_name, cell_type=cell_type,
                region=region_name, cell_count=n_cells,
                positive_pixels=n_pixels,
                positive_area_um2=n_pixels * um2_per_px,
                region_area_px=area_px,
                region_area_um2=area_px * um2_per_px,
                positive_pixel_pct=(100.0 * n_pixels / area_px) if area_px else np.nan,
            ))

        # Per-cell distances + region membership
        if total_cells > 0:
            dists = dist_from_tumour[crow, ccol] * um_per_px
            for r, c, d in zip(crow, ccol, dists):
                rec = dict(
                    **meta, image=image_name, cell_type=cell_type,
                    distance_from_tumour_um=float(d),
                )
                for rname, rmask in regions.items():
                    rec[f"in_{rname}"] = bool(rmask[r, c])
                dist_records.append(rec)

    # --- Colocalisation ---
    coloc_records: List[dict] = []

    # Method 1: Coarse Pearson correlation (Gaussian blur)
    for sigma_um in GAUSS_SIGMAS_UM:
        sigma_px = sigma_um / um_per_px if um_per_px > 0 else sigma_um
        b_blur = gaussian(bcell_raw, sigma=sigma_px)
        c_blur = gaussian(cart_raw, sigma=sigma_px)
        for rname, rmask in regions.items():
            b_vals = b_blur[rmask]
            c_vals = c_blur[rmask]
            r_val = np.nan if len(b_vals) < 20 else float(_stats.pearsonr(b_vals, c_vals)[0])
            coloc_records.append(dict(
                **meta, image=image_name,
                method="coarse_pearson", param_um=sigma_um, region=rname, score=r_val,
            ))

    # Method 2: Binary proximity fraction (EDT)
    cart_edt = distance_transform_edt(~cart_mask)
    b_edt = distance_transform_edt(~bcell_mask)
    for radius_um in PROXIMITY_RADII_UM:
        radius_px = radius_um / um_per_px if um_per_px > 0 else radius_um
        cart_within = cart_edt <= radius_px
        b_within = b_edt <= radius_px
        for rname, rmask in regions.items():
            b_in = bcell_mask & rmask
            n_b = int(b_in.sum())
            frac_b = float((b_in & cart_within).sum()) / n_b if n_b > 0 else np.nan
            c_in = cart_mask & rmask
            n_c = int(c_in.sum())
            frac_c = float((c_in & b_within).sum()) / n_c if n_c > 0 else np.nan
            coloc_records.append(dict(
                **meta, image=image_name,
                method="B_near_CART", param_um=radius_um, region=rname, score=frac_b,
            ))
            coloc_records.append(dict(
                **meta, image=image_name,
                method="CART_near_B", param_um=radius_um, region=rname, score=frac_c,
            ))

    return (
        pd.DataFrame(count_records),
        pd.DataFrame(dist_records),
        pd.DataFrame(coloc_records),
    )


def run(
    manifest_path: str | Path,
    output_dir: Optional[str | Path] = None,
    n_workers: int = 1,
) -> dict:
    """Run Stage 4 for every processable job and write 3 CSVs.

    Returns a dict with keys ``counts_csv``, ``distances_csv``, ``coloc_csv``.
    """
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    if output_dir is None:
        output_dir = output_root
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    processable = manifest_df[manifest_df["should_process"]]
    print(f"Stage 4: analysing {len(processable)}/{len(manifest_df)} images (n_workers={n_workers})")

    jobs = []
    for row in processable.itertuples(index=False):
        name = str(row.expected_output_name)
        tif_path, source = _resolve_tif(output_root, name)
        if tif_path is None:
            print(f"  [SKIP] {name}: no stage2 or stage3 TIF", file=sys.stderr)
            continue
        # Read pixel size from the TIF resolution tags
        try:
            import tifffile
            from .io_utils import get_tiff_pixel_size_um
            with tifffile.TiffFile(str(tif_path)) as tif:
                pixel_size = get_tiff_pixel_size_um(tif)
            if pixel_size is not None:
                um_per_px = float(np.mean(pixel_size))
            else:
                um_per_px = 1.0
        except Exception:
            um_per_px = 1.0
        meta = {
            "patient": str(getattr(row, "patient", "unknown") or "unknown"),
            "day": str(getattr(row, "day", "unknown") or "unknown"),
            "lif_file": Path(str(row.source_file)).stem if row.source_file else "",
            "pixel_size_um": um_per_px,
        }
        jobs.append((tif_path, meta, str(row.image_name), um_per_px, name))

    def _analyse_one(tif_path, meta, image_name, um_per_px, name):
        try:
            df_counts, df_dists, df_coloc = analyse_one(tif_path, meta, image_name, um_per_px)
            return name, df_counts, df_dists, df_coloc, None
        except Exception as exc:
            return name, None, None, None, f"{type(exc).__name__}: {exc}"

    if n_workers and n_workers > 1 and jobs:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, prefer="processes", verbose=0)(
            delayed(_analyse_one)(tif_path, meta, image_name, um_per_px, name)
            for tif_path, meta, image_name, um_per_px, name in jobs
        )
    else:
        results = [
            _analyse_one(tif_path, meta, image_name, um_per_px, name)
            for tif_path, meta, image_name, um_per_px, name in jobs
        ]

    all_counts = []
    all_dists = []
    all_coloc = []
    n_ok = 0
    for name, df_c, df_d, df_col, err in results:
        if err:
            print(f"  [FAIL] {name}: {err}", file=sys.stderr)
        else:
            all_counts.append(df_c)
            all_dists.append(df_d)
            all_coloc.append(df_col)
            n_ok += 1

    counts_csv = output_dir / "all_counts.csv"
    dists_csv = output_dir / "all_distances.csv"
    coloc_csv = output_dir / "all_colocalisation.csv"

    def _safe_concat(frames: list) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    _safe_concat(all_counts).to_csv(counts_csv, index=False)
    _safe_concat(all_dists).to_csv(dists_csv, index=False)
    _safe_concat(all_coloc).to_csv(coloc_csv, index=False)

    print(f"Stage 4 done: {n_ok}/{len(jobs)} images analysed.")
    print(f"  {counts_csv}")
    print(f"  {dists_csv}")
    print(f"  {coloc_csv}")
    return {"counts_csv": counts_csv, "distances_csv": dists_csv, "coloc_csv": coloc_csv}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4 - analyse CART data.")
    parser.add_argument("manifest_path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    run(args.manifest_path, output_dir=args.output_dir, n_workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

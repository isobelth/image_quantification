"""Stage 2 — Auto-detect all segmentation masks and write curation TIFs.

For each processable manifest row, loads the LIF image, runs the full
segmentation pipeline (chip device, vasculature, tumour, B cells, CART cells),
and writes an 11-channel uint8 TIF to ``stage2/<expected_output_name>.tif``.

Channel layout: see cart_id/io_utils.py.
"""

from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import (
    gaussian,
    threshold_li,
    threshold_mean,
    threshold_triangle,
)
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import (
    closing,
    disk,
    erosion,
    h_maxima,
    remove_small_holes,
    remove_small_objects,
    white_tophat,
)
from skimage.segmentation import expand_labels, watershed
from skimage.filters import threshold_yen
from skimage.filters import median as median_filter
from skimage.morphology import square

from .io_utils import (
    load_lif_channels,
    parse_filename_metadata,
    rescale_uint8,
    write_curation_tif,
)

STAGE2_SUBDIR = "stage2"

# Fixed channel indices inside the LIF CYX stack
CH_BCELLS = 0
CH_VASC = 1
CH_BF = 2
# CART cells = last channel

# ---------------------------------------------------------------------------
# Individual segmentation functions (ported verbatim from humanitl notebook)
# ---------------------------------------------------------------------------


def _best_tumour(lbl_img: np.ndarray, img_center: Tuple[float, float]) -> Optional[int]:
    """Pick the tumour region label with the highest area × circularity / distance score."""
    props = pd.DataFrame(regionprops_table(
        lbl_img, properties=("label", "area", "perimeter", "centroid")))
    props = props[props["area"] >= 0.01 * lbl_img.size] if not props.empty else props
    if props.empty:
        return None
    circ = (4 * np.pi * props["area"]) / (props["perimeter"] ** 2 + 1e-9)
    dist = np.hypot(
        props["centroid-0"] - img_center[0],
        props["centroid-1"] - img_center[1],
    )
    props = props.copy()
    props["score"] = props["area"] * circ / (1 + dist)
    return int(props.loc[props["score"].idxmax(), "label"])


def segment_chip(
    bf_2d: np.ndarray,
    save_diag_path: Optional[Path] = None,
) -> dict:
    """Segment the chip device from the brightfield channel.

    Returns a dict with:
        chip_center  (bool 2-D) — interior channel
        left_device  (bool 2-D) — left PDMS wall
        right_device (bool 2-D) — right PDMS wall
    """
    clipped = np.clip(
        bf_2d,
        np.percentile(bf_2d, 25),
        np.percentile(bf_2d, 90),
    )
    smoothed = gaussian(clipped, sigma=7)
    bright = smoothed > threshold_li(smoothed)
    labelled_bright = label(bright)
    h = smoothed.shape[0]

    # --- left wall ---
    left_labels = set(labelled_bright[:, 0]) - {0}
    left_region = np.isin(labelled_bright, list(left_labels))
    left_region = closing(left_region, disk(10))
    left_region = remove_small_holes(left_region, max_size=500_000)

    if left_region[:, 0].sum() > h / 2:
        bright_mean = smoothed > threshold_mean(smoothed)
        lbl_mean = label(bright_mean)
        ll = set(lbl_mean[:, 0]) - {0}
        left_region = np.isin(lbl_mean, list(ll))
        left_region = closing(left_region, disk(10))
        left_region = remove_small_holes(left_region, max_size=500_000)

    # --- right wall ---
    right_labels = set(labelled_bright[:, -1]) - {0}
    right_region = np.isin(labelled_bright, list(right_labels))
    right_region = closing(right_region, disk(10))
    right_region = remove_small_holes(right_region, max_size=500_000)

    if right_region[:, -1].sum() > h / 2:
        bright_mean = smoothed > threshold_mean(smoothed)
        lbl_mean = label(bright_mean)
        rl = set(lbl_mean[:, -1]) - {0}
        right_region = np.isin(lbl_mean, list(rl))
        right_region = closing(right_region, disk(10))
        right_region = remove_small_holes(right_region, max_size=500_000)

    chip_center = (~left_region & ~right_region).astype(bool)

    return {
        "chip_center": chip_center,
        "left_device": left_region.astype(bool),
        "right_device": right_region.astype(bool),
        "smoothed_bf": smoothed,  # needed for tumour segmentation
    }


def segment_vasculature(vasc_2d: np.ndarray) -> np.ndarray:
    """Segment vasculature using triangle threshold with Li fallback.

    Returns a boolean mask.
    """
    vasc_med = median_filter(vasc_2d, footprint=square(7))
    smooth_vasc = gaussian(vasc_med, sigma=3)
    seg = smooth_vasc > threshold_mean(smooth_vasc)
    return seg.astype(bool)

    
def segment_cells(channel_2d: np.ndarray) -> np.ndarray:
    """Segment cells using Yen's threshold. Returns a boolean mask."""

    bg_removed = white_tophat(channel_2d, footprint=disk(5))
    cell_seg = bg_removed > threshold_triangle(bg_removed)
    cell_seg = remove_small_objects(cell_seg, min_size=2)
    return cell_seg.astype(bool)


def segment_tumour(
    smoothed_bf: np.ndarray,
    chip_center: np.ndarray,
    image_name: str = "",
    save_diag_path: Optional[Path] = None,
) -> np.ndarray:
    """Segment the tumour region from the smoothed brightfield inside the chip.

    Uses triangle threshold, _best_tumour selection, two erosion/expansion
    solidity passes, and watershed splitting. Returns a boolean mask.

    If *save_diag_path* is given, writes a watershed diagnostic PNG there.
    """
    img_center = (smoothed_bf.shape[0] / 2, smoothed_bf.shape[1] / 2)

    tumour_raw = (smoothed_bf < threshold_triangle(smoothed_bf)) & chip_center
    lbl = label(tumour_raw)
    chosen = _best_tumour(lbl, img_center)
    tumour = (lbl == chosen).astype(bool) if chosen is not None else tumour_raw.astype(bool)

    # Solidity pass 1: erode 10, expand 10
    tumour_rp = regionprops(label(tumour))
    if tumour_rp and tumour_rp[0].solidity < 0.9:
        eroded = erosion(tumour, disk(10))
        e_lbl = label(eroded)
        chosen_e = _best_tumour(e_lbl, img_center)
        if chosen_e is not None:
            tumour = (expand_labels(e_lbl, distance=10) == chosen_e).astype(bool)

        # Solidity pass 2: erode 50, expand 60
        tumour_rp = regionprops(label(tumour))
        if tumour_rp and tumour_rp[0].solidity < 0.9:
            eroded = erosion(tumour, disk(50))
            e_lbl = label(eroded)
            chosen_e = _best_tumour(e_lbl, img_center)
            if chosen_e is not None:
                tumour = (expand_labels(e_lbl, distance=60) == chosen_e).astype(bool)

    # Watershed to split merged regions
    tumour_mask = remove_small_holes(tumour, max_size=500_000)
    if tumour_mask.any():
        smooth_ws = gaussian(smoothed_bf, sigma=10)
        edt_tumour = ndi.distance_transform_edt(tumour_mask)
        markers = ndi.label(h_maxima(edt_tumour, h=7))[0]
        if markers.max() >= 2:
            ws_labels = watershed(smooth_ws, markers, mask=tumour_mask)
            ws_medians = {
                v: float(np.median(smoothed_bf[ws_labels == v]))
                for v in np.unique(ws_labels[ws_labels > 0])
            }
            darkest = min(ws_medians, key=ws_medians.get)
            tumour = (ws_labels == darkest).astype(bool)
            ws_darkest = tumour

            # --- optional watershed diagnostic figure (only when split attempted) ---
            if save_diag_path is not None:
                _save_watershed_diag(
                    save_diag_path, image_name, smoothed_bf,
                    tumour_mask, edt_tumour, markers, ws_labels, ws_darkest, darkest,
                )

    return tumour.astype(bool)


def _save_watershed_diag(
    path: Path,
    image_name: str,
    smoothed: np.ndarray,
    tumour_mask: np.ndarray,
    edt_tumour: np.ndarray,
    markers: np.ndarray,
    ws_labels: np.ndarray,
    ws_darkest: np.ndarray,
    darkest_label,
) -> None:
    seed_rps = regionprops(markers)
    seed_coords = np.array([rp.centroid for rp in seed_rps]) if seed_rps else np.empty((0, 2))

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), constrained_layout=True)
    fig.suptitle(f"Watershed tumour split — {image_name}", fontsize=11, fontweight="bold")

    disp = np.zeros_like(smoothed)
    disp[tumour_mask] = smoothed[tumour_mask]
    axes[0].imshow(disp, cmap="gray")
    axes[0].set_title("Smoothed (tumour only)", fontsize=9)
    axes[0].axis("off")

    edt_disp = np.zeros_like(edt_tumour)
    edt_disp[tumour_mask] = edt_tumour[tumour_mask]
    axes[1].imshow(edt_disp, cmap="hot")
    if len(seed_coords) >= 2:
        axes[1].scatter(seed_coords[:, 1], seed_coords[:, 0], c="cyan", s=60,
                        edgecolors="white", linewidths=1, zorder=5)
    axes[1].set_title(f"EDT + seeds ({len(seed_coords)} found)", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(smoothed, cmap="gray")
    ws_disp = np.where(tumour_mask, ws_labels, 0).astype(float)
    ws_disp[ws_disp == 0] = np.nan
    axes[2].imshow(ws_disp, cmap="tab10", alpha=0.3)
    axes[2].set_title(f"Watershed regions ({int(ws_labels.max())})", fontsize=9)
    axes[2].axis("off")

    for v in sorted(np.unique(ws_labels[ws_labels > 0])):
        vals = smoothed[ws_labels == v]
        axes[3].hist(vals, bins=60, alpha=0.5, label=f"Region {v} (med={np.median(vals):.3f})")
    axes[3].set_title("Intensity per region", fontsize=9)
    axes[3].set_xlabel("Smoothed intensity")
    axes[3].set_ylabel("Count")
    axes[3].legend(fontsize=7)

    axes[4].imshow(smoothed, cmap="gray")
    overlay = np.zeros((*smoothed.shape, 4), dtype=float)
    overlay[ws_darkest] = [0, 1, 0, 0.3]
    axes[4].imshow(overlay)
    axes[4].set_title(
        f"Kept: region {darkest_label}" if darkest_label else "Single region", fontsize=9
    )
    axes[4].axis("off")

    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-job driver
# ---------------------------------------------------------------------------

def process_job(
    row: pd.Series,
    output_root: Path,
    ch_bf: int = CH_BF,
    ch_bcells: int = CH_BCELLS,
    ch_vasc: int = CH_VASC,
    figures_dir: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """Segment one image and write stage2/<name>.tif.  Returns a status dict."""
    name = str(row["expected_output_name"])
    lif_path = Path(str(row["source_file"]))
    image_index = int(row["image_index"])
    image_name = str(row["image_name"])
    n_channels = int(row["n_channels"])

    status = {
        "expected_output_name": name,
        "skip_reason": "",
        "error": "",
    }

    try:
        stack_cyx, um_per_px = load_lif_channels(lif_path, image_index)
    except Exception as exc:
        status["skip_reason"] = "load_failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()
        return status

    if stack_cyx.shape[0] < 4:
        status["skip_reason"] = f"too_few_channels ({stack_cyx.shape[0]})"
        return status

    bf_raw = stack_cyx[ch_bf]
    bcell_raw = stack_cyx[ch_bcells]
    vasc_raw = stack_cyx[ch_vasc]
    cart_raw = stack_cyx[-1]  # last channel

    # Pixel size as (y_um, x_um) tuple for TIF writing
    pixel_size_um = (um_per_px, um_per_px)

    # ---- Segmentation ----
    try:
        chip_result = segment_chip(bf_raw)
    except Exception as exc:
        status["skip_reason"] = "chip_seg_failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()
        return status

    chip_center = chip_result["chip_center"]
    left_device = chip_result["left_device"]
    right_device = chip_result["right_device"]
    smoothed_bf = chip_result["smoothed_bf"]

    try:
        vasc_mask = segment_vasculature(vasc_raw)
    except Exception as exc:
        warnings.warn(f"[WARN] {name}: vasculature segmentation failed ({exc}); using zeros")
        vasc_mask = np.zeros(bf_raw.shape, dtype=bool)

    try:
        bcell_mask = segment_cells(bcell_raw)
    except Exception as exc:
        warnings.warn(f"[WARN] {name}: B-cell segmentation failed ({exc}); using zeros")
        bcell_mask = np.zeros(bf_raw.shape, dtype=bool)

    try:
        cart_mask = segment_cells(cart_raw)
    except Exception as exc:
        warnings.warn(f"[WARN] {name}: CART-cell segmentation failed ({exc}); using zeros")
        cart_mask = np.zeros(bf_raw.shape, dtype=bool)

    diag_path = (figures_dir / f"{image_name.replace('/', '_')}_diag_watershed.png") if figures_dir else None
    try:
        tumour_mask = segment_tumour(
            smoothed_bf, chip_center,
            image_name=image_name,
            save_diag_path=diag_path,
        )
    except Exception as exc:
        warnings.warn(f"[WARN] {name}: tumour segmentation failed ({exc}); using zeros")
        tumour_mask = np.zeros(bf_raw.shape, dtype=bool)

    # ---- Build 11-channel stack ----
    stack_11ch = np.stack([
        rescale_uint8(bf_raw),          # ch0
        rescale_uint8(bcell_raw),       # ch1
        rescale_uint8(vasc_raw),        # ch2
        rescale_uint8(cart_raw),        # ch3
        chip_center.astype(np.uint8),   # ch4  [editable]
        tumour_mask.astype(np.uint8),   # ch5  [editable]
        left_device.astype(np.uint8),   # ch6
        right_device.astype(np.uint8),  # ch7
        vasc_mask.astype(np.uint8),     # ch8
        bcell_mask.astype(np.uint8),    # ch9
        cart_mask.astype(np.uint8),     # ch10
    ], axis=0)

    out_dir = output_root / STAGE2_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    write_curation_tif(out_dir / f"{name}.tif", stack_11ch, pixel_size_um)

    return status


def run(
    manifest_path: str | Path,
    verbose: bool = False,
    n_workers: int = 1,
    ch_bf: int = CH_BF,
    ch_bcells: int = CH_BCELLS,
    ch_vasc: int = CH_VASC,
    save_diag_figures: bool = True,
) -> None:
    """Run Stage 2 for every processable row in the manifest."""
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest_df = pd.read_csv(manifest_path)
    processable = manifest_df[manifest_df["should_process"]]
    print(
        f"Stage 2: processing {len(processable)}/{len(manifest_df)} images "
        f"(n_workers={n_workers})"
    )

    figures_dir: Optional[Path] = None
    if save_diag_figures:
        figures_dir = output_root / "diag_figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for _, row in processable.iterrows()]

    def _run_one(row: pd.Series) -> tuple:
        status = process_job(row, output_root, ch_bf=ch_bf, ch_bcells=ch_bcells, ch_vasc=ch_vasc, figures_dir=figures_dir, verbose=verbose)
        return status

    if n_workers and n_workers > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, prefer="processes", verbose=0)(
            delayed(_run_one)(row) for row in rows
        )
    else:
        results = [_run_one(row) for row in rows]

    n_ok = 0
    for status in results:
        name = status["expected_output_name"]
        if status["skip_reason"]:
            print(f"  [FAIL] {name}: {status['skip_reason']} — {status['error']}")
        else:
            print(f"  [OK  ] {name}")
            n_ok += 1
    print(f"Stage 2 done: {n_ok}/{len(processable)} images processed.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 2 - auto-detect CART regions.")
    parser.add_argument("manifest_path")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-diag-figures", dest="diag_figures", action="store_false",
                        help="Skip watershed diagnostic PNGs.")
    args = parser.parse_args(argv)
    run(args.manifest_path, verbose=args.verbose, n_workers=args.workers,
        save_diag_figures=args.diag_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 5 — Per-image QC figures and publication plots.

Per-image QC PNG
----------------
A 2-row figure:
  Row 1 (raw channels): BF | B cells | Vasculature | CART cells
  Row 2 (mask overlays): BF + chip overlay | BF + tumour overlay |
                         Vasculature + vasc-mask overlay |
                         CART raw + CART-mask overlay

Saved to ``<figures_dir>/<image_name>.png``.

Publication plots
-----------------
After per-image QC, summary publication plots are generated from the
``all_counts.csv``, ``all_distances.csv``, and ``all_colocalisation.csv``
CSVs written by Stage 4.

Usage:
    python -m cart_id.figures <manifest_path> [--dpi 150] [--workers 4]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import read_curation_tif

STAGE2_SUBDIR = "stage2"
STAGE3_SUBDIR = "stage3"

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


def _resolve_tif(output_root: Path, name: str) -> Optional[Path]:
    override = output_root / STAGE3_SUBDIR / f"{name}.tif"
    if override.exists():
        return override
    auto = output_root / STAGE2_SUBDIR / f"{name}.tif"
    if auto.exists():
        return auto
    return None


def _render_overlay(ax, base, mask, color, alpha: float = 0.4):
    ax.imshow(base, cmap="gray", vmin=0, vmax=255)
    overlay = np.zeros((*base.shape, 4), dtype=float)
    overlay[mask.astype(bool)] = [*color, alpha]
    ax.imshow(overlay)
    ax.axis("off")


def render_qc_figure(
    tif_path: Path,
    image_name: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    stack, _pixel_size = read_curation_tif(tif_path)
    bf = stack[_CH_BF]
    bcell = stack[_CH_BCELL_RAW]
    vasc = stack[_CH_VASC_RAW]
    cart = stack[_CH_CART_RAW]
    chip_mask = stack[_CH_CHIP].astype(bool)
    tumour_mask = stack[_CH_TUMOUR].astype(bool)
    vasc_mask = stack[_CH_VASC_MASK].astype(bool)
    bcell_mask = stack[_CH_BCELL_MASK].astype(bool)
    cart_mask = stack[_CH_CART_MASK].astype(bool)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    fig.suptitle(image_name, fontsize=12, fontweight="bold")

    # Row 0: raw channels (all grayscale)
    for ax, img, title in zip(
        axes[0],
        [bf, bf, bcell, vasc, cart],
        ["BF", "BF", "B cells (raw)", "Vasculature (raw)", "CART cells (raw)"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # Row 1: overlays (raw grayscale + coloured mask)
    _render_overlay(axes[1, 0], bf, chip_mask, [0, 1, 1], alpha=0.35)        # chip = cyan
    axes[1, 0].set_title("BF + chip", fontsize=9)
    _render_overlay(axes[1, 1], bf, tumour_mask, [1, 0.2, 0], alpha=0.4)     # tumour = red-orange
    axes[1, 1].set_title("BF + tumour", fontsize=9)
    _render_overlay(axes[1, 2], bcell, bcell_mask, [0, 1, 0], alpha=1) # B cells = green
    axes[1, 2].set_title("B cells + mask", fontsize=9)
    _render_overlay(axes[1, 3], vasc, vasc_mask, [1, 0.6, 0], alpha=1)     # vasc = orange
    axes[1, 3].set_title("Vasculature + mask", fontsize=9)
    _render_overlay(axes[1, 4], cart, cart_mask, [0, 1, 0], alpha=1)       # CART = green
    axes[1, 4].set_title("CART cells + mask", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_one(job_args: tuple) -> tuple:
    """Worker entry point for joblib (must be module-level for pickling)."""
    matplotlib.use("Agg")
    tif_path, image_name, out_path, dpi = job_args
    try:
        render_qc_figure(Path(tif_path), image_name, Path(out_path), dpi=dpi)
        return image_name, None
    except Exception as exc:
        return image_name, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Publication plots from stage-4 CSVs
# ---------------------------------------------------------------------------

def _counts_barplot(df_counts: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    if df_counts.empty:
        return
    subset = df_counts[df_counts["region"].isin(["chip", "chip_not_tumour", "tumour"])]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pivot = subset.pivot_table(
        index=["patient", "day", "image"],
        columns=["cell_type", "region"],
        values="cell_count",
        aggfunc="sum",
    ).fillna(0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Cell counts per region", fontsize=11)
    ax.set_ylabel("Number of cells")
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=45)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _distance_histogram(df_dists: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    if df_dists.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for cell_type, grp in df_dists.groupby("cell_type"):
        vals = grp["distance_from_tumour_um"].dropna()
        ax.hist(vals, bins=50, alpha=0.5, label=cell_type)
    ax.set_title("Distance from tumour edge", fontsize=11)
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Number of cells")
    ax.legend()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _coloc_lineplot(df_coloc: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    if df_coloc.empty:
        return
    methods = df_coloc["method"].unique()
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 5), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        sub = df_coloc[df_coloc["method"] == method]
        for region, grp in sub.groupby("region"):
            grp = grp.groupby("param_um")["score"].mean().reset_index()
            ax.plot(grp["param_um"], grp["score"], marker="o", label=region)
        ax.set_title(f"Colocalisation: {method}", fontsize=10)
        ax.set_xlabel("Scale (µm)")
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_publication_plots(
    output_dir: Path,
    counts_csv: Optional[Path] = None,
    distances_csv: Optional[Path] = None,
    coloc_csv: Optional[Path] = None,
    dpi: int = 150,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if counts_csv and Path(counts_csv).exists():
        df_counts = pd.read_csv(counts_csv)
        _counts_barplot(df_counts, output_dir / "cell_counts_barplot.png", dpi=dpi)
    if distances_csv and Path(distances_csv).exists():
        df_dists = pd.read_csv(distances_csv)
        _distance_histogram(df_dists, output_dir / "distance_histogram.png", dpi=dpi)
    if coloc_csv and Path(coloc_csv).exists():
        df_coloc = pd.read_csv(coloc_csv)
        _coloc_lineplot(df_coloc, output_dir / "coloc_lineplot.png", dpi=dpi)
    print(f"Publication plots written to {output_dir}")


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------

def run(
    manifest_path: str | Path,
    figures_dir: Optional[str | Path] = None,
    dpi: int = 150,
    n_workers: int = 1,
    skip_publication_plots: bool = False,
) -> None:
    """Generate per-image QC PNGs and publication plots.

    *figures_dir* defaults to ``<manifest_dir>/figures``.
    """
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    if figures_dir is None:
        figures_dir = output_root / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    processable = manifest_df[manifest_df["should_process"]]
    print(f"Stage 5: generating QC figures for {len(processable)}/{len(manifest_df)} images "
          f"(n_workers={n_workers})")

    job_args_list: list[tuple] = []
    for row in processable.itertuples(index=False):
        name = str(row.expected_output_name)
        tif_path = _resolve_tif(output_root, name)
        if tif_path is None:
            print(f"  [SKIP] {name}: no stage2 or stage3 TIF", file=sys.stderr)
            continue
        image_name = str(getattr(row, "image_name", name) or name)
        safe_name = name.replace("/", "_").replace("\\", "_")
        out_path = figures_dir / f"{safe_name}.png"
        job_args_list.append((str(tif_path), image_name, str(out_path), dpi))

    if n_workers and n_workers > 1 and job_args_list:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, prefer="processes", verbose=0)(
            delayed(_render_one)(args) for args in job_args_list
        )
    else:
        results = [_render_one(args) for args in job_args_list]

    n_ok = 0
    for image_name, err in results:
        if err:
            print(f"  [FAIL] {image_name}: {err}", file=sys.stderr)
        else:
            print(f"  [OK  ] {image_name}")
            n_ok += 1
    print(f"Stage 5 QC figures: {n_ok}/{len(job_args_list)} rendered.")

    if not skip_publication_plots:
        pub_dir = figures_dir / "publication"
        make_publication_plots(
            pub_dir,
            counts_csv=output_root / "all_counts.csv",
            distances_csv=output_root / "all_distances.csv",
            coloc_csv=output_root / "all_colocalisation.csv",
            dpi=dpi,
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 5 - CART figures.")
    parser.add_argument("manifest_path")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-pub-plots", dest="pub_plots", action="store_false")
    args = parser.parse_args(argv)
    run(
        args.manifest_path,
        figures_dir=args.figures_dir,
        dpi=args.dpi,
        n_workers=args.workers,
        skip_publication_plots=not args.pub_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 6 — Publication plots from the Stage-4 CSVs.

Generates every figure from the exploratory plotting notebook directly from
the three CSVs written by :mod:`cart_id.analyse`:

    all_counts.csv          → CART scatter / bar / enrichment figures
    all_distances.csv       → distance-from-tumour KDE figures
    all_colocalisation.csv  → B-cell ↔ CART colocalisation line plots

Each figure is driven by exactly one CSV (its single source of truth); no
intermediate notebook state is required.  The experimental group
(``condition`` = ARi or UTD) is extracted from the ``image`` column;
``patient`` and ``day`` already exist as columns in the CSVs.

Usage:
    python -m cart_id.plotting <output_dir> [--plots-dir DIR] [--dpi 150]
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from PIL import ImageColor
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration constants (mirrored from the exploratory notebook)
# ---------------------------------------------------------------------------

CART_PLOT_REGIONS = ["chip", "tumour", "chip_vasculature"]
CART_METRICS = [
    ("cell_count", "CART Cell Count"),
    ("positive_pixels", "CART Positive Pixels"),
    ("positive_pixel_pct", "CART Coverage (% of region)"),
]
ENRICHMENT_REGIONS = ["tumour", "chip_vasculature", "chip_not_vasculature"]

COLOC_REGIONS = ["chip", "chip_vasculature", "chip_not_vasculature"]
COLOC_YLABELS = {
    "coarse_pearson": "Pearson r (blurred channels)",
    "B_near_CART": "Fraction of B cells near a CART cell",
    "CART_near_B": "Fraction of CART cells near a B cell",
}

DEFAULT_PALETTE = {
    "ARi": "#FF8C00",   # orange
    "UTD": "#000000",   # black
}


# ---------------------------------------------------------------------------
# Condition labelling (ARi / UTD)
# ---------------------------------------------------------------------------

def add_condition_label(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a ``condition`` (ARi/UTD) column extracted from the ``image`` column.

    ``patient`` and ``day`` already exist as columns in the Stage-4 CSVs.
    """
    dataframe = dataframe.copy()
    name = dataframe["image"].astype(str).str.lower()
    condition = pd.Series(np.nan, index=dataframe.index, dtype=object)
    condition[name.str.contains("utd", na=False)] = "UTD"
    condition[name.str.contains("ari", na=False)] = "ARi"
    dataframe["condition"] = condition
    return dataframe


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def create_n_valued_palette(base_colour_hex: str, n: int = 5) -> List[str]:
    """Return *n* lightness-graded shades of *base_colour_hex*."""
    r, g, b = ImageColor.getcolor(base_colour_hex, "RGB")
    r, g, b = r / 255, g / 255, b / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    palette = []
    for a in np.linspace(0.5, 1.4, n):
        l_adj = min(max(l * a, 0), 1)
        palette.append(to_hex(colorsys.hls_to_rgb(h, l_adj, s)))
    return palette


def _build_img_color(df: pd.DataFrame, group_col: str = "condition", palette_map=None):
    """Per-sample colours derived from group membership."""
    if palette_map is None:
        palette_map = DEFAULT_PALETTE
    img_color, group_images, group_base_colors = {}, {}, {}
    for group_name, base_hex in palette_map.items():
        grp_imgs = df.loc[df[group_col] == group_name, "sample_id"].unique().tolist()
        if not grp_imgs:
            continue
        shades = create_n_valued_palette(base_hex, max(len(grp_imgs), 2))
        for k, img_name in enumerate(grp_imgs):
            img_color[img_name] = shades[k % len(shades)]
        group_images[group_name] = grp_imgs
        group_base_colors[group_name] = base_hex
    for img_name in df["sample_id"].unique():
        img_color.setdefault(img_name, "gray")
    return img_color, group_images, group_base_colors


# ---------------------------------------------------------------------------
# Counts-based figures (source: all_counts.csv)
# ---------------------------------------------------------------------------

def _cart_counts_subset(counts: pd.DataFrame) -> pd.DataFrame:
    return counts[
        (counts["cell_type"] == "CART cells")
        & (counts["region"].isin(CART_PLOT_REGIONS))
        & (counts["condition"].notna())
    ]


def plot_cart_metrics_by_condition(counts: pd.DataFrame, plots_dir: Path, dpi: int = 150,
                                   save_type: str = "png") -> None:
    """Scatter of CART metrics per region, x-axis = condition."""
    cart_data = _cart_counts_subset(counts)
    if cart_data.empty:
        print("  [SKIP] CART metrics by condition — no labelled CART rows.")
        return
    condition_order = list(dict.fromkeys(cart_data["condition"]))
    condition_colors = {cond: plt.cm.tab20(i / max(len(condition_order) - 1, 1))
                        for i, cond in enumerate(condition_order)}

    for metric, ylabel in CART_METRICS:
        if metric not in cart_data.columns:
            continue
        fig, axes = plt.subplots(1, len(CART_PLOT_REGIONS), figsize=(12, 4),
                                 sharey=False, constrained_layout=True)
        fig.suptitle(ylabel, fontsize=12, fontweight="bold")

        for ax, region in zip(axes, CART_PLOT_REGIONS):
            region_data = cart_data[cart_data["region"] == region]
            for condition_index, condition in enumerate(condition_order):
                values = region_data.loc[region_data["condition"] == condition, metric].values
                jitter = np.linspace(-0.15, 0.15, max(len(values), 1))
                for jitter_offset, value in zip(jitter, values):
                    ax.scatter(condition_index + jitter_offset, value,
                               color=condition_colors[condition], s=30, alpha=0.5,
                               edgecolor="None", linewidth=0.4, zorder=3)
            ax.set_xticks(range(len(condition_order)))
            ax.set_xticklabels(condition_order, fontsize=8, rotation=45, ha="right")
            ax.set_title(region.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel(ylabel if ax is axes[0] else "")

        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=condition_colors[c], markersize=8, label=c)
            for c in condition_order
        ]
        fig.legend(handles=legend_handles, loc="outside right upper", fontsize=7, ncol=1)
        fig.savefig(plots_dir / f"CART_{metric}_by_condition.{save_type}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_cart_enrichment(counts: pd.DataFrame, plots_dir: Path, dpi: int = 150,
                         save_type: str = "png") -> Optional[pd.DataFrame]:
    """Normalised CART enrichment per region; also writes enrichment_quantities.csv."""
    if "region_area_px" not in counts.columns:
        print("  [SKIP] CART enrichment — region_area_px column missing "
              "(re-run Stage 4 to regenerate all_counts.csv).")
        return None

    cart_all = counts[counts["cell_type"] == "CART cells"].copy()
    chip_rows = cart_all[cart_all["region"] == "chip"][
        ["image", "positive_pixels", "region_area_px"]
    ].rename(columns={"positive_pixels": "cart_px_chip", "region_area_px": "area_chip"})

    enrichment = cart_all[cart_all["region"].isin(ENRICHMENT_REGIONS)].merge(
        chip_rows, on="image", how="left")
    if enrichment.empty:
        print("  [SKIP] CART enrichment — no enrichment-region rows.")
        return None
    enrichment["cart_enrichment"] = (
        (enrichment["positive_pixels"] / enrichment["cart_px_chip"].replace(0, np.nan))
        * (enrichment["region_area_px"] / enrichment["area_chip"].replace(0, np.nan))
    )

    fig, axes = plt.subplots(1, len(ENRICHMENT_REGIONS), figsize=(14, 5),
                             sharey=False, constrained_layout=True)
    if len(ENRICHMENT_REGIONS) == 1:
        axes = [axes]
    fig.suptitle("CART Normalised Score\n(CART in region / CART in chip) * (region area / chip area)",
                 fontsize=11, fontweight="bold")

    for ax, region in zip(axes, ENRICHMENT_REGIONS):
        region_data = enrichment[enrichment["region"] == region]
        conditions = list(dict.fromkeys(region_data["condition"].dropna()))
        x_positions = np.arange(len(conditions))
        means = [region_data.loc[region_data["condition"] == c, "cart_enrichment"].mean() for c in conditions]
        sems = [region_data.loc[region_data["condition"] == c, "cart_enrichment"].sem() for c in conditions]
        ax.bar(x_positions, means, yerr=sems, capsize=3, color="#5A9BD5", edgecolor="white")
        for xi, c in enumerate(conditions):
            vals = region_data.loc[region_data["condition"] == c, "cart_enrichment"].values
            jitter = np.linspace(-0.15, 0.15, max(len(vals), 1))
            for j, v in zip(jitter, vals):
                ax.scatter(xi + j, v, color="black", s=15, alpha=0.5, zorder=5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions, fontsize=7, rotation=45, ha="right")
        ax.set_title(region.replace("_", " ").title(), fontsize=10)
        ax.set_ylabel("Normalised CART score" if ax is axes[0] else "")

    fig.savefig(plots_dir / f"CART_normalised_enrichment.{save_type}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    csv_path = plots_dir / "enrichment_quantities.csv"
    enrichment.to_csv(csv_path, index=False)
    print(f"  Saved enrichment CSV → {csv_path}")
    return enrichment


# ---------------------------------------------------------------------------
# Distance KDE (source: all_distances.csv)
# ---------------------------------------------------------------------------

def plot_distance_kde(df: pd.DataFrame, save_path: Path, title_suffix: str = "",
                      group_col: str = "condition", palette_map=None,
                      dpi: int = 150) -> None:
    """KDE of distance from tumour edge per cell type, colour-coded by group."""
    if df.empty:
        print(f"  [SKIP] distance KDE{title_suffix} — no data.")
        return
    img_color, group_images, group_base_colors = _build_img_color(df, group_col, palette_map)
    cell_types = df["cell_type"].unique()
    imgs = df["sample_id"].unique()

    fig, axes = plt.subplots(1, len(cell_types), figsize=(7 * len(cell_types), 4),
                             sharey=False, constrained_layout=True)
    if len(cell_types) == 1:
        axes = [axes]

    x_grid = np.linspace(0, df["distance_from_tumour_um"].max(), 500)

    for ax, ct in zip(axes, cell_types):
        sub_ct = df[df["cell_type"] == ct]
        group_densities = {g: [] for g in group_base_colors}

        for img_name in imgs:
            vals = sub_ct[sub_ct["sample_id"] == img_name]["distance_from_tumour_um"].values
            if len(vals) < 2:
                continue
            y = gaussian_kde(vals)(x_grid)
            ax.plot(x_grid, y, color=img_color.get(img_name, "gray"),
                    alpha=0.5, linewidth=1.2, label=img_name)
            for grp_name, grp_imgs in group_images.items():
                if img_name in grp_imgs:
                    group_densities[grp_name].append(y)
                    break

        for grp_name, densities in group_densities.items():
            if not densities:
                continue
            mean_y = np.mean(densities, axis=0)
            ax.plot(x_grid, mean_y, color=group_base_colors[grp_name],
                    linewidth=2.5, linestyle="--", label=f"Mean {grp_name}")
            if len(densities) > 1:
                sd_y = np.std(densities, axis=0)
                ax.fill_between(x_grid, mean_y - sd_y, mean_y + sd_y,
                                color=group_base_colors[grp_name], alpha=0.15,
                                label=f"\u00b11 SD {grp_name}")

        ax.set_title(ct, fontsize=11, fontweight="bold")
        ax.set_xlabel("Distance from tumour edge (\u00b5m)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7, ncol=1)

    fig.suptitle(f"Distance from tumour edge{title_suffix}", fontsize=11, fontweight="bold")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_distance_by_day(distances: pd.DataFrame, plots_dir: Path, dpi: int = 150,
                         save_type: str = "png") -> None:
    """Per-day CART distance KDEs (in chip, and in chip vasculature)."""
    cart = distances[distances["cell_type"] == "CART cells"]
    days = [d for d in cart["day"].dropna().unique()]
    if not days:
        print("  [SKIP] distance-by-day — no labelled days.")
        return

    for day in sorted(days, key=lambda d: str(d)):
        dist_day = cart[cart["day"] == day]
        plot_distance_kde(
            dist_day[dist_day["in_chip"] == True],
            save_path=plots_dir / f"CART_day{day}_in_chip.{save_type}",
            title_suffix=f" \u2014 Day {day}, in chip (ARi vs UTD)",
            dpi=dpi,
        )
        if "in_chip_vasculature" in dist_day.columns:
            plot_distance_kde(
                dist_day[(dist_day["in_chip"] == True) & (dist_day["in_chip_vasculature"] == True)],
                save_path=plots_dir / f"CART_day{day}_in_chip_vasculature.{save_type}",
                title_suffix=f" \u2014 Day {day}, in vasculature on chip (ARi vs UTD)",
                dpi=dpi,
            )


# ---------------------------------------------------------------------------
# Colocalisation (source: all_colocalisation.csv)
# ---------------------------------------------------------------------------

def plot_colocalisation(df: pd.DataFrame, save_path: Path, title_suffix: str = "",
                        group_col: str = "condition", palette_map=None,
                        methods: Sequence[str] = ("CART_near_B",),
                        plot_regions: Sequence[str] = ("chip",), dpi: int = 150) -> None:
    """Line plot of colocalisation score vs scale (µm); rows=methods, cols=regions."""
    if df.empty:
        print(f"  [SKIP] colocalisation{title_suffix} — no data.")
        return
    img_color, group_images, group_base_colors = _build_img_color(df, group_col, palette_map)
    images = df["sample_id"].unique()
    n_rows, n_cols = len(methods), len(plot_regions)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                             constrained_layout=True, squeeze=False)
    fig.suptitle(f"Colocalisation \u2014 B cells \u2194 CART cells{title_suffix}",
                 fontsize=11, fontweight="bold")

    for row, method in enumerate(methods):
        sub_m = df[df["method"] == method]
        for col, region in enumerate(plot_regions):
            ax = axes[row, col]
            sub = sub_m[sub_m["region"] == region].copy()

            for img_name in images:
                s = (sub[sub["sample_id"] == img_name]
                     .dropna(subset=["score"])
                     .sort_values("param_um"))
                ax.plot(s["param_um"], s["score"],
                        color=img_color.get(img_name, "gray"),
                        alpha=0.7, linewidth=1, linestyle="dashdot", label=img_name)

            for grp_name, base_color in group_base_colors.items():
                mean_grp = (sub[sub[group_col] == grp_name]
                            .groupby("param_um")["score"].mean().reset_index())
                std_grp = (sub[sub[group_col] == grp_name]
                           .groupby("param_um")["score"].std().reset_index()
                           .rename(columns={"score": "sd"}))
                if mean_grp.empty:
                    continue
                merged = (mean_grp.merge(std_grp, on="param_um")
                          .dropna(subset=["score"])
                          .sort_values("param_um"))
                if merged.empty:
                    continue
                ax.plot(merged["param_um"], merged["score"],
                        color=base_color, linewidth=2.5, linestyle="--",
                        label=f"Mean {grp_name}")
                if merged["sd"].notna().any():
                    ax.fill_between(merged["param_um"],
                                    merged["score"] - merged["sd"],
                                    merged["score"] + merged["sd"],
                                    color=base_color, alpha=0.15, label=f"\u00b11 SD {grp_name}")

            ax.set_ylim(-1, 1) if method == "coarse_pearson" else ax.set_ylim(0, 1)
            if method == "coarse_pearson":
                ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
            if row == 0:
                ax.set_title(region, fontsize=9, fontweight="bold")
            ax.set_xlabel("Scale (\u00b5m)", fontsize=8)
            ax.set_ylabel(COLOC_YLABELS.get(method, method) if col == 0 else "", fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:15], labels[:15], loc="outside right upper", fontsize=7, ncol=1)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_colocalisation_all(coloc: pd.DataFrame, plots_dir: Path, dpi: int = 150,
                            save_type: str = "png") -> None:
    """Generate a colocalisation figure for each method present, across available regions."""
    if coloc.empty:
        print("  [SKIP] colocalisation — empty CSV.")
        return
    methods = [m for m in coloc["method"].dropna().unique()]
    regions = [r for r in COLOC_REGIONS if r in set(coloc["region"].dropna().unique())]
    if not regions:
        regions = list(coloc["region"].dropna().unique())[:3]
    for method in methods:
        plot_colocalisation(
            coloc,
            save_path=plots_dir / f"colocalisation_{method}.{save_type}",
            title_suffix=f" — {method}",
            methods=(method,),
            plot_regions=tuple(regions),
            dpi=dpi,
        )


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------

def _load_labelled(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path.name} not found.", file=sys.stderr)
        return None
    df = pd.read_csv(csv_path)
    if "image" in df.columns:
        is_out = df["image"].astype(str).str.lower().str.contains("out", na=False)
        n_dropped = int(is_out.sum())
        if n_dropped:
            print(f"  Dropped {n_dropped} row(s) from {csv_path.name} "
                  f"({df.loc[is_out, 'image'].nunique()} image(s) with 'out' in the name).")
        df = df[~is_out]
    if df.empty:
        print(f"  [SKIP] {csv_path.name} is empty.")
        return None
    labelled = add_condition_label(df)
    # Unique per-sample id: the `image` column alone is NOT unique across
    # patients/lif files (e.g. "ARi_device1" recurs), which would merge two
    # distinct monotonic curves into one zig-zag line when plotted.
    labelled["sample_id"] = (labelled["patient"].astype(str) + " | "
                             + labelled["lif_file"].astype(str) + " | "
                             + labelled["image"].astype(str))
    return labelled


def run(
    output_dir: str | Path,
    plots_dir: Optional[str | Path] = None,
    counts_csv: Optional[str | Path] = None,
    distances_csv: Optional[str | Path] = None,
    coloc_csv: Optional[str | Path] = None,
    dpi: int = 150,
    save_type: str = "png",
) -> Path:
    """Generate all publication plots from the three Stage-4 CSVs.

    Parameters
    ----------
    output_dir:
        Directory containing ``all_counts.csv``, ``all_distances.csv`` and
        ``all_colocalisation.csv`` (the Stage-4 outputs).
    plots_dir:
        Where to write figures.  Defaults to ``<output_dir>/plots``.
    counts_csv, distances_csv, coloc_csv:
        Override paths to individual CSVs (each plot uses one CSV as its
        single source of truth).
    """
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir) if plots_dir is not None else output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    counts_csv = Path(counts_csv) if counts_csv else output_dir / "all_counts.csv"
    distances_csv = Path(distances_csv) if distances_csv else output_dir / "all_distances.csv"
    coloc_csv = Path(coloc_csv) if coloc_csv else output_dir / "all_colocalisation.csv"

    print(f"Stage 6: writing plots to {plots_dir}")

    # --- Counts-based figures ---
    counts = _load_labelled(counts_csv)
    if counts is not None:
        plot_cart_metrics_by_condition(counts, plots_dir, dpi=dpi, save_type=save_type)
        plot_cart_enrichment(counts, plots_dir, dpi=dpi, save_type=save_type)

    # --- Distance KDE figures ---
    distances = _load_labelled(distances_csv)
    if distances is not None:
        plot_distance_by_day(distances, plots_dir, dpi=dpi, save_type=save_type)

    # --- Colocalisation figures ---
    coloc = _load_labelled(coloc_csv)
    if coloc is not None:
        plot_colocalisation_all(coloc, plots_dir, dpi=dpi, save_type=save_type)

    print(f"Stage 6 done: plots written to {plots_dir}")
    return plots_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 6 - publication plots from Stage-4 CSVs.")
    parser.add_argument("output_dir", help="Directory containing the three Stage-4 CSVs.")
    parser.add_argument("--plots-dir", default=None, help="Where to write figures.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--save-type", default="png", help="Figure file extension (png, pdf, ...).")
    args = parser.parse_args(argv)
    run(args.output_dir, plots_dir=args.plots_dir, dpi=args.dpi, save_type=args.save_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

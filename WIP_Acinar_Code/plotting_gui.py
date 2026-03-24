"""
Acinar Analysis Plotting GUI
=============================
Standalone magicgui GUI for plotting outputs from the acinar analysis pipeline.

Launch from a notebook (run ``%gui qt`` first)::

    %gui qt
    from plotting_gui import launch_and_run
    figs = launch_and_run()

Plots d0 (blank) vs soft vs stiff across varying timepoints, using
violin plots with split soft/stiff comparisons.
"""


import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import colorsys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from magicgui import magicgui
from magicgui.widgets import Container, Label, TextEdit
from matplotlib.patches import Patch
from PIL import ImageColor

# ---------------------------------------------------------------------------
#  Custom colour palette
# ---------------------------------------------------------------------------
MYCOL = ["red", "darkorange", "yellow", "limegreen", "dodgerblue", "darkviolet", "deeppink"]


def create_n_valued_palette(base_color_hex, n=14):
    """Generate *n* RGB colours by varying lightness of a base colour."""
    r, g, b = ImageColor.getcolor(base_color_hex, "RGB")
    r, g, b = r / 255, g / 255, b / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    adjustments = np.linspace(0.6, 1.4, n)
    return [colorsys.hls_to_rgb(h, min(max(l * a, 0), 1), s) for a in adjustments]


palette_14_red = create_n_valued_palette("#FF0000")
palette_14_limegreen = create_n_valued_palette("#32CD32")
palette_14_dodgerblue = create_n_valued_palette("#1E90FF")

# Condition → colour mapping (red / limegreen / dodgerblue from the palette)
COLOUR_MAP = {
    "blank": "red",
    "soft": "limegreen",
    "stiff": "dodgerblue",
}

# ---------------------------------------------------------------------------
#  Analysis type detection from CSV columns
# ---------------------------------------------------------------------------
_ANALYSIS_SIGNATURES = {
    "acinus_shape": {"acinus_volume_um3", "acinus_roundness"},
    "cell_nuclear_shape": {"nucleus_volume_um3", "cell_volume_um3", "cell_roundness"},
    "protein_polarisation": {"rounded_distance", "protein_intensity"},
    "apoptosis": {"c3_volume_um3", "normalised_distance", "number_of_nuclei"},
    "protein_proximity": {"dying", "proximity_intensity_in_cell"},
    "proliferation": {"dividing", "number_dividing"},
    "mitochondria": {"number_of_mito", "mito_volume_um3", "mito_cell_vol_ratio"},
}

# Which columns are numeric and meaningful to plot for each analysis
_PLOTTABLE_COLUMNS = {
    "acinus_shape": ["acinus_volume_um3", "acinus_roundness"],
    "cell_nuclear_shape": [
        "nucleus_volume_um3", "cell_volume_um3", "nucleus_cell_volume_ratio",
        "cell_roundness", "nucleus_roundness",
    ],
    "protein_polarisation": ["protein_intensity"],
    "apoptosis": [
        "c3_volume_um3", "normalised_distance", "number_of_nuclei",
    ],
    "protein_proximity": [
        "proximity_intensity_in_cell", "proximity_mean_intensity_in_cell",
        "proximity_intensity_around_cell", "proximity_mean_intensity_neighborhood",
    ],
    "proliferation": [
        "number_dividing", "number_not_dividing",
    ],
    "mitochondria": [
        "number_of_mito", "mito_volume_um3", "mito_cell_vol_ratio",
        "mean_mito_distance_ratio",
    ],
}


def detect_analysis_type(df: pd.DataFrame) -> Optional[str]:
    """Detect which analysis produced this DataFrame based on its columns."""
    cols = set(df.columns)
    for name, signature in _ANALYSIS_SIGNATURES.items():
        if signature.issubset(cols):
            return name
    return None


# ---------------------------------------------------------------------------
#  Plotting functions
# ---------------------------------------------------------------------------

def _sort_days(days):
    """Sort day values numerically."""
    return sorted(days, key=lambda x: float(x))


def plot_violin_by_condition(
    df: pd.DataFrame,
    parameter: str,
    analysis_name: str,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
) -> plt.Figure:
    """Plot split violins: d0 (blank) on left, then soft|stiff per timepoint.

    Returns the Figure.
    """
    if "day" not in df.columns or "condition" not in df.columns:
        return None

    days = _sort_days(df["day"].dropna().unique())
    has_d0 = 0 in days or 0.0 in days
    later_days = [d for d in days if d != 0]
    n_panels = (1 if has_d0 else 0) + (1 if later_days else 0)

    if n_panels == 0:
        return None

    width_ratios = []
    if has_d0:
        width_ratios.append(1)
    if later_days:
        width_ratios.append(max(1, len(later_days)))

    fig, axes = plt.subplots(
        1, n_panels, figsize=(3 * sum(width_ratios), 5),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.0},
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    idx = 0
    # Panel 1: d0 blank
    if has_d0:
        d0 = df[df["day"] == 0]
        if len(d0) > 0:
            sns.violinplot(
                data=d0, y=parameter, palette=["red"],
                ax=axes[idx], cut=0, inner="quartile", linewidth=0.8,
            )
        axes[idx].set_xlabel("d0")
        axes[idx].spines["right"].set_visible(False)
        idx += 1

    # Panel 2: later days with split soft|stiff violins
    if later_days:
        later = df[(df["day"] != 0) & (df["condition"].isin(["soft", "stiff"]))]
        if len(later) > 0:
            sns.violinplot(
                data=later, y=parameter, x="day", hue="condition",
                split=True, palette=COLOUR_MAP, ax=axes[idx],
                cut=0, inner="quartile", linewidth=0.8,
                order=later_days,
            )
            axes[idx].legend_.remove()
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")
        axes[idx].spines["left"].set_visible(False)
        axes[idx].yaxis.set_visible(False)

    # Axis labels & legend
    ylabel = parameter.replace("_", " ")
    axes[0].set_ylabel(ylabel, fontsize=12)
    fig.text(0.5, 0.01, "Day", ha="center", fontsize=12)

    legend_elements = [
        Patch(facecolor="red", edgecolor="k", label="Blank"),
        Patch(facecolor="limegreen", edgecolor="k", label="Soft"),
        Patch(facecolor="dodgerblue", edgecolor="k", label="Stiff"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", framealpha=0)

    title = f"{analysis_name.replace('_', ' ').title()} — {ylabel}"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 0.88, 0.95])

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{analysis_name}_{parameter}.{fmt}"
        fig.savefig(str(out), bbox_inches="tight", transparent=True, dpi=300)

    return fig


def plot_swarm_by_condition(
    df: pd.DataFrame,
    parameter: str,
    analysis_name: str,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
) -> plt.Figure:
    """Swarm + box plot for acinar-level (fewer points) data.

    Groups by condition and day.
    """
    if "day" not in df.columns or "condition" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(max(4, df["day"].nunique() * 2), 5))
    days = _sort_days(df["day"].dropna().unique())

    palette_order = []
    for d in days:
        sub = df[df["day"] == d]
        for cond in ["blank", "soft", "stiff"]:
            if cond in sub["condition"].values:
                palette_order.append(cond)

    sns.swarmplot(
        data=df, x="day", y=parameter, hue="condition",
        dodge=True, palette=COLOUR_MAP, ax=ax, size=4, order=days,
    )
    sns.boxplot(
        data=df, x="day", y=parameter, hue="condition",
        dodge=True, ax=ax, order=days,
        boxprops={"facecolor": "None"}, whiskerprops={"color": "k"},
        capprops={"color": "k"}, medianprops={"color": "k"},
        fliersize=0,
    )

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append((h, l))
    ax.legend(
        [h for h, _ in unique_handles],
        [l for _, l in unique_handles],
        loc="upper right", framealpha=0,
    )

    ylabel = parameter.replace("_", " ")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Day", fontsize=12)
    ax.set_title(f"{analysis_name.replace('_', ' ').title()} — {ylabel}", fontsize=13)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"{analysis_name}_{parameter}_swarm.{fmt}"
        fig.savefig(str(out), bbox_inches="tight", transparent=True, dpi=300)

    return fig


def plot_protein_polarisation(
    df: pd.DataFrame,
    save_dir: Optional[Path] = None,
    fmt: str = "png",
) -> plt.Figure:
    """Line plot of protein intensity vs normalised radial distance.

    One line per condition×day, coloured by condition and grouped by day.
    """
    if "rounded_distance" not in df.columns or "protein_intensity" not in df.columns:
        return None
    if "day" not in df.columns or "condition" not in df.columns:
        return None

    # Normalise distance 0-1 per filename
    if "filename" in df.columns:
        df = df.copy()
        grp = df.groupby("filename")["rounded_distance"]
        df["norm_dist"] = (df["rounded_distance"] - grp.transform("min")) / (
            grp.transform("max") - grp.transform("min")
        )
        df["norm_dist"] = df["norm_dist"].round(2)
        df = df[df["norm_dist"] < 0.99]
        # Scale intensity 0-1 per filename
        imax = df.groupby("filename")["protein_intensity"].transform("max")
        df["protein_intensity"] = df["protein_intensity"] / imax.clip(lower=1e-9)
        x_col = "norm_dist"
    else:
        x_col = "rounded_distance"

    days = _sort_days(df["day"].dropna().unique())
    ncols = len(days)
    fig, axes = plt.subplots(
        1, max(1, ncols), figsize=(5 * max(1, ncols), 5),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for i, day in enumerate(days):
        ax = axes[i]
        sub = df[df["day"] == day]
        for cond in ["blank", "soft", "stiff"]:
            c_sub = sub[sub["condition"] == cond]
            if len(c_sub) == 0:
                continue
            # Per-filename lines (thin, translucent)
            if "filename" in c_sub.columns:
                for fn, grp in c_sub.groupby("filename"):
                    avg = grp.groupby(x_col, as_index=False)["protein_intensity"].mean()
                    ax.plot(avg[x_col], avg["protein_intensity"],
                            color=COLOUR_MAP[cond], alpha=0.2, linewidth=0.8)
            # Average line (thick)
            avg_all = c_sub.groupby(x_col, as_index=False)["protein_intensity"].mean()
            avg_all["smooth"] = avg_all["protein_intensity"].rolling(3, center=True, min_periods=1).mean()
            ax.plot(avg_all[x_col], avg_all["smooth"],
                    color=COLOUR_MAP[cond], linewidth=2, label=cond.title())

        ax.set_title(f"Day {int(day)}", fontsize=12)
        ax.set_xlabel("Normalised Distance", fontsize=11)
        if i == 0:
            ax.set_ylabel("Protein Intensity (norm.)", fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(framealpha=0)

    fig.suptitle("Protein Polarisation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_dir / f"protein_polarisation.{fmt}"),
                    bbox_inches="tight", transparent=True, dpi=300)

    return fig


# ---------------------------------------------------------------------------
#  GUI class
# ---------------------------------------------------------------------------

class PlottingGUI:
    """Magicgui-based GUI for plotting acinar analysis outputs."""

    def __init__(self):
        self._closed = False
        self._config: Optional[dict] = None
        self._figures: List[plt.Figure] = []

        # --- Widgets ---
        self.folder_panel = magicgui(
            self._folder_stub,
            csv_folder={"label": "Results Folder (with CSVs)", "mode": "d"},
            output_folder={"label": "Save Plots To (optional)", "mode": "d"},
            call_button=False,
        )

        self.plot_panel = magicgui(
            self._plot_stub,
            plot_type={
                "label": "Plot Type",
                "choices": ["violin", "swarm", "both"],
            },
            save_format={
                "label": "Save Format",
                "choices": ["png", "pdf", "svg", "tif"],
            },
            call_button=False,
        )

        self._run_btn = magicgui(
            lambda: None,
            call_button="Generate Plots",
        )
        self._run_btn.called.connect(self._on_run_clicked)

        self._log_widget = TextEdit(value="", label="Log")

        self.widget = Container(widgets=[
            Label(value="<h2>Acinar Analysis Plotting</h2>"),
            self.folder_panel,
            self.plot_panel,
            self._run_btn,
            self._log_widget,
        ])
        self.widget.show()

    # --- Stubs ---

    @staticmethod
    def _folder_stub(
        csv_folder: Path = Path(),
        output_folder: Path = Path(),
    ):
        return None

    @staticmethod
    def _plot_stub(
        plot_type: str = "violin",
        save_format: str = "png",
    ):
        return None

    # --- Helpers ---

    def _log(self, msg: str):
        cur = self._log_widget.value.rstrip()
        self._log_widget.value = (cur + "\n" + msg) if cur else msg

    @staticmethod
    def _dir_or_none(path_value) -> Optional[Path]:
        p = Path(str(path_value))
        if str(p) in (".", "") or not p.is_dir():
            return None
        return p

    def _on_run_clicked(self):
        csv_folder = self._dir_or_none(self.folder_panel.csv_folder.value)
        output_folder = self._dir_or_none(self.folder_panel.output_folder.value)
        plot_type = str(self.plot_panel.plot_type.value)
        save_format = str(self.plot_panel.save_format.value)

        if csv_folder is None:
            self._log("[ERROR] Please select a folder containing result CSVs.")
            return

        if output_folder is None:
            self._log("[ERROR] Please select an output folder to save plots.")
            return

        csvs = list(csv_folder.glob("*.csv"))
        if not csvs:
            self._log(f"[ERROR] No CSV files found in {csv_folder}")
            return

        self._config = {
            "csv_folder": csv_folder,
            "output_folder": output_folder,
            "plot_type": plot_type,
            "save_format": save_format,
        }

        self._log(f"[INFO] Found {len(csvs)} CSV file(s)")
        self._log(f"[INFO] Plot type: {plot_type}")
        self._log(f"[INFO] Save format: {save_format}")
        self._log(f"[INFO] Saving to: {output_folder}")
        self._log("[INFO] Closing GUI and generating plots...")

        self._closed = True
        self.widget.close()

    @property
    def config(self) -> Optional[dict]:
        return self._config

    def generate_plots(self) -> List[plt.Figure]:
        """Read CSVs and generate all plots. Call after GUI closes."""
        if self._config is None:
            print("[ERROR] No configuration. Did the user click Generate Plots?")
            return []

        csv_folder = self._config["csv_folder"]
        output_folder = self._config["output_folder"]
        plot_type = self._config["plot_type"]
        save_format = self._config["save_format"]

        csvs = sorted(csv_folder.glob("*.csv"))
        figures = []

        for csv_path in csvs:
            print(f"\nReading {csv_path.name}...")
            df = pd.read_csv(csv_path)
            analysis = detect_analysis_type(df)

            if analysis is None:
                print(f"  [SKIP] Could not detect analysis type for {csv_path.name}")
                continue

            print(f"  Detected: {analysis}")

            # Special case: protein polarisation uses line plots
            if analysis == "protein_polarisation":
                fig = plot_protein_polarisation(df, save_dir=output_folder, fmt=save_format)
                if fig is not None:
                    figures.append(fig)
                    plt.close(fig)
                    print(f"  Created protein polarisation line plot")
                continue

            # Standard metric plots
            plottable = _PLOTTABLE_COLUMNS.get(analysis, [])
            # Only use columns that actually exist
            plottable = [c for c in plottable if c in df.columns]

            for param in plottable:
                if plot_type in ("violin", "both"):
                    fig = plot_violin_by_condition(
                        df, param, analysis, save_dir=output_folder, fmt=save_format)
                    if fig is not None:
                        figures.append(fig)
                        plt.close(fig)
                        print(f"  Created violin plot: {param}")

                if plot_type in ("swarm", "both"):
                    fig = plot_swarm_by_condition(
                        df, param, analysis, save_dir=output_folder, fmt=save_format)
                    if fig is not None:
                        figures.append(fig)
                        plt.close(fig)
                        print(f"  Created swarm plot: {param}")

        print(f"\n{'=' * 50}")
        print(f"Generated {len(figures)} plot(s)")
        if output_folder:
            print(f"Saved to: {output_folder}")
        print("=" * 50)

        self._figures = figures
        return figures


# ---------------------------------------------------------------------------
#  Public launch helpers
# ---------------------------------------------------------------------------

def launch() -> PlottingGUI:
    """Create and show the Plotting GUI. Returns the GUI instance."""
    return PlottingGUI()


def launch_and_run() -> List[plt.Figure]:
    """Launch the GUI, wait for config, then generate and return plots.

    Usage in a notebook::

        %gui qt
        from plotting_gui import launch_and_run
        figs = launch_and_run()
    """
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    gui = PlottingGUI()

    while not gui._closed:
        app.processEvents()
        time.sleep(0.05)

    if gui.config is None:
        print("[INFO] GUI closed without generating plots.")
        return []

    return gui.generate_plots()


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication
    _qapp = QApplication.instance() or QApplication([])
    launch_and_run()

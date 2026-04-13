"""Plotting functions for persistent and snapshot cell-death analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from colours import assign_colours, build_category_colormap


def plot_persistent_percentages(summary_df, fluor_names, title="Persistent positive cells over time"):
    """Line plot of cumulative % positive cells over frames (persistent mode).

    Each fluorophore gets its own coloured line, plus a black "Total" line
    summing all fates. Y-axis is fixed 0-100%.

    Parameters
    ----------
    summary_df : DataFrame
        Output of compute_persistent_percentages().
    fluor_names : list[str]
        Fluorophore names (must match <name>_pct columns in summary_df).
    title : str
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colour_assignments = assign_colours(fluor_names)
    for fluorophore_name in fluor_names:
        colour = colour_assignments[fluorophore_name]["mpl"]
        sns.lineplot(data=summary_df, x="frame", y=f"{fluorophore_name}_pct",
                     color=colour, label=fluorophore_name, ax=ax)
    sns.lineplot(data=summary_df, x="frame", y="total_positive_pct",
                 color="black", label="Total", ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return fig, ax


def plot_snapshot_percentages(summary_df, categories, title="Per-frame categories over time"):
    """Line plot of % cells per category over frames (snapshot mode).

    Each category gets a colour derived from build_category_colormap.
    Unlike persistent plots, lines can go up and down.

    Parameters
    ----------
    summary_df : DataFrame
        Output of compute_snapshot_percentages().
    categories : list[str]
        Category names (must match <cat>_pct columns in summary_df).
    title : str
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    category_colormap = build_category_colormap(categories)
    for category in categories:
        colour = category_colormap.get(category, (0.5, 0.5, 0.5))
        sns.lineplot(data=summary_df, x="frame", y=f"{category}_pct",
                     color=colour, label=category, ax=ax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return fig, ax


def plot_snapshot_trajectories(tracks_df, snapshot_df, title="Snapshot trajectories by category"):
    """Plot cell XY trajectories coloured by snapshot category at each step.

    Each cell's track is drawn as a series of line segments where each
    segment is coloured according to the cell's category in that frame.
    If lineage data is available (parent_track_id column), dashed lines
    connect parent track endpoints to daughter track start points.

    Parameters
    ----------
    tracks_df : DataFrame
        TrackMate tracks with columns: track_id, t, x, y, and optionally
        parent_track_id for lineage connections.
    snapshot_df : DataFrame
        Output of assign_snapshot_fates().
    title : str
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    if tracks_df is None or len(tracks_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No tracks available", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        return fig, ax

    categories = sorted(snapshot_df["category"].unique())
    color_map = build_category_colormap(categories)
    if "negative" not in color_map:
        color_map["negative"] = (0.6, 0.6, 0.6)

    track_points = tracks_df.copy()
    track_points["frame"] = track_points["t"].astype(int)
    track_points["label_id"] = track_points["track_id"].astype(int) + 1
    category_lookup = snapshot_df[["label_id", "frame", "category"]].copy()
    merged_tracks = track_points.merge(category_lookup, on=["label_id", "frame"], how="left")
    merged_tracks["category"] = merged_tracks["category"].fillna("negative")

    fig, ax = plt.subplots(figsize=(8, 6))
    all_x_coords, all_y_coords = [], []
    for label_id, track_group in merged_tracks.groupby("label_id"):
        track_group = track_group.sort_values("frame")
        coordinates = track_group[["x", "y"]].to_numpy(dtype=float)
        if len(coordinates) < 2:
            continue
        all_x_coords.extend(coordinates[:, 0].tolist())
        all_y_coords.extend(coordinates[:, 1].tolist())
        segments = np.stack([coordinates[:-1], coordinates[1:]], axis=1)
        segment_colours = [color_map.get(cat, (0.6, 0.6, 0.6)) for cat in track_group["category"].iloc[:-1]]
        ax.add_collection(LineCollection(segments, colors=segment_colours, linewidths=1.5, alpha=0.85))

    has_lineage = "parent_track_id" in tracks_df.columns
    if has_lineage:
        track_bounds = {}
        for track_id_val, group in merged_tracks.groupby("track_id"):
            group = group.sort_values("frame")
            track_bounds[int(track_id_val)] = {
                "first_xy": (float(group.iloc[0]["x"]), float(group.iloc[0]["y"])),
                "last_xy": (float(group.iloc[-1]["x"]), float(group.iloc[-1]["y"])),
            }
        for row_index, group in merged_tracks.drop_duplicates("track_id").iterrows():
            parent_track_id_val = group.get("parent_track_id")
            if pd.isna(parent_track_id_val):
                continue
            parent_track_id_val = int(parent_track_id_val)
            child_track_id = int(group["track_id"])
            if parent_track_id_val in track_bounds and child_track_id in track_bounds:
                parent_x, parent_y = track_bounds[parent_track_id_val]["last_xy"]
                child_x, child_y = track_bounds[child_track_id]["first_xy"]
                ax.plot([parent_x, child_x], [parent_y, child_y],
                        color="black", linewidth=1.0, alpha=0.5, linestyle="--", zorder=0)

    if all_x_coords and all_y_coords:
        ax.set_xlim(min(all_x_coords) - 10, max(all_x_coords) + 10)
        ax.set_ylim(min(all_y_coords) - 10, max(all_y_coords) + 10)
    ax.invert_yaxis()
    ax.set(xlabel="x", ylabel="y", title=title, aspect="equal")
    handles = [plt.Line2D([0], [0], color=color_map[cat], lw=2, label=cat) for cat in sorted(color_map.keys())]
    ax.legend(handles=handles, title="Snapshot category", loc="best")
    plt.tight_layout()
    return fig, ax


def plot_snapshot_cell_timelines(snapshot_df, tracks_df=None, title="Cell status over time"):
    """Horizontal bar chart showing each cell's category across all frames.

    Each row is a cell (Y-axis), each column is a frame (X-axis), and
    the colour of each bar indicates the cell's snapshot category in that
    frame. This gives an at-a-glance view of when cells transition.

    If lineage information is available (lineage_id, parent_track_id in
    tracks_df), cells are sorted by lineage and dashed vertical lines
    connect parent->daughter division events.

    Parameters
    ----------
    snapshot_df : DataFrame
        Output of assign_snapshot_fates().
    tracks_df : DataFrame or None
        TrackMate tracks CSV. If it contains lineage_id and
        parent_track_id columns, lineage sorting and division lines
        are enabled.
    title : str
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes.
    """
    if snapshot_df is None or len(snapshot_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No snapshot data", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        return fig, ax

    categories = sorted(snapshot_df["category"].unique())
    color_map = build_category_colormap(categories)
    has_lineage = (tracks_df is not None and len(tracks_df) > 0 and "lineage_id" in tracks_df.columns)

    if has_lineage:
        tracks_dataframe = tracks_df.copy()
        tracks_dataframe["label_id"] = tracks_dataframe["track_id"].astype(int) + 1
        lineage_map = tracks_dataframe.drop_duplicates("label_id")[["label_id", "lineage_id", "parent_track_id"]]
        lineage_map = lineage_map.set_index("label_id")

        def lineage_sort_key(label_id):
            lineage_id_string = lineage_map.loc[label_id, "lineage_id"] if label_id in lineage_map.index else str(label_id)
            parts = str(lineage_id_string).split(".")
            return tuple(int(part) if part.isdigit() else 0 for part in parts)

        cell_ids = sorted(snapshot_df["label_id"].unique(), key=lineage_sort_key)
        label_display = {}
        for label_id in cell_ids:
            label_display[label_id] = str(lineage_map.loc[label_id, "lineage_id"]) if label_id in lineage_map.index else str(label_id)
    else:
        cell_ids = sorted(snapshot_df["label_id"].unique())
        label_display = {label_id: str(label_id) for label_id in cell_ids}

    num_cells = len(cell_ids)
    cell_y_position = {cell_id: index for index, cell_id in enumerate(cell_ids)}
    min_frame = int(snapshot_df["frame"].min())
    max_frame = int(snapshot_df["frame"].max())

    height = max(3, min(num_cells * 0.25 + 1, 40))
    fig, ax = plt.subplots(figsize=(max(8, (max_frame - min_frame) * 0.15 + 2), height))

    for row_index, row in snapshot_df.iterrows():
        colour = color_map.get(row["category"], (0.5, 0.5, 0.5))
        ax.barh(cell_y_position[row["label_id"]], width=1, left=int(row["frame"]),
                height=0.8, color=colour, edgecolor="none", linewidth=0)

    if has_lineage:
        for label_id in cell_ids:
            if label_id not in lineage_map.index:
                continue
            parent_track_id_val = lineage_map.loc[label_id, "parent_track_id"]
            if pd.isna(parent_track_id_val):
                continue
            parent_label_id = int(parent_track_id_val) + 1
            if parent_label_id not in cell_y_position:
                continue
            daughter_frames = snapshot_df.loc[snapshot_df["label_id"] == label_id, "frame"]
            if len(daughter_frames) == 0:
                continue
            division_frame = int(daughter_frames.min())
            ax.plot([division_frame, division_frame],
                    [cell_y_position[parent_label_id], cell_y_position[label_id]],
                    color="black", linewidth=0.8, alpha=0.6, linestyle="--")

    ax.set_xlim(min_frame - 0.5, max_frame + 1.5)
    ax.set_ylim(-0.5, num_cells - 0.5)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Cell (lineage ID)" if has_lineage else "Cell", fontsize=11)
    ax.set_title(title, fontsize=12)
    if num_cells <= 60:
        ax.set_yticks(range(num_cells))
        ax.set_yticklabels([label_display[cell_id] for cell_id in cell_ids], fontsize=max(4, 8 - num_cells // 20))
    else:
        ax.set_yticks([])
    handles = [Patch(facecolor=color_map[cat], edgecolor="none", label=cat) for cat in categories]
    ax.legend(handles=handles, title="Category", loc="upper right", fontsize=8, title_fontsize=9, framealpha=0.8)
    fig.tight_layout()
    return fig, ax

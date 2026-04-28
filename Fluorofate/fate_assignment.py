"""Fate assignment, percentage computation, and cell filtering."""

import numpy as np
import pandas as pd

from measurement import measure_all_cells_in_frame


def assign_persistent_fates(linked_labels, frame_cell_pos):
    """Assign each cell a permanent fate based on whichever fluorophore it
    becomes positive for first (irreversible / "once dead, stays dead").

    Logic per cell:
      - Walk through all frames; record the first frame where each
        fluorophore has a positive blob assigned to that cell.
      - The fluorophore with the earliest first-positive frame wins
        and becomes that cell's fate (e.g. "Red" or "Green").
      - Cells that never become positive for anything get fate="negative".

    This mode is appropriate for markers like Annexin V / PI where
    positivity indicates a one-way biological transition.

    Parameters
    ----------
    linked_labels : ndarray (n_frames, H, W), uint32
        Tracked cell segmentation.
    frame_cell_pos : dict[str, dict[int, dict[int, int]]]
        Precomputed centroid-based assignments from compute_cell_positivity().

    Returns
    -------
    fates_dataframe : DataFrame
        One row per cell with columns: label_id, mean_area, mean_roundness,
        first_<fluor>_frame (per channel), <fluor>_positive_area (total),
        first_positive_frame, and fate.
    locked_labels : dict[str, ndarray]
        {fluorophore: (n_frames, H, W) uint32} — label images containing
        only cells assigned to that fate.
    per_frame_dataframe : DataFrame
        Long-format table with one row per (cell, frame) — includes area,
        roundness, per-fluorophore positive area, and the cell's fate.
    """
    num_frames = linked_labels.shape[0]
    fluor_names = list(frame_cell_pos.keys())

    frame_measurements = {}
    for frame_index in range(num_frames):
        frame_measurements[frame_index] = measure_all_cells_in_frame(linked_labels[frame_index])

    all_label_ids = set()
    for frame_index in range(num_frames):
        all_label_ids.update(frame_measurements[frame_index].keys())
    all_label_ids = sorted(all_label_ids)

    summary_rows = []
    frame_rows = []
    for label_id in all_label_ids:
        first_frames = {fluorophore_name: None for fluorophore_name in fluor_names}
        area_counts = {fluorophore_name: 0 for fluorophore_name in fluor_names}
        cell_areas = []
        cell_roundnesses = []
        for frame_index in range(num_frames):
            if label_id not in frame_measurements[frame_index]:
                continue
            area, roundness = frame_measurements[frame_index][label_id]
            cell_areas.append(area)
            cell_roundnesses.append(roundness)
            frame_record = {"label_id": label_id, "frame": frame_index, "area": area, "roundness": roundness}
            for fluorophore_name in fluor_names:
                positive_count = frame_cell_pos[fluorophore_name][frame_index].get(label_id, 0)
                area_counts[fluorophore_name] += positive_count
                frame_record[f"{fluorophore_name}_positive_area"] = positive_count
                if first_frames[fluorophore_name] is None and positive_count > 0:
                    first_frames[fluorophore_name] = frame_index
            frame_rows.append(frame_record)

        fate = "negative"
        first_positive = np.nan
        earliest_frame = num_frames + 1
        for fluorophore_name in fluor_names:
            first_fluorophore_frame = first_frames[fluorophore_name]
            if first_fluorophore_frame is not None and first_fluorophore_frame < earliest_frame:
                earliest_frame = first_fluorophore_frame
                fate = fluorophore_name
                first_positive = first_fluorophore_frame

        cell_row = {"label_id": label_id}
        cell_row["mean_area"] = float(np.mean(cell_areas)) if cell_areas else np.nan
        cell_row["mean_roundness"] = (float(np.nanmean(cell_roundnesses)) if cell_roundnesses else np.nan)
        for fluorophore_name in fluor_names:
            cell_row[f"first_{fluorophore_name}_frame"] = (
                first_frames[fluorophore_name]
                if first_frames[fluorophore_name] is not None else np.nan)
            cell_row[f"{fluorophore_name}_positive_area"] = area_counts[fluorophore_name]
        cell_row["first_positive_frame"] = first_positive
        cell_row["fate"] = fate
        summary_rows.append(cell_row)

    fates_dataframe = (
        pd.DataFrame(summary_rows)
        .sort_values(["fate", "first_positive_frame", "label_id"]).reset_index(drop=True))

    locked_labels = {}
    for fluorophore_name in fluor_names:
        fate_cell_ids = fates_dataframe.loc[
            fates_dataframe["fate"] == fluorophore_name, "label_id"
        ].to_numpy(dtype=np.uint32)
        locked_labels[fluorophore_name] = np.where(
            np.isin(linked_labels, fate_cell_ids), linked_labels, 0
        ).astype(np.uint32)

    per_frame_dataframe = pd.DataFrame(frame_rows)
    if len(per_frame_dataframe) > 0:
        fate_map = fates_dataframe.set_index("label_id")["fate"]
        per_frame_dataframe["fate"] = per_frame_dataframe["label_id"].map(fate_map)
        per_frame_dataframe = per_frame_dataframe.sort_values(["label_id", "frame"]).reset_index(drop=True)

    return fates_dataframe, locked_labels, per_frame_dataframe


def assign_snapshot_fates(linked_labels, frame_cell_pos):
    """Classify each cell independently in every frame (no memory across time).

    Unlike persistent mode, a cell can switch categories between frames.
    The category is built by joining the names of all fluorophores the cell
    is positive for in that frame with "+". Examples:
      - Positive for Red only       -> "Red"
      - Positive for Red AND Green  -> "Red+Green"
      - Positive for nothing        -> "negative"

    Parameters
    ----------
    linked_labels : ndarray (n_frames, H, W), uint32
        Tracked cell segmentation.
    frame_cell_pos : dict[str, dict[int, dict[int, int]]]
        Precomputed centroid-based assignments from compute_cell_positivity().

    Returns
    -------
    DataFrame
        One row per (cell, frame) with columns: label_id, frame, area,
        roundness, one bool column per fluorophore, <fluor>_positive_area,
        and category.
    """
    fluor_names = list(frame_cell_pos.keys())
    snapshot_rows = []
    for frame_index in range(linked_labels.shape[0]):
        cell_shapes = measure_all_cells_in_frame(linked_labels[frame_index])
        for label_id, (area, roundness) in cell_shapes.items():
            is_positive = {
                fluorophore_name: frame_cell_pos[fluorophore_name][frame_index].get(label_id, 0) > 0
                for fluorophore_name in fluor_names}
            category = (
                "+".join(fluorophore_name for fluorophore_name in fluor_names if is_positive[fluorophore_name])
                or "negative"
            )
            cell_frame_row = {"label_id": label_id, "frame": frame_index, "area": area, "roundness": roundness}
            cell_frame_row.update(is_positive)
            for fluorophore_name in fluor_names:
                cell_frame_row[f"{fluorophore_name}_positive_area"] = (
                    frame_cell_pos[fluorophore_name][frame_index].get(label_id, 0)
                )
            cell_frame_row["category"] = category
            snapshot_rows.append(cell_frame_row)

    return pd.DataFrame(snapshot_rows)


# ---------------------------------------------------------------------------
#  Percentage computation
# ---------------------------------------------------------------------------

def compute_persistent_percentages(assignments_df, num_frames, fluor_names):
    """Build a time-series of cumulative % positive cells (persistent mode).

    For each frame f, counts how many cells have fate == fluorophore AND
    first_positive_frame <= f, then divides by total cell count.
    Because fates are permanent, these curves can only go up over time.

    Parameters
    ----------
    assignments_df : DataFrame
        Output of assign_persistent_fates() — one row per cell.
    num_frames : int
        Total number of frames in the time-lapse.
    fluor_names : list[str]
        Fluorophore names to compute percentages for.

    Returns
    -------
    DataFrame
        Columns: frame, <fluor>_pct (one per channel), total_positive_pct.
    """
    total_cells = len(assignments_df)
    if total_cells == 0:
        raise ValueError("No tracked cells.")

    frames = np.arange(num_frames)
    data = {"frame": frames}
    total_positive = np.zeros(num_frames)

    for fluorophore_name in fluor_names:
        fate_cells = assignments_df.loc[
            assignments_df["fate"] == fluorophore_name, "first_positive_frame"
        ].dropna().to_numpy()
        cumulative_counts = np.array([np.sum(fate_cells <= frame_index) for frame_index in frames])
        percentages = 100.0 * cumulative_counts / total_cells
        data[f"{fluorophore_name}_pct"] = percentages
        total_positive += percentages

    data["total_positive_pct"] = total_positive
    return pd.DataFrame(data)


def compute_snapshot_percentages(snapshot_df, num_frames):
    """Build a time-series of % cells in each category (snapshot mode).

    Unlike persistent percentages, these can go up or down because a
    cell can change category between frames.

    Parameters
    ----------
    snapshot_df : DataFrame
        Output of assign_snapshot_fates() — one row per (cell, frame).
    num_frames : int
        Total number of frames in the time-lapse.

    Returns
    -------
    summary_df : DataFrame
        Columns: frame, <category>_pct for each unique category.
    categories : list[str]
        Sorted list of category names (positive categories first,
        "negative" last).
    """
    categories = sorted(snapshot_df["category"].unique(), key=lambda category: (category == "negative", category))

    counts = snapshot_df.groupby(["frame", "category"]).size().unstack(fill_value=0)
    totals_per_frame = counts.sum(axis=1)
    percentages = counts.div(totals_per_frame, axis=0) * 100.0
    percentages = percentages.reindex(range(num_frames), fill_value=0.0)

    data = {"frame": np.arange(num_frames)}
    for category in categories:
        data[f"{category}_pct"] = (
            percentages[category].values if category in percentages.columns
            else np.zeros(num_frames)
        )
    return pd.DataFrame(data), categories


# ---------------------------------------------------------------------------
#  Filtering
# ---------------------------------------------------------------------------

def filter_by_frame_presence(tracks_df, snapshot_df, num_frames, min_pct):
    """Remove cells that appear in too few frames (snapshot mode).

    Cells tracked for only a small fraction of the time-lapse are often
    segmentation artefacts or cells entering/leaving the field of view.
    Keeps only cells present in >= min_pct% of frames.

    Parameters
    ----------
    tracks_df : DataFrame or None
        TrackMate tracks CSV (columns: track_id, t, x, y, ...).
    snapshot_df : DataFrame
        Output of assign_snapshot_fates().
    num_frames : int
        Total frames in the time-lapse.
    min_pct : float
        Minimum % of frames a cell must appear in to be kept (0-100).

    Returns
    -------
    filtered_tracks : DataFrame
        Filtered tracks (or original if tracks_df is None/empty).
    filtered_snapshot : DataFrame
        Filtered snapshot data.
    """
    min_frame_count = num_frames * min_pct / 100.0
    frame_counts = snapshot_df.groupby("label_id")["frame"].nunique()
    keep_ids = frame_counts[frame_counts >= min_frame_count].index
    filtered_snapshot = snapshot_df[snapshot_df["label_id"].isin(keep_ids)].copy()
    if tracks_df is not None and len(tracks_df) > 0:
        keep_track_ids = keep_ids.astype(int) - 1
        filtered_tracks = tracks_df[tracks_df["track_id"].isin(keep_track_ids)].copy()
    else:
        filtered_tracks = tracks_df
    return filtered_tracks, filtered_snapshot


def filter_persistent_by_frame_presence(assignments_df, linked_labels, num_frames, min_pct):
    """Remove cells that appear in too few frames (persistent mode).

    Counts how many frames each cell actually appears in the
    linked_labels stack and removes cells below the threshold.

    Parameters
    ----------
    assignments_df : DataFrame
        Output of assign_persistent_fates().
    linked_labels : ndarray (n_frames, H, W)
        Tracked cell segmentation.
    num_frames : int
        Total frames in the time-lapse.
    min_pct : float
        Minimum % of frames a cell must appear in to be kept (0-100).

    Returns
    -------
    DataFrame
        Filtered copy of assignments_df.
    """
    min_frame_count = num_frames * min_pct / 100.0
    frame_presence = {}
    for frame_index in range(linked_labels.shape[0]):
        for cell_id in np.unique(linked_labels[frame_index]):
            if cell_id > 0:
                frame_presence[cell_id] = frame_presence.get(cell_id, 0) + 1
    keep_ids = [cell_id for cell_id, count in frame_presence.items() if count >= min_frame_count]
    return assignments_df[assignments_df["label_id"].isin(keep_ids)].copy()

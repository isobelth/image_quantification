"""TrackMate integration and lineage-tree construction."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from utils import configure_java_home


def _build_lineage(spot_rows, edges):
    """Build a lineage tree from TrackMate spots and edge pairs.

    Detects division events (one source spot with 2+ outgoing edges to
    different tracks) and assigns hierarchical lineage IDs such as
    ``"1"``, ``"1.1"``, ``"1.2"``, ``"1.1.1"``, etc.

    Parameters
    ----------
    spot_rows : list[dict]
        Each dict has at least ``spot_id``, ``track_id``, ``t``.
    edges : list[tuple[int, int]]
        ``(source_spot_id, target_spot_id)`` pairs from TrackMate, ordered
        so the source frame <= target frame.

    Returns
    -------
    pd.DataFrame
        The input rows extended with ``lineage_id``, ``parent_track_id``,
        and ``generation`` columns.
    """
    df = pd.DataFrame(spot_rows)
    if len(df) == 0:
        df["lineage_id"] = pd.Series(dtype=str)
        df["parent_track_id"] = pd.Series(dtype="Int64")
        df["generation"] = pd.Series(dtype=int)
        return df

    spot_to_track = dict(zip(df["spot_id"], df["track_id"]))

    children_of_spot = defaultdict(list)
    for source_id, target_id in edges:
        children_of_spot[source_id].append(target_id)

    parent_track = {}
    division_idx = {}
    for source_spot_id, target_spot_ids in children_of_spot.items():
        if len(target_spot_ids) < 2:
            continue
        source_track_id = spot_to_track[source_spot_id]
        daughter_track_ids = []
        for target_spot_id in target_spot_ids:
            target_track_id = spot_to_track[target_spot_id]
            if target_track_id != source_track_id and target_track_id not in daughter_track_ids:
                daughter_track_ids.append(target_track_id)
        if len(daughter_track_ids) == 0:
            continue
        all_children = daughter_track_ids[:]
        continues_in_source = any(
            spot_to_track[target_spot_id] == source_track_id for target_spot_id in target_spot_ids
        )
        if continues_in_source:
            if source_track_id not in all_children:
                all_children.insert(0, source_track_id)
        for sibling_index, child_track_id in enumerate(sorted(all_children)):
            if child_track_id not in parent_track:
                parent_track[child_track_id] = source_track_id
                division_idx[child_track_id] = sibling_index + 1

    all_track_ids = sorted(df["track_id"].unique())
    root_track_ids = [track_id for track_id in all_track_ids if track_id not in parent_track]

    lineage_map = {}
    for root_index, track_id in enumerate(root_track_ids, start=1):
        lineage_map[track_id] = str(root_index)

    queue = list(root_track_ids)
    visited = set(root_track_ids)
    while queue:
        track_id = queue.pop(0)
        children = [child_id for child_id, parent_id in parent_track.items() if parent_id == track_id]
        for child_track_id in sorted(children):
            if child_track_id in visited:
                continue
            parent_lineage_id = lineage_map.get(track_id, "?")
            sibling = division_idx.get(child_track_id, 1)
            lineage_map[child_track_id] = f"{parent_lineage_id}.{sibling}"
            visited.add(child_track_id)
            queue.append(child_track_id)

    for track_id in all_track_ids:
        if track_id not in lineage_map:
            lineage_map[track_id] = str(track_id)

    df["lineage_id"] = df["track_id"].map(lineage_map)
    df["parent_track_id"] = df["track_id"].map(
        lambda track_id: parent_track.get(track_id, pd.NA)
    )
    df["generation"] = df["lineage_id"].str.count(r"\\.").astype(int)

    return df


def generate_trackmate_labels(
    masks_path,
    output_directory,
    target_channel=1,
    simplify_contours=False,
    initial_search_radius=30.0,
    search_radius=150.0,
    max_frame_gap=3,
    allow_track_splitting=False,
    splitting_max_distance=15.0,
    allow_track_merging=False,
    ij=None,
):
    """Run TrackMate headless on a masks stack and produce linked labels.

    Parameters
    ----------
    masks_path : str or Path
        Path to a TIFF file containing Cellpose segmentation masks.
    output_directory : str or Path
        Directory where outputs (linked labels TIFF, tracks CSV, lineage
        summary CSV) will be saved.
    target_channel : int
        Which channel TrackMate should use for detection (1-based).
    simplify_contours : bool
        Whether to simplify label contours during detection.
    initial_search_radius : float
        Initial linking distance for the Kalman tracker.
    search_radius : float
        Maximum search radius for the Kalman tracker.
    max_frame_gap : int
        Maximum number of consecutive frames a track can skip.
    allow_track_splitting : bool
        Whether to allow track splitting (cell division).
    splitting_max_distance : float
        Maximum distance for splitting events.
    allow_track_merging : bool
        Whether to allow track merging.
    ij : ImageJ instance or None
        Pass an existing ImageJ instance to reuse it across calls.

    Returns
    -------
    dict
        Keys: "ij" (ImageJ instance), "trackmate_tracks_df" (DataFrame),
        "linked_labels" (ndarray), "linked_labels_path" (Path),
        "tracks_csv" (Path).
    """
    import imagej as _imagej
    import scyjava as _sj
    from imagej import Mode as _Mode

    if ij is None:
        configure_java_home()
        ij = _imagej.init("sc.fiji:fiji", mode=_Mode.HEADLESS, add_legacy=True)

    IJ = _sj.jimport("ij.IJ")
    HashMap = _sj.jimport("java.util.HashMap")
    Integer = _sj.jimport("java.lang.Integer")
    Double = _sj.jimport("java.lang.Double")
    Model = _sj.jimport("fiji.plugin.trackmate.Model")
    Settings = _sj.jimport("fiji.plugin.trackmate.Settings")
    TM = _sj.jimport("fiji.plugin.trackmate.TrackMate")
    Logger = _sj.jimport("fiji.plugin.trackmate.Logger")
    LIDF = _sj.jimport("fiji.plugin.trackmate.detection.LabelImageDetectorFactory")
    AKTF = _sj.jimport("fiji.plugin.trackmate.tracking.kalman.AdvancedKalmanTrackerFactory")

    imp = IJ.openImage(str(masks_path))
    if imp is None:
        raise RuntimeError(f"Could not open: {masks_path}")
    num_timepoints = int(imp.getNFrames())
    if num_timepoints <= 1:
        num_timepoints = int(imp.getStackSize())
    imp.setDimensions(1, 1, num_timepoints)
    imp.setOpenAsHyperStack(True)

    trackmate_model = Model()
    trackmate_model.setLogger(Logger.IJ_LOGGER)
    settings = Settings(imp)
    settings.detectorFactory = LIDF()
    detector_settings = HashMap()
    detector_settings.put("TARGET_CHANNEL", Integer.valueOf(int(target_channel)))
    detector_settings.put("SIMPLIFY_CONTOURS", bool(simplify_contours))
    settings.detectorSettings = detector_settings

    settings.trackerFactory = AKTF()
    tracker_settings = HashMap(settings.trackerFactory.getDefaultSettings())
    tracker_settings.put("KALMAN_SEARCH_RADIUS", Double.valueOf(float(search_radius)))
    tracker_settings.put("LINKING_MAX_DISTANCE", Double.valueOf(float(initial_search_radius)))
    tracker_settings.put("MAX_FRAME_GAP", Integer.valueOf(int(max_frame_gap)))
    tracker_settings.put("ALLOW_TRACK_SPLITTING", allow_track_splitting)
    tracker_settings.put("SPLITTING_MAX_DISTANCE", Double.valueOf(float(splitting_max_distance)))
    tracker_settings.put("ALLOW_TRACK_MERGING", allow_track_merging)
    feature_penalties = HashMap()
    for feature_name in ("POSITION_X", "POSITION_Y", "AREA"):
        feature_penalties.put(feature_name, Double.valueOf(1.0))
    tracker_settings.put("LINKING_FEATURE_PENALTIES", feature_penalties)
    tracker_settings.put("GAP_CLOSING_FEATURE_PENALTIES", feature_penalties)
    settings.trackerSettings = tracker_settings
    settings.addAllAnalyzers()

    trackmate = TM(trackmate_model, settings)
    if not trackmate.checkInput():
        raise RuntimeError(f"TrackMate input: {trackmate.getErrorMessage()}")
    if not trackmate.process():
        raise RuntimeError(f"TrackMate process: {trackmate.getErrorMessage()}")

    track_model = trackmate_model.getTrackModel()

    # Extract spots with a unique spot_id per spot
    spot_info = {}
    rows = []
    for track_id in track_model.trackIDs(True):
        for spot in sorted(
            track_model.trackSpots(track_id),
            key=lambda spot: float(spot.getFeature("FRAME")),
        ):
            spot_id = int(spot.ID())
            info = {
                "spot_id": spot_id,
                "track_id": int(track_id),
                "t": int(float(spot.getFeature("FRAME"))),
                "y": float(spot.getFeature("POSITION_Y")),
                "x": float(spot.getFeature("POSITION_X")),
                "quality": float(spot.getFeature("QUALITY")),
            }
            spot_info[spot_id] = info
            rows.append(info)

    # Extract edges to detect mother-daughter relationships
    edges = []
    for track_id in track_model.trackIDs(True):
        for edge in track_model.trackEdges(track_id):
            source_spot = track_model.getEdgeSource(edge)
            target_spot = track_model.getEdgeTarget(edge)
            source_id = int(source_spot.ID())
            target_id = int(target_spot.ID())
            if spot_info[source_id]["t"] > spot_info[target_id]["t"]:
                source_id, target_id = target_id, source_id
            edges.append((source_id, target_id))

    lineage_df = _build_lineage(rows, edges)

    tracks_df = lineage_df[
        ["track_id", "t", "y", "x", "quality", "lineage_id",
         "parent_track_id", "generation"]
    ].copy()
    csv_path = Path(output_directory) / "trackmate_tracks.csv"
    tracks_df.to_csv(csv_path, index=False)

    if "lineage_id" in tracks_df.columns:
        lineage_summary = (
            tracks_df.groupby("track_id")
            .agg(
                lineage_id=("lineage_id", "first"),
                parent_track_id=("parent_track_id", "first"),
                generation=("generation", "first"),
                first_frame=("t", "min"),
                last_frame=("t", "max"),
                n_frames=("t", "count"),
            )
            .reset_index()
            .sort_values("lineage_id")
        )
        lineage_csv = Path(output_directory) / "lineage_summary.csv"
        lineage_summary.to_csv(lineage_csv, index=False)

    masks = tifffile.imread(str(masks_path))
    linked = np.zeros_like(masks, dtype=np.uint32)
    num_t, num_y, num_x = masks.shape
    for track_id_val, time_val, y_val, x_val in tracks_df[["track_id", "t", "y", "x"]].to_numpy():
        time_index = int(time_val)
        y_index = int(np.clip(round(float(y_val)), 0, num_y - 1))
        x_index = int(np.clip(round(float(x_val)), 0, num_x - 1))
        if 0 <= time_index < num_t:
            label_at_spot = int(masks[time_index, y_index, x_index])
            if label_at_spot > 0:
                linked[time_index, masks[time_index] == label_at_spot] = int(track_id_val) + 1

    linked_labels_path = Path(output_directory) / "linked_labels_trackmate.tiff"
    tifffile.imwrite(str(linked_labels_path), linked)

    return {
        "ij": ij,
        "trackmate_tracks_df": tracks_df,
        "linked_labels": linked,
        "linked_labels_path": linked_labels_path,
        "tracks_csv": csv_path,
    }

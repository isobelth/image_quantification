"""Cell death analysis — napari + magicgui GUI.

Pipeline: Cellpose segmentation -> TrackMate tracking -> fluorescence
thresholding -> fate assignment (persistent / snapshot) -> percentage
trajectories.

Two workflows:
  - Analyse Single Image : full pipeline, all outputs shown in napari.
  - Analyse All in Folder : batch every TIFF in a folder -> CSV.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile

from cellpose import io, models
from magicgui import magicgui
from magicgui.widgets import TextEdit
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)
from skimage.filters import (
    gaussian,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.measure import label, regionprops

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def _running_in_notebook():
    if "ipykernel" in sys.modules:
        return True
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


def _configure_java_home():
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    candidates = []
    try:
        import jdk4py
        candidates.append(str(jdk4py.JAVA_HOME))
    except Exception:
        pass
    candidates.append(str(Path(sys.prefix) / "Library"))
    for home in candidates:
        if (Path(home) / "bin" / "server" / "jvm.dll").exists():
            os.environ["JAVA_HOME"] = str(home)
            os.environ["PATH"] = (
                str(Path(home) / "bin") + os.pathsep + os.environ.get("PATH", "")
            )
            return str(home)
    return None


# ---------------------------------------------------------------------------
#  Processing functions
# ---------------------------------------------------------------------------


def measure_all_cells_in_frame(label_image):
    """Measure area and roundness for every cell in a label image.

    Parameters
    ----------
    label_image : ndarray (H, W), uint32
        Label image where each cell has a unique integer ID (0 = background).

    Returns
    -------
    dict[int, tuple[int, float]]
        {cell_label_id: (area_in_pixels, roundness)} for every cell.
        Roundness = minor_axis / major_axis (1.0 = circle, → 0 = elongated).
    """
    measurements = {}
    for region in regionprops(label_image):
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        roundness = minor_axis / major_axis if major_axis > 0 else np.nan
        measurements[region.label] = (region.area, roundness)
    return measurements


def assign_positive_objects_to_cells(positive_labels_frame, linked_labels_frame):
    """Link thresholded fluorescence blobs to tracked cells using CoMs.

    For each connected component in the thresholded fluorescence image
    (positive_labels_frame), finds its centroid (center-of-mass pixel)
    and assign it to a labelled cell (linked_labels_frame) based on which cell label sits at that pixel.
    If the centroid falls on background (label 0),
    the blob is discarded.

    Parameters
    ----------
    positive_labels_frame : ndarray (H, W), uint32
        Labels of thresholded fluorescence for one frame.
    linked_labels_frame : ndarray (H, W), uint32
        Tracked cell segmentation labels for the same frame (from TrackMate).

    Returns
    -------
    cell_positive_area : dict[int, int]
        {cell_label_id: total fluorescence-positive pixel area} for cells
        that had at least one blob centroid land inside them.
    """
    cell_positive_area = {}
    max_y = linked_labels_frame.shape[0] - 1
    max_x = linked_labels_frame.shape[1] - 1
    for region in regionprops(positive_labels_frame):
        centroid_y = min(max(int(round(region.centroid[0])), 0), max_y)
        centroid_x = min(max(int(round(region.centroid[1])), 0), max_x)
        cell_id = int(linked_labels_frame[centroid_y, centroid_x])
        if cell_id > 0:
            cell_positive_area[cell_id] = cell_positive_area.get(cell_id, 0) + region.area
    return cell_positive_area


def _build_lineage(spot_rows, edges):
    """Build a lineage tree from spot rows and edge pairs.

    Detects division events (one source spot → 2+ target spots) and assigns
    hierarchical ``lineage_id`` strings such as ``"1"``, ``"1.1"``,
    ``"1.2"``, ``"1.1.1"``, etc.

    Parameters
    ----------
    spot_rows : list[dict]
        Each dict has at least ``spot_id``, ``track_id``, ``t``.
    edges : list[tuple[int, int]]
        ``(source_spot_id, target_spot_id)`` pairs from TrackMate, already
        ordered so the source frame ≤ target frame.

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

    # Map spot_id → track_id
    spot_to_track = dict(zip(df["spot_id"], df["track_id"]))

    # Detect which source spots have ≥ 2 outgoing edges (division)
    from collections import defaultdict
    children_of_spot = defaultdict(list)  # source_spot_id → [target_spot_id, ...]
    for src, tgt in edges:
        children_of_spot[src].append(tgt)

    # Build track-level parent → children mapping
    # A division: source_spot in track A connects to spots in track B, C, ...
    # (different track_ids in the targets mean new daughter tracks)
    parent_track = {}   # child_track_id → parent_track_id
    division_idx = {}   # child_track_id → sibling index (1-based)
    for src_spot, tgt_spots in children_of_spot.items():
        if len(tgt_spots) < 2:
            continue
        src_tid = spot_to_track[src_spot]
        # Collect distinct daughter track_ids (different from source)
        daughter_tids = []
        for ts in tgt_spots:
            tgt_tid = spot_to_track[ts]
            if tgt_tid != src_tid and tgt_tid not in daughter_tids:
                daughter_tids.append(tgt_tid)
        # If all targets are in the same track, no real split
        if len(daughter_tids) == 0:
            continue
        # Assign: all daughters (and possibly the continuation in src_tid)
        # get the source as parent.  If one target stays in src_tid, that
        # continuation also becomes a "daughter" for lineage purposes.
        all_children = daughter_tids[:]
        # Check if the source track itself continues (has any target in same track)
        continues_in_src = any(
            spot_to_track[ts] == src_tid for ts in tgt_spots
        )
        if continues_in_src:
            # The continuing track also gets marked as a daughter
            if src_tid not in all_children:
                all_children.insert(0, src_tid)
        for idx, ctid in enumerate(sorted(all_children)):
            if ctid not in parent_track:
                parent_track[ctid] = src_tid
                division_idx[ctid] = idx + 1

    # Build hierarchical lineage IDs
    # First, find root tracks (tracks with no parent)
    all_tids = sorted(df["track_id"].unique())
    root_tids = [t for t in all_tids if t not in parent_track]

    # Assign base lineage IDs to roots (1-based sequential)
    lineage_map = {}  # track_id → lineage_id string
    for i, tid in enumerate(root_tids, start=1):
        lineage_map[tid] = str(i)

    # BFS to assign children
    queue = list(root_tids)
    visited = set(root_tids)
    while queue:
        tid = queue.pop(0)
        # Find children of this track
        children = [c for c, p in parent_track.items() if p == tid]
        for ctid in sorted(children):
            if ctid in visited:
                continue
            parent_lid = lineage_map.get(tid, "?")
            sibling = division_idx.get(ctid, 1)
            lineage_map[ctid] = f"{parent_lid}.{sibling}"
            visited.add(ctid)
            queue.append(ctid)

    # Any tracks still not assigned (should not happen, but safety)
    for tid in all_tids:
        if tid not in lineage_map:
            lineage_map[tid] = str(tid)

    # Compute generation depth (number of dots in lineage_id)
    df["lineage_id"] = df["track_id"].map(lineage_map)
    df["parent_track_id"] = df["track_id"].map(
        lambda t: parent_track.get(t, pd.NA)
    )
    df["generation"] = df["lineage_id"].str.count(r"\\.").astype(int)

    return df


def cellpose_live_segmentation(
    stack,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    model_type="cpsam",
    custom_model_path=None,
    gpu=True,
    progress_callback=None,
):
    """Segment brightfield stack with Cellpose.

    *progress_callback(current_frame, total_frames)* is called after each
    frame so the GUI can update a progress bar.
    """
    if isinstance(stack, (str, Path)):
        stack = io.imread(stack)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    if custom_model_path:
        print(f"[Cellpose] Loading custom model: {custom_model_path}")
        model = models.CellposeModel(
            gpu=gpu, pretrained_model=str(custom_model_path)
        )
    else:
        print(f"[Cellpose] Loading built-in model: {model_type}  (GPU={gpu})")
        model = models.CellposeModel(gpu=gpu, pretrained_model=model_type)
    print("[Cellpose] Model loaded successfully.")

    if diameter is None:
        print("[Cellpose] Auto-estimating diameter from first frame...")
        m0, _, _ = model.eval(
            stack[0],
            diameter=None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        if m0.max() > 0:
            areas = [
                np.sum(m0 == i) for i in range(1, min(m0.max() + 1, 50))
            ]
            diameter = 2 * np.sqrt((np.median(areas) if areas else 700) / np.pi)
        else:
            diameter = 30.0
        print(f"[Cellpose] Estimated diameter: {diameter:.1f} px")

    n_frames = stack.shape[0]
    out = np.zeros(stack.shape, dtype=np.uint16)
    print(f"[Cellpose] Segmenting {n_frames} frame(s) (diameter={diameter:.1f}, flow_thresh={flow_threshold}, cellprob_thresh={cellprob_threshold}, min_size={min_size})...")

    for i in range(n_frames):
        m, _, _ = model.eval(
            stack[i],
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            resample=True,
        )
        n_cells = m.max()
        out[i] = m.astype(np.uint16)
        print(f"[Cellpose] Frame {i + 1}/{n_frames} — {n_cells} cell(s) detected")
        if progress_callback is not None:
            progress_callback(i + 1, n_frames)

    print(f"[Cellpose] Segmentation complete. Processed {n_frames} frame(s).")
    return out[0] if n_frames == 1 else out


def segment_fluorescence(stacks, blur_sigma=1.0, threshold_method="otsu"):
    """Segment fluorescence channels by Gaussian blur + per-frame thresholding.

    For each fluorophore stack:
      1. Apply Gaussian blur
      2. Compute a separate threshold for each frame independently.
      3. Binarise: pixels above that frame's threshold = "positive".
      4. Generate labels of fluorescence-positive "blobs" in each frame.

    Parameters
    ----------
    stacks : dict[str, ndarray]
        {fluorophore_name: array of shape (n_frames, H, W)}.
    blur_sigma : float
        Sigma for Gaussian smoothing
    threshold_method : str or dict[str, str]
        Thresholding algorithm name(s). A single string applies to all
        channels; a dict maps each channel name to its own method.
        Supported: "mean", "minimum", "yen", "otsu", "triangle".

    Returns
    -------
    result : dict
        Top-level keys: "threshold_methods", "blur_sigma", plus one key
        per fluorophore name. Each fluorophore entry is a dictionary with:
        - "blurred"             : ndarray (n_frames, H, W) — smoothed stack
        - "thresholds_per_frame": ndarray (n_frames,) — threshold for each frame
        - "positive"            : ndarray (n_frames, H, W) bool — binary mask
        - "positive_labels"     : ndarray (n_frames, H, W) uint32 — labels of fluorescence-positive blobs per frame
    """
    threshold_functions = {
        "mean": threshold_mean,
        "minimum": threshold_minimum,
        "yen": threshold_yen,
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
    }
    if isinstance(threshold_method, str):
        threshold_map = {name: threshold_method.lower() for name in stacks}
    else:
        threshold_map = {name: str(threshold_method.get(name, "otsu")).lower() for name in stacks}

    result = {"threshold_methods": threshold_map, "blur_sigma": blur_sigma}
    for channel_name, stack in stacks.items():
        threshold_method_name = threshold_map[channel_name]
        if threshold_method_name not in threshold_functions:
            raise ValueError(f"Unsupported threshold for '{channel_name}': {threshold_method_name}")
        threshold_function = threshold_functions[threshold_method_name]
        blurred = np.stack([gaussian(frame, sigma=blur_sigma, preserve_range=True) for frame in stack], axis=0)

        # Per-frame thresholding to handle signal drift over time
        num_frames = blurred.shape[0]
        thresholds_per_frame = np.zeros(num_frames)
        positive = np.zeros_like(blurred, dtype=bool)
        positive_labels = np.zeros(blurred.shape, dtype=np.uint32)
        for frame_index in range(num_frames):
            frame_threshold = threshold_function(blurred[frame_index])
            thresholds_per_frame[frame_index] = frame_threshold
            positive[frame_index] = blurred[frame_index] > frame_threshold
            positive_labels[frame_index] = label(positive[frame_index]).astype(np.uint32)

        result[channel_name] = {
            "blurred": blurred,
            "thresholds_per_frame": thresholds_per_frame,
            "positive": positive,
            "positive_labels": positive_labels,
        }
    return result


def compute_cell_positivity(linked_labels, pos_label_imgs, fluorophore_names):
    """Map fluorescence blobs to tracked cells for every frame & channel.
    Uses assign_positive_objects_to_cells (COM-based) to decide which
    of the identified cells each fluorescence-positive blob belongs to.

    Parameters
    ----------
    linked_labels : ndarray (n_frames, H, W), uint32
        Tracked cell segmentation from TrackMate.
    pos_label_imgs : dict[str, ndarray]
        {fluorophore_name: (n_frames, H, W) uint32 labels}
        from segment_fluorescence().
    fluorophore_names : list[str]
        Names matching the keys in pos_label_imgs.

    Returns
    -------
    frame_cell_pos : dict[str, dict[int, dict[int, int]]]
        Nested lookup: frame_cell_pos[fluorophore][frame_index][cell_id]
        → positive pixel area assigned to that cell in that frame.
        e.g. frame_cell_pos["Red"][5][42] → 187 means cell #42 in frame 5 had 187 pixels of Red fluorescence assigned to it.
        For each fluorophore in each frame, we find every fluorescence-positive blob, determine which cell it belongs to
        (if any), and sum the total positive area assigned to each cell.
    positive_cell_labels : dict[str, ndarray]
        {fluorophore: (n_frames, H, W) uint32} — label images where only
        positive cells are painted with their cell ID, everything else 0.
    """
    num_frames = linked_labels.shape[0]
    frame_cell_pos = {}
    positive_cell_labels = {}
    for fluorophore_name in fluorophore_names:
        frame_cell_pos[fluorophore_name] = {}
        output_labels = np.zeros_like(linked_labels)
        for frame_index in range(num_frames):
            cell_positivity = assign_positive_objects_to_cells(
                pos_label_imgs[fluorophore_name][frame_index],
                linked_labels[frame_index],
            )
            frame_cell_pos[fluorophore_name][frame_index] = cell_positivity
            positive_ids = np.array(list(cell_positivity.keys()), dtype=np.uint32)
            if len(positive_ids) > 0:
                is_positive = np.isin(linked_labels[frame_index], positive_ids)
                output_labels[frame_index] = np.where(is_positive, linked_labels[frame_index], 0)
        positive_cell_labels[fluorophore_name] = output_labels
    return frame_cell_pos, positive_cell_labels


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
    """Run TrackMate headless.  Pass *ij* to reuse an ImageJ instance."""
    import imagej as _imagej
    import scyjava as _sj
    from imagej import Mode as _Mode

    if ij is None:
        _configure_java_home()
        ij = _imagej.init("sc.fiji:fiji", mode=_Mode.HEADLESS, add_legacy=True)

    IJ = _sj.jimport("ij.IJ")
    HashMap = _sj.jimport("java.util.HashMap")
    Integer = _sj.jimport("java.lang.Integer")
    Double = _sj.jimport("java.lang.Double")
    Model = _sj.jimport("fiji.plugin.trackmate.Model")
    Settings = _sj.jimport("fiji.plugin.trackmate.Settings")
    TM = _sj.jimport("fiji.plugin.trackmate.TrackMate")
    Logger = _sj.jimport("fiji.plugin.trackmate.Logger")
    LIDF = _sj.jimport(
        "fiji.plugin.trackmate.detection.LabelImageDetectorFactory"
    )
    AKTF = _sj.jimport(
        "fiji.plugin.trackmate.tracking.kalman.AdvancedKalmanTrackerFactory"
    )

    imp = IJ.openImage(str(masks_path))
    if imp is None:
        raise RuntimeError(f"Could not open: {masks_path}")
    nt = int(imp.getNFrames())
    if nt <= 1:
        nt = int(imp.getStackSize())
    imp.setDimensions(1, 1, nt)
    imp.setOpenAsHyperStack(True)

    mdl = Model()
    mdl.setLogger(Logger.IJ_LOGGER)
    settings = Settings(imp)
    settings.detectorFactory = LIDF()
    ds = HashMap()
    ds.put("TARGET_CHANNEL", Integer.valueOf(int(target_channel)))
    ds.put("SIMPLIFY_CONTOURS", bool(simplify_contours))
    settings.detectorSettings = ds

    settings.trackerFactory = AKTF()
    ts = HashMap(settings.trackerFactory.getDefaultSettings())
    ts.put("KALMAN_SEARCH_RADIUS", Double.valueOf(float(search_radius)))
    ts.put("LINKING_MAX_DISTANCE", Double.valueOf(float(initial_search_radius)))
    ts.put("MAX_FRAME_GAP", Integer.valueOf(int(max_frame_gap)))
    ts.put("ALLOW_TRACK_SPLITTING", allow_track_splitting)
    ts.put("SPLITTING_MAX_DISTANCE", Double.valueOf(float(splitting_max_distance)))
    ts.put("ALLOW_TRACK_MERGING", allow_track_merging)
    fp = HashMap()
    for k in ("POSITION_X", "POSITION_Y", "AREA"):
        fp.put(k, Double.valueOf(1.0))
    ts.put("LINKING_FEATURE_PENALTIES", fp)
    ts.put("GAP_CLOSING_FEATURE_PENALTIES", fp)
    settings.trackerSettings = ts
    settings.addAllAnalyzers()

    trackmate = TM(mdl, settings)
    if not trackmate.checkInput():
        raise RuntimeError(f"TrackMate input: {trackmate.getErrorMessage()}")
    if not trackmate.process():
        raise RuntimeError(f"TrackMate process: {trackmate.getErrorMessage()}")

    tm = mdl.getTrackModel()

    # ------------------------------------------------------------------
    # Extract spots with a unique spot_id per spot
    # ------------------------------------------------------------------
    spot_info = {}  # java_spot_id -> dict
    rows = []
    for tid in tm.trackIDs(True):
        for s in sorted(
            tm.trackSpots(tid), key=lambda s: float(s.getFeature("FRAME"))
        ):
            sid = int(s.ID())
            info = {
                "spot_id": sid,
                "track_id": int(tid),
                "t": int(float(s.getFeature("FRAME"))),
                "y": float(s.getFeature("POSITION_Y")),
                "x": float(s.getFeature("POSITION_X")),
                "quality": float(s.getFeature("QUALITY")),
            }
            spot_info[sid] = info
            rows.append(info)

    # ------------------------------------------------------------------
    # Extract edges to detect mother-daughter relationships
    # ------------------------------------------------------------------
    edges = []  # list of (source_spot_id, target_spot_id)
    for tid in tm.trackIDs(True):
        for edge in tm.trackEdges(tid):
            src = tm.getEdgeSource(edge)
            tgt = tm.getEdgeTarget(edge)
            src_id = int(src.ID())
            tgt_id = int(tgt.ID())
            # Ensure src is the earlier-frame spot
            if spot_info[src_id]["t"] > spot_info[tgt_id]["t"]:
                src_id, tgt_id = tgt_id, src_id
            edges.append((src_id, tgt_id))

    # Build lineage: detect division events (one source → 2+ targets)
    lineage_df = _build_lineage(rows, edges)

    tracks_df = lineage_df[
        ["track_id", "t", "y", "x", "quality", "lineage_id",
         "parent_track_id", "generation"]
    ].copy()
    csv_path = Path(output_directory) / "trackmate_tracks.csv"
    tracks_df.to_csv(csv_path, index=False)

    # Save a concise lineage summary (one row per track)
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
    n_t, n_y, n_x = masks.shape
    for tid, t, y, x in tracks_df[["track_id", "t", "y", "x"]].to_numpy():
        ti = int(t)
        yi = int(np.clip(round(float(y)), 0, n_y - 1))
        xi = int(np.clip(round(float(x)), 0, n_x - 1))
        if 0 <= ti < n_t:
            lid = int(masks[ti, yi, xi])
            if lid > 0:
                linked[ti, masks[ti] == lid] = int(tid) + 1

    lbl_path = Path(output_directory) / "linked_labels_trackmate.tiff"
    tifffile.imwrite(str(lbl_path), linked)

    return {
        "ij": ij,
        "trackmate_tracks_df": tracks_df,
        "linked_labels": linked,
        "linked_labels_path": lbl_path,
        "tracks_csv": csv_path,
    }


# ---------------------------------------------------------------------------
#  Fate assignment
# ---------------------------------------------------------------------------

def assign_persistent_fates(linked_labels, frame_cell_pos):
    """Assign each cell a permanent fate based on whichever fluorophore it
    becomes positive for FIRST (irreversible / "once dead, stays dead").

    Logic per cell:
      - Walk through all frames; record the first frame where each
        fluorophore has a positive blob assigned to that cell.
      - The fluorophore with the earliest first-positive frame wins
        and becomes that cell's "fate" (e.g. "Red" or "Green").
      - Cells that never become positive for anything get fate="negative".

    This is appropriate for markers like Annexin V / PI where positivity
    indicates a one-way biological transition (apoptosis, necrosis).

    Parameters
    ----------
    linked_labels : ndarray (n_frames, H, W), uint32
        Tracked cell segmentation.
    frame_cell_pos : dict[str, dict[int, dict[int, int]]]
        Precomputed COM-based assignments from compute_cell_positivity().

    Returns
    -------
    fates_dataframe : DataFrame
        One row per cell with columns: label_id, mean_area, mean_roundness,
        first_<fluor>_frame (per channel), <fluor>_positive_area (total),
        first_positive_frame, and fate.
    locked_labels : dict[str, ndarray]
        {fluorophore: (n_frames, H, W) uint32} — label images containing
        only cells assigned to that fate (for Napari overlays).
    per_frame_dataframe : DataFrame
        Long-format table with one row per (cell, frame) — includes area,
        roundness, per-fluorophore positive area, and the cell's fate.
    """
    num_frames = linked_labels.shape[0]
    fluor_names = list(frame_cell_pos.keys())

    # Batch-measure all cells in every frame (one regionprops call per frame)
    frame_measurements = {}
    for frame_index in range(num_frames):
        frame_measurements[frame_index] = measure_all_cells_in_frame(linked_labels[frame_index])

    # Collect all unique cell IDs across all frames
    all_label_ids = set()
    for frame_index in range(num_frames):
        all_label_ids.update(frame_measurements[frame_index].keys())
    all_label_ids = sorted(all_label_ids)

    # Build per-cell summaries and per-frame records
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

        # Determine fate: whichever fluorophore appeared first
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

    # Build locked label images (only cells assigned to each fate)
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
      - Positive for Red only       → "Red"
      - Positive for Red AND Green  → "Red+Green"
      - Positive for nothing        → "negative"

    Uses batch regionprops (measure_all_cells_in_frame) so cell shape is
    measured once per frame for ALL cells, not once per cell per frame.

    Parameters
    ----------
    linked_labels : ndarray (n_frames, H, W), uint32
        Tracked cell segmentation.
    frame_cell_pos : dict[str, dict[int, dict[int, int]]]
        Precomputed COM-based assignments from compute_cell_positivity().

    Returns
    -------
    DataFrame
        One row per (cell, frame) with columns: label_id, frame, area,
        roundness, one bool column per fluorophore, <fluor>_positive_area,
        and category (the "+"-joined string).
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
#  Percentages, filtering, and plotting
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
        # For each frame, count how many cells became positive at or before it
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
    # Sort out all the possible categories for consistent ordering in outputs. "Negative" always comes last
    categories = sorted(snapshot_df["category"].unique(), key=lambda category: (category == "negative", category))

    # Pivot: count cells per (frame, category), then divide by total per frame
    counts = snapshot_df.groupby(["frame", "category"]).size().unstack(fill_value=0)
    totals_per_frame = counts.sum(axis=1)
    percentages = counts.div(totals_per_frame, axis=0) * 100.0

    # Reindex to cover all frames (fills missing frames with 0)
    percentages = percentages.reindex(range(num_frames), fill_value=0.0)

    data = {"frame": np.arange(num_frames)}
    for category in categories:
        data[f"{category}_pct"] = (
            percentages[category].values if category in percentages.columns
            else np.zeros(num_frames)
        )
    return pd.DataFrame(data), categories


# --- Colour inference ---

COLOUR_KEYWORDS: Dict[str, dict] = {
    "red":     {"mpl": "tab:red",    "napari": "red",     "rgb": (1.0, 0.0, 0.0)},
    "green":   {"mpl": "tab:green",  "napari": "green",   "rgb": (0.0, 0.8, 0.0)},
    "blue":    {"mpl": "tab:blue",   "napari": "blue",    "rgb": (0.2, 0.4, 1.0)},
    "yellow":  {"mpl": "goldenrod",  "napari": "yellow",  "rgb": (0.9, 0.85, 0.0)},
    "cyan":    {"mpl": "tab:cyan",   "napari": "cyan",    "rgb": (0.0, 0.8, 0.8)},
    "magenta": {"mpl": "tab:pink",   "napari": "magenta", "rgb": (0.9, 0.0, 0.6)},
}

def assign_colours(names):
    """Assign plot colours to a list of fluorophore/category names.

    Names containing a colour keyword (e.g. "Red", "Green") get that
    colour. Combined-positive cells (e.g. "Red+Green") get the average of their components' colours.
    Unknown names get assigned unused colours from the palette in order.

    Parameters
    ----------
    names : list[str]
        Fluorophore or category names.

    Returns
    -------
    dict[str, dict]
        {name: {"mpl": ..., "napari": ..., "rgb": ...}} for each input name.
    """
    all_colour_entries = list(COLOUR_KEYWORDS.values())
    result = {}
    used_palette_indices = set()

    # First pass: keyword matches
    for name in names:
        lowercase_name = name.lower()
        for palette_index, (keyword, entry) in enumerate(COLOUR_KEYWORDS.items()):
            if keyword in lowercase_name:
                result[name] = entry
                used_palette_indices.add(palette_index)
                break

    # Second pass: assign unused colours to unmatched names
    available_indices = [i for i in range(len(all_colour_entries)) if i not in used_palette_indices]
    fallback_counter = 0
    for name in names:
        if name not in result:
            if available_indices:
                result[name] = all_colour_entries[available_indices[fallback_counter % len(available_indices)]]
            else:
                result[name] = all_colour_entries[fallback_counter % len(all_colour_entries)]
            fallback_counter += 1

    return result


def build_category_colormap(categories):
    """Build an RGB colour mapping for snapshot categories.

    Single-fluorophore categories get their inferred colour directly.
    Compound categories like "Red+Green" get the mean of their component
    colours (clamped to [0, 1]). "negative" is always grey.

    Parameters
    ----------
    categories : list[str]
        Category names, e.g. ["Red", "Green", "Red+Green", "negative"].

    Returns
    -------
    dict[str, tuple[float, float, float]]
        {category_name: (R, G, B)} with values in 0-1 range.
    """
    single_fluorophores = []
    for category in categories:
        if category == "negative":
            continue
        for part in category.split("+"):
            if part not in single_fluorophores:
                single_fluorophores.append(part)
    colour_assignments = assign_colours(single_fluorophores)
    single_rgb = {name: np.array(colour_assignments[name]["rgb"]) for name in single_fluorophores}
    colormap = {}
    for category in categories:
        if category == "negative":
            colormap[category] = (0.7, 0.7, 0.7)
            continue
        parts = category.split("+")
        rgb = np.clip(np.mean([single_rgb.get(part, np.array([0.5, 0.5, 0.5])) for part in parts], axis=0), 0, 1)
        colormap[category] = tuple(rgb.tolist())
    return colormap


# --- Plotting ---

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


# --- Filtering ---

def filter_by_frame_presence(tracks_df, snapshot_df, num_frames, min_pct):
    """Remove cells that appear in too few frames (snapshot mode).

    Cells that are only tracked for a small fraction of the time-lapse
    are often segmentation artefacts or cells entering/leaving the FOV.
    This filter keeps only cells present in >= min_pct% of frames.
    Also filters the matching tracks_df rows (track_id = label_id - 1).

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

    Same idea as filter_by_frame_presence, but works on the persistent
    assignments table. Counts how many frames each cell_id actually
    appears in the linked_labels stack (not just the summary table).

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
    keep_ids = [cid for cid, count in frame_presence.items() if count >= min_frame_count]
    return assignments_df[assignments_df["label_id"].isin(keep_ids)].copy()


# --- Trajectory & timeline plots ---

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
        Output of assign_snapshot_fates() — provides category per (label_id, frame).
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
        parent_track_id columns, lineage sorting and division lines are enabled.
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


_CP_DEFAULT_DIAMETER = None
_CP_DEFAULT_FLOW_THRESHOLD = 0.4
_CP_DEFAULT_CELLPROB_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
#  GUI Application
# ---------------------------------------------------------------------------

class CellDeathAnalysisApp:
    """Napari-based GUI that wraps the full cell-death analysis pipeline."""

    # -- placeholder methods for config panels (never called directly) ------

    @staticmethod
    def _file_placeholder(
        single_tiff: Path = Path(),
        batch_folder: Path = Path(),
        output_directory: Path = Path(),
    ):
        pass

    @staticmethod
    def _channel_placeholder(
        brightfield_channel: int = -1,
        fluor_1_name: str = "Annexin_V",
        fluor_1_channel: int = 0,
        fluor_1_threshold: str = "mean",
        fluor_2_name: str = "PI",
        fluor_2_channel: int = 1,
        fluor_2_threshold: str = "mean",
        fluor_3_name: str = "",
        fluor_3_channel: int = 2,
        fluor_3_threshold: str = "mean",
    ):
        pass

    @staticmethod
    def _analysis_placeholder(
        analysis_mode: str = "persistent",
        blur_sigma: float = 1.0,
        custom_model_file: Path = Path(),
    ):
        pass

    @staticmethod
    def _cellpose_placeholder(
        model_type: str = "cpsam",
        min_size: int = 15,
        use_gpu: bool = True,
    ):
        pass

    @staticmethod
    def _trackmate_placeholder(
        initial_search_radius: float = 30.0,
        search_radius: float = 150.0,
        max_frame_gap: int = 3,
        allow_splitting: bool = False,
        splitting_max_distance: float = 15.0,
        allow_merging: bool = False,
    ):
        pass

    # -- init ---------------------------------------------------------------

    def __init__(self):
        self.viewer = napari.Viewer(title="Cell Death Analysis")

        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.setQuitOnLastWindowClosed(not _running_in_notebook())

        # -- state --
        self._ij = None  # ImageJ instance, lazy-initialised & reused
        self._results: List[dict] = []
        self._results_df: Optional[pd.DataFrame] = None

        # -- progress bar --
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Idle")

        # -- log --
        self._log = TextEdit(value="")
        try:
            self._log.native.setReadOnly(True)
        except Exception:
            pass
        self._log.min_height = 100
        self._log.max_height = 300

        # -- file panel --
        self.file_panel = magicgui(
            self._file_placeholder,
            single_tiff={
                "label": "Single TIFF (.tif/.tiff)",
                "mode": "r",
                "filter": "*.tif *.tiff",
            },
            batch_folder={"label": "Batch folder (optional for single)", "mode": "d"},
            output_directory={"label": "Output directory", "mode": "d"},
            call_button=False,
        )

        # -- channel panel --
        self.channel_panel = magicgui(
            self._channel_placeholder,
            brightfield_channel={
                "label": "Brightfield ch (-1=last)",
                "min": -1,
                "max": 20,
            },
            fluor_1_name={"label": "Fluorophore 1 name"},
            fluor_1_channel={
                "label": "Fluorophore 1 channel",
                "min": 0,
                "max": 20,
            },
            fluor_1_threshold={
                "label": "Fluorophore 1 threshold",
                "choices": ["mean", "minimum", "yen", "otsu", "triangle"],
            },
            fluor_2_name={"label": "Fluorophore 2 name"},
            fluor_2_channel={
                "label": "Fluorophore 2 channel",
                "min": 0,
                "max": 20,
            },
            fluor_2_threshold={
                "label": "Fluorophore 2 threshold",
                "choices": ["mean", "minimum", "yen", "otsu", "triangle"],
            },
            fluor_3_name={"label": "Fluorophore 3 (blank=none)"},
            fluor_3_channel={
                "label": "Fluorophore 3 channel",
                "min": 0,
                "max": 20,
            },
            fluor_3_threshold={
                "label": "Fluorophore 3 threshold",
                "choices": ["mean", "minimum", "yen", "otsu", "triangle"],
            },
            call_button=False,
        )

        # -- analysis panel --
        self.analysis_panel = magicgui(
            self._analysis_placeholder,
            analysis_mode={
                "label": "Analysis mode",
                "choices": ["persistent", "snapshot"],
            },
            blur_sigma={
                "label": "Blur sigma",
                "min": 0.1,
                "max": 20.0,
                "step": 0.1,
            },
            custom_model_file={
                "label": "Select file (optional custom model)",
                "mode": "r",
                "filter": "*.pt *.pth",
            },
            call_button=False,
        )
        self.analysis_panel.custom_model_file.changed.connect(
            self._on_custom_model_file_changed
        )

        # -- cellpose panel --
        self.cellpose_panel = magicgui(
            self._cellpose_placeholder,
            model_type={
                "label": "Model",
                "choices": ["cpsam", "cyto3", "cyto2", "cyto", "nuclei"],
            },
            min_size={"label": "Min cell size (px)", "min": 0, "max": 2000},
            use_gpu={"label": "Use GPU"},
            call_button=False,
        )
        self._on_custom_model_file_changed()

        # -- trackmate panel --
        self.trackmate_panel = magicgui(
            self._trackmate_placeholder,
            initial_search_radius={
                "label": "Init search radius",
                "min": 1.0,
                "max": 500.0,
                "step": 1.0,
            },
            search_radius={
                "label": "Search radius",
                "min": 1.0,
                "max": 2000.0,
                "step": 5.0,
            },
            max_frame_gap={"label": "Max frame gap", "min": 0, "max": 50},
            allow_splitting={"label": "Allow splitting"},
            splitting_max_distance={
                "label": "Splitting max distance",
                "min": 1.0,
                "max": 2000.0,
                "step": 5.0,
            },
            allow_merging={"label": "Allow merging"},
            call_button=False,
        )

        # -- action buttons --
        self._btn_segment = magicgui(
            self._run_segmentation_only, call_button="1) Run Segmentation"
        )
        self._btn_track = magicgui(
            self._run_tracking_only, call_button="2) Run Tracking"
        )
        self._btn_persistent = magicgui(
            self._run_persistent_analysis,
            call_button="3a) Run Persistent Analysis",
        )
        self._btn_snapshot = magicgui(
            self._run_snapshot_analysis,
            call_button="3b) Run Snapshot Analysis",
        )
        self._btn_single = magicgui(
            self._run_all_single_image, call_button="Run All (Single Image)"
        )
        self._btn_folder = magicgui(
            self._analyse_folder, call_button="Analyse All in Folder"
        )
        self._btn_save = magicgui(
            self._save_results, call_button="Save Results"
        )

        # -- dock widgets --
        self.viewer.window.add_dock_widget(
            self.file_panel, name="Files", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.channel_panel, name="Channels", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.analysis_panel, name="Analysis", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.cellpose_panel, name="Cellpose", area="right"
        )
        self.viewer.window.add_dock_widget(
            self.trackmate_panel, name="TrackMate", area="right"
        )

        # Run panel: buttons + progress + log
        run_widget = QWidget()
        run_layout = QVBoxLayout(run_widget)
        run_layout.addWidget(self._btn_segment.native)
        run_layout.addWidget(self._btn_track.native)
        run_layout.addWidget(self._btn_persistent.native)
        run_layout.addWidget(self._btn_snapshot.native)
        run_layout.addWidget(self._btn_single.native)
        run_layout.addWidget(self._btn_folder.native)
        run_layout.addWidget(self._btn_save.native)
        run_layout.addWidget(self._progress)
        run_layout.addWidget(self._log.native)
        self.viewer.window.add_dock_widget(
            run_widget, name="Run & Log", area="right"
        )

    # -- helpers ------------------------------------------------------------

    def _append_log(self, msg: str):
        self._log.value = (
            (self._log.value.rstrip() + "\n" + msg)
            if self._log.value
            else msg
        )

    def _set_progress(self, value: int, maximum: int = 100, fmt: str = ""):
        self._progress.setRange(0, maximum)
        self._progress.setValue(min(value, maximum) if maximum > 0 else 0)
        if fmt:
            self._progress.setFormat(fmt)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _set_buttons_enabled(self, enabled: bool):
        self._btn_segment.call_button.enabled = enabled
        self._btn_track.call_button.enabled = enabled
        self._btn_persistent.call_button.enabled = enabled
        self._btn_snapshot.call_button.enabled = enabled
        self._btn_single.call_button.enabled = enabled
        self._btn_folder.call_button.enabled = enabled
        self._btn_save.call_button.enabled = enabled

    def _custom_model_is_selected(self) -> bool:
        custom_value = str(self.analysis_panel.custom_model_file.value).strip()
        return bool(custom_value and custom_value != ".")

    def _on_custom_model_file_changed(self, *_):
        use_custom = self._custom_model_is_selected()
        self.cellpose_panel.model_type.enabled = not use_custom
        self.cellpose_panel.model_type.tooltip = (
            "Disabled because a custom model file is selected in Analysis."
            if use_custom
            else ""
        )

    def _read_fluorophore_config(self):
        """Read channel panel -> (fluor_names, fluor_channels, bf_channel)."""
        bf = int(self.channel_panel.brightfield_channel.value)
        names: List[str] = []
        channels: Dict[str, int] = {}
        thresholds: Dict[str, str] = {}
        for attr_n, attr_c, attr_t in [
            ("fluor_1_name", "fluor_1_channel", "fluor_1_threshold"),
            ("fluor_2_name", "fluor_2_channel", "fluor_2_threshold"),
            ("fluor_3_name", "fluor_3_channel", "fluor_3_threshold"),
        ]:
            n = str(getattr(self.channel_panel, attr_n).value).strip()
            c = int(getattr(self.channel_panel, attr_c).value)
            t = str(getattr(self.channel_panel, attr_t).value)
            if n:
                names.append(n)
                channels[n] = c
                thresholds[n] = t
        if not names:
            raise ValueError("At least one fluorophore name must be provided.")
        return names, channels, bf, thresholds

    def _resolve_single_run_paths(self):
        """Resolve and validate single-image input/output paths."""
        tiff_path = Path(str(self.file_panel.single_tiff.value))
        if not tiff_path.exists() or not tiff_path.is_file():
            raise ValueError("Select a valid TIFF file in the Files panel.")
        if tiff_path.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError("Single image must be a .tif or .tiff file.")

        out_dir = Path(str(self.file_panel.output_directory.value))
        if not str(out_dir).strip() or str(out_dir) == ".":
            out_dir = tiff_path.parent / "cell_death_output"
        work_dir = out_dir / tiff_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        return tiff_path, out_dir, work_dir

    def _load_image_channels(self, tiff_path: Path):
        """Load TIFF and return brightfield + fluorophore stacks."""
        fluor_names, fluor_channels, bf_ch, _thresholds = self._read_fluorophore_config()
        image = tifffile.imread(str(tiff_path))
        if image.ndim != 4:
            raise ValueError(
                f"Expected 4-D TIFF (T, C, Y, X), got {image.ndim}-D "
                f"shape {image.shape}."
            )
        n_channels = image.shape[1]
        for fn, ch in fluor_channels.items():
            if ch >= n_channels:
                raise ValueError(
                    f"Channel {ch} for '{fn}' is out of range "
                    f"(image has {n_channels} channels)."
                )
        bf_image = image[:, bf_ch, :, :]
        fluor_images = {
            fn: image[:, fluor_channels[fn], :, :] for fn in fluor_names
        }
        return bf_image, fluor_images, fluor_names, n_channels

    def _resolve_cellpose_model_config(self):
        """Read model panel and resolve built-in/custom model selection."""
        model_type = str(self.cellpose_panel.model_type.value)
        custom_value = str(self.analysis_panel.custom_model_file.value).strip()
        custom_model_path = None
        if self._custom_model_is_selected():
            custom_model_path = Path(custom_value)
            if not custom_model_path.exists():
                raise ValueError("Custom model path does not exist.")
            self._append_log(
                f"Using custom Cellpose model path: {custom_model_path}"
            )
        else:
            self._append_log(f"Using built-in Cellpose model: {model_type}")
        return model_type, custom_model_path

    def _run_segmentation_stage(self, tiff_path: Path, work_dir: Path):
        """Run Cellpose segmentation and save masks stack."""
        cp_model, cp_custom_model = self._resolve_cellpose_model_config()
        cp_min = int(self.cellpose_panel.min_size.value)
        cp_gpu = bool(self.cellpose_panel.use_gpu.value)

        self._set_progress(2, fmt="Loading image...")
        self._append_log(f"Loading: {tiff_path.name}")
        bf_image, fluor_images, _, n_channels = self._load_image_channels(
            tiff_path
        )
        self._append_log(
            f"  Shape: ({bf_image.shape[0]}, {n_channels}, {bf_image.shape[1]}, "
            f"{bf_image.shape[2]})  ({bf_image.shape[0]} frames, {n_channels} channels)"
        )

        self._set_progress(5, fmt="Cellpose: starting...")
        self._append_log("Running Cellpose segmentation...")

        def _cp_cb(frame, total):
            pct = 5 + int(90 * frame / total)
            self._set_progress(pct, fmt=f"Cellpose: frame {frame}/{total}")

        masks_stack = cellpose_live_segmentation(
            bf_image,
            diameter=_CP_DEFAULT_DIAMETER,
            flow_threshold=_CP_DEFAULT_FLOW_THRESHOLD,
            cellprob_threshold=_CP_DEFAULT_CELLPROB_THRESHOLD,
            min_size=cp_min,
            model_type=cp_model,
            custom_model_path=cp_custom_model,
            gpu=cp_gpu,
            progress_callback=_cp_cb,
        )

        masks_path = work_dir / "masks_stack.tiff"
        tifffile.imwrite(str(masks_path), masks_stack.astype(np.uint16))
        self._set_progress(100, fmt="Segmentation saved")
        self._append_log(f"[OK] Saved segmentation -> {masks_path}")

        return {
            "bf_image": bf_image,
            "fluor_images": fluor_images,
            "masks_stack": masks_stack,
            "masks_path": masks_path,
        }

    def _run_tracking_stage(self, work_dir: Path):
        """Run TrackMate from saved masks stack and save linked labels."""
        tm_init = float(self.trackmate_panel.initial_search_radius.value)
        tm_search = float(self.trackmate_panel.search_radius.value)
        tm_gap = int(self.trackmate_panel.max_frame_gap.value)
        tm_split = bool(self.trackmate_panel.allow_splitting.value)
        tm_split_dist = float(self.trackmate_panel.splitting_max_distance.value)
        tm_merge = bool(self.trackmate_panel.allow_merging.value)

        masks_path = work_dir / "masks_stack.tiff"
        if not masks_path.exists():
            raise FileNotFoundError(
                "No saved segmentation found. Run segmentation first."
            )

        self._set_progress(0, maximum=0, fmt="Running TrackMate...")
        self._append_log("Running TrackMate (UI may be unresponsive)...")
        tm_out = generate_trackmate_labels(
            masks_path=masks_path,
            output_directory=work_dir,
            initial_search_radius=tm_init,
            search_radius=tm_search,
            max_frame_gap=tm_gap,
            allow_track_splitting=tm_split,
            splitting_max_distance=tm_split_dist,
            allow_track_merging=tm_merge,
            ij=self._ij,
        )
        self._ij = tm_out["ij"]
        tracks_df = tm_out["trackmate_tracks_df"]
        linked_labels = tm_out["linked_labels"]
        n_tracks = int(tracks_df["track_id"].nunique())

        self._set_progress(100, fmt="Tracking saved")
        self._append_log(
            f"[OK] Saved tracking -> {tm_out['linked_labels_path']}"
        )
        self._append_log(
            f"  Tracked {n_tracks} cells, {len(tracks_df)} points"
        )

        return {
            "tracks_df": tracks_df,
            "linked_labels": linked_labels,
            "linked_labels_path": tm_out["linked_labels_path"],
            "tracks_csv": tm_out["tracks_csv"],
        }

    def _run_analysis_stage(self, tiff_path: Path, work_dir: Path, mode: str):
        """Run either persistent or snapshot analysis from saved tracking."""
        _fn, _fc, _bf, thresh_map = self._read_fluorophore_config()
        blur_sigma = float(self.analysis_panel.blur_sigma.value)

        linked_path = work_dir / "linked_labels_trackmate.tiff"
        if not linked_path.exists():
            raise FileNotFoundError(
                "No saved tracking found. Run tracking first."
            )

        bf_image, fluor_images, fluor_names, _ = self._load_image_channels(
            tiff_path
        )
        linked_labels = tifffile.imread(str(linked_path)).astype(np.uint32)

        self._set_progress(10, fmt="Fluorescence segmentation...")
        self._append_log("Segmenting fluorescence channels...")
        fl_out = segment_fluorescence(
            fluor_images, blur_sigma=blur_sigma, threshold_method=thresh_map
        )
        pos_label_imgs = {fn: fl_out[fn]["positive_labels"] for fn in fluor_names}
        frame_cell_pos, positive_cell_labels = compute_cell_positivity(
            linked_labels, pos_label_imgs, fluor_names
        )

        self._set_progress(45, fmt=f"Assigning fates ({mode})...")
        self._append_log(f"Assigning fates (mode={mode})...")
        locked_labels = None
        assignments_df = None
        snapshot_df = None

        # Load lineage info from tracks CSV (if available)
        tracks_csv = work_dir / "trackmate_tracks.csv"
        _lineage_cols = ["track_id", "lineage_id", "parent_track_id", "generation"]
        if tracks_csv.exists():
            _tracks_raw = pd.read_csv(tracks_csv)
            if "lineage_id" in _tracks_raw.columns:
                _lineage_map = (
                    _tracks_raw.drop_duplicates("track_id")[_lineage_cols]
                    .copy()
                )
            else:
                _lineage_map = pd.DataFrame(columns=_lineage_cols)
        else:
            _lineage_map = pd.DataFrame(columns=_lineage_cols)

        if mode == "persistent":
            assignments_df, locked_labels, persistent_per_frame = assign_persistent_fates(
                linked_labels, frame_cell_pos,
            )
            assignments_df = assignments_df.sort_values("label_id").reset_index(
                drop=True
            )
            csv = work_dir / "assignments_persistent.csv"
            assignments_df.to_csv(csv, index=False)
            if len(persistent_per_frame) > 0:
                pf_csv = work_dir / "persistent_per_frame.csv"
                persistent_per_frame.to_csv(pf_csv, index=False)
                self._append_log(f"  Per-frame metrics -> {pf_csv.name}")
            persistent_by_cell = assignments_df.rename(
                columns={"label_id": "cell_id"}
            ).copy()
            persistent_by_cell["track_id"] = (
                persistent_by_cell["cell_id"].astype(int) - 1
            )
            if len(_lineage_map) > 0:
                persistent_by_cell = persistent_by_cell.merge(
                    _lineage_map, on="track_id", how="left"
                )
            persistent_by_cell_csv = work_dir / "persistent_by_cell.csv"
            persistent_by_cell.to_csv(persistent_by_cell_csv, index=False)
            self._append_log(
                f"  Fates: {dict(assignments_df['fate'].value_counts())}"
            )
        else:
            snapshot_df = assign_snapshot_fates(
                linked_labels, frame_cell_pos,
            )
            snapshot_df = snapshot_df.sort_values(
                ["label_id", "frame"]
            ).reset_index(drop=True)
            csv = work_dir / "snapshot.csv"
            snapshot_df.to_csv(csv, index=False)
            snapshot_by_cell = snapshot_df.rename(
                columns={"label_id": "cell_id"}
            ).copy()
            snapshot_by_cell["track_id"] = (
                snapshot_by_cell["cell_id"].astype(int) - 1
            )
            if len(_lineage_map) > 0:
                snapshot_by_cell = snapshot_by_cell.merge(
                    _lineage_map, on="track_id", how="left"
                )
            snapshot_by_cell_csv = work_dir / "snapshot_by_cell_long.csv"
            snapshot_by_cell.to_csv(snapshot_by_cell_csv, index=False)

            # Wide-format: one row per cell, columns for each frame's category
            # Include lineage columns alongside the pivoted frame categories
            _pivot_cols = ["cell_id"]
            if "lineage_id" in snapshot_by_cell.columns:
                _pivot_cols += ["lineage_id", "parent_track_id", "generation"]
            snapshot_wide = (
                snapshot_by_cell.pivot(
                    index="cell_id", columns="frame", values="category"
                )
                .reset_index()
                .sort_values("cell_id")
            )
            snapshot_wide.columns = [
                "cell_id"
                if c == "cell_id"
                else f"frame_{int(c)}_category"
                for c in snapshot_wide.columns
            ]
            # Merge lineage info into the wide table
            if "lineage_id" in snapshot_by_cell.columns:
                _lin_per_cell = (
                    snapshot_by_cell.drop_duplicates("cell_id")
                    [["cell_id", "lineage_id", "parent_track_id", "generation"]]
                )
                snapshot_wide = snapshot_wide.merge(
                    _lin_per_cell, on="cell_id", how="left"
                )
                # Move lineage columns right after cell_id
                _front = ["cell_id", "lineage_id", "parent_track_id", "generation"]
                _rest = [c for c in snapshot_wide.columns if c not in _front]
                snapshot_wide = snapshot_wide[_front + _rest]
            snapshot_wide_csv = work_dir / "snapshot_by_cell_wide.csv"
            snapshot_wide.to_csv(snapshot_wide_csv, index=False)
            cats = sorted(snapshot_df["category"].unique())
            self._append_log(f"  Categories: {cats}")

        self._set_progress(75, fmt="Computing percentages...")
        n_fr = linked_labels.shape[0]
        stem = tiff_path.stem
        if mode == "persistent":
            summary_df = compute_persistent_percentages(
                assignments_df, n_fr, fluor_names
            )
            fig, _ = plot_persistent_percentages(
                summary_df, fluor_names, title=stem
            )
            summary_csv = work_dir / "percentages_persistent.csv"
            plot_pdf = work_dir / "percentages_persistent.pdf"

            cutoff_pcts = [30, 40, 50, 60]
            for cutoff in cutoff_pcts:
                filt_df = filter_persistent_by_frame_presence(
                    assignments_df, linked_labels, n_fr, cutoff
                )
                n_cells = len(filt_df)
                if n_cells == 0:
                    self._append_log(
                        f"  No cells at \u2265{cutoff}% frame presence, skipping"
                    )
                    continue
                filt_summary = compute_persistent_percentages(
                    filt_df, n_fr, fluor_names
                )
                fig_pct, _ = plot_persistent_percentages(
                    filt_summary,
                    fluor_names,
                    title=f"{stem} \u2014 \u2265{cutoff}% frames (n={n_cells} cells)",
                )
                pct_pdf = work_dir / f"percentages_persistent_{cutoff}pct.pdf"
                fig_pct.savefig(str(pct_pdf), bbox_inches="tight")
                plt.close(fig_pct)
                filt_summary.to_csv(
                    work_dir / f"percentages_persistent_{cutoff}pct.csv",
                    index=False,
                )
        else:
            summary_df, cats = compute_snapshot_percentages(snapshot_df, n_fr)
            fig, _ = plot_snapshot_percentages(summary_df, cats, title=stem)
            summary_csv = work_dir / "percentages_snapshot.csv"
            plot_pdf = work_dir / "percentages_snapshot.pdf"
            tracks_csv = work_dir / "trackmate_tracks.csv"
            tracks_for_plot = (
                pd.read_csv(tracks_csv)
                if tracks_csv.exists()
                else pd.DataFrame(
                    columns=["track_id", "t", "y", "x", "quality"]
                )
            )

            cutoff_pcts = [30, 40, 50, 60]
            for cutoff in cutoff_pcts:
                ft, fs = filter_by_frame_presence(
                    tracks_for_plot, snapshot_df, n_fr, cutoff
                )
                n_cells = fs["label_id"].nunique() if len(fs) > 0 else 0

                filt_summary, filt_cats = compute_snapshot_percentages(fs, n_fr)
                fig_pct, _ = plot_snapshot_percentages(
                    filt_summary,
                    filt_cats,
                    title=f"{stem} — ≥{cutoff}% frames (n={n_cells} cells)",
                )
                pct_pdf = work_dir / f"percentages_snapshot_{cutoff}pct.pdf"
                fig_pct.savefig(str(pct_pdf), bbox_inches="tight")
                plt.close(fig_pct)
                filt_summary.to_csv(
                    work_dir / f"percentages_snapshot_{cutoff}pct.csv",
                    index=False,
                )

                fig_traj, _ = plot_snapshot_trajectories(
                    ft,
                    fs,
                    title=f"{stem} — trajectories ≥{cutoff}% frames (n={n_cells} cells)",
                )
                traj_pdf = work_dir / f"snapshot_trajectories_{cutoff}pct.pdf"
                fig_traj.savefig(str(traj_pdf), bbox_inches="tight")
                plt.close(fig_traj)

                # Timeline: one row per cell, coloured by category at each frame
                fig_tl, _ = plot_snapshot_cell_timelines(
                    fs,
                    tracks_df=ft,
                    title=f"{stem} — cell timelines ≥{cutoff}% frames (n={n_cells} cells)",
                )
                tl_pdf = work_dir / f"snapshot_timelines_{cutoff}pct.pdf"
                fig_tl.savefig(str(tl_pdf), bbox_inches="tight")
                plt.close(fig_tl)
        summary_df.to_csv(summary_csv, index=False)
        fig.savefig(str(plot_pdf), bbox_inches="tight")
        plt.close(fig)
        self._set_progress(100, fmt=f"{mode.capitalize()} analysis saved")
        self._append_log(
            f"[OK] Saved {mode} outputs -> {summary_csv.name}, "
            f"{plot_pdf.name}, plus cutoff plots at 30/40/50/60%"
        )

        tracks_csv = work_dir / "trackmate_tracks.csv"
        tracks_df = (
            pd.read_csv(tracks_csv)
            if tracks_csv.exists()
            else pd.DataFrame(
                columns=["track_id", "t", "y", "x", "quality"]
            )
        )
        if len(tracks_df) > 0:
            tracks_by_cell = tracks_df.sort_values(["track_id", "t"]).copy()
            tracks_by_cell["cell_id"] = tracks_by_cell["track_id"].astype(int) + 1
            tracks_by_cell["frame"] = tracks_by_cell["t"].astype(int)
            tracks_by_cell_csv = work_dir / "trackmate_tracks_by_cell.csv"
            tracks_by_cell.to_csv(tracks_by_cell_csv, index=False)
            self._append_log(
                f"[OK] Saved plot-ready tracks -> {tracks_by_cell_csv.name}"
            )

        result = {
            "filename": tiff_path.name,
            "analysis_mode": mode,
            "n_frames": n_fr,
            "n_tracked_cells": int(tracks_df["track_id"].nunique())
            if len(tracks_df) > 0
            else 0,
        }
        if mode == "persistent":
            for fn in fluor_names:
                result[f"n_{fn}"] = int((assignments_df["fate"] == fn).sum())
                result[f"final_pct_{fn}"] = float(summary_df[f"{fn}_pct"].iloc[-1])
            result["n_negative"] = int(
                (assignments_df["fate"] == "negative").sum()
            )
            result["final_total_pct"] = float(
                summary_df["total_positive_pct"].iloc[-1]
            )
        else:
            last = snapshot_df[snapshot_df["frame"] == n_fr - 1]
            for cat in sorted(snapshot_df["category"].unique()):
                result[f"final_pct_{cat}"] = (
                    100.0 * (last["category"] == cat).sum() / max(len(last), 1)
                )

        run_data = {
            "bf_image": bf_image,
            "fluor_images": fluor_images,
            "masks_stack": tifffile.imread(
                str(work_dir / "masks_stack.tiff")
            ).astype(np.uint16)
            if (work_dir / "masks_stack.tiff").exists()
            else np.zeros_like(linked_labels, dtype=np.uint16),
            "linked_labels": linked_labels,
            "positive_cell_labels": positive_cell_labels,
            "locked_labels": locked_labels,
            "tracks_df": tracks_df,
            "mode": mode,
        }
        return result, run_data

    def _run_segmentation_only(self):
        """Run only Cellpose segmentation and save masks."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            out = self._run_segmentation_stage(tiff_path, work_dir)
            self._show_in_viewer(
                {
                    "bf_image": out["bf_image"],
                    "fluor_images": out["fluor_images"],
                    "masks_stack": out["masks_stack"],
                    "linked_labels": np.zeros_like(
                        out["masks_stack"], dtype=np.uint32
                    ),
                    "positive_cell_labels": {},
                    "locked_labels": None,
                    "tracks_df": pd.DataFrame(
                        columns=["track_id", "t", "y", "x", "quality"]
                    ),
                    "mode": "snapshot",
                }
            )
        except Exception as e:
            self._set_progress(0, fmt="Error")
            self._append_log(f"[ERROR] {type(e).__name__}: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_tracking_only(self):
        """Run only TrackMate tracking from previously saved masks."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            bf_image, fluor_images, _, _ = self._load_image_channels(tiff_path)
            out = self._run_tracking_stage(work_dir)
            masks_path = work_dir / "masks_stack.tiff"
            masks_stack = (
                tifffile.imread(str(masks_path)).astype(np.uint16)
                if masks_path.exists()
                else np.zeros_like(out["linked_labels"], dtype=np.uint16)
            )
            self._show_in_viewer(
                {
                    "bf_image": bf_image,
                    "fluor_images": fluor_images,
                    "masks_stack": masks_stack,
                    "linked_labels": out["linked_labels"],
                    "positive_cell_labels": {},
                    "locked_labels": None,
                    "tracks_df": out["tracks_df"],
                    "mode": "snapshot",
                }
            )
        except Exception as e:
            self._set_progress(0, fmt="Error")
            self._append_log(f"[ERROR] {type(e).__name__}: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_persistent_analysis(self):
        """Run persistent analysis from saved tracking outputs."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            result, run_data = self._run_analysis_stage(
                tiff_path, work_dir, mode="persistent"
            )
            self._results.append(result)
            self._results_df = pd.DataFrame(self._results)
            self._show_in_viewer(run_data)
        except Exception as e:
            self._set_progress(0, fmt="Error")
            self._append_log(f"[ERROR] {type(e).__name__}: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_snapshot_analysis(self):
        """Run snapshot analysis from saved tracking outputs."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            result, run_data = self._run_analysis_stage(
                tiff_path, work_dir, mode="snapshot"
            )
            self._results.append(result)
            self._results_df = pd.DataFrame(self._results)
            self._show_in_viewer(run_data)
        except Exception as e:
            self._set_progress(0, fmt="Error")
            self._append_log(f"[ERROR] {type(e).__name__}: {e}")
        finally:
            self._set_buttons_enabled(True)

    def _run_all_single_image(self):
        """Run segmentation + tracking + both analyses on one image."""
        self._set_buttons_enabled(False)
        try:
            tiff_path, _, work_dir = self._resolve_single_run_paths()
            self._append_log("[INFO] Running full pipeline (all stages)...")
            self._run_segmentation_stage(tiff_path, work_dir)
            self._run_tracking_stage(work_dir)

            persistent_result, persistent_data = self._run_analysis_stage(
                tiff_path, work_dir, mode="persistent"
            )
            snapshot_result, _ = self._run_analysis_stage(
                tiff_path, work_dir, mode="snapshot"
            )

            self._results.extend([persistent_result, snapshot_result])
            self._results_df = pd.DataFrame(self._results)
            self._show_in_viewer(persistent_data)
            self._append_log(f"[OK] Full pipeline complete: {tiff_path.name}")
        except Exception as e:
            self._set_progress(0, fmt="Error")
            self._append_log(f"[ERROR] {type(e).__name__}: {e}")
        finally:
            self._set_buttons_enabled(True)

    # -- viewer -------------------------------------------------------------

    def _show_in_viewer(self, data: dict):
        """Populate the napari viewer with all pipeline outputs."""
        self.viewer.layers.clear()

        self.viewer.add_image(
            data["bf_image"],
            name="Brightfield",
            colormap="gray",
            blending="translucent",
            opacity=0.7,
        )

        colour_assignments = assign_colours(list(data["fluor_images"].keys()))
        for fn, img in data["fluor_images"].items():
            napari_cmap = colour_assignments[fn]["napari"]
            self.viewer.add_image(
                img, name=f"{fn} fluorescence",
                colormap=napari_cmap, blending="additive",
            )

        for fn, lbl in data["positive_cell_labels"].items():
            self.viewer.add_labels(
                lbl, name=f"{fn} positive cells", opacity=0.35
            )

        self.viewer.add_labels(
            data["linked_labels"], name="Linked labels", opacity=0.20
        )

        if data["mode"] == "persistent" and data["locked_labels"] is not None:
            for fn, lbl in data["locked_labels"].items():
                self.viewer.add_labels(
                    lbl, name=f"Persistent {fn}", opacity=0.60
                )

        self.viewer.add_labels(
            data["masks_stack"].astype(np.uint32),
            name="Cellpose masks",
            opacity=0.15,
        )

        tdf = data["tracks_df"]
        if len(tdf) > 0:
            track_arr = (
                tdf[["track_id", "t", "y", "x"]]
                .sort_values(["track_id", "t"])
                .to_numpy(dtype=float)
            )
            self.viewer.add_tracks(
                track_arr, name="Tracks", tail_length=50, opacity=0.8
            )

    # -- button handlers ----------------------------------------------------

    def _analyse_folder(self):
        """Batch-process every TIFF in a folder, save combined CSV."""
        folder = Path(str(self.file_panel.batch_folder.value))
        if not folder.exists() or not folder.is_dir():
            self._append_log(
                "[ERROR] Select a valid batch folder in the Files panel."
            )
            return
        out_dir = Path(str(self.file_panel.output_directory.value))
        if not str(out_dir).strip() or str(out_dir) == ".":
            out_dir = folder / "cell_death_output"

        tiff_files = sorted(
            f
            for f in folder.iterdir()
            if f.suffix.lower() in (".tif", ".tiff") and f.is_file()
        )
        if not tiff_files:
            self._append_log("[WARN] No .tif / .tiff files found in folder.")
            return

        n = len(tiff_files)
        self._append_log(f"[INFO] Batch: {n} files in {folder.name}")
        batch_results: List[dict] = []

        self._set_buttons_enabled(False)
        try:
            for idx, fp in enumerate(tiff_files):
                self._append_log(
                    f"--- File {idx + 1}/{n}: {fp.name} ---"
                )
                try:
                    work_dir = out_dir / fp.stem
                    work_dir.mkdir(parents=True, exist_ok=True)
                    self._run_segmentation_stage(fp, work_dir)
                    self._run_tracking_stage(work_dir)
                    mode = str(self.analysis_panel.analysis_mode.value)
                    result, _ = self._run_analysis_stage(fp, work_dir, mode)
                    batch_results.append(result)
                except Exception as e:
                    batch_results.append(
                        {"filename": fp.name, "error": str(e)}
                    )
                    self._append_log(
                        f"  FAILED: {type(e).__name__}: {e}"
                    )
                self._set_progress(
                    idx + 1, maximum=n,
                    fmt=f"Files: {idx + 1}/{n}",
                )

            self._results.extend(batch_results)
            self._results_df = pd.DataFrame(self._results)
            summary_path = out_dir / "batch_summary.csv"
            out_dir.mkdir(parents=True, exist_ok=True)
            self._results_df.to_csv(summary_path, index=False)
            self._append_log(f"[OK] Batch done. Summary -> {summary_path}")
        finally:
            self._set_buttons_enabled(True)

    def _save_results(self):
        """Save accumulated results to a user-chosen CSV file."""
        if self._results_df is None or self._results_df.empty:
            self._append_log("[WARN] No results to save yet.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            None,
            "Save results CSV",
            str(Path.home()),
            "CSV files (*.csv)",
        )
        if not filename:
            self._append_log("[INFO] Save cancelled.")
            return
        if not filename.lower().endswith(".csv"):
            filename += ".csv"

        self._results_df.to_csv(filename, index=False)
        self._append_log(f"[OK] Saved results -> {filename}")


# ---------------------------------------------------------------------------
#  Launch helpers
# ---------------------------------------------------------------------------

def launch():
    """Create the GUI and return the app instance (for notebook use)."""
    return CellDeathAnalysisApp()


def main():
    """Launch the standalone GUI."""
    launch()
    napari.run()
    raise SystemExit(0)


if __name__ == "__main__":
    main()

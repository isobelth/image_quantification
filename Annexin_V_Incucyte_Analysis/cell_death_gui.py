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
from skimage.measure import label

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
    model_type="cyto3",
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
        model = models.CellposeModel(
            gpu=gpu, pretrained_model=str(custom_model_path)
        )
    else:
        model = models.CellposeModel(gpu=gpu, model_type=model_type)

    if diameter is None:
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

    n_frames = stack.shape[0]
    out = np.zeros(stack.shape, dtype=np.uint16)

    for i in range(n_frames):
        m, _, _ = model.eval(
            stack[i],
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            resample=True,
        )
        out[i] = m.astype(np.uint16)
        if progress_callback is not None:
            progress_callback(i + 1, n_frames)

    return out[0] if n_frames == 1 else out


def segment_fluorescence(stacks, blur_sigma=1.0, threshold_method="mean"):
    """Blur, threshold, and label fluorescence stacks (1-3 fluorophores).

    *threshold_method* can be a single string (applied to all channels)
    or a ``dict`` mapping channel name to its own threshold method.
    """
    threshold_functions = {
        "mean": threshold_mean,
        "minimum": threshold_minimum,
        "yen": threshold_yen,
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
    }

    # Normalise to a dict keyed by channel name
    if isinstance(threshold_method, str):
        _thresh_map = {name: threshold_method.lower() for name in stacks}
    else:
        _thresh_map = {name: str(threshold_method.get(name, "mean")).lower()
                       for name in stacks}

    result = {"threshold_methods": _thresh_map, "blur_sigma": blur_sigma}

    for name, stack in stacks.items():
        tm_name = _thresh_map[name]
        if tm_name not in threshold_functions:
            raise ValueError(f"Unsupported threshold for '{name}': {tm_name}")
        fn = threshold_functions[tm_name]
        blurred = np.stack(
            [gaussian(f, sigma=blur_sigma, preserve_range=True) for f in stack],
            axis=0,
        )
        thresh = fn(blurred)
        positive = blurred > thresh
        positive_labels = np.stack(
            [label(f) for f in positive], axis=0
        ).astype(np.uint32)
        result[name] = {
            "blurred": blurred,
            "thresh": thresh,
            "positive": positive,
            "positive_labels": positive_labels,
        }
    return result


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

def assign_persistent_fates(linked_labels, positive_masks):
    """Persistent mode: fate = whichever fluorophore appears FIRST."""
    for name, mask in positive_masks.items():
        if linked_labels.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch: linked_labels {linked_labels.shape} "
                f"vs {name} {mask.shape}"
            )

    n_frames = linked_labels.shape[0]
    label_ids = np.unique(linked_labels)
    label_ids = label_ids[label_ids > 0]
    fluor_names = list(positive_masks.keys())

    rows = []
    for label_id in label_ids:
        first_frames = {fn: None for fn in fluor_names}
        area_counts = {fn: 0 for fn in fluor_names}
        for frame in range(n_frames):
            cell = linked_labels[frame] == label_id
            if not np.any(cell):
                continue
            for fn in fluor_names:
                pos_count = int(np.count_nonzero(
                    positive_masks[fn][frame] & cell
                ))
                area_counts[fn] += pos_count
                if first_frames[fn] is None and pos_count > 0:
                    first_frames[fn] = frame

        fate, first_positive = "negative", np.nan
        best = n_frames + 1
        for fn in fluor_names:
            ff = first_frames[fn]
            if ff is not None and ff < best:
                best, fate, first_positive = ff, fn, ff

        row = {"label_id": int(label_id)}
        for fn in fluor_names:
            row[f"first_{fn}_frame"] = (
                first_frames[fn] if first_frames[fn] is not None else np.nan
            )
            row[f"{fn}_positive_area"] = area_counts[fn]
        row["first_positive_frame"] = first_positive
        row["fate"] = fate
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .sort_values(["fate", "first_positive_frame", "label_id"])
        .reset_index(drop=True)
    )

    locked = {}
    for fn in fluor_names:
        ids = df.loc[df["fate"] == fn, "label_id"].to_numpy(dtype=np.uint32)
        locked[fn] = np.where(
            np.isin(linked_labels, ids), linked_labels, 0
        ).astype(np.uint32)

    return df, locked


def assign_snapshot_fates(linked_labels, positive_masks):
    """Snapshot mode: per-frame classification by active fluorophores."""
    for name, mask in positive_masks.items():
        if linked_labels.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch: linked_labels {linked_labels.shape} "
                f"vs {name} {mask.shape}"
            )

    fluor_names = list(positive_masks.keys())
    rows = []
    for frame in range(linked_labels.shape[0]):
        fl = linked_labels[frame]
        for lid in np.unique(fl):
            if lid == 0:
                continue
            cell = fl == lid
            active = {
                fn: bool(np.any(positive_masks[fn][frame] & cell))
                for fn in fluor_names
            }
            cat = "+".join(fn for fn in fluor_names if active[fn]) or "negative"
            row = {"label_id": int(lid), "frame": frame}
            row.update(active)
            for fn in fluor_names:
                row[f"{fn}_positive_area"] = int(np.count_nonzero(
                    positive_masks[fn][frame] & cell
                ))
            row["category"] = cat
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Percentages & plots
# ---------------------------------------------------------------------------

def compute_persistent_percentages(assignments_df, n_frames, fluor_names):
    n = len(assignments_df)
    if n == 0:
        raise ValueError("No tracked cells.")
    data = {"frame": np.arange(n_frames)}
    for fn in fluor_names:
        pcts = []
        for f in range(n_frames):
            c = (
                (assignments_df["fate"] == fn)
                & (assignments_df["first_positive_frame"] <= f)
            ).sum()
            pcts.append(100.0 * c / n)
        data[f"{fn}_pct"] = pcts
    data["total_positive_pct"] = [
        sum(data[f"{fn}_pct"][f] for fn in fluor_names) for f in range(n_frames)
    ]
    return pd.DataFrame(data)


def compute_snapshot_percentages(snapshot_df, n_frames):
    cats = sorted(
        snapshot_df["category"].unique(),
        key=lambda c: (c == "negative", c),
    )
    data = {"frame": np.arange(n_frames)}
    for cat in cats:
        pcts = []
        for f in range(n_frames):
            fd = snapshot_df[snapshot_df["frame"] == f]
            nt = len(fd)
            pcts.append(
                100.0 * (fd["category"] == cat).sum() / nt if nt > 0 else 0.0
            )
        data[f"{cat}_pct"] = pcts
    return pd.DataFrame(data), cats


# ---------------------------------------------------------------------------
#  Colour inference from fluorophore names
# ---------------------------------------------------------------------------

_COLOUR_KEYWORDS: Dict[str, dict] = {
    "red":     {"mpl": "tab:red",    "napari": "red",     "rgb": (1.0, 0.0, 0.0)},
    "green":   {"mpl": "tab:green",  "napari": "green",   "rgb": (0.0, 0.8, 0.0)},
    "blue":    {"mpl": "tab:blue",   "napari": "blue",    "rgb": (0.2, 0.4, 1.0)},
    "yellow":  {"mpl": "goldenrod",  "napari": "yellow",  "rgb": (0.9, 0.85, 0.0)},
    "cyan":    {"mpl": "tab:cyan",   "napari": "cyan",    "rgb": (0.0, 0.8, 0.8)},
    "magenta": {"mpl": "tab:pink",   "napari": "magenta", "rgb": (0.9, 0.0, 0.6)},
}

_FALLBACK_COLOURS = [
    {"mpl": "tab:red",    "napari": "red",   "rgb": (1.0, 0.0, 0.0)},
    {"mpl": "tab:green",  "napari": "green", "rgb": (0.0, 0.8, 0.0)},
    {"mpl": "tab:blue",   "napari": "blue",  "rgb": (0.2, 0.4, 1.0)},
    {"mpl": "tab:orange", "napari": "gray",  "rgb": (1.0, 0.5, 0.0)},
    {"mpl": "tab:purple", "napari": "gray",  "rgb": (0.5, 0.0, 0.8)},
]


def _infer_colour(name: str, index: int = 0) -> dict:
    """Return {"mpl", "napari", "rgb"} colour mapping for a fluorophore name."""
    low = name.lower()
    for kw, entry in _COLOUR_KEYWORDS.items():
        if kw in low:
            return entry
    return _FALLBACK_COLOURS[index % len(_FALLBACK_COLOURS)]


# ---------------------------------------------------------------------------


def plot_persistent_percentages(
    summary_df, fluor_names, title="Persistent positive cells over time"
):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, fn in enumerate(fluor_names):
        c = _infer_colour(fn, i)["mpl"]
        sns.lineplot(
            data=summary_df, x="frame", y=f"{fn}_pct",
            color=c, label=fn, ax=ax,
        )
    sns.lineplot(
        data=summary_df, x="frame", y="total_positive_pct",
        color="black", label="Total", ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return fig, ax


def plot_snapshot_percentages(
    summary_df, categories, title="Per-frame categories over time"
):
    fig, ax = plt.subplots(figsize=(8, 4))
    cat_cmap = _build_category_colormap(categories)
    for cat in categories:
        c = cat_cmap.get(cat, (0.5, 0.5, 0.5))
        sns.lineplot(
            data=summary_df, x="frame", y=f"{cat}_pct",
            color=c, label=cat, ax=ax,
        )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return fig, ax


def _filter_by_frame_presence(tracks_df, snapshot_df, n_frames, min_pct):
    """Keep only cells appearing in >= min_pct% of frames."""
    threshold = n_frames * min_pct / 100.0
    frame_counts = snapshot_df.groupby("label_id")["frame"].nunique()
    keep_ids = frame_counts[frame_counts >= threshold].index
    filt_snap = snapshot_df[snapshot_df["label_id"].isin(keep_ids)].copy()
    if tracks_df is not None and len(tracks_df) > 0:
        keep_track_ids = keep_ids.astype(int) - 1
        filt_tracks = tracks_df[tracks_df["track_id"].isin(keep_track_ids)].copy()
    else:
        filt_tracks = tracks_df
    return filt_tracks, filt_snap


def _filter_persistent_by_frame_presence(
    assignments_df, linked_labels, n_frames, min_pct
):
    """Keep only persistent cells appearing in >= min_pct% of frames."""
    threshold = n_frames * min_pct / 100.0
    all_ids = []
    for frame in range(linked_labels.shape[0]):
        ids = np.unique(linked_labels[frame])
        ids = ids[ids > 0]
        all_ids.extend(ids.tolist())
    frame_counts = pd.Series(all_ids).value_counts()
    keep_ids = frame_counts[frame_counts >= threshold].index
    return assignments_df[assignments_df["label_id"].isin(keep_ids)].copy()


def plot_snapshot_trajectories(
    tracks_df,
    snapshot_df,
    title="Snapshot trajectories by category",
):
    """Plot trajectories colored by per-frame snapshot category.

    When ``tracks_df`` contains ``lineage_id`` and ``parent_track_id``
    columns (from TrackMate splitting), dashed lines are drawn connecting
    the last position of a mother cell to the first position of each
    daughter, making division events visible.
    """
    if tracks_df is None or len(tracks_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No tracks available", ha="center", va="center")
        ax.axis("off")
        plt.tight_layout()
        return fig, ax

    categories = sorted(snapshot_df["category"].unique())
    color_map = _build_category_colormap(categories)
    if "negative" not in color_map:
        color_map["negative"] = (0.6, 0.6, 0.6)

    pts = tracks_df.copy()
    pts["frame"] = pts["t"].astype(int)
    pts["label_id"] = pts["track_id"].astype(int) + 1

    cat_map = snapshot_df[["label_id", "frame", "category"]].copy()
    merged = pts.merge(cat_map, on=["label_id", "frame"], how="left")
    merged["category"] = merged["category"].fillna("negative")

    fig, ax = plt.subplots(figsize=(8, 6))
    all_x, all_y = [], []

    for lid, tr in merged.groupby("label_id"):
        tr = tr.sort_values("frame")
        coords = tr[["x", "y"]].to_numpy(dtype=float)
        if len(coords) < 2:
            continue
        all_x.extend(coords[:, 0].tolist())
        all_y.extend(coords[:, 1].tolist())
        segments = np.stack([coords[:-1], coords[1:]], axis=1)
        seg_colors = [
            color_map.get(cat, (0.6, 0.6, 0.6))
            for cat in tr["category"].iloc[:-1]
        ]
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.5, alpha=0.85)
        ax.add_collection(lc)

    # Draw division connectors (mother → daughter dashed lines)
    has_lineage = "parent_track_id" in tracks_df.columns
    if has_lineage:
        # Build per-track first/last positions
        track_bounds = {}
        for tid, grp in merged.groupby("track_id"):
            grp = grp.sort_values("frame")
            first = grp.iloc[0]
            last = grp.iloc[-1]
            track_bounds[int(tid)] = {
                "first_xy": (float(first["x"]), float(first["y"])),
                "last_xy": (float(last["x"]), float(last["y"])),
            }
        for tid, grp in merged.drop_duplicates("track_id").iterrows():
            ptid = grp.get("parent_track_id")
            if pd.isna(ptid):
                continue
            ptid = int(ptid)
            child_tid = int(grp["track_id"])
            if ptid in track_bounds and child_tid in track_bounds:
                px, py = track_bounds[ptid]["last_xy"]
                cx, cy = track_bounds[child_tid]["first_xy"]
                ax.plot(
                    [px, cx], [py, cy],
                    color="black", linewidth=1.0, alpha=0.5,
                    linestyle="--", zorder=0,
                )

    if all_x and all_y:
        ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
        ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")

    handles = [
        plt.Line2D([0], [0], color=color_map[c], lw=2, label=c)
        for c in sorted(color_map.keys())
    ]
    ax.legend(handles=handles, title="Snapshot category", loc="best")
    plt.tight_layout()
    return fig, ax


def _build_category_colormap(categories):
    """Build a colour map for snapshot categories.

    Single fluorophores are coloured by name inference (e.g. a name
    containing 'red' gets red, 'green' gets green).  Combinations get
    blended RGB values.  'negative' is always grey.
    """
    # Discover unique single-fluorophore names in the order they first appear
    singles = []
    for cat in categories:
        if cat == "negative":
            continue
        for part in cat.split("+"):
            if part not in singles:
                singles.append(part)

    # Map each single fluorophore to its inferred RGB
    single_rgb = {
        name: np.array(_infer_colour(name, i)["rgb"])
        for i, name in enumerate(singles)
    }

    cmap = {}
    for cat in categories:
        if cat == "negative":
            cmap[cat] = (0.7, 0.7, 0.7)
            continue
        parts = cat.split("+")
        rgb = np.zeros(3)
        for p in parts:
            rgb += single_rgb.get(p, np.array([0.5, 0.5, 0.5]))
        rgb = np.clip(rgb, 0, 1)
        cmap[cat] = tuple(rgb.tolist())
    return cmap


def plot_snapshot_cell_timelines(
    snapshot_df,
    tracks_df=None,
    title="Cell status over time",
):
    """Swimlane / timeline plot showing each cell's category at every frame.

    Each row is one tracked cell; the x-axis is frame (time).  Segments are
    coloured by the cell's snapshot category at that frame, producing a
    visual timeline of transitions (e.g. red → red+green → green for FUCCI).

    When *tracks_df* contains ``lineage_id`` columns (from TrackMate splitting),
    cells are sorted by lineage so that mother and daughter cells are grouped
    together, and a vertical line marks the division frame.

    Returns (fig, ax).
    """
    if snapshot_df is None or len(snapshot_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(title)
        ax.text(0.5, 0.5, "No snapshot data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        return fig, ax

    categories = sorted(snapshot_df["category"].unique())
    color_map = _build_category_colormap(categories)

    # ------- lineage-aware ordering -------
    has_lineage = (
        tracks_df is not None
        and len(tracks_df) > 0
        and "lineage_id" in tracks_df.columns
    )
    if has_lineage:
        # Build label_id → lineage_id mapping
        tdf = tracks_df.copy()
        tdf["label_id"] = tdf["track_id"].astype(int) + 1
        lin_map = tdf.drop_duplicates("label_id")[["label_id", "lineage_id", "parent_track_id"]]
        lin_map = lin_map.set_index("label_id")

        present_ids = set(snapshot_df["label_id"].unique())

        def _lineage_sort_key(lid):
            """Sort key that groups families together.

            Uses the root number + lineage suffix so e.g.
            1, 1.1, 1.2, 2, 3, 3.1, 3.2 stay together.
            """
            lid_str = lin_map.loc[lid, "lineage_id"] if lid in lin_map.index else str(lid)
            # Pad each numeric part for correct lexicographic sorting
            parts = str(lid_str).split(".")
            return tuple(int(p) if p.isdigit() else 0 for p in parts)

        cell_ids = sorted(
            [lid for lid in snapshot_df["label_id"].unique()],
            key=_lineage_sort_key,
        )
        label_display = {}
        for lid in cell_ids:
            if lid in lin_map.index:
                label_display[lid] = str(lin_map.loc[lid, "lineage_id"])
            else:
                label_display[lid] = str(lid)
    else:
        cell_ids = sorted(snapshot_df["label_id"].unique())
        label_display = {lid: str(lid) for lid in cell_ids}

    n_cells = len(cell_ids)
    cell_y = {cid: i for i, cid in enumerate(cell_ids)}

    # Determine frame range
    min_frame = int(snapshot_df["frame"].min())
    max_frame = int(snapshot_df["frame"].max())

    height = max(3, min(n_cells * 0.25 + 1, 40))
    fig, ax = plt.subplots(figsize=(max(8, (max_frame - min_frame) * 0.15 + 2), height))

    # Draw one coloured rectangle per cell per frame
    for _, row in snapshot_df.iterrows():
        lid = row["label_id"]
        frame = int(row["frame"])
        cat = row["category"]
        colour = color_map.get(cat, (0.5, 0.5, 0.5))
        y = cell_y[lid]
        ax.barh(y, width=1, left=frame, height=0.8, color=colour,
                edgecolor="none", linewidth=0)

    # ------- draw division connectors -------
    if has_lineage:
        # For each daughter, draw a line from where the parent row ends
        # to where the daughter row begins
        for lid in cell_ids:
            if lid not in lin_map.index:
                continue
            parent_tid = lin_map.loc[lid, "parent_track_id"]
            if pd.isna(parent_tid):
                continue
            parent_lid = int(parent_tid) + 1
            if parent_lid not in cell_y:
                continue
            # Find the frame where the daughter first appears
            daughter_frames = snapshot_df.loc[
                snapshot_df["label_id"] == lid, "frame"
            ]
            if len(daughter_frames) == 0:
                continue
            div_frame = int(daughter_frames.min())
            y_parent = cell_y[parent_lid]
            y_daughter = cell_y[lid]
            ax.plot(
                [div_frame, div_frame],
                [y_parent, y_daughter],
                color="black", linewidth=0.8, alpha=0.6,
                linestyle="--",
            )

    ax.set_xlim(min_frame - 0.5, max_frame + 1.5)
    ax.set_ylim(-0.5, n_cells - 0.5)
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Cell (lineage ID)" if has_lineage else "Cell", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Thin y-tick labels only when there are few cells
    if n_cells <= 60:
        ax.set_yticks(range(n_cells))
        ax.set_yticklabels(
            [label_display[c] for c in cell_ids],
            fontsize=max(4, 8 - n_cells // 20),
        )
    else:
        ax.set_yticks([])

    # Legend
    handles = [
        Patch(facecolor=color_map[c], edgecolor="none", label=c)
        for c in categories
    ]
    ax.legend(handles=handles, title="Category", loc="upper right",
              fontsize=8, title_fontsize=9, framealpha=0.8)

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
        model_type: str = "cyto3",
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
                "choices": ["cyto3", "cyto2", "cyto", "nuclei"],
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
        pos_masks = {fn: fl_out[fn]["positive"] for fn in fluor_names}
        pos_labels = {fn: fl_out[fn]["positive_labels"] for fn in fluor_names}

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
            assignments_df, locked_labels = assign_persistent_fates(
                linked_labels, pos_masks,
            )
            assignments_df = assignments_df.sort_values("label_id").reset_index(
                drop=True
            )
            csv = work_dir / "assignments_persistent.csv"
            assignments_df.to_csv(csv, index=False)
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
                linked_labels, pos_masks,
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
                filt_df = _filter_persistent_by_frame_presence(
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
                ft, fs = _filter_by_frame_presence(
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
            "pos_labels": pos_labels,
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
                    "pos_labels": {},
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
                    "pos_labels": {},
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

        for i, (fn, img) in enumerate(data["fluor_images"].items()):
            cmap = _infer_colour(fn, i)["napari"]
            self.viewer.add_image(
                img, name=f"{fn} fluorescence",
                colormap=cmap, blending="additive",
            )

        for fn, lbl in data["pos_labels"].items():
            self.viewer.add_labels(
                lbl, name=f"{fn} positive", opacity=0.35
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

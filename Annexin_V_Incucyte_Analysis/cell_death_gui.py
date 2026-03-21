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

def cellpose_live_segmentation(
    stack,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    model_type="cyto3",
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

    model = models.CellposeModel(gpu=gpu, model_type=model_type)

    if diameter is None:
        m0, _, _ = model.eval(
            stack[0],
            diameter=None,
            channels=[0, 0],
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
            channels=[0, 0],
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
    """Blur, threshold, and label fluorescence stacks (1-3 fluorophores)."""
    threshold_functions = {
        "mean": threshold_mean,
        "yen": threshold_yen,
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
    }
    threshold_method = str(threshold_method).lower()
    if threshold_method not in threshold_functions:
        raise ValueError(f"Unsupported threshold: {threshold_method}")

    fn = threshold_functions[threshold_method]
    result = {"threshold_method": threshold_method, "blur_sigma": blur_sigma}

    for name, stack in stacks.items():
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
    rows = []
    for tid in tm.trackIDs(True):
        for s in sorted(
            tm.trackSpots(tid), key=lambda s: float(s.getFeature("FRAME"))
        ):
            rows.append(
                {
                    "track_id": int(tid),
                    "t": int(float(s.getFeature("FRAME"))),
                    "y": float(s.getFeature("POSITION_Y")),
                    "x": float(s.getFeature("POSITION_X")),
                    "quality": float(s.getFeature("QUALITY")),
                }
            )
    tracks_df = pd.DataFrame(
        rows, columns=["track_id", "t", "y", "x", "quality"]
    )
    csv_path = Path(output_directory) / "trackmate_tracks.csv"
    tracks_df.to_csv(csv_path, index=False)

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
        for frame in range(n_frames):
            cell = linked_labels[frame] == label_id
            if not np.any(cell):
                continue
            for fn in fluor_names:
                if first_frames[fn] is None and np.any(
                    positive_masks[fn][frame] & cell
                ):
                    first_frames[fn] = frame
            if all(v is not None for v in first_frames.values()):
                break

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


_FLUOR_COLORS = ["tab:red", "tab:green", "tab:blue"]


def plot_persistent_percentages(
    summary_df, fluor_names, title="Persistent positive cells over time"
):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, fn in enumerate(fluor_names):
        c = _FLUOR_COLORS[i] if i < len(_FLUOR_COLORS) else f"C{i}"
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
    pal = sns.color_palette("husl", len(categories))
    for cat, c in zip(categories, pal):
        sns.lineplot(
            data=summary_df, x="frame", y=f"{cat}_pct",
            color=c, label=cat, ax=ax,
        )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel="Frame", ylabel="% Cells", ylim=(0, 100), title=title)
    plt.tight_layout()
    return fig, ax


_VIEWER_CMAPS = ["red", "green", "blue"]


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
        fluor_2_name: str = "PI",
        fluor_2_channel: int = 1,
        fluor_3_name: str = "",
        fluor_3_channel: int = 2,
    ):
        pass

    @staticmethod
    def _analysis_placeholder(
        analysis_mode: str = "persistent",
        threshold_method: str = "mean",
        blur_sigma: float = 1.0,
    ):
        pass

    @staticmethod
    def _cellpose_placeholder(
        model_type: str = "cyto3",
        diameter: int = 0,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
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
                "label": "Single TIFF",
                "mode": "r",
                "filter": "*.tif",
            },
            batch_folder={"label": "Batch folder", "mode": "d"},
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
            fluor_2_name={"label": "Fluorophore 2 name"},
            fluor_2_channel={
                "label": "Fluorophore 2 channel",
                "min": 0,
                "max": 20,
            },
            fluor_3_name={"label": "Fluorophore 3 (blank=none)"},
            fluor_3_channel={
                "label": "Fluorophore 3 channel",
                "min": 0,
                "max": 20,
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
            threshold_method={
                "label": "Threshold",
                "choices": ["mean", "yen", "otsu", "triangle"],
            },
            blur_sigma={
                "label": "Blur sigma",
                "min": 0.1,
                "max": 20.0,
                "step": 0.1,
            },
            call_button=False,
        )

        # -- cellpose panel --
        self.cellpose_panel = magicgui(
            self._cellpose_placeholder,
            model_type={
                "label": "Model",
                "choices": ["cyto3", "cyto2", "cyto", "nuclei"],
            },
            diameter={"label": "Diameter (0=auto)", "min": 0, "max": 500},
            flow_threshold={
                "label": "Flow threshold",
                "min": 0.0,
                "max": 3.0,
                "step": 0.05,
            },
            cellprob_threshold={
                "label": "Cell prob threshold",
                "min": -6.0,
                "max": 6.0,
                "step": 0.5,
            },
            min_size={"label": "Min cell size (px)", "min": 0, "max": 2000},
            use_gpu={"label": "Use GPU"},
            call_button=False,
        )

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
            allow_merging={"label": "Allow merging"},
            call_button=False,
        )

        # -- action buttons --
        self._btn_single = magicgui(
            self._analyse_single_image, call_button="Analyse Single Image"
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
        self._btn_single.call_button.enabled = enabled
        self._btn_folder.call_button.enabled = enabled
        self._btn_save.call_button.enabled = enabled

    def _read_fluorophore_config(self):
        """Read channel panel -> (fluor_names, fluor_channels, bf_channel)."""
        bf = int(self.channel_panel.brightfield_channel.value)
        names: List[str] = []
        channels: Dict[str, int] = {}
        for attr_n, attr_c in [
            ("fluor_1_name", "fluor_1_channel"),
            ("fluor_2_name", "fluor_2_channel"),
            ("fluor_3_name", "fluor_3_channel"),
        ]:
            n = str(getattr(self.channel_panel, attr_n).value).strip()
            c = int(getattr(self.channel_panel, attr_c).value)
            if n:
                names.append(n)
                channels[n] = c
        if not names:
            raise ValueError("At least one fluorophore name must be provided.")
        return names, channels, bf

    # -- pipeline -----------------------------------------------------------

    def _run_pipeline(self, tiff_path: Path, out_dir: Path):
        """Run the full pipeline on one multi-channel timelapse TIFF.

        Returns *(result_dict, run_data)* where *run_data* holds numpy
        arrays suitable for populating the napari viewer.
        """
        work_dir = out_dir / tiff_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)

        fluor_names, fluor_channels, bf_ch = self._read_fluorophore_config()
        mode = str(self.analysis_panel.analysis_mode.value)
        thresh_method = str(self.analysis_panel.threshold_method.value)
        blur_sigma = float(self.analysis_panel.blur_sigma.value)

        cp_model = str(self.cellpose_panel.model_type.value)
        cp_diam = int(self.cellpose_panel.diameter.value)
        cp_flow = float(self.cellpose_panel.flow_threshold.value)
        cp_prob = float(self.cellpose_panel.cellprob_threshold.value)
        cp_min = int(self.cellpose_panel.min_size.value)
        cp_gpu = bool(self.cellpose_panel.use_gpu.value)

        tm_init = float(self.trackmate_panel.initial_search_radius.value)
        tm_search = float(self.trackmate_panel.search_radius.value)
        tm_gap = int(self.trackmate_panel.max_frame_gap.value)
        tm_split = bool(self.trackmate_panel.allow_splitting.value)
        tm_merge = bool(self.trackmate_panel.allow_merging.value)

        # 1. Load image  (expect 4-D: T, C, Y, X)
        self._set_progress(2, fmt="Loading image...")
        self._append_log(f"Loading: {tiff_path.name}")
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
        self._append_log(
            f"  Shape: {image.shape}  ({image.shape[0]} frames, "
            f"{n_channels} channels)"
        )

        # 2. Cellpose segmentation
        self._set_progress(5, fmt="Cellpose: starting...")
        self._append_log("Running Cellpose segmentation...")

        def _cp_cb(frame, total):
            pct = 5 + int(50 * frame / total)
            self._set_progress(pct, fmt=f"Cellpose: frame {frame}/{total}")

        masks_stack = cellpose_live_segmentation(
            bf_image,
            diameter=cp_diam if cp_diam > 0 else None,
            flow_threshold=cp_flow,
            cellprob_threshold=cp_prob,
            min_size=cp_min,
            model_type=cp_model,
            gpu=cp_gpu,
            progress_callback=_cp_cb,
        )
        masks_path = work_dir / "masks_stack.tiff"
        tifffile.imwrite(str(masks_path), masks_stack.astype(np.uint16))
        self._append_log(f"  Saved masks -> {masks_path.name}")

        # 3. TrackMate tracking
        self._set_progress(0, maximum=0, fmt="Running TrackMate...")
        self._append_log("Running TrackMate (UI may be unresponsive)...")
        tm_out = generate_trackmate_labels(
            masks_path=masks_path,
            output_directory=work_dir,
            initial_search_radius=tm_init,
            search_radius=tm_search,
            max_frame_gap=tm_gap,
            allow_track_splitting=tm_split,
            allow_track_merging=tm_merge,
            ij=self._ij,
        )
        self._ij = tm_out["ij"]  # cache for reuse
        tracks_df = tm_out["trackmate_tracks_df"]
        linked_labels = tm_out["linked_labels"]
        n_tracks = int(tracks_df["track_id"].nunique())
        self._set_progress(80, fmt="TrackMate done")
        self._append_log(
            f"  Tracked {n_tracks} cells, {len(tracks_df)} points"
        )

        # 4. Fluorescence segmentation
        self._set_progress(85, fmt="Fluorescence segmentation...")
        self._append_log("Segmenting fluorescence channels...")
        fl_out = segment_fluorescence(
            fluor_images, blur_sigma=blur_sigma, threshold_method=thresh_method
        )
        pos_masks = {fn: fl_out[fn]["positive"] for fn in fluor_names}
        pos_labels = {fn: fl_out[fn]["positive_labels"] for fn in fluor_names}

        # 5. Fate assignment
        self._set_progress(90, fmt="Assigning fates...")
        self._append_log(f"Assigning fates (mode={mode})...")
        locked_labels = None
        assignments_df = None
        snapshot_df = None

        if mode == "persistent":
            assignments_df, locked_labels = assign_persistent_fates(
                linked_labels, pos_masks
            )
            csv = work_dir / "assignments.csv"
            assignments_df.to_csv(csv, index=False)
            self._append_log(
                f"  Fates: {dict(assignments_df['fate'].value_counts())}"
            )
        else:
            snapshot_df = assign_snapshot_fates(linked_labels, pos_masks)
            csv = work_dir / "snapshot.csv"
            snapshot_df.to_csv(csv, index=False)
            cats = sorted(snapshot_df["category"].unique())
            self._append_log(f"  Categories: {cats}")

        # 6. Percentages + plot
        self._set_progress(95, fmt="Computing percentages...")
        n_fr = linked_labels.shape[0]
        stem = tiff_path.stem

        if mode == "persistent":
            summary_df = compute_persistent_percentages(
                assignments_df, n_fr, fluor_names
            )
            fig, _ = plot_persistent_percentages(
                summary_df, fluor_names, title=stem
            )
        else:
            summary_df, cats = compute_snapshot_percentages(
                snapshot_df, n_fr
            )
            fig, _ = plot_snapshot_percentages(summary_df, cats, title=stem)

        summary_csv = work_dir / "percentages.csv"
        plot_pdf = work_dir / "percentages.pdf"
        summary_df.to_csv(summary_csv, index=False)
        fig.savefig(str(plot_pdf), bbox_inches="tight")
        plt.close(fig)
        self._append_log(
            f"  Saved: {summary_csv.name}, {plot_pdf.name}"
        )
        self._set_progress(100, fmt="Done")

        # -- build summary result row --
        result = {
            "filename": tiff_path.name,
            "analysis_mode": mode,
            "n_frames": n_fr,
            "n_tracked_cells": n_tracks,
        }
        if mode == "persistent":
            for fn in fluor_names:
                result[f"n_{fn}"] = int(
                    (assignments_df["fate"] == fn).sum()
                )
                result[f"final_pct_{fn}"] = float(
                    summary_df[f"{fn}_pct"].iloc[-1]
                )
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
            "masks_stack": masks_stack,
            "linked_labels": linked_labels,
            "pos_labels": pos_labels,
            "locked_labels": locked_labels,
            "tracks_df": tracks_df,
            "mode": mode,
        }
        return result, run_data

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
            cmap = _VIEWER_CMAPS[i] if i < len(_VIEWER_CMAPS) else "gray"
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
            pts = tdf[["t", "y", "x"]].to_numpy(dtype=float)
            self.viewer.add_points(
                pts, name="Track points",
                size=3, face_color="yellow", opacity=0.6,
            )

    # -- button handlers ----------------------------------------------------

    def _analyse_single_image(self):
        """Run the full pipeline on a single TIFF and show in napari."""
        tiff_path = Path(str(self.file_panel.single_tiff.value))
        if not tiff_path.exists() or not tiff_path.is_file():
            self._append_log(
                "[ERROR] Select a valid TIFF file in the Files panel."
            )
            return
        out_dir = Path(str(self.file_panel.output_directory.value))
        if not str(out_dir).strip() or str(out_dir) == ".":
            out_dir = tiff_path.parent / "cell_death_output"

        self._set_buttons_enabled(False)
        try:
            result, run_data = self._run_pipeline(tiff_path, out_dir)
            self._results.append(result)
            self._results_df = pd.DataFrame(self._results)
            self._show_in_viewer(run_data)
            self._append_log(
                f"[OK] Analysis complete: {tiff_path.name}"
            )
        except Exception as e:
            self._set_progress(0, fmt="Error")
            self._append_log(f"[ERROR] {type(e).__name__}: {e}")
        finally:
            self._set_buttons_enabled(True)

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
                    result, _ = self._run_pipeline(fp, out_dir)
                    batch_results.append(result)
                except Exception as e:
                    batch_results.append(
                        {"filename": fp.name, "error": str(e)}
                    )
                    self._append_log(
                        f"  FAILED: {type(e).__name__}: {e}"
                    )
                # Update batch-level progress after each file
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

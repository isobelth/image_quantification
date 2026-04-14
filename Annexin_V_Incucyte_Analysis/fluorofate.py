"""FluoroFate — napari + magicgui GUI.

Pipeline: Cellpose segmentation -> TrackMate tracking -> fluorescence
thresholding -> fate assignment (persistent / snapshot) -> percentage
trajectories.

Two workflows:
  - Analyse Single Image : full pipeline, all outputs shown in napari.
  - Analyse All in Folder : batch every TIFF in a folder -> CSV.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import tifffile

from magicgui import magicgui
from magicgui.widgets import TextEdit
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from utils import running_in_notebook
from colours import assign_colours
from measurement import compute_cell_positivity
from segmentation import cellpose_live_segmentation, segment_fluorescence
from tracking import generate_trackmate_labels
from fate_assignment import (
    assign_persistent_fates,
    assign_snapshot_fates,
    compute_persistent_percentages,
    compute_snapshot_percentages,
    filter_by_frame_presence,
    filter_persistent_by_frame_presence,
)
from plotting import (
    plot_persistent_percentages,
    plot_snapshot_percentages,
    plot_snapshot_trajectories,
    plot_snapshot_cell_timelines,
)

warnings.filterwarnings("ignore")


# Cellpose default parameters
_CP_DEFAULT_DIAMETER = None
_CP_DEFAULT_FLOW_THRESHOLD = 0.4
_CP_DEFAULT_CELLPROB_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
#  GUI Application
# ---------------------------------------------------------------------------

class FluoroFateApp:
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
        fluor_1_name: str = "Green",
        fluor_1_channel: int = 0,
        fluor_1_threshold: str = "otsu",
        fluor_2_name: str = "Red",
        fluor_2_channel: int = 1,
        fluor_2_threshold: str = "otsu",
        fluor_3_name: str = "",
        fluor_3_channel: int = 2,
        fluor_3_threshold: str = "otsu",
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
        max_frame_gap: int = 2,
        allow_splitting: bool = False,
        splitting_max_distance: float = 15.0,
        allow_merging: bool = False,
    ):
        pass

    # -- init ---------------------------------------------------------------

    def __init__(self):
        self.viewer = napari.Viewer(title="FluoroFate")

        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.setQuitOnLastWindowClosed(not running_in_notebook())

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
    return FluoroFateApp()


def main():
    """Launch the standalone GUI."""
    launch()
    napari.run()
    raise SystemExit(0)


if __name__ == "__main__":
    main()

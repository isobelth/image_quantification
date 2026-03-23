"""
Acinar Analysis GUI
===================
Standalone magicgui GUI for batch 3-D acinar image analysis.

Launch from a notebook (run ``%gui qt`` first)::

    %gui qt
    from acinar_gui import launch
    app = launch()

Or standalone::

    python acinar_gui.py
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from magicgui import magicgui
from magicgui.widgets import Container, Label, TextEdit, ProgressBar

from acinar_analysis import batch_analyse


# ---------------------------------------------------------------------------
#  Per-analysis requirements
# ---------------------------------------------------------------------------

_REQUIREMENTS: Dict[str, dict] = {
    "acinus_shape": {
        "folders": [],
        "channels": [],
        "label": "Acinus Shape",
    },
    "cell_nuclear_shape": {
        "folders": ["nuclear_mask_dir", "membrane_mask_dir"],
        "channels": [],
        "label": "Cell & Nuclear Shape",
    },
    "protein_polarisation": {
        "folders": [],
        "channels": ["protein_channel"],
        "label": "Protein Polarisation",
    },
    "apoptosis": {
        "folders": ["c3_mask_dir", "nuclear_mask_dir"],
        "channels": ["c3_channel"],
        "label": "Apoptosis (C3)",
    },
    "protein_proximity": {
        "folders": ["c3_mask_dir", "nuclear_mask_dir"],
        "channels": ["c3_channel", "proximity_protein_channel"],
        "label": "Protein Proximity",
    },
    "proliferation": {
        "folders": ["edu_mask_dir", "nuclear_mask_dir"],
        "channels": ["edu_channel"],
        "label": "Proliferation (EdU)",
    },
    "mitochondria": {
        "folders": ["nuclear_mask_dir", "membrane_mask_dir", "mito_mask_dir"],
        "channels": [],
        "label": "Mitochondria",
    },
}

_FOLDER_LABELS = {
    "nuclear_mask_dir": "Nuclear Mask Folder",
    "membrane_mask_dir": "Membrane Mask Folder",
    "c3_mask_dir": "C3 Mask Folder",
    "edu_mask_dir": "EdU Mask Folder",
    "mito_mask_dir": "Mito Mask Folder",
}

_CHANNEL_LABELS = {
    "protein_channel": "Protein Channel",
    "c3_channel": "C3 Channel",
    "edu_channel": "EdU Channel",
    "proximity_protein_channel": "Proximity Protein Channel",
}


# ---------------------------------------------------------------------------
#  GUI class
# ---------------------------------------------------------------------------

class AcinarAnalysisGUI:
    """Standalone magicgui-based interface for batch acinar image analysis."""

    def __init__(self):
        self._running = False
        self._results: List[pd.DataFrame] = []

        # ---- Build magicgui panels ----

        self.folder_panel = magicgui(
            self._folder_stub,
            image_dir={"label": "Image Folder", "mode": "d"},
            nuclear_mask_dir={"label": "Nuclear Mask Folder", "mode": "d"},
            membrane_mask_dir={"label": "Membrane Mask Folder", "mode": "d"},
            c3_mask_dir={"label": "C3 Mask Folder", "mode": "d"},
            edu_mask_dir={"label": "EdU Mask Folder", "mode": "d"},
            mito_mask_dir={"label": "Mito Mask Folder", "mode": "d"},
            call_button=False,
        )

        self.channel_panel = magicgui(
            self._channel_stub,
            nuclear_channel={"label": "Nuclear Channel", "value": 0, "min": 0, "max": 20},
            membrane_channel={"label": "Membrane Ch (-1=none)", "value": 2, "min": -1, "max": 20},
            protein_channel={"label": "Protein Ch (-1=none)", "value": -1, "min": -1, "max": 20},
            c3_channel={"label": "C3 Ch (-1=none)", "value": 3, "min": -1, "max": 20},
            edu_channel={"label": "EdU Ch (-1=none)", "value": -1, "min": -1, "max": 20},
            mito_channel={"label": "Mito Ch (-1=none)", "value": -1, "min": -1, "max": 20},
            proximity_protein_channel={"label": "Prox. Protein Ch (-1=none)", "value": -1, "min": -1, "max": 20},
            file_extension={"label": "File Extension", "value": "tif"},
            n_jobs={"label": "Parallel Jobs", "value": 3, "min": 1, "max": 32},
            call_button=False,
        )

        self.analysis_panel = magicgui(
            self._analysis_stub,
            acinus_shape={"label": "Acinus Shape", "value": False},
            cell_nuclear_shape={"label": "Cell & Nuclear Shape", "value": False},
            protein_polarisation={"label": "Protein Polarisation", "value": False},
            apoptosis={"label": "Apoptosis (C3)", "value": False},
            protein_proximity={"label": "Protein Proximity", "value": False},
            proliferation={"label": "Proliferation (EdU)", "value": False},
            mitochondria={"label": "Mitochondria", "value": False},
            save_qc_plots={"label": "Save QC Plots", "value": True},
            output_csv={"label": "Output CSV", "mode": "w", "value": "acinar_results.csv"},
            call_button=False,
        )

        self._btn_run = magicgui(self._on_run_clicked, call_button="Run Analysis")

        self.progress_bar = ProgressBar(min=0, max=100, value=0, label="Progress")
        self.progress_bar.visible = False

        # Build welcome / requirements text
        welcome = "Select folders, set channels, tick analyses, click Run.\n\n"
        for info in _REQUIREMENTS.values():
            reqs = [_FOLDER_LABELS[f] for f in info["folders"]]
            reqs += [f"{_CHANNEL_LABELS[c]} >= 0" for c in info["channels"]]
            req_str = ", ".join(reqs) if reqs else "(Image Folder only)"
            welcome += f"  {info['label']:25s} {req_str}\n"

        self._log_widget = TextEdit(value=welcome)
        self._log_widget.min_height = 180
        try:
            self._log_widget.native.setReadOnly(True)
        except Exception:
            pass

        # ---- Assemble into one Container ----
        self.widget = Container(
            widgets=[
                Label(value="── Input Folders ──"),
                self.folder_panel,
                Label(value="── Channel Config ──"),
                self.channel_panel,
                Label(value="── Analyses ──"),
                self.analysis_panel,
                Label(value="── Run ──"),
                self._btn_run,
                self.progress_bar,
                self._log_widget,
            ],
            label="Acinar Analysis",
        )
        self.widget.native.setWindowTitle("Acinar Analysis")
        self.widget.native.setMinimumWidth(520)

        self.folder_panel.image_dir.changed.connect(self._on_image_dir_changed)

        self.widget.show()

    # ------------------------------------------------------------------
    #  Stub functions (no call_button → panels are purely config)
    # ------------------------------------------------------------------

    @staticmethod
    def _folder_stub(
        image_dir: Path = Path(),
        nuclear_mask_dir: Path = Path(),
        membrane_mask_dir: Path = Path(),
        c3_mask_dir: Path = Path(),
        edu_mask_dir: Path = Path(),
        mito_mask_dir: Path = Path(),
    ):
        return None

    @staticmethod
    def _analysis_stub(
        acinus_shape: bool = False,
        cell_nuclear_shape: bool = False,
        protein_polarisation: bool = False,
        apoptosis: bool = False,
        protein_proximity: bool = False,
        proliferation: bool = False,
        mitochondria: bool = False,
        save_qc_plots: bool = True,
        output_csv: Path = Path("acinar_results.csv"),
    ):
        return None

    @staticmethod
    def _channel_stub(
        nuclear_channel: int = 0,
        membrane_channel: int = 2,
        protein_channel: int = -1,
        c3_channel: int = 3,
        edu_channel: int = -1,
        mito_channel: int = -1,
        proximity_protein_channel: int = -1,
        file_extension: str = "tif",
        n_jobs: int = 3,
    ):
        return None

    # ------------------------------------------------------------------
    #  Running state helpers
    # ------------------------------------------------------------------

    def _set_running(self, running: bool):
        self._running = running
        self._btn_run.call_button.enabled = not running
        self.progress_bar.visible = running
        if running:
            self.progress_bar.value = 0

    def _update_progress(self, completed: int, total: int):
        self.progress_bar.max = total
        self.progress_bar.value = completed
        self._log(f"  [PROGRESS] {completed}/{total} images processed")

    def _log(self, msg: str):
        cur = self._log_widget.value.rstrip()
        self._log_widget.value = (cur + "\n" + msg) if cur else msg

    # ------------------------------------------------------------------
    #  Read GUI values
    # ------------------------------------------------------------------

    @staticmethod
    def _dir_or_none(path_value) -> Optional[str]:
        p = Path(str(path_value))
        if str(p) in (".", "") or not p.is_dir():
            return None
        return str(p)

    @staticmethod
    def _channel_or_none(value: int) -> Optional[int]:
        return value if value >= 0 else None

    def _read_folders(self) -> Dict[str, Optional[str]]:
        return {
            k: self._dir_or_none(getattr(self.folder_panel, k).value)
            for k in (
                "image_dir", "nuclear_mask_dir", "membrane_mask_dir",
                "c3_mask_dir", "edu_mask_dir", "mito_mask_dir",
            )
        }

    def _read_channels(self) -> Dict[str, Optional[int]]:
        return {
            k: self._channel_or_none(int(getattr(self.channel_panel, k).value))
            for k in (
                "nuclear_channel", "membrane_channel", "protein_channel",
                "c3_channel", "edu_channel", "mito_channel",
                "proximity_protein_channel",
            )
        }

    def _selected_analyses(self) -> List[str]:
        return [n for n in _REQUIREMENTS if getattr(self.analysis_panel, n).value]

    def _validate(self, analyses, folders, channels) -> List[str]:
        errors: List[str] = []
        if not analyses:
            errors.append("No analyses selected.")
            return errors
        if folders.get("image_dir") is None:
            errors.append("Image Folder is required.")
        for name in analyses:
            reqs = _REQUIREMENTS[name]
            for fkey in reqs["folders"]:
                if folders.get(fkey) is None:
                    errors.append(f"'{reqs['label']}' requires {_FOLDER_LABELS[fkey]}.")
            for ckey in reqs["channels"]:
                if channels.get(ckey) is None:
                    errors.append(f"'{reqs['label']}' requires {_CHANNEL_LABELS[ckey]} (>= 0).")
        return errors

    # ------------------------------------------------------------------
    #  Callbacks
    # ------------------------------------------------------------------

    def _on_image_dir_changed(self, value):
        d = self._dir_or_none(value)
        if d is None:
            return
        ext = str(self.channel_panel.file_extension.value)
        n = len(list(Path(d).rglob(f"*.{ext}")))
        self._log(f"[INFO] Found {n} .{ext} file(s) in {d}")

    def _on_run_clicked(self):
        if self._running:
            self._log("[WARN] Analysis already running.")
            return

        analyses = self._selected_analyses()
        folders = self._read_folders()
        channels = self._read_channels()

        errors = self._validate(analyses, folders, channels)
        if errors:
            self._log("--- Validation failed ---")
            for e in errors:
                self._log(f"  [ERROR] {e}")
            return

        out_path = str(self.analysis_panel.output_csv.value)
        if not out_path.lower().endswith(".csv"):
            out_path += ".csv"

        qc_dir = None
        if self.analysis_panel.save_qc_plots.value:
            qc_dir = str(Path(out_path).parent / "qc_plots")

        self._log("=" * 50)
        self._log(f"[INFO] Analyses: {', '.join(analyses)}")
        self._log(f"[INFO] Images : {folders['image_dir']}")
        self._log(f"[INFO] Output : {out_path}")
        if qc_dir:
            self._log(f"[INFO] QC plots: {qc_dir}")
        self._set_running(True)

        threading.Thread(
            target=self._run_batch,
            args=(folders, channels, analyses, out_path, qc_dir),
            daemon=True,
        ).start()

    def _run_batch(self, folders, channels, analyses, out_path, qc_dir):
        try:
            results = batch_analyse(
                image_dir=folders["image_dir"],
                analyses=analyses,
                file_extension=str(self.channel_panel.file_extension.value),
                n_jobs=int(self.channel_panel.n_jobs.value),
                output_csv=out_path,
                nuclear_mask_dir=folders.get("nuclear_mask_dir"),
                membrane_mask_dir=folders.get("membrane_mask_dir"),
                c3_mask_dir=folders.get("c3_mask_dir"),
                edu_mask_dir=folders.get("edu_mask_dir"),
                mito_mask_dir=folders.get("mito_mask_dir"),
                nuclear_channel=channels.get("nuclear_channel", 0),
                membrane_channel=channels.get("membrane_channel"),
                protein_channel=channels.get("protein_channel"),
                c3_channel=channels.get("c3_channel"),
                edu_channel=channels.get("edu_channel"),
                mito_channel=channels.get("mito_channel"),
                proximity_protein_channel=channels.get("proximity_protein_channel"),
                qc_dir=qc_dir,
                progress_callback=self._update_progress,
            )
            for name, df in results.items():
                n = len(df) if df is not None else 0
                self._log(f"  [OK] {name}: {n} row(s)")
                if df is not None and not df.empty:
                    self._results.append(df)
            self._log(f"[OK] Complete. CSVs saved with prefix: {out_path}")
        except Exception as exc:
            self._log(f"[ERROR] {type(exc).__name__}: {exc}")
        finally:
            self._set_running(False)
            self._log("[INFO] Ready for next run.\n")


# ---------------------------------------------------------------------------
#  Public launch helper
# ---------------------------------------------------------------------------

def launch() -> AcinarAnalysisGUI:
    """Create and show the Acinar Analysis GUI. Returns the app instance."""
    return AcinarAnalysisGUI()


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication
    _qapp = QApplication.instance() or QApplication([])
    gui = launch()
    _qapp.exec_()

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

The GUI collects all settings (folders, channels, analyses) and validates
them.  When the user clicks "Run Analysis" the window closes and
``batch_analyse`` runs directly — with full tqdm progress visible in
the terminal/notebook.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from magicgui import magicgui
from magicgui.widgets import Container, Label, TextEdit

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
    "membrane_upregulation": {
        "folders": [],
        "channels": ["membrane_channel"],
        "label": "Membrane Upregulation",
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
    "membrane_channel": "Membrane Channel",
}


# ---------------------------------------------------------------------------
#  GUI class
# ---------------------------------------------------------------------------

class AcinarAnalysisGUI:
    """Standalone magicgui-based interface for batch acinar image analysis.

    Collects all settings, validates them, then closes the window and
    runs ``batch_analyse`` directly so tqdm progress is visible in the
    terminal / notebook output.
    """

    def __init__(self):
        self._results: Optional[Dict[str, pd.DataFrame]] = None
        self._config: Optional[dict] = None  # filled on Run click
        self._closed = False

        # ---- Build magicgui panels ----

        self.folder_panel = magicgui(
            self._folder_stub,
            image_dir={"label": "Image Folder", "mode": "d"},
            nuclear_mask_dir={"label": "Nuclear Mask Folder", "mode": "d"},
            membrane_mask_dir={"label": "Membrane Mask Folder", "mode": "d"},
            c3_mask_dir={"label": "C3 Mask Folder", "mode": "d"},
            edu_mask_dir={"label": "EdU Mask Folder", "mode": "d"},
            mito_mask_dir={"label": "Mito Mask Folder", "mode": "d"},
            output_dir={"label": "Output Directory", "mode": "d"},
            imaging_record={"label": "Imaging Record (.yml)", "mode": "r",
                            "filter": "*.yml *.yaml"},
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

        self.override_panel = magicgui(
            self._override_stub,
            cell_type={"label": "Cell Type (blank = infer from filename)", "value": ""},
            treatment={"label": "Treatment (blank = infer from filename)", "value": ""},
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
            membrane_upregulation={"label": "Membrane Upregulation", "value": False},
            save_qc_plots={"label": "Save QC Plots", "value": True},
            call_button=False,
        )

        self._btn_run = magicgui(self._on_run_clicked, call_button="Run Analysis")

        # Build welcome / requirements text
        welcome = "Select folders, set channels, tick analyses, click Run.\n"
        welcome += "The window will close and analysis will run with progress in the terminal.\n\n"
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
                Label(value="── Metadata Overrides (optional) ──"),
                self.override_panel,
                Label(value="── Analyses ──"),
                self.analysis_panel,
                Label(value="── Run ──"),
                self._btn_run,
                self._log_widget,
            ],
            label="Acinar Analysis",
        )
        self.widget.native.setWindowTitle("Acinar Analysis")
        self.widget.native.setMinimumWidth(520)

        # Ensure _closed is set if the user closes the window via X button
        _self = self
        _orig_close = self.widget.native.closeEvent
        def _on_native_close(event):
            _self._closed = True
            if _orig_close:
                _orig_close(event)
        self.widget.native.closeEvent = _on_native_close

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
        output_dir: Path = Path(),
        imaging_record: Path = Path(),
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
        membrane_upregulation: bool = False,
        save_qc_plots: bool = True,
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

    @staticmethod
    def _override_stub(
        cell_type: str = "",
        treatment: str = "",
    ):
        return None

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        cur = self._log_widget.value.rstrip()
        self._log_widget.value = (cur + "\n" + msg) if cur else msg

    @staticmethod
    def _dir_or_none(path_value) -> Optional[str]:
        p = Path(str(path_value))
        if str(p) in (".", "") or not p.is_dir():
            return None
        return str(p)

    @staticmethod
    def _file_or_none(path_value) -> Optional[str]:
        p = Path(str(path_value))
        if str(p) in (".", "") or not p.is_file():
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
                "c3_mask_dir", "edu_mask_dir", "mito_mask_dir", "output_dir",
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
        if folders.get("output_dir") is None:
            errors.append("Output Directory is required.")
        for name in analyses:
            reqs = _REQUIREMENTS[name]
            for fkey in reqs["folders"]:
                if folders.get(fkey) is None:
                    errors.append(f"'{reqs['label']}' requires {_FOLDER_LABELS[fkey]}.")
            for ckey in reqs["channels"]:
                if channels.get(ckey) is None:
                    errors.append(f"'{reqs['label']}' requires {_CHANNEL_LABELS[ckey]} (>= 0).")

        # Check that mask directories have the same number of files as images
        ext = str(self.channel_panel.file_extension.value)
        image_dir = folders.get("image_dir")
        if image_dir is not None:
            n_images = len(list(Path(image_dir).rglob(f"*.{ext}")))
            mask_dirs = {
                "nuclear_mask_dir": "Nuclear Mask Folder",
                "membrane_mask_dir": "Membrane Mask Folder",
                "c3_mask_dir": "C3 Mask Folder",
                "edu_mask_dir": "EdU Mask Folder",
                "mito_mask_dir": "Mito Mask Folder",
            }
            for key, label in mask_dirs.items():
                d = folders.get(key)
                if d is not None:
                    n_masks = len(list(Path(d).rglob(f"*.{ext}")))
                    if n_masks != n_images:
                        errors.append(
                            f"{label} has {n_masks} .{ext} file(s) but Image "
                            f"Folder has {n_images}. They must match."
                        )

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
        """Validate settings, store config, then close the window."""
        analyses = self._selected_analyses()
        folders = self._read_folders()
        channels = self._read_channels()

        errors = self._validate(analyses, folders, channels)
        if errors:
            self._log("--- Validation failed ---")
            for e in errors:
                self._log(f"  [ERROR] {e}")
            return

        # Store the validated configuration
        output_dir = folders.pop("output_dir")
        output_csv = str(Path(output_dir) / "acinar_results.csv")
        qc_dir = str(Path(output_dir) / "qc_plots") if self.analysis_panel.save_qc_plots.value else None

        # Read metadata overrides (empty string → None → infer from filename)
        cell_type_override = str(self.override_panel.cell_type.value).strip() or None
        treatment_override = str(self.override_panel.treatment.value).strip() or None

        self._config = {
            "image_dir": folders["image_dir"],
            "analyses": analyses,
            "file_extension": str(self.channel_panel.file_extension.value),
            "n_jobs": int(self.channel_panel.n_jobs.value),
            "output_csv": output_csv,
            "nuclear_mask_dir": folders.get("nuclear_mask_dir"),
            "membrane_mask_dir": folders.get("membrane_mask_dir"),
            "c3_mask_dir": folders.get("c3_mask_dir"),
            "edu_mask_dir": folders.get("edu_mask_dir"),
            "mito_mask_dir": folders.get("mito_mask_dir"),
            "nuclear_channel": channels.get("nuclear_channel", 0),
            "membrane_channel": channels.get("membrane_channel"),
            "protein_channel": channels.get("protein_channel"),
            "c3_channel": channels.get("c3_channel"),
            "edu_channel": channels.get("edu_channel"),
            "mito_channel": channels.get("mito_channel"),
            "proximity_protein_channel": channels.get("proximity_protein_channel"),
            "qc_dir": qc_dir,
            "cell_type_override": cell_type_override,
            "treatment_override": treatment_override,
        }

        self._log("=" * 50)
        self._log(f"[INFO] Analyses: {', '.join(analyses)}")
        self._log(f"[INFO] Images  : {folders['image_dir']}")
        self._log(f"[INFO] Output  : {output_csv}")
        if qc_dir:
            self._log(f"[INFO] QC plots: {qc_dir}")
        self._log("[INFO] Closing GUI and starting analysis...")

        # Close the GUI window
        self._closed = True
        self.widget.close()

    # ------------------------------------------------------------------
    #  Public methods
    # ------------------------------------------------------------------

    @property
    def config(self) -> Optional[dict]:
        """Returns the validated config after the user clicks Run, or None."""
        return self._config

    def run_analysis(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Run batch_analyse using the collected config. Call after GUI closes."""
        if self._config is None:
            print("[ERROR] No configuration set. Did the user click Run Analysis?")
            return None

        print("=" * 60)
        print(f"Running analyses: {', '.join(self._config['analyses'])}")
        print(f"Image folder:     {self._config['image_dir']}")
        print(f"Output CSV:       {self._config['output_csv']}")
        if self._config["qc_dir"]:
            print(f"QC plots:         {self._config['qc_dir']}")
        print("=" * 60)

        self._results = batch_analyse(**self._config)

        print("\n" + "=" * 60)
        print("COMPLETE!")
        for name, df in self._results.items():
            n = len(df) if df is not None else 0
            print(f"  {name}: {n} row(s)")
        print("=" * 60)

        return self._results


# ---------------------------------------------------------------------------
#  Public launch helper
# ---------------------------------------------------------------------------

def launch() -> AcinarAnalysisGUI:
    """Create and show the Acinar Analysis GUI.

    Returns the app instance.  After the user clicks 'Run Analysis',
    the window closes and you can call ``app.run_analysis()`` to execute
    the batch processing.

    Example (notebook)::

        %gui qt
        from acinar_gui import launch
        app = launch()
        # ... user configures and clicks Run ...
        # Then in the next cell:
        results = app.run_analysis()
    """
    return AcinarAnalysisGUI()


def launch_and_run():
    """Launch the GUI, wait for the user to click Run, then execute the analysis.

    This is the simplest way to use the GUI — one call does everything.
    Intended for use from a notebook cell::

        %gui qt
        from acinar_gui import launch_and_run
        results = launch_and_run()
    """
    import time
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    gui = AcinarAnalysisGUI()

    # Process events until the GUI is closed
    while not gui._closed:
        app.processEvents()
        time.sleep(0.05)  # prevent CPU spinning / re-entrancy

    if gui.config is None:
        print("[INFO] GUI closed without running analysis.")
        return None

    return gui.run_analysis()


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication
    _qapp = QApplication.instance() or QApplication([])
    results = launch_and_run()
    if results is None:
        sys.exit(0)

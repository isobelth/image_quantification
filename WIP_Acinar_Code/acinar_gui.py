"""
Acinar Analysis GUI
===================
Napari-based GUI for batch 3-D acinar image analysis.

Launch from a notebook::

    from acinar_gui import launch
    launch()

Or standalone::

    python acinar_gui.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import napari
import pandas as pd
from magicgui import magicgui
from magicgui.widgets import Container, TextEdit
from napari.qt.threading import create_worker

from acinar_analysis import VALID_ANALYSES, batch_analyse

# ---------------------------------------------------------------------------
#  Per-analysis requirements
# ---------------------------------------------------------------------------

_REQUIREMENTS: Dict[str, dict] = {
    "acinus_shape": {
        "folders": [],
        "channels": [],
        "label": "Acinus Shape",
        "description": "Volume & roundness of the acinus",
    },
    "cell_nuclear_shape": {
        "folders": ["nuclear_mask_dir", "membrane_mask_dir"],
        "channels": [],
        "label": "Cell & Nuclear Shape",
        "description": "Per-cell volume, roundness, neighbour analysis",
    },
    "protein_polarisation": {
        "folders": [],
        "channels": ["protein_channel"],
        "label": "Protein Polarisation",
        "description": "Protein intensity vs radial distance",
    },
    "apoptosis": {
        "folders": ["c3_mask_dir", "dapi_mask_dir"],
        "channels": ["c3_channel"],
        "label": "Apoptosis (C3)",
        "description": "C3-positive apoptotic cell counting",
    },
    "protein_proximity": {
        "folders": ["c3_mask_dir", "dapi_mask_dir"],
        "channels": ["c3_channel", "proximity_protein_channel"],
        "label": "Protein Proximity",
        "description": "Protein intensity near dying vs non-dying cells",
    },
    "proliferation": {
        "folders": ["edu_mask_dir", "dapi_mask_dir"],
        "channels": ["edu_channel"],
        "label": "Proliferation (EdU)",
        "description": "EdU-positive dividing vs non-dividing cells",
    },
    "mitochondria": {
        "folders": ["nuclear_mask_dir", "membrane_mask_dir", "mito_mask_dir"],
        "channels": [],
        "label": "Mitochondria",
        "description": "Per-cell mito count, volume, distance from nucleus",
    },
}

_FOLDER_LABELS = {
    "nuclear_mask_dir": "Nuclear Mask Folder",
    "membrane_mask_dir": "Membrane Mask Folder",
    "c3_mask_dir": "C3 Mask Folder",
    "dapi_mask_dir": "DAPI Mask Folder",
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
    """Napari-based graphical interface for batch acinar image analysis."""

    def __init__(self):
        self.viewer = napari.Viewer(title="Acinar Analysis")
        self._running = False
        self._results: List[pd.DataFrame] = []

        # ---- Log panel ----
        welcome = (
            "=== Acinar Analysis GUI ===\n"
            "1) Set input folders (Image Folder is always required).\n"
            "2) Configure channel indices (-1 = not used).\n"
            "3) Tick the analyses you want, then click Run.\n"
            "\nRequirements per analysis:\n"
        )
        for info in _REQUIREMENTS.values():
            reqs = []
            for f in info["folders"]:
                reqs.append(_FOLDER_LABELS[f])
            for c in info["channels"]:
                reqs.append(f"{_CHANNEL_LABELS[c]} >= 0")
            req_str = ", ".join(reqs) if reqs else "(Image Folder only)"
            welcome += f"  {info['label']:25s} {req_str}\n"

        self.log_output = TextEdit(value=welcome)
        self.log_output.min_height = 200
        self.log_output.max_height = 500
        try:
            self.log_output.native.setReadOnly(True)
        except Exception:
            pass

        # ---- Folder selectors (no submit button) ----
        self.folder_panel = magicgui(
            self._folder_stub,
            image_dir={"label": "Image Folder (required)", "mode": "d"},
            nuclear_mask_dir={"label": "Nuclear Mask Folder", "mode": "d"},
            membrane_mask_dir={"label": "Membrane Mask Folder", "mode": "d"},
            c3_mask_dir={"label": "C3 Mask Folder", "mode": "d"},
            dapi_mask_dir={"label": "DAPI Mask Folder", "mode": "d"},
            edu_mask_dir={"label": "EdU Mask Folder", "mode": "d"},
            mito_mask_dir={"label": "Mito Mask Folder", "mode": "d"},
            call_button=False,
        )

        # ---- Channel config (no submit button) ----
        self.channel_panel = magicgui(
            self._channel_stub,
            dapi_channel={
                "label": "DAPI Channel",
                "value": 0, "min": 0, "max": 20,
            },
            membrane_channel={
                "label": "Membrane Channel (-1 = none)",
                "value": 2, "min": -1, "max": 20,
            },
            protein_channel={
                "label": "Protein Channel (-1 = none)",
                "value": -1, "min": -1, "max": 20,
            },
            c3_channel={
                "label": "C3 Channel (-1 = none)",
                "value": 3, "min": -1, "max": 20,
            },
            edu_channel={
                "label": "EdU Channel (-1 = none)",
                "value": -1, "min": -1, "max": 20,
            },
            mito_channel={
                "label": "Mito Channel (-1 = none)",
                "value": -1, "min": -1, "max": 20,
            },
            proximity_protein_channel={
                "label": "Prox. Protein Ch. (-1 = none)",
                "value": -1, "min": -1, "max": 20,
            },
            file_extension={
                "label": "File Extension",
                "value": "tif",
            },
            n_jobs={
                "label": "Parallel Jobs",
                "value": 3, "min": 1, "max": 32,
            },
            call_button=False,
        )

        # ---- Analysis checkboxes + Run button ----
        self.run_panel = magicgui(
            self._on_run_clicked,
            acinus_shape={"label": "Acinus Shape", "value": False},
            cell_nuclear_shape={
                "label": "Cell & Nuclear Shape", "value": False,
            },
            protein_polarisation={
                "label": "Protein Polarisation", "value": False,
            },
            apoptosis={"label": "Apoptosis (C3)", "value": False},
            protein_proximity={
                "label": "Protein Proximity", "value": False,
            },
            proliferation={
                "label": "Proliferation (EdU)", "value": False,
            },
            mitochondria={"label": "Mitochondria", "value": False},
            output_csv={
                "label": "Output CSV",
                "mode": "w",
                "value": "acinar_results.csv",
            },
            call_button="Run Selected Analyses",
        )

        # ---- Dock everything ----
        self.viewer.window.add_dock_widget(
            self.folder_panel, name="Input Folders", area="right",
        )
        self.viewer.window.add_dock_widget(
            self.channel_panel, name="Channel Config", area="right",
        )
        self.viewer.window.add_dock_widget(
            self.run_panel, name="Analyses", area="right",
        )
        self.viewer.window.add_dock_widget(
            Container(widgets=[self.log_output]), name="Log", area="right",
        )

        # Show image count when the user selects the image folder
        self.folder_panel.image_dir.changed.connect(self._on_image_dir_changed)

    # ------------------------------------------------------------------
    #  Stub functions for the panels without a submit button
    # ------------------------------------------------------------------

    @staticmethod
    def _folder_stub(
        image_dir: Path = Path(),
        nuclear_mask_dir: Path = Path(),
        membrane_mask_dir: Path = Path(),
        c3_mask_dir: Path = Path(),
        dapi_mask_dir: Path = Path(),
        edu_mask_dir: Path = Path(),
        mito_mask_dir: Path = Path(),
    ):
        return None

    @staticmethod
    def _channel_stub(
        dapi_channel: int = 0,
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
    #  Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        cur = self.log_output.value.rstrip()
        self.log_output.value = (cur + "\n" + msg) if cur else msg

    @staticmethod
    def _dir_or_none(path_value) -> Optional[str]:
        """Return the path string if the user selected a real directory, else None."""
        p = Path(str(path_value))
        if str(p) in (".", "") or not p.is_dir():
            return None
        return str(p)

    @staticmethod
    def _channel_or_none(value: int) -> Optional[int]:
        """Convert -1 → None (meaning 'not set')."""
        return value if value >= 0 else None

    def _read_folders(self) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        for key in (
            "image_dir", "nuclear_mask_dir", "membrane_mask_dir",
            "c3_mask_dir", "dapi_mask_dir", "edu_mask_dir", "mito_mask_dir",
        ):
            out[key] = self._dir_or_none(getattr(self.folder_panel, key).value)
        return out

    def _read_channels(self) -> Dict[str, Optional[int]]:
        out: Dict[str, Optional[int]] = {}
        for key in (
            "dapi_channel", "membrane_channel", "protein_channel",
            "c3_channel", "edu_channel", "mito_channel",
            "proximity_protein_channel",
        ):
            out[key] = self._channel_or_none(
                int(getattr(self.channel_panel, key).value)
            )
        return out

    def _selected_analyses(self) -> List[str]:
        return [
            name for name in _REQUIREMENTS
            if getattr(self.run_panel, name).value
        ]

    def _validate(
        self,
        analyses: List[str],
        folders: Dict[str, Optional[str]],
        channels: Dict[str, Optional[int]],
    ) -> List[str]:
        """Return a list of validation error strings (empty → valid)."""
        errors: List[str] = []

        if not analyses:
            errors.append("No analyses selected.")
            return errors

        if folders.get("image_dir") is None:
            errors.append("Image Folder is required for all analyses.")

        for name in analyses:
            reqs = _REQUIREMENTS[name]
            for fkey in reqs["folders"]:
                if folders.get(fkey) is None:
                    errors.append(
                        f"'{reqs['label']}' requires {_FOLDER_LABELS[fkey]}."
                    )
            for ckey in reqs["channels"]:
                if channels.get(ckey) is None:
                    errors.append(
                        f"'{reqs['label']}' requires "
                        f"{_CHANNEL_LABELS[ckey]} (set to >= 0)."
                    )
        return errors

    # ------------------------------------------------------------------
    #  Folder-changed callback
    # ------------------------------------------------------------------

    def _on_image_dir_changed(self, value):
        d = self._dir_or_none(value)
        if d is None:
            return
        ext = str(self.channel_panel.file_extension.value)
        n = len(list(Path(d).rglob(f"*.{ext}")))
        self._log(f"[INFO] Image folder: found {n} .{ext} file(s) in {d}")

    # ------------------------------------------------------------------
    #  Run handler
    # ------------------------------------------------------------------

    def _on_run_clicked(
        self,
        acinus_shape: bool = False,
        cell_nuclear_shape: bool = False,
        protein_polarisation: bool = False,
        apoptosis: bool = False,
        protein_proximity: bool = False,
        proliferation: bool = False,
        mitochondria: bool = False,
        output_csv: Path = Path("acinar_results.csv"),
    ):
        if self._running:
            self._log("[WARN] Analysis is already running. Please wait.")
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

        out_path = str(output_csv)
        if not out_path.lower().endswith(".csv"):
            out_path += ".csv"

        self._log("=" * 50)
        self._log(f"[INFO] Running: {', '.join(analyses)}")
        self._log(f"[INFO] Images : {folders['image_dir']}")
        self._log(f"[INFO] Output : {out_path}")
        self._running = True

        worker = create_worker(
            batch_analyse,
            image_dir=folders["image_dir"],
            analyses=analyses,
            file_extension=str(self.channel_panel.file_extension.value),
            n_jobs=int(self.channel_panel.n_jobs.value),
            output_csv=out_path,
            nuclear_mask_dir=folders.get("nuclear_mask_dir"),
            membrane_mask_dir=folders.get("membrane_mask_dir"),
            c3_mask_dir=folders.get("c3_mask_dir"),
            dapi_mask_dir=folders.get("dapi_mask_dir"),
            edu_mask_dir=folders.get("edu_mask_dir"),
            mito_mask_dir=folders.get("mito_mask_dir"),
            dapi_channel=channels.get("dapi_channel", 0),
            membrane_channel=channels.get("membrane_channel"),
            protein_channel=channels.get("protein_channel"),
            c3_channel=channels.get("c3_channel"),
            edu_channel=channels.get("edu_channel"),
            mito_channel=channels.get("mito_channel"),
            proximity_protein_channel=channels.get("proximity_protein_channel"),
        )
        worker.returned.connect(
            lambda results: self._on_success(results, out_path)
        )
        worker.errored.connect(self._on_error)
        worker.finished.connect(self._on_finished)
        worker.start()

    # ------------------------------------------------------------------
    #  Worker callbacks (run on the main / Qt thread)
    # ------------------------------------------------------------------

    def _on_success(self, results: Dict[str, pd.DataFrame], output_path: str):
        for name, df in results.items():
            n = len(df) if df is not None else 0
            self._log(f"  [OK] {name}: {n} row(s)")
            if df is not None and not df.empty:
                self._results.append(df)
        self._log(f"[OK] Complete. CSVs saved with prefix: {output_path}")

    def _on_error(self, exc: Exception):
        self._log(f"[ERROR] {type(exc).__name__}: {exc}")

    def _on_finished(self):
        self._running = False
        self._log("[INFO] Ready for next run.\n")


# ---------------------------------------------------------------------------
#  Public launch helper
# ---------------------------------------------------------------------------

def launch() -> AcinarAnalysisGUI:
    """Create and show the Acinar Analysis GUI. Returns the app instance."""
    return AcinarAnalysisGUI()


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


if __name__ == "__main__":
    app = launch()
    if not _running_in_notebook():
        napari.run()

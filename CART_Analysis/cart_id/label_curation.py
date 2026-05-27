"""Stage 3 — Interactive label curation (optional, napari-based).

Opens a napari viewer for each image in the manifest.  Two masks are
editable by the user:

    * Chip center (channel 4)
    * Tumour     (channel 5)

The remaining 9 channels are loaded as read-only reference layers.

The viewer prefers ``stage3/<name>.tif`` (a previous curation session) and
falls back to ``stage2/<name>.tif``.  Every time the user navigates to a
new image, if the current image was edited, its stage3 TIF is written
immediately (dirty-flag incremental save).  Only genuinely edited images
produce stage3 TIFs; Stage 4 falls back to stage2 for the rest.

Usage:
    python -m cart_id.label_curation <manifest_path>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .io_utils import (
    N_CHANNELS,
    read_curation_tif,
    write_curation_tif,
)

STAGE2_SUBDIR = "stage2"
STAGE3_SUBDIR = "stage3"

# Indices inside the 11-channel TIF
_CH_BF = 0
_CH_BCELL_RAW = 1
_CH_VASC_RAW = 2
_CH_CART_RAW = 3
_CH_CHIP = 4       # editable
_CH_TUMOUR = 5     # editable
_CH_LEFT = 6
_CH_RIGHT = 7
_CH_VASC_MASK = 8
_CH_BCELL_MASK = 9
_CH_CART_MASK = 10

# Keep active CurationApp instances alive during a Jupyter session.
_ACTIVE_APPS: list["CurationApp"] = []


@dataclass
class CurationJob:
    job_id: str
    expected_output_name: str
    source_file: str
    image_index: int
    image_name: str
    # Raw channels (display only)
    bf_raw: np.ndarray          # (Y, X) uint8
    bcell_raw: np.ndarray       # (Y, X) uint8
    vasc_raw: np.ndarray        # (Y, X) uint8
    cart_raw: np.ndarray        # (Y, X) uint8
    # Editable masks
    chip_mask: np.ndarray       # (Y, X) bool
    tumour_mask: np.ndarray     # (Y, X) bool
    # Auto-only masks (preserved unchanged on write)
    left_mask: np.ndarray       # (Y, X) bool
    right_mask: np.ndarray      # (Y, X) bool
    vasc_mask: np.ndarray       # (Y, X) bool
    bcell_mask: np.ndarray      # (Y, X) bool
    cart_mask: np.ndarray       # (Y, X) bool
    pixel_size_um: Tuple[float, float]  # (y_um, x_um)
    loaded_from: str = "stage2"
    dirty: bool = False


def _load_curation_jobs(
    manifest_df: pd.DataFrame,
    output_root: Path,
) -> List[CurationJob]:
    jobs: List[CurationJob] = []
    stage2_dir = output_root / STAGE2_SUBDIR
    stage3_dir = output_root / STAGE3_SUBDIR
    n_resumed = 0

    for row in manifest_df.itertuples(index=False):
        name = str(row.expected_output_name)
        stage3_tif = stage3_dir / f"{name}.tif"
        stage2_tif = stage2_dir / f"{name}.tif"
        if stage3_tif.exists():
            tif_path = stage3_tif
            loaded_from = "stage3"
            n_resumed += 1
        elif stage2_tif.exists():
            tif_path = stage2_tif
            loaded_from = "stage2"
        else:
            print(f"[SKIP] {name}: no stage2 or stage3 TIF found.", file=sys.stderr)
            continue

        try:
            stack, pixel_size = read_curation_tif(tif_path)
        except Exception as exc:
            print(f"[SKIP] {name}: {exc}", file=sys.stderr)
            continue

        pixel_size_um = pixel_size if pixel_size is not None else (float("nan"), float("nan"))

        jobs.append(CurationJob(
            job_id=str(row.job_id),
            expected_output_name=name,
            source_file=str(getattr(row, "source_file", "") or ""),
            image_index=int(getattr(row, "image_index", 0) or 0),
            image_name=str(getattr(row, "image_name", "") or ""),
            bf_raw=stack[_CH_BF],
            bcell_raw=stack[_CH_BCELL_RAW],
            vasc_raw=stack[_CH_VASC_RAW],
            cart_raw=stack[_CH_CART_RAW],
            chip_mask=stack[_CH_CHIP].astype(bool),
            tumour_mask=stack[_CH_TUMOUR].astype(bool),
            left_mask=stack[_CH_LEFT].astype(bool),
            right_mask=stack[_CH_RIGHT].astype(bool),
            vasc_mask=stack[_CH_VASC_MASK].astype(bool),
            bcell_mask=stack[_CH_BCELL_MASK].astype(bool),
            cart_mask=stack[_CH_CART_MASK].astype(bool),
            pixel_size_um=pixel_size_um,
            loaded_from=loaded_from,
        ))

    if n_resumed:
        print(f"[INFO] Resumed {n_resumed} job(s) from existing stage3/ outputs.")
    return jobs


class CurationApp:
    """Napari curation GUI for CART analysis (chip + tumour masks editable)."""

    def __init__(self, jobs: List[CurationJob], output_root: Path):
        if not jobs:
            raise ValueError("No jobs to curate.")
        self.jobs = jobs
        self.output_root = output_root
        self.current_index = 0
        self.is_syncing = False
        self._suppress_dirty = False

    # ------------------------------------------------------------------
    def open(self) -> None:
        import napari
        from magicgui.widgets import ComboBox, Container, Label, PushButton

        first = self.jobs[0]
        ph = np.zeros(first.bf_raw.shape, dtype=np.uint8)

        self.viewer = napari.Viewer(title="CART Label Curation (Stage 3)")

        # --- Image layers (display only) ---
        self.bf_layer = self.viewer.add_image(ph, name="BF", colormap="gray")

        # --- Editable label layers ---
        self.chip_label_layer = self.viewer.add_labels(
            np.zeros_like(ph, dtype=np.int32), name="Chip center [edit]")
        self.tumour_label_layer = self.viewer.add_labels(
            np.zeros_like(ph, dtype=np.int32), name="Tumour [edit]")
        self.chip_label_layer.opacity = 0.4
        self.tumour_label_layer.opacity = 0.4

        # Dirty tracking — only on the two editable label layers
        def _mark_dirty(*_args):
            if self._suppress_dirty:
                return
            if 0 <= self.current_index < len(self.jobs):
                self.jobs[self.current_index].dirty = True

        for lyr in (self.chip_label_layer, self.tumour_label_layer):
            for ev_name in ("paint", "set_data"):
                sig = getattr(lyr.events, ev_name, None)
                if sig is not None:
                    try:
                        sig.connect(_mark_dirty)
                    except Exception:
                        pass

        # --- Control panel ---
        job_choices = [f"{i}: {j.expected_output_name}" for i, j in enumerate(self.jobs)]
        self.job_dropdown = ComboBox(label="Image", choices=job_choices)
        self.job_dropdown.changed.connect(self._on_job_dropdown_changed)
        self.status_label = Label(value="")
        self.btn_prev = PushButton(text="◀ Previous")
        self.btn_prev.clicked.connect(self._go_prev)
        self.btn_next = PushButton(text="Next ▶")
        self.btn_next.clicked.connect(self._go_next)
        self.btn_done = PushButton(text="Done — Write Overrides")
        self.btn_done.clicked.connect(self._finish)
        self.progress_label = Label(value="")

        panel = Container(widgets=[
            self.job_dropdown, self.status_label,
            self.btn_prev, self.btn_next, self.btn_done,
            self.progress_label,
        ])
        self.viewer.window.add_dock_widget(panel, area="right", name="Curation")

        self._show_index(0)

        _ACTIVE_APPS.append(self)

        def _drop_ref(*_args):
            if self in _ACTIVE_APPS:
                _ACTIVE_APPS.remove(self)

        for owner, attr in (
            (getattr(self.viewer.window, "qt_viewer", None), "destroyed"),
            (getattr(self.viewer.window, "_qt_window", None), "destroyed"),
        ):
            sig = getattr(owner, attr, None) if owner is not None else None
            if sig is not None:
                try:
                    sig.connect(_drop_ref)
                    break
                except Exception:
                    continue

        napari.run()

    # ------------------------------------------------------------------
    def _write_job_override(self, job: CurationJob) -> bool:
        stage3_dir = self.output_root / STAGE3_SUBDIR
        stage3_dir.mkdir(parents=True, exist_ok=True)
        try:
            stack_11ch = np.stack([
                job.bf_raw,
                job.bcell_raw,
                job.vasc_raw,
                job.cart_raw,
                job.chip_mask.astype(np.uint8),
                job.tumour_mask.astype(np.uint8),
                job.left_mask.astype(np.uint8),
                job.right_mask.astype(np.uint8),
                job.vasc_mask.astype(np.uint8),
                job.bcell_mask.astype(np.uint8),
                job.cart_mask.astype(np.uint8),
            ], axis=0)
            write_curation_tif(
                stage3_dir / f"{job.expected_output_name}.tif",
                stack_11ch,
                job.pixel_size_um,
            )
            return True
        except Exception as exc:
            print(f"[FAIL] {job.expected_output_name}: {exc}", file=sys.stderr)
            return False

    def _save_current_state(self) -> None:
        job = self.jobs[self.current_index]
        chip_arr = np.asarray(self.chip_label_layer.data)
        if chip_arr.shape == job.chip_mask.shape:
            job.chip_mask = chip_arr > 0
        tumour_arr = np.asarray(self.tumour_label_layer.data)
        if tumour_arr.shape == job.tumour_mask.shape:
            job.tumour_mask = tumour_arr > 0
        if job.dirty:
            if self._write_job_override(job):
                job.dirty = False
                job.loaded_from = "stage3"
                print(f"  saved stage3/{job.expected_output_name}.tif")

    def _show_index(self, index: int) -> None:
        if index < 0 or index >= len(self.jobs):
            return
        self.current_index = index
        job = self.jobs[index]
        self._suppress_dirty = True

        # Raw image layers
        self.bf_layer.data = job.bf_raw
        bf_min, bf_max = float(job.bf_raw.min()), float(job.bf_raw.max())
        if bf_max > bf_min:
            self.bf_layer.contrast_limits = (bf_min, bf_max)

        # Editable label layers
        chip_labels = np.zeros(job.bf_raw.shape, dtype=np.int32)
        chip_labels[job.chip_mask] = 1
        self.chip_label_layer.data = chip_labels

        tumour_labels = np.zeros(job.bf_raw.shape, dtype=np.int32)
        tumour_labels[job.tumour_mask] = 1
        self.tumour_label_layer.data = tumour_labels

        self.viewer.reset_view()
        job.dirty = False
        self._suppress_dirty = False

        self.is_syncing = True
        try:
            self.job_dropdown.value = f"{index}: {job.expected_output_name}"
        finally:
            self.is_syncing = False
        self._update_status_labels()

    def _update_status_labels(self) -> None:
        job = self.jobs[self.current_index]
        origin = f" - {job.loaded_from}"
        src = Path(job.source_file).name if job.source_file else ""
        src_info = f"\n{src}  [img {job.image_index}]" if src else ""
        self.status_label.value = (
            f"Image {self.current_index + 1}/{len(self.jobs)}{origin}{src_info}"
        )
        n_edited = sum(1 for j in self.jobs if j.loaded_from == "stage3" or j.dirty)
        self.progress_label.value = f"{n_edited}/{len(self.jobs)} have stage3 overrides"

    def _on_job_dropdown_changed(self, value) -> None:
        if self.is_syncing:
            return
        try:
            new_index = int(str(value).split(":")[0])
        except (ValueError, IndexError):
            return
        if new_index == self.current_index or not (0 <= new_index < len(self.jobs)):
            return
        self._save_current_state()
        self._show_index(new_index)

    def _go_prev(self, *_args) -> None:
        if self.current_index > 0:
            self._save_current_state()
            self._show_index(self.current_index - 1)

    def _go_next(self, *_args) -> None:
        if self.current_index < len(self.jobs) - 1:
            self._save_current_state()
            self._show_index(self.current_index + 1)

    def _finish(self, *_args) -> None:
        self._save_current_state()
        n_overrides = sum(1 for j in self.jobs if j.loaded_from == "stage3")
        self.status_label.value = f"Done. {n_overrides} stage3 override(s) on disk."
        print(f"Stage 3 complete: {n_overrides}/{len(self.jobs)} jobs have stage3 overrides.")
        try:
            self.viewer.close()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 - interactive CART label curation.")
    parser.add_argument("manifest_path")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest_path)
    output_root = manifest_path.parent
    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df["should_process"]]
    jobs = _load_curation_jobs(manifest_df, output_root)
    if not jobs:
        print("[INFO] No jobs with stage2/<name>.tif found. Run Stage 2 first.")
        return 0
    print(f"Opening curation viewer for {len(jobs)} job(s)...")
    CurationApp(jobs, output_root=output_root).open()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

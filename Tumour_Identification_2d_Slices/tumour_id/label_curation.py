"""Stage 3 - Interactive label curation (optional, napari).

For each manifest row with <output_root>/stage2/<name>.tif on disk:
- Open a napari viewer with the BF raw image + BF label layer; if the job has
  fluorescence, also add FL image + FL label layer.
- Let the user edit the label layers, navigate with Prev / Next, and Done.
- On Done, write <output_root>/stage3/<name>.tif with the same 4-channel
  layout as Stage 2 (raw channels + edited labels) and the pixel size
  re-embedded in the TIF resolution tags.

<output_root> is the directory containing the manifest CSV.

Usage:
    python -m tumour_id.label_curation <manifest_path>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

from .io_utils import get_tiff_pixel_size_um

STAGE2_SUBDIR = "stage2"
STAGE3_SUBDIR = "stage3"

# Keep references to active CurationApp instances so that, when launched
# from Jupyter (where ``napari.run()`` returns immediately because a Qt
# event loop is already running), the app — and therefore the Qt signal
# connections wired to its bound methods — are not garbage-collected.
_ACTIVE_APPS: list["CurationApp"] = []


@dataclass
class CurationJob:
    job_id: str
    expected_output_name: str
    bf_raw: np.ndarray              # 2D uint8
    fl_raw: Optional[np.ndarray]    # 2D uint8 or None
    bf_mask: np.ndarray             # 2D bool
    fl_mask: Optional[np.ndarray]   # 2D bool or None
    has_fluorescence: bool
    pixel_size_um: Tuple[float, float]   # (y_um, x_um)
    source_file: str = ""           # for status label / debugging
    image_index: int = 0
    # ``loaded_from`` records which folder the job was read from ("stage2"
    # or "stage3"). ``dirty`` flips to True only when the user actually
    # paints/erases on a label layer for this job - that controls whether
    # we re-write the stage3 TIF on navigation.
    loaded_from: str = "stage2"
    dirty: bool = False


def _load_curation_jobs(manifest_df: pd.DataFrame, output_root: Path) -> List[CurationJob]:
    jobs: List[CurationJob] = []
    stage2_dir = output_root / STAGE2_SUBDIR
    stage3_dir = output_root / STAGE3_SUBDIR
    n_resumed = 0
    for row in manifest_df.itertuples(index=False):
        name = str(row.expected_output_name)
        # Prefer a previously curated stage3 TIF (so partial work persists
        # across sessions); fall back to the auto stage2 TIF.
        stage3_tif = stage3_dir / f"{name}.tif"
        stage2_tif = stage2_dir / f"{name}.tif"
        if stage3_tif.exists():
            curation_tif = stage3_tif
            loaded_from = "stage3"
            n_resumed += 1
        elif stage2_tif.exists():
            curation_tif = stage2_tif
            loaded_from = "stage2"
        else:
            print(f"[SKIP] {name}: no stage2/{name}.tif or stage3/{name}.tif", file=sys.stderr)
            continue

        with tifffile.TiffFile(curation_tif) as tif:
            stack = np.asarray(tif.asarray())
            pixel_size_um = get_tiff_pixel_size_um(tif) or (float("nan"), float("nan"))

        if stack.ndim != 3 or stack.shape[0] != 4:
            print(f"[SKIP] {name}: unexpected curation TIF shape {stack.shape}", file=sys.stderr)
            continue

        bf_raw = stack[0].astype(np.uint8)
        fl_raw_channel = stack[1].astype(np.uint8)
        bf_mask = stack[2].astype(bool)
        fl_mask_channel = stack[3].astype(bool)

        has_fluorescence = bool(fl_raw_channel.any())
        fl_raw = fl_raw_channel if has_fluorescence else None
        fl_mask = fl_mask_channel if has_fluorescence else None

        jobs.append(CurationJob(
            job_id=str(row.job_id),
            expected_output_name=name,
            bf_raw=bf_raw,
            fl_raw=fl_raw,
            bf_mask=bf_mask,
            fl_mask=fl_mask,
            has_fluorescence=has_fluorescence,
            pixel_size_um=pixel_size_um,
            source_file=str(getattr(row, "source_file", "") or ""),
            image_index=int(getattr(row, "image_index", 0) or 0),
            loaded_from=loaded_from,
        ))
    if n_resumed:
        print(f"[INFO] Resumed {n_resumed} job(s) from existing stage3/ outputs.")
    return jobs


class CurationApp:
    """Napari multi-image curation GUI for tumour labels."""

    def __init__(self, jobs: List[CurationJob], output_root: Path):
        if not jobs:
            raise ValueError("No jobs to curate.")
        self.jobs = jobs
        self.output_root = output_root
        self.current_index = 0
        self.is_syncing = False

    def open(self) -> None:
        import napari
        from magicgui.widgets import ComboBox, Container, Label, PushButton

        first = self.jobs[0]
        placeholder = np.zeros_like(first.bf_raw)

        self.viewer = napari.Viewer(title="Tumour Label Curation (Stage 3)")
        self.bf_image_layer = self.viewer.add_image(placeholder, name="BF", colormap="gray")
        self.fl_image_layer = self.viewer.add_image(placeholder, name="FL", colormap="green", visible=False)
        self.bf_label_layer = self.viewer.add_labels(np.zeros_like(placeholder, dtype=np.int32), name="BF tumour")
        self.fl_label_layer = self.viewer.add_labels(np.zeros_like(placeholder, dtype=np.int32), name="FL tumour")
        self.bf_label_layer.opacity = 0.5
        self.fl_label_layer.opacity = 0.5

        # Mark the active job dirty whenever the user paints/erases on either
        # label layer, so we only write a stage3 override for genuinely
        # edited images. ``_suppress_dirty`` is flipped to True while
        # ``_show_index`` swaps layer data programmatically so loading a job
        # is not mistaken for an edit.
        self._suppress_dirty = False

        def _mark_dirty(*_args):
            if self._suppress_dirty:
                return
            if 0 <= self.current_index < len(self.jobs):
                self.jobs[self.current_index].dirty = True

        for label_layer in (self.bf_label_layer, self.fl_label_layer):
            # ``paint`` fires on every brush stroke; ``set_data`` covers
            # programmatic edits (e.g. fill bucket). Connect both when
            # available, silently skipping unsupported napari versions.
            for event_name in ("paint", "set_data"):
                signal = getattr(label_layer.events, event_name, None)
                if signal is not None:
                    try:
                        signal.connect(_mark_dirty)
                    except Exception:
                        pass

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
            self.btn_prev, self.btn_next, self.btn_done, self.progress_label,
        ])
        self.viewer.window.add_dock_widget(panel, area="right", name="Curation")

        # Keyboard shortcuts: n = next, b = back, Ctrl+S = save current edits
        @self.viewer.bind_key("n", overwrite=True)
        def _key_next(_viewer):
            self._go_next()

        @self.viewer.bind_key("b", overwrite=True)
        def _key_prev(_viewer):
            self._go_prev()

        self._show_index(0)

        # Keep this instance alive for the lifetime of the viewer so that
        # the Qt signal connections to bound methods keep working when
        # ``open()`` returns (e.g. when launched from a Jupyter cell).
        _ACTIVE_APPS.append(self)

        def _drop_ref(*_args):
            if self in _ACTIVE_APPS:
                _ACTIVE_APPS.remove(self)

        # ``napari.Viewer`` exposes a ``closed`` Qt signal on its underlying
        # window in some versions; fall back silently if it is unavailable.
        for owner, name in (
            (getattr(self.viewer.window, "qt_viewer", None), "destroyed"),
            (getattr(self.viewer.window, "_qt_window", None), "destroyed"),
        ):
            signal = getattr(owner, name, None) if owner is not None else None
            if signal is not None:
                try:
                    signal.connect(_drop_ref)
                    break
                except Exception:
                    continue

        # ``napari.run()`` blocks when launched as a script and returns
        # immediately under Jupyter (where a Qt loop is already running),
        # so it is safe to call in both contexts.
        napari.run()

    # ------------------------------------------------------------------
    def _write_job_override(self, job: CurationJob) -> bool:
        """Persist a single job to ``<output_root>/stage3/<name>.tif``."""
        stage3_dir = self.output_root / STAGE3_SUBDIR
        stage3_dir.mkdir(parents=True, exist_ok=True)
        try:
            zeros = np.zeros(job.bf_raw.shape, dtype=np.uint8)
            stack_cyx = np.stack([
                job.bf_raw.astype(np.uint8),
                job.fl_raw.astype(np.uint8) if (job.has_fluorescence and job.fl_raw is not None) else zeros,
                job.bf_mask.astype(np.uint8),
                job.fl_mask.astype(np.uint8) if (job.has_fluorescence and job.fl_mask is not None) else zeros,
            ], axis=0)

            pixel_size_y, pixel_size_x = job.pixel_size_um
            write_kwargs: dict = dict(imagej=True, metadata={"axes": "CYX", "unit": "um"})
            if np.isfinite(pixel_size_x) and np.isfinite(pixel_size_y) and pixel_size_x > 0 and pixel_size_y > 0:
                write_kwargs["resolution"] = (1.0 / pixel_size_x, 1.0 / pixel_size_y)
            tifffile.imwrite(stage3_dir / f"{job.expected_output_name}.tif", stack_cyx, **write_kwargs)
            return True
        except Exception as exc:
            print(f"[FAIL] {job.expected_output_name}: {exc}", file=sys.stderr)
            return False

    def _save_current_state(self) -> None:
        """Copy the visible label layers back into the active job and, if it was
        edited, write a stage3 override immediately."""
        job = self.jobs[self.current_index]
        bf_label_arr = np.asarray(self.bf_label_layer.data)
        if bf_label_arr.shape == job.bf_mask.shape:
            job.bf_mask = bf_label_arr > 0
        if job.has_fluorescence:
            fl_label_arr = np.asarray(self.fl_label_layer.data)
            if job.fl_mask is not None and fl_label_arr.shape == job.fl_mask.shape:
                job.fl_mask = fl_label_arr > 0
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

        # Programmatic data swaps below trigger label-layer ``set_data``
        # events; suppress the dirty flag so loading a job is not mistaken
        # for a user edit.
        self._suppress_dirty = True

        # --- BF layers (always rebuilt from the current job) ---
        self.bf_image_layer.data = job.bf_raw
        bf_min, bf_max = float(job.bf_raw.min()), float(job.bf_raw.max())
        if bf_max > bf_min:
            self.bf_image_layer.contrast_limits = (bf_min, bf_max)

        bf_labels = np.zeros(job.bf_raw.shape, dtype=np.int32)
        bf_labels[job.bf_mask.astype(bool)] = 1
        self.bf_label_layer.data = bf_labels

        # --- FL layers: ALWAYS rebuilt, using zeros sized to the BF image
        # when this job has no fluorescence. This prevents the previous
        # job's FL raw / label from lingering in the layer and being
        # misread as belonging to the current image (which is what caused
        # mismatches between BF and FL after navigating past jobs that
        # toggled fluorescence on/off).
        if job.has_fluorescence and job.fl_raw is not None:
            self.fl_image_layer.data = job.fl_raw
            fl_min, fl_max = float(job.fl_raw.min()), float(job.fl_raw.max())
            if fl_max > fl_min:
                self.fl_image_layer.contrast_limits = (fl_min, fl_max)
            self.fl_image_layer.visible = True
            fl_labels = np.zeros(job.fl_raw.shape, dtype=np.int32)
            if job.fl_mask is not None:
                fl_labels[job.fl_mask.astype(bool)] = 1
            self.fl_label_layer.data = fl_labels
            self.fl_label_layer.visible = True
        else:
            blank_fl = np.zeros(job.bf_raw.shape, dtype=np.uint8)
            self.fl_image_layer.data = blank_fl
            self.fl_image_layer.visible = False
            self.fl_label_layer.data = np.zeros(job.bf_raw.shape, dtype=np.int32)
            self.fl_label_layer.visible = False

        self.viewer.reset_view()

        # The job has just been loaded fresh from disk - it is not dirty.
        job.dirty = False
        self._suppress_dirty = False

        self.is_syncing = True
        try:
            self.job_dropdown.value = f"{index}: {self.jobs[index].expected_output_name}"
        finally:
            self.is_syncing = False
        self._update_status_labels()

    def _update_status_labels(self) -> None:
        job = self.jobs[self.current_index]
        kind = "BF+FL" if job.has_fluorescence else "BF only"
        origin = f" - loaded from {job.loaded_from}" if job.loaded_from else ""
        source_basename = Path(job.source_file).name if job.source_file else ""
        source_info = f"\n{source_basename}  [img {job.image_index}]" if source_basename else ""
        self.status_label.value = (
            f"Image {self.current_index + 1} / {len(self.jobs)}  ({kind}){origin}"
            f"{source_info}"
        )
        n_edited = sum(1 for j in self.jobs if j.loaded_from == "stage3" or j.dirty)
        self.progress_label.value = f"{n_edited}/{len(self.jobs)} have stage3 overrides"

    # ------------------------------------------------------------------
    def _on_job_dropdown_changed(self, value) -> None:
        if self.is_syncing:
            return
        try:
            new_index = int(str(value).split(":")[0])
        except (ValueError, IndexError):
            return
        if new_index < 0 or new_index >= len(self.jobs):
            return
        if new_index == self.current_index:
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
        # Save the currently displayed job if the user edited it. Untouched
        # jobs are NOT flushed - stage3/ therefore contains only genuine
        # overrides and Stage 4 falls back to stage2/ for the rest.
        self._save_current_state()

        n_overrides = sum(1 for j in self.jobs if j.loaded_from == "stage3")
        self.status_label.value = f"Done. {n_overrides} stage3 override(s) on disk."
        print(f"Stage 3 complete: {n_overrides}/{len(self.jobs)} jobs have stage3 overrides.")

        try:
            self.viewer.close()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 3 - interactive label curation.")
    parser.add_argument("manifest_path")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest_path)
    output_root = manifest_path.parent
    manifest_df = pd.read_csv(manifest_path)
    manifest_df = manifest_df[manifest_df["should_process"]]
    jobs = _load_curation_jobs(manifest_df, output_root=output_root)
    if not jobs:
        print("[INFO] No jobs with stage2/<name>.tif found. Run Stage 2 first.")
        return 0
    print(f"Opening curation viewer for {len(jobs)} job(s)...")
    CurationApp(jobs, output_root=output_root).open()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

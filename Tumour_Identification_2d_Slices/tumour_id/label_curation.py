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


def _load_curation_jobs(manifest_df: pd.DataFrame, output_root: Path) -> List[CurationJob]:
    jobs: List[CurationJob] = []
    stage2_dir = output_root / STAGE2_SUBDIR
    for row in manifest_df.itertuples(index=False):
        name = str(row.expected_output_name)
        curation_tif = stage2_dir / f"{name}.tif"
        if not curation_tif.exists():
            print(f"[SKIP] {name}: no stage2/{name}.tif", file=sys.stderr)
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
        ))
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
        from qtpy.QtWidgets import (
            QComboBox,
            QLabel,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        first = self.jobs[0]
        placeholder = np.zeros_like(first.bf_raw)

        self.viewer = napari.Viewer(title="Tumour Label Curation (Stage 3)")
        self.bf_image_layer = self.viewer.add_image(placeholder, name="BF", colormap="gray")
        self.fl_image_layer = self.viewer.add_image(placeholder, name="FL", colormap="green", visible=False)
        self.bf_label_layer = self.viewer.add_labels(np.zeros_like(placeholder, dtype=np.int32), name="BF tumour")
        self.fl_label_layer = self.viewer.add_labels(np.zeros_like(placeholder, dtype=np.int32), name="FL tumour")
        self.bf_label_layer.opacity = 0.5
        self.fl_label_layer.opacity = 0.5

        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.job_dropdown = QComboBox()
        for i, j in enumerate(self.jobs):
            self.job_dropdown.addItem(f"{i}: {j.expected_output_name}")
        self.job_dropdown.currentIndexChanged.connect(self._on_job_dropdown_changed)
        layout.addWidget(QLabel("Image:"))
        layout.addWidget(self.job_dropdown)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.btn_prev = QPushButton("◀ Previous")
        self.btn_prev.clicked.connect(self._go_prev)
        layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self._go_next)
        layout.addWidget(self.btn_next)

        self.btn_done = QPushButton("Done — Write Overrides")
        self.btn_done.clicked.connect(self._finish)
        layout.addWidget(self.btn_done)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        layout.addStretch(1)

        self.viewer.window.add_dock_widget(panel, area="right", name="Curation")

        self._show_index(0)
        napari.run()

    # ------------------------------------------------------------------
    def _save_current_state(self) -> None:
        job = self.jobs[self.current_index]
        bf_label_arr = np.asarray(self.bf_label_layer.data)
        if bf_label_arr.shape == job.bf_mask.shape:
            job.bf_mask = bf_label_arr > 0
        if job.has_fluorescence:
            fl_label_arr = np.asarray(self.fl_label_layer.data)
            if job.fl_mask is not None and fl_label_arr.shape == job.fl_mask.shape:
                job.fl_mask = fl_label_arr > 0

    def _show_index(self, index: int) -> None:
        if index < 0 or index >= len(self.jobs):
            return
        self.current_index = index
        job = self.jobs[index]

        self.bf_image_layer.data = job.bf_raw
        bf_min, bf_max = float(job.bf_raw.min()), float(job.bf_raw.max())
        if bf_max > bf_min:
            self.bf_image_layer.contrast_limits = (bf_min, bf_max)

        bf_labels = np.zeros(job.bf_raw.shape, dtype=np.int32)
        bf_labels[job.bf_mask.astype(bool)] = 1
        self.bf_label_layer.data = bf_labels

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
            self.fl_image_layer.visible = False
            self.fl_label_layer.visible = False

        self.viewer.reset_view()

        self.is_syncing = True
        try:
            self.job_dropdown.setCurrentIndex(index)
        finally:
            self.is_syncing = False
        self._update_status_labels()

    def _update_status_labels(self) -> None:
        job = self.jobs[self.current_index]
        kind = "BF+FL" if job.has_fluorescence else "BF only"
        self.status_label.setText(f"Image {self.current_index + 1} / {len(self.jobs)}  ({kind})")
        self.progress_label.setText("")

    # ------------------------------------------------------------------
    def _on_job_dropdown_changed(self, new_index: int) -> None:
        if self.is_syncing:
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
        self._save_current_state()

        self.status_label.setText("Writing overrides...")
        try:
            self.viewer.window.qt_window.repaint()
        except Exception:
            pass

        n_written = 0
        stage3_dir = self.output_root / STAGE3_SUBDIR
        stage3_dir.mkdir(parents=True, exist_ok=True)
        for job in self.jobs:
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
                n_written += 1
            except Exception as exc:
                print(f"[FAIL] {job.expected_output_name}: {exc}", file=sys.stderr)
        print(f"Stage 3 complete: wrote overrides for {n_written}/{len(self.jobs)} jobs.")

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

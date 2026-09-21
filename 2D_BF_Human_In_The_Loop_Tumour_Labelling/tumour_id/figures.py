"""Stage 5 (optional) - save a 4-panel summary figure per image.

For each processable manifest row, prefer ``stage3/<name>.tif`` (curated)
and fall back to ``stage2/<name>.tif`` (auto). Each figure has four axes:

    ax[0]  Brightfield (grayscale)
    ax[1]  Brightfield + semi-transparent BF label overlay
    ax[2]  Fluorescence (or fully black when no FL channel present)
    ax[3]  Fluorescence + semi-transparent FL label overlay
           (or fully black when no FL channel present)

Figures are written to ``<output_root>/figures/<name>.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe in worker processes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

STAGE2_SUBDIR = "stage2"
STAGE3_SUBDIR = "stage3"
FIGURES_SUBDIR = "figures"


def _resolve_curation_tif(output_root: Path, name: str) -> tuple[Optional[Path], str]:
    override = output_root / STAGE3_SUBDIR / f"{name}.tif"
    if override.exists():
        return override, "curated"
    auto = output_root / STAGE2_SUBDIR / f"{name}.tif"
    if auto.exists():
        return auto, "auto"
    return None, "missing"


def _overlay_rgba(mask: np.ndarray, color: tuple[float, float, float], alpha: float = 0.4) -> np.ndarray:
    """Return an RGBA image that is the given colour where ``mask`` is True
    and fully transparent elsewhere, suitable for ``ax.imshow`` over a
    grayscale base image."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = alpha
    return rgba


def save_figure_one(
    curation_tif_path: Path,
    out_path: Path,
    title: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Write a 1x4 summary figure for a single 4-channel curation TIF."""
    with tifffile.TiffFile(curation_tif_path) as tif:
        stack = np.asarray(tif.asarray())

    if stack.ndim != 3 or stack.shape[0] != 4:
        raise ValueError(f"Unexpected curation TIF shape {stack.shape} in {curation_tif_path}")

    bf_raw = stack[0]
    fl_raw = stack[1]
    bf_mask = stack[2].astype(bool)
    fl_mask = stack[3].astype(bool)
    has_fluorescence = bool(fl_raw.any())

    # Use the BF shape for the black FL placeholder so the four panels share
    # extent and aspect.
    black = np.zeros(bf_raw.shape, dtype=bf_raw.dtype)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # ax[0]: brightfield
    axes[0].imshow(bf_raw, cmap="gray")
    axes[0].set_title("Brightfield")

    # ax[1]: brightfield + BF label overlay (red)
    axes[1].imshow(bf_raw, cmap="gray")
    axes[1].imshow(_overlay_rgba(bf_mask, color=(1.0, 0.0, 0.0), alpha=0.4))
    axes[1].set_title("BF + label")

    # ax[2]: fluorescence (or black)
    if has_fluorescence:
        axes[2].imshow(fl_raw, cmap="gray")
        axes[2].set_title("Fluorescence")
    else:
        axes[2].imshow(black, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("Fluorescence (none)")

    # ax[3]: fluorescence + FL label overlay (or black)
    if has_fluorescence:
        axes[3].imshow(fl_raw, cmap="gray")
        if fl_mask.any():
            axes[3].imshow(_overlay_rgba(fl_mask, color=(0.0, 1.0, 0.0), alpha=0.4))
            axes[3].set_title("FL + label")
        else:
            axes[3].set_title("FL (no label)")
    else:
        axes[3].imshow(black, cmap="gray", vmin=0, vmax=255)
        axes[3].set_title("FL + label (none)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_one(curation_tif: Path, out_path: Path, title: str, dpi: int) -> tuple[str, Optional[str]]:
    """Worker entry point. Returns (name, error_or_None)."""
    try:
        save_figure_one(curation_tif, out_path, title=title, dpi=dpi)
        return out_path.stem, None
    except Exception as exc:
        return out_path.stem, f"{type(exc).__name__}: {exc}"


def run(
    manifest_path: str | Path,
    figures_dir: Optional[str | Path] = None,
    dpi: int = 150,
    n_workers: int = 1,
) -> Path:
    """Iterate the manifest and write one figure per processable job."""
    manifest_path = Path(manifest_path)
    output_root = manifest_path.parent
    manifest_df = pd.read_csv(manifest_path)

    if figures_dir is None:
        figures_dir = output_root / FIGURES_SUBDIR
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    jobs: List[tuple] = []
    for row in manifest_df.itertuples(index=False):
        if not row.should_process:
            continue
        name = str(row.expected_output_name)
        curation_tif, source = _resolve_curation_tif(output_root, name)
        if curation_tif is None:
            print(f"  [SKIP] {name}: no stage2/stage3 TIF")
            continue
        jobs.append((curation_tif, figures_dir / f"{name}.png", f"{name}  ({source})"))

    if n_workers and n_workers > 1 and jobs:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_workers, prefer="processes", verbose=0)(
            delayed(_render_one)(curation_tif, out_path, title, dpi)
            for curation_tif, out_path, title in jobs
        )
    else:
        results = [_render_one(curation_tif, out_path, title, dpi)
                   for curation_tif, out_path, title in jobs]

    n_ok = 0
    for name, err in results:
        if err is None:
            n_ok += 1
        else:
            print(f"  [FAIL] {name}: {err}", file=sys.stderr)
    print(f"Figures done: wrote {n_ok}/{len(jobs)} figures to {figures_dir}")
    return figures_dir


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Save per-image 4-panel summary figures.")
    parser.add_argument("manifest_path")
    parser.add_argument("--figures-dir", default=None,
                        help="Output directory (default: <manifest_dir>/figures)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1).")
    args = parser.parse_args(argv)
    run(args.manifest_path, figures_dir=args.figures_dir, dpi=args.dpi, n_workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

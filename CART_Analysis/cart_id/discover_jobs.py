"""Stage 1 — Discover LIF images and write manifest.csv.

Scans ``source_dir`` recursively for ``.lif`` files.  Each image inside a
LIF becomes one row in the manifest.  Only images with at least 4 channels
(B cells, Vasculature, BF, CART) are marked ``should_process=True``.
Z-stacks are handled by max-projection during Stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import List, Optional

import liffile
import pandas as pd

from .io_utils import parse_filename_metadata

MANIFEST_COLUMNS = [
    "job_id",
    "source_file",
    "image_index",
    "image_name",
    "expected_output_name",
    "n_channels",
    "patient",
    "day",
    "should_process",
    "skip_reason",
]

STAGE2_SUBDIR = "stage2"

# Fixed channel convention for this experiment type
BCELL_CHANNEL = 0
VASC_CHANNEL = 1
BF_CHANNEL = 2
# CART cells = last channel (index n_channels - 1)


def _sanitize(name: str) -> str:
    """Remove characters that are illegal in filenames."""
    return re.sub(r'[<>:"/\\|?*\s]+', "_", name).strip("_")


def _job_id(source_path: Path, image_index: int, image_name: str) -> str:
    raw = f"{source_path}|{image_index}|{image_name}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def build_manifest(
    source_dir: str | Path,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Scan *source_dir* for LIF files and return a manifest DataFrame.

    Parameters
    ----------
    source_dir:
        Directory to search recursively for ``.lif`` files.
    output_dir:
        If given, marks ``should_process=False`` for any image whose
        ``stage2/<expected_output_name>.tif`` already exists there.
    """
    source_dir = Path(source_dir)
    lif_files = sorted(source_dir.rglob("*.lif"))

    rows: List[dict] = []
    for lif_path in lif_files:
        try:
            with liffile.LifFile(str(lif_path)) as lif:
                images = list(lif.images)
        except Exception as exc:
            print(f"  [WARN] Could not open {lif_path.name}: {exc}", file=sys.stderr)
            continue

        lif_stem = lif_path.stem
        meta = parse_filename_metadata(lif_stem)

        for img_idx, image in enumerate(images):
            img_name = getattr(image, "name", "") or f"image_{img_idx}"
            sizes = {k.upper(): v for k, v in image.sizes.items()}
            n_channels = int(sizes.get("C", 1))

            output_name = f"{_sanitize(lif_stem)}_{_sanitize(img_name)}_img{img_idx}"
            job_id = _job_id(lif_path, img_idx, img_name)

            # Determine whether to process
            skip_reason = ""
            should_process = True
            if n_channels < 4:
                should_process = False
                skip_reason = f"too_few_channels ({n_channels})"
            elif output_dir is not None:
                tif_path = Path(output_dir) / STAGE2_SUBDIR / f"{output_name}.tif"
                if tif_path.exists():
                    should_process = False
                    skip_reason = "already_processed"

            # Per-image metadata may override the per-LIF metadata
            img_meta = parse_filename_metadata(img_name)
            patient = img_meta["patient"] if img_meta["patient"] != "unknown" else meta["patient"]
            day = img_meta["day"] if img_meta["day"] != "unknown" else meta["day"]

            rows.append({
                "job_id": job_id,
                "source_file": str(lif_path),
                "image_index": img_idx,
                "image_name": img_name,
                "expected_output_name": output_name,
                "n_channels": n_channels,
                "patient": patient,
                "day": day,
                "should_process": should_process,
                "skip_reason": skip_reason,
            })

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return df


def build_and_write_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    manifest_path: Optional[str | Path] = None,
) -> Path:
    """Build the manifest, write it to *output_dir*/manifest.csv, and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is None:
        manifest_path = output_dir / "manifest.csv"
    manifest_path = Path(manifest_path)

    df = build_manifest(source_dir, output_dir=output_dir)
    df.to_csv(manifest_path, index=False)

    n_total = len(df)
    n_process = int(df["should_process"].sum())
    print(f"Manifest written: {manifest_path}")
    print(f"  {n_process}/{n_total} images to process")
    return manifest_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1 - discover CART analysis jobs.")
    parser.add_argument("source_dir", help="Directory containing LIF files.")
    parser.add_argument("output_dir", help="Pipeline output directory.")
    args = parser.parse_args(argv)
    build_and_write_manifest(args.source_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 1 - Discover images and write a manifest CSV.

Scans the source directory for .tif/.tiff/.lif files (LIFs expanded so one
manifest row = one image = one downstream job) and writes
<output_dir>/manifest.csv.

Channel convention (auto-inferred per image):
- brightfield channel = 0 (or the only channel in a 1-channel image)
- fluorescence channel = the last channel, when n_channels > 1; otherwise none

Usage:
    from tumour_id.discover_jobs import build_and_write_manifest
    build_and_write_manifest(source_dir="...", output_dir="...")

    # or from the command line:
    python -m tumour_id.discover_jobs <source_dir> <output_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from .io_utils import enumerate_images_in_file


MANIFEST_COLUMNS: List[str] = [
    "job_id", "source_file", "image_index", "image_name",
    "expected_output_name",
    "bf_channel", "fl_channel",
    "should_process", "skip_reason",
]


def _sanitize(name: str) -> str:
    return "".join("_" if c in '<>:"/\\|?* ' else c for c in name).strip().strip(".")


def build_manifest(
    source_dir: str | Path,
    skip_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Discover images under source_dir (recursive) and build a manifest DataFrame.

    bf_channel is always 0; fl_channel is auto-set to the last channel index
    when the image has more than one channel, otherwise left blank.
    Rows whose expected output TIF already exists in skip_dir are kept with
    should_process=False and skip_reason='output_already_exists'.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    if skip_dir is None or not Path(skip_dir).is_dir():
        existing_output_names: Set[str] = set()
    else:
        existing_output_names = {p.stem for p in Path(skip_dir).iterdir() if p.is_file() and p.suffix.lower() in (".tif", ".tiff")}

    source_files = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".lif")
    )

    rows: List[dict] = []
    for source_path in source_files:
        try:
            discovered = list(enumerate_images_in_file(source_path))
        except Exception as exc:
            print(f"[SKIP] {source_path.name}: {exc}", file=sys.stderr)
            continue

        for image in discovered:
            # Build per-image output folder name
            if source_path.suffix.lower() == ".lif":
                sanitized_image_name = _sanitize(image.image_name)
                expected_output_name = f"{source_path.stem}_{sanitized_image_name}_img{image.image_index}"
            else:
                expected_output_name = source_path.stem

            # Decide skip / process
            if image.has_zt:
                should_process = False
                skip_reason = "has_zt_dim"
            elif expected_output_name in existing_output_names:
                should_process = False
                skip_reason = "output_already_exists"
            else:
                should_process = True
                skip_reason = ""

            # BF is always channel 0; FL is the last channel when n_channels > 1
            bf_channel = 0
            effective_fl = (image.n_channels - 1) if image.n_channels > 1 else None

            hash_key = f"{source_path.resolve()}|{image.image_index}|{image.image_name}"
            rows.append({
                "job_id": hashlib.sha1(hash_key.encode("utf-8")).hexdigest()[:12],
                "source_file": str(source_path.resolve()),
                "image_index": image.image_index,
                "image_name": image.image_name,
                "expected_output_name": expected_output_name,
                "bf_channel": bf_channel,
                "fl_channel": "" if effective_fl is None else int(effective_fl),
                "should_process": should_process,
                "skip_reason": skip_reason,
            })

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_and_write_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    skip_dir: Optional[str | Path] = None,
    manifest_path: Optional[str | Path] = None,
) -> Path:
    """Build the manifest and write it to <output_dir>/manifest.csv (overridable)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = output_dir / "manifest.csv"
    manifest_path = Path(manifest_path)

    manifest_df = build_manifest(
        source_dir=source_dir,
        skip_dir=skip_dir,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, index=False)

    total = len(manifest_df)
    processable = int(manifest_df["should_process"].sum())
    print(f"Wrote manifest: {manifest_path}  ({processable}/{total} to process)")
    return manifest_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1 - discover images and write a manifest CSV.")
    parser.add_argument("source_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--skip-dir", default=None, help="if set, skip images whose expected output already exists here")
    args = parser.parse_args(argv)
    build_and_write_manifest(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        skip_dir=args.skip_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from tifffile import imwrite
from skimage.measure import regionprops

from CropNuclei import CropNuclei
from SpotDetection import SpotDetector
from NucleusStats import NucleusStats
from ConcatNucleiStats import ConcatCSV
from CalculateThresholds import CalculateThreshold
from Normalization import Normalization
from ConcatSpots import ConcatSpots
from FilterSpots import FilterSpots
from Decoding import Decoding
from CountTable import CountTable


def _crop_nuclei_from_arrays(image_name, spots_stack, dapi_stack, labels_3d, crop_buffer, output_dir):
    seg_path = output_dir / f"{image_name}-seg_mask.tif"
    imwrite(seg_path, labels_3d)

    records = []
    for prop in regionprops(labels_3d):
        out = CropNuclei.crop_image(
            (prop.bbox, prop.label),
            labels_3d.shape,
            labels_3d,
            dapi_stack,
            spots_stack,
            int(crop_buffer),
            image_name,
        )
        if out is not None:
            records.append(out)

    if not records:
        raise RuntimeError("No valid nucleus crops were produced from labels_3d.")

    location_df = pd.concat(records, axis=0, ignore_index=True)
    location_path = output_dir / f"{image_name}-NucleiLocation.csv"
    location_df.to_csv(location_path, index=False)

    cropped_files = sorted(glob.glob(str(output_dir / f"{image_name}-nucleus*_cropped.tif")))
    if not cropped_files:
        raise RuntimeError("No cropped nuclei files produced from input arrays.")

    return str(seg_path), str(location_path), cropped_files


def _filter_spots_within_global_nuclei(spots_csv_path, nuclei_location_csv_path, labels_3d):
    spots_csv_path = Path(spots_csv_path)
    nuclei_location_csv_path = Path(nuclei_location_csv_path)

    spots_df = pd.read_csv(spots_csv_path)
    if spots_df.empty:
        return str(spots_csv_path), 0, 0

    required_spot_cols = {"nucleus", "y", "x"}
    if not required_spot_cols.issubset(spots_df.columns):
        raise ValueError(f"Spots table missing required columns: {required_spot_cols}")

    loc_df = pd.read_csv(nuclei_location_csv_path)
    required_loc_cols = {"nucleus", "ystart", "xstart"}
    if not required_loc_cols.issubset(loc_df.columns):
        raise ValueError(f"Nuclei location table missing required columns: {required_loc_cols}")

    spots_df["nucleus"] = spots_df["nucleus"].astype(int)
    loc_df["nucleus"] = loc_df["nucleus"].astype(int)

    merged = spots_df.merge(loc_df[["nucleus", "ystart", "xstart"]], on="nucleus", how="left")
    merged = merged.dropna(subset=["ystart", "xstart"]).copy()

    merged["x_global"] = merged["x"].astype(float) + merged["xstart"].astype(float)
    merged["y_global"] = merged["y"].astype(float) + merged["ystart"].astype(float)

    labels_2d = (np.asarray(labels_3d).max(axis=0) > 0)
    y_idx = np.rint(merged["y_global"].values).astype(int)
    x_idx = np.rint(merged["x_global"].values).astype(int)

    in_bounds = (
        (y_idx >= 0)
        & (x_idx >= 0)
        & (y_idx < labels_2d.shape[0])
        & (x_idx < labels_2d.shape[1])
    )

    inside_mask = np.zeros(len(merged), dtype=bool)
    inside_mask[in_bounds] = labels_2d[y_idx[in_bounds], x_idx[in_bounds]]

    filtered = merged.loc[inside_mask].copy()
    filtered.to_csv(spots_csv_path, index=False)
    return str(spots_csv_path), len(merged), len(filtered)


def build_codebook_and_channels_from_metadata(metadata_path, output_dir):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_path not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)
    required_cols = {
        "section",
        "gene",
        "color1",
        "color2",
        "color3",
        "color4",
        "channel",
        "fluorophore",
    }
    missing = required_cols - set(metadata_df.columns)
    if missing:
        raise ValueError(f"metadata file missing required columns: {sorted(missing)}")

    channel_df = metadata_df[metadata_df["section"].astype(str).str.lower() == "channel_info"].copy()
    codebook_df = metadata_df[metadata_df["section"].astype(str).str.lower() == "codebook"].copy()
    if channel_df.empty or codebook_df.empty:
        raise ValueError("metadata file must contain both section='channel_info' and section='codebook' rows")

    channel_df = channel_df[["channel", "fluorophore"]].dropna().copy()
    channel_df["channel"] = channel_df["channel"].astype(int)
    channel_df["fluorophore"] = channel_df["fluorophore"].astype(str)

    codebook_df = codebook_df[["gene", "color1", "color2", "color3", "color4"]].fillna("").copy()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    codebook_path = output_dir / "_tmp_codebook.csv"
    channel_info_path = output_dir / "_tmp_channels.txt"
    codebook_df.to_csv(codebook_path, index=False)
    channel_df.to_csv(channel_info_path, index=False, sep=" ", header=False)

    return codebook_path, channel_info_path


def run_legacy_pipeline_from_arrays(
    image_name,
    spots_stack,
    dapi_stack,
    labels_3d,
    metadata_path=None,
    codebook_path=None,
    channel_info_path=None,
    output_dir=None,
    spotiflow_model="general",
    min_distance=3,
    crop_buffer=2,
    default_p99=3.0,
    global_nucleus_normalisation=True,
    kneepoint_factor=2.0,
    do_global_filtering=True,
    decoding_search_radius=5,
    decoding_overlap_threshold=0.1,
    decoding_balance_threshold=0.1,
    run_normalization=False,
    normalization_sigma1=9.0,
    normalization_sigma2=2.0,
    make_count_table=False,
    write_full_inputs=False,
    postdecode_mask_sanity_check=False,
):
    spots_stack = np.asarray(spots_stack)
    dapi_stack = np.asarray(dapi_stack)
    labels_3d = np.asarray(labels_3d)

    if spots_stack.ndim != 4:
        raise ValueError(f"spots_stack must be (C,Z,Y,X), got {spots_stack.shape}")
    if dapi_stack.ndim != 3:
        raise ValueError(f"dapi_stack must be (Z,Y,X), got {dapi_stack.shape}")
    if labels_3d.ndim != 3:
        raise ValueError(f"labels_3d must be (Z,Y,X), got {labels_3d.shape}")
    if spots_stack.shape[1:] != dapi_stack.shape or dapi_stack.shape != labels_3d.shape:
        raise ValueError(
            f"Shape mismatch: spots={spots_stack.shape}, dapi={dapi_stack.shape}, labels={labels_3d.shape}"
        )

    cwd_at_start = Path.cwd()

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if not metadata_path.is_absolute():
            metadata_path = (cwd_at_start / metadata_path)
        metadata_path = metadata_path.resolve()

    if codebook_path is not None:
        codebook_path = Path(codebook_path)
        if not codebook_path.is_absolute():
            codebook_path = (cwd_at_start / codebook_path)
        codebook_path = codebook_path.resolve()

    if channel_info_path is not None:
        channel_info_path = Path(channel_info_path)
        if not channel_info_path.is_absolute():
            channel_info_path = (cwd_at_start / channel_info_path)
        channel_info_path = channel_info_path.resolve()

    if output_dir is None:
        output_dir = cwd_at_start
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = (cwd_at_start / output_dir)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prev_cwd = Path.cwd()
    try:
        import os

        os.chdir(output_dir)

        spots_path = None
        dapi_path = None
        mask_path = None

        if write_full_inputs:
            spots_path = output_dir / f"{image_name}_spotchannels.tif"
            dapi_path = output_dir / f"{image_name}_dapi.tif"
            mask_path = output_dir / f"{image_name}_segmask.tif"

            imwrite(spots_path, spots_stack, metadata={"axes": "CZYX"})
            imwrite(dapi_path, dapi_stack, metadata={"axes": "ZYX"})
            imwrite(mask_path, labels_3d.astype(np.uint16), metadata={"axes": "ZYX"})

            CropNuclei(
                image_name=image_name,
                spotchannels=str(spots_path),
                dapichannel=str(dapi_path),
                seg_mask=str(mask_path),
                buffer=int(crop_buffer),
            ).process()

            nuclei_location_path = output_dir / f"{image_name}-NucleiLocation.csv"

            cropped_files = sorted(glob.glob(str(output_dir / f"{image_name}-nucleus*_cropped.tif")))
            if not cropped_files:
                raise RuntimeError("No cropped nuclei files produced by CropNuclei.process().")
        else:
            mask_path, nuclei_location_path, cropped_files = _crop_nuclei_from_arrays(
                image_name=image_name,
                spots_stack=spots_stack,
                dapi_stack=dapi_stack,
                labels_3d=labels_3d,
                crop_buffer=int(crop_buffer),
                output_dir=output_dir,
            )

        SpotDetector(image_name=image_name, image_path=cropped_files).detect(
            model=spotiflow_model,
            min_distance=int(min_distance),
        )
        raw_spot_files = sorted(glob.glob(str(output_dir / f"{image_name}-nuclei*-spots.csv")))
        if not raw_spot_files:
            raise RuntimeError("No spot csv produced by SpotDetector.detect().")

        raw_nucleus_stats_parts = []
        for crop_file in cropped_files:
            NucleusStats(image_name=image_name, files=[crop_file]).calculate_stats(flag=True)
            nucleus_id = int(Path(crop_file).stem.split("nucleus")[-1].split("_")[0])
            part_path = output_dir / f"{image_name}_{nucleus_id}-nucleus_stats.csv"
            if part_path.exists():
                raw_nucleus_stats_parts.append(str(part_path))

        if not raw_nucleus_stats_parts:
            raise RuntimeError("No raw nucleus stats csv files were produced.")

        ConcatCSV(image_name=image_name, files=raw_nucleus_stats_parts).concat(datatype="raw")
        nuclei_stats_raw_path = output_dir / f"{image_name}_NucleiStats.csv"

        CalculateThreshold([str(nuclei_stats_raw_path)]).calculate_thresholds(
            default_p99=float(default_p99),
            global_nucleus_normalisation=bool(global_nucleus_normalisation),
        )
        normalization_threshold_path = output_dir / "normalization_threshold.csv"

        normalized_nucleus_stats_path = None
        if run_normalization:
            Normalization(
                image_name=image_name,
                image_path=cropped_files,
                threshold=str(normalization_threshold_path),
                sigma1=float(normalization_sigma1),
                sigma2=float(normalization_sigma2),
            ).normalize()

            norm_files = sorted(glob.glob(str(output_dir / "nuclei_stats_normalized_nucleus*.csv")))
            if norm_files:
                ConcatCSV(image_name=image_name, files=norm_files).concat(datatype="normalised")
                normalized_nucleus_stats_path = output_dir / f"{image_name}_NucleiStats_normalized.csv"

        ConcatSpots(image_name=image_name, files=raw_spot_files).concat()
        concat_spots_path = output_dir / f"{image_name}_spots_concat.csv"

        FilterSpots(
            spots_file=[str(concat_spots_path)],
            kneepoint_factor=float(kneepoint_factor),
            do_global_normalization=bool(global_nucleus_normalisation),
            do_global_filtering=bool(do_global_filtering),
        ).process()

        filtered_map_path = output_dir / "Filtered_Spots.csv"
        filtered_map_df = pd.read_csv(filtered_map_path)
        filtered_spots_path = Path(filtered_map_df.loc[0, "path"])

        if codebook_path is None or channel_info_path is None:
            if metadata_path is None:
                raise ValueError(
                    "Need either (codebook_path + channel_info_path) or metadata_path to run decoding."
                )
            codebook_path, channel_info_path = build_codebook_and_channels_from_metadata(
                metadata_path=metadata_path,
                output_dir=output_dir,
            )

        number_of_spotchannels = int(spots_stack.shape[0])

        decoder = Decoding(
            image_name=image_name,
            spots=str(filtered_spots_path),
            noc=number_of_spotchannels,
            radius=int(decoding_search_radius),
            overlap_threshold=float(decoding_overlap_threshold),
            balance_threshold=float(decoding_balance_threshold),
        )
        decoder.add_codebook(
            codebook_path=str(codebook_path),
            channel_info=str(channel_info_path),
            number_of_channels=number_of_spotchannels,
        )
        decoder.decode()

        decoded_spots_path = output_dir / f"{image_name}_spots_decoded.csv"
        decoded_before_filter = None
        decoded_after_filter = None
        if postdecode_mask_sanity_check:
            decoded_spots_path, decoded_before_filter, decoded_after_filter = _filter_spots_within_global_nuclei(
                spots_csv_path=decoded_spots_path,
                nuclei_location_csv_path=nuclei_location_path,
                labels_3d=labels_3d,
            )

        count_table_path = None
        if make_count_table:
            CountTable(
                image_name=image_name,
                decoded_spots=str(decoded_spots_path),
                codebook=str(codebook_path),
                nucleus_stats=str(nuclei_stats_raw_path),
            ).create_table()
            count_table_path = output_dir / f"{image_name}_count_table.csv"

        return {
            "spots_path": str(spots_path) if spots_path is not None else None,
            "dapi_path": str(dapi_path) if dapi_path is not None else None,
            "mask_path": str(mask_path) if mask_path is not None else None,
            "cropped_files": cropped_files,
            "raw_spot_files": raw_spot_files,
            "nuclei_stats_raw_path": str(nuclei_stats_raw_path),
            "normalization_threshold_path": str(normalization_threshold_path),
            "normalized_nucleus_stats_path": str(normalized_nucleus_stats_path) if normalized_nucleus_stats_path else None,
            "concat_spots_path": str(concat_spots_path),
            "filtered_spots_path": str(filtered_spots_path),
            "decoded_spots_path": str(decoded_spots_path),
            "decoded_spots_before_nuclei_filter": int(decoded_before_filter) if decoded_before_filter is not None else None,
            "decoded_spots_after_nuclei_filter": int(decoded_after_filter) if decoded_after_filter is not None else None,
            "count_table_path": str(count_table_path) if count_table_path else None,
            "codebook_path": str(codebook_path),
            "channel_info_path": str(channel_info_path),
        }
    finally:
        import os

        os.chdir(prev_cwd)

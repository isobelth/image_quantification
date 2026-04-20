"""Cellpose and fluorescence segmentation functions."""

from pathlib import Path

import numpy as np
from cellpose import io, models
from skimage.filters import (
    gaussian,
    threshold_mean,
    threshold_minimum,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from skimage.measure import label


def cellpose_live_segmentation(
    stack,
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    model_type="cpsam",
    custom_model_path=None,
    gpu=True,
    progress_callback=None,
):
    """Segment a brightfield stack frame-by-frame with Cellpose.

    Parameters
    ----------
    stack : ndarray or str or Path
        Brightfield image stack of shape (n_frames, H, W), a single 2-D
        image, or a file path that Cellpose can read.
    diameter : float or None
        Expected cell diameter in pixels. If None, estimated automatically
        from the first frame.
    flow_threshold : float
        Cellpose flow threshold (lower = stricter boundaries).
    cellprob_threshold : float
        Cellpose cell-probability threshold (higher = fewer cells).
    min_size : int
        Minimum cell area in pixels; smaller objects are removed.
    model_type : str
        Built-in Cellpose model name (e.g. "cpsam", "cyto3", "nuclei").
        Ignored when *custom_model_path* is provided.
    custom_model_path : str or Path or None
        Path to a user-trained Cellpose model file. Overrides *model_type*.
    gpu : bool
        Whether to use GPU acceleration.
    progress_callback : callable or None
        Called as ``progress_callback(current_frame, total_frames)`` after
        each frame so the GUI can update a progress bar.

    Returns
    -------
    ndarray, uint16
        Segmentation masks with the same spatial shape as the input. If the
        input was a single 2-D image, a 2-D array is returned; otherwise
        a 3-D stack of shape (n_frames, H, W).
    """
    if isinstance(stack, (str, Path)):
        stack = io.imread(stack)
    if stack.ndim == 2:
        stack = stack[np.newaxis, :, :]

    if custom_model_path:
        print(f"[Cellpose] Loading custom model: {custom_model_path}")
        model = models.CellposeModel(
            gpu=gpu, pretrained_model=str(custom_model_path)
        )
    else:
        print(f"[Cellpose] Loading model: {model_type}")
        model = models.CellposeModel(gpu=gpu, model_type=model_type)

    if diameter is None:
        first_frame_masks, _, _ = model.eval(
            stack[0], diameter=30, flow_threshold=flow_threshold, min_size=min_size
        )
        if first_frame_masks.max() > 0:
            areas = [
                np.sum(first_frame_masks == i)
                for i in range(1, min(first_frame_masks.max() + 1, 50))
            ]
            diameter = 2 * np.sqrt((np.median(areas) if areas else 700) / np.pi)
        else:
            diameter = 30.0
        print(f"[Cellpose] Estimated diameter: {diameter:.1f} px")

    n_frames = stack.shape[0]
    out = np.zeros(stack.shape, dtype=np.uint16)
    print(
        f"[Cellpose] Segmenting {n_frames} frame(s) "
        f"(diameter={diameter:.1f}, flow_thresh={flow_threshold}, "
        f"cellprob_thresh={cellprob_threshold}, min_size={min_size})..."
    )

    for frame_index in range(n_frames):
        frame_masks, _, _ = model.eval(
            stack[frame_index],
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            resample=True,
        )
        n_cells = frame_masks.max()
        out[frame_index] = frame_masks.astype(np.uint16)
        print(f"[Cellpose] Frame {frame_index + 1}/{n_frames} — {n_cells} cell(s) detected")
        if progress_callback is not None:
            progress_callback(frame_index + 1, n_frames)

    print(f"[Cellpose] Segmentation complete. Processed {n_frames} frame(s).")
    return out[0] if n_frames == 1 else out


def segment_fluorescence(stacks, blur_sigma=1.0, threshold_method="otsu"):
    """Segment fluorescence channels by Gaussian blur + per-frame thresholding.

    For each fluorophore stack:
      1. Apply Gaussian blur.
      2. Compute a separate threshold for each frame independently
         (handles signal drift over time).
      3. Binarise: pixels above that frame's threshold = "positive".
      4. Label connected components of fluorescence-positive pixels.

    Parameters
    ----------
    stacks : dict[str, ndarray]
        {fluorophore_name: array of shape (n_frames, H, W)}.
    blur_sigma : float
        Sigma for Gaussian smoothing.
    threshold_method : str or dict[str, str]
        Thresholding algorithm name(s). A single string applies to all
        channels; a dict maps each channel name to its own method.
        Supported: "mean", "minimum", "yen", "otsu", "triangle".

    Returns
    -------
    result : dict
        Top-level keys: "threshold_methods", "blur_sigma", plus one key
        per fluorophore name. Each fluorophore entry is a dictionary with:
        - "blurred"             : ndarray (n_frames, H, W) — smoothed stack
        - "thresholds_per_frame": ndarray (n_frames,) — threshold for each frame
        - "positive"            : ndarray (n_frames, H, W) bool — binary mask
        - "positive_labels"     : ndarray (n_frames, H, W) uint32 — labelled blobs
    """
    threshold_functions = {
        "mean": threshold_mean,
        "minimum": threshold_minimum,
        "yen": threshold_yen,
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
    }
    if isinstance(threshold_method, str):
        threshold_map = {name: threshold_method.lower() for name in stacks}
    else:
        threshold_map = {name: str(threshold_method.get(name, "otsu")).lower() for name in stacks}

    result = {"threshold_methods": threshold_map, "blur_sigma": blur_sigma}
    for channel_name, stack in stacks.items():
        threshold_method_name = threshold_map[channel_name]
        if threshold_method_name not in threshold_functions:
            raise ValueError(f"Unsupported threshold for '{channel_name}': {threshold_method_name}")
        threshold_function = threshold_functions[threshold_method_name]
        blurred = np.stack([gaussian(frame, sigma=blur_sigma, preserve_range=True) for frame in stack], axis=0)

        num_frames = blurred.shape[0]
        thresholds_per_frame = np.zeros(num_frames)
        positive = np.zeros_like(blurred, dtype=bool)
        positive_labels = np.zeros(blurred.shape, dtype=np.uint32)
        for frame_index in range(num_frames):
            frame_threshold = threshold_function(blurred[frame_index])
            thresholds_per_frame[frame_index] = frame_threshold
            positive[frame_index] = blurred[frame_index] > frame_threshold
            positive_labels[frame_index] = label(positive[frame_index]).astype(np.uint32)

        result[channel_name] = {
            "blurred": blurred,
            "thresholds_per_frame": thresholds_per_frame,
            "positive": positive,
            "positive_labels": positive_labels,
        }
    return result

"""Cell measurement and fluorescence-to-cell assignment functions."""

import numpy as np
from skimage.measure import regionprops


def measure_all_cells_in_frame(label_image):
    """Measure area and roundness for every cell in a label image.

    Runs a single regionprops call for the entire frame and returns a
    dictionary keyed by cell label.

    Parameters
    ----------
    label_image : ndarray (H, W), uint32
        Label image where each cell has a unique integer ID (0 = background).

    Returns
    -------
    dict[int, tuple[int, float]]
        {cell_label_id: (area_in_pixels, roundness)} for every cell.
        Roundness = minor_axis / major_axis (1.0 = circle, closer to 0 = elongated).
    """
    measurements = {}
    for region in regionprops(label_image):
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        roundness = minor_axis / major_axis if major_axis > 0 else np.nan
        measurements[region.label] = (region.area, roundness)
    return measurements


def assign_positive_objects_to_cells(positive_labels_frame, linked_labels_frame):
    """Link thresholded fluorescence blobs to tracked cells using centroids.

    For each connected component in the thresholded fluorescence image,
    finds its centroid (centre-of-mass pixel) and assigns it to whichever
    tracked cell label sits at that pixel. If the centroid falls on
    background (label 0), the blob is discarded.

    Parameters
    ----------
    positive_labels_frame : ndarray (H, W), uint32
        Labels of thresholded fluorescence blobs for one frame.
    linked_labels_frame : ndarray (H, W), uint32
        Tracked cell segmentation labels for the same frame.

    Returns
    -------
    cell_positive_area : dict[int, int]
        {cell_label_id: total fluorescence-positive pixel area} for cells
        that had at least one blob centroid land inside them.
    """
    cell_positive_area = {}
    max_y = linked_labels_frame.shape[0] - 1
    max_x = linked_labels_frame.shape[1] - 1
    for region in regionprops(positive_labels_frame):
        centroid_y = min(max(int(round(region.centroid[0])), 0), max_y)
        centroid_x = min(max(int(round(region.centroid[1])), 0), max_x)
        cell_id = int(linked_labels_frame[centroid_y, centroid_x])
        if cell_id > 0:
            cell_positive_area[cell_id] = cell_positive_area.get(cell_id, 0) + region.area
    return cell_positive_area


def compute_cell_positivity(linked_labels, pos_label_imgs, fluorophore_names):
    """Map fluorescence blobs to tracked cells for every frame and channel.

    Uses assign_positive_objects_to_cells (centroid-based) to decide which
    tracked cell each fluorescence-positive blob belongs to.

    Parameters
    ----------
    linked_labels : ndarray (n_frames, H, W), uint32
        Tracked cell segmentation from TrackMate.
    pos_label_imgs : dict[str, ndarray]
        {fluorophore_name: (n_frames, H, W) uint32 labels}
        from segment_fluorescence().
    fluorophore_names : list[str]
        Names matching the keys in pos_label_imgs.

    Returns
    -------
    frame_cell_pos : dict[str, dict[int, dict[int, int]]]
        Nested lookup: frame_cell_pos[fluorophore][frame_index][cell_id]
        gives the positive pixel area assigned to that cell in that frame.
    positive_cell_labels : dict[str, ndarray]
        {fluorophore: (n_frames, H, W) uint32} — label images where only
        positive cells are painted with their cell ID, everything else 0.
    """
    num_frames = linked_labels.shape[0]
    frame_cell_pos = {}
    positive_cell_labels = {}
    for fluorophore_name in fluorophore_names:
        frame_cell_pos[fluorophore_name] = {}
        output_labels = np.zeros_like(linked_labels)
        for frame_index in range(num_frames):
            cell_positivity = assign_positive_objects_to_cells(
                pos_label_imgs[fluorophore_name][frame_index],
                linked_labels[frame_index],
            )
            frame_cell_pos[fluorophore_name][frame_index] = cell_positivity
            positive_ids = np.array(list(cell_positivity.keys()), dtype=np.uint32)
            if len(positive_ids) > 0:
                is_positive = np.isin(linked_labels[frame_index], positive_ids)
                output_labels[frame_index] = np.where(is_positive, linked_labels[frame_index], 0)
        positive_cell_labels[fluorophore_name] = output_labels
    return frame_cell_pos, positive_cell_labels

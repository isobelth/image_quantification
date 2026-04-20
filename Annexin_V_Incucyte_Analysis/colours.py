"""Colour inference and category colour-mapping for plots and Napari."""

import colorsys

import matplotlib.colors as mcolors
import numpy as np
from typing import Dict


COLOUR_KEYWORDS: Dict[str, dict] = {
    "red":     {"mpl": "tab:red",    "napari": "red",     "rgb": (1.0, 0.0, 0.0)},
    "green":   {"mpl": "tab:green",  "napari": "green",   "rgb": (0.0, 0.8, 0.0)},
    "blue":    {"mpl": "tab:blue",   "napari": "blue",    "rgb": (0.2, 0.4, 1.0)},
    "yellow":  {"mpl": "goldenrod",  "napari": "yellow",  "rgb": (0.9, 0.85, 0.0)},
    "cyan":    {"mpl": "tab:cyan",   "napari": "cyan",    "rgb": (0.0, 0.8, 0.8)},
    "magenta": {"mpl": "tab:pink",   "napari": "magenta", "rgb": (0.9, 0.0, 0.6)},
}


def assign_colours(names):
    """Assign plot colours to a list of fluorophore or category names.

    Names containing a colour keyword (e.g. "Red", "Green") get that
    colour. Unknown names get assigned unused colours from the palette
    in order, avoiding collisions with already-matched names.

    Parameters
    ----------
    names : list[str]
        Fluorophore or category names.

    Returns
    -------
    dict[str, dict]
        {name: {"mpl": ..., "napari": ..., "rgb": ...}} for each input name.
    """
    all_colour_entries = list(COLOUR_KEYWORDS.values())
    result = {}
    used_palette_indices = set()

    for name in names:
        lowercase_name = name.lower()
        for palette_index, (keyword, entry) in enumerate(COLOUR_KEYWORDS.items()):
            if keyword in lowercase_name:
                result[name] = entry
                used_palette_indices.add(palette_index)
                break

    available_indices = [i for i in range(len(all_colour_entries)) if i not in used_palette_indices]
    fallback_counter = 0
    for name in names:
        if name not in result:
            if available_indices:
                result[name] = all_colour_entries[available_indices[fallback_counter % len(available_indices)]]
            else:
                result[name] = all_colour_entries[fallback_counter % len(all_colour_entries)]
            fallback_counter += 1

    return result


def build_category_colormap(categories):
    """Build an RGB colour mapping for snapshot categories.

    Single-fluorophore categories get their inferred colour directly.
    Compound categories like "Red+Green" get the mean of their component
    colours (clamped to [0, 1]). "negative" is always grey.

    Parameters
    ----------
    categories : list[str]
        Category names, e.g. ["Red", "Green", "Red+Green", "negative"].

    Returns
    -------
    dict[str, tuple[float, float, float]]
        {category_name: (R, G, B)} with values in 0-1 range.
    """
    single_fluorophores = []
    for category in categories:
        if category == "negative":
            continue
        for part in category.split("+"):
            if part not in single_fluorophores:
                single_fluorophores.append(part)
    colour_assignments = assign_colours(single_fluorophores)
    single_rgb = {name: np.array(colour_assignments[name]["rgb"]) for name in single_fluorophores}
    colormap = {}
    for category in categories:
        if category == "negative":
            colormap[category] = (0.7, 0.7, 0.7)
            continue
        parts = category.split("+")
        rgb = np.clip(np.mean([single_rgb.get(part, np.array([0.5, 0.5, 0.5])) for part in parts], axis=0), 0, 1)
        colormap[category] = tuple(rgb.tolist())
    return colormap


# --------------- Napari label-colouring helpers ---------------

def get_fluor_base_colour(name):
    """Map a fluorophore name to a matplotlib base colour string.

    Parameters
    ----------
    name : str
        Fluorophore name (e.g. "Green", "Red", "Annexin V").

    Returns
    -------
    str
        A matplotlib named colour (e.g. "limegreen", "red").
    """
    name_lower = name.lower()
    if "green" in name_lower:
        return "limegreen"
    elif "red" in name_lower:
        return "red"
    elif "blue" in name_lower:
        return "dodgerblue"
    return "cyan"


def build_direct_label_colormap(label_stack, base_colour,
                                hue_span=0.02, sat_span=0.45, light_span=0.45,
                                seed=0, alpha=1.0):
    """Build a ``DirectLabelColormap`` with jittered shades keyed by actual label IDs.

    Parameters
    ----------
    label_stack : np.ndarray
        Integer label image (2-D, 3-D, or 4-D). Background must be 0.
    base_colour : str
        Any matplotlib named colour (e.g. "red", "limegreen").
    hue_span, sat_span, light_span : float
        Half-width of the uniform jitter applied to hue / saturation / lightness.
    seed : int
        RNG seed for reproducibility.
    alpha : float
        Alpha channel value for every label colour.

    Returns
    -------
    DirectLabelColormap
    """
    from napari.utils.colormaps import DirectLabelColormap

    unique_ids = np.unique(label_stack)
    unique_ids = unique_ids[unique_ids != 0]
    n = len(unique_ids)

    colour_dict = {0: (0.0, 0.0, 0.0, 0.0)}
    if n > 0:
        rng = np.random.default_rng(seed)
        r, g, b = mcolors.to_rgb(base_colour)
        base_h, base_l, base_s = colorsys.rgb_to_hls(r, g, b)
        hues = (base_h + rng.uniform(-hue_span, hue_span, n)) % 1.0
        sats = np.clip(base_s + rng.uniform(-sat_span, sat_span, n), 0.20, 1.00)
        lights = np.clip(base_l + rng.uniform(-light_span, light_span, n), 0.18, 0.88)
        for i, lid in enumerate(unique_ids):
            pr, pg, pb = colorsys.hls_to_rgb(float(hues[i]), float(lights[i]), float(sats[i]))
            colour_dict[int(lid)] = (pr, pg, pb, float(alpha))

    return DirectLabelColormap(color_dict=colour_dict)


def add_coloured_labels(viewer, label_stack, name, base_colour, opacity=0.5, **kwargs):
    """Add a labels layer with jittered shades of *base_colour*.

    Convenience wrapper around :func:`build_direct_label_colormap` that
    creates the colourmap and adds the layer in one call.

    Parameters
    ----------
    viewer : napari.Viewer
    label_stack : np.ndarray
    name : str
    base_colour : str
    opacity : float
    **kwargs
        Forwarded to ``viewer.add_labels()``.

    Returns
    -------
    napari.layers.Labels
    """
    cmap = build_direct_label_colormap(label_stack, base_colour)
    return viewer.add_labels(label_stack.astype(np.int32), name=name,
                             opacity=opacity, colormap=cmap, **kwargs)

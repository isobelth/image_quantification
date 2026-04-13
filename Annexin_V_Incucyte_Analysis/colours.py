"""Colour inference and category colour-mapping for plots and Napari."""

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

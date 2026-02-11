# Placenta Perfusion GUI (terminal runnable)

This folder is a lightweight wrapper around the notebook-based GUI in:
- `Vasculature/placenta_perfusion_individal_viewing.ipynb`

## Run without installing

From this folder:
- `python placenta_permeability.py`

Or (equivalent):
- `python -m placenta_permeability`

## Install and run as a command (optional)

From this folder (note: `-e` takes a *path*, not the project name):
- `python -m pip install -e .`
- `placenta-perfusion-gui`

From the repo root (equivalent):
- `python -m pip install -e .\placenta_perfusion_gui_project`
- `placenta-perfusion-gui`

## Notes

- This project expects the same Python environment as the notebook (napari + magicgui + liffile + scientific stack).
- The GUI has a **Time separation (s)** control; it defaults to `180`.
- Closing the napari window exits the process when run from the terminal.

## Developer

- To regenerate `your_gui.py` from the notebook, use `tools/export_gui_from_notebook.py`.

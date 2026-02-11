# Placenta Perfusion GUI (terminal runnable)

This folder is a lightweight wrapper around the notebook-based GUI in:
- `Vasculature/placenta_perfusion.ipynb`

## Run without installing

From this folder:
- `python your_gui.py`

## Install and run as a command (optional)

From this folder:
- `pip install -e .`
- `placenta-perfusion-gui`

## Notes

- This project expects the same Python environment as the notebook (napari + magicgui + liffile + scientific stack).
- The GUI has a **Time separation (s)** control; it defaults to `180`.

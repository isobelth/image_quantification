# Placenta Perfusion GUI (terminal runnable)

This folder is a lightweight wrapper around the notebook-based GUI in:
- `Vasculature/placenta_perfusion_individal_viewing.ipynb`

## First-time setup (Windows, beginner-friendly)

Follow these steps in order.

1. Install Visual Studio Code
	- Download and install from: https://code.visualstudio.com/

2. Install Miniconda (and add it to PATH)
	- Download Miniconda for Windows from: https://docs.conda.io/en/latest/miniconda.html
	- Run the installer.
	- In **Advanced Options**, check **Add Miniconda to my PATH environment variable**.

3. Open VS Code and confirm conda is working
	- Open this project folder in VS Code.
	- Open a terminal in VS Code: **Terminal → New Terminal**.
	- Run:
	  - `conda --version`
	- If you see a version number, conda is working.

4. Create the environment from `Vasculature/environment.yml`
	- In the same VS Code terminal, make sure you are in `placenta_perfusion_gui_project`.
	- Run:
	  - `conda env create -f ..\Vasculature\environment.yml`

5. Activate the environment
	- Run:
	  - `conda activate nap-ij`

## Run the GUI

From `placenta_perfusion_gui_project`, with the environment activated:
- `python placenta_permeability.py`

Or (equivalent):
- `python -m placenta_permeability`

## Notes

- The GUI has a **Time separation (s)** control; it defaults to `180`. This is the time (seconds) between first and last image in your stack
- Closing the napari window exits the process when run from the terminal.

## Developer

- To regenerate `your_gui.py` from the notebook, use `tools/export_gui_from_notebook.py`.

"""Command-line entrypoint for the Placenta Perfusion napari GUI.

Typical usage:
  python cli.py

After installing (e.g. `pip install -e .`), you can run the console script:
  placenta-perfusion-gui
"""

from __future__ import annotations


def main() -> None:
    # Import lazily so `python -m pip install ...` etc doesn't import napari.
    from your_gui import main as run_gui

    run_gui()


if __name__ == "__main__":
    main()

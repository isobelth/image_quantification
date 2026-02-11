from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cell_source_to_text(cell_source) -> str:
    if isinstance(cell_source, str):
        return cell_source.rstrip() + "\n"
    if isinstance(cell_source, list):
        return "\n".join([str(line).rstrip("\n") for line in cell_source]).rstrip() + "\n"
    return ""


def export(notebook_path: Path, output_path: Path, n_code_cells: int = 3) -> None:
    obj = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [c for c in obj.get("cells", []) if c.get("cell_type") == "code"]

    if len(code_cells) < n_code_cells:
        raise SystemExit(f"Expected >={n_code_cells} code cells, found {len(code_cells)}")

    parts: list[str] = []
    for idx in range(n_code_cells):
        parts.append(_cell_source_to_text(code_cells[idx].get("source", [])))
        parts.append("\n\n")

    parts.append(
        "\n\n"
        "def main():\n"
        "    \"\"\"Launch the napari GUI.\"\"\"\n"
        "    _app = PlacentaPerfusionApp()\n"
        "    import napari as _napari\n"
        "    _napari.run()\n"
        "\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    output_path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--n-code-cells", type=int, default=3)
    args = parser.parse_args()

    export(args.notebook, args.output, n_code_cells=int(args.n_code_cells))
    print(f"Wrote: {args.output.resolve()}")


if __name__ == "__main__":
    main()

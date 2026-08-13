"""Generate the original-dimension Pupper shell model and design metadata."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from curl_robot_2d.pupper_shell_geometry import PupperShellDesign, solve_compact_geometry
from curl_robot_2d.pupper_shell_model import write_pupper_shell_mjcf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/pupper_original_geometry_shell_r127p5"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-radius-mm", type=float, default=127.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    design = replace(
        PupperShellDesign(), shell_outer_radius=args.shell_radius_mm / 1000.0
    )
    output_dir = args.output_dir.expanduser().resolve()
    model_path = write_pupper_shell_mjcf(output_dir / "model.xml", design)
    solution = solve_compact_geometry(design)
    metadata = {"design": asdict(design), "compact_solution": asdict(solution), "model": str(model_path.resolve())}
    (output_dir / "geometry.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

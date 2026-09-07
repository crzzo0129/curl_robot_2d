"""Bake front/rear abduction offsets into a copy of any RollingQuad MJCF.

Unlike build_primitive_collision.py (which replaces meshes with primitives),
this only rewrites the keyframes and leaves the collision geometry untouched,
so it can produce an abducted MESH model for CEM/viewing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.build_primitive_collision import _override_keyframe_abduction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/rollingquad_description_2/mjcf/rollingquad.xml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--front-abduction-deg", type=float, required=True)
    parser.add_argument("--rear-abduction-deg", type=float, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force")
    tree = ET.parse(args.input)
    _override_keyframe_abduction(
        tree,
        math.radians(args.front_abduction_deg),
        math.radians(args.rear_abduction_deg),
    )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=False)

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(output))
    print(f"compiled ngeom={model.ngeom} nbody={model.nbody} nu={model.nu}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bake a flat hfield floor into terrain variants of the RollingQuad MJCFs.

The hfield keeps flat heights (base only); the environment or the domain
randomization callback overwrites ``model.hfield_data`` with the slope profile
at run time.  Baking keeps the relative ``../meshes`` references valid.
"""

from __future__ import annotations

from pathlib import Path

from curl_robot_2d_mjx.terrain_3d import (
    SlopeTerrainConfig,
    inject_hfield_into_mjcf,
)


MJCF_DIR = Path("assets/rollingquad_description_2/mjcf")

VARIANTS = {
    "rollingquad_primitive_abd10.xml": "rollingquad_primitive_abd10_terrain.xml",
    "rollingquad_abd10.xml": "rollingquad_abd10_terrain.xml",
}


def main() -> None:
    config = SlopeTerrainConfig(slope_angle_deg=0.0)
    for input_name, output_name in VARIANTS.items():
        inject_hfield_into_mjcf(
            MJCF_DIR / input_name,
            MJCF_DIR / output_name,
            config,
            force=True,
        )
        print(f"baked {output_name}")


if __name__ == "__main__":
    main()

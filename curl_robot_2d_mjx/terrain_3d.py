"""Slope terrain for 3-D rolling: a flat -> slope -> flat heightfield.

The rolling robot travels along world +x.  Terrain height therefore varies only
along x and is constant across the lateral (y) direction.  The profile is a
smoothstep-ramped linear slope so the robot enters and leaves the incline
gradually instead of hitting a sharp corner.

MuJoCo hfield notes (verified empirically):
  * actual surface height = hfield_size[3] (base) + hfield_size[2] (scale) * data
  * ``hfield_data`` is column-major: ``data[col * nrow + row]``, where ``col``
    indexes x and ``row`` indexes y.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SlopeTerrainConfig:
    """Parameters for one flat -> slope -> flat heightfield."""

    slope_angle_deg: float = 0.0
    slope_start_distance_m: float = 3.0
    transition_length_m: float = 0.5
    slope_length_m: float = 2.0
    # hfield grid (constant along y).
    extent_x_m: float = 16.0
    extent_y_m: float = 3.0
    ncol: int = 800
    nrow: int = 6
    # MuJoCo hfield height mapping: surface = base + scale * data.
    hfield_base_z: float = 0.001
    hfield_scale_z: float = 1.0

    @property
    def rise_rate(self) -> float:
        """Signed dz/dx of the constant-slope section."""

        return math.tan(math.radians(self.slope_angle_deg))

    @property
    def total_rise(self) -> float:
        """Total height change across both transitions plus the slope."""

        return self.rise_rate * (self.transition_length_m + self.slope_length_m)

    @property
    def is_flat(self) -> bool:
        return abs(self.slope_angle_deg) < 1e-9


def validate_slope_terrain_config(config: SlopeTerrainConfig) -> None:
    if not math.isfinite(config.slope_angle_deg):
        raise ValueError("slope_angle_deg must be finite")
    for value, name in (
        (config.slope_start_distance_m, "slope_start_distance_m"),
        (config.transition_length_m, "transition_length_m"),
        (config.slope_length_m, "slope_length_m"),
        (config.extent_x_m, "extent_x_m"),
        (config.extent_y_m, "extent_y_m"),
        (config.hfield_scale_z, "hfield_scale_z"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.hfield_base_z) or config.hfield_base_z <= 0.0:
        raise ValueError("hfield_base_z must be finite and positive")
    if config.ncol < 2 or config.nrow < 2:
        raise ValueError("ncol and nrow must be at least 2")


def smoothstep(u: np.ndarray) -> np.ndarray:
    clamped = np.clip(u, 0.0, 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _smoothstep_integral(u: np.ndarray) -> np.ndarray:
    """Integral of smoothstep over [0, u]; equals 1/2 at u == 1."""

    clamped = np.clip(u, 0.0, 1.0)
    return clamped**3 - 0.5 * clamped**4


def terrain_height_array(x: np.ndarray, config: SlopeTerrainConfig) -> np.ndarray:
    """Return terrain surface height (relative to the start) at world x.

    The profile is flat, then a smoothstep entry transition, a constant-slope
    section, a smoothstep exit transition, then flat again.  Heights are signed:
    positive for uphill, negative for downhill.
    """

    validate_slope_terrain_config(config)
    if config.is_flat:
        return np.zeros_like(x, dtype=np.float64)

    rise = config.rise_rate
    length = config.transition_length_m
    x0 = config.slope_start_distance_m
    x1 = x0 + length
    x2 = x1 + config.slope_length_m
    x3 = x2 + length

    height = np.zeros_like(x, dtype=np.float64)
    # Entry transition: rate ramps 0 -> rise.
    entry = (x > x0) & (x <= x1)
    height[entry] = rise * length * _smoothstep_integral(
        (x[entry] - x0) / length
    )
    # Constant slope.
    constant = (x > x1) & (x <= x2)
    height[constant] = rise * length * 0.5 + rise * (x[constant] - x1)
    # Exit transition: rate ramps rise -> 0, so the remaining-rate integral is
    # u - smoothstep_integral(u).
    exit_ = (x > x2) & (x <= x3)
    u_exit = (x[exit_] - x2) / length
    height[exit_] = (
        rise * length * 0.5
        + rise * config.slope_length_m
        + rise * length * (np.clip(u_exit, 0.0, 1.0) - _smoothstep_integral(u_exit))
    )
    # Flat top.
    height[x > x3] = config.total_rise
    return height


def terrain_height(x: float, config: SlopeTerrainConfig) -> float:
    return float(terrain_height_array(np.asarray([x]), config)[0])


def column_centers_x(config: SlopeTerrainConfig) -> np.ndarray:
    """World x of each hfield column center."""

    validate_slope_terrain_config(config)
    return np.linspace(
        -0.5 * config.extent_x_m,
        0.5 * config.extent_x_m,
        config.ncol,
    )


def hfield_data_3d(config: SlopeTerrainConfig) -> np.ndarray:
    """Return (nrow, ncol) hfield data for the terrain.

    The returned matrix matches the MuJoCo logical layout (row=y, col=x); the
    caller reshapes it to the flat column-major ``hfield_data`` order when
    writing ``model.hfield_data``.
    """

    validate_slope_terrain_config(config)
    heights = terrain_height_array(column_centers_x(config), config)
    # MuJoCo surface = base + scale * data.  Solve for data so the surface
    # equals ``heights`` while keeping data >= 0: shift the profile up by its
    # minimum and absorb the base offset.
    shifted = heights - float(np.min(heights))
    data = (shifted - config.hfield_base_z) / config.hfield_scale_z
    return np.tile(data[None, :], (config.nrow, 1))


def flat_hfield_data_3d(config: SlopeTerrainConfig) -> np.ndarray:
    """Return all-flat hfield data matching the same base/scale mapping."""

    validate_slope_terrain_config(config)
    data = (-config.hfield_base_z) / config.hfield_scale_z
    return np.full((config.nrow, config.ncol), data)


def hfield_data_flat_column_major(config: SlopeTerrainConfig, data_2d: np.ndarray) -> np.ndarray:
    """Reshape (nrow, ncol) data into MuJoCo's flat column-major order."""

    if data_2d.shape != (config.nrow, config.ncol):
        raise ValueError(
            f"hfield data must be ({config.nrow}, {config.ncol}), "
            f"got {data_2d.shape}"
        )
    return np.asarray(data_2d, dtype=np.float64).T.reshape(-1)


def inject_hfield_into_mjcf(
    input_xml: Path,
    output_xml: Path,
    config: SlopeTerrainConfig,
    *,
    force: bool = False,
) -> None:
    """Bake a flat hfield floor into a copy of a RollingQuad MJCF.

    The original ``floor`` plane is replaced by a ``type="hfield"`` geom whose
    grid parameters come from ``config``.  Heights are left flat (base only);
    the environment or the domain-randomization callback overwrites
    ``model.hfield_data`` with the slope profile at run time.
    """

    validate_slope_terrain_config(config)
    output_xml = output_xml.resolve()
    if output_xml.exists() and not force:
        raise FileExistsError(f"{output_xml} exists; pass force=True")

    tree = ET.parse(input_xml)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    existing = asset.find("hfield[@name='terrain']")
    if existing is not None:
        asset.remove(existing)
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": "terrain",
            "nrow": str(config.nrow),
            "ncol": str(config.ncol),
            "size": (
                f"{0.5 * config.extent_x_m:.6g} "
                f"{0.5 * config.extent_y_m:.6g} "
                f"{config.hfield_scale_z:.6g} "
                f"{config.hfield_base_z:.6g}"
            ),
        },
    )

    worldbody = root.find("worldbody")
    floor = worldbody.find("geom[@name='floor']")
    if floor is None:
        raise ValueError("input model is missing the 'floor' geom")
    # Keep the floor's contype/conaffinity and friction so the self-collision
    # contract and rolling contact rewards are unchanged.
    for key in ("size", "type", "material"):
        floor.attrib.pop(key, None)
    floor.set("type", "hfield")
    floor.set("hfield", "terrain")

    ET.indent(tree, space="  ")
    tree.write(str(output_xml), encoding="utf-8", xml_declaration=False)


def slope_terrain_config_from_task(task) -> SlopeTerrainConfig:
    """Build a :class:`SlopeTerrainConfig` from a ``Rolling3DConfig``."""

    return SlopeTerrainConfig(
        slope_angle_deg=task.terrain_slope_angle_deg,
        slope_start_distance_m=task.terrain_slope_start_distance_m,
        transition_length_m=task.terrain_transition_length_m,
        slope_length_m=task.terrain_slope_length_m,
        extent_x_m=task.terrain_extent_x_m,
        extent_y_m=task.terrain_extent_y_m,
    )


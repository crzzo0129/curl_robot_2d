"""Small 2-D rough heightfields for the deployment-compatible walking task.

No JAX/MuJoCo imports: geometry helpers accept numpy or jax.numpy. MuJoCo
stores rows along y, columns along x, flattened in C order. The surface is
geom_z + size_z * data; size[3] is depth BELOW zero, not a surface offset.
"""

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class RoughTerrainConfig:
    half_size_m: float = 8.0       # 16 x 16 m, with 10 cm grid spacing
    grid_size: int = 161
    min_height_m: float = 0.005    # total valley-to-peak height, not +/- height
    max_height_m: float = 0.015
    flat_probability: float = 0.40
    spawn_radius_m: float = 0.45
    transition_m: float = 0.35
    edge_margin_m: float = 0.50
    base_depth_m: float = 0.05
    seed: int = 731

    def validate(self):
        for name in ("half_size_m", "min_height_m", "max_height_m",
                     "spawn_radius_m", "transition_m", "edge_margin_m",
                     "base_depth_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"terrain {name} must be finite and positive")
        if self.min_height_m > self.max_height_m:
            raise ValueError("terrain min height exceeds max height")
        if not 0.0 <= self.flat_probability <= 1.0:
            raise ValueError("terrain flat probability must be in [0, 1]")
        if self.grid_size < 3 or self.grid_size % 2 != 1:
            raise ValueError("terrain grid size must be odd and >= 3")
        if self.spawn_radius_m + self.transition_m >= self.half_size_m - self.edge_margin_m:
            raise ValueError("terrain spawn/transition region leaves no walking area")


def inject_heightfield(xml, config):
    """Replace the ground plane, preserving robot geometry and keyframes."""
    config.validate()
    root = ET.fromstring(xml)
    asset = root.find("asset")
    floor = root.find("./worldbody/geom[@name='floor']")
    if asset is None or floor is None or floor.get("type") != "plane":
        raise ValueError("deploy terrain requires an asset block and named floor plane")
    if root.findall("./asset/hfield"):
        raise ValueError("deploy terrain expects a source without existing heightfields")
    ET.SubElement(asset, "hfield", {
        "name": "deploy_terrain",
        "nrow": str(config.grid_size), "ncol": str(config.grid_size),
        "size": (f"{config.half_size_m} {config.half_size_m} "
                 f"{config.max_height_m} {config.base_depth_m}"),
    })
    # Sample data is supplied directly to MjModel before conversion to MJX.
    # Identity placement makes both collision and reward lookup use world xy.
    for name in ("size", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
        floor.attrib.pop(name, None)
    floor.set("type", "hfield")
    floor.set("hfield", "deploy_terrain")
    floor.set("pos", "0 0 0")
    floor.set("contype", "1")
    floor.set("conaffinity", "0")
    return ET.tostring(root, encoding="unicode")


def terrain_data(xp, noise, height_m, config):
    """Smooth independent 2-D noise, then flatten the spawn pad and edges.

    Returned normalized samples use the same fixed vertical scale for all
    environments. A flat environment passes height_m=0, retaining identical
    collision topology and tensor shapes for vmap.
    """
    # A separable binomial filter makes gentle bumps instead of sharp spikes.
    smooth = (xp.roll(noise, 1, axis=0) + 2.0 * noise
              + xp.roll(noise, -1, axis=0)) / 4.0
    smooth = (xp.roll(smooth, 1, axis=1) + 2.0 * smooth
              + xp.roll(smooth, -1, axis=1)) / 4.0
    normalized = (smooth - xp.min(smooth)) / xp.maximum(
        xp.max(smooth) - xp.min(smooth), 1e-6)
    axis = xp.linspace(-config.half_size_m, config.half_size_m, config.grid_size)
    xx, yy = xp.meshgrid(axis, axis, indexing="xy")
    radius = xp.sqrt(xx * xx + yy * yy)
    gate = xp.clip((radius - config.spawn_radius_m) / config.transition_m, 0., 1.)
    gate = gate * gate * (3.0 - 2.0 * gate)
    edge = xp.clip((config.half_size_m - xp.maximum(xp.abs(xx), xp.abs(yy)))
                   / config.edge_margin_m, 0., 1.)
    edge = edge * edge * (3.0 - 2.0 * edge)
    return (normalized * gate * edge * (height_m / config.max_height_m)).reshape(-1)


def reference_terrain_data(config):
    """Deterministic rough field for unwrapped evaluation/video rendering."""
    config.validate()
    noise = np.random.default_rng(config.seed).uniform(
        size=(config.grid_size, config.grid_size)).astype(np.float32)
    return terrain_data(np, noise, config.max_height_m, config).astype(np.float32)


def surface_height(xp, xy, data, config):
    """Vertical surface height matching MuJoCo's two triangles per grid cell.

    The diagonal joins (x0,y0) to (x1,y1). Bilinear interpolation is deliberately
    avoided because it would give a different surface within the cell.
    Queries outside the field clamp to the edge; the environment terminates
    before the robot can reach that edge.
    """
    grid = data.reshape((config.grid_size, config.grid_size))
    uv = xp.clip((xy + config.half_size_m) * ((config.grid_size - 1)
                 / (2.0 * config.half_size_m)), 0., config.grid_size - 1.)
    ij = xp.minimum(xp.floor(uv).astype(xp.int32), config.grid_size - 2)
    frac = uv - ij
    ix, iy = ij[..., 0], ij[..., 1]
    fx, fy = frac[..., 0], frac[..., 1]
    z00, z10 = grid[iy, ix], grid[iy, ix + 1]
    z01, z11 = grid[iy + 1, ix], grid[iy + 1, ix + 1]
    lower = z00 + fx * (z10 - z00) + fy * (z11 - z10)
    upper = z00 + fx * (z11 - z01) + fy * (z01 - z00)
    return config.max_height_m * xp.where(fy <= fx, lower, upper)

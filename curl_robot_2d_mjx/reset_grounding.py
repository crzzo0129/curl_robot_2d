"""Geometric floor clearance, shared by CPU collection and MJX reset.

Mesh support uses compiled vertices in the geom frame (also exact for the
convex collision hull in a plane-normal direction). No dynamics settling.
"""
import numpy as np


def make_floor_clearance(model, floor_id, xp):
    import mujoco
    if (int(model.geom_type[floor_id]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
            or model.geom_bodyid[floor_id] != 0
            or not np.allclose(model.geom_quat[floor_id], [1, 0, 0, 0])):
        raise ValueError("Reset grounding requires a horizontal world-body floor plane")
    explicit = set()
    for a, b in zip(model.pair_geom1, model.pair_geom2):
        if a == floor_id:
            explicit.add(int(b))
        if b == floor_id:
            explicit.add(int(a))
    specs = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        collides = ((int(model.geom_contype[g]) & int(model.geom_conaffinity[floor_id]))
                    or (int(model.geom_contype[floor_id]) & int(model.geom_conaffinity[g])))
        if not collides and g not in explicit:
            continue
        kind = int(model.geom_type[g])
        vertices = None
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = xp.asarray(np.array(model.mesh_vert[start:start + count], copy=True))
        elif kind not in (2, 3, 4, 5, 6):
            raise ValueError(f"Unsupported floor collision geom type {kind}")
        specs.append((g, kind, xp.asarray(np.array(model.geom_size[g], copy=True)), vertices))
    if not specs:
        raise ValueError("No robot geometries collide with floor")

    def clearance(data):
        lowest = []
        for g, kind, size, vertices in specs:
            row = data.geom_xmat[g].reshape(3, 3)[2]
            if vertices is not None:
                bottom = xp.min(vertices @ row)
            elif kind == 2:  # sphere
                bottom = -size[0]
            elif kind == 3:  # capsule
                bottom = -size[0] - size[1] * xp.abs(row[2])
            elif kind == 4:  # ellipsoid
                bottom = -xp.linalg.norm(row * size)
            elif kind == 5:  # cylinder
                bottom = -size[0] * xp.linalg.norm(row[:2]) - size[1] * xp.abs(row[2])
            else:  # box
                bottom = -xp.sum(xp.abs(row) * size)
            lowest.append(data.geom_xpos[g, 2] + bottom)
        return xp.min(xp.stack(lowest)) - data.geom_xpos[floor_id, 2]
    return clearance

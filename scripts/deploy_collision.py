"""Walking-only native sphere contacts; XML helpers do not import JAX."""
import math
import xml.etree.ElementTree as ET


def foot_sphere_only_xml(xml, legs, radius):
    """Keep CAD for rendering and explicit inertias, with four sphere-floor pairs.

    The foot sites also define reward clearance, so contact and reward use
    exactly the same centers/radius. There is deliberately no torso contact.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("foot sphere radius must be finite and positive")
    legs = tuple(legs)
    if len(legs) != 4 or len(set(legs)) != 4:
        raise ValueError("four distinct leg names are required")
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    floor = root.find("./worldbody/geom[@name='floor']")
    torso = root.find("./worldbody/body[@name='torso']")
    if world is None or floor is None or torso is None:
        raise ValueError("walking collision model requires floor and torso")
    if floor.get("type") not in ("plane", "hfield"):
        raise ValueError("walking floor must be a plane or heightfield")
    for body in torso.iter("body"):
        if body.find("geom") is not None and body.find("inertial") is None:
            raise ValueError(f"explicit inertia required on {body.get('name')}")
    # Disable every explicit geom override, as well as inherited masks.
    for geom in list(root.findall(".//default/geom")) + list(world.iter("geom")):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    floor.set("contype", "1")
    # Explicit pairs bypass masks. No source pair should survive this mode.
    for contact in root.findall("contact"):
        for pair in list(contact.findall("pair")):
            contact.remove(pair)
    for leg in legs:
        body = torso.find(f".//body[@name='{leg}_shank']")
        site = None if body is None else body.find(f"site[@name='{leg}_foot_site']")
        if site is None or not site.get("pos"):
            raise ValueError(f"missing {leg} shank/foot site")
        name = f"{leg}_foot_collision"
        if world.find(f".//geom[@name='{name}']") is not None:
            raise ValueError(f"duplicate walking sphere: {name}")
        ET.SubElement(body, "geom", {
            "name": name, "type": "sphere", "pos": site.get("pos"),
            "size": str(radius), "contype": "0", "conaffinity": "1",
            "condim": "3", "mass": "0", "group": "3",
            "rgba": "1 0.45 0.05 0.6",
        })
    return ET.tostring(root, encoding="unicode")


def audit_foot_sphere_model(mj, legs, radius):
    """Check compiled masks, sphere centers and explicit pairs using CPU MuJoCo."""
    import numpy as np
    import mujoco

    def geom_id(name):
        value = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_GEOM, name)
        if value < 0:
            raise ValueError(f"missing collision geom: {name}")
        return value

    floor = geom_id("floor")
    spheres = [geom_id(f"{leg}_foot_collision") for leg in legs]
    expected_type = np.zeros(mj.ngeom, dtype=int)
    expected_affinity = np.zeros(mj.ngeom, dtype=int)
    expected_type[floor] = 1
    expected_affinity[spheres] = 1
    if (not np.array_equal(mj.geom_contype, expected_type)
            or not np.array_equal(mj.geom_conaffinity, expected_affinity)
            or mj.npair != 0):
        raise ValueError("only four sphere-floor collision pairs are allowed")
    for leg, gid in zip(legs, spheres):
        sid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
        if (sid < 0 or mj.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE
                or mj.geom_bodyid[gid] != mj.site_bodyid[sid]
                or not np.allclose(mj.geom_pos[gid], mj.site_pos[sid], atol=1e-10, rtol=0)
                or not np.isclose(mj.geom_size[gid, 0], radius, atol=1e-10, rtol=0)):
            raise ValueError(f"foot collision/reward geometry mismatch: {leg}")
    return {"mode": "foot-spheres", "radius_m": float(radius),
            "ground_pairs": 4, "robot_pairs": 0, "explicit_pairs": 0}


def main():
    """Generate a portable native MuJoCo model without importing the trainer."""
    import argparse
    import os
    from pathlib import Path
    import mujoco

    source = (Path(__file__).resolve().parents[1] / "assets" /
              "rollingquad_description_2" / "mjcf" / "rollingquad.xml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=source.with_name("rollingquad_walk_foot_spheres.xml"))
    args = parser.parse_args()
    legs = ("front_left", "front_right", "rear_left", "rear_right")
    root = ET.fromstring(foot_sphere_only_xml(source.read_text(encoding="utf-8"), legs, 0.0195))
    output = args.out.expanduser().resolve()
    if output == source:
        parser.error("output must not overwrite the source CAD model")
    compiler = root.find("compiler")
    mesh_dir = (source.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", os.path.relpath(mesh_dir, output.parent).replace("\\", "/"))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    mj = mujoco.MjModel.from_xml_path(str(output))
    print(audit_foot_sphere_model(mj, legs, 0.0195))
    print(output)


if __name__ == "__main__":
    main()

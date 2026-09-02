"""Build a primitive-collision RollingQuad MJCF from the CAD mesh model.

Keeps joints, masses, inertias, keyframes, sites, actuators and sensors
verbatim, and replaces every CAD mesh collision geom with MuJoCo analytic
primitives:

- shell   -> capsule arc (300 deg: torso 150 deg + thigh 45 deg + shank 30 deg
             per side, with a 60 deg opening at the bottom), matching
             curl_robot_3d_pupper_r127p5_open60_width120.xml.
- hip link-> small cylinder (abduction motor mount)
- thigh   -> cylinder (hip motor) + capsule (upper leg)
- shank+foot -> capsule (lower leg) + sphere (foot)

The shell is distributed across the torso and the four legs so the rolling
surface follows the folded body, exactly like the analytic shell model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

from curl_robot_2d.pupper_shell_geometry import (
    PupperShellDesign,
    solve_compact_geometry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad.xml"
)
DEFAULT_OUTPUT_XML = DEFAULT_XML.with_name("rollingquad_primitive.xml")

MOTOR_RADIUS = 0.030           # cylinder radius: 60 mm diameter
MOTOR_HALF_LENGTH = 0.0165     # motor_half_thickness_y
FOOT_RADIUS = 0.0195           # foot_radius
HIP_CYLINDER_HALF_LENGTH = 0.0315
THIGH_LINK_RADIUS = 0.023      # upper-leg half-thickness
SHANK_LINK_RADIUS = 0.0195     # lower-leg half-thickness (matches foot)
SHELL_HALF_WIDTH = 0.060       # side_rail_half_width_override
SHELL_CONTYPE = 16             # shell collides as the torso category
SHELL_CONAFFINITY = 7
FOOT_SHELL_CLEARANCE = 0.010   # pupper_shank_shell_foot_clearance
TORSO_BOX_HALF = (0.060, 0.060, 0.04507)  # torso_box_height / 2
TORSO_BOX_POS = (0.0, 0.0, -0.04507)

# Quaternion mapping MuJoCo's local +z cylinder axis onto the body +y axis.
AXIS_Y_QUAT = "0.7071067811865476 0.7071067811865476 0 0"
# Quaternion rotating 90 degrees about +y: maps the local +z axis onto +x.
AXIS_X_QUAT = "0.7071067811865476 0 0.7071067811865476 0"


def _load_stl(path: Path) -> np.ndarray:
    """Return de-duplicated finite STL vertices (binary or ASCII)."""
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if 84 + triangle_count * 50 == len(payload):
            record_dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            )
            records = np.frombuffer(
                payload, dtype=record_dtype, count=triangle_count, offset=84
            )
            vertices = np.asarray(records["vertices"], dtype=np.float64).reshape(-1, 3)
        else:
            vertices = _load_ascii_stl(payload, path)
    else:
        vertices = _load_ascii_stl(payload, path)
    vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
    return np.unique(vertices, axis=0)


def _load_ascii_stl(payload: bytes, path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    text = payload.decode("utf-8", errors="replace")
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    if not vertices:
        raise ValueError(f"No STL vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def _parse_xml(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{float(x):.8g}" for x in v)


def _kind(mesh_name: str) -> str:
    lowered = mesh_name.lower()
    if "torso" in lowered:
        return "torso"
    if "abduction" in lowered:
        return "hip"
    if "upperleg" in lowered:
        return "thigh"
    if "lower_leg" in lowered:
        return "foot"
    raise ValueError(f"unknown mesh kind: {mesh_name}")


def _primitive_specs(mesh_name: str, lo: np.ndarray, hi: np.ndarray, ctr: np.ndarray,
                     foot_site: np.ndarray | None, knee_pos: np.ndarray | None) -> list[dict]:
    kind = _kind(mesh_name)
    if kind == "torso":
        return []  # replaced by the shell arc
    if kind == "hip":
        return [
            {
                "type": "cylinder",
                "size": f"{MOTOR_RADIUS:.8g} {HIP_CYLINDER_HALF_LENGTH:.8g}",
                "pos": _fmt(ctr),
                "quat": AXIS_X_QUAT,
            }
        ]
    if kind == "thigh":
        motor_center = np.asarray([ctr[0], hi[1] - MOTOR_HALF_LENGTH, ctr[2]])
        motor_bottom = np.asarray([ctr[0], hi[1] - 2.0 * MOTOR_HALF_LENGTH, ctr[2]])
        knee = np.asarray(knee_pos)
        return [
            {
                "type": "capsule",
                "size": f"{THIGH_LINK_RADIUS:.8g}",
                "fromto": f"{_fmt(motor_bottom)} {_fmt(knee)}",
            },
            {
                "type": "cylinder",
                "size": f"{MOTOR_RADIUS:.8g} {MOTOR_HALF_LENGTH:.8g}",
                "pos": _fmt(motor_center),
                "quat": AXIS_X_QUAT,
            },
        ]
    # foot: shank capsule + foot sphere
    foot_center = np.asarray(foot_site)
    foot_top = np.asarray([foot_site[0], foot_site[1] + FOOT_RADIUS, foot_site[2]])
    knee = np.asarray([ctr[0], hi[1], ctr[2]])
    return [
        {
            "type": "sphere",
            "size": f"{FOOT_RADIUS:.8g}",
            "pos": _fmt(foot_center),
        },
        {
            "type": "capsule",
            "size": f"{SHANK_LINK_RADIUS:.8g}",
            "fromto": f"{_fmt(knee)} {_fmt(foot_top)}",
        },
    ]


def _sibling_name(original_name: str, kind: str) -> str:
    if kind == "thigh":
        return original_name.removesuffix("_geom") + "_motor"
    if kind == "foot":
        return original_name.replace("_foot_proxy", "_shank_geom")
    raise ValueError(f"no sibling for {kind}")


def _foot_site_pos(tree: ET.ElementTree) -> dict[str, np.ndarray]:
    sites: dict[str, np.ndarray] = {}
    for site in tree.getroot().iter("site"):
        name = site.get("name", "")
        pos = site.get("pos")
        if name.endswith("_foot_site") and pos:
            sites[name] = np.asarray([float(x) for x in pos.split()], dtype=np.float64)
    return sites


def _mesh_asset_files(tree: ET.ElementTree, xml_path: Path) -> dict[str, Path]:
    asset = tree.getroot().find("asset")
    mapping: dict[str, Path] = {}
    if asset is None:
        return mapping
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        file_ = mesh.get("file")
        if name and file_:
            mapping[name] = (xml_path.parent / file_).resolve()
    return mapping


def _angular_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dz = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dz * dz
    if denominator == 0.0:
        return math.dist(point, start)
    fraction = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz)
            / denominator,
        ),
    )
    closest = (start[0] + fraction * dx, start[1] + fraction * dz)
    return math.dist(point, closest)


def _shell_capsules(model, data) -> list[dict]:
    """Return capsule specs for the shell arc, one per body local frame.

    Replicates model._pupper_shell_geoms' explicit 150/45/30 degree split (with
    a 60 degree foot opening) and folds each segment into the CAD body's local
    frame using the compact keyframe pose.
    """

    design = PupperShellDesign()
    sol = solve_compact_geometry(design)
    radius = design.shell_centerline_radius
    capsule_radius = design.shell_capsule_radius
    center = np.asarray([0.0, 0.0, -sol.shell_center_below_hip])  # torso frame
    front_foot = (sol.foot_x, -sol.foot_below_hip)
    rear_foot = (-sol.foot_x, -sol.foot_below_hip)
    torso_half = math.radians(150.0) / 2.0

    poses = {
        model.body(bid).name: (data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy())
        for bid in range(model.nbody)
    }
    torso_pos, torso_mat = poses["torso"]

    def world_point(x: float, z: float, y: float) -> np.ndarray:
        return torso_pos + torso_mat @ np.asarray([x, y, z])

    def to_local(body_name: str, world: np.ndarray) -> np.ndarray:
        pos, mat = poses[body_name]
        return mat.T @ (world - pos)

    leg_map = {
        "front_thigh": ("front_left_thigh", "front_right_thigh"),
        "front_shank": ("front_left_shank", "front_right_shank"),
        "rear_thigh": ("rear_left_thigh", "rear_right_thigh"),
        "rear_shank": ("rear_left_shank", "rear_right_shank"),
    }

    capsules: list[dict] = []
    for index in range(design.shell_segments):
        angle_a = 2.0 * math.pi * index / design.shell_segments
        angle_b = 2.0 * math.pi * (index + 1) / design.shell_segments
        middle = 0.5 * (angle_a + angle_b)
        angle = middle % (2.0 * math.pi)
        torso_start = math.pi / 2.0 - torso_half
        torso_end = math.pi / 2.0 + torso_half
        if torso_start <= angle < torso_end:
            region = "torso"
        elif angle < torso_start or angle >= 11.0 * math.pi / 6.0:
            region = "front_thigh"
        elif angle < 7.0 * math.pi / 6.0:
            region = "rear_thigh"
        elif angle < 4.0 * math.pi / 3.0:
            region = "rear_shank"
        elif angle >= 5.0 * math.pi / 3.0:
            region = "front_shank"
        else:
            region = "rear_shank" if angle < 3.0 * math.pi / 2.0 else "front_shank"

        xa = center[0] + radius * math.cos(angle_a)
        za = center[2] + radius * math.sin(angle_a)
        xb = center[0] + radius * math.cos(angle_b)
        zb = center[2] + radius * math.sin(angle_b)

        # Remove the near-foot opening (60 deg): segments whose centreline runs
        # too close to a foot are dropped, matching the analytic model.
        same_foot = {
            "front_shank": front_foot,
            "rear_shank": rear_foot,
        }.get(region)
        if same_foot is not None and _point_segment_distance(
            same_foot, (xa, za), (xb, zb)
        ) < (FOOT_RADIUS + capsule_radius + FOOT_SHELL_CLEARANCE):
            continue

        if region == "torso":
            for rail_y in (-SHELL_HALF_WIDTH, 0.0, SHELL_HALF_WIDTH):
                capsules.append({
                    "body": "torso",
                    "fromto": np.array([xa, rail_y, za, xb, rail_y, zb]),
                })
        else:
            for body_name in leg_map[region]:
                pos, _ = poses[body_name]
                world_y = float(pos[1])
                a_local = to_local(body_name, world_point(xa, za, world_y))
                b_local = to_local(body_name, world_point(xb, zb, world_y))
                capsules.append({
                    "body": body_name,
                    "fromto": np.array([*a_local, *b_local]),
                })

    for capsule in capsules:
        capsule["size"] = f"{capsule_radius:.8g}"
    return capsules


def build(args: argparse.Namespace) -> dict[str, object]:
    input_xml = args.input.resolve()
    output_xml = args.output.resolve()
    if input_xml == output_xml:
        raise ValueError("Output MJCF must not overwrite the source MJCF")
    if not input_xml.is_file():
        raise FileNotFoundError(input_xml)
    if output_xml.exists() and not args.force:
        raise FileExistsError(f"{output_xml} already exists; pass --force to replace")

    tree = _parse_xml(input_xml)
    root = tree.getroot()
    mesh_files = _mesh_asset_files(tree, input_xml)
    foot_sites = _foot_site_pos(tree)
    body_pos = {
        body.get("name"): np.asarray(
            [float(x) for x in body.get("pos").split()], dtype=np.float64
        )
        for body in root.iter("body")
        if body.get("name") and body.get("pos")
    }

    # Compute the shell arc from the compact keyframe via MuJoCo FK.
    import mujoco

    shell_model = mujoco.MjModel.from_xml_path(str(input_xml))
    shell_data = mujoco.MjData(shell_model)
    mujoco.mj_resetDataKeyframe(shell_model, shell_data, shell_model.key("compact").id)
    mujoco.mj_forward(shell_model, shell_data)
    shell_capsules = _shell_capsules(shell_model, shell_data)

    collision_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.get("type") == "mesh"
        and geom.get("mesh") in mesh_files
        and not (geom.get("contype") == "0" and geom.get("conaffinity") == "0")
    ]

    occupied_geom_names = {
        geom.get("name") for geom in root.iter("geom") if geom.get("name")
    }
    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    body_elements = {
        body.get("name"): body for body in root.iter("body") if body.get("name")
    }
    report_geoms: dict[str, object] = {}

    for geom in collision_geoms:
        mesh_name = geom.get("mesh")
        geom_name = geom.get("name") or mesh_name
        contype = geom.get("contype")
        conaffinity = geom.get("conaffinity")
        rgba = geom.get("rgba")
        kind = _kind(mesh_name)

        if kind == "torso":
            # Remove the torso mesh; the shell arc replaces it.
            parent = parent_by_child[geom]
            parent.remove(geom)
            report_geoms[geom_name] = {"mesh": mesh_name, "kind": "torso", "primitives": []}
            continue

        vertices = _load_stl(mesh_files[mesh_name])
        lo = vertices.min(axis=0)
        hi = vertices.max(axis=0)
        ctr = vertices.mean(axis=0)

        foot_site = None
        knee_pos = None
        if kind == "foot":
            site_name = geom_name.replace("_foot_proxy", "_foot_site")
            foot_site = foot_sites.get(site_name)
            if foot_site is None:
                raise ValueError(f"missing foot site for {geom_name}")
        elif kind == "thigh":
            prefix = geom_name.removesuffix("_thigh_geom")
            knee_pos = body_pos.get(f"{prefix}_shank")
            if knee_pos is None:
                raise ValueError(f"missing shank body for {geom_name}")

        specs = _primitive_specs(mesh_name, lo, hi, ctr, foot_site, knee_pos)

        first = specs[0]
        for key in ("mesh", "type", "size", "pos", "quat", "fromto"):
            if key in geom.attrib:
                del geom.attrib[key]
        geom.set("type", first["type"])
        geom.set("size", first["size"])
        for key in ("pos", "quat", "fromto"):
            if key in first:
                geom.set(key, first[key])

        for spec in specs[1:]:
            sibling = ET.Element("geom")
            sibling.set("name", _sibling_name(geom_name, kind))
            sibling.set("type", spec["type"])
            sibling.set("size", spec["size"])
            for key in ("pos", "quat", "fromto"):
                if key in spec:
                    sibling.set(key, spec[key])
            if contype is not None:
                sibling.set("contype", contype)
            if conaffinity is not None:
                sibling.set("conaffinity", conaffinity)
            if rgba is not None:
                sibling.set("rgba", rgba)
            parent = parent_by_child[geom]
            index = list(parent).index(geom)
            parent.insert(index + 1, sibling)

        report_geoms[geom_name] = {
            "mesh": mesh_name,
            "kind": kind,
            "primitives": [
                {k: v for k, v in spec.items() if k != "fromto"}
                | ({"fromto": spec["fromto"]} if "fromto" in spec else {})
                for spec in specs
            ],
        }

    # Append the shell capsule geoms to their bodies.
    shell_counts: dict[str, int] = {}
    for capsule in shell_capsules:
        body_name = capsule["body"]
        parent = body_elements[body_name]
        shell_counts[body_name] = shell_counts.get(body_name, 0) + 1
        name = f"{body_name}_shell_{shell_counts[body_name]:02d}"
        geom = ET.Element("geom")
        geom.set("name", name)
        geom.set("type", "capsule")
        geom.set("size", capsule["size"])
        geom.set("fromto", _fmt(capsule["fromto"]))
        geom.set("contype", str(SHELL_CONTYPE))
        geom.set("conaffinity", str(SHELL_CONAFFINITY))
        geom.set("rgba", "0.55 0.78 0.95 0.60")
        parent.append(geom)

    # Add a solid torso core (the "body"), colliding as the torso category.
    # MJX does not implement cylinder-vs-box collision, so the core is an
    # ellipsoid (a rounded box) rather than a box; ellipsoid-vs-cylinder and
    # ellipsoid-vs-capsule are both supported.
    torso_box = ET.Element("geom")
    torso_box.set("name", "torso_box_proxy")
    torso_box.set("type", "ellipsoid")
    torso_box.set("pos", " ".join(f"{v:.8g}" for v in TORSO_BOX_POS))
    torso_box.set("size", " ".join(f"{v:.8g}" for v in TORSO_BOX_HALF))
    torso_box.set("contype", str(SHELL_CONTYPE))
    torso_box.set("conaffinity", str(SHELL_CONAFFINITY))
    torso_box.set("rgba", "0.30 0.32 0.36 1")
    body_elements["torso"].append(torso_box)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)

    report: dict[str, object] = {
        "source_xml": str(input_xml),
        "output_xml": str(output_xml),
        "shell_outer_radius": PupperShellDesign().shell_outer_radius,
        "shell_capsule_radius": PupperShellDesign().shell_capsule_radius,
        "shell_segments": PupperShellDesign().shell_segments,
        "shell_geoms": len(shell_capsules),
        "shell_counts": shell_counts,
        "leg_geoms": report_geoms,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.skip_compile_check:
        try:
            import mujoco as _mj
        except ImportError:
            print("MuJoCo is not installed; skipped MJCF compile check.", file=sys.stderr)
        else:
            check = _mj.MjModel.from_xml_path(str(output_xml))
            print(
                f"MuJoCo compile check passed: ngeom={check.ngeom}, "
                f"nbody={check.nbody}, nu={check.nu}, npair={check.npair}"
            )

    print(f"Wrote MJCF: {output_xml}")
    print(f"Wrote report: {args.report.resolve()}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_XML, help="Source MJCF")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_XML, help="Generated MJCF")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_XML.with_suffix(".report.json"),
        help="JSON primitive-layout report",
    )
    parser.add_argument("--skip-compile-check", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing outputs")
    return parser


def main() -> None:
    args = _parser().parse_args()
    build(args)


if __name__ == "__main__":
    main()

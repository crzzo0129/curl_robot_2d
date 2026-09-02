"""Build a low-vertex, one-convex-hull-per-link RollingQuad MJCF.

The source STL meshes are not modified.  For every mesh geom that can take
part in collision, this script:

1. computes the exact convex hull of the STL vertices;
2. selects a bounded set of support vertices over many directions;
3. writes a triangular OBJ convex hull; and
4. writes a new MJCF whose collision geom keeps its original name and masks.

By default the output renders and collides with only the simplified hulls.  This
keeps the existing training model's exact geom-name contract.  The optional
``--visual-mode original`` retains the original STL as a non-colliding visual
geom, but that preview model has extra geoms and is therefore not accepted by
the current RollingQuad training contract validator.

This is deliberately a *single* convex hull per source mesh.  It cannot retain
concavities.  The JSON report quantifies the support-surface loss so that the
result can be rejected before an expensive training run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    from scipy.spatial import ConvexHull, QhullError
except ImportError as exc:  # pragma: no cover - exercised by the user environment
    raise SystemExit(
        "This script requires SciPy. Install it with: python -m pip install scipy"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    PROJECT_ROOT
    / "assets"
    / "rollingquad_description_2"
    / "mjcf"
    / "rollingquad.xml"
)
DEFAULT_OUTPUT_XML = DEFAULT_XML.with_name("rollingquad_simple_convex.xml")
DEFAULT_MESH_DIR = DEFAULT_XML.parent.parent / "meshes_simple_convex"


def _load_stl(path: Path) -> np.ndarray:
    """Returns de-duplicated STL vertices for binary or ASCII STL files."""
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected_size = 84 + triangle_count * 50
        if expected_size == len(payload):
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
    vertices = np.unique(vertices, axis=0)
    if len(vertices) < 4:
        raise ValueError(f"{path} contains fewer than four valid unique vertices")
    return vertices


def _load_ascii_stl(payload: bytes, path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not identify binary or ASCII STL: {path}") from exc
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    if not vertices:
        raise ValueError(f"No STL vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def _fibonacci_directions(count: int) -> np.ndarray:
    """Deterministic, approximately uniform directions on a unit sphere."""
    index = np.arange(count, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    directions = np.column_stack(
        (radius * np.cos(golden_angle * index), radius * np.sin(golden_angle * index), z)
    )
    axes = np.vstack((np.eye(3), -np.eye(3)))
    return np.vstack((axes, directions))


def _support(vertices: np.ndarray, directions: np.ndarray, chunk: int = 256):
    values = np.empty(len(directions), dtype=np.float64)
    indices = np.empty(len(directions), dtype=np.int64)
    for start in range(0, len(directions), chunk):
        stop = min(start + chunk, len(directions))
        dots = vertices @ directions[start:stop].T
        local_indices = np.argmax(dots, axis=0)
        indices[start:stop] = local_indices
        values[start:stop] = dots[local_indices, np.arange(stop - start)]
    return values, indices


def _simplified_hull(
    vertices: np.ndarray, max_vertices: int, direction_count: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    try:
        exact = ConvexHull(vertices)
    except QhullError as exc:
        raise ValueError("Input mesh does not span a valid 3-D convex hull") from exc

    exact_vertices = vertices[np.asarray(exact.vertices)]
    directions = _fibonacci_directions(direction_count)
    exact_support, support_indices = _support(exact_vertices, directions)

    if len(exact_vertices) <= max_vertices:
        selected = exact_vertices
    else:
        # Preserve all six axis extrema first.  Greedily add the source vertex
        # responsible for the largest remaining support-function error.
        axis_indices = support_indices[:6]
        chosen: list[int] = list(dict.fromkeys(int(i) for i in axis_indices))
        chosen_set = set(chosen)

        while len(chosen) < max_vertices:
            current_support = np.max(exact_vertices[chosen] @ directions.T, axis=0)
            gaps = exact_support - current_support
            added = False
            for direction_index in np.argsort(gaps)[::-1]:
                vertex_index = int(support_indices[direction_index])
                if vertex_index not in chosen_set:
                    chosen.append(vertex_index)
                    chosen_set.add(vertex_index)
                    added = True
                    break
            if not added:
                break
        selected = exact_vertices[chosen]

    try:
        simple = ConvexHull(selected)
    except QhullError as exc:
        raise ValueError("Selected support points do not span a valid 3-D hull") from exc

    # Drop any selected point that became interior and remap triangle indices.
    used = np.asarray(simple.vertices, dtype=np.int64)
    remap = np.full(len(selected), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    faces = remap[np.asarray(simple.simplices, dtype=np.int64)]
    simple_vertices = selected[used]

    simple_support = np.max(simple_vertices @ directions.T, axis=0)
    loss = np.maximum(0.0, exact_support - simple_support)
    stats: dict[str, float | int] = {
        "source_unique_vertices": int(len(vertices)),
        "exact_hull_vertices": int(len(exact_vertices)),
        "simple_hull_vertices": int(len(simple_vertices)),
        "simple_hull_triangles": int(len(faces)),
        "support_loss_mean_mesh_units": float(np.mean(loss)),
        "support_loss_p95_mesh_units": float(np.percentile(loss, 95.0)),
        "support_loss_max_mesh_units": float(np.max(loss)),
    }
    return simple_vertices, faces, stats


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = ["# Generated by scripts.build_simple_convex_collision\n"]
    lines.extend(f"v {x:.10g} {y:.10g} {z:.10g}\n" for x, y, z in vertices)
    lines.extend(
        f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n" for a, b, c in faces
    )
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def _unique_name(base: str, occupied: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in occupied:
        candidate = f"{base}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def _relative_xml_path(path: Path, xml_directory: Path) -> str:
    return os.path.relpath(path, xml_directory).replace(os.sep, "/")


def _mesh_budget(name: str, args: argparse.Namespace) -> int:
    lowered = name.lower()
    if "lower_leg" in lowered or "foot" in lowered:
        return args.foot_max_vertices
    if "torso" in lowered:
        return args.torso_max_vertices
    return args.max_vertices


def _parse_xml(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser=parser)


def build(args: argparse.Namespace) -> dict[str, object]:
    input_xml = args.input.resolve()
    output_xml = args.output.resolve()
    mesh_directory = args.mesh_dir.resolve()

    if input_xml == output_xml:
        raise ValueError("Output MJCF must not overwrite the source MJCF")
    if not input_xml.is_file():
        raise FileNotFoundError(input_xml)
    if not args.force:
        existing = [path for path in (output_xml, args.report.resolve()) if path.exists()]
        if existing:
            joined = ", ".join(str(path) for path in existing)
            raise FileExistsError(f"Output already exists: {joined}; pass --force to replace")

    tree = _parse_xml(input_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MJCF has no <asset> section")

    mesh_assets = {
        element.get("name"): element
        for element in asset.findall("mesh")
        if element.get("name") and element.get("file")
    }
    collision_geoms = [
        geom
        for geom in root.iter("geom")
        if geom.get("type") == "mesh"
        and geom.get("mesh") in mesh_assets
        and not (geom.get("contype") == "0" and geom.get("conaffinity") == "0")
    ]
    if not collision_geoms:
        raise ValueError("No colliding mesh geoms were found")

    required_meshes = list(dict.fromkeys(geom.get("mesh") for geom in collision_geoms))
    mesh_directory.mkdir(parents=True, exist_ok=True)
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    occupied_mesh_names = {name for name in mesh_assets if name is not None}
    replacement_names: dict[str, str] = {}
    generated_paths: dict[str, Path] = {}
    mesh_reports: dict[str, object] = {}

    for source_name in required_meshes:
        source_asset = mesh_assets[source_name]
        source_file = (input_xml.parent / source_asset.get("file", "")).resolve()
        if source_file.suffix.lower() != ".stl":
            raise ValueError(f"Only STL input is supported, got {source_file}")

        budget = _mesh_budget(source_name, args)
        vertices = _load_stl(source_file)
        simple_vertices, faces, stats = _simplified_hull(
            vertices, max_vertices=budget, direction_count=args.directions
        )

        output_mesh = mesh_directory / f"{source_file.stem}_simple_hull.obj"
        if output_mesh.exists() and not args.force:
            raise FileExistsError(f"Output mesh already exists: {output_mesh}")
        _write_obj(output_mesh, simple_vertices, faces)

        hull_name = _unique_name(f"{source_name}__simple_hull", occupied_mesh_names)
        hull_asset = ET.Element("mesh")
        hull_asset.set("name", hull_name)
        hull_asset.set("file", _relative_xml_path(output_mesh, output_xml.parent))
        for key, value in source_asset.attrib.items():
            if key not in {"name", "file", "content_type", "maxhullvert"}:
                hull_asset.set(key, value)
        asset.append(hull_asset)

        replacement_names[source_name] = hull_name
        generated_paths[source_name] = output_mesh
        mesh_reports[source_name] = {
            "source_file": str(source_file),
            "output_file": str(output_mesh),
            "vertex_budget": budget,
            **stats,
        }
        print(
            f"{source_name}: exact={stats['exact_hull_vertices']} -> "
            f"simple={stats['simple_hull_vertices']} vertices, "
            f"max support loss={stats['support_loss_max_mesh_units']:.6g} mesh units"
        )

    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    occupied_geom_names = {
        geom.get("name") for geom in root.iter("geom") if geom.get("name")
    }
    for geom in collision_geoms:
        source_name = geom.get("mesh")
        if args.visual_mode == "original":
            parent = parent_by_child[geom]
            visual = copy.deepcopy(geom)
            original_name = geom.get("name", f"{source_name}_geom")
            visual.set("name", _unique_name(f"{original_name}_visual", occupied_geom_names))
            visual.set("contype", "0")
            visual.set("conaffinity", "0")
            visual.set("group", "2")
            parent.insert(list(parent).index(geom), visual)
            geom.set("rgba", "0 0 0 0")
            geom.set("group", "3")
        geom.set("mesh", replacement_names[source_name])

    # The output may be placed outside the source MJCF directory.  Preserve
    # every existing file-backed asset by rewriting its path relative to the
    # new XML.  Newly generated hull assets are already output-relative.
    generated_asset_names = set(replacement_names.values())
    for element in asset:
        file_name = element.get("file")
        if not file_name or element.get("name") in generated_asset_names:
            continue
        source_path = (input_xml.parent / file_name).resolve()
        element.set("file", _relative_xml_path(source_path, output_xml.parent))

    if args.visual_mode == "hull":
        used_mesh_names = {
            geom.get("mesh") for geom in root.iter("geom") if geom.get("mesh")
        }
        for source_name in required_meshes:
            source_asset = mesh_assets[source_name]
            if source_name not in used_mesh_names:
                asset.remove(source_asset)

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)

    report: dict[str, object] = {
        "source_xml": str(input_xml),
        "output_xml": str(output_xml),
        "visual_mode": args.visual_mode,
        "directions": args.directions,
        "default_max_vertices": args.max_vertices,
        "foot_max_vertices": args.foot_max_vertices,
        "torso_max_vertices": args.torso_max_vertices,
        "note": (
            "Support loss is reported in raw mesh units. The RollingQuad STL files "
            "use metres, so multiply by 1000 to read millimetres."
        ),
        "meshes": mesh_reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.skip_compile_check:
        try:
            import mujoco
        except ImportError:
            print("MuJoCo is not installed; skipped MJCF compile check.", file=sys.stderr)
        else:
            model = mujoco.MjModel.from_xml_path(str(output_xml))
            print(
                f"MuJoCo compile check passed: ngeom={model.ngeom}, "
                f"nmesh={model.nmesh}, npair={model.npair}"
            )

    print(f"Wrote MJCF: {output_xml}")
    print(f"Wrote report: {args.report.resolve()}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_XML, help="Source MJCF")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_XML, help="Generated MJCF"
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help="Directory for generated OBJ convex hulls",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_XML.with_suffix(".report.json"),
        help="JSON geometry-error report",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=96,
        help="Vertex budget for hip and thigh meshes (default: 96)",
    )
    parser.add_argument(
        "--foot-max-vertices",
        type=int,
        default=192,
        help="Vertex budget for lower-leg/foot meshes (default: 192)",
    )
    parser.add_argument(
        "--torso-max-vertices",
        type=int,
        default=128,
        help="Vertex budget for the torso mesh (default: 128)",
    )
    parser.add_argument(
        "--directions",
        type=int,
        default=4096,
        help="Support directions used for simplification and error measurement",
    )
    parser.add_argument(
        "--visual-mode",
        choices=("original", "hull"),
        default="hull",
        help=(
            "Render/load only the hull meshes (training default), or keep original "
            "STL visuals in a preview model with extra non-colliding geoms"
        ),
    )
    parser.add_argument("--skip-compile-check", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace existing outputs")
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in ("max_vertices", "foot_max_vertices", "torso_max_vertices"):
        if getattr(args, name) < 4:
            raise SystemExit(f"--{name.replace('_', '-')} must be at least 4")
    if args.directions < 64:
        raise SystemExit("--directions must be at least 64")
    build(args)


if __name__ == "__main__":
    main()

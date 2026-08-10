"""Render static inspection views of the real-geometry 3-D candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets/curl_robot_3d_real_geometry.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "renders/real_geometry_3d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keyframe",
        choices=("open", "stand", "park", "compact", "all"),
        default="all",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = model.body("torso").id
    camera.azimuth = 135.0
    camera.elevation = -18.0
    camera.distance = 0.9
    scene_option = mujoco.MjvOption()

    names = (
        ("open", "stand", "park", "compact")
        if args.keyframe == "all"
        else (args.keyframe,)
    )
    for name in names:
        mujoco.mj_resetDataKeyframe(model, data, model.key(name).id)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        output = args.output_dir / f"{name}.png"
        Image.fromarray(renderer.render()).save(output)
        print(output)
    renderer.close()


if __name__ == "__main__":
    main()

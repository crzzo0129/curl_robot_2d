from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "renders"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyframe",
        choices=("open", "walk", "compact", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Overlay MuJoCo center-of-mass and contact diagnostics.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
    renderer = mujoco.Renderer(model, height=720, width=960)
    scene_option = mujoco.MjvOption()
    if args.diagnostics:
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    names = (
        ("open", "walk", "compact")
        if args.keyframe == "all"
        else (args.keyframe,)
    )
    for name in names:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
        pixels = renderer.render()
        output = args.output_dir / f"{name}.png"
        Image.fromarray(pixels).save(output)
        print(output)

    renderer.close()


if __name__ == "__main__":
    main()

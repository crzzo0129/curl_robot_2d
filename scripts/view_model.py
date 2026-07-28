from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyframe",
        choices=("open", "walk", "compact"),
        default="compact",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Step physics in real time. Without this flag the selected pose stays frozen.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("auto", "fixed", "track"),
        default="auto",
        help=(
            "Camera behavior. 'auto' tracks the torso while simulating and "
            "uses the fixed XML camera for a frozen pose."
        ),
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=0.75,
        help="Tracking-camera distance in meters; increase it to zoom out.",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        camera_mode = args.camera_mode
        if camera_mode == "auto":
            camera_mode = "track" if args.simulate else "fixed"

        if camera_mode == "track":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
            viewer.cam.azimuth = 90
            viewer.cam.elevation = 0
            viewer.cam.distance = args.camera_distance
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "side"
            )
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        while viewer.is_running():
            start = time.perf_counter()
            if args.simulate:
                mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()

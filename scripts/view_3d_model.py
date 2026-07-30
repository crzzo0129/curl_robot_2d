from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_3d.xml"


def parse_args(argv=None):
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
    parser.add_argument("--camera-distance", type=float, default=0.75)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-18.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = model.body("torso").id
        viewer.cam.azimuth = args.azimuth
        viewer.cam.elevation = args.elevation
        viewer.cam.distance = args.camera_distance
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

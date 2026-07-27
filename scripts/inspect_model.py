from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.parameters import FIXED_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "assets" / "curl_robot_2d.xml"


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, object_type, name)
    if value < 0:
        raise ValueError(f"Missing {object_type.name}: {name}")
    return value


def reset_keyframe(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    key_id = _id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe", choices=("open", "compact"), default="compact")
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    reset_keyframe(model, data, args.keyframe)

    foot_ids = [
        _id(model, mujoco.mjtObj.mjOBJ_SITE, "front_foot_site"),
        _id(model, mujoco.mjtObj.mjOBJ_SITE, "rear_foot_site"),
    ]
    torso_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    foot_positions = np.asarray(data.site_xpos)[foot_ids]
    subtree_com = np.asarray(data.subtree_com)[torso_id]
    print(f"model={MODEL_PATH}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"total_mass_expected={FIXED_PARAMETERS.total_mass:.6f}")
    print(f"total_mass_compiled={float(model.body_mass.sum()):.6f}")
    print(f"keyframe={args.keyframe} steps={args.steps}")
    print(f"qpos={np.array2string(data.qpos, precision=6)}")
    print(f"subtree_com={np.array2string(subtree_com, precision=6)}")
    print(f"front_foot={np.array2string(foot_positions[0], precision=6)}")
    print(f"rear_foot={np.array2string(foot_positions[1], precision=6)}")
    print(f"contacts={data.ncon}")


if __name__ == "__main__":
    main()

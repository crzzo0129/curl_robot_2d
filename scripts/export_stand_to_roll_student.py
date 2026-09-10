"""Export symmetry student inference params to standalone neural_controller RTNeural JSON.

Run on the cloud with the student and original BC normalizer. Input clipping
is represented exactly using a ReLU layer, requiring no custom C++ operator.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.export_rtneural import _load_checkpoint, _split_params, _dense_layers, _array, _write_json, _run_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-params", type=Path, required=True)
    parser.add_argument("--bc-params", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.student_params.parent / "student_source.json").read_text(encoding="utf-8"))
    if source["bc_sha256"] != hashlib.sha256(args.bc_params.read_bytes()).hexdigest():
        parser.error("BC normalizer differs from student training")
    if args.out.exists():
        parser.error("Output already exists; choose a new --out")
    from curl_robot_2d_mjx.config_stand_to_roll import StandToRollConfig
    from curl_robot_2d_mjx.stand_to_roll_training import action_center_and_scale, BC_CONTRACT_VERSION
    from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D, HARDWARE_CONTROLLER_JOINT_NAMES_3D
    from curl_robot_2d_mjx.environment_3d import model_path_3d
    import mujoco

    task = StandToRollConfig(**source["task"])
    norm, _ = _load_checkpoint(str(args.bc_params))
    _, actor = _split_params(_load_checkpoint(str(args.student_params)))
    if int(norm.get("contract_version", 0)) != BC_CONTRACT_VERSION:
        parser.error("Expected v2 BC normalizer")
    mean, std = _array(norm["mean"], "mean"), _array(norm["std"], "std")
    clip = float(norm["clip"])
    if (mean.shape != (720,) or std.shape != (720,) or np.any(std <= 0)
            or not np.isfinite(clip) or clip <= 0):
        parser.error("Invalid 720-channel clipped normalizer")
    dense = _dense_layers(actor)
    if dense[0][1].shape[0] != 720 or dense[-1][1].shape[1] != 24:
        parser.error("Expected 720-input, 24-logit tanh-normal student actor")

    def layer(kernel, bias, activation):
        return {"type": "dense", "shape": [1, len(bias)],
                "weights": [kernel.tolist(), bias.tolist()], "activation": activation}

    # clip(z,-c,c) = relu(z+c) - relu(z-c) - c.
    # Encode both branches in one dense/ReLU layer; merge their difference
    # and constant directly into the student's original first ELU layer.
    inv_std = np.diag(1.0 / std).astype(np.float32)
    layers = [layer(np.concatenate((inv_std, inv_std), axis=1),
                    np.concatenate((-mean / std + clip, -mean / std - clip)), "relu")]
    for index, (_, kernel, bias) in enumerate(dense):
        kernel, bias = kernel.copy(), bias.copy()
        if index == 0:
            bias -= clip * np.sum(kernel, axis=0)
            kernel = np.concatenate((kernel, -kernel), axis=0)
        if index == len(dense) - 1:
            kernel, bias = kernel[:, :12], bias[:12]
        layers.append(layer(kernel, bias, "tanh" if index == len(dense) - 1 else "elu"))

    center, scale = action_center_and_scale(task)
    model = mujoco.MjModel.from_xml_path(str(model_path_3d(task.geometry)))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in CONTROLLER_JOINT_NAMES_3D]
    aids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_servo") for name in CONTROLLER_JOINT_NAMES_3D]
    if min(ids + aids) < 0:
        raise ValueError("Missing controlled joints/actuators")
    kp, kd = model.actuator_gainprm[aids, 0], -model.actuator_biasprm[aids, 2]
    if not np.allclose(kp, kp[0]) or not np.allclose(kd, kd[0]):
        raise ValueError("Controller JSON kp/kd loader expects uniform gains")
    document = {
        "in_shape": [1, 720], "out_shape": [1, 12], "layers": layers,
        "observation_history": 20, "default_joint_pos": center.tolist(),
        "action_scale": scale.tolist(), "joint_lower_limits": model.jnt_range[ids, 0].tolist(),
        "joint_upper_limits": model.jnt_range[ids, 1].tolist(), "kp": float(kp[0]), "kd": float(kd[0]),
        "use_imu": True, "control_orientation": False,
        "joint_names": list(HARDWARE_CONTROLLER_JOINT_NAMES_3D),
        "export_contract": "stand_to_roll_student_clipped_bc_norm_v1",
        "student_sha256": hashlib.sha256(args.student_params.read_bytes()).hexdigest(),
        "bc_sha256": source["bc_sha256"], "symmetric_action_projection": False,
        "controller_requirements": {
            "policy_period_s": task.control_timestep, "action_type": "position",
            "raw_observation_limit": task.observation_limit,
            "command": [0, 0, 0], "desired_world_z_observation": [0, 0, 1],
            "initial_state": "stand with zero velocity; action center is not the startup stand pose",
            "joint_order": list(HARDWARE_CONTROLLER_JOINT_NAMES_3D),
            "note": "Requirements are metadata, not automatically applied ROS parameters. Rolling-compatible body-angle handling and explicit handoff/stop logic are required.",
        },
    }
    # Cloud numerical conversion check, including inputs outside the clipping
    # interval. This is not a hardware or C++ runtime validation.
    rng = np.random.default_rng(7)
    obs = (mean + std * rng.uniform(-2 * clip, 2 * clip, (64, 720))).astype(np.float32)
    value = np.clip((obs - mean) / std, -clip, clip)
    for index, (_, kernel, bias) in enumerate(dense):
        if index == len(dense) - 1:
            value = np.tanh(value @ kernel[:, :12] + bias[:12])
        else:
            value = value @ kernel + bias
            value = np.where(value > 0, value, np.expm1(np.minimum(value, 0)))
    exported = _run_layers(obs, layers)
    error = float(np.max(np.abs(exported - value)))
    if not np.isfinite(error) or error > 1e-4:
        raise RuntimeError(f"JSON conversion mismatch: {error}; not writing model")
    document["conversion_max_action_error"] = error
    _write_json(str(args.out.resolve()), document)
    print(f"Exported {args.out}: 720 -> 12, max action error={error:.3g}, no symmetry projection")
    print("Clipped normalization adds a 1440-unit ReLU layer; measure controller inference latency before use.")


if __name__ == "__main__":
    main()

from pathlib import Path
import math
import unittest

import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d_mjx.config_walking_3d import (
    Walking3DConfig,
    WalkingReference3DConfig,
    validate_walking_3d_config,
    walking_physics_profile_3d,
)
from curl_robot_2d_mjx.environment_walking_3d import (
    EXPECTED_WALKING_JOINT_AXES_3D,
    WALKING_ACTION_SIZE_3D,
    WALKING_MODEL_PATH_3D,
    WALKING_OBSERVATION_SIZE_3D,
    validate_walking_morphology_3d,
)
from curl_robot_2d_mjx.walking_reference_3d import (
    LEG_WORLD_X_SIGNS_3D,
    leg_forward_kinematics,
    leg_inverse_kinematics,
    walking_reference_3d,
)
from scripts import mjx_3d_walking_smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MJX3DWalkingContractTest(unittest.TestCase):
    def test_task_uses_walk_keyframe_and_eight_joint_residual(self) -> None:
        config = Walking3DConfig()

        self.assertEqual(
            WALKING_MODEL_PATH_3D,
            PROJECT_ROOT / "assets" / "curl_robot_3d.xml",
        )
        self.assertEqual(config.reset_keyframe_name, "walk")
        self.assertEqual(len(JOINT_NAMES_3D), WALKING_ACTION_SIZE_3D)
        self.assertEqual(WALKING_OBSERVATION_SIZE_3D, 74)
        self.assertEqual(config.reference.phase_offsets, (0.0, 0.0, 0.5, 0.5))
        self.assertEqual(config.reference.initial_phase_fraction, 0.0)
        self.assertEqual(config.reset_reference_weight, 1.0)

    def test_model_matches_mirrored_planar_leg_morphology(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(WALKING_MODEL_PATH_3D))

        validate_walking_morphology_3d(model)

        for name, expected_axis in EXPECTED_WALKING_JOINT_AXES_3D.items():
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            np.testing.assert_allclose(model.jnt_axis[joint_id], expected_axis)
            self.assertAlmostEqual(abs(model.jnt_axis[joint_id, 1]), 1.0)

    def test_reference_ik_respects_rear_world_x_mirroring(self) -> None:
        config = WalkingReference3DConfig()
        reference = walking_reference_3d(
            np, np.asarray(1.9 * np.pi), config
        )

        self.assertEqual(reference["joint_targets"].shape, (8,))
        self.assertEqual(reference["stance"].tolist(), [False, False, True, True])
        for leg_index, world_x_sign in enumerate(LEG_WORLD_X_SIGNS_3D):
            hip, knee = reference["joint_targets"][2 * leg_index : 2 * leg_index + 2]
            effective_x, _ = leg_forward_kinematics(
                np,
                hip,
                knee,
                config.upper_length_m,
                config.lower_length_m,
            )
            actual_world_x = world_x_sign * effective_x
            self.assertAlmostEqual(
                float(actual_world_x),
                float(reference["foot_world_x_m"][leg_index]),
                places=7,
            )

    def test_general_two_link_ik_round_trip(self) -> None:
        upper = 0.14
        lower = 0.16
        for outward, depth in ((0.0, 0.25), (0.04, 0.24), (-0.03, 0.22)):
            hip, knee = leg_inverse_kinematics(
                np, outward, depth, upper, lower
            )
            actual_outward, actual_depth = leg_forward_kinematics(
                np, hip, knee, upper, lower
            )
            self.assertAlmostEqual(float(actual_outward), outward, places=9)
            self.assertAlmostEqual(float(actual_depth), depth, places=9)

    def test_zero_residual_reference_has_stable_one_second_start(self) -> None:
        config = Walking3DConfig()
        model = mujoco.MjModel.from_xml_path(str(WALKING_MODEL_PATH_3D))
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            config.reset_keyframe_name,
        )
        joint_qpos = np.asarray(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                ]
                for name in JOINT_NAMES_3D
            ],
            dtype=int,
        )
        initial_phase = (
            2.0 * math.pi * config.reference.initial_phase_fraction
        )
        initial_reference = walking_reference_3d(
            np, np.asarray(initial_phase), config.reference
        )["joint_targets"]
        startup_target = (
            model.key_ctrl[key_id]
            + config.reset_reference_weight
            * (initial_reference - model.key_ctrl[key_id])
        )
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.qpos[joint_qpos] = startup_target
        data.ctrl[:] = startup_target
        mujoco.mj_forward(model, data)
        initial_x = float(data.qpos[0])
        initial_y = float(data.qpos[1])
        torso_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        maximum_tilt = 0.0
        steps = int(round(1.0 / model.opt.timestep))
        for step in range(steps):
            elapsed = step * model.opt.timestep
            if step % config.action_repeat == 0:
                ramp = min(
                    max(
                        elapsed / config.startup_reference_ramp_s,
                        0.0,
                    ),
                    1.0,
                )
                blend = ramp * ramp * (3.0 - 2.0 * ramp)
                phase = (
                    initial_phase
                    + 2.0
                    * math.pi
                    * config.reference.frequency_hz
                    * elapsed
                ) % (2.0 * math.pi)
                reference_target = walking_reference_3d(
                    np, np.asarray(phase), config.reference
                )["joint_targets"]
                data.ctrl[:] = startup_target + blend * (
                    reference_target - startup_target
                )
            mujoco.mj_step(model, data)
            body_z = data.xmat[torso_id].reshape(3, 3)[:, 2]
            maximum_tilt = max(
                maximum_tilt,
                float(np.arccos(np.clip(body_z[2], -1.0, 1.0))),
            )

        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertGreater(float(data.qpos[2]), 0.25)
        self.assertLess(maximum_tilt, 0.35)
        self.assertGreater(float(data.qpos[0]) - initial_x, 0.0)
        self.assertAlmostEqual(float(data.qpos[1]) - initial_y, 0.0, places=8)

    def test_config_validation_and_fast_profile(self) -> None:
        validate_walking_3d_config(Walking3DConfig())
        cg12 = walking_physics_profile_3d("cg12")

        self.assertAlmostEqual(cg12.control_timestep, 0.02)
        self.assertEqual(cg12.solver_name, "cg")
        self.assertEqual(cg12.solver_iterations, 12)
        for values in (
            {"action_scales": (1.0,)},
            {"terminate_root_z_min": 0.5},
            {"reset_reference_weight": 1.1},
            {
                "reference": WalkingReference3DConfig(
                    phase_offsets=(0.0, 0.5)
                )
            },
            {
                "reference": WalkingReference3DConfig(
                    initial_phase_fraction=1.0
                )
            },
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_walking_3d_config(Walking3DConfig(**values))

    def test_smoke_entry_imports_without_jax(self) -> None:
        args = mjx_3d_walking_smoke.parse_args([])

        self.assertTrue(callable(mjx_3d_walking_smoke.main))
        self.assertEqual(args.physics_profile, "cg12")
        self.assertEqual(args.steps, 8)
        self.assertEqual(args.frequency_hz, 0.70)
        self.assertEqual(args.residual_gain, 0.65)
        self.assertEqual(args.reset_reference_weight, 1.0)

    def test_environment_declares_walking_failures_and_guards(self) -> None:
        source = (
            PROJECT_ROOT / "curl_robot_2d_mjx" / "environment_walking_3d.py"
        ).read_text(encoding="utf-8")
        for token in (
            "failure_root_low",
            "failure_upright_tilt",
            "failure_airborne",
            "failure_nonfoot_contact",
            "failure_self_contact",
            "walking_reference_3d",
            "startup_reference_ramp_s",
            "jax.lax.cond",
            "transition_finite",
            "jp.nan_to_num",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()

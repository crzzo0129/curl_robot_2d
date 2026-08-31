from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from curl_robot_2d_mjx.config_transition_3d import (
    TRANSITION_ACTION_SIZE_3D,
    TRANSITION_ACTOR_OBSERVATION_SIZE_3D,
    TRANSITION_CRITIC_OBSERVATION_SIZE_3D,
    TRANSITION_CURRICULUM_STAGE_NAMES_3D,
    ReadyToWalkSample3D,
    Transition3DConfig,
    TransitionMode3D,
    is_ready_to_walk_3d,
    transition_curriculum_config_3d,
    validate_transition_config_3d,
)
from curl_robot_2d_mjx.environment_transition_3d import (
    TRANSITION_KEYFRAME_NAMES_3D,
    TRANSITION_MODEL_PATH_3D,
    transition_reference_ctrl_3d,
)
from curl_robot_2d_mjx.transition_initialization_3d import (
    walking_start_state_3d, transition_target_ctrl_3d,
    transition_action_from_ctrl_3d, save_roll_snapshots_3d,
    load_roll_snapshots_3d,
)
from curl_robot_2d_mjx.environment_walking_3d import (
    WALKING_JOINT_NAMES_3D, validate_walking_morphology_3d,
)
from curl_robot_2d_mjx.environment_3d import apply_physics_options_3d
from curl_robot_2d_mjx.reward_transition_3d import (
    TRANSITION_REWARD_TERM_NAMES_3D,
    Transition3DRewardConfig,
    reward_terms_transition_3d,
)
from curl_robot_2d_mjx.supervisor_transition_3d import (
    PolicyRoute3D,
    RollToWalkSupervisor3D,
    ThreePolicyController3D,
)
from scripts.mjx_3d_transition_smoke import parse_args as parse_smoke_args
from scripts.train_mjx_3d_transition_ppo import (
    build_task,
    parse_args as parse_train_args,
)


class Transition3DConfigTests(unittest.TestCase):
    def test_all_curriculum_stages_validate(self):
        modes = []
        for stage in TRANSITION_CURRICULUM_STAGE_NAMES_3D:
            task = transition_curriculum_config_3d(stage)
            validate_transition_config_3d(task)
            modes.append(task.reset_start_mode)
        self.assertEqual(modes, [int(TransitionMode3D.STABILIZE),
                                int(TransitionMode3D.DEPLOY),
                                int(TransitionMode3D.DEPLOY),
                                int(TransitionMode3D.BRAKE),
                                int(TransitionMode3D.BRAKE)])
        self.assertEqual(TRANSITION_CURRICULUM_STAGE_NAMES_3D[0], "walking_start")

    def test_observation_and_action_contract(self):
        self.assertEqual(TRANSITION_ACTION_SIZE_3D, 12)
        self.assertEqual(TRANSITION_ACTOR_OBSERVATION_SIZE_3D, 720)
        self.assertEqual(TRANSITION_CRITIC_OBSERVATION_SIZE_3D, 86)

    def test_reference_is_only_walking_target(self):
        stand = np.full(12, 1.0)
        np.testing.assert_array_equal(
            transition_reference_ctrl_3d(np, stand), stand)
        self.assertNotIn("park", TRANSITION_KEYFRAME_NAMES_3D)


class Transition3DSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.config = Transition3DConfig(ready_hold_s=0.06)
        self.ready = ReadyToWalkSample3D(
            linear_speed_m_s=0.02,
            angular_speed_rad_s=0.05,
            upright_tilt_rad=0.03,
            joint_error_rad=0.04,
            root_height_m=0.17,
            foot_contacts=4,
        )

    def test_ready_requires_continuous_hold(self):
        supervisor = RollToWalkSupervisor3D(self.config)
        output = supervisor.update(
            stop_requested=True, sample=self.ready, dt=0.02
        )
        self.assertEqual(output.route, PolicyRoute3D.TRANSITION)
        output = supervisor.update(
            stop_requested=True, sample=self.ready, dt=0.02
        )
        self.assertEqual(output.route, PolicyRoute3D.TRANSITION)
        output = supervisor.update(
            stop_requested=True, sample=self.ready, dt=0.02
        )
        self.assertEqual(output.route, PolicyRoute3D.WALK)

    def test_bad_sample_resets_hold_and_cannot_return_to_roll(self):
        supervisor = RollToWalkSupervisor3D(self.config)
        supervisor.update(stop_requested=True, sample=self.ready, dt=0.02)
        bad = ReadyToWalkSample3D(
            **{**self.ready.__dict__, "angular_speed_rad_s": 1.0}
        )
        output = supervisor.update(
            stop_requested=False, sample=bad, dt=0.02
        )
        self.assertEqual(output.route, PolicyRoute3D.TRANSITION)
        self.assertEqual(output.ready_hold_s, 0.0)
        self.assertIn("angular_speed", output.gate_failures)

    def test_ready_gate_scalar_contract(self):
        self.assertTrue(is_ready_to_walk_3d(self.ready, self.config))

    def test_nonfinite_measurement_cannot_pass_ready(self):
        self.assertFalse(is_ready_to_walk_3d(
            replace(self.ready, linear_speed_m_s=float("nan")), self.config))

    def test_three_policy_controller_routes_native_observations(self):
        calls = []

        def policy(name):
            def run(observation):
                calls.append((name, observation))
                return f"{name}-action"

            return run

        controller = ThreePolicyController3D(
            roll_policy=policy("roll"),
            transition_policy=policy("transition"),
            walk_policy=policy("walk"),
            config=self.config,
        )
        observations = {route: route.value for route in PolicyRoute3D}
        output = controller.control(
            stop_requested=False,
            ready_sample=self.ready,
            observations=observations,
            dt=0.02,
        )
        self.assertEqual((output.route, output.action), (PolicyRoute3D.ROLL, "roll-action"))
        output = controller.control(
            stop_requested=True,
            ready_sample=self.ready,
            observations=observations,
            dt=0.02,
        )
        self.assertEqual(output.route, PolicyRoute3D.TRANSITION)
        controller.control(
            stop_requested=True,
            ready_sample=self.ready,
            observations=observations,
            dt=0.02,
        )
        output = controller.control(
            stop_requested=True,
            ready_sample=self.ready,
            observations=observations,
            dt=0.02,
        )
        self.assertEqual(output.route, PolicyRoute3D.WALK)
        self.assertEqual(calls[-1], ("walk", "walk"))


class Transition3DRewardTests(unittest.TestCase):
    def _inputs(self):
        return {
            "mode_brake": 1.0,
            "mode_deploy": 0.0,
            "mode_stabilize": 0.0,
            "combined_speed": 0.5,
            "previous_combined_speed": 0.8,
            "reference_pose_error_rms": 0.2,
            "previous_reference_pose_error_rms": 0.3,
            "upright_tilt": 0.1,
            "root_height_error": 0.01,
            "support_fraction": 1.0,
            "newly_ready": 0.0,
            "action_rate_squared": 0.1,
            "action_squared": 0.1,
            "joint_velocity_squared": 1.0,
            "foot_slip_velocity_squared": 0.01,
            "contact_force_peak_n": 10.0,
            "nonfoot_contact_count": 0.0,
            "failed": 0.0,
        }

    def test_reward_has_complete_finite_named_terms(self):
        terms = reward_terms_transition_3d(
            np, Transition3DRewardConfig(), self._inputs()
        )
        self.assertEqual(tuple(terms), TRANSITION_REWARD_TERM_NAMES_3D)
        self.assertTrue(all(math.isfinite(float(value)) for value in terms.values()))

    def test_braking_progress_is_positive_when_speed_decreases(self):
        terms = reward_terms_transition_3d(
            np, Transition3DRewardConfig(), self._inputs()
        )
        self.assertGreater(float(terms["brake_progress"]), 0.0)


class Transition3DModelAndCliTests(unittest.TestCase):
    def test_final_model_has_12_actuators_and_required_keys(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo is unavailable")
        model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        self.assertEqual((model.nq, model.nv, model.nu), (19, 18, 12))
        validate_walking_morphology_3d(model, geometry_name="rollingquad_2")
        keys = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, index)
            for index in range(model.nkey)
        }
        self.assertTrue({"compact", "stand"}.issubset(keys))

    def test_stand_terminal_state_satisfies_ready_gate_on_cpu(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo is unavailable")
        model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        apply_physics_options_3d(model, Transition3DConfig())
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )
        torso_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        floor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        foot_ids = {
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_proxy"
            )
            for leg in ("front_left", "front_right", "rear_left", "rear_right")
        }
        initial = walking_start_state_3d(model, Transition3DConfig())
        data.qpos[:] = initial["qpos"]
        data.qvel[:] = initial["qvel"]
        data.ctrl[:] = initial["ctrl"]
        mujoco.mj_forward(model, data)
        for _ in range(round(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        rotation = data.xmat[torso_id].reshape(3, 3)
        contacts = set()
        for index in range(data.ncon):
            pair = {int(data.contact[index].geom1), int(data.contact[index].geom2)}
            if floor_id in pair:
                contacts.update(pair & foot_ids)
        sample = ReadyToWalkSample3D(
            linear_speed_m_s=float(np.linalg.norm(data.qvel[:3])),
            angular_speed_rad_s=float(np.linalg.norm(data.qvel[3:6])),
            upright_tilt_rad=float(
                np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
            ),
            joint_error_rad=float(
                np.sqrt(np.mean(np.square(data.qpos[[
                    model.jnt_qposadr[model.joint(name).id]
                    for name in WALKING_JOINT_NAMES_3D
                ]] - model.key_ctrl[key_id])))
            ),
            root_height_m=float(data.qpos[2]),
            foot_contacts=len(contacts),
        )
        self.assertTrue(is_ready_to_walk_3d(sample), str(sample))

    def test_named_walking_start_matches_current_deploy_env(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        initial = walking_start_state_3d(model, Transition3DConfig())
        indices = [model.jnt_qposadr[model.joint(name).id]
                   for name in WALKING_JOINT_NAMES_3D]
        self.assertNotEqual(indices, list(range(7, 19)))
        np.testing.assert_allclose(initial["qpos"][indices], [0.0, .90, 1.15] * 4)
        np.testing.assert_array_equal(initial["qvel"], np.zeros(18))
        self.assertAlmostEqual(initial["qpos"][2], 0.1580029248 + .0005)

    def test_transition_physics_options_match_shared_model_api(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        apply_physics_options_3d(model, Transition3DConfig())
        self.assertEqual(model.opt.iterations, 4)
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], 5.0)

    def test_action_covers_new_model_compact_and_stand(self):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        ids = [model.joint(name).id for name in WALKING_JOINT_NAMES_3D]
        low, high = model.jnt_range[ids].T
        stand = model.key_ctrl[model.key("stand").id]
        compact = model.key_ctrl[model.key("compact").id]
        for target in (stand, compact, low, high):
            action = transition_action_from_ctrl_3d(np, target, stand, low, high)
            self.assertTrue(np.all(np.abs(action) <= 1.0))
            np.testing.assert_allclose(
                transition_target_ctrl_3d(np, action, stand, low, high), target)
        np.testing.assert_array_equal(
            transition_target_ctrl_3d(np, np.zeros(12), stand, low, high), stand)

    def test_training_dry_configuration_requires_no_jax(self):
        args = parse_train_args(
            ["--stage", "brake_low", "--preset", "smoke", "--dry-run"]
        )
        task = build_task(args)
        self.assertEqual(task.curriculum_stage, "brake_low")
        self.assertEqual(task.reset_start_mode, int(TransitionMode3D.BRAKE))
        self.assertEqual(task.geometry, "rollingquad_2")

    def test_default_cli_starts_from_walking(self):
        self.assertEqual(build_task(parse_train_args([])).curriculum_stage, "walking_start")
        self.assertEqual(parse_smoke_args([]).stage, "walking_start")


class TransitionRollSnapshotTests(unittest.TestCase):
    def setUp(self):
        import mujoco
        self.model = mujoco.MjModel.from_xml_path(str(TRANSITION_MODEL_PATH_3D))
        self.config = Transition3DConfig()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "roll.npz"
        initial = walking_start_state_3d(self.model, self.config)
        self.qpos = np.tile(initial["qpos"], (4, 1))
        self.qvel = np.zeros((4, 18))
        self.qvel[:, 0] = [.1, .2, .3, .4]
        self.qvel[:, 4] = [1., 2., 3., 4.]
        self.ctrl = np.tile(initial["ctrl"], (4, 1))
        save_roll_snapshots_3d(
            self.path, self.model, self.config, qpos=self.qpos,
            qvel=self.qvel, ctrl=self.ctrl, time_s=np.arange(4.),
            episode_id=np.zeros(4), source_policy="unit-test fixture, not trained",
        )

    def test_tail_filter_retains_original_velocity_not_slowed(self):
        task = replace(self.config, snapshot_tail_fraction=.5)
        bank = load_roll_snapshots_3d(self.path, self.model, task)
        np.testing.assert_array_equal(bank["qpos"], self.qpos[2:])
        np.testing.assert_array_equal(bank["qvel"], self.qvel[2:])
        np.testing.assert_array_equal(bank["ctrl"], self.ctrl[2:])

    def test_low_speed_course_filters_without_scaling(self):
        task = transition_curriculum_config_3d("brake_low", self.config)
        bank = load_roll_snapshots_3d(self.path, self.model, task)
        np.testing.assert_array_equal(bank["qvel"], self.qvel[:3])
        full = load_roll_snapshots_3d(
            self.path, self.model, transition_curriculum_config_3d("brake_full"))
        np.testing.assert_array_equal(full["qvel"], self.qvel)

    def test_empty_filter_raises_instead_of_modifying_velocities(self):
        task = replace(self.config, snapshot_max_linear_speed_m_s=.01)
        with self.assertRaisesRegex(ValueError, "no ROLL snapshots"):
            load_roll_snapshots_3d(self.path, self.model, task)

    def test_wrong_model_and_pose_only_banks_rejected(self):
        with np.load(self.path) as data:
            arrays = dict(data)
        arrays["geometry"] = np.asarray("pupper_open60")
        wrong = Path(self.directory.name) / "wrong.npz"
        np.savez(wrong, **arrays)
        with self.assertRaisesRegex(ValueError, "model/order mismatch"):
            load_roll_snapshots_3d(wrong, self.model, self.config)
        arrays["geometry"] = np.asarray("rollingquad_2")
        del arrays["qvel"]
        np.savez(wrong, **arrays)
        with self.assertRaisesRegex(ValueError, "qvel"):
            load_roll_snapshots_3d(wrong, self.model, self.config)

    def test_smoke_parser_requires_no_jax(self):
        args = parse_smoke_args(["--stage", "deploy_capture", "--steps", "2"])
        self.assertEqual(args.stage, "deploy_capture")
        self.assertEqual(args.steps, 2)


if __name__ == "__main__":
    unittest.main()

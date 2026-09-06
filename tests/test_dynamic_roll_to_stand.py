"""Regression contracts for primitive dynamic recovery, independent of PPO."""

import unittest
import json
import tempfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
from scripts.train_mjx_3d_transition_ppo import parse_args, build_task
from curl_robot_2d_mjx.environment_3d import model_path_3d, geometry_parameters_3d
from curl_robot_2d_mjx.environment_walking_3d import validate_walking_morphology_3d
from curl_robot_2d_mjx.reward_transition_3d import Transition3DRewardConfig, reward_terms_roll_to_stand_3d


class DynamicContracts(unittest.TestCase):
    def test_reference_dynamics_and_split(self):
        from curl_robot_2d_mjx.reference_bank_contract_3d import validate_reference_split
        task = build_task(parse_args(["--dynamic-roll-to-stand", "--geometry", "rollingquad_2_primitive"]))
        with tempfile.TemporaryDirectory() as directory:
            train, evaluation = Path(directory) / "train.npz", Path(directory) / "eval.npz"
            report = dict(source_kind="cem_reference_zero_residual", status="ok",
                          task=asdict(task), reference_sha256="same", seed=0)
            def write(path, payload):
                path.with_suffix(".summary.json").write_text(json.dumps(payload), encoding="utf-8")
            write(train, report)
            write(evaluation, {**report, "seed": 1000})
            self.assertTrue(validate_reference_split(train, evaluation, task)["nominal_dynamics_match"])
            write(evaluation, report)
            with self.assertRaisesRegex(ValueError, "seeds overlap"):
                validate_reference_split(train, evaluation, task)
            write(evaluation, {**report, "seed": 1000, "task": {**report["task"], "solver_iterations": 20}})
            with self.assertRaisesRegex(ValueError, "dynamics differ"):
                validate_reference_split(train, evaluation, task)

    def test_mesh_training_rejected_before_runtime(self):
        with self.assertRaisesRegex(ValueError, "require primitive"):
            build_task(parse_args(["--dynamic-roll-to-stand"]))

    def test_primitive_config_and_stand_duration(self):
        task = build_task(parse_args(["--dynamic-roll-to-stand", "--geometry", "rollingquad_2_primitive"]))
        self.assertTrue(task.dynamic_roll_to_stand)
        self.assertEqual(task.ready_hold_s + task.stand_verification_s, 3.0)
        self.assertEqual(task.episode_length, 500)
        self.assertEqual(task.control_timestep, 0.02)

    def test_primitive_actuators_and_contacts(self):
        import mujoco
        name = "rollingquad_2_primitive"
        m = mujoco.MjModel.from_xml_path(str(model_path_3d(name)))
        validate_walking_morphology_3d(m, geometry_parameters_3d(name), geometry_name=name)
        active = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.assertFalse(np.any(active & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH)))
        self.assertEqual(m.nu, 12)

    def test_stopping_without_feet_has_no_stable_bonus(self):
        inputs = dict(mode_brake=0., mode_deploy=1., mode_stabilize=0.,
            combined_speed=0., previous_combined_speed=2., reference_pose_error_rms=0.,
            previous_reference_pose_error_rms=1., upright_tilt=0., root_height_error=0.,
            support_fraction=0., nonfoot_contact_count=4., newly_ready=0.,
            action_rate_squared=0., action_squared=0., joint_velocity_squared=0.,
            foot_slip_velocity_squared=0., contact_force_peak_n=0., failed=0.)
        terms = reward_terms_roll_to_stand_3d(np, Transition3DRewardConfig(), inputs)
        self.assertEqual(terms["stabilize"], 0.)
        self.assertEqual(terms["brake_progress"], 0.)
        self.assertEqual(terms["brake_speed"], 0.)
        standing = reward_terms_roll_to_stand_3d(np, Transition3DRewardConfig(),
            {**inputs, "support_fraction": 1., "nonfoot_contact_count": 0.})
        self.assertGreater(standing["stabilize"], terms["stabilize"])


if __name__ == "__main__":
    unittest.main()

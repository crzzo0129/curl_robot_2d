"""CPU checks for the v4 training fixes; no JAX/Brax imports required."""

from dataclasses import replace
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.config_transition_3d import (
    Transition3DConfig, TransitionMode3D, stabilize_failure_update_3d,
    validate_transition_config_3d,
)
from curl_robot_2d_mjx.reward_transition_3d import Transition3DRewardConfig, reward_terms_transition_3d
from curl_robot_2d_mjx.training_transition_3d import (
    initialize_transition_actor, transition_scale_logit, transition_curriculum_acceptance,
)
from curl_robot_2d_mjx.wrappers_transition_3d import select_reset_lanes
from scripts.train_mjx_3d_transition_ppo import main


class InitializationTests(unittest.TestCase):
    def test_zero_mean_small_std_keeps_hidden_layers_and_schema(self):
        params = {"params": {
            "hidden_0": {"kernel": np.ones((720, 16)), "bias": np.zeros(16)},
            "hidden_1": {"kernel": np.ones((16, 24)), "bias": np.ones(24)},
        }}
        actual = initialize_transition_actor(np, params, (16,))
        self.assertIs(actual["params"]["hidden_0"], params["params"]["hidden_0"])
        head = actual["params"]["hidden_1"]
        np.testing.assert_array_equal(head["kernel"], np.zeros((16, 24)))
        np.testing.assert_array_equal(head["bias"][:12], np.zeros(12))
        np.testing.assert_allclose(np.logaddexp(0, head["bias"][12:]) + .001, .05)
        # No in-place edits to input parameters, including potential restore trees.
        np.testing.assert_array_equal(params["params"]["hidden_1"]["kernel"], 1)
        np.testing.assert_array_equal(params["params"]["hidden_1"]["bias"], 1)

    def test_scale_validation(self):
        for std in (0, .001, -1, float("nan"), float("inf")):
            with self.subTest(std=std), self.assertRaises(ValueError):
                transition_scale_logit(std)
        for std in (.002, .05, .2, 25):
            self.assertAlmostEqual(np.logaddexp(0, transition_scale_logit(std)) + .001, std)

    def test_reset_mask_selects_whole_history_per_lane(self):
        fresh, current = np.zeros((3, 720)), np.ones((3, 720))
        actual = select_reset_lanes(np, np.array([True, False, True]), fresh, current)
        np.testing.assert_array_equal(actual[[0, 2]], 0)
        np.testing.assert_array_equal(actual[1], 1)
        np.testing.assert_array_equal(current, 1)


class StandingGuardTests(unittest.TestCase):
    def setUp(self):
        self.config = Transition3DConfig()
        self.bad = dict(mode=int(TransitionMode3D.STABILIZE), mode_steps=100,
                        previous_bad_steps=0, root_height=.104, joint_error=1.81,
                        tilt=.16, foot_contacts=0, nonfoot_contacts=4)

    def update(self, **overrides):
        return stabilize_failure_update_3d(np, self.config, **{**self.bad, **overrides})

    def test_bad_support_requires_consecutive_hold(self):
        hold = math.ceil(self.config.stabilize_bad_support_hold_s / self.config.control_timestep)
        count = 0
        for index in range(hold):
            count, failed = self.update(previous_bad_steps=count)
            self.assertEqual(bool(failed), index == hold - 1)

    def test_good_sample_resets_counter_and_grace_suppresses_failure(self):
        count, failed = self.update(previous_bad_steps=100, root_height=.154,
                                   joint_error=.03, foot_contacts=4, nonfoot_contacts=0)
        self.assertEqual(int(count), 0)
        self.assertFalse(failed)
        self.assertEqual(int(self.update(mode_steps=0, previous_bad_steps=100)[0]), 0)

    def test_brake_and_deploy_shell_contact_is_not_stabilize_failure(self):
        for mode in (TransitionMode3D.BRAKE, TransitionMode3D.DEPLOY):
            count, failed = self.update(mode=int(mode), previous_bad_steps=100)
            self.assertEqual(int(count), 0)
            self.assertFalse(failed)

    def test_individual_standing_faults(self):
        good = dict(root_height=.154, joint_error=.03, tilt=.02,
                    foot_contacts=4, nonfoot_contacts=0, previous_bad_steps=100)
        for fault in ({"root_height": .10}, {"joint_error": 1.8}, {"tilt": 1.0},
                      {"foot_contacts": 1}, {"nonfoot_contacts": 1}):
            with self.subTest(fault=fault):
                self.assertTrue(self.update(**{**good, **fault})[1])

    def test_config_rejects_unsafe_guard_bounds(self):
        for override in ({"stabilize_bad_support_hold_s": 0},
                         {"stabilize_failure_root_height_min_m": .14},
                         {"stabilize_failure_min_foot_contacts": 4}):
            with self.assertRaises(ValueError):
                validate_transition_config_3d(replace(self.config, **override))


class StandingRewardTests(unittest.TestCase):
    def inputs(self, **overrides):
        return dict(mode_brake=0., mode_deploy=0., mode_stabilize=1., combined_speed=0.,
                    previous_combined_speed=0., reference_pose_error_rms=0.,
                    previous_reference_pose_error_rms=0., upright_tilt=0., root_height_error=0.,
                    support_fraction=1., newly_ready=0., action_rate_squared=0., action_squared=0.,
                    joint_velocity_squared=0., foot_slip_velocity_squared=0.,
                    contact_force_peak_n=0., nonfoot_contact_count=0., failed=0.) | overrides

    def terms(self, **overrides):
        return reward_terms_transition_3d(np, Transition3DRewardConfig(), self.inputs(**overrides))

    def test_logged_belly_pose_cannot_harvest_upright_reward(self):
        terms = self.terms(reference_pose_error_rms=1.81, root_height_error=.104-.158,
                           upright_tilt=.16, support_fraction=0., nonfoot_contact_count=4.)
        self.assertEqual(float(terms["upright"]), 0.)
        self.assertEqual(float(terms["height"]), 0.)
        self.assertEqual(float(terms["stabilize"]), 0.)
        self.assertLess(float(sum(terms.values())), 0.)

    def test_standing_has_positive_reward_and_rewards_lower_speed(self):
        self.assertGreater(float(sum(self.terms().values())), 0.)
        self.assertGreater(float(self.terms()["stabilize"]),
                           float(self.terms(combined_speed=.5)["stabilize"]))

    def test_deploy_can_receive_upright_shaping_before_foot_landing(self):
        terms = self.terms(mode_deploy=1., mode_stabilize=0., support_fraction=0.)
        self.assertGreater(float(terms["upright"]), 0.)
        self.assertEqual(float(terms["stabilize_pose"]), 0.)


class CurriculumAcceptanceTests(unittest.TestCase):
    def row(self, step, success=1., failure=0., timeout=0.):
        return {"step": step, "eval/episode_transition_success": success,
                "eval/episode_failed": failure, "eval/episode_timeout": timeout}

    def test_all_timeout_never_advances(self):
        gate = transition_curriculum_acceptance([self.row(1, 0, 0, 1), self.row(2, 0, 0, 1)])
        self.assertFalse(gate["passed"])
        self.assertIn("timeout_above_threshold", gate["reasons"])

    def test_requires_two_consecutive_post_training_passes(self):
        self.assertFalse(transition_curriculum_acceptance([self.row(0), self.row(1)])["passed"])
        self.assertTrue(transition_curriculum_acceptance([self.row(1), self.row(2)])["passed"])
        self.assertFalse(transition_curriculum_acceptance(
            [self.row(1), self.row(2, .5, .5, 0)])["passed"])

    def test_incomplete_nonfinite_or_double_counted_outcomes_fail(self):
        for row in (self.row(2, float("nan")), self.row(2, 1, 0, 1), self.row(2, .95)):
            with self.subTest(row=row):
                self.assertFalse(transition_curriculum_acceptance([self.row(1), row])["passed"])

    def test_dry_run_reports_correct_revision_and_preserves_abi(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main(["--dry-run"])
        config = json.loads(stream.getvalue())
        self.assertEqual(config["actor_observation_size"], 720)
        self.assertEqual(config["training"]["initial_policy_std"], .05)
        self.assertEqual(config["training"]["entropy_cost"], .0003)
        self.assertEqual(config["task"]["action_range_fraction"], 1.)
        self.assertEqual(config["curriculum_next_stage"], "deploy_near_stand")
        self.assertNotIn("next_stage", config)

    def test_existing_output_is_rejected_before_jax_import(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory) / "walking_start"
            stage.mkdir()
            # An existing subdirectory suffices, without creating checkpoint files.
            (stage / "ppo_checkpoint").mkdir()
            with self.assertRaisesRegex(SystemExit, "not be overwritten"):
                main(["--out", directory])


if __name__ == "__main__":
    unittest.main()

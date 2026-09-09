import unittest

import numpy as np

from curl_robot_2d_mjx.reward_transition_3d import (
    Transition3DRewardConfig, reward_terms_roll_to_stand_3d,
    smooth_stand_reward_config_3d,
)
from scripts.train_mjx_3d_transition_ppo import parse_args, build_task


class SmoothStandRewardTests(unittest.TestCase):
    def terms(self, config=None, **changes):
        inputs = dict(
            mode_brake=0., mode_deploy=1., mode_stabilize=0.,
            combined_speed=0., previous_combined_speed=0.,
            reference_pose_error_rms=0., previous_reference_pose_error_rms=0.,
            upright_tilt=0., root_height_error=0., support_fraction=1.,
            nonfoot_contact_count=0., newly_ready=0., action_rate_squared=0.,
            action_squared=0., joint_velocity_squared=0., target_rate_squared=0.,
            foot_slip_velocity_squared=0., contact_force_peak_n=0., failed=0.,
        )
        return reward_terms_roll_to_stand_3d(
            np, config or smooth_stand_reward_config_3d(), {**inputs, **changes})

    def test_target_jump_penalty_and_legacy_opt_out(self):
        self.assertLess(self.terms(target_rate_squared=400.)["target_rate"],
                        self.terms(target_rate_squared=100.)["target_rate"])
        self.assertEqual(self.terms(Transition3DRewardConfig(),
                                   target_rate_squared=400.)["target_rate"], 0.)

    def test_hold_pose_cost_requires_support(self):
        self.assertLess(self.terms(reference_pose_error_rms=.2)["stabilize_pose"], 0.)
        self.assertEqual(self.terms(reference_pose_error_rms=.2,
                                   support_fraction=0.)["stabilize_pose"], 0.)

    def test_quiet_supported_stand_preferred(self):
        quiet = sum(self.terms().values())
        moving = sum(self.terms(combined_speed=.3, joint_velocity_squared=16.,
                                foot_slip_velocity_squared=.04).values())
        self.assertGreater(quiet, moving)
        self.assertEqual(self.terms(support_fraction=0.)["stabilize"], 0.)
        self.assertLess(self.terms(nonfoot_contact_count=1.)["nonfoot_contact"], 0.)

    def test_profile_preserves_action_contract(self):
        args = ["--geometry", "rollingquad_2_abd10_no_self_collision",
                "--dynamic-roll-to-stand", "--stand-abduction-zero"]
        baseline = build_task(parse_args(args))
        self.assertEqual(baseline, build_task(parse_args(args + [
            "--reward-profile", "smooth_stand"])))
        with self.assertRaises(ValueError):
            build_task(parse_args(args + ["--reward-profile", "smooth_stand",
                                         "--handcrafted-reference-residual"]))


if __name__ == "__main__":
    unittest.main()

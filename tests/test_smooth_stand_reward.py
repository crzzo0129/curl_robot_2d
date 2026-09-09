import unittest

import numpy as np

from curl_robot_2d_mjx.reward_transition_3d import (
    Transition3DRewardConfig, reward_terms_roll_to_stand_3d,
    smooth_stand_reward_config_3d,
    smooth_deploy_reward_config_3d, deploy_window_fraction_3d,
    smooth_deploy_v3_reward_config_3d,
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
            deploy_window_fraction=1.,
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

    def test_window_uses_elapsed_time_not_mode_or_pose(self):
        fractions = [deploy_window_fraction_3d(np, i, .02, .15) for i in range(500)]
        self.assertAlmostEqual(sum(fractions) * .02, .15)
        self.assertAlmostEqual(fractions[7], .5)
        self.assertEqual(fractions[8], 0.)
        self.assertEqual(fractions[-1], 0.)

    def test_v2_smoothing_stops_even_if_deploy_mode_never_ends(self):
        config = smooth_deploy_reward_config_3d()
        kwargs = dict(action_rate_squared=1., target_rate_squared=100.,
                      joint_velocity_squared=64., foot_slip_velocity_squared=.1)
        early = self.terms(config, **kwargs)
        late = self.terms(config, deploy_window_fraction=0., **kwargs)
        for name in ("action_rate", "target_rate", "joint_velocity", "foot_slip"):
            self.assertLess(early[name], 0.)
            self.assertEqual(late[name], 0.)
        self.assertGreater(early["stabilize"], 0.)
        self.assertEqual(late["stabilize"], 0.)

    def test_v2_no_perpetual_near_stand_bonus(self):
        config = smooth_deploy_reward_config_3d()
        exact = sum(self.terms(config, deploy_window_fraction=0.).values())
        offset = sum(self.terms(config, deploy_window_fraction=0.,
                               reference_pose_error_rms=.23,
                               previous_reference_pose_error_rms=.23).values())
        self.assertAlmostEqual(exact, 0.)
        self.assertLess(offset, exact)

    def test_v3_airborne_reversals_and_post_deploy_motion_cost(self):
        config = smooth_deploy_v3_reward_config_3d()
        airborne = self.terms(config, support_fraction=0.,
                              target_acceleration_squared=250000.)
        self.assertLess(airborne["target_acceleration"], 0.)
        late = self.terms(config, deploy_window_fraction=0.,
                          joint_velocity_squared=64., target_rate_squared=100.,
                          combined_speed=.4)
        self.assertLess(late["hold_joint_motion"], 0.)
        self.assertEqual(late["stabilize"], 0.)
        self.assertEqual(late["deploy_instability"], 0.)

    def test_v3_default_window_is_300ms(self):
        args = ["--geometry", "rollingquad_2_abd10_no_self_collision",
                "--dynamic-roll-to-stand", "--reward-profile", "smooth_deploy_v3"]
        self.assertEqual(build_task(parse_args(args)).reference_deploy_duration_s, .3)
        self.assertAlmostEqual(sum(deploy_window_fraction_3d(np, i, .02, .3)
                                   for i in range(500)) * .02, .3)


if __name__ == "__main__":
    unittest.main()

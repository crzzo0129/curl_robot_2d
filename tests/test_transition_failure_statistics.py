import unittest

import numpy as np

from curl_robot_2d_mjx.failure_transition_3d import (
    TRANSITION_FAILURE_CAUSE_NAMES_3D,
    transition_failure_breakdown_3d,
    transition_failure_causes_3d,
)


class FailureCauseTests(unittest.TestCase):
    def classify(self, **overrides):
        values = dict(failed=True, action_finite=True, physics_finite=True,
                      root_height_low=False, root_height_high=False,
                      brake_timeout=False, deploy_timeout=False,
                      stabilize_guard=False)
        values.update(overrides)
        return transition_failure_causes_3d(
            np, **{name: np.asarray(value) for name, value in values.items()})

    def test_each_terminal_predicate_has_exactly_one_named_cause(self):
        cases = {
            "action_nonfinite": {"action_finite": False},
            "physics_nonfinite": {"physics_finite": False},
            "root_height_low": {"root_height_low": True},
            "root_height_high": {"root_height_high": True},
            "brake_timeout": {"brake_timeout": True},
            "deploy_timeout": {"deploy_timeout": True},
            "stabilize_guard": {"stabilize_guard": True},
            "other": {},
        }
        for expected, inputs in cases.items():
            with self.subTest(expected=expected):
                result = self.classify(**inputs)
                self.assertEqual(tuple(result), TRANSITION_FAILURE_CAUSE_NAMES_3D)
                self.assertEqual(sum(bool(value) for value in result.values()), 1)
                self.assertTrue(result[expected])

    def test_nonfailure_has_no_cause(self):
        result = self.classify(failed=False, root_height_low=True, brake_timeout=True)
        self.assertFalse(any(bool(value) for value in result.values()))

    def test_priority_makes_simultaneous_conditions_mutually_exclusive(self):
        result = self.classify(action_finite=False, physics_finite=False,
                               root_height_low=True, brake_timeout=True)
        self.assertTrue(result["action_nonfinite"])
        self.assertEqual(sum(bool(value) for value in result.values()), 1)
        result = self.classify(root_height_low=True, brake_timeout=True,
                               stabilize_guard=True)
        self.assertTrue(result["root_height_low"])
        self.assertEqual(sum(bool(value) for value in result.values()), 1)

    def test_batched_lanes_remain_one_hot(self):
        result = transition_failure_causes_3d(np,
            failed=np.array([1, 1, 1, 0], bool),
            action_finite=np.array([0, 1, 1, 1], bool),
            physics_finite=np.ones(4, bool),
            root_height_low=np.array([1, 1, 0, 0], bool),
            root_height_high=np.zeros(4, bool),
            brake_timeout=np.array([0, 1, 1, 1], bool),
            deploy_timeout=np.zeros(4, bool), stabilize_guard=np.zeros(4, bool))
        matrix = np.stack(tuple(result.values()))
        np.testing.assert_array_equal(matrix.sum(axis=0), [1, 1, 1, 0])
        np.testing.assert_array_equal(result["action_nonfinite"], [1, 0, 0, 0])
        np.testing.assert_array_equal(result["root_height_low"], [0, 1, 0, 0])
        np.testing.assert_array_equal(result["brake_timeout"], [0, 0, 1, 0])

    def test_summary_exposes_dominant_cause_and_consistency(self):
        metrics = {"eval/episode_failed": .4,
                   "eval/episode_failure_root_height_low": .3,
                   "eval/episode_failure_brake_timeout": .1}
        report = transition_failure_breakdown_3d(metrics)
        self.assertAlmostEqual(report["sum"], .4)
        self.assertAlmostEqual(report["consistency_error"], 0.)
        self.assertEqual(report["dominant_cause"], "root_height_low")
        self.assertTrue(report["mutually_exclusive"])


if __name__ == "__main__":
    unittest.main()

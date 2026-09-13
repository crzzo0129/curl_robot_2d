"""Cloud-side unit checks; these do not initialize JAX or run physics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.deployment_rolling_3d import CONTROLLER_JOINT_NAMES_3D
from curl_robot_2d_mjx.rolling_speed_tracking import heading_displacement, takeover_commands
from curl_robot_2d_mjx.rolling_student_snapshot_pool import read_handoff_bank, handoff_split_indices


class HandoffTrackingTest(unittest.TestCase):
    def test_heading_velocity_is_invariant_to_world_rotation(self):
        headings = np.array([0., np.pi/2, np.pi, -np.pi/2])
        displacement = heading_displacement(np, .6*np.cos(headings), .6*np.sin(headings), headings, headings)
        np.testing.assert_allclose(displacement, .6, atol=1e-12)

    def test_heading_wrap_uses_short_arc(self):
        result = heading_displacement(np, -.6, 0., np.deg2rad(179), np.deg2rad(-179))
        self.assertAlmostEqual(float(result), .6)

    def test_slowdown_ramp_and_completion(self):
        t = np.array([0., 1., 2., 10.])
        vx, yaw, active = takeover_commands(np, .9, .6, -.07, t, .15, .07)
        np.testing.assert_allclose(vx, [.9,.75,.6,.6])
        np.testing.assert_allclose(yaw, [0.,-.07,-.07,-.07])
        np.testing.assert_array_equal(active, [True,True,False,False])

    def test_ramp_accelerates_without_overshoot(self):
        vx, yaw, active = takeover_commands(np,.45,.75,.07,np.array([-1.,0.,1.,3.]),.15,.07)
        np.testing.assert_allclose(vx,[.45,.45,.6,.75])
        np.testing.assert_allclose(yaw,[0,0,.07,.07])
        self.assertFalse(active[-1])

    def test_split_keeps_every_frame_of_one_trajectory_together(self):
        ids = np.repeat(np.arange(12),3)
        train, train_ids = handoff_split_indices(ids)
        evaluation, eval_ids = handoff_split_indices(ids,evaluation=True)
        self.assertFalse(set(train_ids) & set(eval_ids))
        self.assertEqual(len(train)+len(evaluation),len(ids))
        order = np.random.default_rng(7).permutation(len(ids))
        _, shuffled_eval_ids = handoff_split_indices(ids[order],evaluation=True)
        np.testing.assert_array_equal(eval_ids,shuffled_eval_ids)

    def test_bank_rejects_bad_history_and_nonfinite_physics(self):
        n = 8
        data = dict(qpos=np.zeros((n,19)),qvel=np.zeros((n,18)),ctrl=np.zeros((n,12)),
            time=np.ones(n),history=np.zeros((n,720)),previous_action=np.zeros((n,12)),
            speed=np.full(n,.9),rolling_phase=np.ones(n),trajectory_id=np.arange(n),
            metadata=np.asarray(json.dumps(dict(schema='rolling_handoff_bank_v1',
                history_order='newest_first_past_frames',controller_joint_names=list(CONTROLLER_JOINT_NAMES_3D)))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bank.npz'
            np.savez(path,**data)
            bank,_ = read_handoff_bank(path)
            self.assertEqual(bank['history'].shape,(8,720))
            np.savez(path,**{**data,'history':np.zeros((n,684))})
            with self.assertRaisesRegex(ValueError,'720 history'):
                read_handoff_bank(path)
            bad = data['qvel'].copy()
            bad[0,0] = np.nan
            np.savez(path,**{**data,'qvel':bad})
            with self.assertRaisesRegex(ValueError,'finite states'):
                read_handoff_bank(path)
            np.savez(path,**{**data,'speed':np.zeros((n,1))})
            with self.assertRaisesRegex(ValueError,'scalar fields'):
                read_handoff_bank(path)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from curl_robot_2d_mjx.rolling_velocity_estimator import (
    EstimatorConfig, MotionLabeler, ObservationHistory, RollingVelocityEstimator,
    prepare_dataset, save_estimator, split_episodes,
)


class MotionLabelTest(unittest.TestCase):
    def test_straight_forward_reverse_and_side_slip(self):
        labeler = MotionLabeler(EstimatorConfig(velocity_filter_tau_s=0))
        labeler.update([1, 0, 0], [0, 1, 0])
        target, mask = labeler.update([1, 0, 5], [0, 1, 0])
        np.testing.assert_allclose(target, [1, 0])
        self.assertTrue(mask.all())
        labeler.reset()
        target, _ = labeler.update([-1, 0, 0], [0, 1, 0])
        self.assertEqual(target[0], -1)
        # Lateral displacement isn't counted as forward rolling speed.
        target, _ = labeler.update([0, 1, 0], [0, 1, 0])
        self.assertEqual(target[0], 0)

    def test_constant_radius_circle_both_directions_and_angle_wrap(self):
        dt = .02
        for angular_rate in (-.8, .8):
            labeler = MotionLabeler(EstimatorConfig(control_dt=dt, velocity_filter_tau_s=0))
            # Cross +pi/-pi while moving on a radius-2 circle: speed = 1.6.
            for step in range(100):
                heading = np.pi - .2 + angular_rate * step * dt
                direction = np.array([np.cos(heading), np.sin(heading), 0.])
                axis = np.array([-np.sin(heading), np.cos(heading), 0.])
                target, mask = labeler.update(1.6 * direction, axis)
                self.assertAlmostEqual(float(target[0]), 1.6, places=6)
                if step:
                    self.assertTrue(mask[1])
                    self.assertAlmostEqual(float(target[1]), angular_rate, places=5)

    def test_body_roll_does_not_change_heading(self):
        yaw = .7
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        labeler = MotionLabeler(EstimatorConfig())
        for pitch in np.linspace(0, 2 * np.pi, 101):
            ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0],
                           [-np.sin(pitch), 0, np.cos(pitch)]])
            target, _ = labeler.update(rz @ np.array([.6, 0, 0]), (rz @ ry)[:, 1])
            np.testing.assert_allclose(target, [.6, 0], atol=1e-6)

    def test_stop_masks_turn_even_when_ema_is_still_moving(self):
        labeler = MotionLabeler(EstimatorConfig())
        labeler.update([1, 0, 0], [0, 1, 0])
        _, mask = labeler.update([0, 0, 0], [0, 1, 0])
        self.assertFalse(mask[1])
        _, mask = labeler.update([1, 0, 0], [0, 1, 0])
        self.assertFalse(mask[1])
        labeler.reset()
        _, mask = labeler.update([0, 1, 0], [0, 0, 1])
        self.assertFalse(mask.any())

    def test_causal_filter_matches_analytic_ema(self):
        c = EstimatorConfig(control_dt=.02, velocity_filter_tau_s=.1)
        labeler = MotionLabeler(c)
        labeler.update([1, 0, 0], [0, 1, 0])
        target, mask = labeler.update([0, 1, 0], [0, 1, 0])
        alpha = 1 - np.exp(-.02 / .1)
        expected = np.arctan2(alpha, 1 - alpha) / .02
        self.assertTrue(mask[1])
        self.assertAlmostEqual(float(target[1]), expected, places=5)


class DataAndRuntimeTest(unittest.TestCase):
    def test_command_invariance_and_reset(self):
        one, two = ObservationHistory(2), ObservationHistory(2)
        for i in range(3):
            frame = np.full(36, i, dtype=np.float32)
            other = frame.copy()
            other[6:12] = 1000
            x, ready = one.update(frame)
            y, _ = two.update(other)
            np.testing.assert_array_equal(x, y)
            self.assertEqual(ready, i >= 1)
        one.reset()
        self.assertFalse(one.update(frame)[1])

    def test_offline_windows_equal_streaming_and_never_cross_episodes(self):
        config = EstimatorConfig(history=3, control_dt=.02)
        rng = np.random.default_rng(22)
        frames = rng.normal(size=(30, 36)).astype(np.float32)
        episode = np.repeat(np.arange(5), 6)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.npz"
            np.savez(path, frames=frames, episode=episode, time_s=np.tile(np.arange(6) * .02, 5),
                     velocity_world=np.tile([1., 0, 0], (30, 1)),
                     body_y_world=np.tile([0., 1, 0], (30, 1)),
                     metadata_json=np.array(json.dumps({"control_dt": .02})))
            data = prepare_dataset(path, config)
            self.assertEqual(len(data["x"]), 20)
            self.assertTrue(np.all(data["mask"]))
            layers = [(rng.normal(size=(90, 8)).astype(np.float32), np.zeros(8, np.float32)),
                      (rng.normal(size=(8, 2)).astype(np.float32), np.zeros(2, np.float32))]
            model_path = Path(temp) / "estimator.npz"
            save_estimator(model_path, config, layers, np.zeros(90), np.ones(90),
                           np.zeros(2), np.ones(2), {})
            model = RollingVelocityEstimator(model_path)
            online = []
            for ep in range(5):
                model.reset()
                for frame in frames[episode == ep]:
                    prediction = model.update(frame)
                    if prediction is not None:
                        online.append(prediction)
            np.testing.assert_allclose(online, model.predict_features(data["x"]), atol=1e-5)
            split = split_episodes(data["episode"])
            sets = [set(data["episode"][rows]) for rows in split.values()]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            self.assertEqual(set.union(*sets), set(range(5)))
            with self.assertRaisesRegex(ValueError, "control_dt"):
                prepare_dataset(path, EstimatorConfig(control_dt=.01))

    def test_root_velocity_reference_with_offset_center_of_mass(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0" timestep="0.000001"/>
            <worldbody><body name="torso"><freejoint name="root"/>
            <geom type="sphere" size="0.1" pos="0.3 0 0" mass="1"/>
            </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[:] = [1, 2, 0, 0, 0, 3]
        mujoco.mj_forward(model, data)
        start = data.xpos[1].copy()
        root_velocity = data.qvel[:3].copy()
        self.assertGreater(np.linalg.norm(data.cvel[1, 3:] - root_velocity), .5)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose((data.xpos[1] - start) / model.opt.timestep,
                                   root_velocity, atol=1e-5)


class CEMCollectionTest(unittest.TestCase):
    def test_nominal_target_matches_existing_cem_replay(self):
        from scripts import collect_rolling_velocity_data as collector
        from scripts import evaluate_3d_symmetric_cem_reference as bridge
        from curl_robot_2d_mjx.cem_reference import load_cem_reference
        bridge.activate_planar_geometry(bridge.PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
        reference = load_cem_reference(collector.DEFAULT_CONTROLLER)
        compact = np.zeros(12)
        compact[collector.ACTIVE] = bridge.map_planar_to_curl_3d_targets(bridge.PLANAR_COMPACT)
        for phase in np.linspace(0, 2 * np.pi, 9):
            expected = bridge.map_planar_to_curl_3d_targets(bridge.planar_cem_target(phase, reference))
            actual = collector.cem_motor_target(phase, reference, compact,
                                                np.full(12, -10.), np.full(12, 10.), 1., 0.)
            np.testing.assert_allclose(actual[collector.ACTIVE], expected, atol=1e-12)

    def test_cem_action_is_actual_clipped_motor_target(self):
        from scripts import collect_rolling_velocity_data as collector
        from scripts import evaluate_3d_symmetric_cem_reference as bridge
        from curl_robot_2d_mjx.cem_reference import load_cem_reference
        bridge.activate_planar_geometry(bridge.PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
        reference = load_cem_reference(collector.DEFAULT_CONTROLLER)
        compact = np.tile([.1, .2, .5], 4)
        lower, upper = compact - .1, compact + .1
        target = collector.cem_motor_target(1., reference, compact, lower, upper, 1., .2)
        action = collector.encode_motor_target(target, compact)
        np.testing.assert_allclose(compact + action * collector.ACTION_SCALES, target, atol=1e-7)
        np.testing.assert_array_equal(action[::3], 0.)
        self.assertTrue(np.all(target >= lower) and np.all(target <= upper))

    def test_collected_frames_match_recorded_cem_targets_and_time(self):
        from scripts import collect_rolling_velocity_data as collector
        args = collector.parse_args(["--out", "unused.npz", "--duration", "0.5",
                                     "--dr-strength", "0", "--observation-noise", "0"])
        settings = {k: v for k, v in vars(args).items() if k not in ("controller", "out", "model")}
        settings.update(controllers=[str(args.controller[0])], model=str(args.model))
        records, summary = collector.collect_episode((settings, 0))
        compact = np.asarray(summary["compact_joint_position"])
        np.testing.assert_allclose(
            compact + records["frames"][:, 24:36] * collector.ACTION_SCALES,
            records["motor_target"], atol=1e-7,
        )
        np.testing.assert_allclose(np.diff(records["time_s"]), 1 / 52., atol=1e-10)
        self.assertEqual(summary["source"], "cem")
        self.assertGreater(np.ptp(records["cem_phase"]), 0.)
        self.assertGreater(np.linalg.norm(records["velocity_world"], axis=1).max(), .01)


if __name__ == "__main__":
    unittest.main()

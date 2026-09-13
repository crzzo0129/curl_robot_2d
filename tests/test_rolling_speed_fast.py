"""Cloud-side numerical/contract checks; no robot or simulator rollouts."""
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

from curl_robot_2d_mjx.rolling_speed_tracking import (
    update_speed_window, update_speed_settling, select_speed_checkpoint, speed_checkpoint_eligible)
from scripts.run_rolling_speed_fast import build_command


class SpeedTrackingTest(unittest.TestCase):
    def test_window_rejects_warmup_and_ramp_contamination(self):
        buffer = np.zeros((1,3,3))
        for step in range(1,4):
            buffer,error,valid = update_speed_window(np,buffer,np.array([.7]),np.array([.6]),
                np.array([float(step==1)]),np.array([step]))
            self.assertFalse(valid[0])
        _,error,valid = update_speed_window(np,buffer,np.array([.7]),np.array([.6]),np.array([0.]),np.array([4]))
        self.assertTrue(valid[0])
        self.assertAlmostEqual(error[0],.1)

    def test_brief_crossing_and_failure_do_not_count_as_settling(self):
        streak,first = np.array([0]),np.array([-1.])
        for step,(error,healthy) in enumerate([(0.,True),(.1,True),(0.,True),(0.,False),(0.,True),(0.,True),(0.,True)],1):
            streak,first,_ = update_speed_settling(np,streak,first,np.array([error]),np.array([True]),
                np.array([healthy]),step*.02,dt=.02,hold_steps=3)
            if step<7: self.assertEqual(first[0],-1.)
        self.assertAlmostEqual(first[0],.1)

    def test_speed_selection_preserves_survival(self):
        def row(step,error,success=1.,full=1.):
            return dict(step=step,success_rate=success,yaw_mae_rad_s=.05,
                tracking_by_reset_source={'handoff':dict(episodes=100,full_horizon_rate=full,
                    windowed_forward_mae_m_s=error,steady_forward_mae_m_s=.12),
                    'mature':dict(episodes=100,full_horizon_rate=1.)})
        baseline,good,unsafe = row(0,.10),row(1,.08,.98),row(2,.01,1.,.95)
        self.assertFalse(speed_checkpoint_eligible(unsafe,baseline,.03))
        self.assertEqual(select_speed_checkpoint([baseline,good,unsafe])['step'],1)
        self.assertEqual(select_speed_checkpoint([baseline,row(3,.11)])['step'],0)

    def test_runner_command_is_accepted_and_restores_best_actor(self):
        from scripts.train_mjx_3d_roll_student_dr_ppo import parse_args
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); checkpoint=root/'checkpoints/000000737280';checkpoint.mkdir(parents=True)
            for name in ('student_params','params'): (checkpoint/name).write_bytes(b'placeholder')
            bank=root/'bank.npz';bank.write_bytes(b'placeholder')
            controller=root/'cem.json';controller.write_text('{}')
            saved={'args':dict(forward_velocity_frame='heading',handoff_bank=str(bank),
                command_conditioned=True,rolling_snapshots=True,controller=str(controller),
                forward_command_min_m_s=.45,forward_command_max_m_s=.75,
                preset='h200',fixed_eval_envs=256,episode_length=500,command_interval_s=10)}
            cmd=build_command(saved,source=root,step=737280,bank=bank,out=root/'out',steps=491520,
                              learning_rate=1e-5,max_learning_rate=3e-5)
            args=parse_args(cmd[4:])
            self.assertEqual(args.restore_ppo,checkpoint/'params')
            self.assertFalse(args.critic_only)
            self.assertEqual(args.max_learning_rate,3e-5)
            self.assertTrue(args.gradient_diagnostics)
            self.assertEqual(args.checkpoint_selection,'handoff_speed')

    def test_gradient_diagnostics_preserve_gradients_and_restore_hook_on_failure(self):
        import jax.numpy as jp
        from curl_robot_2d_mjx.rolling_ppo_gradient_diagnostics import train_with_gradient_diagnostics
        Gradients=namedtuple('Gradients','policy value')
        gradients=Gradients({'x':jp.array([3.])},{'x':jp.array([4.])})
        original=SimpleNamespace(loss_and_pgrad=lambda *a,**kw: lambda *a: ((jp.array(1.),{}),gradients))
        ppo=SimpleNamespace(gradients=original)
        def train(**kwargs):
            (_,metrics),actual=ppo.gradients.loss_and_pgrad(None,has_aux=True)()
            self.assertIs(actual,gradients)
            self.assertAlmostEqual(float(metrics['actor_grad_norm']),3.)
            self.assertAlmostEqual(float(metrics['critic_grad_norm']),4.)
            self.assertAlmostEqual(float(metrics['grad_clip_scale']),.1,places=6)
            raise RuntimeError('expected test interruption')
        ppo.train=train
        with self.assertRaisesRegex(RuntimeError,'expected test interruption'):
            train_with_gradient_diagnostics(ppo,enabled=True,max_grad_norm=.5)
        self.assertIs(ppo.gradients,original)


if __name__ == '__main__': unittest.main()

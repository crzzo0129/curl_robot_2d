"""CPU-only sinusoidal walker and CEM objective for the RollingQuad 2 CAD model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets/rollingquad_description_2/mjcf/rollingquad.xml"
LEGS = ("front_left", "front_right", "rear_left", "rear_right")


@dataclass(frozen=True)
class GaitParameters:
    frequency_hz: float = 2.0
    stride_m: float = 0.035
    lift_m: float = 0.025
    front_hip_bias: float = 0.0
    rear_hip_bias: float = 0.0
    front_knee_bias: float = 0.0
    rear_knee_bias: float = 0.0
    rear_phase: float = 0.5
    pitch_gain: float = 0.025
    roll_gain: float = 0.025
    pitch_rate_gain: float = 0.003
    roll_rate_gain: float = 0.003

    def vector(self):
        return np.array(list(asdict(self).values()))

    @classmethod
    def from_vector(cls, vector):
        vector = np.asarray(vector, dtype=float)
        if vector.shape != (len(BOUNDS),) or not np.isfinite(vector).all():
            raise ValueError("Expected twelve finite gait parameters")
        if np.any(vector < BOUNDS[:, 0]) or np.any(vector > BOUNDS[:, 1]):
            raise ValueError("Gait parameters outside supported bounds")
        return cls(*vector.tolist())


BOUNDS = np.array([
    [0.8, 3.5], [0.0, 0.10], [0.005, 0.05],
    [-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.4], [-0.3, 0.4],
    [0.35, 0.65], [-0.06, 0.06], [-0.06, 0.06],
    [-0.015, 0.015], [-0.015, 0.015],
])


class SineWalkingController:
    """50 Hz joint-position targets; exact stand reset, then a smooth gait ramp.

    Foot displacements are mapped using the CAD model's stand Jacobians.
    CEM tunes this small periodic policy offline, not an online MPC sequence.
    All arrays exposed by targets() use model actuator order.
    """

    def __init__(self, model, parameters=None, *, hold_s=0.5, ramp_s=1.0, heading_feedback=False):
        self.model = model
        self.parameters = parameters or GaitParameters()
        GaitParameters.from_vector(self.parameters.vector())
        if not math.isfinite(hold_s) or hold_s < 0 or not math.isfinite(ramp_s) or ramp_s <= 0:
            raise ValueError("hold_s must be nonnegative and ramp_s positive")
        self.hold_s, self.ramp_s = hold_s, ramp_s
        self.heading_feedback = heading_feedback
        self.key = model.key("stand").id
        self.torso = model.body("torso").id
        self.nominal = model.key_ctrl[self.key].copy()
        self.actuators = np.array([[model.actuator(f"{leg}_{joint}_servo").id
                                  for joint in ("hip_abduction", "hip", "knee")]
                                 for leg in LEGS])
        joint_ids = model.actuator_trnid[self.actuators, 0]
        self.qadr = model.jnt_qposadr[joint_ids]
        self.dadr = model.jnt_dofadr[joint_ids]
        self.lower = np.maximum(model.actuator_ctrlrange[:, 0],
                                model.jnt_range[model.actuator_trnid[:, 0], 0])
        self.upper = np.minimum(model.actuator_ctrlrange[:, 1],
                                model.jnt_range[model.actuator_trnid[:, 0], 1])
        reference = mujoco.MjData(model)
        self.reset(reference)
        self.inverse_jacobians = []
        for i, leg in enumerate(LEGS):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, reference, jac, None, model.site(f"{leg}_foot_site").id)
            self.inverse_jacobians.append(np.linalg.pinv(jac[:, self.dadr[i]], rcond=1e-4))
        self.inverse_jacobians = np.array(self.inverse_jacobians)

    def reset(self, data):
        mujoco.mj_resetDataKeyframe(self.model, data, self.key)
        data.ctrl[:] = self.nominal
        mujoco.mj_forward(self.model, data)

    def targets(self, time_s, data=None):
        p = self.parameters
        t = max(0.0, float(time_s) - self.hold_s)
        u = min(t / self.ramp_s, 1.0)
        blend = u * u * (3 - 2 * u)
        if blend == 0:
            return self.nominal.copy()
        phase = 2 * math.pi * (t * p.frequency_hz + np.array([0, 0.5, p.rear_phase, p.rear_phase + 0.5]))
        displacement = np.zeros((4, 3))
        # During sin(phase)>0 the foot swings forward with a positive lift.
        displacement[:, 0] = -0.5 * p.stride_m * np.cos(phase)
        displacement[:, 2] = p.lift_m * np.maximum(np.sin(phase), 0.0) ** 2
        if data is not None:
            rotation = data.xmat[self.torso].reshape(3, 3)
            pitch = math.asin(float(np.clip(-rotation[2, 0], -1, 1)))
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            # Differential stride closes the heading/lateral loop about +world X.
            heading_error = np.clip(yaw + math.atan2(float(data.qpos[1]), 0.75), -0.6, 0.6)
            if not self.heading_feedback:
                heading_error = 0.0
            displacement[:, 0] -= 0.03 * heading_error * np.cos(phase) * np.array([1, -1, 1, -1])
            displacement[:, 1] += 0.02 * heading_error * np.array([1, 1, -1, -1])
            pitch_correction = np.clip(p.pitch_gain * pitch + p.pitch_rate_gain * data.qvel[4], -0.025, 0.025)
            roll_correction = np.clip(p.roll_gain * roll + p.roll_rate_gain * data.qvel[3], -0.025, 0.025)
            displacement[:, 2] += pitch_correction * np.array([1, 1, -1, -1])
            displacement[:, 2] += roll_correction * np.array([-1, 1, -1, 1])
        delta = np.einsum("ijk,ik->ij", self.inverse_jacobians, displacement)
        delta[:, 1] += [p.front_hip_bias] * 2 + [p.rear_hip_bias] * 2
        delta[:, 2] += [p.front_knee_bias] * 2 + [p.rear_knee_bias] * 2
        result = self.nominal.copy()
        result[self.actuators] += blend * delta
        return np.clip(result, self.lower, self.upper)


def rollout(model, controller, *, duration_s=10.0, target_speed=0.15, record=False, callback=None):
    if not math.isfinite(duration_s) or duration_s <= controller.hold_s + controller.ramp_s:
        raise ValueError("duration must exceed hold + ramp")
    if not math.isfinite(target_speed) or target_speed <= 0:
        raise ValueError("target_speed must be finite and positive")
    data = mujoco.MjData(model)
    controller.reset(data)
    control_steps = round(0.02 / model.opt.timestep)
    if control_steps < 1 or not math.isclose(control_steps * model.opt.timestep, 0.02, abs_tol=1e-8):
        raise ValueError("model timestep must divide the 0.02 s control interval")
    floor = model.geom("floor").id
    feet = {model.geom(f"{leg}_foot_proxy").id for leg in LEGS}
    foot_ids = [model.geom(f"{leg}_foot_proxy").id for leg in LEGS]
    previous_contact = np.ones(4, dtype=bool)
    liftoffs = np.zeros(4, dtype=int)
    swing_samples = np.zeros(4, dtype=int)
    rows, controls, times = [data.qpos.copy()], [data.ctrl.copy()], [0.0]
    squared_tilt = lateral = speed_error = effort = nonfoot = 0.0
    max_tilt, min_height, count = 0.0, float(data.qpos[2]), 0
    failed, reason = False, ""
    initial = data.qpos[:3].copy()
    steady_start_x = None
    steady_start_time = controller.hold_s + controller.ramp_s
    target = data.ctrl.copy()
    for _ in range(math.ceil(duration_s / 0.02)):
        target = controller.targets(data.time, data)
        data.ctrl[:] = target
        mujoco.mj_step(model, data, nstep=control_steps)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.time < count * 0.02:
            failed, reason = True, "nonfinite state or physics reset"
            break
        tilt = math.acos(float(np.clip(data.xmat[controller.torso].reshape(3, 3)[2, 2], -1, 1)))
        max_tilt = max(max_tilt, tilt)
        min_height = min(min_height, float(data.qpos[2]))
        bad_contact = any(floor in (c.geom1, c.geom2) and
                          (c.geom2 if c.geom1 == floor else c.geom1) not in feet
                          for c in data.contact)
        nonfoot += float(bad_contact)
        contacting = {int(c.geom2 if c.geom1 == floor else c.geom1)
                      for c in data.contact if floor in (c.geom1, c.geom2)}
        foot_contact = np.array([f in contacting for f in foot_ids])
        if data.time > controller.hold_s + controller.ramp_s:
            liftoffs += previous_contact & ~foot_contact
            swing_samples += ~foot_contact
        previous_contact = foot_contact
        squared_tilt += tilt * tilt
        lateral += float(data.qpos[1] ** 2 + 0.1 * data.qvel[1] ** 2)
        desired = target_speed * min(max((data.time - controller.hold_s) / controller.ramp_s, 0), 1)
        speed_error += float((data.qvel[0] - desired) ** 2)
        effort += float(np.mean(data.actuator_force ** 2))
        count += 1
        if steady_start_x is None and data.time >= steady_start_time:
            steady_start_x = float(data.qpos[0])
        if record:
            rows.append(data.qpos.copy()); controls.append(target.copy()); times.append(float(data.time))
        if tilt > math.radians(50) or data.qpos[2] < 0.085:
            failed, reason = True, "fall: tilt > 50 deg or base height < 0.085 m"
            break
        if callback is not None and callback(model, data) is False:
            reason = "viewer closed"
            break
    n = max(count, 1)
    dx = float(data.qpos[0] - initial[0])
    completed = not failed and data.time >= duration_s - 1e-8
    score = (4 * dx / duration_s - 12 * speed_error / n - 3 * squared_tilt / n
             - 12 * lateral / n - 4 * nonfoot / n - 0.002 * effort / n
             - (10 + 5 * (1 - data.time / duration_s) if failed else 0))
    summary = dict(score=float(score), completed=completed, failed=failed, failure_reason=reason,
                   duration_s=float(data.time), requested_duration_s=duration_s, distance_x_m=dx,
                   drift_y_m=float(data.qpos[1] - initial[1]), mean_speed_m_s=dx / max(data.time, 0.02),
                   steady_speed_m_s=(float(data.qpos[0]) - steady_start_x) / max(data.time - steady_start_time, 0.02)
                   if steady_start_x is not None else 0.0,
                   max_tilt_deg=math.degrees(max_tilt), min_base_height_m=min_height,
                   nonfoot_contact_fraction=nonfoot / n, target_speed_m_s=target_speed)
    summary["foot_liftoffs"] = dict(zip(LEGS, liftoffs.tolist()))
    summary["foot_swing_samples"] = dict(zip(LEGS, swing_samples.tolist()))
    return summary, dict(qpos=np.asarray(rows), ctrl=np.asarray(controls), time=np.asarray(times))

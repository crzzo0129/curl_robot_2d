# RollingQuad MJCF

This directory contains two MuJoCo models derived from
`../urdf/rollingquad_description_2.urdf`:

- `rollingquad_from_urdf.xml`: the direct MuJoCo URDF import. It is kept as a
  traceable reference and has a fixed root with no actuators.
- `rollingquad.xml`: the simulation/training model. It adds a free base, floor,
  position servos, state sensors, foot sites, contact settings and reset
  keyframes.

## Preserved robot data

The training model preserves the URDF mesh geometry, link transforms, joint
origins, joint axes, joint limits, masses and inertias. In particular, the four
inclined abduction axes remain unchanged:

| MJCF joint | URDF joint | Axis in the joint/body frame |
| --- | --- | --- |
| `front_right_hip_abduction` | `abduction_mirror_1__Upperleg_with_motor_2` | `0.872069 -0.489382 0` |
| `rear_left_hip_abduction` | `abduction_mirror_2__Upperleg_with_motor_1` | `-0.872069 -0.489382 0` |
| `front_left_hip_abduction` | `abduction_1__Upperleg_with_motor_mirror_1` | `-0.872069 0.489382 0` |
| `rear_right_hip_abduction` | `abduction_9__Upperleg_with_motor_mirror_2` | `0.872069 0.489382 0` |

This corrected URDF uses the reference model's physical flexion directions:
both front hip axes point along world `-Y`, both rear hip axes point along
world `+Y`, both front knee axes point along world `+Y`, and both rear knee
axes point along world `-Y`. The original package URI paths are retained in
the source URDF; `../urdf/rollingquad_description_2.mujoco.urdf` is the same
model with MuJoCo-loadable relative mesh paths.

Leg names are assigned from the abduction joint mounting location on the base
(`+x` front, `+y` left). At the exported zero pose, a lower leg may cross the
base centerline; this is intentionally not used to rename the leg.

## Training interface

The model exposes 12 position servos in the policy's canonical order:

1. front left: abduction, hip, knee
2. front right: abduction, hip, knee
3. rear left: abduction, hip, knee
4. rear right: abduction, hip, knee

The body hierarchy and `qpos` retain the URDF's original `FR, RL, FL, RR`
order. Keyframe controls are independently remapped to the canonical actuator
order, so policy actions and named keyframe poses address the same legs.

Following `curl_robot_3d_pupper_r127p5_open60_width120.xml`, every servo uses
`kp=5`, `kd=0.1` and a `[-3, 3] N m` output-force limit. Control ranges follow
the corresponding URDF joint limits. Latency, gearbox friction and electrical
limits still need to be identified separately before a final sim-to-real run.

The `open`, `stand_previous`, `park` and `compact` keyframes are ported from
the reference XML. Their joint values are remapped from the reference order
`(abduction, hip, knee)` into this URDF's kinematic order
`(hip, abduction, knee)`. The active walking `stand` is a mildly crouched
`(abduction, hip, knee) = (0, 0.90, 1.15)` pose with a root height of
`0.1580029248 m`; this leaves about 1 mm of CAD floor clearance and provides
extension travel for swing-foot lift. Root heights are recomputed against the
exported STL geometry; this notably changes the stale `park` height from the
reference model. The original `open` pose sets every joint to zero; this places
the knee below the URDF lower limit, so `extended` is also provided as a
limit-valid reset pose.

Robot mesh geoms collide with the floor but not with one another. This avoids
expensive and often unwanted self-contact while retaining the exported outer
geometry for rolling and ground contact.

## Quick load check

```python
import mujoco

model = mujoco.MjModel.from_xml_path("assets/rollingquad_description_2/mjcf/rollingquad.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
```

From the `curl_robot_2d` project root, this is the fixed source model used by
`python -m scripts.train_ppo_walk3d` and
`python -m scripts.train_ppo_deploy`.  Both scripts remap the URDF body-tree
qpos order to the canonical policy/controller joint order by name.

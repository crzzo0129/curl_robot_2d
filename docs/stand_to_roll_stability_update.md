# Crouch stability update (cloud validation required)

No local simulation, training or tests were run for this change.

At each control step (dt=0.02 s), the additional reward is:

    -4.0 * (root_y / 0.60)^2 * dt + 1.0 * sustain_seconds

Thus a 0.3 m offset costs 1 reward/second and a 0.6 m offset costs
4 reward/second. Sustain pays at most 1 reward/second, only after capture,
positive angular speed >0.5 rad/s and net positive x speed >0.02 m/s have
held for 4 consecutive control steps. A pause/reversal/failure resets this
counter; no sustain reward is given merely for being alive. Thresholds can
make the term intermittent during periodic contact transitions; inspect
sustain_seconds before judging its effectiveness. Existing progress reward,
capture bonus, failure thresholds and stage-pass criteria are unchanged.

For non-snapshot reset, final joint interpolation/noise is applied first.
Then the minimum height of all robot geometries eligible for floor collision
is computed, using compiled mesh vertices or exact primitive support. Only
root z changes to leave 0.5 mm floor clearance. Velocity is unchanged (zero
under --static-curriculum). No settling dynamics or momentum are injected.
This assumes a horizontal world-body plane and guarantees geometric placement,
not static balance. Recorded dynamic snapshots retain their original height.
The startup collector uses the same correction for future data collection.

New episode metrics:

- reward_lateral: summed negative reward contribution.
- reward_sustain and sustain_seconds: earned sustained forward rolling time.
- lateral_cost: summed squared normalized offset (divide by episode length
  before interpreting as a time average).
- reset_z_correction_m: signed correction, emitted only on the first step.
- reset_floor_gap_m: initial minimum clearance, emitted only on the first step;
  static episodes should report approximately 0.0005 m.

Restart crouch from the previously passing slightly_open checkpoint into a
NEW output directory, with the same BC parameters, learning rate 5e-6,
--max-kl 0.2 and --static-curriculum. The weights remain compatible, but the
reset distribution and reward changed; compare initial step=0 evaluation
before attributing changes to PPO updates. Old BC data/checkpoints are not
retroactively height-corrected. Do not overwrite or regenerate them blindly.

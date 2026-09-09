"""Check reference provenance and nominal dynamics before dynamic training."""

import json
import math
from pathlib import Path

PHYSICS_FIELDS = (
    "geometry", "physics_timestep", "action_repeat", "solver_name",
    "integrator_name", "cone_name", "jacobian_name", "solver_iterations",
    "solver_ls_iterations", "geom_friction_scale", "floor_friction_scale",
    "floor_contact_friction_override", "body_mass_scale", "body_mass_left_scale",
    "body_mass_right_scale", "actuator_gain_scale", "disable_root_damping",
)


def validate_reference_bank(path, task):
    path = Path(path)
    report = json.loads(path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    source_kind = report.get("source_kind")
    # Residual training is defined relative to the +90° handoff ctrl, so it
    # only accepts the handcrafted +90° bank. Absolute (non-residual) training
    # resets from the same +90° handoff states, so it accepts both that bank
    # and the legacy rolling-reference bank.
    if task.handcrafted_reference_residual:
        if source_kind != "handcrafted_roll_to_stand_90_reference":
            raise ValueError(
                "handcrafted reference training requires a qualified "
                "handcrafted_roll_to_stand_90_reference bank"
            )
    elif source_kind not in ("cem_reference_zero_residual",
                             "handcrafted_roll_to_stand_90_reference",
                             "roll_to_stand_phase_window_reference"):
        raise ValueError(
            "dynamic training requires a qualified "
            "cem_reference_zero_residual or handcrafted_roll_to_stand_90_reference bank"
        )
    if report.get("status") != "ok":
        raise ValueError("reference bank is not marked ok")
    if source_kind == "roll_to_stand_phase_window_reference":
        rows = report.get("handoffs", [])
        targets = report.get("pitch_targets_deg", [])
        if (not rows or not targets or report.get("stand_abduction_deg") != [0.0]*4
                or any(not math.isfinite(p) or not 80 <= p <= 100 for p in targets)):
            raise ValueError("invalid phase-window bank metadata")
        counts = [sum(r.get("target_pitch_deg") == p for r in rows) for p in targets]
        if min(counts) < 1 or len(set(counts)) != 1 or sum(counts) != len(rows):
            raise ValueError("phase-window bank must balance all target pitches")
        for row in rows:
            if (not math.isfinite(row["pitch_deg"])
                    or abs(row["pitch_deg"] - row["target_pitch_deg"]) > 1.0
                    or not math.isfinite(row["pitch_rate_rad_s"]) or row["pitch_rate_rad_s"] >= 0):
                raise ValueError("invalid measured phase or roll direction")
    if source_kind == "handcrafted_roll_to_stand_90_reference":
        if report.get("trigger_pitch_deg") != 90.0:
            raise ValueError("handcrafted reference bank must trigger at +90 degrees")
        if report.get("stand_abduction_deg") != [0.0] * 4:
            raise ValueError("handcrafted reference bank must target zero stand abduction")
    mismatches = [name for name in PHYSICS_FIELDS
                  if report.get("task", {}).get(name) != getattr(task, name)]
    if mismatches:
        raise ValueError(f"reference and recovery dynamics differ: {mismatches}")
    return report


def validate_reference_split(train_path, eval_path, task):
    train = validate_reference_bank(train_path, task)
    evaluation = validate_reference_bank(eval_path, task)
    if train["reference_sha256"] != evaluation["reference_sha256"]:
        raise ValueError("train/eval reference controllers differ")
    if train["seed"] == evaluation["seed"]:
        raise ValueError("train/eval reference collection seeds overlap")
    if "minimum_turns" in (train.get("handoffs") or [{}])[0] and "minimum_turns" in (evaluation.get("handoffs") or [{}])[0]:
        train_cycles = {r["minimum_turns"] for r in train["handoffs"]}
        eval_cycles = {r["minimum_turns"] for r in evaluation["handoffs"]}
        if train_cycles & eval_cycles:
            raise ValueError("deterministic train/eval rolling cycles overlap")
    return {"train_seed": train["seed"], "eval_seed": evaluation["seed"],
            "reference_sha256": train["reference_sha256"], "nominal_dynamics_match": True}

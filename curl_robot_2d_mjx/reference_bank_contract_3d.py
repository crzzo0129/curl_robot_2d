"""Check reference provenance and nominal dynamics before dynamic training."""

import json
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
    if report.get("source_kind") != "cem_reference_zero_residual" or report.get("status") != "ok":
        raise ValueError("dynamic training requires a qualified reference bank, not residual-policy states")
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
    return {"train_seed": train["seed"], "eval_seed": evaluation["seed"],
            "reference_sha256": train["reference_sha256"], "nominal_dynamics_match": True}

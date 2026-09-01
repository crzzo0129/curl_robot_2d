"""Dependency-light Transition initialization and curriculum acceptance rules."""

import math
from pathlib import Path


TRANSITION_TRAINING_REVISION = "v5_failure_breakdown"
TRANSITION_INITIAL_POLICY_STD = 0.05  # Pre-tanh Gaussian std; action ABI unchanged.


def resolve_transition_checkpoint(path):
    """Accept a completed Brax step directory or select latest completed child.

    Brax writes ppo_network_config.json AFTER Orbax save completes. Never pick
    temp/non-numeric children or mistake params_final for an Orbax checkpoint.
    """
    path = Path(path).resolve()
    complete = lambda p: (p.is_dir() and (p / "_METADATA").is_file()
                          and (p / "ppo_network_config.json").is_file())
    if complete(path):
        return path
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}; use ppo_checkpoint, not params_final")
    if path.name.isdigit() or (path / "_METADATA").exists():
        raise ValueError(f"incomplete checkpoint directory: {path}")
    candidates = [p for p in path.iterdir() if p.name.isascii() and p.name.isdigit() and complete(p)]
    if not candidates:
        raise ValueError(f"no completed numeric Brax checkpoints under {path}")
    return max(candidates, key=lambda p: int(p.name))


def transition_scale_logit(initial_std):
    # Brax 0.14 NormalTanhDistribution: softplus(scale_logit) + 0.001.
    if not math.isfinite(initial_std) or initial_std <= 0.001:
        raise ValueError("initial_policy_std must be finite and greater than 0.001")
    adjusted = initial_std - 0.001
    return adjusted if adjusted > 20 else math.log(math.expm1(adjusted))


def initialize_transition_actor(xp, params, hidden_layers, action_size=12,
                                initial_std=TRANSITION_INITIAL_POLICY_STD):
    """Replace ONLY fresh actor's final dense initialization, never a restore.

    Keep Brax's single 24-logit dense head and parameter names unchanged so
    ELU/tanh-location export remains identical. Random hidden layers are kept.
    """
    scale_logit = transition_scale_logit(initial_std)
    name = f"hidden_{len(hidden_layers)}"
    head = params["params"][name]
    if head["kernel"].shape[-1] != 2 * action_size or head["bias"].shape != (2 * action_size,):
        raise ValueError("expected Brax tanh-normal combined location/scale dense head")
    bias = xp.concatenate((xp.zeros(action_size, dtype=head["bias"].dtype),
                           xp.full(action_size, scale_logit, dtype=head["bias"].dtype)))
    return {**params, "params": {**params["params"], name: {
        **head, "kernel": xp.zeros_like(head["kernel"]), "bias": bias}}}


def transition_curriculum_acceptance(history, *, required_evals=2,
                                      minimum_success=0.9, maximum_failure=0.05,
                                      maximum_timeout=0.05):
    """Only consecutive post-training evaluations can recommend advancing.

    These are empirical training gates, not a hardware safety certificate.
    The three event metrics must be mutually exclusive episode-end pulses.
    """
    evaluations = [row for row in history if row.get("step", 0) > 0
                   and "eval/episode_transition_success" in row]
    recent = evaluations[-required_evals:]
    reasons = []
    if len(recent) < required_evals:
        reasons.append("insufficient_post_training_evaluations")
    for row in recent:
        values = [row.get(key, float("nan")) for key in (
            "eval/episode_transition_success", "eval/episode_failed", "eval/episode_timeout")]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            reasons.append("invalid_outcome_rates")
            continue
        success, failure, timeout = values
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            reasons.append("outcome_rates_do_not_sum_to_one")
        if success < minimum_success:
            reasons.append("success_below_threshold")
        if failure > maximum_failure:
            reasons.append("failure_above_threshold")
        if timeout > maximum_timeout:
            reasons.append("timeout_above_threshold")
    return {"passed": not reasons, "reasons": sorted(set(reasons)),
            "evaluated_steps": [row["step"] for row in recent],
            "required_evals": required_evals, "minimum_success": minimum_success,
            "maximum_failure": maximum_failure, "maximum_timeout": maximum_timeout}

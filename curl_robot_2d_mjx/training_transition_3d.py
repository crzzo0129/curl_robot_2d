"""Dependency-light Transition initialization and curriculum acceptance rules."""

import math


TRANSITION_TRAINING_REVISION = "v4_full_reset_standing"
TRANSITION_INITIAL_POLICY_STD = 0.05  # Pre-tanh Gaussian std; action ABI unchanged.


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

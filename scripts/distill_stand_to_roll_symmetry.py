"""Cloud-only action distillation from a frozen, left/right averaged PPO teacher."""
from pathlib import Path
import json


def distill(env, networks, params, *, out: Path, count: int, updates: int, seed: int):
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax
    from brax.io import model as model_io

    # Brax inference parameters: normalization, actor, and optionally critic.
    if not isinstance(params, (tuple, list)) or len(params) < 2:
        raise ValueError("Expected Brax inference parameters (normalizer, actor, ...)")
    actor_initial = jax.tree_util.tree_map(jp.asarray, params[1])
    partner = jp.asarray([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

    def act(actor, obs):
        logits = networks.policy_network.apply(params[0], actor, obs)
        return networks.parametric_action_distribution.mode(logits)

    @jax.jit
    def rollout(actor, key, use_teacher):
        states = jax.vmap(env.reset)(jax.random.split(key, count))
        alive = jp.ones(count, dtype=bool)
        def step(carry, _):
            states, alive = carry
            raw = act(actor_initial, states.obs)
            labels = (raw + jp.take(raw, partner, axis=-1)) * 0.5
            action = jp.where(use_teacher, labels, act(actor, states.obs))
            nxt = jax.vmap(env.step)(states, action)
            output = (states.obs, labels, alive,
                      {name: jp.where(alive, value, 0.0) for name, value in nxt.metrics.items()})
            nxt = jax.tree_util.tree_map(
                lambda new, old: jp.where(alive.reshape((count,) + (1,) * (new.ndim - 1)), new, old),
                nxt, states)
            return (nxt, alive & (~nxt.done.astype(bool))), output
        return jax.lax.scan(step, (states, alive), None, length=env.config.episode_length)[1]

    def summarize(output):
        values = {name: np.asarray(value).sum(axis=0) for name, value in output[3].items()}
        captured = values["captured"] > 0.5
        angle = values["capture_axis_error_rad"][captured]
        return {
            "capture": float(values["captured"].mean()),
            "insurance": float(values["insurance_success"].mean()),
            "failed": float(values["failed"].mean()),
            "axis_mean_deg": float(np.rad2deg(angle).mean()) if angle.size else 180.0,
            "axis_p95_deg": float(np.percentile(np.rad2deg(angle), 95)) if angle.size else 180.0,
            "axis_under5_fraction_of_captured": float((angle < np.deg2rad(5)).mean()) if angle.size else 0.0,
            "capture_abs_y_m": float(values["capture_abs_y_m"][captured].mean()) if angle.size else None,
            "capture_abs_vy_m_s": float(values["capture_abs_vy_m_s"][captured].mean()) if angle.size else None,
        }

    def evaluate(actor, offset, teacher=False):
        return summarize(rollout(actor, jax.random.PRNGKey(seed + offset), jp.asarray(teacher)))

    def score(report):
        return (report["capture"], report["insurance"], -report["axis_p95_deg"])

    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-5))
    @jax.jit
    def update(actor, opt_state, obs, labels):
        loss, grads = jax.value_and_grad(lambda p: jp.mean(jp.square(act(p, obs) - labels)))(actor)
        delta, opt_state = optimizer.update(grads, opt_state, actor)
        return optax.apply_updates(actor, delta), opt_state, loss

    out.mkdir(parents=True, exist_ok=True)
    report = {"method": "Frozen PPO teacher with averaged actions; student evaluated WITHOUT averaging",
              "episodes_per_batch": count, "updates_per_round": updates, "seed": seed, "rounds": []}
    def save():
        (out / "distillation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("Evaluating original and projected teacher; first compilation may take several minutes...", flush=True)
    report["baseline"] = evaluate(actor_initial, 10000)
    report["teacher"] = evaluate(actor_initial, 10000, True)
    save()
    print("Teacher: " + json.dumps(report["teacher"]), flush=True)
    if (report["teacher"]["capture"] < 0.95 or report["teacher"]["insurance"] < 0.95
            or report["teacher"]["axis_p95_deg"] >= 5):
        report["status"] = "teacher_gate_failed"
        save()
        return
    best_actor, best_report = actor_initial, report["baseline"]
    obs_chunks, label_chunks = [], []
    for iteration in range(3):
        # First use teacher trajectories, then label states visited by the
        # current student. Teacher remains frozen; history uses applied actions.
        output = rollout(best_actor, jax.random.PRNGKey(seed + iteration), jp.asarray(iteration == 0))
        valid = np.asarray(output[2]).reshape(-1)
        obs_chunks.append(np.asarray(output[0]).reshape(-1, 720)[valid])
        label_chunks.append(np.asarray(output[1]).reshape(-1, 12)[valid])
        obs_bank, labels = jp.asarray(np.concatenate(obs_chunks)), jp.asarray(np.concatenate(label_chunks))
        actor = best_actor
        opt_state = optimizer.init(actor)
        rng = jax.random.PRNGKey(seed + 100 + iteration)
        for _ in range(updates):
            rng, key = jax.random.split(rng)
            index = jax.random.randint(key, (256,), 0, len(obs_bank))
            actor, opt_state, loss = update(actor, opt_state, obs_bank[index], labels[index])
        if not np.isfinite(float(loss)):
            raise RuntimeError("Nonfinite distillation loss")
        candidate = evaluate(actor, 10000)
        accepted = score(candidate) > score(best_report)
        if accepted:
            best_actor, best_report = actor, candidate
        report["rounds"].append({"round": iteration + 1, "loss": float(loss),
                                 "accepted": accepted, "student": candidate})
        save()
        print(f"Round {iteration + 1} (no projection): " + json.dumps(candidate), flush=True)

    baseline = evaluate(actor_initial, 20000)
    student = evaluate(best_actor, 20000)
    report.update(baseline_holdout=baseline, student_holdout=student)
    passed = (student["capture"] >= max(0.95, baseline["capture"])
              and student["insurance"] >= max(0.95, baseline["insurance"])
              and student["failed"] <= baseline["failed"]
              and student["axis_p95_deg"] < 5
              and student["axis_p95_deg"] < baseline["axis_p95_deg"])
    report["status"] = "passed" if passed else "student_gate_failed"
    if passed:
        exported = list(params)
        exported[1] = jax.tree_util.tree_map(np.asarray, best_actor)
        model_io.save_params(out / "student_params", tuple(exported))
        report["critic_note"] = "Critic retained from original; only deterministic actor was distilled. No PPO optimizer state exported."
    save()
    print(f"Distillation {report['status']}; report: {out / 'distillation_report.json'}", flush=True)

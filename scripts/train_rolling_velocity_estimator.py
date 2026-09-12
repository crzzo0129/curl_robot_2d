"""Supervised JAX/Optax training; export a standalone NumPy estimator.

No PPO, policy modification, or feedback into the rollout controller.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.rolling_velocity_estimator import (
    EstimatorConfig, OUTPUT_NAMES, RollingVelocityEstimator, prepare_dataset,
    regression_metrics, save_estimator, split_episodes,
)


def evaluate(model, data, rows, constant):
    target, mask = data["y"][rows], data["mask"][rows]
    predicted = model.predict_features(data["x"][rows])
    # World-vertical gyro component is a useful comparator, NOT the definition
    # of trajectory turn rate, particularly in slip or wobble.
    frame = data["x"][rows, :30]
    gyro_baseline = np.broadcast_to(constant, target.shape).copy()
    gyro_baseline[:, 1] = -np.sum(frame[:, :3] * frame[:, 3:6], axis=1)
    groups = {"all": np.ones(len(target), bool),
              "forward": target[:, 0] > .08, "reverse": target[:, 0] < -.08,
              "low_forward_speed": np.abs(target[:, 0]) <= .08,
              "left_turn": mask[:, 1] & (target[:, 1] > .1),
              "right_turn": mask[:, 1] & (target[:, 1] < -.1)}
    if all(key in data for key in ("command", "command_age_s", "command_segment",
                                   "speed_command_delta", "turn_command_delta")):
        changed = data["command_segment"][rows] > 0
        transition = changed & (data["command_age_s"][rows] < .5)
        speed_delta = data["speed_command_delta"][rows]
        turn = data["command"][rows, 2]
        previous_turn = turn - data["turn_command_delta"][rows]
        groups.update(command_transition_0p5s=transition,
                      after_command_transition=changed & ~transition,
                      speed_increase_transition=transition & (speed_delta > 1e-6),
                      speed_decrease_transition=transition & (speed_delta < -1e-6),
                      turn_reversal_transition=transition & (turn * previous_turn < 0),
                      commanded_straight=np.abs(turn) < 1e-6,
                      commanded_left=turn > 0, commanded_right=turn < 0)
    return {
        "estimator": regression_metrics(predicted, target, mask),
        "training_mean_baseline": regression_metrics(np.broadcast_to(constant, target.shape), target, mask),
        "vertical_gyro_baseline": regression_metrics(gyro_baseline, target, mask),
        "groups": {key: regression_metrics(predicted, target, mask & selected[:, None])
                   for key, selected in groups.items()},
    }, predicted


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--history", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--velocity-filter-tau", type=float, default=.06)
    p.add_argument("--min-turn-speed", type=float, default=.08)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    if min(args.epochs, args.batch_size, args.history, *args.hidden) < 1 or args.seed < 0:
        p.error("counts must be positive and seed must be nonnegative")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("learning-rate must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    with np.load(args.data, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    config = EstimatorConfig(history=args.history, control_dt=metadata["control_dt"],
                             velocity_filter_tau_s=args.velocity_filter_tau,
                             min_turn_speed_m_s=args.min_turn_speed)
    data = prepare_dataset(args.data, config)
    splits = split_episodes(data["episode"], args.seed)
    train, valid = splits["train"], splits["validation"]
    for name, rows in splits.items():
        if not np.all(data["mask"][rows].sum(axis=0) > 0):
            raise ValueError(f"{name} lacks valid labels for one output; collect more moving episodes")
    x_mean = data["x"][train].mean(axis=0)
    x_std = np.maximum(data["x"][train].std(axis=0), .01)
    y_mean = np.array([data["y"][train, i][data["mask"][train, i]].mean() for i in range(2)])
    y_std = np.maximum(np.array([data["y"][train, i][data["mask"][train, i]].std()
                                for i in range(2)]), [.05, .1]).astype(np.float32)
    x = ((data["x"] - x_mean) / x_std).astype(np.float32)
    y = ((data["y"] - y_mean) / y_std).astype(np.float32)

    import jax
    import jax.numpy as jnp
    import optax

    rng = np.random.default_rng(args.seed)
    dimensions = [x.shape[1], *args.hidden, 2]
    params = [(jnp.asarray(rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out)).astype(np.float32)),
               jnp.zeros(fan_out)) for fan_in, fan_out in zip(dimensions[:-1], dimensions[1:])]

    def forward(parameters, features):
        for i, (weight, bias) in enumerate(parameters):
            features = features @ weight + bias
            if i + 1 < len(parameters):
                features = jax.nn.relu(features)
        return features

    def loss(parameters, features, labels, masks):
        losses = optax.huber_loss(forward(parameters, features), labels, delta=1.) * masks
        # Normalize each target by its own valid count; a masked turn label
        # does not reduce the speed output's contribution.
        count = masks.sum(axis=0)
        per_output = losses.sum(axis=0) / jnp.maximum(count, 1)
        return per_output.sum() / jnp.maximum((count > 0).sum(), 1)

    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(args.learning_rate, weight_decay=1e-5))
    state = optimizer.init(params)

    @jax.jit
    def update(parameters, opt_state, features, labels, masks):
        value, gradient = jax.value_and_grad(loss)(parameters, features, labels, masks)
        changes, opt_state = optimizer.update(gradient, opt_state, parameters)
        return optax.apply_updates(parameters, changes), opt_state, value

    validation_loss = jax.jit(loss)
    valid_x, valid_y = jnp.asarray(x[valid]), jnp.asarray(y[valid])
    valid_mask = jnp.asarray(data["mask"][valid], jnp.float32)
    train_rows = np.flatnonzero(train)
    best_loss, best_params, best_epoch = np.inf, None, 0
    curve = []
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"devices={jax.devices()}, samples={len(x)}, input={x.shape[1]}, "
          f"split episodes={ {k: len(np.unique(data['episode'][v])) for k, v in splits.items()} }", flush=True)
    for epoch in range(1, args.epochs + 1):
        ordered = rng.permutation(train_rows)
        losses = []
        for start in range(0, len(ordered), args.batch_size):
            rows = ordered[start:start + args.batch_size]
            # Zero-mask padding keeps one compiled batch shape without throwing
            # away training samples or overweighting duplicate examples.
            size = len(rows)
            rows = np.pad(rows, (0, args.batch_size - size), mode="edge")
            mask = data["mask"][rows].astype(np.float32)
            mask[size:] = 0
            params, state, value = update(params, state, x[rows], y[rows], mask)
            losses.append(float(value))
        score = float(validation_loss(params, valid_x, valid_y, valid_mask))
        if not np.isfinite(score) or not np.all(np.isfinite(losses)):
            raise FloatingPointError("nonfinite training loss")
        if score < best_loss:
            best_loss, best_epoch = score, epoch
            best_params = [(np.asarray(w).copy(), np.asarray(b).copy()) for w, b in params]
        curve.append([epoch, float(np.mean(losses)), score])
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch}: train={curve[-1][1]:.5f} val={score:.5f} best={best_epoch}", flush=True)
    fingerprint = hashlib.sha256(args.data.read_bytes()).hexdigest()
    provenance = dict(dataset=str(args.data.resolve()), dataset_sha256=fingerprint,
                      source=metadata.get("source", "unspecified"),
                      collection_settings=metadata.get("settings", {}),
                      source_sha256=metadata.get("sha256", {}),
                      observation_contract=metadata.get("observation_contract", {}),
                      split_episodes={key: np.unique(data["episode"][rows]).tolist() for key, rows in splits.items()},
                      seed=args.seed, best_epoch=best_epoch, hidden=args.hidden,
                      label_definition="signed raw root forward speed; causal EMA trajectory direction difference",
                      normalization="training episodes only", activation="relu", output_activation="linear")
    model_path = args.out / "estimator.npz"
    save_estimator(model_path, config, best_params, x_mean, x_std, y_mean, y_std, provenance)
    model = RollingVelocityEstimator(model_path)
    # Check the shipped NumPy runtime against the training implementation.
    expected = np.asarray(forward(best_params, jnp.asarray(x[:32]))) * y_std + y_mean
    np.testing.assert_allclose(model.predict_features(data["x"][:32]), expected, rtol=2e-4, atol=2e-5)
    metrics = {}
    for name, rows in splits.items():
        metrics[name], predicted = evaluate(model, data, rows, y_mean)
        if name == "test":
            np.savez_compressed(args.out / "test_predictions.npz", prediction=predicted,
                                target=data["y"][rows], mask=data["mask"][rows],
                                episode=data["episode"][rows], source_row=data["source_row"][rows],
                                output_names=np.asarray(OUTPUT_NAMES),
                                **{k: data[k][rows] for k in ("command", "command_age_s", "command_segment") if k in data})
    target_coverage = {name: {"valid_samples": int(data["mask"][:, i].sum()),
                              "quantiles": np.quantile(data["y"][:, i][data["mask"][:, i]],
                                                       [0, .05, .5, .95, 1]).tolist()}
                       for i, name in enumerate(OUTPUT_NAMES)}
    report = dict(config=asdict(config), provenance=provenance, metrics=metrics,
                  target_coverage=target_coverage, best_validation_loss=best_loss,
                  wall_seconds=time.perf_counter() - started,
                  training_args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  devices=[str(d) for d in jax.devices()],
                  limitations=["same-source episode holdout, not sim-to-real validation",
                               "turn rate undefined at low translation speed; no runtime confidence head",
                               "EMA turn label is causal but delayed; linear speed label is instantaneous"])
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    np.savetxt(args.out / "learning_curve.csv", curve, delimiter=",",
               header="epoch,train_huber,validation_huber", comments="")
    print(json.dumps(metrics["test"]["estimator"], indent=2), flush=True)
    print(f"saved {model_path}; elapsed {report['wall_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()

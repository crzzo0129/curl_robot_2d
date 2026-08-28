#!/usr/bin/env python3
"""Convert a Brax PPO parameter checkpoint to an RTNeural policy JSON.

The deployment policy is a deterministic tanh-normal actor.  Brax stores the
actor's final Gaussian location and scale parameters together; only the first
half (the location) contributes to deterministic inference.  This exporter
therefore keeps the location columns and appends a tanh activation.

Observation normalization is folded into the first dense layer so the C++
controller can feed raw observations directly to RTNeural.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import tempfile
from collections.abc import Mapping
from typing import Any

import numpy as np


SUPPORTED_ACTIVATIONS = {"elu", "relu", "sigmoid", "tanh"}


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _field(value: Any, name: str) -> Any:
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, Mapping) and name in value:
        return value[name]
    raise ValueError(f"normalizer has no {name!r} field")


def _load_checkpoint(path: str) -> Any:
    # Brax currently uses pickle in brax.io.model.  Importing through Brax is
    # preferred, while the fallback keeps the self-test and simple checkpoints
    # usable in a lightweight Python environment.
    try:
        from brax.io import model  # pylint: disable=import-outside-toplevel

        return model.load_params(path)
    except ImportError:
        with open(path, "rb") as checkpoint_file:
            return pickle.load(checkpoint_file)


def _natural_path_key(path: tuple[str, ...]) -> tuple[Any, ...]:
    key: list[Any] = []
    for part in path:
        for token in re.split(r"(\d+)", part):
            key.append(int(token) if token.isdigit() else token)
    return tuple(key)


def _dense_layers(tree: Any) -> list[tuple[tuple[str, ...], np.ndarray, np.ndarray]]:
    found: list[tuple[tuple[str, ...], np.ndarray, np.ndarray]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if not isinstance(node, Mapping):
            return
        if "kernel" in node and "bias" in node:
            kernel = _array(node["kernel"], ".".join(path) + ".kernel")
            bias = _array(node["bias"], ".".join(path) + ".bias")
            if kernel.ndim != 2 or bias.ndim != 1:
                raise ValueError(
                    f"dense layer {'.'.join(path)} must have a 2-D kernel "
                    "and a 1-D bias"
                )
            if kernel.shape[1] != bias.shape[0]:
                raise ValueError(
                    f"dense layer {'.'.join(path)} kernel/bias mismatch: "
                    f"{kernel.shape} vs {bias.shape}"
                )
            found.append((path, kernel, bias))
            return
        for key, child in node.items():
            visit(child, path + (str(key),))

    visit(tree, ())
    found.sort(key=lambda item: _natural_path_key(item[0]))
    if not found:
        raise ValueError("no Flax dense layers were found in policy parameters")
    for previous, current in zip(found, found[1:]):
        if previous[1].shape[1] != current[1].shape[0]:
            raise ValueError(
                "dense layers do not form a chain: "
                f"{'.'.join(previous[0])} {previous[1].shape} -> "
                f"{'.'.join(current[0])} {current[1].shape}"
            )
    return found


def _split_params(checkpoint: Any) -> tuple[Any, Any]:
    if not isinstance(checkpoint, (tuple, list)) or len(checkpoint) < 2:
        raise ValueError(
            "expected a Brax PPO checkpoint tuple: "
            "(normalizer_params, policy_params, value_params)"
        )
    return checkpoint[0], checkpoint[1]


def _read_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config JSON must contain an object at its root")
    return config


def convert(
    checkpoint: Any,
    config: dict[str, Any],
    activation: str = "elu",
    observation_history: int | None = None,
) -> dict[str, Any]:
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"unsupported RTNeural activation {activation!r}; choose one of "
            f"{sorted(SUPPORTED_ACTIVATIONS)}"
        )

    normalizer, policy_params = _split_params(checkpoint)
    dense = _dense_layers(policy_params)
    mean = _array(_field(normalizer, "mean"), "normalizer.mean").reshape(-1)
    std = _array(_field(normalizer, "std"), "normalizer.std").reshape(-1)

    input_size = dense[0][1].shape[0]
    if mean.shape != (input_size,) or std.shape != (input_size,):
        raise ValueError(
            f"normalizer size must be {input_size}, got mean={mean.shape}, "
            f"std={std.shape}"
        )
    if np.any(std <= 0.0):
        raise ValueError("normalizer.std must be strictly positive")

    configured_action_size = len(config.get("action_scale", []))
    final_size = dense[-1][1].shape[1]
    action_size = configured_action_size or final_size // 2
    if action_size <= 0:
        raise ValueError("could not infer a positive action size")
    if final_size == 2 * action_size:
        # NormalTanhDistribution: [location, unconstrained scale].  The
        # deterministic policy is tanh(location), so discard scale columns.
        final_columns = slice(0, action_size)
    elif final_size == action_size:
        final_columns = slice(None)
    else:
        raise ValueError(
            f"policy output has {final_size} values but action size is "
            f"{action_size}; expected {action_size} or {2 * action_size}"
        )

    converted: list[tuple[np.ndarray, np.ndarray, str]] = []
    for index, (_, kernel, bias) in enumerate(dense):
        if index == len(dense) - 1:
            kernel = kernel[:, final_columns]
            bias = bias[final_columns]
            layer_activation = "tanh"
        else:
            layer_activation = activation
        converted.append((kernel.copy(), bias.copy(), layer_activation))

    # Brax evaluates (obs - mean) / std before the actor.  Fold that affine
    # transform into layer 0:
    #   ((x - mean) / std) @ W + b == x @ (W / std) + b - mean @ (W / std)
    first_kernel, first_bias, first_activation = converted[0]
    first_kernel /= std[:, None]
    first_bias -= mean @ first_kernel
    converted[0] = (first_kernel, first_bias, first_activation)

    history_in_config = config.get("observation_history")
    if observation_history is None and history_in_config is not None:
        observation_history = int(history_in_config)
    if observation_history is not None:
        if observation_history <= 0:
            raise ValueError("observation history must be positive")
        if history_in_config is not None and int(history_in_config) != observation_history:
            raise ValueError(
                "--obs-history does not match config observation_history: "
                f"{observation_history} vs {history_in_config}"
            )
        if input_size % observation_history != 0:
            raise ValueError(
                f"input size {input_size} is not divisible by observation "
                f"history {observation_history}"
            )

    layers = []
    for kernel, bias, layer_activation in converted:
        layers.append(
            {
                "type": "dense",
                "shape": [1, int(bias.shape[0])],
                "weights": [kernel.tolist(), bias.tolist()],
                "activation": layer_activation,
            }
        )

    output = dict(config)
    output.update(
        {
            "in_shape": [1, int(input_size)],
            "out_shape": [1, int(action_size)],
            "layers": layers,
        }
    )
    return output


def _activation(name: str, value: np.ndarray) -> np.ndarray:
    if name == "elu":
        return np.where(value > 0.0, value, np.expm1(value))
    if name == "relu":
        return np.maximum(value, 0.0)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-value))
    if name == "tanh":
        return np.tanh(value)
    raise ValueError(name)


def _run_layers(value: np.ndarray, layers: list[dict[str, Any]]) -> np.ndarray:
    for layer in layers:
        kernel = np.asarray(layer["weights"][0], dtype=np.float32)
        bias = np.asarray(layer["weights"][1], dtype=np.float32)
        value = _activation(layer["activation"], value @ kernel + bias)
    return value


def self_test() -> None:
    rng = np.random.default_rng(7)
    mean = rng.normal(size=6).astype(np.float32)
    std = rng.uniform(0.3, 2.0, size=6).astype(np.float32)
    kernels = [
        rng.normal(size=(6, 5)).astype(np.float32),
        rng.normal(size=(5, 3)).astype(np.float32),
        rng.normal(size=(3, 4)).astype(np.float32),
    ]
    biases = [rng.normal(size=k.shape[1]).astype(np.float32) for k in kernels]
    policy = {
        "params": {
            f"hidden_{i}": {"kernel": kernel, "bias": bias}
            for i, (kernel, bias) in enumerate(zip(kernels, biases))
        }
    }
    checkpoint = ({"mean": mean, "std": std}, policy, {})
    document = convert(
        checkpoint,
        {"action_scale": [0.25, 0.25], "observation_history": 2},
        activation="elu",
        observation_history=2,
    )

    observations = rng.normal(size=(32, 6)).astype(np.float32)
    expected = (observations - mean) / std
    expected = _activation("elu", expected @ kernels[0] + biases[0])
    expected = _activation("elu", expected @ kernels[1] + biases[1])
    expected = np.tanh((expected @ kernels[2] + biases[2])[:, :2])
    actual = _run_layers(observations, document["layers"])
    error = float(np.max(np.abs(expected - actual)))
    if not math.isfinite(error) or error > 2e-5:
        raise RuntimeError(f"exporter self-test failed: max error {error}")
    json.loads(json.dumps(document, separators=(",", ":")))
    print(f"exporter self-test passed (max error {error:.3g})")


def _write_json(path: str, document: dict[str, Any]) -> None:
    output_path = os.path.abspath(path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=os.path.basename(output_path) + ".",
        suffix=".tmp",
        dir=output_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            json.dump(document, output_file, separators=(",", ":"))
            output_file.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Brax PPO checkpoint to RTNeural JSON"
    )
    parser.add_argument("checkpoint", nargs="?", help="Brax policy .bin file")
    parser.add_argument("output", nargs="?", help="output RTNeural .json file")
    parser.add_argument("--activation", default="elu", choices=sorted(SUPPORTED_ACTIVATIONS))
    parser.add_argument("--config", help="controller metadata JSON to merge")
    parser.add_argument("--obs-history", type=int, help="expected observation history")
    parser.add_argument("--self-test", action="store_true", help="run an in-memory conversion test")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.checkpoint or not args.output:
        parser.error("checkpoint and output are required unless --self-test is used")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    config = _read_config(args.config)
    checkpoint = _load_checkpoint(args.checkpoint)
    document = convert(
        checkpoint,
        config,
        activation=args.activation,
        observation_history=args.obs_history,
    )
    _write_json(args.output, document)
    print(
        f"wrote {args.output}: in={document['in_shape'][-1]} "
        f"out={document['out_shape'][-1]} layers={len(document['layers'])}"
    )


if __name__ == "__main__":
    main()

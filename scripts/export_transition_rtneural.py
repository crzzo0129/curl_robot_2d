"""Export the asymmetric 720-input Transition actor to neural_controller JSON.

Only the state normalizer and actor are exported; privileged_state and the
critic never enter the hardware model. Reuses the Walking RTNeural exporter.
"""

import argparse
import math
from collections.abc import Mapping

from scripts.export_rtneural import (
    convert, _field, _split_params, _dense_layers,
    _load_checkpoint, _read_config, _write_json,
)


def convert_transition(checkpoint, config, *, runtime="static"):
    config = dict(config)
    if config.get("requires_target_rate_limiter") and runtime != "pitch_hot_switch":
        raise ValueError("This policy needs the updated neural_controller hot-switch runtime. "
                         "Export with --runtime pitch_hot_switch and load via "
                         "roll_to_stand_model_path; do not disable the limiter metadata.")
    if runtime == "pitch_hot_switch":
        if (config.get("action_semantics") != "absolute_about_default" or
                not config.get("requires_target_rate_limiter") or
                config.get("target_rate_limit_initialization") != "last_executed_roll_ctrl" or
                config.get("target_rate_limit_order") !=
                "joint_limits_then_rate_limit_once_per_policy_tick"):
            raise ValueError("pitch_hot_switch requires absolute targets and the rolling-target limiter contract")
        rates = config.get("target_rate_limits_rad_s", ())
        dt = float(config.get("target_rate_limit_timestep_s", 0))
        hz = float(config.get("policy_frequency_hz", 0))
        if (len(rates) != 12 or any(not math.isfinite(v) or v <= 0 for v in rates) or
                not math.isfinite(dt) or dt <= 0 or not math.isfinite(hz) or hz <= 0 or
                not math.isclose(dt * hz, 1.0, abs_tol=1e-6)):
            raise ValueError("Invalid target rate limits or policy timing")
        config["required_controller_runtime"] = "neural_controller_pitch_hot_switch_v1"
    if config.get("contract_version") != "transition_neural_controller_36x20_v3":
        raise ValueError("expected the current Transition deployment_config.json")
    if config.get("observation_history") != 20 or config.get("single_observation_size") != 36:
        raise ValueError("Transition requires exactly 36 x 20 observations")
    if config.get("activation") != "elu" or config.get("actor_output") != "tanh_location":
        raise ValueError("export supports the ELU/tanh-normal Transition actor only")
    if len(config.get("action_scale", ())) != 12:
        raise ValueError("Transition requires 12 action scales")
    normalizer, actor = _split_params(checkpoint)
    dense = _dense_layers(actor)
    if dense[0][1].shape[0] != 720 or dense[-1][1].shape[1] != 24:
        raise ValueError("expected a 720-input, 24-logit tanh-normal actor; "
                         "old 66-input or unsquashed normal actors are incompatible")
    mean = _field(normalizer, "mean")
    std = _field(normalizer, "std")
    if not isinstance(mean, Mapping) or not isinstance(std, Mapping):
        raise ValueError("expected asymmetric normalizer with state/privileged_state keys")
    actor_checkpoint = ({"mean": mean["state"], "std": std["state"]}, actor, {})
    return convert(actor_checkpoint, config, activation="elu", observation_history=20)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--config", required=True,
                        help="deployment_config.json written by Transition training")
    parser.add_argument("--runtime", choices=("static", "pitch_hot_switch"), default="static",
                        help="pitch_hot_switch requires the updated C++ runtime with target limiting")
    args = parser.parse_args(argv)
    document = convert_transition(_load_checkpoint(args.checkpoint), _read_config(args.config),
                                  runtime=args.runtime)
    _write_json(args.output, document)
    print(f"Exported Actor only: {document['in_shape']} -> {document['out_shape']}")


if __name__ == "__main__":
    main()

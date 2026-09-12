"""CPU reproduction of historical steering authority and current prior amplitude.

Uses the existing CPU reference evaluator. These are compact-start, 10-second
authority probes, not MJX snapshot-takeover policy success evaluations. The CPU
evaluator applies a constant steering offset including startup, while the MJX
teacher ramps the prior during startup. No PPO/student is loaded or modified.
"""
from pathlib import Path
import json
import time

import mujoco
import numpy as np

from scripts import evaluate_3d_symmetric_cem_reference as evaluator
from curl_robot_2d_mjx.environment_3d import (
    model_path_3d, forward_command_to_target_scale_3d,
)


def main():
    out = Path("results/cem_steering_comparison_20260912")
    out.mkdir(parents=True, exist_ok=False)
    old = Path("results/pupper_r127p5_open60_shell150_45_three_stage_cem/03_strict_forbidden_collision/best_phase_controller.json")
    new = Path("results/rollingquad_abd10_high_speed_zero_contact_refine_smoke/01_zero_contact_speed_refine/best_phase_controller.json")
    old_xml = Path("assets/rollingquad_description_2/mjcf/rollingquad_abd10.xml")
    new_xml = model_path_3d("rollingquad_2_abd10_no_self_collision")
    cases = [
        ("historical_zero", old, old_xml, 1., 0., .30),
        ("historical_positive", old, old_xml, 1., .5, .30),
        ("historical_negative", old, old_xml, 1., -.5, .30),
    ]
    for speed in (.60, .80):
        scale = float(forward_command_to_target_scale_3d(np, speed))
        for name, raw, gain in (("zero", 0., .15), ("cmd_p08", .4, .15),
                                ("cmd_n08", -.4, .15), ("old_authority", .5, .30)):
            cases.append((f"current_v{speed:.2f}_{name}", new, new_xml, scale, raw, gain))
    summary = []
    for name, controller, xml, scale, raw, gain in cases:
        start = time.perf_counter()
        print(f"Starting {name}", flush=True)
        args = evaluator.parse_args([
            "--xml", str(xml), "--controller", str(controller),
            "--geometry", "rollingquad_2", "--physics-profile", "cg20",
            "--duration", "10", "--control-dt", ".02",
            "--front-abduction-deg", "-10", "--rear-abduction-deg", "10",
            "--target-scale", str(scale), "--residual-gain", str(gain),
            "--differential-scale", ".25", "--differential-residual",
            str(raw), str(raw), str(raw), str(-raw),
        ])
        result = evaluator.run_smoke(args)
        result["mujoco_version"] = mujoco.__version__
        result["probe_walltime_s"] = time.perf_counter() - start
        (out / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        row = {"case": name, "action_offset": raw * gain * .25,
               "heading_rate_rad_s": result["rolling_axis_heading_rate_rad_s"],
               "world_vx_m_s": result["distance_x_m"] / result["elapsed_s"],
               "self_contact_fraction": result["self_contact_fraction"],
               "nonfinite": result["nonfinite"],
               "axis_elevation_rms_rad": result["rolling_axis_elevation_rms_rad"]}
        summary.append(row)
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

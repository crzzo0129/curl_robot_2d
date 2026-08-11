"""Render the old reference on the original and enlarged 2-D shells."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from curl_robot_2d.parameters import FIXED_PARAMETERS
from scripts import optimize_phase_controller as phase_controller
from scripts.evaluate_fixed_reference_shell_radius import (
    DEFAULT_CONTROLLER,
    DEFAULT_OUTPUT,
    _controller_settings,
)
from scripts.replay_active_controller import (
    advance_controller,
    configure_tracking_camera,
    initialize_simulation,
    load_controller_options,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shell-radius-mm", type=float, default=160.0)
    parser.add_argument(
        "--mode",
        choices=("radius", "cem"),
        default="radius",
        help="Compare shell radii, or compare warm-start and CEM on the enlarged shell.",
    )
    parser.add_argument(
        "--optimized-controller",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "shell_radius_160mm_warm_start_cem"
            / "best_phase_controller.json"
        ),
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--camera-distance", type=float, default=0.72)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _make_scene(model, parameters, minimum_gap_m, width, height, distance, diagnostics):
    data, pitch_address, _ = initialize_simulation(
        model, minimum_gap_m, parameters
    )
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    configure_tracking_camera(model, camera, distance=distance)
    option = mujoco.MjvOption()
    if diagnostics:
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    return data, pitch_address, renderer, camera, option


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.duration <= 0 or args.fps <= 0:
        raise SystemExit("--duration and --fps must be positive")

    controller_path = args.controller.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_path = (
        args.output
        or DEFAULT_OUTPUT
        / (
            "shell_160mm_cem_before_after.gif"
            if args.mode == "cem"
            else "old_reference_radius_comparison.gif"
        )
    ).expanduser().resolve()
    enlarged = replace(
        FIXED_PARAMETERS,
        shell_contact_radius_override=args.shell_radius_mm / 1000.0,
        shell_arc_coverage_angle_override=FIXED_PARAMETERS.shell_arc_coverage_angle,
    )
    enlarged_model = input_dir / f"shell_{args.shell_radius_mm:g}mm" / "model.xml"
    if args.mode == "cem":
        variants = (
            ("160 mm: old warm-start", enlarged, enlarged_model, controller_path),
            (
                "160 mm: CEM optimized",
                enlarged,
                enlarged_model,
                args.optimized_controller.expanduser().resolve(),
            ),
        )
    else:
        variants = (
            (
                "Original shell: 147.55 mm",
                FIXED_PARAMETERS,
                input_dir / "original_no_shell_shell" / "model.xml",
                controller_path,
            ),
            (
                f"Enlarged shell: {args.shell_radius_mm:g} mm",
                enlarged,
                enlarged_model,
                controller_path,
            ),
        )
    for _, _, model_path, variant_controller in variants:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not variant_controller.exists():
            raise FileNotFoundError(variant_controller)

    states = []
    for label, parameters, model_path, variant_controller in variants:
        (
            coefficients,
            oscillator_rate,
            oscillator_coupling,
            minimum_gap_m,
            tracking_margin_m,
            knee_bias_rad,
        ) = load_controller_options(variant_controller)
        evaluated_gap_m, evaluated_margin_m, _ = _controller_settings(
            variant_controller
        )
        minimum_gap_m = evaluated_gap_m
        tracking_margin_m = evaluated_margin_m
        model = mujoco.MjModel.from_xml_path(str(model_path))
        scene = _make_scene(
            model,
            parameters,
            minimum_gap_m,
            args.panel_width,
            args.panel_height,
            args.camera_distance,
            args.diagnostics,
        )
        states.append(
            {
                "label": label,
                "parameters": parameters,
                "controller": str(variant_controller),
                "coefficients": coefficients,
                "oscillator_rate": oscillator_rate,
                "oscillator_coupling": oscillator_coupling,
                "minimum_gap_m": minimum_gap_m,
                "tracking_margin_m": tracking_margin_m,
                "knee_bias_rad": knee_bias_rad,
                "model": model,
                "data": scene[0],
                "pitch_address": scene[1],
                "renderer": scene[2],
                "camera": scene[3],
                "option": scene[4],
                "oscillator_phase": 0.0,
                "start_pitch": float(scene[0].qpos[scene[1]]),
                "start_x": float(scene[0].qpos[0]),
            }
        )

    frames: list[Image.Image] = []
    next_frame_time = 0.0
    frame_period = 1.0 / args.fps
    try:
        while min(float(state["data"].time) for state in states) < args.duration:
            for state in states:
                phase_controller._activate_geometry(state["parameters"])
                state["oscillator_phase"] = advance_controller(
                    state["model"],
                    state["data"],
                    state["coefficients"],
                    state["oscillator_rate"],
                    state["oscillator_coupling"],
                    state["oscillator_phase"],
                    state["pitch_address"],
                    state["minimum_gap_m"],
                    state["tracking_margin_m"],
                    state["knee_bias_rad"],
                )
            if min(float(state["data"].time) for state in states) + 1e-12 < next_frame_time:
                continue

            panels = []
            for state in states:
                state["renderer"].update_scene(
                    state["data"],
                    camera=state["camera"],
                    scene_option=state["option"],
                )
                panel = Image.fromarray(state["renderer"].render())
                draw = ImageDraw.Draw(panel)
                elapsed = float(state["data"].time)
                turns = (
                    float(state["data"].qpos[state["pitch_address"]])
                    - state["start_pitch"]
                ) / (2.0 * math.pi)
                distance = float(state["data"].qpos[0]) - state["start_x"]
                draw.rectangle((8, 8, args.panel_width - 8, 58), fill=(18, 24, 32))
                draw.text((18, 15), state["label"], fill=(245, 247, 250))
                draw.text(
                    (18, 36),
                    f"t={elapsed:4.1f}s   roll={turns:5.2f} turns   x={distance:5.2f}m",
                    fill=(190, 205, 220),
                )
                panels.append(panel)
            combined = Image.new(
                "RGB", (2 * args.panel_width, args.panel_height), (12, 16, 22)
            )
            combined.paste(panels[0], (0, 0))
            combined.paste(panels[1], (args.panel_width, 0))
            frames.append(
                combined.quantize(colors=128, method=Image.Quantize.FASTOCTREE)
            )
            next_frame_time += frame_period
    finally:
        for state in states:
            state["renderer"].close()

    if not frames:
        raise RuntimeError("No frames rendered")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000.0 / args.fps),
        loop=0,
        optimize=False,
        disposal=2,
    )
    summary = {
        "output": str(output_path),
        "duration_s": args.duration,
        "fps": args.fps,
        "frames": len(frames),
        "variants": [
            {
                "label": state["label"],
                "controller": state["controller"],
                "turns": (
                    float(state["data"].qpos[state["pitch_address"]])
                    - state["start_pitch"]
                ) / (2.0 * math.pi),
                "distance_x_m": float(state["data"].qpos[0]) - state["start_x"],
            }
            for state in states
        ],
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

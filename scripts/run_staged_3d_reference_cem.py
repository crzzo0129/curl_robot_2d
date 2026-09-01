"""Warm-start a three-stage CEM reference on the rollingquad_2 3-D model.

The historical controller was optimized in the planar model.  This search
keeps its symmetric phase-oscillator structure, but evaluates every candidate
against the real 3-D CAD geometry with the model's selective self-collision
contract enabled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from curl_robot_2d.model_3d import JOINT_NAMES_3D
from curl_robot_2d.parameters import PUPPER_ORIGINAL_SHELL_60_PARAMETERS
from curl_robot_2d_mjx.cem_reference import (
    COEFFICIENT_NAMES,
    CEMReferenceConfig,
    advance_oscillator,
    load_cem_reference,
    wrapped_phase_error,
)
from curl_robot_2d_mjx.config_3d import Rolling3DConfig, physics_profile_3d
from curl_robot_2d_mjx.environment_3d import (
    PUPPER_OPEN60_CEM_CONTROLLER,
    ROLLINGQUAD_2_MODEL_PATH_3D,
    apply_physics_options_3d,
    validate_rollingquad_self_collision_contract_3d,
)
from scripts.evaluate_3d_symmetric_cem_reference import (
    FOOT_GEOM_NAMES_3D,
    _rolling_axis_tilt,
    activate_planar_geometry,
    map_planar_to_curl_3d_targets,
    override_reference_foot_gap,
    planar_cem_target,
    scaled_planar_target,
    startup_target_scale,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "rollingquad2_three_stage_3d_self_collision_cem"
)

COEFFICIENT_BOUNDS = np.asarray(
    (
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.0, 1.0),
        (-1.0, 1.0),
        (-1.4, 1.4),
        (-1.4, 1.4),
    ),
    dtype=np.float64,
)
RATE_BOUNDS = (0.5, 6.0)
COUPLING_BOUNDS = (0.0, 8.0)
FOOT_GAP_BOUNDS_M = (0.0, 0.006)
PARAMETER_NAMES = (*COEFFICIENT_NAMES, "oscillator_rate_rad_s", "oscillator_coupling_per_s", "minimum_foot_surface_gap_m")


@dataclass(frozen=True)
class StageConfig:
    name: str
    description: str
    generations: int
    population: int
    elite_count: int
    duration_s: float
    minimum_turns: float
    progress_reward_margin_turns: float
    progress_weight: float
    contact_time_weight: float
    penetration_integral_weight: float
    maximum_penetration_weight: float
    lateral_weight: float
    tilt_rms_weight: float
    tilt_max_weight: float
    phase_error_weight: float
    saturation_weight: float
    seed: int


FULL_STAGES = (
    StageConfig(
        name="01_recover_roll",
        description="Recover sustained rolling from the planar warm start",
        generations=8,
        population=32,
        elite_count=6,
        duration_s=3.0,
        minimum_turns=1.2,
        progress_reward_margin_turns=0.8,
        progress_weight=24.0,
        contact_time_weight=0.5,
        penetration_integral_weight=250.0,
        maximum_penetration_weight=30.0,
        lateral_weight=1.0,
        tilt_rms_weight=1.0,
        tilt_max_weight=0.5,
        phase_error_weight=0.5,
        saturation_weight=4.0,
        seed=17,
    ),
    StageConfig(
        name="02_reduce_contact",
        description="Reduce measured 3-D self-contact while retaining progress",
        generations=10,
        population=40,
        elite_count=8,
        duration_s=6.0,
        minimum_turns=2.8,
        progress_reward_margin_turns=0.4,
        progress_weight=18.0,
        contact_time_weight=2.5,
        penetration_integral_weight=1400.0,
        maximum_penetration_weight=180.0,
        lateral_weight=3.0,
        tilt_rms_weight=2.0,
        tilt_max_weight=1.0,
        phase_error_weight=1.0,
        saturation_weight=8.0,
        seed=29,
    ),
    StageConfig(
        name="03_strict_10s",
        description="Strict 10 s selection with progress gating and collision cost",
        generations=12,
        population=48,
        elite_count=8,
        duration_s=10.0,
        minimum_turns=5.0,
        progress_reward_margin_turns=0.5,
        progress_weight=14.0,
        contact_time_weight=4.0,
        penetration_integral_weight=2400.0,
        maximum_penetration_weight=300.0,
        lateral_weight=5.0,
        tilt_rms_weight=3.0,
        tilt_max_weight=2.0,
        phase_error_weight=1.5,
        saturation_weight=12.0,
        seed=43,
    ),
)


SMOKE_STAGES = tuple(
    replace(
        stage,
        generations=2,
        population=8,
        elite_count=2,
        duration_s=(1.5, 2.5, 4.0)[index],
        minimum_turns=(0.35, 0.8, 1.6)[index],
    )
    for index, stage in enumerate(FULL_STAGES)
)


@dataclass(frozen=True)
class RolloutResult:
    score: float
    summary: dict[str, object]


class ReferenceRollout3D:
    def __init__(
        self,
        xml_path: Path,
        *,
        physics_profile: str,
        control_dt: float,
        kp: float,
        kd: float,
        torque_limit: float,
        tracking_margin_m: float,
    ) -> None:
        activate_planar_geometry(PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
        validate_rollingquad_self_collision_contract_3d(self.model)
        task = physics_profile_3d(physics_profile, Rolling3DConfig())
        apply_physics_options_3d(self.model, task)
        self.data = mujoco.MjData(self.model)
        self.control_repeat = max(1, round(control_dt / self.model.opt.timestep))
        self.control_dt = self.control_repeat * float(self.model.opt.timestep)
        self.tracking_margin_m = float(tracking_margin_m)
        self.torque_limit = float(torque_limit)

        joint_ids = np.asarray(
            [self.model.joint(name).id for name in JOINT_NAMES_3D], dtype=np.int32
        )
        self.qpos_indices = np.asarray(
            [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids],
            dtype=np.int32,
        )
        self.actuator_ids = np.asarray(
            [self.model.actuator(f"{name}_servo").id for name in JOINT_NAMES_3D],
            dtype=np.int32,
        )
        self.ctrl_low = np.asarray(
            self.model.actuator_ctrlrange[self.actuator_ids, 0], dtype=np.float64
        )
        self.ctrl_high = np.asarray(
            self.model.actuator_ctrlrange[self.actuator_ids, 1], dtype=np.float64
        )
        self.model.actuator_gainprm[self.actuator_ids, 0] = kp
        self.model.actuator_biasprm[self.actuator_ids, 1] = -kp
        self.model.actuator_biasprm[self.actuator_ids, 2] = -kd
        self.model.actuator_forcerange[self.actuator_ids, 0] = -torque_limit
        self.model.actuator_forcerange[self.actuator_ids, 1] = torque_limit

        self.compact_key_id = self.model.key("compact").id
        self.torso_body_id = self.model.body("torso").id
        self.floor_geom_id = self.model.geom("floor").id
        self.foot_geom_ids = {
            self.model.geom(name).id for name in FOOT_GEOM_NAMES_3D
        }

    def _config(self, parameters: np.ndarray) -> CEMReferenceConfig:
        gap_m = float(parameters[10])
        base = CEMReferenceConfig(
            coefficients=tuple(float(value) for value in parameters[:8]),
            oscillator_rate_rad_s=float(parameters[8]),
            oscillator_coupling_per_s=float(parameters[9]),
            foot_gap_tracking_margin_m=self.tracking_margin_m,
        )
        if gap_m <= 0.0:
            return base
        return override_reference_foot_gap(
            base, 1000.0 * gap_m, None
        )

    def run(self, parameters: np.ndarray, stage: StageConfig) -> RolloutResult:
        config = self._config(parameters)
        initial_planar = planar_cem_target(0.0, config)
        initial_planar = scaled_planar_target(
            initial_planar,
            startup_target_scale(
                0.0,
                target_scale=1.0,
                startup_scale=0.0,
                ramp_duration_s=0.25,
                startup_boost=0.0,
                startup_boost_duration_s=0.25,
            ),
        )
        initial_ctrl = np.clip(
            map_planar_to_curl_3d_targets(initial_planar),
            self.ctrl_low,
            self.ctrl_high,
        )
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.compact_key_id)
        self.data.qpos[self.qpos_indices] = initial_ctrl
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.actuator_ids] = initial_ctrl
        mujoco.mj_forward(self.model, self.data)

        start_x = float(self.data.qpos[0])
        start_y = float(self.data.qpos[1])
        phase = 0.0
        rolling_phase = 0.0
        absolute_rotation = 0.0
        self_contact_steps = 0
        penetration_integral = 0.0
        maximum_penetration = 0.0
        saturation_sum = 0.0
        tilt_squared_sum = 0.0
        maximum_tilt = 0.0
        phase_error_squared_sum = 0.0
        pair_steps: dict[str, int] = {}
        pair_maximum_penetration: dict[str, float] = {}
        nonfinite = False
        physics_steps = 0
        control_steps = max(1, round(stage.duration_s / self.control_dt))

        for _ in range(control_steps):
            for _ in range(self.control_repeat):
                phase = float(
                    advance_oscillator(
                        np,
                        rolling_phase,
                        phase,
                        float(self.model.opt.timestep),
                        config,
                    )
                )
                planar = planar_cem_target(phase, config)
                planar = scaled_planar_target(
                    planar,
                    startup_target_scale(
                        float(self.data.time),
                        target_scale=1.0,
                        startup_scale=0.0,
                        ramp_duration_s=0.25,
                        startup_boost=0.0,
                        startup_boost_duration_s=0.25,
                    ),
                )
                ctrl = np.clip(
                    map_planar_to_curl_3d_targets(planar),
                    self.ctrl_low,
                    self.ctrl_high,
                )
                self.data.ctrl[self.actuator_ids] = ctrl
                mujoco.mj_step(self.model, self.data)
                dt = float(self.model.opt.timestep)
                increment = float(self.data.qvel[4]) * dt
                rolling_phase += increment
                absolute_rotation += abs(increment)
                physics_steps += 1

                if not (
                    np.isfinite(self.data.qpos).all()
                    and np.isfinite(self.data.qvel).all()
                ):
                    nonfinite = True
                    break

                saturation_sum += float(
                    np.mean(
                        np.abs(self.data.actuator_force[self.actuator_ids])
                        >= 0.99 * self.torque_limit
                    )
                )
                active_pairs: dict[str, float] = {}
                for contact in self.data.contact:
                    geom1, geom2 = int(contact.geom1), int(contact.geom2)
                    if self.floor_geom_id in (geom1, geom2):
                        continue
                    names = sorted(
                        (
                            mujoco.mj_id2name(
                                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1
                            )
                            or f"geom_{geom1}",
                            mujoco.mj_id2name(
                                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2
                            )
                            or f"geom_{geom2}",
                        )
                    )
                    pair = "__".join(names)
                    active_pairs[pair] = max(
                        active_pairs.get(pair, 0.0), max(-float(contact.dist), 0.0)
                    )
                if active_pairs:
                    self_contact_steps += 1
                    step_penetration = max(active_pairs.values())
                    penetration_integral += step_penetration * dt
                    maximum_penetration = max(maximum_penetration, step_penetration)
                for pair, penetration in active_pairs.items():
                    pair_steps[pair] = pair_steps.get(pair, 0) + 1
                    pair_maximum_penetration[pair] = max(
                        pair_maximum_penetration.get(pair, 0.0), penetration
                    )

                rotation = self.data.xmat[self.torso_body_id].reshape(3, 3)
                tilt = _rolling_axis_tilt(rotation)
                maximum_tilt = max(maximum_tilt, tilt)
                tilt_squared_sum += tilt * tilt
                phase_error = float(wrapped_phase_error(np, rolling_phase, phase))
                phase_error_squared_sum += phase_error * phase_error
            if nonfinite:
                break

        elapsed = max(physics_steps * float(self.model.opt.timestep), 1.0e-9)
        rolling_turns = rolling_phase / (2.0 * math.pi)
        distance_x = float(self.data.qpos[0]) - start_x
        distance_y = float(self.data.qpos[1]) - start_y
        distance_turns = distance_x / (
            2.0
            * math.pi
            * PUPPER_ORIGINAL_SHELL_60_PARAMETERS.shell_contact_radius
        )
        conservative_turns = max(min(rolling_turns, distance_turns), 0.0)
        contact_time = self_contact_steps * float(self.model.opt.timestep)
        divisor = max(physics_steps, 1)
        tilt_rms = math.sqrt(tilt_squared_sum / divisor)
        phase_error_rms = math.sqrt(phase_error_squared_sum / divisor)
        saturation_fraction = saturation_sum / divisor
        progress_deficit = max(stage.minimum_turns - conservative_turns, 0.0)
        failure_penalty = 120.0 * progress_deficit * progress_deficit
        rewarded_turns = min(
            conservative_turns,
            stage.minimum_turns + stage.progress_reward_margin_turns,
        )
        if nonfinite:
            failure_penalty += 10000.0
        score = (
            stage.progress_weight * rewarded_turns
            - failure_penalty
            - stage.contact_time_weight * contact_time
            - stage.penetration_integral_weight * penetration_integral
            - stage.maximum_penetration_weight * maximum_penetration
            - stage.lateral_weight * abs(distance_y)
            - stage.tilt_rms_weight * tilt_rms
            - stage.tilt_max_weight * maximum_tilt
            - stage.phase_error_weight * phase_error_rms
            - stage.saturation_weight * saturation_fraction
        )
        summary: dict[str, object] = {
            "score": float(score),
            "duration_s": float(elapsed),
            "rolling_turns": float(rolling_turns),
            "distance_turns": float(distance_turns),
            "conservative_rolling_turns": float(conservative_turns),
            "absolute_rotation_turns": float(absolute_rotation / (2.0 * math.pi)),
            "reference_turns": float(phase / (2.0 * math.pi)),
            "distance_x_m": float(distance_x),
            "distance_y_m": float(distance_y),
            "minimum_required_turns": float(stage.minimum_turns),
            "progress_reward_cap_turns": float(
                stage.minimum_turns + stage.progress_reward_margin_turns
            ),
            "rewarded_turns": float(rewarded_turns),
            "progress_deficit_turns": float(progress_deficit),
            "self_contact_fraction": float(self_contact_steps / divisor),
            "self_contact_total_s": float(contact_time),
            "self_penetration_integral_m_s": float(penetration_integral),
            "maximum_self_penetration_m": float(maximum_penetration),
            "rolling_axis_tilt_rms_rad": float(tilt_rms),
            "rolling_axis_tilt_max_rad": float(maximum_tilt),
            "phase_error_rms_rad": float(phase_error_rms),
            "torque_saturation_fraction": float(saturation_fraction),
            "nonfinite": bool(nonfinite),
            "self_contact_pairs": {
                pair: {
                    "duration_s": count * float(self.model.opt.timestep),
                    "maximum_penetration_m": pair_maximum_penetration[pair],
                }
                for pair, count in sorted(pair_steps.items())
            },
        }
        return RolloutResult(float(score), summary)


_WORKER: ReferenceRollout3D | None = None


def _initialize_worker(
    xml_path: str,
    physics_profile: str,
    control_dt: float,
    kp: float,
    kd: float,
    torque_limit: float,
    tracking_margin_m: float,
) -> None:
    global _WORKER
    _WORKER = ReferenceRollout3D(
        Path(xml_path),
        physics_profile=physics_profile,
        control_dt=control_dt,
        kp=kp,
        kd=kd,
        torque_limit=torque_limit,
        tracking_margin_m=tracking_margin_m,
    )


def _run_worker(task: tuple[np.ndarray, StageConfig]) -> RolloutResult:
    if _WORKER is None:
        raise RuntimeError("3-D CEM worker was not initialized")
    parameters, stage = task
    return _WORKER.run(parameters, stage)


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate(
        (
            COEFFICIENT_BOUNDS[:, 0],
            np.asarray((RATE_BOUNDS[0], COUPLING_BOUNDS[0], FOOT_GAP_BOUNDS_M[0])),
        )
    )
    upper = np.concatenate(
        (
            COEFFICIENT_BOUNDS[:, 1],
            np.asarray((RATE_BOUNDS[1], COUPLING_BOUNDS[1], FOOT_GAP_BOUNDS_M[1])),
        )
    )
    return lower, upper


def controller_parameters(path: Path, *, initial_gap_m: float) -> np.ndarray:
    config = load_cem_reference(path)
    return np.asarray(
        (
            *config.coefficients,
            config.oscillator_rate_rad_s,
            config.oscillator_coupling_per_s,
            initial_gap_m,
        ),
        dtype=np.float64,
    )


def _initial_std(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    std = (upper - lower) / 12.0
    std[8] = 0.30
    std[9] = 0.45
    std[10] = 0.0012
    return std


def _update_distribution(
    samples: np.ndarray,
    scores: np.ndarray,
    *,
    elite_count: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    elites = samples[np.argsort(scores)[-elite_count:]]
    next_mean = 0.7 * elites.mean(axis=0) + 0.3 * mean
    next_std = 0.7 * elites.std(axis=0) + 0.3 * std
    minimum_std = np.asarray((0.01,) * 8 + (0.03, 0.05, 0.00015))
    return next_mean, np.maximum(next_std, minimum_std)


def optimize_stage(
    stage: StageConfig,
    initial_parameters: np.ndarray,
    *,
    runner: ReferenceRollout3D | None,
    executor: ProcessPoolExecutor | None,
) -> tuple[np.ndarray, RolloutResult, list[dict[str, object]]]:
    lower, upper = parameter_bounds()
    rng = np.random.default_rng(stage.seed)
    mean = np.clip(initial_parameters, lower, upper)
    std = _initial_std(lower, upper)
    best_parameters = mean.copy()
    if runner is not None:
        best = runner.run(best_parameters, stage)
    else:
        assert executor is not None
        best = next(iter(executor.map(_run_worker, [(best_parameters, stage)])))
    history: list[dict[str, object]] = []

    for generation in range(stage.generations):
        samples = rng.normal(mean, std, size=(stage.population, len(mean)))
        samples = np.clip(samples, lower, upper)
        samples[0] = best_parameters
        tasks = [(sample, stage) for sample in samples]
        if runner is not None:
            rollouts = [runner.run(sample, stage) for sample in samples]
        else:
            assert executor is not None
            rollouts = list(
                executor.map(
                    _run_worker,
                    tasks,
                    chunksize=max(1, stage.population // 16),
                )
            )
        scores = np.asarray([rollout.score for rollout in rollouts])
        generation_best_index = int(np.argmax(scores))
        generation_best = rollouts[generation_best_index]
        if generation_best.score > best.score:
            best = generation_best
            best_parameters = samples[generation_best_index].copy()
        mean, std = _update_distribution(
            samples,
            scores,
            elite_count=stage.elite_count,
            mean=mean,
            std=std,
        )
        mean = np.clip(mean, lower, upper)
        record = {
            "generation": generation + 1,
            "generation_best_score": generation_best.score,
            "global_best_score": best.score,
            "population_mean_score": float(np.mean(scores)),
            "population_std_score": float(np.std(scores)),
            "global_best_turns": best.summary["conservative_rolling_turns"],
            "global_best_contact_s": best.summary["self_contact_total_s"],
            "global_best_maximum_penetration_mm": 1000.0
            * float(best.summary["maximum_self_penetration_m"]),
            "global_best_lateral_m": best.summary["distance_y_m"],
            "global_best_gap_mm": 1000.0 * float(best_parameters[10]),
        }
        history.append(record)
        print(
            f"stage={stage.name} generation={generation + 1:02d}/{stage.generations} "
            f"score={best.score:+.3f} "
            f"turns={float(best.summary['conservative_rolling_turns']):.3f} "
            f"contact={float(best.summary['self_contact_total_s']):.3f}s "
            f"penetration={1000.0*float(best.summary['maximum_self_penetration_m']):.3f}mm "
            f"lateral={float(best.summary['distance_y_m']):+.3f}m "
            f"gap={1000.0*float(best_parameters[10]):.2f}mm",
            flush=True,
        )
    return best_parameters, best, history


def controller_payload(
    parameters: np.ndarray,
    rollout: RolloutResult,
    *,
    stage: StageConfig,
    source_controller: Path,
    tracking_margin_m: float,
) -> dict[str, object]:
    raw = {
        name: float(value)
        for name, value in zip(COEFFICIENT_NAMES, parameters[:8], strict=True)
    }
    base_config = CEMReferenceConfig(
        coefficients=tuple(parameters[:8]),
        oscillator_rate_rad_s=float(parameters[8]),
        oscillator_coupling_per_s=float(parameters[9]),
        foot_gap_tracking_margin_m=tracking_margin_m,
    )
    knee_bias_rad = (
        0.0
        if float(parameters[10]) <= 0.0
        else override_reference_foot_gap(
            base_config,
            1000.0 * float(parameters[10]),
            None,
        ).knee_bias_rad
    )
    return {
        "controller": "phase_locked_oscillator",
        "oscillator_rate_rad_s": float(parameters[8]),
        "oscillator_period_s": float(2.0 * math.pi / parameters[8]),
        "oscillator_coupling_per_s": float(parameters[9]),
        "minimum_foot_surface_gap_m": float(parameters[10]),
        "nominal_knee_bias_rad": float(knee_bias_rad),
        "foot_gap_tracking_margin_m": float(tracking_margin_m),
        "raw_coefficients": raw,
        "optimization": {
            "method": "three_stage_3d_cem",
            "stage": stage.name,
            "source_controller": str(source_controller.resolve()),
            "parameters": list(PARAMETER_NAMES),
            "stage_config": asdict(stage),
        },
        "collision_objective": {
            "physics": "rollingquad_2 selective self-collision",
            "all_enabled_self_contact_penalized": True,
            "contact_time_weight": stage.contact_time_weight,
            "penetration_integral_weight": stage.penetration_integral_weight,
            "maximum_penetration_weight": stage.maximum_penetration_weight,
        },
        "rollout_summary": rollout.summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=ROLLINGQUAD_2_MODEL_PATH_3D)
    parser.add_argument(
        "--initial-controller", type=Path, default=PUPPER_OPEN60_CEM_CONTROLLER
    )
    parser.add_argument("--initial-gap-mm", type=float, default=2.0)
    parser.add_argument("--tracking-margin-mm", type=float, default=4.0)
    parser.add_argument("--physics-profile", default="cg20")
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if not 0.0 <= args.initial_gap_mm <= 1000.0 * FOOT_GAP_BOUNDS_M[1]:
        raise SystemExit("--initial-gap-mm must be in [0, 6]")
    if args.tracking_margin_mm < 0.0:
        raise SystemExit("--tracking-margin-mm must be nonnegative")
    if args.control_dt <= 0.0:
        raise SystemExit("--control-dt must be positive")
    if args.kp < 0.0 or args.kd < 0.0 or args.torque_limit <= 0.0:
        raise SystemExit("invalid actuator parameters")
    if not args.xml.exists():
        raise SystemExit(f"3-D XML not found: {args.xml}")
    if not args.initial_controller.exists():
        raise SystemExit(f"initial controller not found: {args.initial_controller}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    stages = SMOKE_STAGES if args.preset == "smoke" else FULL_STAGES
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = controller_parameters(
        args.initial_controller.expanduser().resolve(),
        initial_gap_m=args.initial_gap_mm / 1000.0,
    )
    runner = None
    executor = None
    worker_args = (
        str(args.xml.expanduser().resolve()),
        args.physics_profile,
        args.control_dt,
        args.kp,
        args.kd,
        args.torque_limit,
        args.tracking_margin_mm / 1000.0,
    )
    if args.workers == 1:
        runner = ReferenceRollout3D(
            args.xml,
            physics_profile=args.physics_profile,
            control_dt=args.control_dt,
            kp=args.kp,
            kd=args.kd,
            torque_limit=args.torque_limit,
            tracking_margin_m=args.tracking_margin_mm / 1000.0,
        )
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialize_worker,
            initargs=worker_args,
        )

    stage_summaries: list[dict[str, object]] = []
    try:
        for stage in stages:
            stage_dir = output_dir / stage.name
            controller_path = stage_dir / "best_phase_controller.json"
            result_path = stage_dir / "result.json"
            if controller_path.exists() and result_path.exists() and not args.restart:
                parameters = controller_parameters(
                    controller_path,
                    initial_gap_m=float(
                        json.loads(controller_path.read_text(encoding="utf-8"))[
                            "minimum_foot_surface_gap_m"
                        ]
                    ),
                )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                stage_summaries.append(result)
                print(f"stage={stage.name} resume controller={controller_path}", flush=True)
                continue

            parameters, rollout, history = optimize_stage(
                stage,
                parameters,
                runner=runner,
                executor=executor,
            )
            stage_dir.mkdir(parents=True, exist_ok=True)
            payload = controller_payload(
                parameters,
                rollout,
                stage=stage,
                source_controller=args.initial_controller,
                tracking_margin_m=args.tracking_margin_mm / 1000.0,
            )
            controller_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            (stage_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            result = {
                "stage": stage.name,
                "description": stage.description,
                "controller_path": str(controller_path),
                "parameters": {
                    name: float(value)
                    for name, value in zip(PARAMETER_NAMES, parameters, strict=True)
                },
                **rollout.summary,
            }
            result_path.write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            stage_summaries.append(result)
            (output_dir / "summary.json").write_text(
                json.dumps(stage_summaries, indent=2) + "\n", encoding="utf-8"
            )
    finally:
        if executor is not None:
            executor.shutdown()

    print(f"controller={output_dir / stages[-1].name / 'best_phase_controller.json'}")
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

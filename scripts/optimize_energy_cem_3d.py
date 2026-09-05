"""Warm-start CEM minimizing absolute mechanical J/m with motion gates.

Uses the exact paired-energy evaluator. This is a mechanical-work objective,
not a battery model. Geometry, foot-gap projection and servo gains stay fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import compare_locomotion_energy_3d as bench
from curl_robot_2d_mjx.cem_reference import COEFFICIENT_NAMES, load_cem_reference

_ARGS = None
_BASE = None


def initialize(args):
    global _ARGS, _BASE
    _ARGS = args
    _BASE = load_cem_reference(args.controller)


def config_from_vector(base, x):
    return replace(base, coefficients=tuple(x[:8]),
                   oscillator_rate_rad_s=float(x[8]),
                   oscillator_coupling_per_s=float(x[9]))


def evaluate(x):
    return bench.run_case(_ARGS, 'roll', None, 'candidate',
        config_override=config_from_vector(_BASE, x), save=False, quiet=True)


def objective(result, baseline):
    w = result['windows']['post_startup']
    b = baseline['windows']['post_startup']
    full = result['windows']['full']
    bf = baseline['windows']['full']
    speed_ratio = w['actual_speed_m_s'] / b['actual_speed_m_s']
    violations = {
        'speed': max(0., abs(speed_ratio - 1.) - .05) / .05,
        'turns': max(0., .9 - result['rolling_turns']/baseline['rolling_turns']) / .1,
        'lateral': max(0., abs(full['lateral_displacement_m']) - max(.25, abs(bf['lateral_displacement_m'])+.05)) / .1,
        'axis_tilt': max(0., full['max_axis_tilt_deg'] - max(10., bf['max_axis_tilt_deg']+2.)) / 5.,
        'self_penetration': max(0., full['max_self_penetration_m'] - bf['max_self_penetration_m'] - .0005) / .0005,
        'self_contact': max(0., full['self_contact_fraction'] - bf['self_contact_fraction'] - .01) / .01,
        'saturation': max(0., full['saturation_fraction'] - max(.01, bf['saturation_fraction'])) / .01,
        'positive_work': max(0., (w['positive_j_per_m'] or 1e6) / b['positive_j_per_m'] - 1.05) / .05,
    }
    feasible = all(v <= 1e-10 for v in violations.values())
    energy_ratio = (w['cot_absolute'] or 1e6) / b['cot_absolute']
    score = energy_ratio + .2 * abs(speed_ratio - 1.)
    if not feasible:
        score += 100. + 10. * sum(violations.values())
    return float(score), feasible, violations


def payload(base_path, x):
    base = json.loads(base_path.read_text())
    return dict(controller='phase_locked_oscillator',
        oscillator_rate_rad_s=float(x[8]), oscillator_period_s=float(2*np.pi/x[8]),
        oscillator_coupling_per_s=float(x[9]),
        minimum_foot_surface_gap_m=base['minimum_foot_surface_gap_m'],
        nominal_knee_bias_rad=base['nominal_knee_bias_rad'],
        foot_gap_tracking_margin_m=base['foot_gap_tracking_margin_m'],
        raw_coefficients=dict(zip(COEFFICIENT_NAMES, map(float, x[:8]))),
        optimization=dict(method='energy_cem_3d', source=str(base_path.resolve()),
            objective='absolute mechanical J/m plus small speed tracking cost',
            acceptance='speed within 5%, >=90% baseline turns; contact, tilt, drift, positive work gates'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controller', type=Path, default=bench.ref.DEFAULT_CONTROLLER_PATH)
    p.add_argument('--xml', type=Path, default=bench.ref.DEFAULT_XML_PATH)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--generations', type=int, default=6)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 43])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true', help='Resume completed generations without rerunning them')
    args = p.parse_args()
    if args.generations < 1 or args.population < 4 or args.workers < 1:
        p.error('Require generations >=1, population >=4, workers >=1')
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        p.error('Output directory must be empty')
    args.out.mkdir(parents=True, exist_ok=True)
    args.duration, args.warmup, args.dt = 10., 2., .002
    manifest = dict(status='RUNNING', hypothesis='Energy-aware CEM reduces absolute mechanical J/m by >=10% while retaining baseline rolling speed and motion gates.',
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (args.controller, args.xml, Path(__file__), Path(bench.__file__))},
        mujoco_version=bench.mj.__version__)
    if args.resume:
        previous = json.loads((args.out/'manifest.json').read_text())
        for key in ('generations', 'population', 'seeds', 'duration', 'warmup', 'dt'):
            if previous['settings'][key] != manifest['settings'][key]:
                p.error(f'Resume setting differs: {key}')
        for path in (args.controller, args.xml, Path(bench.__file__)):
            if previous['hashes'][str(path)] != manifest['hashes'][str(path)]:
                p.error(f'Resume input changed: {path}')
        manifest['previous_optimizer_sha256'] = previous['hashes'].get(str(Path(__file__)))
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    initialize(args)
    initial = np.array([*_BASE.coefficients, _BASE.oscillator_rate_rad_s, _BASE.oscillator_coupling_per_s])
    baseline = (json.loads((args.out/'baseline.json').read_text()) if args.resume else
                bench.run_case(args, 'roll', None, 'baseline', quiet=True))
    (args.out/'baseline.json').write_text(json.dumps(baseline, indent=2))
    print('Baseline '+json.dumps(baseline['windows']['post_startup']), flush=True)
    lower = np.array([-1,-1,-1.4,-1.4,-1,-1,-1.4,-1.4,.5,0])
    upper = np.array([1,1,1.4,1.4,1,1,1.4,1.4,6,8])
    winners = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args,)) as executor:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            mean = initial.copy()
            std = np.array([.06]*8 + [.18,.3])
            best_x, best = initial.copy(), baseline
            best_score = objective(best, baseline)[0]
            records = []
            start_generation = 0
            records_path = args.out/f'seed{seed}_candidates.json'
            if args.resume and records_path.exists():
                records = json.loads(records_path.read_text())
                start_generation = max(r['generation'] for r in records) + 1
                for generation in range(start_generation):
                    saved = [r for r in records if r['generation'] == generation]
                    if len(saved) != args.population:
                        raise ValueError('Incomplete saved generation')
                    rng.normal(mean, std, (args.population,10))
                    scores = np.array([r['score'] for r in saved])
                    samples = np.array([r['parameters'] for r in saved])
                    idx = int(np.argmin(scores))
                    if scores[idx] < best_score:
                        best_x, best, best_score = samples[idx].copy(), saved[idx]['result'], float(scores[idx])
                    elite = samples[np.argsort(scores)[:max(4,args.population//4)]]
                    mean = .7*elite.mean(0) + .3*mean
                    std = np.maximum(.7*elite.std(0)+.3*std, [.005]*8+[.015,.025])
                print(f'Resumed seed {seed} after {start_generation} generations',flush=True)
            for generation in range(start_generation, args.generations):
                samples = np.clip(rng.normal(mean, std, (args.population,10)), lower, upper)
                samples[0] = best_x
                samples[1] = mean
                results = list(executor.map(evaluate, samples))
                assessments = [objective(r, baseline) for r in results]
                scores = np.array([a[0] for a in assessments])
                idx = int(np.argmin(scores))
                if scores[idx] < best_score:
                    best_x, best, best_score = samples[idx].copy(), results[idx], float(scores[idx])
                elite = samples[np.argsort(scores)[:max(4,args.population//4)]]
                mean = .7*elite.mean(0) + .3*mean
                std = np.maximum(.7*elite.std(0)+.3*std, [.005]*8+[.015,.025])
                for x, result, assessment in zip(samples, results, assessments):
                    records.append(dict(generation=generation, parameters=x.tolist(),
                        score=assessment[0], feasible=assessment[1], violations=assessment[2], result=result))
                (args.out/f'seed{seed}_candidates.json').write_text(json.dumps(records, indent=2))
                (args.out/f'seed{seed}_best_controller.json').write_text(json.dumps(payload(args.controller,best_x),indent=2))
                print(f'Seed {seed} generation {generation+1}/{args.generations}: feasible {sum(a[1] for a in assessments)}/{args.population}; best abs CoT={best["windows"]["post_startup"]["cot_absolute"]:.4f}, speed={best["windows"]["post_startup"]["actual_speed_m_s"]:.4f}',flush=True)
            winners.append(dict(seed=seed, score=best_score, parameters=best_x.tolist(), result=best))
    winner = min(winners, key=lambda r:r['score'])
    (args.out/'best_controller.json').write_text(json.dumps(payload(args.controller,winner['parameters']),indent=2))
    (args.out/'winners.json').write_text(json.dumps(winners,indent=2))
    manifest['status']='COMPLETED'
    manifest['best_seed']=winner['seed']
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()

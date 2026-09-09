"""Add verified CEM rolling prehistory to an existing MJX transition rollout."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from scripts.collect_reference_roll_to_stand import (
    _collect_handoff, MODEL, REFERENCE, activate_planar_geometry,
    PUPPER_ORIGINAL_SHELL_60_PARAMETERS, load_cem_reference,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--xml", type=Path, default=MODEL)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if args.output.exists():
        parser.error("--output must be new")
    with np.load(args.rollout, allow_pickle=False) as archive:
        rollout = {k: archive[k] for k in archive.files}
    with np.load(args.bank, allow_pickle=False) as archive:
        bank = {k: archive[k] for k in archive.files}
    if "handoff_index" in rollout:
        raise ValueError("rollout already contains rolling prehistory")
    report = json.loads(args.bank.with_suffix(".summary.json").read_text(encoding="utf-8"))
    if hashlib.sha256(args.reference.read_bytes()).hexdigest() != report["reference_sha256"]:
        raise ValueError("reference hash does not match the handoff bank")
    if hashlib.sha256(args.xml.read_bytes()).hexdigest() != str(bank["model_xml_sha256"].item()):
        raise ValueError("model hash does not match the handoff bank")
    matches = np.ones(len(bank["qpos"]), dtype=bool)
    for name, tol in (("qpos", 1e-4), ("qvel", 1e-3), ("ctrl", 1e-4)):
        matches &= np.max(np.abs(bank[name] - rollout[name][0]), axis=1) <= tol
    ids = np.flatnonzero(matches)
    if len(ids) != 1:
        raise ValueError("cannot uniquely match rollout initial state to bank")
    index = int(ids[0])
    activate_planar_geometry(PUPPER_ORIGINAL_SHELL_60_PARAMETERS)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    control_dt = float(rollout.get("control_dt", .02))
    stride = round(control_dt / model.opt.timestep)
    if stride < 1 or not np.isclose(stride * model.opt.timestep, control_dt):
        raise ValueError("control timestep must be a multiple of physics timestep")
    stop = float(bank["time_s"][index])
    row = _collect_handoff(model, load_cem_reference(args.reference), minimum_turns=0,
        replay_until_s=stop, max_time_s=stop+1., preroll_s=args.seconds)
    # Reject discontinuous splices caused by runtime/physics differences.
    for name, tol in (("qpos", 1e-4), ("qvel", 1e-3), ("ctrl", 1e-4)):
        error = float(np.max(np.abs(row[name] - bank[name][index])))
        if error > tol:
            raise ValueError(f"CEM replay endpoint {name} differs by {error:g}; use collection runtime/physics")
    history = row["history"]
    indices = list(range(len(history)-1-stride, -1, -stride))[::-1]
    prefix_count = len(indices)
    if not prefix_count:
        raise ValueError("not enough prehistory for one control frame")
    result = {}
    for name, values in rollout.items():
        if values.ndim and len(values) == len(rollout["qpos"]):
            if name not in ("qpos", "qvel", "ctrl"):
                raise ValueError(f"unsupported time-series field: {name}")
        else:
            result[name] = values
    for field, name in enumerate(("qpos", "qvel", "ctrl")):
        prefix = np.stack([history[i][field] for i in indices])
        result[name] = np.concatenate((prefix, rollout[name]))
    result.update(handoff_index=prefix_count, handoff_time_s=prefix_count*control_dt,
                  source_bank_index=index, mode="CEM roll -> RL stand")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **result)
    print(f"[roll-to-stand replay] pre-roll={prefix_count*control_dt:.2f}s "
          f"bank_index={index} endpoint verified | {args.output}")


if __name__ == "__main__":
    main()

"""Plot compact and direct stand-to-reference rolling evaluations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CASES = (
    ("compact_baseline_10s", "Compact baseline", "#333333"),
    ("standard_ramp_10s", "Stand direct, standard ramp", "#0072B2"),
    ("full_reference_10s", "Stand direct, full reference", "#D55E00"),
)


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    series = [(label, color, load(args.root / folder / "roll.csv"))
              for folder, label, color in CASES]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    panels = (
        (axes[0, 0], "x_m", "Forward position (m)", None),
        (axes[0, 1], "roll_phase_rad", "Rolling phase (turns)", 2 * np.pi),
        (axes[1, 0], "height_m", "Torso height (m)", None),
    )
    for axis, key, ylabel, divisor in panels:
        for label, color, values in series:
            y = values[key] if divisor is None else values[key] / divisor
            axis.plot(values["time_s"], y, color=color, lw=1.7, label=label)
        axis.set(xlabel="Time (s)", ylabel=ylabel)
        axis.grid(alpha=0.25)

    power_axis = axes[1, 1]
    for label, color, values in series:
        mask = values["time_s"] <= 0.5
        absolute_power = values["positive_power_w"] + values["negative_power_w"]
        power_axis.plot(values["time_s"][mask], absolute_power[mask],
                        color=color, lw=1.7, label=label)
    power_axis.set(xlabel="Time (s)", ylabel="Absolute mechanical power (W)")
    power_axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Direct stand-to-rolling reference fails to enter the rolling cycle",
                 fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    fig.savefig(args.out.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()

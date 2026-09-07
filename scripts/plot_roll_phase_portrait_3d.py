"""Plot rolling-axis phase portraits from a 3-D energy rollout CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
import matplotlib as mpl
from matplotlib.collections import LineCollection
import numpy as np


def moving_average(values: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 1:
        return values.copy()
    if samples % 2 == 0:
        samples += 1
    pad = samples // 2
    padded = np.pad(values, pad, mode='edge')
    return np.convolve(padded, np.ones(samples) / samples, mode='valid')


def colored_segments(ax, x, y, t, *, cmap, norm, linewidth=0.8,
                     alpha=0.75, break_wrap=False):
    points = np.column_stack((x, y))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    keep = np.isfinite(segments).all(axis=(1, 2))
    if break_wrap:
        keep &= np.abs(np.diff(x)) < np.pi
    collection = LineCollection(segments[keep], cmap=cmap, norm=norm,
                                linewidth=linewidth, alpha=alpha,
                                rasterized=False)
    collection.set_array(t[:-1][keep])
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True,
                        help='Output path without extension')
    parser.add_argument('--startup-s', type=float, default=2.0)
    parser.add_argument('--smooth-ms', type=float, default=20.0)
    parser.add_argument('--start-s', type=float, default=0.0)
    parser.add_argument('--end-s', type=float, default=None)
    args = parser.parse_args()

    with args.csv.open(newline='') as handle:
        records = list(csv.DictReader(handle))
    required = ('time_s', 'roll_phase_wrapped_rad', 'roll_rate_rad_s')
    if not records or any(name not in records[0] for name in required):
        raise ValueError('CSV lacks rolling phase portrait fields')
    t = np.asarray([float(r['time_s']) for r in records])
    phase = np.asarray([float(r['roll_phase_wrapped_rad']) for r in records])
    rate_raw = np.asarray([float(r['roll_rate_rad_s']) for r in records])
    dt = float(np.median(np.diff(t)))
    smooth_samples = max(1, round(args.smooth_ms / 1000.0 / dt))
    rate = moving_average(rate_raw, smooth_samples)
    acceleration = np.gradient(rate, dt)

    end_s = float(t[-1] + dt) if args.end_s is None else args.end_s
    if not 0 <= args.start_s < end_s <= float(t[-1] + dt) + 1e-9:
        raise ValueError('Require 0 <= start-s < end-s <= rollout duration')
    interval = (t >= args.start_s) & (t < end_s)
    startup = interval & (t < args.startup_s)
    analysis = interval & ~startup
    if analysis.sum() < 2:
        raise ValueError('No samples after startup interval')

    mpl.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.linewidth': 0.8,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.05), constrained_layout=True)
    signal = '#0F4D92'
    startup_color = '#B9B9B9'
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'time_blue', ['#A9C7E8', '#3775BA', signal])
    color_start = max(args.start_s, args.startup_s)
    norm = mpl.colors.Normalize(vmin=color_start, vmax=end_s)

    ax = axes[0]
    if startup.sum() >= 2:
        colored_segments(ax, phase[startup], rate[startup], t[startup],
                         cmap=mpl.colors.ListedColormap([startup_color]),
                         norm=mpl.colors.Normalize(0, 1), linewidth=0.65,
                         alpha=0.6, break_wrap=True)
    colored_segments(ax, phase[analysis], rate[analysis], t[analysis],
                     cmap=cmap, norm=norm, linewidth=0.85,
                     alpha=0.78, break_wrap=True)
    first_interval = int(np.flatnonzero(interval)[0])
    ax.scatter(phase[first_interval], rate[first_interval], s=20,
               facecolor='white', edgecolor='#333333',
               linewidth=0.8, zorder=5, label='start')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                  [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel(r'Wrapped roll phase, $\theta_y$ (rad)')
    ax.set_ylabel(r'Roll rate, $\omega_y$ (rad s$^{-1}$)')
    ax.set_title('Phase–rate portrait', loc='left', fontsize=8, pad=7)
    if startup.any():
        ax.text(0.02, 0.03, f'grey: 0–{args.startup_s:g} s',
                transform=ax.transAxes, color='#767676', fontsize=6.5,
                ha='left', va='bottom')
    ax.text(-0.13, 1.06, 'a', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')

    ax = axes[1]
    colored_segments(ax, rate[analysis], acceleration[analysis], t[analysis],
                     cmap=cmap, norm=norm, linewidth=0.7, alpha=0.65)
    first_analysis = int(np.flatnonzero(analysis)[0])
    ax.scatter(rate[first_analysis], acceleration[first_analysis], s=20, facecolor='white',
               edgecolor='#333333', linewidth=0.8, zorder=5)
    ax.set_xlabel(r'Roll rate, $\omega_y$ (rad s$^{-1}$)')
    ax.set_ylabel(r'Roll acceleration, $\alpha_y$ (rad s$^{-2}$)')
    ax.set_title(f'Rate–acceleration portrait, {color_start:g}–{end_s:g} s '
                 f'({args.smooth_ms:g} ms smoothing)',
                 loc='left', fontsize=8, pad=7)
    ax.text(-0.13, 1.06, 'b', transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top')

    for ax in axes:
        ax.axhline(0, color='#D8D8D8', lw=0.6, zorder=0)
        ax.axvline(0, color='#D8D8D8', lw=0.6, zorder=0)
        ax.grid(False)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.035, pad=0.025)
    cbar.set_label('Time (s)')
    cbar.outline.set_linewidth(0.6)
    fig.suptitle(f'Rolling-axis state portraits, {args.start_s:g}–{end_s:g} s '
                 '· front ABD −10°, rear ABD +10°',
                 fontsize=9, y=1.04)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for extension, kwargs in (
        ('svg', {}), ('pdf', {}), ('tiff', {'dpi': 600}),
        ('png', {'dpi': 300})):
        fig.savefig(args.out.with_suffix('.' + extension),
                    bbox_inches='tight', facecolor='white', **kwargs)
    plt.close(fig)

    summary = args.out.with_name(args.out.name + '_qa.txt')
    summary.write_text(
        f'source_csv={args.csv.resolve()}\n'
        f'duration_s={t[-1] + dt:.6f}\n'
        f'dt_s={dt:.6f}\n'
        f'startup_s={args.startup_s:.6f}\n'
        f'plot_interval_s={args.start_s:.6f},{end_s:.6f}\n'
        f'smoothing_ms={args.smooth_ms:.6f}\n'
        f'raw_roll_rate_range_rad_s={rate_raw.min():.6f},{rate_raw.max():.6f}\n'
        f'smoothed_roll_rate_range_rad_s={rate.min():.6f},{rate.max():.6f}\n'
        f'smoothed_acceleration_range_rad_s2={acceleration.min():.6f},{acceleration.max():.6f}\n'
        'acceleration_display=gradient of centered moving-average roll rate; raw MuJoCo acceleration remains in CSV\n',
        encoding='utf-8')


if __name__ == '__main__':
    main()

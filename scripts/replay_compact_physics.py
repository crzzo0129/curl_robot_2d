"""View saved compact diagnostic states, or export a GIF; no new simulation."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from curl_robot_2d_mjx.autonomous_startup_3d import validate_model_fingerprint
from curl_robot_2d_mjx.config_3d import Rolling3DConfig
from curl_robot_2d_mjx.environment_3d import model_path_3d, apply_physics_options_3d
from scripts.diagnose_compact_physics import fixed_fixture_xml


def frame_indices(times, fps, speed):
    times = np.asarray(times)
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('trajectory timestamps must be finite and strictly increasing')
    samples = np.arange(times[0], times[-1], speed / fps)
    return np.append(np.clip(np.searchsorted(times, samples), 0, len(times)-1), len(times)-1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='fixed or ground subdirectory containing trajectory.npz and summary.json')
    p.add_argument('--gif', type=Path, help='export GIF instead of opening a desktop window')
    p.add_argument('--speed', type=float, default=.5, help='playback speed; default half speed')
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--loop', action='store_true', help='loop desktop playback until window is closed')
    p.add_argument('--azimuth', type=float, default=90.)
    args = p.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0 or not 1 <= args.fps <= 60 or not math.isfinite(args.azimuth):
        p.error('speed must be positive and finite, fps 1..60, azimuth finite')
    if args.gif and args.gif.exists():
        p.error('GIF already exists; choose a new output path')
    summary = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
    with np.load(args.run / 'trajectory.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in ('time', 'qpos', 'qvel', 'ctrl', 'metrics', 'metric_names')}
    indices = frame_indices(arrays['time'], args.fps, args.speed)
    source = model_path_3d('rollingquad_2_primitive')
    validate_model_fingerprint(summary, source, context='diagnostic replay')
    import mujoco as mj
    if summary['mode'] == 'fixed':
        model = mj.MjModel.from_xml_string(fixed_fixture_xml(source, summary['options']['fixture_height_m']))
    else:
        model = mj.MjModel.from_xml_path(str(source))
    apply_physics_options_3d(model, Rolling3DConfig(**summary['physics']))
    for key, size in (('qpos', model.nq), ('qvel', model.nv), ('ctrl', model.nu)):
        if arrays[key].shape != (len(arrays['time']), size) or not np.isfinite(arrays[key]).all():
            raise ValueError(f'invalid or nonfinite {key} trajectory')
    data = mj.MjData(model)
    torso = model.body('torso').id

    def set_frame(i):
        # Deliberate state replay for visualization only. No mj_step and no
        # modification of the original diagnostic trajectories or results.
        data.qpos[:] = arrays['qpos'][i]
        data.qvel[:] = arrays['qvel'][i]
        data.ctrl[:] = arrays['ctrl'][i]
        data.time = arrays['time'][i]
        mj.mj_forward(model, data)

    def camera_setup(camera):
        camera.lookat[:] = data.xpos[torso]
        camera.distance = .75
        camera.azimuth = args.azimuth
        camera.elevation = -15.

    set_frame(0)
    stop = ', '.join(summary['stop_reasons']) or 'schedule ended'
    print(f'[replay] {summary["mode"]}, speed={args.speed}x, recorded stop: {stop}', flush=True)
    if args.gif:
        from PIL import Image, ImageDraw
        camera = mj.MjvCamera()
        camera_setup(camera)
        images = []
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 640)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 480)
        with mj.Renderer(model, height=480, width=640) as renderer:
            for i in indices:
                set_frame(i)
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 48), fill=(15, 20, 28))
                row = dict(zip(arrays['metric_names'], arrays['metrics'][i]))
                draw.text((10, 6), f'SAVED PHYSICS REPLAY | {summary["mode"]} | t={data.time:.2f}s | {args.speed:g}x', fill='white')
                draw.text((10, 24), f'fold={row["progress"]:.0%}  tilt={row["tilt_rad"]:.3f}rad  vz={row["root_vz_m_s"]:.3f}m/s  stop={stop}', fill='white')
                images.append(frame.convert('P', palette=Image.Palette.ADAPTIVE, colors=192))
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        durations = [round(1000 / args.fps)] * len(images)
        durations[-1] = 1500
        images[0].save(args.gif, save_all=True, append_images=images[1:], duration=durations, loop=0)
        print(f'[replay] saved {args.gif}', flush=True)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                camera_setup(viewer.cam)
            while viewer.is_running():
                started = time.monotonic()
                for n, i in enumerate(indices):
                    if not viewer.is_running():
                        return
                    with viewer.lock():
                        set_frame(i)
                    viewer.sync()
                    time.sleep(max(0., started + (n+1)/args.fps - time.monotonic()))
                if not args.loop:
                    break


if __name__ == '__main__':
    main()

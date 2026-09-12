"""Export videos of the selected student from its actual MJX evaluation states."""

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import sys
import zipfile

from scripts.run_rolling_low_speed_recovery import distill_command, run, write_json


def choose_episodes(report):
    """Choose near bin-center commands before looking at success/failure."""
    edges = report['speed_bin_edges_m_s']
    selected = []
    for bin_index, speed in enumerate(('low', 'medium', 'high')):
        center = (edges[bin_index] + edges[bin_index + 1]) / 2
        for direction, yaw_target in (('straight', 0.), ('left', .05), ('right', -.05)):
            candidates = []
            for episode in report['per_episode']:
                vx, yaw = episode['forward_command_m_s'], episode['yaw_command_rad_s']
                in_bin = (edges[bin_index] <= vx < edges[bin_index + 1]
                          or bin_index == 2 and vx == edges[-1])
                in_turn = abs(yaw) <= .001 if direction == 'straight' else yaw * yaw_target > 0
                if in_bin and in_turn:
                    candidates.append(episode)
            if not candidates:
                raise ValueError(f'No episode for {speed}/{direction}')
            episode = min(candidates, key=lambda e: (
                ((e['forward_command_m_s'] - center) / (edges[bin_index+1] - edges[bin_index]))**2
                + ((e['yaw_command_rad_s'] - yaw_target) / .06)**2, e['episode']))
            selected.append({'label': f'{speed}_{direction}', 'episode': episode['episode'],
                             'vx_command': episode['forward_command_m_s'],
                             'yaw_command': episode['yaw_command_rad_s']})
    return selected


def write_gallery(out, cases, student):
    speed_labels = {'low': '低速', 'medium': '中速', 'high': '高速'}
    turn_labels = {'straight': '直行', 'left': '左转', 'right': '右转'}
    cards = []
    for case in cases:
        speed, turn = case['label'].split('_')
        stats = case['summary']
        videos = ''.join(f'<div><small>{"俯视" if v["view"] == "top" else "斜视"}</small>'
                         f'<video controls muted playsinline preload="metadata" src="{html.escape(v["file"])}"></video></div>'
                         for v in case['videos'])
        status = '通过当前滚动判据' if stats['success'] else '未通过当前滚动判据'
        cards.append(f'<article><h2>{speed_labels[speed]} · {turn_labels[turn]}</h2>'
                     f'<p>目标 vx {case["vx_command"]:.3f} m/s · yaw {case["yaw_command"]:+.3f} rad/s</p>'
                     f'{videos}<p>{status} · 学生运行 {stats["student_duration_s"]:.2f} s<br>'
                     f'vx MAE {stats["forward_mae_m_s"]:.3f} m/s · yaw MAE {stats["yaw_mae_rad_s"]:.3f} rad/s</p></article>')
    content = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>滚动策略可视化</title>
<style>body{margin:28px;background:#101720;color:#e6edf6;font:16px system-ui}h1{font-size:28px}
main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}article{background:#1c2736;padding:16px;border-radius:12px}
h2{font-size:20px}p,small{color:#b9c8dc}video{width:100%;margin:8px 0}button{padding:10px 18px;margin:12px 8px 20px 0}
@media(max-width:850px){main{grid-template-columns:1fr}}</style>
<h1>滚动策略 · 三种速度与转向</h1><p>从教师预热后的滚动态接管，学生最多运行 10 秒。视频回放实际 MJX 轨迹。
场景按命令接近分组中心选取，保留失败结果；这 9 个场景不代表整体成功率。</p>
<button onclick="document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})})">从头播放全部</button>
<button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">暂停全部</button><main>'''
    (out/'index.html').write_text(content + ''.join(cards) + '</main></html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/rolling_low_speed_20260912_065438'))
    parser.add_argument('--student', type=Path, help='optional alternate student_params; defaults to selected_student')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--views', choices=('both', 'oblique', 'top'), default='both')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.run/'recovery.json').read_text(encoding='utf-8'))
    settings = manifest['settings']
    student = args.student or Path(manifest['selected_student'])
    review = Path(manifest['rechecks'][-1]) if manifest.get('rechecks') else args.run
    panel = json.loads((review/'baseline/command_evaluation.json').read_text(encoding='utf-8'))
    cases = choose_episodes(panel)
    out = (args.out or Path(f'results/rolling_policy_video_{datetime.now():%Y%m%d_%H%M%S}')).resolve()
    cache = review/'evaluation_snapshots.npz'
    if not cache.is_file():
        cache = out/'evaluation_snapshots.npz'
    command = distill_command(settings, student, out/'evaluation', seed=settings['seed'],
                              evaluation=True, chunk_steps=500, snapshot_cache=cache)
    command.remove('--record-diagnostics')  # No extra teacher label queries are needed for a movie.
    command += ['--record-rollout-episodes', *[str(c['episode']) for c in cases]]
    if args.dry_run:
        print(json.dumps({'student': str(student), 'cases': cases, 'command': command}, indent=2))
        return
    if not student.is_file():
        parser.error(f'Missing student checkpoint: {student}')
    if out.exists() or out.with_suffix('.zip').exists():
        parser.error(f'Output exists: {out}; choose another --out')
    try:
        import imageio.v2
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        parser.error(f'Video dependency missing: {error}; install with python -m pip install imageio imageio-ffmpeg')
    out.mkdir(parents=True)
    record = {'student': str(student), 'cases': cases, 'evaluation_command': command,
              'selection': 'Closest to speed-bin center and yaw 0/+0.05/-0.05; outcomes ignored when choosing.'}
    write_json(out/'visualization.json', record)
    try:
        run(command, out/'evaluation.log')
        evaluation = json.loads((out/'evaluation/command_evaluation.json').read_text())
        for case in cases:
            case['summary'] = evaluation['per_episode'][case['episode']]
            case['vx_command'] = case['summary']['forward_command_m_s']
            case['yaw_command'] = case['summary']['yaw_command_rad_s']
            case['videos'] = []
            views = ('oblique', 'top') if args.views == 'both' else (args.views,)
            for view in views:
                video = f'{case["label"]}_{view}.mp4'
                render = [sys.executable, '-u', '-m', 'scripts.render_mjx_3d_policy',
                          str(out/f'evaluation/rollouts/episode_{case["episode"]:03d}.npz'),
                          '--geometry', settings['geometry'], '--physics-profile', 'cg20',
                          '--output', str(out/video), '--fps', '25', '--width', '720', '--height', '540',
                          '--camera-distance', '2.0', '--azimuth', '135',
                          '--elevation', '-85' if view == 'top' else '-25', '--mujoco-gl', 'egl']
                run(render, out/f'{case["label"]}_{view}.log')
                case['videos'].append({'view': view, 'file': video})
                write_json(out/'visualization.json', record)
        write_gallery(out, cases, student)
    finally:
        with zipfile.ZipFile(out.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob('*')):
                if path.is_file() and path.name != 'evaluation_snapshots.npz':
                    archive.write(path, path.relative_to(out.parent))
        print(f'[video bundle] {out.with_suffix(".zip")}', flush=True)
    print(f'[watch after download] {out / "index.html"}', flush=True)


if __name__ == '__main__':
    main()

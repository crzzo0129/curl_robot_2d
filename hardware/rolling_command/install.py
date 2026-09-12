"""Stage or install reviewed robot sources; never launches a controller."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path,
        default=Path('/home/pi/pupperv3-monorepo/ros2_ws'))
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--stage', type=Path, help='New isolated colcon workspace')
    choice.add_argument('--install', action='store_true', help='Back up and install source files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    package = args.workspace.resolve() / 'src/neural_controller'
    baseline = json.loads((root / 'baseline_sha256.json').read_text())
    for relative, expected in baseline.items():
        if digest(package / relative) != expected:
            parser.error('Robot source changed since review; reconcile before installing: ' + relative)
    payload = root / 'neural_controller'
    files = {str(p.relative_to(payload)): p.read_bytes().replace(b'\r\n', b'\n')
             for p in payload.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    cmake = (package / 'CMakeLists.txt').read_text()
    if cmake.count('ament_package()') != 1:
        parser.error('Unexpected CMakeLists.txt')
    cmake = cmake.replace('ament_package()',
        'install(PROGRAMS scripts/rolling_gamepad_mapping.py DESTINATION lib/${PROJECT_NAME})\n\nament_package()')
    files['CMakeLists.txt'] = cmake.encode()
    if args.stage:
        stage = args.stage.resolve()
        if stage.exists():
            parser.error('Stage exists; select a new path')
        destination = stage / 'src/neural_controller'
        shutil.copytree(package, destination, ignore=shutil.ignore_patterns('modules', 'models', 'launch'))
        for name in ('modules', 'models', 'launch'):
            (destination / name).symlink_to(package / name, target_is_directory=True)
    else:
        destination = package
        backup = args.workspace / ('rolling_command_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        backup.mkdir()
        (backup / 'COLCON_IGNORE').touch()
        for relative in files:
            source = package / relative
            if source.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, saved)
        (backup / 'manifest.json').write_text(json.dumps({
            'baseline_sha256': baseline, 'installed': list(files)}, indent=2))
        print('Backup:', backup)
    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if relative.startswith('scripts/'):
            path.chmod(path.stat().st_mode | 0o111)
    print('Prepared:', destination)
    print('No launch, motor enable, model replacement or gamepad config replacement performed.')


if __name__ == '__main__':
    main()

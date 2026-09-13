"""Back up and install the rolling-only P=5.5 candidate; never starts/stops motors."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_copy(source, destination):
    destination = destination.resolve()
    temporary = destination.with_name(destination.name + '.pd55-incoming')
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def restore(backup):
    manifest = json.loads((backup / 'manifest.json').read_text())
    for item in manifest['restore_files']:
        saved = backup / item['backup_name']
        if digest(saved) != item['sha256']:
            raise RuntimeError('Backup hash mismatch: ' + str(saved))
    for item in manifest['restore_files']:
        atomic_copy(backup / item['backup_name'], Path(item['destination']))
    print('Restored source, library and configuration. Restart the original launch when stationary.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('/home/pi/pupperv3-monorepo/ros2_ws'))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--install', action='store_true')
    mode.add_argument('--rollback', type=Path)
    args = parser.parse_args()
    if args.rollback:
        restore(args.rollback.resolve())
        return
    workspace = args.workspace.resolve()
    package = workspace / 'src/neural_controller'
    payload = Path(__file__).resolve().parent
    runtime = package / 'include/neural_controller/rolling_command_runtime.inc'
    config = package / 'launch/config_rollingquad_gamepad.yaml'
    model = package / 'models/rolling_command_ppo_000000081920.json'
    library = (workspace / 'install/neural_controller/lib/libneural_controller.so').resolve()
    if not library.is_file():
        library = (workspace / 'build/neural_controller/libneural_controller.so').resolve()
    if digest(runtime) != '0e71371e3432648ed7153af41ed402aef061b7a7c4f6f215c30fa35bdf881697':
        raise RuntimeError('Rolling runtime changed since review; refusing to overwrite it')
    if digest(model) != '902d8c08f530cfe752e394332e45d5ccbe7e641fba7f5bcf316b4c4938f74e97':
        raise RuntimeError('Original actor changed since review')
    new_model_name = 'rolling_command_ppo_000000081920_pd55.json'
    new_model = payload / new_model_name
    old_doc, new_doc = (json.loads(p.read_text()) for p in (model, new_model))
    assert old_doc['layers'] == new_doc['layers'], 'Policy weights must be identical'
    assert new_doc['kps'] == [5.0, 5.5, 5.5] * 4
    assert new_doc['kds'] == [0.1] * 12
    assert new_doc['export_contract'] == 'rolling_command_ppo_36x20_batchnorm_v2_joint_gains'
    text = config.read_text()
    old_setting = 'rolling_model_path: ' + str(model)
    if text.count(old_setting) != 1:
        raise RuntimeError('Unexpected active rolling model configuration')
    backup = workspace / ('rolling_pd55_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    backup.mkdir()
    (backup / 'COLCON_IGNORE').touch()
    entries = []
    for name, path in [('runtime.inc', runtime), ('config.yaml', config), ('model.json', model), ('library.so', library)]:
        shutil.copy2(path, backup / name)
        entries.append(dict(backup_name=name, destination=str(path.resolve()), sha256=digest(path)))
    manifest = dict(status='backed_up', restore_files=entries, backup=str(backup),
                    motors_started=False, restart_required=True)
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('BACKUP: ' + str(backup), flush=True)
    stage = backup / 'isolated_build'
    staged_package = stage / 'src/neural_controller'
    shutil.copytree(package, staged_package, ignore=shutil.ignore_patterns('models', 'modules', '__pycache__'))
    for name in ['models', 'modules']:
        if (package / name).exists():
            (staged_package / name).symlink_to(package / name, target_is_directory=True)
    new_runtime = payload / 'neural_controller/include/neural_controller/rolling_command_runtime.inc'
    shutil.copy2(new_runtime, staged_package / 'include/neural_controller/rolling_command_runtime.inc')
    build_script = stage / 'build.sh'
    build_script.write_text('set -euo pipefail\nsource /opt/ros/jazzy/setup.bash\n'
        'source "$ROLLING_PD55_WORKSPACE/install/setup.bash"\n'
        'colcon build --packages-select neural_controller --executor sequential '
        '--cmake-args -DCMAKE_BUILD_TYPE=Release\n')
    env = dict(os.environ, ROLLING_PD55_WORKSPACE=str(workspace), CMAKE_BUILD_PARALLEL_LEVEL='2')
    with (backup / 'build.log').open('w') as log:
        subprocess.run(['bash', str(build_script)], cwd=stage, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    built = stage / 'build/neural_controller/libneural_controller.so'
    dependencies = subprocess.run(['ldd', str(built)], check=True, capture_output=True, text=True)
    if 'not found' in dependencies.stdout:
        raise RuntimeError('Built library has unresolved dependencies: ' + dependencies.stdout)
    # Check again after the build so concurrent edits are never overwritten.
    for item in entries:
        if digest(Path(item['destination'])) != item['sha256']:
            raise RuntimeError('Robot files changed during build; installation cancelled')
    new_config = backup / 'config_pd55.yaml'
    new_config.write_text(text.replace(old_setting, 'rolling_model_path: ' + str(package / 'models' / new_model_name)))
    try:
        atomic_copy(new_model, package / 'models' / new_model_name)
        atomic_copy(new_runtime, runtime)
        # Atomic replacement keeps any currently mapped old library intact.
        atomic_copy(built, library)
        atomic_copy(new_config, config)
    except BaseException:
        restore(backup)
        raise
    manifest.update(status='installed', runtime_sha256=digest(runtime), model_sha256=digest(package / 'models' / new_model_name),
                    library_sha256=digest(library), kps=new_doc['kps'], kds=new_doc['kds'])
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (workspace / 'rolling_pd55_upgrade.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print('Installed. Restart the original launch when stationary; no motor commands were sent.')


if __name__ == '__main__':
    main()

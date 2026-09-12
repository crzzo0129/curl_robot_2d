"""Array-only evaluation snapshot cache; never deserialize executable objects."""

import hashlib
import json
from pathlib import Path

import numpy as np


def evaluation_environment_seed(training_seed, eval_seed, explicit_seed=None):
    if explicit_seed is not None:
        return explicit_seed
    return eval_seed if eval_seed is not None else training_seed


def snapshot_digest(arrays, paths):
    digest = hashlib.sha256()
    for path, value in zip(paths, arrays):
        value = np.asarray(value)
        digest.update(json.dumps([path, value.dtype.str, list(value.shape)]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def save_snapshot_arrays(path, arrays, paths, contract):
    arrays = [np.asarray(a) for a in arrays]
    if len(arrays) != len(paths):
        raise ValueError('Snapshot leaf names and arrays differ')
    if any(a.dtype.hasobject for a in arrays):
        raise ValueError('Object arrays are forbidden in evaluation snapshots')
    manifest = {'version': 1, 'contract': contract, 'paths': paths,
                'initial_state_sha256': snapshot_digest(arrays, paths)}
    path = Path(path)
    # Exclusive creation preserves the first actual state pool for all candidates.
    with path.open('xb') as handle:
        np.savez_compressed(handle, metadata=np.asarray(json.dumps(manifest)),
                            **{f'leaf_{i}': a for i, a in enumerate(arrays)})
    return manifest


def load_snapshot_arrays(path, templates, paths, contract):
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive['metadata'].item()))
        if (manifest.get('version') != 1 or manifest.get('contract') != contract
                or manifest.get('paths') != paths):
            raise ValueError('Evaluation snapshot contract/schema mismatch; choose a new cache path')
        if len(archive.files) != len(templates) + 1:
            raise ValueError('Evaluation snapshot leaf count mismatch')
        arrays = [archive[f'leaf_{i}'] for i in range(len(templates))]
    for value, template in zip(arrays, templates):
        if value.shape != template.shape or value.dtype != template.dtype:
            raise ValueError('Evaluation snapshot shape/dtype mismatch')
    if snapshot_digest(arrays, paths) != manifest['initial_state_sha256']:
        raise ValueError('Evaluation snapshot checksum mismatch')
    return arrays, manifest

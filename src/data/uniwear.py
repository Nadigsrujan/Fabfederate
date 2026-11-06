"""Admit the publisher-smoothed NUAA subset without mixing PHM/NASA semantics."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fabfederate.contracts import PreparedData
from fabfederate.data.partitioning import assign_clients
from fabfederate.data.splitting import build_group_splits
from fabfederate.utils.io import file_hash, write_json

SOURCE_URL = 'https://github.com/katulu-io/uniwear-dataset'
FEATURES = ['force_z', 'vibration_x', 'vibration_y']
TARGET_NOTE = ('Publisher-smoothed/downsampled wear values in mm, approximately 2 Hz. '
               'Derived targets are not independent original per-row wear measurements. '
               'Experiment-heldout offline benchmark; no NASA/PHM pooling or causal online claim.')


def prepare_uniwear(config: dict, output_dir: Path) -> PreparedData:
    path = Path(config['data_path'])
    source = pd.read_csv(path)
    required = set(FEATURES) | {'timestamp', 'tool_wear', 'dataset_tag', 'experiment_tag'}
    if not required.issubset(source):
        raise ValueError('Uniwear CSV lacks its documented signal/target/group schema')
    frame = source.loc[source.dataset_tag.eq('nuaa')].copy()
    if frame.empty or set(frame.experiment_tag.unique()) != {f'W{i}' for i in range(1, 10)}:
        raise ValueError('Expected the complete nine-experiment NUAA subset')
    numeric = frame[FEATURES + ['timestamp', 'tool_wear']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or frame.tool_wear.lt(0).any():
        raise ValueError('Uniwear values must be finite with nonnegative wear')
    if frame.duplicated(['experiment_tag', 'timestamp']).any():
        raise ValueError('Ambiguous experiment/timestamp identity')
    if config['wear_unit'] != 'mm':
        raise ValueError('Uniwear targets are in mm')
    groups = [SimpleNamespace(group_id=g) for g in sorted(frame.experiment_tag.unique())]
    splits = build_group_splits(groups, config['split_seed'], config['test_fraction'],
                                config['validation_fraction'])
    frame = frame.sort_values(['experiment_tag', 'timestamp']).reset_index(drop=True)
    frame['window_id'] = frame.groupby('experiment_tag').cumcount()
    frame['dataset_id'] = 'uniwear_nuaa'
    frame['group_id'] = frame.experiment_tag
    frame['run_id'] = 'nuaa_' + frame.experiment_tag
    frame['split'] = frame.group_id.map(splits)
    frame['wear_target'] = frame.tool_wear
    frame['wear_unit'] = 'mm'
    frame['condition'] = 'publisher_nuaa_orthogonal'
    frame['window_start'] = frame.window_id
    frame['window_end'] = frame.window_id + 1
    names = FEATURES.copy()
    frame = frame[['dataset_id', 'run_id', 'group_id', 'window_id', 'window_start',
                   'window_end', 'split', 'wear_target', 'wear_unit', 'condition', *names]]
    frame, clients = assign_clients(frame, config['scenario'], config['num_clients'],
                                    config['partition_seed'], config['few_shot_groups'])
    metadata = {'schema_version': '1', 'feature_names': names, 'wear_unit': 'mm',
                'source_url': SOURCE_URL, 'source_sha256': file_hash(path),
                'target_provenance': TARGET_NOTE, 'config': config,
                'definitions': {'features': 'publisher-smoothed signal values; no timestamp, group or target inputs'}}
    audit = {'run_count': 9, 'labeled_run_count': 9, 'unlabeled_run_count': 0,
             'independent_group_count': 9, 'target_provenance': TARGET_NOTE,
             'channels': {name: {'samples': len(frame), 'nonfinite_samples': 0} for name in names}}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / 'features.parquet', index=False)
    for name, value in [('schema', metadata), ('audit', audit),
                        ('split_manifest', {'group_splits': splits, 'policy': 'whole experiments held out before model fitting'}),
                        ('client_manifest', clients)]:
        write_json(output_dir / f'{name}.json', value)
    return PreparedData(frame, names, metadata)


def acquire_uniwear(root: str | Path = 'data/raw/uniwear') -> dict:
    """Download data only from the pinned author release, with a 10 MB budget."""
    import requests
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    commit = '66c894f2449f4393126e906e086485a8f9d95171'
    url = f'https://raw.githubusercontent.com/katulu-io/uniwear-dataset/{commit}/data/uniwear.csv'
    target = root / 'uniwear.csv'
    temporary = root / 'uniwear.download'
    if not target.exists():
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                total = 0
                with temporary.open('wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        total += len(block)
                        if total > 10_000_000:
                            raise ValueError('Uniwear source exceeds the declared 10 MB budget')
                        stream.write(block)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {'url': url, 'commit': commit, 'sha256': file_hash(target),
                'bytes': target.stat().st_size, 'license': 'CC BY 4.0 (bundled data)',
                'attribution': 'Katulu Uniwear Dataset; original NUAA and PHM2010 dataset authors',
                'processing': TARGET_NOTE}
    write_json(root / 'source_manifest.json', manifest)
    return manifest

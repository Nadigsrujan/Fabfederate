"""Versioned inference bundles. Load only trusted project-produced artifacts."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from fabfederate.contracts import METADATA_COLUMNS

SCHEMA_VERSION = 1


@dataclass
class ModelBundle:
    feature_names: list[str]
    wear_unit: str
    model_kind: str
    model: Any
    imputer: Any
    scaler: Any
    metadata: dict = field(default_factory=dict)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError('prediction requires a named feature DataFrame')
        if frame.columns.duplicated().any():
            raise ValueError('duplicate feature columns')
        missing = set(self.feature_names) - set(frame.columns)
        unknown = set(frame.columns) - set(self.feature_names) - METADATA_COLUMNS
        if missing:
            raise ValueError(f'missing predictor columns: {sorted(missing)}')
        if unknown:
            raise ValueError(f'unknown predictor columns: {sorted(unknown)}')
        if 'wear_unit' in frame and not frame.wear_unit.eq(self.wear_unit).all():
            raise ValueError(f'wear unit must be {self.wear_unit}')
        if frame.attrs.get('schema_version', SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError('unsupported feature schema version')
        try:
            raw = frame[self.feature_names].to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError('predictor features must be numeric') from exc
        if np.isinf(raw).any():
            raise ValueError('predictor features must be finite or missing (NaN)')
        if len(raw) == 0:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return self.scaler.transform(self.imputer.transform(raw))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = self.transform(frame)
        if len(features) == 0:
            return np.array([], dtype=np.float64)
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(torch.tensor(features, dtype=torch.float32)).numpy().ravel()
        else:
            prediction = np.asarray(self.model.predict(features)).ravel()
        if not np.isfinite(prediction).all():
            raise ValueError('model produced nonfinite predictions')
        return prediction.astype(np.float64)

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        """SHAP callable: raw features in the bundle's documented feature order."""
        values = np.asarray(features)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError('feature array does not match bundle schema')
        return self.predict(pd.DataFrame(values, columns=self.feature_names))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = path / 'bundle.joblib'
        joblib.dump(self, payload)
        manifest = dict(schema_version=SCHEMA_VERSION,
                        sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                        feature_names=self.feature_names, wear_unit=self.wear_unit,
                        model_kind=self.model_kind, metadata=self.metadata)
        (path / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
        (path / 'model_card.md').write_text(
            f'# {self.model_kind}\n\nWear unit: {self.wear_unit}.\n\n'
            'Research prototype; no machine-control or calibrated uncertainty claim. '
            'Load only trusted project artifacts; joblib uses executable Python serialization.\n\n'
            'Checkpoint selection uses validation only; test rows are excluded from training.\n\n'
            '```json\n' + json.dumps(self.metadata, indent=2, default=str) + '\n```\n')


def load_bundle(path: Path) -> ModelBundle:
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported bundle schema version')
    payload = path / 'bundle.joblib'
    if hashlib.sha256(payload.read_bytes()).hexdigest() != manifest.get('sha256'):
        raise ValueError('bundle hash mismatch')
    bundle = joblib.load(payload)
    if not isinstance(bundle, ModelBundle):
        raise ValueError('artifact is not a ModelBundle')
    if (bundle.feature_names != manifest['feature_names'] or
            bundle.wear_unit != manifest['wear_unit'] or
            bundle.model_kind != manifest['model_kind']):
        raise ValueError('bundle schema differs from manifest')
    return bundle

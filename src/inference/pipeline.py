"""File-based prediction and SHAP using one trusted saved model and its transform."""

from pathlib import Path

import pandas as pd

from fabfederate.explainability.shap_explainer import explain_prediction, save_explanation
from fabfederate.models.bundle import load_bundle


def predict_file(model_path: Path, input_path: Path, output_path: Path) -> Path:
    bundle = load_bundle(model_path)
    frame = pd.read_csv(input_path)
    prediction = bundle.predict(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'predicted_wear': prediction, 'wear_unit': bundle.wear_unit,
                  'model': bundle.model_kind}).to_csv(output, index=False)
    return output


def explain_file(model_path: Path, input_path: Path, background_path: Path,
                 output_path: Path) -> Path:
    bundle = load_bundle(model_path)
    frame = pd.read_csv(input_path)
    bundle.predict(frame)  # Validate the complete input schema, including optional metadata.
    background = pd.read_parquet(background_path)
    if 'split' not in background or not background.split.eq('train').any():
        raise ValueError('Background must include a documented nonempty train split')
    explanation = explain_prediction(
        bundle, background.loc[background.split.eq('train'), bundle.feature_names],
        frame[bundle.feature_names])
    return save_explanation(explanation, Path(output_path))

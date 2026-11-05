import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


def create_trial_directory(artifacts_dir, model_name):
    """Create a unique directory for an experiment."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    trial_dir = (
        Path(artifacts_dir)
        / model_name
        / f"trial_{timestamp}"
    )

    trial_dir.mkdir(parents=True, exist_ok=False)

    return trial_dir


def save_model(model, trial_dir):
    joblib.dump(model, trial_dir / "model.joblib")


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def save_predictions(predictions, trial_dir):
    """Save actual values, predictions, and residuals."""

    prediction_dir = trial_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for split_name, result in predictions.items():
        df = pd.DataFrame({
            "actual": result["actual"],
            "predicted": result["predicted"],
            "residual": result["actual"] - result["predicted"],
        })

        df.to_parquet(
            prediction_dir / f"{split_name}_predictions.parquet",
            index=False,
        )
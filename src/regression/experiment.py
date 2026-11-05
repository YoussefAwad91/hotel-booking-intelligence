import pickle
import platform
from datetime import datetime

import numpy as np
import sklearn

from src.regression import config, data, evaluate, save, visualization
from src.regression.linear_regression import model as linear_regression
from src.preprocessing.preprocess import HotelPreprocessor


def load_preprocessor():
    # with open(config.PREPROCESSOR_PATH, "rb") as file:
    #     return pickle.load(file)
    return HotelPreprocessor.load(config.PREPROCESSOR_PATH)


def run_experiment():
    # Load data and preprocessing object
    datasets = data.load_data()
    preprocessor = load_preprocessor()

    train_data = datasets["train"]
    val_data = datasets["validation"]

    # Train model
    model = linear_regression.train(
        train_data["X"],
        train_data["y"],
        config.MODEL_CONFIG,
    )

    # Generate train and validation predictions
    predictions = {}

    for split_name, split_data in [
        ("train", train_data),
        ("validation", val_data),
    ]:
        y_pred = linear_regression.predict(model, split_data["X"])

        predictions[split_name] = {
            "actual": np.asarray(split_data["y"]),
            "predicted": np.asarray(y_pred),
        }

    # Evaluate
    metrics = {
        split_name: evaluate.evaluate(
            result["actual"],
            result["predicted"],
            preprocessor,
        )
        for split_name, result in predictions.items()
    }

    # Create trial directory
    trial_dir = save.create_trial_directory(
        config.ARTIFACTS_DIR,
        config.MODEL_NAME,
    )

    # Save model and predictions
    save.save_model(model, trial_dir)
    save.save_predictions(predictions, trial_dir)

    # Save metrics
    save.save_json(metrics, trial_dir / "metrics.json")

    # Save experiment metadata
    metadata = {
        "model": config.MODEL_NAME,
        "target": config.TARGET,
        "seed": config.SEED,
        "features": list(train_data["X"].columns),
        "excluded_features": config.DROP_COLUMNS,
        "train_rows": len(train_data["X"]),
        "validation_rows": len(val_data["X"]),
        "model_config": config.MODEL_CONFIG,
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }

    save.save_json(metadata, trial_dir / "metadata.json")

    # Generate visualizations
    visualization.generate_visualizations(
        model,
        predictions,
        metrics,
        train_data["X"].columns,
        trial_dir / "visualizations",
    )

    print(f"Experiment completed: {trial_dir}")
    print("\nMetrics:")
    print(metrics)


if __name__ == "__main__":
    run_experiment()
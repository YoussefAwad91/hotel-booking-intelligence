import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_transform_adr(values, preprocessor):
    """Convert MinMax-scaled ADR values back to their original scale."""

    lo = preprocessor.lo_["ADR"]
    hi = preprocessor.hi_["ADR"]

    return np.asarray(values) * (hi - lo) + lo


def calculate_metrics(y_true, y_pred):
    """Calculate standard regression metrics."""

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate(y_true, y_pred, preprocessor):
    """Calculate metrics in scaled and original ADR units."""

    scaled_metrics = calculate_metrics(y_true, y_pred)

    actual_original = inverse_transform_adr(y_true, preprocessor)
    predicted_original = inverse_transform_adr(y_pred, preprocessor)

    original_metrics = calculate_metrics(
        actual_original,
        predicted_original,
    )

    return {
        "scaled": scaled_metrics,
        "original": original_metrics,
    }
import pandas as pd

from src.regression import config


def load_dataset(path):
    """Load a processed Parquet dataset."""
    return pd.read_parquet(path)


def load_splits():
    """Load train, validation, and test datasets."""

    train = load_dataset(config.TRAIN_PATH)
    val = load_dataset(config.VAL_PATH)
    test = load_dataset(config.TEST_PATH)

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


def validate_splits(datasets):
    """Validate target availability and consistent feature schemas."""

    reference_columns = list(datasets["train"].columns)

    for split_name, df in datasets.items():
        if config.TARGET not in df.columns:
            raise ValueError(
                f"Target '{config.TARGET}' missing from {split_name} dataset."
            )

        if list(df.columns) != reference_columns:
            raise ValueError(
                f"Column mismatch between training and {split_name} datasets."
            )

        if df.isna().any().any():
            raise ValueError(f"Missing values found in {split_name} dataset.")

    missing_drop_columns = [
        col for col in config.DROP_COLUMNS
        if col not in reference_columns
    ]

    if missing_drop_columns:
        raise ValueError(
            f"Expected columns to drop were not found: {missing_drop_columns}"
        )


def prepare_splits(datasets):
    """Separate predictors and target while retaining original row indices."""

    excluded = set(config.DROP_COLUMNS + [config.TARGET])

    feature_columns = [
        col for col in datasets["train"].columns
        if col not in excluded
    ]

    prepared = {}

    for split_name, df in datasets.items():
        prepared[split_name] = {
            "X": df[feature_columns].copy(),
            "y": df[config.TARGET].copy(),
        }

    return prepared


def load_data():
    """Complete data-loading workflow."""

    datasets = load_splits()
    validate_splits(datasets)

    return prepare_splits(datasets)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def actual_vs_predicted(y_true, y_pred, split_name, output_dir):
    """Plot actual versus predicted values."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    lower = min(
        np.percentile(y_true, 1),
        np.percentile(y_pred, 1),
    )

    upper = max(
        np.percentile(y_true, 99),
        np.percentile(y_pred, 99),
    )

    padding = (upper - lower) * 0.05

    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.35, s=12)

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color="red",
        linewidth=1.5,
        label="Ideal prediction",
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Actual ADR")
    ax.set_ylabel("Predicted ADR")
    ax.set_title(f"{split_name.title()}: Actual vs Predicted")

    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{split_name}_actual_vs_predicted.png",
        dpi=300,
    )
    plt.close(fig)


def residuals_vs_predicted(y_true, y_pred, split_name, output_dir):
    """Plot residuals against predicted values."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_pred, residuals, alpha=0.35, s=12)

    ax.axhline(
        0,
        linestyle="--",
        color="red",
        linewidth=1.5,
        label="Zero residual",
    )

    lower = np.percentile(residuals, 1)
    upper = np.percentile(residuals, 99)

    padding = (upper - lower) * 0.1
    limit = max(abs(lower - padding), abs(upper + padding), 1e-6)

    ax.set_ylim(-limit, limit)
    ax.margins(x=0.05)

    ax.set_xlabel("Predicted ADR")
    ax.set_ylabel("Residual")
    ax.set_title(f"{split_name.title()}: Residuals vs Predicted")

    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{split_name}_residuals_vs_predicted.png",
        dpi=300,
    )
    plt.close(fig)


def residual_distribution(y_true, y_pred, split_name, output_dir):
    """Plot the distribution of residuals."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    residuals = y_true - y_pred

    lower = np.percentile(residuals, 1)
    upper = np.percentile(residuals, 99)

    padding = (upper - lower) * 0.1

    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        residuals,
        bins=40,
        range=(lower, upper),
        edgecolor="black",
        alpha=0.75,
    )

    ax.axvline(
        0,
        linestyle="--",
        color="red",
        linewidth=1.5,
        label="Zero residual",
    )

    ax.set_xlim(lower, upper)

    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{split_name.title()}: Residual Distribution")

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{split_name}_residual_distribution.png",
        dpi=300,
    )
    plt.close(fig)


def metrics_comparison(metrics, output_dir):
    """Compare training and validation metrics."""

    metric_names = ["MAE", "RMSE", "R2"]
    splits = ["train", "validation"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric in zip(axes, metric_names):

        values = np.array([
            metrics[split]["original"][metric]
            for split in splits
        ])

        ax.bar(splits, values)

        minimum = np.min(values)
        maximum = np.max(values)

        difference = maximum - minimum

        if difference > 0:
            padding = difference * 0.5
        else:
            padding = max(abs(maximum) * 0.1, 0.1)

        ax.set_ylim(
            minimum - padding,
            maximum + padding,
        )

        ax.set_title(metric)
        ax.set_ylabel("Score")

        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle("Training vs Validation Performance")

    fig.tight_layout()
    fig.savefig(output_dir / "metrics_comparison.png", dpi=300)
    plt.close(fig)


def coefficient_plot(model, feature_names, output_dir):
    """Plot linear regression coefficients."""

    coefficients = np.asarray(model.coef_).ravel()

    order = np.argsort(np.abs(coefficients))

    max_abs = np.max(np.abs(coefficients))
    limit = max(max_abs * 1.1, 1e-6)

    fig, ax = plt.subplots(
        figsize=(10, max(6, len(feature_names) * 0.22))
    )

    ax.barh(
        np.array(feature_names)[order],
        coefficients[order],
    )

    ax.axvline(
        0,
        linestyle="--",
        color="black",
        linewidth=1,
    )

    ax.set_xlim(-limit, limit)

    ax.set_xlabel("Coefficient")
    ax.set_title("Linear Regression Coefficients")

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_dir / "coefficients.png", dpi=300)
    plt.close(fig)


def generate_visualizations(
    model,
    predictions,
    metrics,
    feature_names,
    output_dir,
):
    """Generate all experiment visualizations."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, result in predictions.items():

        actual_vs_predicted(
            result["actual"],
            result["predicted"],
            split_name,
            output_dir,
        )

        residuals_vs_predicted(
            result["actual"],
            result["predicted"],
            split_name,
            output_dir,
        )

        residual_distribution(
            result["actual"],
            result["predicted"],
            split_name,
            output_dir,
        )

    metrics_comparison(metrics, output_dir)

    coefficient_plot(
        model,
        feature_names,
        output_dir,
    )
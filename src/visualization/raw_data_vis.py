
import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/H2.csv"
OUTPUT_DIR = "visualization/eda_output"

ADR_COL = "ADR"
CANCEL_COL = "IsCanceled"

TOP_N_CATEGORIES = 15
MAX_SCATTER_POINTS = 5000
N_COLS = 3

sns.set_theme(style="whitegrid", palette="deep")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for folder in ["distributions", "vs_adr", "vs_cancellation"]:
    os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)


# ============================================================
# 2. LOAD AND CLASSIFY DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

# Do not automatically remove duplicates
df = df.replace([np.inf, -np.inf], np.nan)

# Explicit column classification based on your dataset
categorical_cols = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "AssignedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
    "ReservationStatus",
    "ReservationStatusDate"
]

numerical_cols = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests"
]

# Convert categorical columns to strings while preserving missing values
for col in categorical_cols:
    df[col] = df[col].astype("string")

# Check columns
assert set(categorical_cols + numerical_cols) == set(df.columns), (
    "Column classification does not match dataset columns. "
    f"Unclassified: {set(df.columns) - set(categorical_cols + numerical_cols)}"
)

print("Categorical variables:", categorical_cols)
print("Numerical variables:", numerical_cols)

# Save classification summary
classification = pd.DataFrame({
    "Variable": df.columns,
    "Data Type": [
        "Categorical" if c in categorical_cols else "Numerical"
        for c in df.columns
    ],
    "Unique Values": [df[c].nunique() for c in df.columns],
    "Missing Values": [df[c].isna().sum() for c in df.columns]
})

classification.to_csv(
    os.path.join(OUTPUT_DIR, "variable_classification.csv"),
    index=False
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    return "".join(
        c if c.isalnum() or c in "_-" else "_"
        for c in str(name)
    )


def top_categories(series, n=TOP_N_CATEGORIES):
    """
    Keep the most frequent categories.
    Combine the remaining categories into 'Other'.
    """

    counts = series.value_counts(dropna=True)

    if len(counts) <= n:
        return series

    top = counts.nlargest(n).index

    return series.where(series.isin(top), other="Other")


def get_mode(series):
    modes = series.mode(dropna=True)

    if len(modes) == 0:
        return "N/A"

    return str(modes.iloc[0])


# ============================================================
# 4. INDIVIDUAL VARIABLE DISTRIBUTIONS
# ============================================================

print("\nGenerating individual distributions...")

# ------------------------------------------------------------
# 4.1 Categorical distributions
# ------------------------------------------------------------

for col in categorical_cols:

    fig, ax = plt.subplots(figsize=(11, 6))

    counts = df[col].value_counts(dropna=False)

    # Limit displayed categories for readability
    if len(counts) > TOP_N_CATEGORIES:
        display_counts = counts.nlargest(TOP_N_CATEGORIES).copy()
        display_counts["Other"] = (
            counts.iloc[TOP_N_CATEGORIES:].sum()
        )
    else:
        display_counts = counts

    display_counts.plot(
        kind="bar",
        ax=ax,
        color="steelblue",
        edgecolor="black"
    )

    mode = get_mode(df[col])

    ax.set_title(
        f"{col}\nMode: {mode} | Unique categories: {df[col].nunique()}",
        fontsize=12
    )

    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "distributions",
            f"{safe_filename(col)}_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ------------------------------------------------------------
# 4.2 Numerical distributions
# ------------------------------------------------------------

for col in numerical_cols:

    data = df[col].dropna()

    if len(data) == 0:
        continue

    mean = data.mean()
    median = data.median()
    std = data.std()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data,
        kde=True,
        bins=40,
        ax=ax,
        color="steelblue",
        edgecolor="black"
    )

    # Mean
    ax.axvline(
        mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean:.2f}"
    )

    # Median
    ax.axvline(
        median,
        color="green",
        linestyle="-.",
        linewidth=2,
        label=f"Median = {median:.2f}"
    )

    ax.set_title(
        f"{col}\nStandard Deviation = {std:.2f}",
        fontsize=12
    )

    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "distributions",
            f"{safe_filename(col)}_distribution.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# ============================================================
# 5. RELATIONSHIP WITH ADR
# ============================================================

print("\nGenerating ADR relationship plots...")

adr_features = [
    c for c in df.columns
    if c != ADR_COL
]

# Each variable gets its own subplot.
# All subplots are saved together in one figure.

n_plots = len(adr_features)
n_rows = math.ceil(n_plots / N_COLS)

fig, axes = plt.subplots(
    n_rows,
    N_COLS,
    figsize=(7 * N_COLS, 5 * n_rows)
)

axes = np.array(axes).flatten()

for i, col in enumerate(adr_features):

    ax = axes[i]

    temp = df[[col, ADR_COL]].dropna()

    if temp.empty:
        ax.set_visible(False)
        continue

    if col in categorical_cols:

        # Boxplot: ADR distribution for each category
        temp[col] = top_categories(temp[col])

        order = temp.groupby(col, observed=True)[ADR_COL].median()
        order = order.sort_values().index

        sns.boxplot(
            data=temp,
            x=col,
            y=ADR_COL,
            order=order,
            ax=ax,
            showfliers=False
        )

        ax.set_title(f"ADR vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("ADR")
        ax.tick_params(axis="x", rotation=45)

    elif col in numerical_cols:

        # Scatterplot + binned median trend
        if len(temp) > MAX_SCATTER_POINTS:
            temp = temp.sample(
                MAX_SCATTER_POINTS,
                random_state=42
            )

        sns.scatterplot(
            data=temp,
            x=col,
            y=ADR_COL,
            alpha=0.25,
            s=15,
            ax=ax
        )

        # Calculate median ADR across 20 bins
        try:
            temp["bin"] = pd.qcut(
                temp[col],
                q=20,
                duplicates="drop"
            )

            trend = temp.groupby(
                "bin",
                observed=True
            ).agg(
                x=(col, "median"),
                y=(ADR_COL, "median")
            ).dropna()

            ax.plot(
                trend["x"],
                trend["y"],
                color="red",
                linewidth=2,
                marker="o",
                label="Binned median ADR"
            )

            ax.legend()

        except (ValueError, TypeError):
            pass

        ax.set_title(f"ADR vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("ADR")

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Relationship of All Variables with ADR",
    fontsize=20,
    y=1.002
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "vs_adr", "all_variables_vs_adr.png"),
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. RELATIONSHIP WITH IS_CANCELED
# ============================================================

print("\nGenerating cancellation relationship plots...")

cancel_features = [
    c for c in df.columns
    if c != CANCEL_COL
]

n_plots = len(cancel_features)
n_rows = math.ceil(n_plots / N_COLS)

fig, axes = plt.subplots(
    n_rows,
    N_COLS,
    figsize=(7 * N_COLS, 5 * n_rows)
)

axes = np.array(axes).flatten()

for i, col in enumerate(cancel_features):

    ax = axes[i]

    temp = df[[col, CANCEL_COL]].dropna()

    if temp.empty:
        ax.set_visible(False)
        continue

    if col in categorical_cols:

        # Cancellation rate per category
        temp[col] = top_categories(temp[col])

        cancellation_rate = (
            temp.groupby(col, observed=True)[CANCEL_COL]
            .mean() * 100
        )

        cancellation_rate.plot(
            kind="bar",
            ax=ax,
            color="indianred",
            edgecolor="black"
        )

        ax.set_title(f"Cancellation Rate vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Cancellation Rate (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=45)

    elif col in numerical_cols:

        # Compare numerical distributions between
        # canceled and non-canceled bookings.

        sns.boxplot(
            data=temp,
            x=CANCEL_COL,
            y=col,
            hue=CANCEL_COL,
            legend=False,
            showfliers=False,
            ax=ax
        )

        ax.set_title(f"{col} vs Cancellation Status")
        ax.set_xlabel("Is Canceled (0 = No, 1 = Yes)")
        ax.set_ylabel(col)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not Canceled", "Canceled"])


# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Relationship of All Variables with Cancellation",
    fontsize=20,
    y=1.002
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "vs_cancellation",
        "all_variables_vs_cancellation.png"
    ),
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. SUMMARY STATISTICS
# ============================================================

# Numerical statistics
numerical_summary = df[numerical_cols].describe().T

numerical_summary["mode"] = [
    get_mode(df[col]) for col in numerical_cols
]

numerical_summary.to_csv(
    os.path.join(OUTPUT_DIR, "numerical_summary.csv")
)

# Categorical statistics
categorical_summary = pd.DataFrame({
    "Mode": [get_mode(df[col]) for col in categorical_cols],
    "Mode Frequency": [
        df[col].value_counts().iloc[0]
        if df[col].notna().any() else 0
        for col in categorical_cols
    ],
    "Unique Values": [
        df[col].nunique() for col in categorical_cols
    ],
    "Missing Values": [
        df[col].isna().sum() for col in categorical_cols
    ]
}, index=categorical_cols)

categorical_summary.to_csv(
    os.path.join(OUTPUT_DIR, "categorical_summary.csv")
)

print("\n===================================")
print("VISUALIZATION COMPLETE")
print("===================================")
print(f"All plots and summaries saved in: {OUTPUT_DIR}")


import pandas as pd
import numpy as np


file_path = "data/raw/H2.csv" 

data = pd.read_csv(file_path)

print("HOTEL BOOKING DEMAND DATASET - EXPLORATORY ANALYSIS")
print("-" * 80)
print(f"\nDataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")

missing_placeholders = [
    "NA", "N/A", "NULL", "NONE", "NAN",
    "UNKNOWN", "?", "", "MISSING"
]


report = []

for col in data.columns:

    series = data[col]
    actual_missing = series.isna().sum()

    if series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):
        placeholders = series.astype("string").str.strip().str.upper().isin(missing_placeholders).sum()
    else:
        placeholders = 0

    total_missing = actual_missing + placeholders
    valid_count = len(series) - total_missing

    print("\n" + "-" * 80)
    print(f"COLUMN: {col}")
    print("-" * 80)

    print(f"Data Type: {series.dtype}")
    print(f"Total Values: {len(series)}")
    print(f"Actual NaN Values: {actual_missing}")
    print(f"Missing Placeholders: {placeholders}")
    print(f"Total Missing: {total_missing}")
    print(f"Missing Percentage: {total_missing / len(series) * 100:.2f}%")

    # ======================================
    # CATEGORICAL VARIABLES
    # ======================================

    if series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):

        print("\nVariable Type: CATEGORICAL")
        print(f"Unique Values: {series.nunique(dropna=True)}")

        print("\nAll Values and Frequencies:")

        value_counts = series.value_counts(dropna=False)

        print(value_counts.to_string())

        report.append({
            "Column": col,
            "Type": "Categorical",
            "Unique Values": series.nunique(dropna=True),
            "Missing Values": total_missing,
            "Missing %": round(total_missing / len(data) * 100, 2),
            "Details": value_counts.to_dict()
        })

    # ======================================
    # NUMERICAL VARIABLES
    # ======================================

    elif pd.api.types.is_numeric_dtype(series):

        print("\nVariable Type: NUMERICAL")
        print(f"Unique Values: {series.nunique(dropna=True)}")
        print(f"Valid Values: {valid_count}")

        print("\nDescriptive Statistics:")
        print(series.describe().to_string())

        print("\nAdditional Statistics:")
        print(f"Mean: {series.mean()}")
        print(f"Median: {series.median()}")
        print(f"Standard Deviation: {series.std()}")
        print(f"Minimum: {series.min()}")
        print(f"Maximum: {series.max()}")

        report.append({
            "Column": col,
            "Type": "Numerical",
            "Unique Values": series.nunique(dropna=True),
            "Missing Values": total_missing,
            "Missing %": round(total_missing / len(data) * 100, 2),
            "Details": series.describe().to_dict()
        })

    # ======================================
    # OTHER DATA TYPES
    # ======================================

    else:

        print("\nVariable Type: OTHER")
        print("\nValue Counts:")
        print(series.value_counts(dropna=False).to_string())

        report.append({
            "Column": col,
            "Type": str(series.dtype),
            "Unique Values": series.nunique(dropna=True),
            "Missing Values": total_missing,
            "Missing %": round(total_missing / len(data) * 100, 2),
            "Details": series.value_counts(dropna=False).to_dict()
        })


# ==========================================
# 4. OVERALL DATASET SUMMARY
# ==========================================

print("\n" + "=" * 80)
print("OVERALL DATASET SUMMARY")
print("=" * 80)

summary = pd.DataFrame(report)

print(summary[
    ["Column", "Type", "Unique Values", "Missing Values", "Missing %"]
].to_string(index=False))


# ==========================================
# 5. EXPORT REPORTS
# ==========================================

# General column summary
summary.drop(columns=["Details"]).to_csv("src/preprocessing/eda.csv",
    index=False
)

# Detailed report
with open("src/preprocessing/eda_detailed_report.txt", "w", encoding="utf-8") as f:

    for item in report:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"COLUMN: {item['Column']}\n")
        f.write(f"TYPE: {item['Type']}\n")
        f.write(f"UNIQUE VALUES: {item['Unique Values']}\n")
        f.write(f"MISSING VALUES: {item['Missing Values']}\n")
        f.write(f"MISSING %: {item['Missing %']}%\n")
        f.write(f"\nDETAILS:\n")

        for key, value in item["Details"].items():
            f.write(f"{key}: {value}\n")

print("\nAnalysis completed successfully!")
print("Generated files:")
print("- hotel_booking_column_summary.csv")
print("- hotel_booking_detailed_report.txt")

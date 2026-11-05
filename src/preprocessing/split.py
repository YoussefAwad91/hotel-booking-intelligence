
import pandas as pd
import argparse
import os
import numpy as np


def split_csv(input_file, output_dir, split_ratios, seed=None):

    # Validate split ratios
    if not split_ratios:
        raise ValueError("At least one split ratio is required.")

    if any(r <= 0 for r in split_ratios):
        raise ValueError("All split percentages must be positive.")

    if not np.isclose(sum(split_ratios), 100):
        raise ValueError(
            f"Split percentages must add up to 100%. "
            f"Current total: {sum(split_ratios)}%"
        )

    # Read CSV
    df = pd.read_csv(input_file)

    # Randomize records
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Calculate split sizes
    total_records = len(df)
    split_sizes = [
        int(total_records * ratio / 100)
        for ratio in split_ratios
    ]

    # Assign remaining records to the last split
    split_sizes[-1] = total_records - sum(split_sizes[:-1])

    # Generate output files
    start = 0

    for i, size in enumerate(split_sizes):
        end = start + size

        split_df = df.iloc[start:end]

        output_file = os.path.join(
            output_dir, f"split_{i + 1}_{split_ratios[i]}.csv"
        )

        split_df.to_csv(output_file, index=False)

        print(
            f"Created {output_file}: "
            f"{size} records ({size / total_records * 100:.2f}%)"
        )

        start = end

    print(f"\nTotal records: {total_records}")
    print(f"Successfully created {len(split_ratios)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly shuffle and split a CSV file."
    )

    parser.add_argument(
        "--input",
        help="Input CSV file path"
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        type=float,
        required=True,
        help="Split percentages (must add up to 100)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    split_csv(
        args.input,
        args.output_dir,
        args.splits,
        args.seed
    )

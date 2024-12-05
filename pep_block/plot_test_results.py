"""
Module for analyzing peptide test results and generating plots.

This module loads peptide test data from a Parquet file, \
    calculates the number of tests passed by each peptide, \
    and generates plots. It includes:
- A histogram of tests passed by peptides.
- A panel of bar plots showing the pass/fail percentages for each test \
    (3 plots per row).

Usage:
    python script.py -i <input_file> -o <output_directory>

Arguments:
    - `-i` or `--input`: Path to the input Parquet file (required).
    - `-o` or `--output`: Directory to save the output plots (required).
"""
import argparse
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(parquet_file):
    """Load the peptide test results from a Parquet file into a DataFrame."""
    return pd.read_parquet(parquet_file)


def calculate_passed_tests(df):
    """
    Calculate the sum of tests passed for each peptide.

    Args:
        df (pd.DataFrame): DataFrame containing the test results.

    Returns:
        pd.Series: A Series with the sum of tests passed for each peptide.
    """
    test_columns = [col for col in df.columns if col.startswith("Test")]
    if not test_columns:
        raise ValueError("No columns starting with 'Test' found in the\
             DataFrame.")
    return df[test_columns].sum(axis=1)


def plot_passed_tests_distribution(passed_tests, output_dir):
    """
    Plot the distribution of the sum of tests passed.

    Args:
        passed_tests (pd.Series): Series containing the sum of tests passed\
            for each peptide.
        output_dir (str): Directory to save the histogram plot.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(passed_tests, bins=passed_tests.nunique(), kde=False,
                 color="teal")
    plt.title("Distribution of Tests Passed by Peptides")
    plt.xlabel("Number of Tests Passed")
    plt.ylabel("Number of Peptides")
    plt.tight_layout()

    output_path = f"{output_dir}/passed_tests_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram saved to {output_path}")


def plot_percentage_per_test_panel(df, output_file):
    """
    Plot the percentage of peptides that passed or failed each test.

    Args:
        df (pd.DataFrame): DataFrame containing the test results.
        output_file (str): Path to save the plot.
    """
    sns.set(style="whitegrid")

    # Identify test columns
    test_columns = [col for col in df.columns if (col.startswith("Test") or col.startswith("AllTestsPassed"))]
    if not test_columns:
        raise ValueError("No columns starting with 'Test' found in the\
            DataFrame.")

    # Calculate number of rows needed
    num_tests = len(test_columns)
    num_rows = math.ceil(num_tests / 3)

    # Set up the panel (grid of subplots)
    _, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows),
                           sharey=True)
    axes = axes.flatten()  # Flatten to iterate easily

    for i, test in enumerate(test_columns):
        passed_percentage = (df[test].sum() / len(df)) * 100
        failed_percentage = 100 - passed_percentage

        # Data for plotting
        plot_data = pd.DataFrame({
            "Result": ["Passed", "Failed"],
            "Percentage": [passed_percentage, failed_percentage]
        })

        sns.barplot(x="Result", y="Percentage", data=plot_data, ax=axes[i],
                    palette="viridis")
        axes[i].set_title(f"{test}: Pass/Fail Percentages", fontsize=14)
        axes[i].set_ylim(0, 100)
        axes[i].set_xlabel("")
        if i % 3 == 0:
            axes[i].set_ylabel("Percentage")
        else:
            axes[i].set_ylabel("")

    # Hide unused subplots
    for j in range(len(test_columns), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Panel plot saved to {output_file}")


def main():
    """Run main function to analyze test results and generate plots."""
    parser = argparse.ArgumentParser(description="Analyze\
         test results and generate \
            distribution and panel plots.")
    parser.add_argument('-i', '--input', type=str, required=True, help='Path\
         to the input Parquet file containing test results.')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Directory to save the output plots.')
    args = parser.parse_args()

    # Load data
    df = load_data(args.input)

    # Calculate the sum of tests passed
    passed_tests = calculate_passed_tests(df)

    # Generate the histogram of the sum of tests passed
    plot_passed_tests_distribution(passed_tests, args.output)

    # Generate the panel plot for pass/fail percentages
    output_file = f"{args.output}/percentage_per_test_panel.png"
    plot_percentage_per_test_panel(df, output_file)


if __name__ == "__main__":
    main()

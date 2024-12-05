"""
Peptide Filtering Module.

This module processes peptide data in Parquet files by:
- Calculating the Hamming distance from a reference peptide.
- Adding a column to indicate if all tests were passed ("AllTestsPassed").
- Filtering peptides based on Hamming distance and test success.

Functions:
- hamming_distance: Computes Hamming distance between two peptide sequences.
- add_columns_to_parquet: Adds "AllTestsPassed" and "HammingDistance" columns.
- filter_peptides: Filters peptides by Hamming distance and test success.
- main: Command-line interface for processing and filtering peptides.

Usage:
    python peptide_filtering.py -i input.parquet -r SFLLRN
    -f intermediate.parquet -o output.parquet

Dependencies:
- Python 3.7+, Pandas, PyArrow, argparse
"""
import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


def hamming_distance(peptide1, peptide2):
    """
    Calculate the Hamming distance between two peptides.

    The Hamming distance is the number of positions at which the corresponding
    elements of two sequences differ. It assumes that both sequences have the
    same length.

    Args:
        peptide1 (str): The first peptide sequence.
        peptide2 (str): The second peptide sequence.

    Returns:
        int: The Hamming distance between the two peptide sequences.
    """
    return sum(el1 != el2 for el1, el2 in zip(str(peptide1), str(peptide2)))


def add_columns_to_parquet(parquet_file, reference_peptide, inter_file):
    """
    Add 'AllTestsPassed' and 'HammingDistance' columns to a Parquet file.

    The function reads a Parquet file containing peptides and their test
    results, adds a column indicating whether all tests were passed, and
    another column calculating the Hamming distance from a reference peptide.
    The modified data is saved to an intermediate Parquet file.

    Args:
        parquet_file (str): Path to the input Parquet file.
        reference_peptide (str): The reference peptide used to calculate
        Hamming distance.
        inter_file (str): Path to the output intermediate Parquet file.

    Returns:
        None
    """
    # Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(parquet_file)

    # Add "AllTestsPassed" column (True if all test columns are True)
    test_columns = [col for col in df.columns if col.startswith('Test')]
    df['AllTestsPassed'] = df[test_columns].all(axis=1)

    # Add "HammingDistance" column (distance from the reference peptide)
    hamming_distances = []
    for peptide in df['Peptide']:
        distance = hamming_distance(peptide, reference_peptide)
        hamming_distances.append(distance)
    df['HammingDistance'] = hamming_distances

    # Define the schema explicitly
    schema = pa.schema([
        ('Peptide', pa.string()),
        ('Test 1', pa.bool_()),
        ('Test 2', pa.bool_()),
        ('Test 3', pa.bool_()),
        ('Test 4', pa.bool_()),
        ('Test 5', pa.bool_()),
        ('Test 6', pa.bool_()),
        ('Test 7', pa.bool_()),
        ('Test 8', pa.bool_()),
        ('AllTestsPassed', pa.bool_()),
        ('HammingDistance', pa.int32())
    ])

    # Convert the Pandas DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(df, schema=schema)

    # Save the modified table to an intermediate Parquet file
    pq.write_table(table, inter_file)
    print(f"Modified table saved to: {inter_file}")


def filter_peptides(inter_file, hamming_threshold, output_file):
    """
    Filter peptides based on Hamming distance and test results.

    The function reads the intermediate Parquet file, filters peptides
    based on a maximum Hamming distance and whether all tests passed,
    and saves the filtered data to an output Parquet file.

    Args:
        inter_file (str): Path to the intermediate Parquet file.
        hamming_threshold (int): Maximum allowed Hamming distance.
        output_file (str): Path to the output Parquet file.

    Returns:
        None
    """
    # Read and filter the Parquet table
    table = pq.read_table(inter_file, filters=[
        ('HammingDistance', '<=', hamming_threshold),
        ("AllTestsPassed", "=", True)])
    pq.write_table(table, output_file)
    print(f"Number of filtered peptides saved: {table.num_rows}")
    print(f"First filtered peptides: {table.to_pandas().head()}")


def main():
    """
    Run main function to add columns and filter peptides in Parquet files.

    This script reads an input Parquet file containing peptide data, adds
    new columns for "AllTestsPassed" and "HammingDistance", and filters
    the peptides based on these criteria. The filtered results are saved
    to an output Parquet file.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Add 'AllTestsPassed' and \
                                     'HammingDistance' to a Parquet file.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path \
                        to the input Parquet file.")
    parser.add_argument('-r', '--reference', type=str, default='SFLLRN',
                        help="Reference peptide for Hamming distance \
                            calculation.")
    parser.add_argument('-f', '--file_inter', type=str, required=True,
                        help="Path to the intermediate Parquet file.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Path to the output Parquet file.")

    args = parser.parse_args()

    # Add columns and filter peptides
    add_columns_to_parquet(args.input, args.reference, args.file_inter)
    filter_peptides(args.file_inter, 2, args.output)


if __name__ == "__main__":
    main()

"""
This module provides functions to process peptide sequences, \
    perform various tests on them, and store the results in a Parquet file. \
    It includes functionality for loading amino acid group definitions \
    and metadata from JSON files, running tests to validate peptide\
     properties, and outputting the results to a Parquet file with metadata.

Functions:
    load_json(file_path): Load a JSON file and return its contents.
    test_hydrophobic_percentage(counter, aas): Test if hydrophobic residues \
        are less than 50% of the peptide.
    test_charged_residues(counter, aas): Test if the peptide contains at \
        least one charged residue.
    test_polar_percentage(counter, aas): Test if polar residues are less \
        than or equal to 75% of the peptide.
    test_n_terminal(sequence, aas): Test if the N-terminal residue is valid.
    test_c_terminal(sequence, aas): Test if the C-terminal residue is valid.
    test_max_2_restricted(counter, aas): Test if the number of restricted \
        residues is at most 2.
    test_no_consecutive(sequence, aas): Test if the peptide contains no \
        invalid consecutive residue pairs.
    test_no_repeated_adjacent(sequence, aas): Test if the peptide contains\
         no repeated adjacent residues in the forbidden set.
    all_tests(sequence, aas): Run all peptide tests and return the results\
         as a dictionary.
    main(): Main function to process peptide sequences, perform tests,\
         and save results to a Parquet file.
"""
import argparse
import json
from collections import Counter
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq


def load_json(file_path):
    """
    Load a JSON file and return its contents.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a Python dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"The file {file_path} was not found.")\
             from exc
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f"Error decoding JSON in file {file_path}:\
                                    {exc.msg}", exc.doc, exc.pos) from exc


def test_hydrophobic_percentage(counter, aas):
    """
    Test if the hydrophobic residues are less than 50% of the peptide.

    Args:
        counter (collections.Counter): Counter of amino acids in the peptide.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if hydrophobic residues < 50%, otherwise False.
    """
    hydrophobic_residues = aas['hydrophobic_residues']
    hydrophobic_count = sum(counter[aa] for aa in hydrophobic_residues)
    total_count = sum(counter.values())
    return (hydrophobic_count / total_count) < 0.5


def test_charged_residues(counter, aas):
    """
    Test if the peptide contains at least one charged residue.

    Args:
        counter (collections.Counter): Counter of amino acids in the peptide.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if at least one charged residue is present, otherwise False.
    """
    charged_residues = aas['charged_residues']
    charged_count = sum(counter[aa] for aa in charged_residues)
    return charged_count > 0


def test_polar_percentage(counter, aas):
    """
    Test if polar residues are less than or equal to 75% of the peptide.

    Args:
        counter (collections.Counter): Counter of amino acids in the peptide.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if polar residues <= 75%, otherwise False.
    """
    polar_residues = aas['polar_residues']
    polar_count = sum(counter[aa] for aa in polar_residues)
    total_count = sum(counter.values())
    return (polar_count / total_count) <= 0.75


def test_n_terminal(sequence, aas):
    """
    Test if the N-terminal residue is valid.

    Args:
        sequence (str): The peptide sequence.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if the N-terminal residue is valid, otherwise False.
    """
    invalid_n_ter = aas['invalid_n_ter']
    return sequence[0] not in invalid_n_ter


def test_c_terminal(sequence, aas):
    """
    Test if the C-terminal residue is valid.

    Args:
        sequence (str): The peptide sequence.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if the C-terminal residue is valid, otherwise False.
    """
    invalid_c_ter = aas['invalid_c_ter']
    return sequence[-1] not in invalid_c_ter


def test_max_2_restricted(counter, aas):
    """
    Test if the number of restricted residues is at most 2.

    Args:
        counter (collections.Counter): Counter of amino acids in the peptide.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if restricted residues <= 2, otherwise False.
    """
    restricted_residues = aas['restricted_residues']
    restricted_count = sum(counter[aa] for aa in restricted_residues)
    return restricted_count <= 2


def test_no_consecutive(sequence, aas):
    """
    Test if the peptide contains no invalid consecutive residue pairs.

    Args:
        sequence (str): The peptide sequence.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if no invalid consecutive pairs are present otherwise False.
    """
    if "GGGG" in sequence:
        return False

    invalid_pairs = aas['invalid_pairs']
    for i in range(len(sequence) - 1):
        pair = sequence[i] + sequence[i + 1]
        if pair in invalid_pairs:
            return False

    return True


def test_no_repeated_adjacent(sequence, aas):
    """
    Test if the peptide contains no repeated adjacent residues in the \
        forbidden set.

    Args:
        sequence (str): The peptide sequence.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        bool: True if no forbidden adjacent residues are repeated \
            otherwise False.
    """
    forbidden_residues = aas['forbidden_residues']
    for i in range(len(sequence) - 1):
        if (sequence[i] == sequence[i + 1] and sequence[i] in
                forbidden_residues):
            return False

    return True


def all_tests(sequence, aas):
    """
    Run all peptide tests and return the results as a dictionary.

    Args:
        sequence (str): The peptide sequence.
        aas (dict): Dictionary containing the amino acid groups.

    Returns:
        dict: Dictionary containing the peptide and test results.
    """
    counter = Counter(sequence)

    return {
        "Peptide": sequence,
        "Test 1": test_hydrophobic_percentage(counter, aas),
        "Test 2": test_charged_residues(counter, aas),
        "Test 3": test_polar_percentage(counter, aas),
        "Test 4": test_n_terminal(sequence, aas),
        "Test 5": test_c_terminal(sequence, aas),
        "Test 6": test_max_2_restricted(counter, aas),
        "Test 7": test_no_consecutive(sequence, aas),
        "Test 8": test_no_repeated_adjacent(sequence, aas),
    }


def main():
    """
    Main function to process peptide sequences, perform tests, \
        and save results to a Parquet file.

    The function uses argparse to parse input arguments for the input \
        Parquet file,
    output file, amino acid configuration file (JSON) and metadata file (JSON).
    """
    parser = argparse.ArgumentParser(description="Process peptide sequences \
        with various tests.")
    parser.add_argument("-i", type=str, required=True, help="Path to the \
        input Parquet file.")
    parser.add_argument("-o", type=str, required=True, help="Path to the \
        output Parquet file.")
    parser.add_argument("-a", type=str, required=True, help="Path to the \
        JSON file defining amino acid groups.")
    parser.add_argument("-m", type=str, required=True, help="Path to the \
        JSON metadata file.")
    args = parser.parse_args()

    # Load amino acid groups and metadata using load_json
    aas = load_json(args.a)
    metadata = load_json(args.m)

    peptides_processed = []

    try:
        reader = pq.ParquetFile(args.i)
        num_row_groups = reader.num_row_groups

        for batch_idx in range(num_row_groups):
            table = reader.read_row_group(batch_idx)
            sequences = table['pep_seq'].to_pylist()

            with multiprocessing.Pool() as pool:
                results = pool.starmap(all_tests, [(seq, aas) for seq
                                                   in sequences])

            peptides_processed.extend(results)

    except (pq.lib.ArrowInvalid, FileNotFoundError) as exc:
        print(f"Error reading Parquet file: {exc}")
        return

    schema = pa.schema([
        ("Peptide", pa.string()),
        ("Test 1", pa.bool_()),
        ("Test 2", pa.bool_()),
        ("Test 3", pa.bool_()),
        ("Test 4", pa.bool_()),
        ("Test 5", pa.bool_()),
        ("Test 6", pa.bool_()),
        ("Test 7", pa.bool_()),
        ("Test 8", pa.bool_())
    ])

    table = pa.Table.from_pylist(peptides_processed, schema=schema)
    table = table.replace_schema_metadata(metadata)

    print(table.to_pandas().head())
    pq.write_table(table, args.o, compression='BROTLI', compression_level=4)

    print(f"{len(peptides_processed)} Peptides processed and written to {args.o}")


if __name__ == "__main__":
    main()

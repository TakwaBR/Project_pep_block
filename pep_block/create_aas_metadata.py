"""
Generate JSON files for amino acid groups and peptide test metadata.

This script creates two JSON files:
1. `aas.json`: Defines amino acid classifications (e.g., hydrophobic, charged).
2. `metadata.json`: Describes peptide test criteria and their purposes.

Functions:
-----------
- `save_to_json(data, file_path)`: Saves a dictionary to a JSON file.
- `create_aas_dict()`: Builds the amino acid classifications dictionary.
- `create_metadata_dict()`: Builds the peptide test metadata dictionary.
- `main()`: Parses command-line arguments and saves JSON files.

Usage:
------
Run with optional arguments to specify output file paths:
    ```
    python script.py --aas_output <aas.json> --metadata_output <metadata.json>
    ```

Dependencies:
-------------
- `argparse`: For argument parsing.
- `json`: For JSON file handling.
"""

import json
import argparse


def save_to_json(data, file_path):
    """
    Save a dictionary to a JSON file.

    Args:
        data (dict): The data to save.
        file_path (str): The path to the JSON file.

    Returns:
        None
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"Data successfully saved to {file_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error saving data to {file_path}: {e}")


def create_aas_dict():
    """
    Create a dictionary of amino acid groups.

    Returns:
        dict: Dictionary of amino acid groups.
    """
    return {
        "hydrophobic_residues": ['L', 'V', 'I', 'F'],
        "charged_residues": ['D', 'E', 'K', 'R', 'H'],
        "polar_residues": ['D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y'],
        "invalid_n_ter": ['Q', 'N', 'E', 'L', 'F', 'D', 'K', 'R'],
        "invalid_c_ter": ['D'],
        "restricted_residues": ['E', 'I', 'L', 'F', 'T', 'Y', 'V', 'P', 'D'],
        "invalid_pairs": ["SS", "DG", "DP", "DS"],
        "forbidden_residues": ['V', 'I', 'Y', 'F', 'L', 'Q', 'T']
    }


def create_metadata_dict():
    """
    Create a dictionary of metadata for peptide tests.

    Returns:
        dict: Dictionary of test metadata.
    """
    return {
        "Test 1": "Test on the percentage of hydrophobic amino acids \
            (L, V, I, F) in the peptide. Passes if < 50%.",
        "Test 2": "Test on charged residues (D, E, K, R, H). \
            Passes if the peptide contains charged residues.",
        "Test 3": "Test on the percentage of polar amino acids \
            (D, E, H, K, N, Q, R, S, T, Y). Passes if <= 75%.",
        "Test 4": "Test on the N-terminal residue. \
            Passes if the first amino acid is not in the list: \
                Q, N, E, L, F, D, K, R.",
        "Test 5": "Test on the C-terminal residue. \
            Passes if the last amino acid is not D.",
        "Test 6": "Test on specific residues: E, I, L, F, T, Y, V. \
            Passes if the peptide contains <= 2.",
        "Test 7": "Test on invalid consecutive residue pairs. \
            Passes if no pairs like 'GGGG', 'SS', 'DG', 'DP', 'DS' \
                are present.",
        "Test 8": "Test on repeated adjacent residues \
            (V, I, Y, F, W, L, Q, T). Passes if none of these residues \
                are repeated consecutively."
    }


def main():
    """Run the peptide processing and save the results."""
    parser = argparse.ArgumentParser(description="Generate JSON files \
                                     for amino acid groups and metadata.")
    parser.add_argument(
        "-a",
        type=str,
        default="aas.json",
        help="Path to save the amino acid groups JSON file. \
            Default is 'aas.json'."
    )
    parser.add_argument(
        "-m",
        type=str,
        default="metadata.json",
        help="Path to save the metadata JSON file. Default is 'metadata.json'."
    )
    args = parser.parse_args()

    # Create the dictionaries
    aas = create_aas_dict()
    metadata = create_metadata_dict()

    # Save to JSON files
    save_to_json(aas, args.a)
    save_to_json(metadata, args.m)


if __name__ == "__main__":
    main()

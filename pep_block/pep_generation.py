"""
Peptide Sequence Generator and Writer to Parquet.

This module generates peptide sequences of a specified length from a given set
of amino acids and writes them to a Parquet file in batches.\
    It supports appending to an existing file or creating a new one.

Functions:
    write_to_parquet(buffer, pq_writer, schema, writer=None):
        Writes data from a buffer to a Parquet file.
    main():
        Generates peptide sequences and saves them to a Parquet file using \
            command-line args.

Command-line Arguments:
    -o, --output: Output Parquet file (required)
    -l, --length: Length of the peptide sequences (required)
    -b, --buffer_size: Buffer size for batch writing (optional, \
        default=1,000,000)
"""
import argparse
import itertools
import pyarrow as pa
import pyarrow.parquet as pq


def write_to_parquet(buffer, pq_writer, schema, writer=None):
    """
    Write data from a buffer to a Parquet file.

    If a writer is provided, it appends the data; otherwise,\
    it creates a new file.

    Args:
        buffer (list): The buffer containing data to write.
        pq_writer (str): Path to the Parquet file.
        schema (pa.Schema): The schema for the Parquet file.
        writer (pq.ParquetWriter, optional): An existing Parquet writer.\
            Defaults to None.

    Returns:
        pq.ParquetWriter: The Parquet writer used for writing the data.
    """
    # Convert the buffer into a PyArrow table
    table = pa.Table.from_pylist(buffer, schema=schema)

    if writer is None:  # If the writer does not exist yet, create it
        writer = pq.ParquetWriter(pq_writer, schema, compression='BROTLI',
                                  compression_level=4)

    # Write the table to the Parquet file
    writer.write_table(table)

    return writer


def main():
    """
    Run Main function to generate peptide sequences and write them to a \
        Parquet file.

    Uses argparse to handle input parameters such as output path, \
        sequence length, and buffer size.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate peptide sequences \
                                     and save them to a Parquet file.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Path to the output Parquet file.")
    parser.add_argument("-l", "--length", type=int, default=6,
                        help="Length of the peptide sequences to generate.")
    parser.add_argument("-b", "--buffer_size", type=int, default=1419857,
                        help="Number of sequences to store in memory before \
                            writing to the file.")
    args = parser.parse_args()

    # Define the amino acid sequence and schema
    sequence_aa = 'ARNDEQGHILKFPSTYV'  # Amino acid sequence
    schema = pa.schema([
        ('pep_seq', pa.string())
        # Schema defining peptide sequences as strings
    ])

    buffer = []  # Buffer to hold data before writing
    pq_writer = args.output  # Output Parquet file path
    buffer_size = args.buffer_size  # Batch size for writing to Parquet
    sequence_length = args.length  # Length of peptide sequences to generate

    # Initialize the writer as None
    parquet_writer = None

    # Generate peptide sequences and write them in batches
    for pep_seq in itertools.product(sequence_aa, repeat=sequence_length):
        buffer.append({"pep_seq": ''.join(pep_seq)})

        # If the buffer reaches the batch size, write it to the Parquet file
        if len(buffer) == buffer_size:
            parquet_writer = write_to_parquet(buffer, pq_writer, schema,
                                              parquet_writer)
            buffer.clear()

    # Write any remaining sequences in the buffer
    if buffer:
        parquet_writer = write_to_parquet(buffer, pq_writer, schema,
                                          parquet_writer)

    # Close the writer after finishing
    if parquet_writer:
        parquet_writer.close()

    print(f"Peptide sequences successfully written to {pq_writer}.")


if __name__ == "__main__":
    main()

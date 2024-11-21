import itertools
import pyarrow as pa
import pyarrow.parquet as pq

def write_to_parquet(buffer, pq_writer, schema, writer=None):
    """
    Écrit les données d'un buffer dans un fichier Parquet.
    Si un writer est fourni, il ajoute les données. Sinon, il crée le fichier.
    """
    # Convertir le buffer en une table PyArrow
    table = pa.Table.from_pylist(buffer, schema=schema)

    if writer is None:  # Si le writer n'existe pas encore, on le crée
        writer = pq.ParquetWriter(pq_writer, schema, compression='BROTLI', compression_level=4)
    
    # Écrire les données dans le fichier Parquet
    writer.write_table(table)
    
    return writer

sequence_AA = 'ARNDEQGHILKFPSTYV'
schema = pa.schema([
    ('pep_seq', pa.string())
])

buffer = []
buffer_size = 1419857
pq_writer = '../results/peptides.parquet'

# Initialiser le writer comme None
parquet_writer = None

# Générer les séquences et écrire par lots
for pep_seq in itertools.product(sequence_AA, repeat=6):
    buffer.append({"pep_seq": ''.join(pep_seq)})

    if len(buffer) == buffer_size:
        parquet_writer = write_to_parquet(buffer, pq_writer, schema, parquet_writer)
        buffer.clear()

# Écrire les séquences restantes
if buffer:
    parquet_writer = write_to_parquet(buffer, pq_writer, schema, parquet_writer)

# Fermer le writer après usage
if parquet_writer:
    parquet_writer.close()



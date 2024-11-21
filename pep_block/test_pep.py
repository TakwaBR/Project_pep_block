from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import multiprocessing

hydrophobic_residues = ['L', 'V', 'I', 'F']
charged_residues = ['D', 'E', 'K', 'R', 'H']
polar_residues = ['D', 'E', 'H', 'K', 'N', 'Q', 'R', 'S', 'T', 'Y']
invalid_n_ter = ['Q', 'N', 'E', 'L', 'F', 'D', 'K', 'R']
invalid_c_ter = ['D']
restricted_residues = ['E', 'I', 'L', 'F', 'T', 'Y', 'V', 'P', 'D']
invalid_pairs = ["SS", "DG", "DP", "DS"]
forbidden_residues = {'V', 'I', 'Y', 'F', 'L', 'Q', 'T'}

def test_hydrophobic_percentage(counter):
    hydrophobic_count = sum(counter[aa] for aa in hydrophobic_residues)
    total_count = sum(counter.values())
    return (hydrophobic_count / total_count) < 0.5

def test_charged_residues(counter):
    charged_count = sum(counter[aa] for aa in charged_residues)
    return charged_count > 0

def test_polar_percentage(counter):
    polar_count = sum(counter[aa] for aa in polar_residues)
    total_count = sum(counter.values())
    return (polar_count / total_count) <= 0.75

def test_n_terminal(sequence):
    return sequence[0] not in invalid_n_ter

def test_c_terminal(sequence):
    return sequence[0] not in invalid_c_ter

def test_max_2_restricted(counter):
    restricted_count = sum(counter[aa] for aa in restricted_residues)
    return restricted_count <= 2

def test_no_consecutive(sequence):
    if "GGGG" in sequence:
        return False

    for i in range(len(sequence) - 1):
        pair = sequence[i] + sequence[i + 1]
        if pair in invalid_pairs:
            return False
    
    return True

def test_no_repeated_adjacent(sequence):
    for i in range(len(sequence) - 1):
        if sequence[i] == sequence[i + 1] and sequence[i] in forbidden_residues:
            return False
    
    return True

def all_tests(sequence):
    """Effectue tous les tests sur un peptide et retourne les résultats sous forme de dictionnaire."""
    counter = Counter(sequence)

    # Effectuer les tests
    hydrophobic_result = test_hydrophobic_percentage(counter)
    charged_result = test_charged_residues(counter)
    polar_result = test_polar_percentage(counter)
    n_terminal_result = test_n_terminal(sequence)
    c_terminal_result = test_c_terminal(sequence)
    max_2_restricted_result = test_max_2_restricted(counter)
    consecutive_result = test_no_consecutive(sequence)
    repeated_adjacent_result = test_no_repeated_adjacent(sequence)  # Nouveau test

    # Créer un dictionnaire pour la séquence avec les résultats des tests
    result_dict = {
        "Peptide": sequence,
        "Test 1": hydrophobic_result,
        "Test 2": charged_result,
        "Test 3": polar_result,
        "Test 4": n_terminal_result,
        "Test 5": c_terminal_result,
        "Test 6": max_2_restricted_result,
        "Test 7": consecutive_result,
        "Test 8": repeated_adjacent_result,
    }

    return result_dict


def main():
    parquet_file = "../results/peptides.parquet"
    output_file = "../results/peptides_with_all_tests.parquet"

    # Liste pour stocker tous les peptides
    peptides_passed = []

    try:
        reader = pq.ParquetFile(parquet_file)
        num_row_groups = reader.num_row_groups

        for batch_idx in range(num_row_groups):
            # Lire un paquet de données
            table = reader.read_row_group(batch_idx)
            sequences = table['pep_seq'].to_pylist()

            # Utiliser multiprocessing.Pool pour traiter les peptides en parallèle
            with multiprocessing.Pool() as pool:
                results = pool.map(all_tests, sequences)

            # Ajouter tous les peptides dans la liste
            peptides_passed.extend(results)

    except Exception as e:
        print(f"Erreur lors de la lecture du fichier Parquet : {e}")
        return

    # Ajouter des métadonnées expliquant les tests
    metadata = {
        "Test 1": "Test sur le pourcentage d'acides aminés hydrophobes (L, V, I, F) dans le peptide. Le test passe si < 50%.",
        "Test 2": "Test sur les résidus chargés (D, E, K, R, H). Le test passe si le peptide contient des résidus chargés.",
        "Test 3": "Test sur le pourcentage d'acides aminés polaires (D, E, H, K, N, Q, R, S, T, Y). Le test passe si <= 75%.",
        "Test 4": "Test sur le résidu N-terminal. Le test passe si le premier acide aminé n'est pas l'un des suivants : Q, N, E, L, F, D, K, R.",
        "Test 5": "Test sur le résidu C-terminal. Le test passe si le dernier acide aminé n'est pas D.",
        "Test 6": "Test sur les résidus spécifiques : Glu, Ile, Leu, Phe, Thr, Tyr, Val. Le test passe si le peptide en contient <= 2.",
        "Test 7": "Test sur les paires invalides de résidus consécutifs. Le test passe si aucune paire comme 'GGGG', 'SS', 'DG', 'DP', 'DS' n'est présente.",
        "Test 8": "Test sur les répétitions adjacentes de certains acides aminés (V, I, Y, F, W, L, Q, T). Le test passe si aucun de ces acides aminés n'est répété et adjacent."
    }

    # Écrire les peptides dans un fichier Parquet avec des métadonnées
    if peptides_passed:
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
        
        # Créer une table avec les données et les métadonnées
        table = pa.Table.from_pylist(peptides_passed, schema=schema)

        # Ajouter des métadonnées
        table = table.replace_schema_metadata(metadata)

        # Convertir en DataFrame pour afficher les 10 premiers peptides
        df = table.to_pandas()
        print(df.head(10))  # Afficher les 10 premiers peptides

        # Écrire le fichier Parquet
        pq.write_table(table, output_file, compression='BROTLI', compression_level=4)

        print(f"\nFichier Parquet écrit avec {len(peptides_passed)} peptides : {output_file}")
    else:
        print("\nAucun peptide n'a été traité.")


if __name__ == "__main__":
    main()

from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import PDB
from Bio.PDB.Polypeptide import PPBuilder
from meeko import MoleculePreparation
import pandas as pd
import os
import argparse
from multiprocessing import Pool, cpu_count
from rdkit.Chem.rdmolops import GetAdjacencyMatrix


def extract_chain_by_sequence(input_pdb, target_sequence):
    """
    Extract a specific chain from a PDB file by matching the sequence.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)
    ppb = PPBuilder()  # To extract polypeptides and sequences

    for model in structure:
        for chain in model:
            # Extract the sequence of the current chain
            polypeptides = ppb.build_peptides(chain)
            for poly in polypeptides:
                sequence = str(poly.get_sequence())
                print(f"Extracted sequence from chain {chain.id}: {sequence}")
                if sequence == target_sequence:
                    print(f"Found matching sequence in chain {chain.id}")
                    return chain
    print(f"No chain found matching sequence: {target_sequence}")
    return None  # Sequence not found


def read_parquet(file_path, peptide_col):
    """
    Read a Parquet file and extract peptide sequences.
    """
    df = pd.read_parquet(file_path)
    if peptide_col not in df.columns:
        raise ValueError(f"Column '{peptide_col}' not found in the Parquet file.")
    return df[peptide_col].tolist()


def generate_2d_structure(peptide_sequence):
    """
    Generate a 2D structure for a given peptide sequence.
    """
    mol = Chem.MolFromSequence(peptide_sequence)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return mol


def generate_3d_structure_from_2d(mol_2d):
    """
    Generate a 3D structure from a 2D molecule, ensuring chemical validity.
    """
    mol_3d = Chem.AddHs(mol_2d)  # Add hydrogen atoms
    success = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())  # Generate the 3D structure
    if success != 0:  # Embedding failed
        print("Failed to embed the 3D structure.")
        return None
    AllChem.UFFOptimizeMolecule(mol_3d)  # Optimize the 3D structure
    return mol_3d


def save_as_pdbqt(mol_3d, output_dir, file_prefix, idx):
    """
    Save a molecule in PDBQT format using Meeko.
    """
    try:
        preparator = MoleculePreparation(hydrate=True)
        preparator.prepare(mol_3d)
        pdbqt_path = os.path.join(output_dir, f"{file_prefix}_{idx}.pdbqt")
        preparator.write_pdbqt_file(pdbqt_path)
        print(f"PDBQT saved: {pdbqt_path}")
    except Exception as e:
        print(f"Failed to generate PDBQT for molecule {idx}: {e}")


def compare_atom_symbols(mol1, mol2):
    symbols1 = [atom.GetSymbol() for atom in mol1.GetAtoms()]
    print(symbols1)
    symbols2 = [atom.GetSymbol() for atom in mol2.GetAtoms()]
    print(symbols2)
    if symbols1 != symbols2:
        print("Les atomes diffèrent :", symbols1, symbols2)
        return False
    print("Les atomes sont identiques.")
    return True

def compare_bond_types(mol1, mol2):
    bonds1 = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in mol1.GetBonds()]
    bonds2 = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()) for bond in mol2.GetBonds()]
    if bonds1 != bonds2:
        print("Les types de liaison diffèrent :", bonds1, bonds2)
        return False
    print("Les types de liaison sont identiques.")
    return True


def process_peptide(peptide, idx, prefix, output_dir, template=None):
    """
    Process a single peptide to generate 2D and optionally 3D structures.
    """
    print(f"Processing peptide {idx}: {peptide}")
    mol_2d = generate_2d_structure(peptide)
    if not mol_2d:
        print(f"Failed to generate 2D structure for peptide {idx}")
        return None

    mol_3d = generate_3d_structure_from_2d(mol_2d)
    if not mol_3d:
        print(f"Failed to generate 3D structure for peptide {idx}")
        return None

    if template is not None:
        try:
            print(f"Assigning bond orders for peptide {idx} using the template...")
            mol_3d = AllChem.AssignBondOrdersFromTemplate(template, mol_3d)
            print(f"Successfully assigned bond orders for peptide {idx}.")
        except Exception as e:
            print(f"Error while assigning bond orders for peptide {idx}: {e}")
            return None

    save_as_pdbqt(mol_3d, output_dir, prefix, idx)

    return mol_3d


def main():
    """
    Main function to process peptides from a Parquet file using multiprocessing.
    """
    parser = argparse.ArgumentParser(description="Generate 2D and 3D constrained structures for peptides.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input Parquet file containing peptides.")
    parser.add_argument('-c', '--column', type=str, required=True, help="Column name containing peptide sequences.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Directory to save output structures.")
    parser.add_argument('--prefix', type=str, default="peptide", help="Prefix for saved structure files. Default is 'peptide'.")
    parser.add_argument('-t', '--template', type=str, help="Path to the template PDB file.")
    parser.add_argument('-s', '--sequence', type=str, default='SFLLRN', help="Target sequence for extracting the chain.")

    args = parser.parse_args()

    print(f"Reading peptides from: {args.input}")
    peptide_sequences = read_parquet(args.input, args.column)
    print(f"Number of peptides read: {len(peptide_sequences)}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    template = None
    if args.template:
        if not args.sequence:
            raise ValueError("You must provide a target sequence to extract the chain using the '--sequence' argument.")
        print(f"Loading template structure from: {args.template}")
        chain = extract_chain_by_sequence(args.template, args.sequence)
        if not chain:
            raise ValueError("Failed to find a matching chain for the provided sequence.")
        temp_pdb = os.path.join(args.output, "extracted_chain.pdb")
        io = PDB.PDBIO()
        io.set_structure(chain)
        io.save(temp_pdb)

        template_from_seq = Chem.MolFromSequence(args.sequence)
        if not template_from_seq:
            raise ValueError("Failed to create 3D structure from sequence.")

        print("Template chain loaded successfully.")

        template = Chem.MolFromPDBFile(temp_pdb, removeHs=False)
        template_from_seq = Chem.RemoveHs(template_from_seq)
        template = Chem.RemoveHs(template)

        AllChem.Compute2DCoords(template_from_seq)
        AllChem.Compute2DCoords(template)    
        template = Chem.AddHs(template)  # Re-add hydrogens if needed
        template_from_seq = Chem.AddHs(template_from_seq)
        AllChem.EmbedMolecule(template_from_seq, AllChem.ETKDG())
        AllChem.EmbedMolecule(template, AllChem.ETKDG())
        #AllChem.UFFOptimizeMolecule(template_from_seq)
        #AllChem.UFFOptimizeMolecule(template)

        print(len([1 for b in template.GetBonds() if b.GetBondTypeAsDouble() == 1.0]))
        print(len([1 for b in template_from_seq.GetBonds() if b.GetBondTypeAsDouble() == 1.0]))
        
        #template_from_seq = Chem.MolFromPDBFile(output_pdb_path, removeHs=True)
        #rmsd = AllChem.AlignMol(template, template_from_seq)

        #atom_mapping = template_from_seq.GetSubstructMatch(template)
        #print(atom_mapping)
        #template = Chem.RenumberAtoms(template, list(atom_mapping))

        print(GetAdjacencyMatrix(template))
        print(GetAdjacencyMatrix(template_from_seq))
        print(len([1 for b in template.GetBonds() if b.GetBondTypeAsDouble() == 1.0]))
        print("\nComparaison des atomes...")
        atoms_ok = compare_atom_symbols(template_from_seq, template)

        print("\nComparaison des types de liaison...")
        bonds_ok = compare_bond_types(template_from_seq, template)
        print("Template molecule loaded from PDB.")
        template = AllChem.AssignBondOrdersFromTemplate(template_from_seq, template)

    print(f"Generating structures for {len(peptide_sequences)} peptides using {cpu_count()} CPUs...")

    tasks = [(peptide, idx, args.prefix, args.output, template) for idx, peptide in enumerate(peptide_sequences)]

    with Pool() as pool:
        results = pool.starmap(process_peptide, tasks)

    valid_results = [result for result in results if result is not None]
    print(f"Processed {len(valid_results)} peptides out of {len(peptide_sequences)}.")

    print(f"Processing completed. Results saved in: {args.output}")


if __name__ == "__main__":
    main()

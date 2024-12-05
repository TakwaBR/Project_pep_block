from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import os
import argparse
from Bio import PDB
from Bio.PDB.Polypeptide import PPBuilder


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


def constrain_to_template(mol_3d, template):
    """
    Constrain a molecule to a given template using RDKit's ConstrainedEmbed.
    """
    try:
        Chem.SanitizeMol(mol_3d)
        constrained_mol = Chem.Mol(mol_3d)  # Create a copy to avoid modifying input
        AllChem.ConstrainedEmbed(constrained_mol, template)
        return constrained_mol
    except Exception as e:
        print(f"Constrained embedding failed: {e}")
        return None


def extract_chain_by_sequence(input_pdb, target_sequence):
    """
    Extract a specific chain from a PDB file by matching the sequence.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)
    ppb = PPBuilder()  # To extract polypeptides and sequences

    # Iterate over all models and chains
    for model in structure:
        for chain in model:
            # Get the sequence of the current chain
            polypeptides = ppb.build_peptides(chain)
            for poly in polypeptides:
                sequence = str(poly.get_sequence())
                print(f"Extracted sequence from chain {chain.id}: {sequence}")
                if sequence == target_sequence:
                    print(f"Found matching sequence in chain {chain.id}")
                    return chain
    return None  # Sequence not found


def process_peptide(peptide, idx, skip_3d, prefix, output_dir, template=None):
    """
    Process a single peptide to generate 2D and optionally 3D structures.
    """
    print(f"Processing peptide {idx}: {peptide}")
    mol_2d = generate_2d_structure(peptide)
    if not mol_2d:
        print(f"Failed to generate 2D structure for peptide {idx}")
        return None

    mol_3d = None
    if not skip_3d:
        mol_3d = generate_3d_structure_from_2d(mol_2d)
        if not mol_3d:
            print(f"Failed to generate 3D structure for peptide {idx}")
            return None

        if template is not None:
            mol_3d_constrained = constrain_to_template(mol_3d, template)
            if not mol_3d_constrained:
                print(f"Failed to constrain peptide {idx} to the template.")
                return None
            print(f"Successfully constrained peptide {idx} to the template.")

        #pdb_path = os.path.join(output_dir, f"{prefix}_3d_{idx}.pdb")
        #Chem.MolToPDBFile(mol_3d, pdb_path)
        #print(f"Saved 3D structure at: {pdb_path}")

    return mol_3d


def main():
    """
    Main function to process peptides from a Parquet file using multiprocessing.
    """
    parser = argparse.ArgumentParser(description="Generate 2D and 3D constrained structures for peptides.")
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input Parquet file containing peptides.")
    parser.add_argument('-c', '--column', type=str, required=True, help="Column name containing peptide sequences.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Directory to save output structures and images.")
    parser.add_argument('-n', '--num_images', type=int, default=2, help="Number of 2D images to save for visualization. Default is 2.")
    parser.add_argument('--skip_3d', action='store_true', help="Skip the generation of 3D structures.")
    parser.add_argument('--prefix', type=str, default="peptide", help="Prefix for saved structure files. Default is 'peptide'.")
    parser.add_argument('-t', '--template', type=str, default=None, help="Path to a PDB file of the template structure for constrained embedding.")
    parser.add_argument('-s', '--sequence', type=str, default='SFLLRN', help="Target sequence to extract the template chain.")

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

        template = Chem.MolFromPDBFile(temp_pdb, removeHs=False)
        Chem.SanitizeMol(template)
        if not template:
            raise ValueError("Failed to load the template structure into RDKit.")
        print("Template chain loaded successfully.")

    print(f"Generating structures for {len(peptide_sequences)} peptides using {cpu_count()} CPUs...")

    tasks = [(peptide, idx, args.skip_3d, args.prefix, args.output, template) for idx, peptide in enumerate(peptide_sequences)]

    with Pool() as pool:
        results = pool.starmap(process_peptide, tasks)

    valid_results = [result for result in results if result is not None]
    print(f"Processed {len(valid_results)} peptides out of {len(peptide_sequences)}.")

    if valid_results:
        img = Draw.MolsToGridImage(valid_results[:args.num_images], molsPerRow=2, subImgSize=(300, 300))
        img_path = os.path.join(args.output, f"{args.prefix}_2d_structures.png")
        img.save(img_path)
        print(f"Saved 2D structure images at: {img_path}")

    print(f"Processing completed. Results saved in: {args.output}")


if __name__ == "__main__":
    main()

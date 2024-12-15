"""
Module for 2D and 3D generation peptides structures and PDBQT saving.

Main functions:
- extract_chain_by_sequence: Extracts a chain from a PDB file based
    on a sequence match.
- generate_2d_structure: Generates a 2D structure for a peptide sequence.
- generate_3d_structure_from_2d: Converts a 2D structure to 3D.
- neutralize_atoms: Neutralizes atom charges in a molecule.
- save_as_pdbqt: Saves a molecule in PDBQT format.
- process_peptide: Processes a peptide, generates 2D/3D structures,
    and optionally aligns to a template.
- main: Runs the main workflow for peptide processing using multiprocessing.
"""

import os
import argparse
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from Bio import PDB
from Bio.PDB import PDBIO
from Bio.PDB.Polypeptide import PPBuilder
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
import pandas as pd


def extract_chain_by_sequence(input_pdb, target_sequence):
    """
    Extract a specific chain from a PDB file by matching the sequence.

    Args:
        input_pdb (str): Path to the input PDB file.
        target_sequence (str): Target sequence to find in the structure.

    Returns:
        Bio.PDB.Chain.Chain: The extracted chain if a match is found,
        otherwise None.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)
    ppb = PPBuilder()

    for model in structure:
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            for poly in polypeptides:
                sequence = str(poly.get_sequence())
                print(f"Extracted sequence from chain {chain.id}: {sequence}")
                if sequence == target_sequence:
                    print(f"Found matching sequence in chain {chain.id}")
                    return chain
    print(f"No chain found matching sequence: {target_sequence}")
    return None


def save_extracted_chain(chain, output_path):
    """
    Save the extracted chain to a PDB file.

    Args:
        chain (Bio.PDB.Chain.Chain): The chain object to save.
        output_path (str): Path to save the extracted chain in PDB format.

    Returns:
        None
    """
    io = PDBIO()
    io.set_structure(chain)
    io.save(output_path)
    print(f"Extracted chain saved to {output_path}")


def read_parquet(file_path, peptide_col):
    """
    Read a Parquet file and extract peptide sequences.

    Args:
        file_path (str): Path to the Parquet file.
        peptide_col (str): Name of the column containing peptide sequences.

    Returns:
        list: List of peptide sequences.
    """
    df = pd.read_parquet(file_path)
    if peptide_col not in df.columns:
        raise ValueError(f"Column '{peptide_col}' not found in the Parquet\
                         file.")
    return df[peptide_col].tolist()


def generate_2d_structure(peptide_sequence):
    """
    Generate a 2D molecular structure from a peptide sequence.

    Args:
        peptide_sequence (str): The peptide sequence.

    Returns:
        rdkit.Chem.Mol: The 2D molecular structure or None if generation fails.
    """
    mol = Chem.MolFromSequence(peptide_sequence)
    if mol:
        AllChem.Compute2DCoords(mol)
    return mol


def generate_2d_images(peptides, output_image_path, n_images):
    """
    Generate a 2D image grid of peptide molecules and save it.

    Args:
        peptides (list): List of peptide sequences.
        output_image_path (str): Path to save the generated image.
        n_images (int): Number of peptides to include in the grid.

    Returns:
        None
    """
    print(f"Generating 2D images for {n_images} peptides...")
    molecules = [generate_2d_structure(peptide) for peptide in
                 peptides[:n_images]]
    molecules = [mol for mol in molecules if mol is not None]

    img = Draw.MolsToGridImage(
        molecules,
        molsPerRow=2,  # Number of molecules per row
        subImgSize=(200, 200),  # Size of each molecule image
        legends=[f"Peptide {i+1}" for i in range(len(molecules))]
    )
    img.save(output_image_path)
    print(f"2D image grid saved to {output_image_path}")


def generate_3d_structure_from_2d(mol_2d):
    """
    Generate a 3D structure from a 2D molecular structure.

    Args:
        mol_2d (rdkit.Chem.Mol): The 2D molecular structure.

    Returns:
        rdkit.Chem.Mol: The optimized 3D molecular structure or None if
        embedding fails.
    """
    mol_3d = Chem.AddHs(mol_2d)
    success = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
    if success != 0:
        print("Failed to embed the 3D structure.")
        return None
    AllChem.UFFOptimizeMolecule(mol_3d)
    return mol_3d


def neutralize_atoms(mol):
    """
    Neutralize charges on atoms in a molecule.

    Args:
        mol (rdkit.Chem.Mol): The input molecule.

    Returns:
        rdkit.Chem.Mol: The molecule with neutralized charges.
    """
    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def save_as_pdbqt(mol_3d, output_dir, file_prefix, peptide):
    """
    Save a 3D molecular structure in PDBQT format.

    Args:
        mol_3d (rdkit.Chem.Mol): The 3D molecular structure.
        output_dir (str): Directory to save the PDBQT file.
        file_prefix (str): Prefix for the file name.
        idx (int): Index of the molecule.

    Returns:
        None
    """
    try:
        preparator = MoleculePreparation()
        molecule_setups = preparator.prepare(mol_3d)

        pdbqt_path = os.path.join(output_dir, f"{file_prefix}_{peptide}.pdbqt")

        with open(pdbqt_path, 'w', encoding='utf-8') as pdbqt_file:
            writer = PDBQTWriterLegacy()
            for setup in molecule_setups:
                pdbqt_content = writer.write_string(setup)
                pdbqt_content = pdbqt_content[0]
                pdbqt_file.write(pdbqt_content)

        print(f"PDBQT saved: {pdbqt_path}")
    except FileNotFoundError as fnf_error:
        print(f"File not found error while generating PDBQT for molecule\
              {peptide}: {fnf_error}")
    except PermissionError as perm_error:
        print(f"Permission error while generating PDBQT for molecule\
              {peptide}: {perm_error}")
    except AttributeError as attr_error:
        print(f"Attribute error while generating PDBQT for molecule\
              {peptide}: {attr_error}")


def save_as_pdb(mol_3d, pdb_output_dir, file_prefix, peptide):
    """
    Save a 3D molecular structure in PDB format.

    Args:
        mol_3d (rdkit.Chem.Mol): The 3D molecular structure.
        pdb_output_dir (str): Directory to save the PDB file.
        file_prefix (str): Prefix for the file name.
        peptide (str): Peptide sequence for naming.

    Returns:
        None
    """
    pdb_path = os.path.join(pdb_output_dir, f"{file_prefix}_{peptide}.pdb")
    writer = Chem.rdmolfiles.PDBWriter(pdb_path)
    try:
        writer.write(mol_3d)
        print(f"PDB saved: {pdb_path}")
    except FileNotFoundError:
        print(f"File not found: Unable to save PDB for peptide {peptide}.\
              Check the directory path.")
    except PermissionError:
        print(f"Permission denied: Unable to save PDB for peptide {peptide}.\
              Verify your access rights.")
    except AttributeError as attr_error:
        print(f"Attribute error while saving PDB for peptide\
              {peptide}: {attr_error}")
    except Chem.rdchem.MolSanitizeException as mol_error:
        print(f"Sanitization error for peptide {peptide}: {mol_error}")
    finally:
        writer.close()


def process_peptide(peptide, idx, prefix, output_dir, pdb_output_dir,
                    template):
    """
    Process a peptide to generate 3D structures and save it in pdbqt files.

    Args:
        peptide (str): The peptide sequence.
        idx (int): Index of the peptide.
        prefix (str): Prefix for saved file names.
        output_dir (str): Directory to save the output.
        template (rdkit.Chem.Mol): Template molecule for alignment (optional).

    Returns:
        rdkit.Chem.Mol: The processed 3D molecule or None if processing fails.
    """
    print(f"Processing peptide {idx}: {peptide}")
    mol_2d = generate_2d_structure(peptide)
    if not mol_2d:
        print(f"Failed to generate 2D structure for peptide {idx}: {peptide}")
        return None

    mol_3d = generate_3d_structure_from_2d(mol_2d)
    if not mol_3d:
        print(f"Failed to generate 3D structure for peptide {idx}: {peptide}")
        return None

    if template:
        try:
            mol_pair = [mol_3d, template]
            mcs = rdFMCS.FindMCS(mol_pair, threshold=0.9,
                                 completeRingsOnly=True)
            smarts = Chem.MolFromSmarts(mcs.smartsString)
            core = AllChem.ReplaceSidechains(template, smarts)
            core = AllChem.DeleteSubstructs(core, Chem.MolFromSmiles("*"))
            AllChem.ConstrainedEmbed(mol_3d, core)
        except ValueError as value_error:
            print(f"Value error while aligning to template for peptide\
                  {idx}: {value_error}")
            return None
        except TypeError as type_error:
            print(f"Type error while aligning to template for peptide\
                  {idx}: {type_error}")
            return None
        except Chem.rdchem.RDKitError as rdkit_error:
            print(f"RDKit error while aligning to template for peptide\
                  {idx}: {rdkit_error}")
            return None
    save_as_pdb(mol_3d, pdb_output_dir, prefix, peptide)
    save_as_pdbqt(mol_3d, output_dir, prefix, peptide)
    return mol_3d


def main():
    """Run main function to process peptides from a Parquet file."""
    parser = argparse.ArgumentParser(description="Generate 2D and 3D\
                                     constrained structures for peptides.")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Path to the input Parquet file containing\
                            peptides.")
    parser.add_argument('-c', '--column', type=str, required=True,
                        help="Column name containing peptide sequences.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="Directory to save output structures.")
    parser.add_argument('-op', '--pdb_output', type=str, required=True,
                        help="Directory to save PDB files separately.")
    parser.add_argument('--prefix', type=str, default="peptide",
                        help="Prefix for saved structure files.")
    parser.add_argument('-t', '--template', type=str,
                        help="Path to the template PDB file.")
    parser.add_argument('-s', '--sequence', type=str, default='SFLLRN',
                        help="Target sequence for extracting the chain.")
    parser.add_argument('-p', '--images', type=str,
                        help="Path to save the combined 2D image.")
    parser.add_argument('--n_images', type=int, default=4,
                        help="Number of 2D images to include in the\
                            combined image.")
    parser.add_argument('-other', '--other', type=str, nargs="+",
                        help="Other peptides to be processed.")

    args = parser.parse_args()

    peptide_sequences = read_parquet(args.input, args.column)
    print(f"Read {len(peptide_sequences)} peptides from {args.input}.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.pdb_output):
        os.makedirs(args.pdb_output)

    if args.template:
        chain = extract_chain_by_sequence(args.template, args.sequence)
        if chain:
            template_path = os.path.join(args.pdb_output,
                                         f"template_{args.sequence}.pdb")
            save_extracted_chain(chain, template_path)
        else:
            print(f"Template chain matching sequence\
                  {args.sequence} not found in {args.template}.")
            return
    print(f"Loading template from: {args.template}")
    template = Chem.MolFromPDBFile(template_path)
    if not template:
        print(f"Error: Unable to load the template from '{args.template}'.")
        return

    template = neutralize_atoms(template)

    if args.images:
        generate_2d_images(peptide_sequences, args.images, args.n_images)

    print("Processing peptides...")
    tasks = [(peptide, idx, args.prefix, args.output, args.pdb_output,
              template)
             for idx, peptide in enumerate(peptide_sequences)]

    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_peptide, tasks)

    if args.other:
        for idx, arg in enumerate(args.other):
            print(f"Processing the other peptide {arg}...")
            process_peptide(arg, idx, args.prefix, args.output,
                            args.pdb_output, template)

    valid_results = [res for res in results if res]
    print(f"Processed {len(valid_results)} out of {len(peptide_sequences)}\
           peptides.")
    print(f"Results saved in {args.output}.")


if __name__ == "__main__":
    main()

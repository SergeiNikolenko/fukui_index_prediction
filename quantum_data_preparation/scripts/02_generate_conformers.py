# Generates multiple conformations for each molecule and saves them in SDF format.
import os
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Create output directory for SDF files
sdf_files_dir = './sdf_files'
os.makedirs(sdf_files_dir, exist_ok=True)

# Configure logging
logging.basicConfig(filename='generate_conformers.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to generate conformers and save them as SDF
def generate_conformations_sdf(mol, sdf_path):
    try:
        logging.info(f"Generating SDF file at {sdf_path}")
        writer = Chem.SDWriter(sdf_path)
        mol = Chem.AddHs(mol)  # Add hydrogens
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=5)  # Generate 5 conformers
        for id in ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=id)
            ff.Minimize()  # Minimize energy
            energy_value = ff.CalcEnergy()  # Calculate energy
            logging.info(f"Conformation {id}: Energy = {energy_value:.2f}")
            mol.SetProp('ENERGY', '{0:.2f}'.format(energy_value))  # Save energy value
            writer.write(mol, confId=id)  # Write to SDF
        logging.info(f"Completed SDF file generation for {sdf_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating conformations for {sdf_path}: {e}")
        return False

# Load the SMILES data
csv_file = Path('smiles_id.csv')
if not csv_file.is_file():
    raise FileNotFoundError(f"File {csv_file} not found.")

data = pd.read_csv(csv_file)
smiles_list = data['smiles'].tolist()
indices = data['index'].tolist()

# Process each SMILES string and generate conformers
for idx, smiles in tqdm(zip(indices, smiles_list), desc='Generating conformers', total=len(smiles_list)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        sdf_file = Path(sdf_files_dir) / f'{idx}.sdf'
        if generate_conformations_sdf(mol, str(sdf_file)):
            logging.info(f"Successfully processed SMILES {smiles} with index {idx}")
        else:
            logging.error(f"Failed to generate conformers for SMILES {smiles} with index {idx}")
    else:
        logging.warning(f"SMILES {smiles} could not be converted to a molecule")

logging.info('Conformer generation complete.')

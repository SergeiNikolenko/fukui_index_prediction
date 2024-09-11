# Executes MOPAC calculations for geometry optimization.
import os
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atom, Atoms
from ase.calculators.mopac import MOPAC
from pathlib import Path
import pandas as pd

sdf_files_dir = './sdf_files'

# Create a new directory for MOPAC calculations
existing_directories = [dir_name for dir_name in os.listdir() if dir_name.startswith('calc_')]
existing_numbers = [int(dir_name.replace('calc_', '')) for dir_name in existing_directories if dir_name.replace('calc_', '').isdigit()]

next_number = max(existing_numbers) + 1 if existing_numbers else 1
new_directory_name = f'calc_{next_number}'
os.makedirs(new_directory_name, exist_ok=True)

# Set up logging
log_file_name = f'{new_directory_name}.log'
logging.basicConfig(filename=log_file_name, level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Function to generate conformations and save them as SDF
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

# Load SMILES data
csv_file = Path('smiles_id.csv')
if not csv_file.is_file():
    raise FileNotFoundError(f"File {csv_file} not found.")

data = pd.read_csv(csv_file)
smiles = data['smiles'].tolist()
indices = data['index'].tolist()

# Function to convert XYZ to atom objects
def xyz_to_atoms(xyz_path):
    with open(xyz_path) as file:
        lines = file.readlines()
    n_atoms = int(lines[0].strip())
    atoms = []
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        atoms.append(Atom(parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
    return Atoms(atoms)

# MOPAC calculation class
class MYMOPAC(MOPAC):
    def __init__(self, label, task):
        super().__init__(label=os.path.join(new_directory_name, label), task=task)

    def read_results(self):
        with open(self.label + '.out') as fd:
            lines = fd.readlines()

        num_atoms = int([line for line in lines if 'Empirical Formula' in line][0].split()[-2])
        atoms = []
        for line in lines:
            if 'CARTESIAN COORDINATES' in line:
                for atom_line in lines[lines.index(line)+1:lines.index(line)+1+num_atoms]:
                    parts = atom_line.split()
                    atoms.append(Atom(parts[1], (float(parts[2]), float(parts[3]), float(parts[4]))))
                break
        self.atom_objects = Atoms(atoms)

    def get_atom_objects(self):
        return self.atom_objects

# MOPAC optimization function
def mopac_opt_thermo(e, conformations):
    for i, atoms in enumerate(conformations):
        optimization_label = f'{e}_conformation_optimized_{i}'
        atoms.calc = MYMOPAC(label=optimization_label, task='XYZ CHARGE=0 Singlet BONDS OPT AUX')
        try:
            atoms.get_potential_energy()  # Run MOPAC optimization
            logging.info(f"Optimization successful for {optimization_label}")
        except Exception as ex:
            logging.error(f"Error in MOPAC optimization: {ex}")
        atoms.calc.read_results()

# Main processing loop
for idx, smi in zip(indices, smiles):
    logging.info(f"Processing SMILES {smi} with index {idx}")
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        sdf_file = Path(sdf_files_dir) / f'{idx}.sdf'
        xyz_file = Path(sdf_files_dir) / f'{idx}.xyz'
        if generate_conformations_sdf(mol, str(sdf_file)):
            conformations = xyz_to_atoms(str(xyz_file))
            mopac_opt_thermo(idx, conformations)
        else:
            logging.error(f"Failed to generate conformations for SMILES {smi} with index {idx}")
    else:
        logging.warning(f"SMILES {smi} could not be converted to a molecule")

logging.info('MOPAC optimization complete.')

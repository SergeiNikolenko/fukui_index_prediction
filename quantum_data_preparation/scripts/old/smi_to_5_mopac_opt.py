import logging
import os
from pathlib import Path

import pandas as pd
from ase import Atom, Atoms
from ase.calculators.mopac import MOPAC
from rdkit import Chem
from rdkit.Chem import AllChem

existing_directories = [
    dir_name for dir_name in os.listdir() if dir_name.startswith("calc_")
]
existing_numbers = [
    int(dir_name.replace("calc_", ""))
    for dir_name in existing_directories
    if dir_name.replace("calc_", "").isdigit()
]

next_number = max(existing_numbers) + 1 if existing_numbers else 1
new_directory_name = f"calc_{next_number}"

os.makedirs(new_directory_name, exist_ok=True)

log_file_name = f"{new_directory_name}.log"
logging.basicConfig(
    filename=log_file_name,
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


# Function to generate sdf files with 5 conformations per molecule


def generate_conformations_sdf(mol, sdf_path):
    try:
        logging.info(f"Generating SDF file at {sdf_path}")
        writer = Chem.SDWriter(sdf_path)
        mol = Chem.AddHs(mol)
        logging.info("Added hydrogens to the molecule.")
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=5)
        for id in ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=id)
            ff.Minimize()
            energy_value = ff.CalcEnergy()
            logging.info(f"Conformation {id}: Energy = {energy_value:.2f}")
            mol.SetProp("ENERGY", f"{energy_value:.2f}")
            writer.write(mol, confId=id)
        logging.info(f"Completed SDF file generation for {sdf_path}")

        logging.info(f"Completed SDF file generation for {sdf_path}")
        return True
    except Exception as e:
        logging.error(f"Error generating conformations for {sdf_path}: {e}")
        return False


# Create sdf-files directory inside the new directory
sdf_files_dir = f"{new_directory_name}/sdf_files"
if not os.path.exists(sdf_files_dir):
    os.mkdir(sdf_files_dir)


# Function to convert sdf to xyz
def sdf_to_xyz(sdf_path, xyz_path):
    with open(sdf_path) as file:
        array = [row.strip() for row in file]
        j = 0
        while j < len(array):
            if "RDKit" in array[j]:
                temp_array_a = array[j + 2].split()
                n_atoms = int(temp_array_a[0])
                with open(xyz_path, "a") as out_file:
                    out_file.write(f"{n_atoms}\n")
                for i in range(n_atoms):
                    temp_array_b = array[j + 3 + i].split()
                    x, y, z, atomic_symbol = temp_array_b[0:4]
                    with open(xyz_path, "a") as out_file:
                        out_file.write(f"{atomic_symbol} {x:14} {y:12} {z:12}\n")
                j += 3 + n_atoms
            else:
                j += 1


# Function to convert xyz to atom objects
def xyz_to_atoms(xyz_path):
    with open(xyz_path) as file:
        lines = file.readlines()
    n = int(lines[0])
    molecula = []
    for i in range(5):
        names, xyzs = [], []
        for j in range(n):
            line = lines[i * (n + 1) + 1 + j].split()
            names.append(line[0])
            xyzs += line[1:]
        myatoms = [
            Atom(
                names[k],
                (float(xyzs[3 * k]), float(xyzs[3 * k + 1]), float(xyzs[3 * k + 2])),
            )
            for k in range(len(names))
        ]
        molecula.append(Atoms(myatoms))
    return molecula


class MYMOPAC(MOPAC):
    def __init__(self, label, task):
        super().__init__(label=os.path.join(new_directory_name, label), task=task)

    def read_results(self):
        with open(self.label + ".out") as fd:
            lines = fd.readlines()

        for i in range(len(lines)):
            if lines[i].find("Empirical Formula") != -1:
                self.num_atoms = int(lines[i].split()[-2])
                break
        flag = True
        for i in range(-2, -len(lines), -1):
            if (
                lines[i - 1].find("CARTESIAN COORDINATES") != -1
                and lines[i].isspace() == True
            ):
                number = i
                names = []
                xyzs = []
                break
        for j in range(self.num_atoms):
            print(f"{j+1}/{self.num_atoms}", lines[number + 1 + j])
            xyzs += lines[number + 1 + j].split()[2:]
            names.append(lines[number + 1 + j].split()[1])
        myatoms = []
        i = 0
        while i < len(names):
            A = Atom(
                str(names[i]),
                (float(xyzs[3 * i]), float(xyzs[3 * i + 1]), float(xyzs[3 * i + 2])),
            )
            myatoms.append(A)
            i += 1
        self.atom_objects = Atoms(myatoms)

    def get_atom_objects(self):
        return self.atom_objects

    def get_num_atoms(self):
        return self.num_atoms


############   Функция оптимизации и расчета e-й молекулы без растворителя #####################


def mopac_opt_thermo(e):
    for i in range(5):
        atoms = conformations[i]
        optimization_label = f"{e}_conformation_optimized_{i}"
        thermo_label = f"{e}_conformation_thermo_{i}"

        atoms.calc = MYMOPAC(
            label=optimization_label, task="XYZ CHARGE=0 Singlet BONDS OPT AUX"
        )
        try:
            atoms.get_potential_energy()
            logging.info(
                f"Potential energy calculated successfully for {optimization_label}"
            )
        except Exception as ex:
            pass

        atoms.calc.read_results()
        logging.info(f"Results read successfully for {optimization_label}")


#######################################################################


csv_file = Path("smiles_id.csv")
if not csv_file.is_file():
    raise FileNotFoundError(f"File {csv_file} not found.")

data = pd.read_csv(csv_file)
smiles = data["smiles"].tolist()
indices = data["index"].tolist()

for idx, smi in zip(indices, smiles):
    logging.info(f"Processing SMILES {smi} with index {idx}")
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        sdf_file = Path(sdf_files_dir) / f"{idx}.sdf"
        xyz_file = Path(sdf_files_dir) / f"{idx}.xyz"
        if generate_conformations_sdf(mol, str(sdf_file)):
            sdf_to_xyz(str(sdf_file), str(xyz_file))
            conformations = xyz_to_atoms(str(xyz_file))
            mopac_opt_thermo(idx)
        else:
            logging.error(
                f"Skipping molecule with index {idx} due to errors in conformation generation."
            )
        logging.info(f"Completed processing for SMILES {smi} with index {idx}")
    else:
        logging.warning(f"SMILES {smi} could not be converted to a molecule")

logging.info("Done!")

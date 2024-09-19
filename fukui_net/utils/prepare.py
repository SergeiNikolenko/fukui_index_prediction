import ast
import os
from typing import List, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from rdkit import Chem
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

tqdm.pandas()


class FeaturizationParameters:
    """
    A class to store parameters used for atom featurization.

    Attributes:
        max_atomic_num (int): Maximum atomic number considered.
        atom_features (dict): Dictionary of atomic features with possible values for each feature.
        atom_fdim (int): Dimensionality of atom feature vector.
    """

    def __init__(self):
        self.max_atomic_num = 100
        self.atom_features = {
            "atomic_num": list(range(self.max_atomic_num)),
            "degree": [0, 1, 2, 3, 4, 5],
            "formal_charge": [-1, -2, 1, 2, 0],
            "chiral_tag": [0, 1, 2, 3],
            "num_Hs": [0, 1, 2, 3, 4],
            "hybridization": [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        }
        self.atom_fdim = (
            sum(len(choices) + 1 for choices in self.atom_features.values()) + 2
        )


def onek_encoding_unk(value, choices):
    """
    One-hot encoding for a given value with support for unknown values.

    Args:
        value (int): The value to encode.
        choices (list): List of possible values.

    Returns:
        list: One-hot encoded list.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_skipatom_vector(atom_symbol, skipatom_model):
    """
    Retrieves the SkipAtom vector for a given atom symbol.

    Args:
        atom_symbol (str): The chemical symbol of the atom.
        skipatom_model: The SkipAtom model object.

    Returns:
        list: SkipAtom feature vector.
    """
    if atom_symbol in skipatom_model.dictionary:
        return skipatom_model.vectors[skipatom_model.dictionary[atom_symbol]].tolist()
    else:
        return [0] * skipatom_model.vectors.shape[1]


def atom_features(atom, params, skipatom_model=None):
    """
    Generates features for a given atom.

    Args:
        atom: RDKit atom object.
        params: FeaturizationParameters object.
        skipatom_model: SkipAtom model object.

    Returns:
        list: Atom feature vector.
    """
    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, params.atom_features["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), params.atom_features["degree"])
        + onek_encoding_unk(
            atom.GetFormalCharge(), params.atom_features["formal_charge"]
        )
        + onek_encoding_unk(
            int(atom.GetChiralTag()), params.atom_features["chiral_tag"]
        )
        + onek_encoding_unk(int(atom.GetTotalNumHs()), params.atom_features["num_Hs"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), params.atom_features["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )  # Scale mass

    if skipatom_model is not None:
        atom_symbol = atom.GetSymbol()
        skipatom_features = get_skipatom_vector(atom_symbol, skipatom_model)
        features += skipatom_features

    return features


PARAMS = {"BOND_FDIM": 10}


def bond_features(
    bond: Chem.rdchem.Bond, skipatom_model=None
) -> list[bool | int | float | np.ndarray]:
    """
    Generates features for a given bond.

    Args:
        bond: RDKit bond object.
        skipatom_model: SkipAtom model object.

    Returns:
        list: Bond feature vector.
    """
    if bond is None:
        fbond = [1] + [0] * (PARAMS["BOND_FDIM"] - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated() if bt is not None else 0,
            bond.IsInRing() if bt is not None else 0,
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond


class MoleculeData:
    """
    A class to represent a molecule and its features for graph-based learning.

    Attributes:
        smiles (str): SMILES string of the molecule.
        target (torch.Tensor): Target value(s) for the molecule.
        mol: RDKit molecule object.
        params: FeaturizationParameters object.
        skipatom_model: SkipAtom model object.
        edge_index (torch.Tensor): Edge indices of the molecular graph.
        edge_attr (torch.Tensor): Edge attributes of the molecular graph.
    """

    def __init__(self, smiles, target, addHs=True, skipatom_model=None):
        self.smiles = smiles
        self.target = torch.tensor(target, dtype=torch.float)
        self.mol = Chem.MolFromSmiles(smiles)
        if addHs:
            self.mol = Chem.AddHs(self.mol)
        self.params = FeaturizationParameters()
        self.skipatom_model = skipatom_model
        self.edge_index, self.edge_attr = self.construct_graph()

    def construct_graph(self):
        """
        Constructs a graph representation of the molecule with edge indices and attributes.

        Returns:
            edge_index (torch.Tensor): Edge indices.
            edge_attr (torch.Tensor): Edge attributes.
        """
        edge_index = []
        edge_attr = []
        for bond in self.mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.extend([[start, end], [end, start]])
            edge_attr.extend(
                [
                    bond_features(bond, self.skipatom_model),
                    bond_features(bond, self.skipatom_model),
                ]
            )
        return torch.tensor(edge_index).t().contiguous(), torch.tensor(
            edge_attr, dtype=torch.float
        )

    def generate_atom_features(self):
        """
        Generates features for each atom in the molecule.

        Returns:
            torch.Tensor: Atom features.
        """
        features = []
        for atom in self.mol.GetAtoms():
            features.append(atom_features(atom, self.params, self.skipatom_model))
        return torch.tensor(features, dtype=torch.float)


class MoleculeDataset(Dataset):
    """
    A PyTorch dataset class to represent a collection of molecules.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing SMILES and target values.
        smiles_column (str): Column name for SMILES strings.
        target_column (str): Column name for target values.
        addHs (bool): Whether to add hydrogens to molecules.
        n_jobs (int): Number of parallel jobs to use for data processing.
        skipatom_model: SkipAtom model object.
        data_list (list): List of MoleculeData objects.
    """

    def __init__(
        self,
        dataframe,
        smiles_column="smiles",
        target_column="target",
        addHs=True,
        n_jobs=-1,
        skipatom_model=None,
    ):
        super().__init__()
        self.use_skipatom = skipatom_model is not None
        self.data_list = Parallel(n_jobs=n_jobs)(
            delayed(
                lambda row: MoleculeData(
                    row[smiles_column], row[target_column], addHs, skipatom_model
                )
            )(row)
            for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0])
        )

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        """
        Retrieves a molecule from the dataset at the given index.

        Args:
            idx (int): Index of the molecule.

        Returns:
            Data: PyTorch Geometric Data object containing the graph representation of the molecule.
        """
        molecule_data = self.data_list[idx]
        x = molecule_data.generate_atom_features()
        edge_index = molecule_data.edge_index
        edge_attr = molecule_data.edge_attr
        y = molecule_data.target

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.smiles = molecule_data.smiles

        return data


def convert_string_to_list(string):
    """
    Converts a string representation of a list into an actual list object.

    Args:
        string (str): String representation of a list.

    Returns:
        list: Parsed list object, or an empty list if parsing fails.
    """
    try:
        return ast.literal_eval(string)
    except ValueError:
        return []

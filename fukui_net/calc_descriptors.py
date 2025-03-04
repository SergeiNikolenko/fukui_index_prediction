import warnings

import datamol as dm
import pandas as pd
from joblib import Parallel, delayed
from pandarallel import pandarallel
from rdkit import rdBase

rdBase.DisableLog("rdApp.*")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

pandarallel.initialize(progress_bar=False)

# Загрузка данных
data = pd.read_csv("../data/processed/qm_calc_data.csv")
print("data load")

print(data.shape)
data["smiles"] = data["smiles"].str.split(".")
data = data.explode("smiles")
print(data.shape)

smiles_column = "smiles"

data_unique = data.drop_duplicates(subset=[smiles_column]).copy()

data_unique = data_unique[data_unique[smiles_column].notna()].copy()

data_unique["mol"] = data_unique[smiles_column].parallel_apply(dm.to_mol)

data_unique = data_unique[data_unique["mol"].parallel_apply(lambda x: x is not None)]

mols = data_unique["mol"].tolist()

num_unique_structures = len(data_unique)
print(
    "Количество уникальных структур после удаления дубликатов и некорректных молекул:",
    num_unique_structures,
)


def safe_compute_descriptors(mol):
    try:
        return dm.descriptors.compute_many_descriptors(mol)
    except Exception:
        return None


# Вычисляем дескрипторы с обработкой исключений на уровне молекул
descriptors = Parallel(n_jobs=-1)(
    delayed(safe_compute_descriptors)(mol) for mol in mols
)

# Фильтруем успешные вычисления
descriptors = [desc for desc in descriptors if desc is not None]

if descriptors:
    descriptors_df = pd.DataFrame(descriptors)
    descriptors_df.insert(
        0, smiles_column, data_unique[smiles_column].values[: len(descriptors_df)]
    )
    descriptors_df.to_parquet("../data/qm_calc_descriptors.parquet")
    print("Descriptors saved to parquet file.")
else:
    print("Descriptor computation failed for all molecules, no file was saved.")

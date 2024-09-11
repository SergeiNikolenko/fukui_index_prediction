# Handles data preprocessing, such as canonicalizing SMILES and merging datasets.
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os

# Function to canonicalize SMILES
def canonical_smiles(smiles):
    if pd.isna(smiles):
        return None, False, True
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol), True, False
        else:
            return None, False, False
    except:
        return None, False, False

# Load the primary dataset
data = pd.read_csv('../data/QM_137k.csv')

# Process SMILES and convert to canonical form
data['Canonical_smiles'], data['Conversion_Success'], data['Is_NaN'] = zip(*[canonical_smiles(smile) for smile in tqdm(data['smiles'], desc='Processing data')])

# Check for duplicates in SMILES
duplicate_counts = data['smiles'].value_counts()
print(f"Duplicate SMILES count: {duplicate_counts[duplicate_counts > 1].count()}")

# Load additional datasets from the './data' folder
folder_path = './data'
combined_df = pd.DataFrame()

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, sep=';')
        df['Canonical_Smiles'], df['Conversion_Success'], df['Is_NaN'] = zip(*[canonical_smiles(smile) for smile in tqdm(df['Smiles'], desc=f'Processing {file}')])
        combined_df = pd.concat([combined_df, df])

# Filter successful conversions and sort by canonical SMILES
combined_df_sorted = combined_df[combined_df['Conversion_Success'] == True]
combined_df_sorted = combined_df_sorted[['Canonical_Smiles']]

# Find intersection between primary data and additional datasets
data_intersection = data[data['Canonical_smiles'].isin(combined_df_sorted['Canonical_Smiles'])].copy()
data_intersection = data_intersection.drop(['Canonical_smiles', 'Conversion_Success', 'Is_NaN'], axis=1)
data_intersection.reset_index(inplace=True)

# Save the result as 'smiles_id.csv'
df = data_intersection[['index', 'smiles']].copy()
df.to_csv('smiles_id.csv', index=False)

print('SMILES data processing complete.')

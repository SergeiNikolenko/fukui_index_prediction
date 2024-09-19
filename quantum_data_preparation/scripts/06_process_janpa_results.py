from pathlib import Path

import numpy as np
import pandas as pd

# Set the path to the folder containing JANPA results
nao_folder = Path("/home/nikolenko/work/qm/janpa_files")


def read_populations(input_file):
    """Function to read populations and charges from the JANPA output file."""
    try:
        with open(input_file) as inp:
            lines = [line.strip() for line in inp.readlines()]
            start = lines.index("charge\t population\t  population\tcharge")
            populations = []
            charges = []
            i = 1
            while len(lines[start + i]) > 0:
                line = lines[start + i]
                populations.append([line.split()[0], float(line.split()[2])])
                charges.append(float(line.split()[4]))
                i += 1
        return {"populations": populations, "charges": charges}
    except FileNotFoundError:
        return None


# Function to calculate indexes based on charge states
def calc_indexes(minus, zero, plus):
    """Calculates charge states and handles missing data by filling them with appropriate values."""

    # Если отсутствуют данные для какого-то состояния, заполняем их
    if minus is None and zero is None and plus is not None:
        # Если есть только +1, заполняем недостающие состояния одинаковыми значениями
        minus, zero = plus, plus

    elif minus is None and zero is not None and plus is None:
        # Если есть только нейтральное состояние, остальные будут такими же
        minus, plus = zero, zero

    elif minus is not None and zero is None and plus is None:
        # Если есть только -1, остальные будут такими же
        zero, plus = minus, minus

    elif minus is None and zero is not None and plus is not None:
        # Если есть 0 и +1, -1 заполняем средним значением
        minus = {
            "charges": [
                (zero_charge + plus_charge) / 2
                for zero_charge, plus_charge in zip(zero["charges"], plus["charges"])
            ]
        }

    elif minus is not None and zero is None and plus is not None:
        # Если есть -1 и +1, 0 заполняем средним значением
        zero = {
            "charges": [
                (minus_charge + plus_charge) / 2
                for minus_charge, plus_charge in zip(minus["charges"], plus["charges"])
            ]
        }

    elif minus is not None and zero is not None and plus is None:
        # Если есть -1 и 0, +1 заполняем средним значением
        plus = {
            "charges": [
                (minus_charge + zero_charge) / 2
                for minus_charge, zero_charge in zip(minus["charges"], zero["charges"])
            ]
        }

    return minus["charges"], zero["charges"], plus["charges"]


# Function to process a molecule and extract charges for different states
def process_mol(filename):
    """Processes a molecule and fills missing charge states based on available data."""
    minus_file = filename + "-1.out"
    neutral_file = filename + "0.out"
    plus_file = filename + "1.out"

    minus = read_populations(minus_file)
    zero = read_populations(neutral_file)
    plus = read_populations(plus_file)

    neg_charges, neutral_charges, pos_charges = calc_indexes(minus, zero, plus)

    return neg_charges, neutral_charges, pos_charges


# Function to check valid molecule files (no more exclusions based on missing data)
def check_mol_list(input_folder):
    """Collects all molecules that have at least one charge state file."""
    files = list(input_folder.glob("*_conformation_optimized_*"))
    base_names = {f.stem[:-2] for f in files}
    return base_names


# Function to calculate CDD
def calculate_cdd(row):
    return [
        2 * hc - hfe - hfn
        for hc, hfe, hfn in zip(
            row["hirshfeld_charges"],
            row["hirshfeld_fukui_elec"],
            row["hirshfeld_fukui_neu"],
        )
    ]


# Function to apply log to CDD values
def calculate_log_cdd(cdd_values):
    return [np.log10(abs(value)) if value != 0 else np.nan for value in cdd_values]


# Root Mean Square Error (RMSE) calculation
def rmse(series):
    return np.sqrt(np.mean(series**2))


# Main processing function to extract data and calculate descriptors for all molecules
def process_all_mols(input_folder):
    """Processes all molecules, filling missing charge states and calculating descriptors."""
    files_list = check_mol_list(input_folder)
    combined_data = []

    for file in files_list:
        try:
            neg_charges, neutral_charges, pos_charges = process_mol(
                str(input_folder / file)
            )
            base_name = file.split("_")[0]  # Extract molecule name
            conf_number = int(file.split("_")[3])  # Conformation number
            combined_data.append(
                {
                    "Molecule": base_name,
                    "Conformation": conf_number,
                    "hirshfeld_charges": neg_charges,
                    "hirshfeld_fukui_elec": neutral_charges,
                    "hirshfeld_fukui_neu": pos_charges,
                }
            )
        except ValueError:
            print(f"Broken file: {file}")
            continue

    # Create DataFrame from the collected data
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values(by=["Molecule", "Conformation"])

    # Calculate CDD and Log_CDD
    combined_df["CDD"] = combined_df.apply(calculate_cdd, axis=1)
    combined_df["Log_CDD"] = combined_df["CDD"].apply(calculate_log_cdd)

    # Grouping by molecule and calculating RMSE for each group
    grouped = combined_df.groupby("Molecule")

    rmse_results = []
    for molecule, group in grouped:
        rmse_hirshfeld_charges = rmse(
            pd.Series(np.concatenate(group["hirshfeld_charges"].values))
        )
        rmse_hirshfeld_fukui_elec = rmse(
            pd.Series(np.concatenate(group["hirshfeld_fukui_elec"].values))
        )
        rmse_hirshfeld_fukui_neu = rmse(
            pd.Series(np.concatenate(group["hirshfeld_fukui_neu"].values))
        )
        rmse_results.append(
            {
                "Molecule": molecule,
                "RMSE_hirshfeld_charges": rmse_hirshfeld_charges,
                "RMSE_hirshfeld_fukui_elec": rmse_hirshfeld_fukui_elec,
                "RMSE_hirshfeld_fukui_neu": rmse_hirshfeld_fukui_neu,
            }
        )

    # Create DataFrame with RMSE results
    rmse_df = pd.DataFrame(rmse_results)

    return combined_df, rmse_df


# Process all molecules in the input folder
combined_df, rmse_df = process_all_mols(input_folder=nao_folder)

# Save the results to CSV files
combined_df.to_csv("molecules_combined_data.csv", index=False)
rmse_df.to_csv("molecules_rmse_data.csv", index=False)

print(
    "Processing complete. Data saved to molecules_combined_data.csv and molecules_rmse_data.csv."
)

# FukuiNet: Chebyshev Graph-Based Kolmogorov-Arnold Networks for Molecular Reactivity Prediction

FukuiNet is a machine learning model that combines Chebyshev Graph Convolutions with Kolmogorov-Arnold Networks (KAN) to predict Fukui indices – key descriptors for assessing molecular reactivity.

---

## Project Overview

- **Objective**: Develop an efficient and accurate model to predict molecular reactivity by estimating Fukui indices.
- **Key Features**:
  - **Chebyshev Graph Convolutions** for localized and efficient extraction of graph features.
  - **Kolmogorov-Arnold Networks (KAN)** for advanced non-linear function approximation and feature aggregation.
  - An enriched dataset with additional descriptors such as the Conduction Dual Descriptor (CDD), computed as:

    \[
    \text{CDD} = 2 \times \text{Hirshfeld Charges} - \text{Fukui Index (Electrophilic)} - \text{Fukui Index (Nucleophilic)}
    \]

---

## File Structure

Below is a high-level overview of the most important components of the repository:

```
.
├── LICENSE                # Project license
├── README.md              # Project overview and instructions
├── Makefile               # Convenience commands for training and testing
├── data
│   ├── raw                # Raw datasets (e.g., QM_137k.parquet)
│   ├── processed          # Processed datasets (e.g., QM_137k.pt)
│   └── external           # External resources and pre-trained models
├── fukui_net              # Core source code (model definition, training utilities, etc.)
├── notebooks              # Jupyter notebooks for experiments and analyses
│   ├── 0-preprocessing.ipynb
│   ├── 1-model.ipynb
│   ├── 2-hyperopt.ipynb
│   ├── 5-finetune.ipynb
│   └── analysis.ipynb
├── quantum_data_preparation   # Scripts for quantum chemical data processing and descriptor extraction
├── reports                # Manuscript, figures, and additional analysis reports
├── setup.py, pyproject.toml, environment.yml  # Build and dependency configurations
```

---

## Running the Project

For ease of use and reproducibility, all main experiments and model training are provided as **Jupyter Notebooks** in the `notebooks` directory. We recommend running these notebooks rather than the standalone scripts. Notebooks offer a clear, step-by-step workflow and detailed explanations of the processes.

**Key notebooks include:**

- **0-preprocessing.ipynb**: Data preprocessing and preparation.
- **1-model.ipynb**: Model instantiation, training (with cross-validation), and evaluation.
- **2-hyperopt.ipynb**: Hyperparameter optimization.
- **5-finetune.ipynb**: Model fine-tuning and transfer learning experiments.
- **analysis.ipynb**: Post-training analysis and result visualization.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SergeiNikolenko/fukui_index_prediction.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, use the provided `environment.yml` to create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate fukui_index_prediction
   ```

---

## Usage

1. **Data Preparation**:
   Run the notebook `0-preprocessing.ipynb` to preprocess the raw molecular data and generate the `QM_137k.pt` file in the `data/processed` directory.

2. **Model Training**:
   Use the notebook `1-model.ipynb` for training the FukuiNet model with Chebyshev Graph Convolutions and KAN (Kolmogorov-Arnold Networks). This notebook covers cross-validation and evaluation of the model.

3. **Hyperparameter Optimization**:
   The notebook `2-hyperopt.ipynb` demonstrates the hyperparameter search using Optuna.

4. **Fine-Tuning**:
   The notebook `5-finetune.ipynb` shows how to fine-tune the pre-trained model on a high-quality subset of the data.

5. **Analysis & Visualization**:
   The `analysis.ipynb` notebook and the notebooks in `reports/other` provide visualization and detailed analysis of the results.

---

## Dataset Description

The primary dataset is stored as `QM_137k.parquet` in the `data/raw` directory. An enriched and processed version is available as `QM_137k.pt` in the `data/processed` folder. This dataset is a modified version of the original QM9 dataset and includes additional descriptors—most notably the **Conduction Dual Descriptor (CDD)**, which is computed as:

\[
\text{CDD} = 2 \times \text{Hirshfeld Charges} - \text{Fukui Index (Electrophilic)} - \text{Fukui Index (Nucleophilic)}
\]

This enhanced dataset is designed to improve the accuracy of molecular reactivity predictions.

---

## Model Details

FukuiNet is built upon:
- **Chebyshev Graph Convolutions**: For efficient extraction of both local and global graph features.
- **Kolmogorov-Arnold Networks (KAN)**: Inspired by the Kolmogorov-Arnold representation theorem, these networks serve as non-linear function approximators to aggregate features.
- **Architecture**: Consists of multiple preprocessing layers, Chebyshev convolutional layers, and postprocessing layers, culminating in a final prediction layer.
- **Optimization**: The model is trained using the Lion optimizer with carefully tuned learning rate, weight decay, and step-size schedules.

---

## Contribution

Contributions are welcome! Please open an issue or submit a pull request with improvements, bug fixes, or new ideas.

---

## License

This project is licensed under the MIT License.
```

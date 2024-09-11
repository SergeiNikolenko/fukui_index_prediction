Sure! Here’s a concise README for your project:

---

# FukuiNet: Chebyshev Graph-Based KAN for Molecular Reactivity Prediction

This project introduces FukuiNet, a machine learning model leveraging Chebyshev graph convolutions within a Kernel-based Attention Network (KAN) to predict Fukui indices for assessing molecular reactivity. 

## Project Overview

- **Objective**: To develop an efficient and accurate model for predicting molecular reactivity using advanced graph-based techniques and kernel-based attention mechanisms.
- **Key Features**: Utilizes Chebyshev polynomials and Kernel-based Attention Networks for enhanced performance and speed compared to traditional methods.

## Project Structure

```
├── LICENSE            <- License information
├── Makefile           <- Convenience commands (e.g., `make train`)
├── README.md          <- Project overview and instructions
├── data
│   ├── external       <- Third-party data
│   ├── interim        <- Transformed data
│   ├── processed      <- Final datasets for modeling
│   └── raw            <- Original data
│
├── docs               <- Documentation
│
├── models             <- Trained models and predictions
│
├── notebooks          <- Jupyter notebooks for exploration
│
├── quantum_data_pre   <- Scripts for quantum chemical calculations
│
├── pyproject.toml     <- Project configuration
│
├── references         <- Data dictionaries and manuals
│
├── reports            <- Analysis reports and figures
│
├── requirements.txt   <- Dependencies
│
├── setup.cfg          <- Flake8 configuration
│
└── fukuinet           <- Source code
    ├── __init__.py    <- Makes `fukuinet` a Python module
    ├── config.py      <- Configuration settings
    ├── dataset.py     <- Data handling scripts
    ├── features.py    <- Feature engineering
    ├── modeling       
    │   ├── __init__.py 
    │   ├── predict.py <- Model inference code
    │   └── train.py   <- Model training code
    └── plots.py       <- Visualization scripts
```

## Installation

1. Clone the repository: `git clone https://github.com/your-repo/fukuiNet.git`
2. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Prepare data using scripts in `quantum_data_pre`.
2. Train the model using `modeling/train.py`.
3. Evaluate predictions with `modeling/predict.py`.
4. Visualize results using `plots.py`.

## Dataset

The dataset `QM_137k.parquet` is an enhanced version of QM9 with additional features for reactivity predictions, including the Conduction Dual Descriptor (CDD).

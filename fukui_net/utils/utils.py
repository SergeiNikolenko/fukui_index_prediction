import os
import csv
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import DataLoader as GeoDataLoader
from tqdm import tqdm


class MoleculeDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling molecule datasets.

    Attributes:
        dataset (Dataset): The dataset containing molecular data.
        batch_size (int): Number of samples per batch.
        val_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.
        num_workers (int): Number of worker processes for data loading.
        _train_dataset (Subset): Training subset of the dataset.
        _val_dataset (Subset): Validation subset of the dataset.
        _test_dataset (Subset): Test subset of the dataset.
    """
    
    def __init__(self, dataset, batch_size=128, val_split=0.1, test_split=0.2, num_workers=1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
    
    def setup(self, stage=None):
        """
        Splits the dataset into training, validation, and test subsets.

        Args:
            stage (str): Stage of the experiment ('fit', 'validate', 'test', etc.).
        """
        indices = list(range(len(self.dataset)))
        train_val_indices, test_indices = train_test_split(indices, test_size=self.test_split, random_state=42)
        train_indices, val_indices = train_test_split(train_val_indices, test_size=self.val_split / (1 - self.test_split), random_state=42)
        
        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)
    
    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        """
        Returns the test DataLoader.

        Returns:
            DataLoader: DataLoader for test data.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def evaluate_model(model, data_module):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The trained model to be evaluated.
        data_module (MoleculeDataModule): The data module providing the test data.

    Returns:
        None: Prints the evaluation metrics.
    """
    test_dl = data_module.test_dataloader()
    model.eval()  
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in test_dl:
            y_hat = model(batch.x, batch.edge_index, batch.edge_attr)
            all_pred.extend(y_hat.cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())

    all_pred, all_true = np.array(all_pred), np.array(all_true)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)

    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test R²: {r2:.4f}')


def evaluate_model_full(model, dataset, batch_size, num_workers):
    dataloader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(model.device)
            y_hat = model(batch.x, batch.edge_index, batch.edge_attr)
            all_pred.extend(y_hat.cpu().numpy())
            all_true.extend(batch.y.cpu().numpy())

    all_pred, all_true = np.array(all_pred), np.array(all_true)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    r2 = r2_score(all_true, all_pred)

    print(f'Total RMSE: {rmse:.4f}')
    print(f'Total R²: {r2:.4f}')


def create_hyperopt_dir(base_dir='hyperopt_'):
    """
    Creates a new directory for storing hyperparameter optimization results.

    Args:
        base_dir (str): Base name for the directory.

    Returns:
        str: The path to the newly created directory.
    """
    idx = 1
    while True:
        dir_name = f"{base_dir}{idx}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        idx += 1

def initialize_cuda():
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        print("cuda", torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()
    else:
        print("CUDA is not available.")


def save_trial_to_csv(trial, hyperopt_dir, trial_value):
    """
    Saves the results of a hyperparameter optimization trial to a CSV file.

    Args:
        trial: Optuna trial object containing the trial results.
        hyperopt_dir (str): Directory where the results should be saved.
        trial_value: The value of the objective function for the trial.

    Returns:
        None: Writes the trial data to a CSV file.
    """
    csv_path = os.path.join(hyperopt_dir, 'optuna_results.csv')
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.path.getsize(csv_path) == 0:  
            headers = ['Trial'] + ['Value'] + [key for key in trial.params.keys()]
            writer.writerow(headers)
        row = [trial.number] + [trial_value] + list(trial.params.values())
        writer.writerow(row)






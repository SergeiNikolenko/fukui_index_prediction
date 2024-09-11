import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from torch_geometric.loader import DataLoader

from fukui_net.utils.efficient_kan import KAN, KANLinear


class MoleculeModel(pl.LightningModule):
    """
    A PyTorch Lightning module for training, validating, and testing a model for molecular property prediction.

    Attributes:
        model_backbone (nn.Module): The backbone model architecture.
        optimizer_class (torch.optim.Optimizer): Optimizer class for training.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) coefficient.
        step_size (int): Step size for the learning rate scheduler.
        gamma (float): Multiplicative factor for learning rate decay.
        batch_size (int): Number of samples per batch.
        metric (function): Metric function used to compute the loss.
    """
    
    def __init__(self, model_backbone, optimizer_class, learning_rate, weight_decay, step_size, gamma, batch_size, metric='rmse'):
        super(MoleculeModel, self).__init__()
        self.model_backbone = model_backbone
        self.batch_size = batch_size
        self.metric = self.get_metric(metric)
        self.save_hyperparameters(ignore=['model_backbone'])

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Node features (atom features).
            edge_index (torch.Tensor): Graph edge indices.
            edge_attr (torch.Tensor): Edge features (bond features).

        Returns:
            torch.Tensor: Model predictions.
        """
        return self.model_backbone(x, edge_index, edge_attr)
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            tuple: A tuple containing the optimizer and scheduler.
        """
        optimizer = self.hparams.optimizer_class(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def on_train_start(self):
        """
        Hook to log activations at the start of training.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(self.log_activations_hook(name))

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch: The input batch for training.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss.
        """
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        loss = self.metric(batch.y, y_hat)
        self.log('train_loss', loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch: The input batch for validation.
            batch_idx (int): The index of the batch.
        """
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        val_loss = self.metric(batch.y, y_hat)
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.

        Args:
            batch: The input batch for testing.
            batch_idx (int): The index of the batch.

        Returns:
            list: A list of dictionaries containing SMILES strings, predictions, and true values.
        """
        y_hat = self(batch.x, batch.edge_index, batch.edge_attr)
        preds_np = y_hat.detach().cpu().numpy()
        true_values_np = batch.y.detach().cpu().numpy()

        data = []
        start_idx = 0
        for i, num_atoms in enumerate(batch.ptr[:-1]): 
            end_idx = batch.ptr[i+1].item()
            molecule_preds = preds_np[start_idx:end_idx]
            molecule_true_values = true_values_np[start_idx:end_idx]

            data.append({
                'smiles': batch.smiles[i],
                'predictions': molecule_preds,
                'true_values': molecule_true_values
            })

            start_idx = end_idx
        return data

    def on_test_epoch_end(self, outputs):
        """
        Hook to process the outputs at the end of the test epoch.

        Args:
            outputs: The outputs of all test steps.

        Returns:
            pd.DataFrame: DataFrame containing all test results.
        """
        all_data = [item for batch_data in outputs for item in batch_data]
        self.df_results = pd.DataFrame(all_data)

        all_predictions = np.concatenate(self.df_results['predictions'].values)
        all_true_values = np.concatenate(self.df_results['true_values'].values)

        rmse = np.sqrt(mean_squared_error(all_true_values, all_predictions))
        mse = mean_squared_error(all_true_values, all_predictions)
        r2 = r2_score(all_true_values, all_predictions)
        mae = mean_absolute_error(all_true_values, all_predictions)

        self.log('test_rmse', rmse)
        self.log('test_mse', mse)
        self.log('test_r2', r2)
        self.log('test_mae', mae)

        print(f'Test RMSE: {rmse:.4f}')
        print(f'Test MSE: {mse:.4f}')
        print(f'Test R²: {r2:.4f}')
        print(f'Test MAE: {mae:.4f}')

        return self.df_results
    
    def on_epoch_end(self):
        """
        Hook to log parameter histograms at the end of each epoch.
        """
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)
            
    def log_activations_hook(self, layer_name):
        """
        Hook to log activations for a given layer.

        Args:
            layer_name (str): The name of the layer.

        Returns:
            function: A hook function.
        """
        def hook(module, input, output):
            if self.logger: 
                self.logger.experiment.add_histogram(f"{layer_name}_activations", output, self.current_epoch)
        return hook

    def get_metric(self, metric_name):
        """
        Returns the appropriate metric function based on the provided name.

        Args:
            metric_name (str): The name of the metric ('mse' or 'rmse').

        Returns:
            function: The corresponding metric function.

        Raises:
            ValueError: If an unknown metric name is provided.
        """
        if metric_name == 'mse':
            def mse(y_true, y_pred):
                return F.mse_loss(y_pred, y_true)
            return mse

        elif metric_name == 'rmse':
            def rmse(y_true, y_pred):
                return torch.sqrt(F.mse_loss(y_pred, y_true))
            return rmse

        else:
            raise ValueError(f"Unknown metric name: {metric_name}")

class AtomEdgeInteraction(nn.Module):
    """
    A neural network module to model the interaction between atoms and edges in a molecular graph.

    Attributes:
        in_features (int): Number of input features for atoms.
        edge_features (int): Number of input features for edges.
        out_features (int): Number of output features.
        edge_importance (float): Scaling factor for edge features.
        interaction (nn.Linear): Linear layer to process combined atom and edge features.
    """
    
    def __init__(self, in_features, edge_features, out_features, edge_importance=1.0, dropout_rate=0.1, use_batch_norm=True):
        super(AtomEdgeInteraction, self).__init__()
        self.edge_importance = edge_importance
        self.interaction = KANLinear(in_features + edge_features, out_features)
        self.activation = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()


    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the AtomEdgeInteraction module.

        Args:
            x (torch.Tensor): Node features (atom features).
            edge_index (torch.Tensor): Graph edge indices.
            edge_attr (torch.Tensor): Edge features (bond features).

        Returns:
            torch.Tensor: Updated node features after interaction with edge features.
        """
        row, col = edge_index
        edge_features = edge_attr * self.edge_importance
        atom_features = x[row]
        combined_features = torch.cat([atom_features, edge_features], dim=-1)
        updated_features = self.interaction(combined_features)
        updated_features = self.activation(updated_features)
        updated_features = self.batch_norm(updated_features)
        updated_features = self.dropout(updated_features)
        residual_features = self.residual(x)
        x = scatter_mean(updated_features, col, dim=0, dim_size=x.size(0))
        return x + residual_features



class Model(nn.Module):
    """
    A neural network model designed for molecular graph data, leveraging KANLinear layers, 
    Chebyshev convolutions, and various preprocessing and postprocessing layers. 

    Attributes:
        atom_preprocess (nn.ModuleList): List of preprocessing layers for node features.
        cheb_convolutions (nn.ModuleList): List of Chebyshev convolutional layers.
        postprocess (nn.ModuleList): List of postprocessing layers after the convolutional layers.
        output_layer (nn.Module): Final linear layer to produce the output.

    Args:
        atom_in_features (int): Number of input features for atoms.
        edge_attr_dim (int): Number of input features for edges.
        preprocess_hidden_features (list of int): List of hidden layer sizes for the preprocessing layers.
        cheb_hidden_features (list of int): List of hidden layer sizes for the Chebyshev convolution layers.
        K (list of int): List of polynomial orders for the Chebyshev convolutions.
        cheb_normalizations (list of str): List of normalization techniques for Chebyshev convolutions.
        dropout_rates (list of float): List of dropout rates for each layer.
        activation_fns (list of nn.Module): List of activation functions used in the layers.
        use_batch_norm (list of bool): Flags indicating whether to use batch normalization in each layer.
        postprocess_hidden_features (list of int): List of hidden layer sizes for the postprocessing layers.
        out_features (int): Number of output features.
    """

    def __init__(self, atom_in_features, edge_attr_dim, preprocess_hidden_features, cheb_hidden_features, K, cheb_normalizations, dropout_rates, activation_fns, use_batch_norm, postprocess_hidden_features, out_features):
        super(Model, self).__init__()

        self.atom_preprocess = nn.ModuleList([AtomEdgeInteraction(atom_in_features, edge_attr_dim, preprocess_hidden_features[0], dropout_rate=dropout_rates[0], use_batch_norm=use_batch_norm[0])])
        for i in range(1, len(preprocess_hidden_features)):
            layer = nn.Sequential(
                KANLinear(preprocess_hidden_features[i-1], preprocess_hidden_features[i]),
                nn.BatchNorm1d(preprocess_hidden_features[i]) if use_batch_norm[i] else nn.Identity(),
                activation_fns[i](),
                nn.Dropout(dropout_rates[i])
            )
            self.atom_preprocess.append(layer)

        self.cheb_convolutions = nn.ModuleList()
        in_channels = preprocess_hidden_features[-1]
        for i in range(len(cheb_hidden_features)):
            self.cheb_convolutions.append(ChebConv(in_channels, cheb_hidden_features[i], K[i], normalization=cheb_normalizations[i]))
            in_channels = cheb_hidden_features[i]

        self.postprocess = nn.ModuleList()
        for i in range(len(postprocess_hidden_features)):
            layer = nn.Sequential(
                KANLinear(cheb_hidden_features[i-1] if i > 0 else cheb_hidden_features[-1], postprocess_hidden_features[i]),
                nn.BatchNorm1d(postprocess_hidden_features[i]) if use_batch_norm[len(preprocess_hidden_features) + i] else nn.Identity(),
                activation_fns[len(preprocess_hidden_features) + i](),
                nn.Dropout(dropout_rates[len(preprocess_hidden_features) + i])
            )
            self.postprocess.append(layer)

        self.output_layer = KANLinear(postprocess_hidden_features[-1], out_features)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_preprocess[0](x, edge_index, edge_attr)
        for layer in self.atom_preprocess[1:]:
            x = layer(x)

        for conv in self.cheb_convolutions:
            x = F.relu(conv(x, edge_index))

        for layer in self.postprocess:
            x = layer(x)

        return self.output_layer(x).squeeze(-1)


class CrossValDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=128, num_workers=1):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

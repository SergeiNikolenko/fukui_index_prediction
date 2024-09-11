# %% [markdown]
# ## Import Libraries and Set Up Environment

# %%
# Import necessary libraries
import warnings
import torch
import numpy as np
import random
from torch.utils.data import Subset
from sklearn.model_selection import KFold
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer
from pytorch_lightning import Trainer
from fukui_net.utils.utils import MoleculeDataModule, initialize_cuda, evaluate_model_full
from fukui_net.utils.train import MoleculeModel, Model, CrossValDataModule
from lion_pytorch import Lion

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Initialize CUDA settings
initialize_cuda()

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.connectors.data_connector")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm")

# Ensure deterministic behavior in CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% [markdown]
# ### Load Dataset and Define Model Parameters

# %%
# Load dataset
dataset = torch.load(f'../data/processed/QM_137k.pt')
in_features = dataset[0].x.shape[1]
out_features = 1
edge_attr_dim = dataset[0].edge_attr.shape[1]

# Data module settings
batch_size = 1024 
num_workers = 8  

# Define the model architecture parameters
preprocess_hidden_features = [128] * 9
postprocess_hidden_features = [128, 128]
cheb_hidden_features = [128, 128]
K = [10, 16]
cheb_normalization = ['sym', 'sym']

dropout_rates = [0.0] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
activation_fns = [torch.nn.PReLU] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))
use_batch_norm = [True] * (len(preprocess_hidden_features) + len(postprocess_hidden_features))

# Optimizer settings
optimizer_class = Lion
learning_rate = 2.2e-5
weight_decay = 3e-5
step_size = 80
gamma = 0.2
metric = 'rmse'

# %% [markdown]
# ## Instantiate Model Backbone

# %%
# Instantiate model backbone
backbone = Model(
    atom_in_features=in_features,
    edge_attr_dim=edge_attr_dim,
    preprocess_hidden_features=preprocess_hidden_features,
    cheb_hidden_features=cheb_hidden_features,
    K=K,
    cheb_normalizations=cheb_normalization,
    dropout_rates=dropout_rates,
    activation_fns=activation_fns,
    use_batch_norm=use_batch_norm,
    postprocess_hidden_features=postprocess_hidden_features,
    out_features=out_features
)

# %%
print("Model Backbone Architecture:\n", backbone)

# %% [markdown]
# ## K-Fold cross-validation setup

# %%
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
data_indices = list(range(len(dataset)))

fold_results = []
best_val_loss = float('inf')
best_model = None

# %% [markdown]
# ## Training Loop for Cross-Validation

# %%
# Perform K-Fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(data_indices)):
    print(f"Fold {fold+1}/{n_splits}")

    # Create data subsets for this fold
    train_subset = Subset(dataset, train_index)
    val_subset = Subset(dataset, val_index)
    data_module = CrossValDataModule(train_subset, val_subset, batch_size=batch_size, num_workers=num_workers)

    # Initialize model for this fold
    model = MoleculeModel(
        model_backbone=backbone,
        optimizer_class=optimizer_class,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        batch_size=batch_size,
        metric=metric
    )

    # Callbacks for model checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True, dirpath='.', filename='best_model')
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    timer = Timer()
    logger = pl.loggers.TensorBoardLogger('../reports/tb_logs', name=f'KAN_fold_{fold+1}')

    # Trainer configuration
    trainer = Trainer(
        max_epochs=100,
        enable_checkpointing=True,
        callbacks=[early_stop_callback, timer],
        enable_progress_bar=False,
        logger=logger,
        accelerator='gpu',
        devices=1
    )

    # Train the model
    trainer.fit(model, data_module)

    # Get validation loss
    val_loss = trainer.callback_metrics["val_loss"].item()
    fold_results.append(val_loss)

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model


    seconds = timer.time_elapsed()
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    print(f"Training time for fold {fold+1}: {h}:{m:02d}:{s:02d}")

# %% [markdown]
# ## Cross-Validation Results and Final Model Save

# %%
# Print cross-validation results
print("Cross-validation results:")
for i, val_loss in enumerate(fold_results):
    print(f"Fold {i+1}: Validation error (RMSE) = {val_loss:.4f}")

mean_val_loss = sum(fold_results) / len(fold_results)
print(f"Average validation error (RMSE): {mean_val_loss:.4f}")

# Save the final best model

torch.save(model, "../model/final_best_model.ckpt")
print(f"Final best model saved!")

# %% [markdown]
# ## Evaluate the Final Model

# %%
# Evaluate the final model on the entire dataset
evaluate_model_full(best_model, dataset, batch_size, num_workers)



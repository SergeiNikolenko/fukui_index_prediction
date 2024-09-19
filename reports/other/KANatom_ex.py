# %% [markdown]
# скипатом у реюбят
# и без атомов
#
#

import pytorch_lightning as pl
import torch
import torch.nn as nn
from lion_pytorch import Lion
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    print("cuda", torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
else:
    print("CUDA is not available.")

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pytorch_lightning.trainer.connectors.data_connector",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="lightning_fabric.plugins.environments.slurm"
)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision("medium")

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fukui_index_prediction.FukuiNet.utils.add_skipatom import (
    add_skipatom_features_to_dataset,
)
from fukui_index_prediction.FukuiNet.utils.train import MoleculeModel
from fukui_index_prediction.FukuiNet.utils.utils import (
    MoleculeDataModule,
    create_hyperopt_dir,
    evaluate_model,
    save_trial_to_csv,
)

# %%
dataset = torch.load(f"../data/QM_137k.pt")

# %%
# dataset = add_skipatom_features_to_dataset(dataset, min_count=2e7, top_n=4, device='cpu', progress_bar=True, scaler=StandardScaler())

import pytorch_lightning as pl

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATv2Conv, TransformerConv
from torch_scatter import scatter_mean

from fukui_index_prediction.FukuiNet.utils.efficient_kan import KAN, KANLinear
from fukui_index_prediction.FukuiNet.utils.train import MoleculeModel


class AtomEdgeInteraction(nn.Module):
    def __init__(
        self,
        in_features,
        edge_features,
        out_features,
        edge_importance=1.0,
        dropout_rate=0.1,
        use_batch_norm=True,
    ):
        super().__init__()
        self.edge_importance = edge_importance
        self.interaction = KANLinear(in_features + edge_features, out_features)
        self.activation = nn.ReLU()
        self.batch_norm = (
            nn.BatchNorm1d(out_features) if use_batch_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr):
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


# Теперь обновим остальную часть модели, чтобы интегрировать новый класс
class Model(nn.Module):
    def __init__(
        self,
        atom_in_features,
        edge_attr_dim,
        preprocess_hidden_features,
        cheb_hidden_features,
        K,
        cheb_normalizations,
        dropout_rates,
        activation_fns,
        use_batch_norm,
        postprocess_hidden_features,
        out_features,
    ):
        super().__init__()

        self.atom_preprocess = nn.ModuleList(
            [
                AtomEdgeInteraction(
                    atom_in_features,
                    edge_attr_dim,
                    preprocess_hidden_features[0],
                    dropout_rate=dropout_rates[0],
                    use_batch_norm=use_batch_norm[0],
                )
            ]
        )
        for i in range(1, len(preprocess_hidden_features)):
            layer = nn.Sequential(
                KANLinear(
                    preprocess_hidden_features[i - 1], preprocess_hidden_features[i]
                ),
                nn.BatchNorm1d(preprocess_hidden_features[i])
                if use_batch_norm[i]
                else nn.Identity(),
                activation_fns[i](),
                nn.Dropout(dropout_rates[i]),
            )
            self.atom_preprocess.append(layer)

        self.cheb_convolutions = nn.ModuleList()
        in_channels = preprocess_hidden_features[-1]
        for i in range(len(cheb_hidden_features)):
            self.cheb_convolutions.append(
                ChebConv(
                    in_channels,
                    cheb_hidden_features[i],
                    K[i],
                    normalization=cheb_normalizations[i],
                )
            )
            in_channels = cheb_hidden_features[i]

        self.postprocess = nn.ModuleList()
        for i in range(len(postprocess_hidden_features)):
            layer = nn.Sequential(
                KANLinear(
                    cheb_hidden_features[i - 1] if i > 0 else cheb_hidden_features[-1],
                    postprocess_hidden_features[i],
                ),
                nn.BatchNorm1d(postprocess_hidden_features[i])
                if use_batch_norm[len(preprocess_hidden_features) + i]
                else nn.Identity(),
                activation_fns[len(preprocess_hidden_features) + i](),
                nn.Dropout(dropout_rates[len(preprocess_hidden_features) + i]),
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


# %%
in_features = dataset[0].x.shape[1]
out_features = 1
edge_attr_dim = dataset[0].edge_attr.shape[1]

batch_size = 1024
num_workers = 8

data_module = MoleculeDataModule(
    dataset, batch_size=batch_size, num_workers=num_workers
)


preprocess_hidden_features = [128] * 9
postprocess_hidden_features = [128, 128]
cheb_hidden_features = [128, 128]
K = [10, 16]
cheb_normalization = ["sym", "sym"]

dropout_rates = [0.0] * (
    len(preprocess_hidden_features) + len(postprocess_hidden_features)
)
activation_fns = [nn.PReLU] * (
    len(preprocess_hidden_features) + len(postprocess_hidden_features)
)
use_batch_norm = [True] * (
    len(preprocess_hidden_features) + len(postprocess_hidden_features)
)

# Initialize the backbone with its specific parameters


optimizer_class = Lion
learning_rate = 2.2e-5
weight_decay = 3e-5
step_size = 80
gamma = 0.2
batch_size = 1024
metric = "rmse"


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
    out_features=out_features,
)

model = MoleculeModel(
    model_backbone=backbone,
    optimizer_class=optimizer_class,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    step_size=step_size,
    gamma=gamma,
    batch_size=batch_size,
    metric=metric,
)

print("Model:\n", model)


from pytorch_lightning import Trainer, callbacks

checkpoint_callback = callbacks.ModelCheckpoint(
    monitor="val_loss", mode="min", save_top_k=1, verbose=True
)
early_stop_callback = callbacks.EarlyStopping(
    monitor="val_loss", patience=5, verbose=True, mode="min"
)
timer = callbacks.Timer()
logger = pl.loggers.TensorBoardLogger("tb_logs", name="ChebConv")

trainer = Trainer(
    max_epochs=100,
    enable_checkpointing=False,
    callbacks=[early_stop_callback, timer],
    enable_progress_bar=False,
    logger=logger,
    accelerator="gpu",
    devices=1,
)

trainer.fit(model, data_module)

# %%
seconds = timer.time_elapsed()
h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)

print(f"Время обучения: {h}:{m:02d}:{s:02d}")

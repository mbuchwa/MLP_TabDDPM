import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchmetrics import Accuracy
from torchmetrics.regression import MeanSquaredError


class MLPModel(pl.LightningModule):
    def __init__(self, config, tune_config=None):
        super().__init__()
        self.config = config
        self.mlp_type = config['mlp_type']
        self.lr = tune_config['lr'] if tune_config is not None else config['train']['main']['lr']
        self.hidden_units = config['model_params']['rtdl_params']['d_layers']
        # new PL attributes:
        if self.mlp_type == 'binary_classification':
            self.train_metric = Accuracy(task="binary")
            self.valid_metric = Accuracy(task="binary")
            self.test_metric = Accuracy(task="binary")
            self.loss = nn.BCELoss()
            self.out = nn.Sigmoid()
        elif self.mlp_type == 'mc_classification':
            self.train_metric = Accuracy(task="multiclass")
            self.valid_metric = Accuracy(task="multiclass")
            self.test_metric = Accuracy(task="multiclass")
            self.loss = nn.CrossEntropyLoss()
            self.out = nn.Softmax(dim=2)
        elif self.mlp_type == 'regression':
            self.metric = MeanSquaredError()

        else:
            raise ValueError(f'Type {self.mlp_type} not implemented!')

        all_layers = []
        input_dim = config['model_params']['d_in']

        if tune_config is None:
            # Model similar to previous section:
            for hidden_unit in self.hidden_units:
                layer = nn.Linear(input_dim, hidden_unit)
                all_layers.append(layer)
                all_layers.append(nn.LeakyReLU())
                input_dim = hidden_unit
            all_layers.append(nn.Linear(self.hidden_units[-1], 1))

        else:
            all_layers.append(nn.Linear(input_dim, tune_config['num_units']))
            all_layers.append(nn.LeakyReLU())
            for _ in range(1, tune_config['num_layers']):
                layer = nn.Linear(tune_config['num_units'], tune_config['num_units'])
                all_layers.append(layer)
                all_layers.append(nn.LeakyReLU())
            all_layers.append(nn.Linear(tune_config['num_units'], 1))


        if self.mlp_type != 'regression':
            all_layers.append(self.out)
        self.model = nn.Sequential(*all_layers)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mlp_type in ['binary_classification', 'regression']:
            y = y.to(torch.float)
        out = self(x)
        loss = self.loss(out, y.unsqueeze(dim=-1))
        if self.mlp_type == 'mc_classification':
            preds = torch.argmax(out, dim=1)
            self.train_metric.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.mlp_type in ['binary_classification', 'regression']:
            y = y.to(torch.float)
        out = self(x)
        loss = self.loss(out, y.unsqueeze(dim=-1))
        if self.mlp_type == 'mc_classification':
            preds = torch.argmax(out, dim=1)
            self.valid_metric.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.mlp_type in ['binary_classification', 'regression']:
            y = y.to(torch.float)
        out = self(x)
        loss = self.loss(out, y.unsqueeze(dim=-1))
        preds = torch.argmax(out, dim=1)
        if self.mlp_type != 'regression':
            self.test_metric.update(preds, y)
            self.log("test_metric", self.test_metric.compute(), prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

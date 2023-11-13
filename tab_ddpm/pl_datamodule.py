import pytorch_lightning as pl
from scripts.utils_train import make_dataset, get_model_params
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, raw_config, tune_config=None, change_val=False):
        super().__init__()
        self.raw_config = raw_config
        self.batch_size = tune_config['batch_size'] if tune_config is not None else raw_config['train']['main']['batch_size']
        self.change_val = change_val

    def setup(self, stage=None):
        """load dataset"""
        dataset, model_params, K = get_model_params(self.raw_config['parent_dir'],
                                                    self.raw_config['real_data_path'],
                                                    self.raw_config['model_params'],
                                                    self.raw_config['train']['T'],
                                                    0, self.change_val)
        if dataset.X_cat is not None:
            if dataset.X_num is not None:
                self.X_train = torch.from_numpy(np.concatenate([dataset.X_num['train'], dataset.X_cat['train']], axis=1)).float()
                self.X_val = torch.from_numpy(np.concatenate([dataset.X_num['val'], dataset.X_cat['val']], axis=1)).float()
                self.X_test = torch.from_numpy(np.concatenate([dataset.X_num['test'], dataset.X_cat['test']], axis=1)).float()
            else:
                self.X_train = torch.from_numpy(dataset.X_cat['train']).float()
                self.X_val = torch.from_numpy(dataset.X_cat['val']).float()
                self.X_test = torch.from_numpy(dataset.X_cat['test']).float()
        else:
            self.X_train = torch.from_numpy(dataset.X_num['train']).float()
            self.X_val = torch.from_numpy(dataset.X_num['val']).float()
            self.X_test = torch.from_numpy(dataset.X_num['test']).float()

        self.Y_train = torch.from_numpy(dataset.y['train'])
        self.Y_val = torch.from_numpy(dataset.y['val'])
        self.Y_test = torch.from_numpy(dataset.y['test'])

    def train_dataloader(self):
        train_dataset = TensorDataset(self.X_train, self.Y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = TensorDataset(self.X_val, self.Y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
        return val_loader

    def test_dataloader(self):
        test_dataset = TensorDataset(self.X_test, self.Y_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
        return test_loader

import tomli
import shutil
import os
import argparse
import zero
import lib
from scripts.utils_train import make_dataset, get_model_params

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import argparse
from pathlib import Path
import sys
import warnings
from tab_ddpm.pl_datamodule import CustomDataModule
from plmodel import TabDDPM


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


sys.path.append(str(Path.cwd()))


def main(raw_config, args):
    """load dataset"""
    dataset, model_params, K = get_model_params(raw_config['parent_dir'],
                                                raw_config['real_data_path'],
                                                raw_config['model_params'],
                                                raw_config['train']['T'],
                                                0, args.change_val)

    datamodule = CustomDataModule(raw_config, change_val=args.change_val)
    model = TabDDPM(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            K=K,
            num_numerical_features=raw_config['num_numerical_features'],
            dataset=args.dataset,
            use_mlp_classification_regression=True,
    )
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./logs/")
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=tb_logger,
                         callbacks=[LearningRateMonitor(logging_interval='epoch'),
                                    EarlyStopping(monitor="val_loss", patience=100, mode="min")
                                    ],
                         max_epochs=500,
                         check_val_every_n_epoch=10)

    trainer.fit(model, datamodule=datamodule)


def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='churn2')
    parser.add_argument('--model', default='ddpm_mlp_best')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--change_val', action='store_true', default=False)

    args = parser.parse_args()
    config_path = f'exp/{args.dataset}/{args.model}/config.toml'
    raw_config = lib.load_config(config_path)

    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), config_path)

    main(raw_config, args)
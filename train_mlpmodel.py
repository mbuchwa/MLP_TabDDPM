import tomli
import shutil
import os
import zero
import lib
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
import argparse
from pathlib import Path
import sys
import warnings

from tab_ddpm.mlp_model import MLPModel
from tab_ddpm.pl_datamodule import CustomDataModule

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(str(Path.cwd()))


def main(raw_config):
    datamodule = CustomDataModule(raw_config, change_val=args.change_val)
    model = MLPModel(raw_config)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./mlp_logs/")
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=tb_logger,
                         callbacks=[LearningRateMonitor(logging_interval='epoch'),
                                    EarlyStopping(monitor="val_loss", patience=1000, mode="min"),
                                    ModelCheckpoint(dirpath=raw_config['parent_dir'])
                                    ],
                         max_epochs=raw_config['train']['main']['epochs'],
                         check_val_every_n_epoch=10)

    trainer.fit(model, datamodule)


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
    parser.add_argument('--config', metavar='FILE', default='exp/churn2/mlp/config.toml')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--change_val', action='store_true', default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)

    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    main(raw_config)
import torch
from scripts.utils_train import *
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
from tab_ddpm import GaussianMultinomialDiffusion
from tab_ddpm import load_config


class TabDDPM(pl.LightningModule):
    def __init__(self,
                 lr,
                 weight_decay,
                 K=0,
                 num_numerical_features=0,
                 model_type='mlp',
                 model_params=None,
                 num_timesteps=1000,
                 gaussian_loss_type='mse',
                 scheduler='cosine',
                 steps=1000,
                 log_every=100,
                 print_every=500,
                 ema_every=1000,
                 batch_size=1028,
                 device=torch.device('cuda:0'),
                 dataset='diabetes',
                 use_mlp_classification_regression=True,
                 ):
        super().__init__()

        model = get_model(
            model_type,
            model_params
        )
        model.to(device)

        if use_mlp_classification_regression:
            self.mlp_config = load_config(f'exp/{dataset}/mlp/config.toml')
        else:
            self.mlp_config = None

        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=model,
            gaussian_loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            scheduler=scheduler,
            dataset=dataset,
            mlp_config=self.mlp_config
        )
        self.diffusion.to(device)
        self.diffusion.train()

        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = log_every
        self.print_every = print_every
        self.ema_every = ema_every

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        for k in out_dict:
            out_dict[k] = out_dict[k].long()
        self.optimizer.zero_grad()
        loss_multi, loss_gauss, loss_mlp = self.diffusion.mixed_loss(x, out_dict, self.mlp_config['mlp_type'])
        loss = loss_multi + loss_gauss
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss_multi, loss_gauss, loss_mlp

    def training_step(self, batch, batch_idx):
        x, out_dict = batch
        out_dict = {'y': out_dict}
        batch_loss_multi, batch_loss_gauss, batch_loss_mlp = self._run_step(x, out_dict)

        self._anneal_lr(batch_idx)
        curr_count = len(x)
        curr_loss_multi = batch_loss_multi * curr_count
        curr_loss_gauss = batch_loss_gauss * curr_count
        curr_loss_mlp = batch_loss_mlp * curr_count
        loss = curr_loss_multi + curr_loss_gauss + curr_loss_mlp
        self.log('train_loss', loss)

        update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, out_dict = batch
        out_dict = {'y': out_dict}
        loss_multi, loss_gauss, loss_mlp = self.diffusion.mixed_loss(x, out_dict, self.mlp_config['mlp_type'])
        curr_count = len(x)
        curr_loss_multi = loss_multi * curr_count
        curr_loss_gauss = loss_gauss * curr_count
        curr_loss_mlp = loss_mlp * curr_count
        loss = curr_loss_multi + curr_loss_gauss + curr_loss_mlp

        self.log('val_loss', loss)

        return {'val_loss': loss}

    def forward(self, x, out_dict):
        for k in out_dict:
            out_dict[k] = out_dict[k].long()
        loss_multi, loss_gauss, loss_mlp = self.diffusion.mixed_loss(x, out_dict, self.mlp_config['mlp_type'])
        curr_count = len(x)
        curr_loss_multi = loss_multi * curr_count
        curr_loss_gauss = loss_gauss * curr_count
        curr_loss_mlp = loss_mlp * curr_count
        loss = curr_loss_multi + curr_loss_gauss + curr_loss_mlp
        return {'test_loss': loss}

    def configure_optimizers(self):
        return self.optimizer


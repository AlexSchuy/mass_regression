import time
from collections import OrderedDict
from typing import Callable, List

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor


class DNN(pl.LightningModule):
    def __init__(self, cr: float, cs: List[int], input_dim: int, output_dim: int, optimizer_cfg: dict, scheduler_cfg: dict, criterion_cfg: dict, output_mean: Tensor, output_std: Tensor, target_mean: Tensor, target_std: Tensor, dropout_prob: float = 0.5, steps_per_epoch: int = None):
        super().__init__()
        cs = [int(cr * x) for x in cs]
        self.save_hyperparameters()

        self.optimizer_factory = hydra.utils.instantiate(optimizer_cfg)
        self.steps_per_epoch = steps_per_epoch
        self.scheduler_cfg = scheduler_cfg
        self.lr = None

        self.register_buffer('output_mean', output_mean)
        self.register_buffer('output_std', output_std)
        self.register_buffer('target_mean', target_mean)
        self.register_buffer('target_std', target_std)

        self.criterion = hydra.utils.instantiate(
            criterion_cfg, output_mean=output_mean, output_std=output_std, target_mean=target_mean, target_std=target_std)

        self.input_stage = nn.Sequential(
            nn.Linear(input_dim, cs[0]), nn.ReLU())

        hidden_modules = []
        for i in range(1, len(cs)):
            hidden_modules.append(nn.Linear(cs[i-1], cs[i]))
            hidden_modules.append(nn.ReLU())
            hidden_modules.append(nn.Dropout(p=dropout_prob))
        self.hidden_stage = nn.Sequential(*hidden_modules)

        self.output_stage = nn.Linear(cs[-1], output_dim)

    def forward(self, x):
        x1 = self.input_stage(x)

        x2 = self.hidden_stage(x1)

        out = self.output_stage(x2)

        return out

    def configure_optimizers(self):
        if self.lr is not None:
            optimizer = self.optimizer_factory(self.parameters(), lr=self.lr)
        else:
            optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_cfg is not None:
            if self.scheduler_cfg._target_ == 'torch.optim.lr_scheduler.OneCycleLR':
                if self.lr is None:
                    return optimizer
                scheduler = hydra.utils.instantiate(
                    self.scheduler_cfg, optimizer=optimizer, steps_per_epoch=self.steps_per_epoch, max_lr=self.lr)
                scheduler_dict = {'scheduler': scheduler,
                                  'interval': 'step', 'frequency': 1}
            elif self.scheduler_cfg._target_ == 'torch.optim.lr_scheduler.ReduceLROnPlateau':
                scheduler = hydra.utils.instantiate(
                    self.scheduler_cfg, optimizer=optimizer)
                scheduler_dict = {
                    'scheduler': scheduler, 'monitor': 'val_checkpoint_on', 'interval': 'epoch', 'frequency': 1}
            else:
                raise NotImplementedError('Unknown scheduler target!')
            return [optimizer], [scheduler_dict]
        return optimizer

    def step(self, batch, batch_idx, split):
        inputs, _, targets, attributes = batch
        if batch_idx == 471:
            breakpoint()
        outputs = self(inputs)
        loss = self.criterion(outputs, targets, attributes)
        if split == 'train':
            result = pl.TrainResult(loss)
        else:
            result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log(f'{split}_loss', loss, prog_bar=True,
                   sync_dist=True, on_epoch=True)
        return result

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')

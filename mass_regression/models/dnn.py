import time
from collections import OrderedDict
from typing import Callable, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics import Metric

import hydra


class DNN(pl.LightningModule):
    def __init__(self, cr: float, cs: List[int], input_dim: int, output_dim: int, metrics_cfg: dict, optimizer_cfg: dict, criterion_cfg: dict):
        super().__init__()
        cs = [int(cr * x) for x in cs]
        self.save_hyperparameters()
        self.metrics = hydra.utils.call(metrics_cfg)
        self.optimizer_factory = hydra.utils.instantiate(optimizer_cfg)
        
        self.input_stage = nn.Sequential(nn.Linear(input_dim, cs[0]), nn.ReLU())

        hidden_modules = []
        for i in range(1, len(cs) - 1):
            hidden_modules.append(nn.Linear(cs[i-1], cs[i]))
            hidden_modules.append(nn.ReLU())
            hidden_modules.append(nn.Dropout())
        self.hidden_stage = nn.Sequential(*hidden_modules)
    
        self.output_stage = nn.Linear(cs[-1], output_dim)

    def forward(self, x):
        x1 = self.input_stage(x)
        
        x2 = self.hidden_stage(x1)

        out = self.output_stage(x2)

        return out

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())
        return optimizer

    def step(self, batch, batch_idx, split):
        inputs, targets, attributes, loss_targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets, attributes, loss_targets)
        if split == 'train':
            result = pl.TrainResult(loss)
        else:
            result = pl.EvalResult(checkpoint_on=loss)
        result.log(f'{split}_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        # Hack to record 
        if split == 'test':
            if batch_idx == 0:
                self.predictions = []
            self.predictions.append(outputs.cpu().numpy())
        return result

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, split='test')
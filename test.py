import argparse
import copy
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as df
import pytorch_lightning as pl
import torch
import wandb
import yaml
from criterion.higgs import calc_tree
from omegaconf import OmegaConf
from tqdm import tqdm


def load_from_run(run_dir: str):
    run_dir = Path(run_dir)
    config_path = run_dir / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    checkpoint = str([p for p in sorted(run_dir.glob('*.ckpt'))][-1])
    model = cfg.model._target_
    if model == 'models.dnn.DNN':
        from models.dnn import DNN
        model = DNN
    else:
        raise NotImplementedError()
    model = model.load_from_checkpoint(checkpoint)
    wandb_dir = run_dir / 'wandb'
    run_path = [p for p in sorted(wandb_dir.glob('run*'))][-1]
    wandb_id = run_path.name.split('-')[-1]
    project = cfg['wandb']['project']
    feature_transform, output_transform, target_transform = hydra.utils.instantiate(
        cfg.transforms)
    datamodule = hydra.utils.instantiate(
        cfg.dataset, testing_masses='all', targets=cfg.dataset_criterion.targets, feature_transform=feature_transform, output_transform=output_transform, target_transform=target_transform)
    return model, checkpoint, project, wandb_id, datamodule, output_transform


def save_predictions(model, datamodule, output_dir: Path, output_transform):
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()
    dataset = test_loader.dataset
    model.eval()
    
    output_df = dataset.events.copy()
    with torch.no_grad():
        features, _, _, attributes = dataset[:]
        outputs_ = model(features)
        outputs = output_transform.inverse_transform(outputs_)
        Nbx_pred, Nby_pred, Wam_pred, Wbm_pred, Hm_pred = calc_tree(
            outputs, attributes)
    output_df['Na_Predx'] = outputs[:, 0].numpy()
    output_df['Na_Predy'] = outputs[:, 1].numpy()
    output_df['Na_Predz'] = outputs[:, 2].numpy()
    output_df['Nb_Predz'] = outputs[:, 3].numpy()
    output_df['Nb_Predx'] = Nbx_pred.numpy()
    output_df['Nb_Predy'] = Nby_pred.numpy()
    output_df['Wa_Predm'] = Wam_pred.numpy()
    output_df['Wb_Predm'] = Wbm_pred.numpy()
    output_df['H_Predm'] = Hm_pred.numpy()
    output_df.to_pickle(output_dir / 'events.pkl')


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('run_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    model, checkpoint, project, wandb_id, datamodule, output_transform, = load_from_run(
        args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_predictions(model, datamodule, output_dir, output_transform)


if __name__ == '__main__':
    main()

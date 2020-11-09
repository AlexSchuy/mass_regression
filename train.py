import argparse
import copy
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import submitit
import torch
from omegaconf import DictConfig, OmegaConf

import wandb
from mass_regression.utils import StandardScaler
from mass_regression.training.checkpoint import CheckpointEveryNSteps


def train(cfg: DictConfig, output_dir: Path) -> None:
    logging.info('Beginning training...')

    # Instantiate the standard scalar transforms.
    feature_transform, output_transform, target_transform = hydra.utils.instantiate(
        cfg.transforms)

    # Instantiate the dataset.
    datamodule = hydra.utils.instantiate(
        cfg.dataset, targets=cfg.dataset_criterion.targets, feature_transform=feature_transform, output_transform=output_transform, target_transform=target_transform)

    steps_per_epoch = None
    scheduler_cfg = None
    if 'scheduler' in cfg:
        scheduler_cfg = cfg.scheduler
        if scheduler_cfg._target_ == 'torch.optim.lr_scheduler.OneCycleLR':
            datamodule.setup('fit')
            steps_per_epoch = len(datamodule.train_dataloader())

    # Instantiate the model (pass configs and mean/std to avoid pickle issues in checkpointing).
    model = hydra.utils.instantiate(
        cfg.model, optimizer_cfg=cfg.optimizer.factory, scheduler_cfg=scheduler_cfg, criterion_cfg=cfg.dataset_criterion,
        output_mean=output_transform.mean, output_std=output_transform.std, target_mean=target_transform.mean,
        target_std=target_transform.std, steps_per_epoch=steps_per_epoch)

    callbacks = []

    # Set up checkpointing.
    if cfg.init_ckpt is not None:
        logging.info(f'Loading checkpoint={cfg.init_ckpt}')
        resume_from_checkpoint = cfg.init_ckpt
    else:
        resume_from_checkpoint = None
    checkpoint_callback = hydra.utils.instantiate(
        cfg.checkpoint, filepath=f'{str(output_dir)}/{{epoch:02d}}')

    # Set up early stopping.
    if 'early_stopping' in cfg:
        early_stop_callback = hydra.utils.instantiate(cfg.early_stopping)
        callbacks.append(early_stop_callback)

    # Set up lr monitor.
    lr_monitor = pl.callbacks.LearningRateMonitor('step')
    callbacks.append(lr_monitor)

    # Set up step checkpoints.
    step_checkpoint = CheckpointEveryNSteps(save_step_frequency=1000)
    callbacks.append(step_checkpoint)

    # Set up wandb logging.
    wandb_id = cfg.wandb.id
    if wandb_id is None:
        wandb_id = (output_dir.parent.name +
                    output_dir.name).replace('-', '')
    logger = hydra.utils.instantiate(
        cfg.wandb, save_dir=str(output_dir), id=wandb_id)

    # train
    trainer = pl.Trainer(gpus=cfg.train.gpus, logger=logger, weights_save_path=str(
        output_dir), max_epochs=cfg.train.num_epochs, checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=resume_from_checkpoint, deterministic=True, distributed_backend=cfg.train.distributed_backend,
        gradient_clip_val=cfg.train.gradient_clip_val, callbacks=callbacks, terminate_on_nan=True, auto_lr_find=cfg.optimizer.auto_lr)
    trainer.logger.log_hyperparams(cfg._content)  # pylint: disable=no-member
    trainer.tune(model=model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    # Set up python logging.
    logger = logging.getLogger()
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=cfg.log_level,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(cfg.pretty())
    if 'slurm' in cfg.train:
        slurm_dir = Path.cwd() / 'slurm'
        slurm_dir.mkdir()
        executor = submitit.AutoExecutor(slurm_dir)
        executor.update_parameters(slurm_gpus_per_node=cfg.train.slurm.gpus_per_node, slurm_nodes=cfg.train.slurm.nodes, slurm_ntasks_per_node=cfg.train.slurm.gpus_per_node,
                                   slurm_cpus_per_task=cfg.train.slurm.cpus_per_task, slurm_time=cfg.train.slurm.time, slurm_additional_parameters={'constraint': 'gpu', 'account': cfg.train.slurm.account})
        job = executor.submit(train, cfg=cfg, output_dir=Path.cwd())
        logging.info(f'submitted job {job.job_id}.')
    else:
        train(cfg, output_dir=Path.cwd())


if __name__ == '__main__':
    hydra_main()  # pylint: disable=no-value-for-parameter

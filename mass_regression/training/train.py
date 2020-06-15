"""Training

These routines handle training the ML models that are studied (quantum and classical for comparison).
"""
import json
import math
import os
import shutil
import time

import altair as alt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.distributions import randint
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers  # pylint: disable=import-error
from tensorflow.keras.losses import MSE  # pylint: disable=import-error
from tensorflow.keras.metrics import \
    MeanAbsoluteError  # pylint: disable=import-error
from tensorflow.keras.metrics import \
    MeanAbsolutePercentageError  # pylint: disable=import-error
from tensorflow.keras.metrics import \
    RootMeanSquaredError  # pylint: disable=import-error
from tensorflow.keras.mixed_precision import experimental as mixed_precision  # pylint: disable=import-error

import definitions
import training.wlnu as wlnu
from training import data, train
from training.base_trainer import BaseTrainer
from training.loguniform import LogUniform
from training.models.model_v1 import Model_V1
from training.steploguniform import StepLogUniform
from training.stepuniform import StepUniform
from training.wlnu import WlnuDataset


def calc_Wm(x, y_pred):
    METx = x[:, 0]
    METy = x[:, 1]
    Lax_gen = x[:, 2]
    Lay_gen = x[:, 3]
    Laz_gen = x[:, 4]
    Lam_gen = x[:, 5]
    Lbx_gen = x[:, 6]
    Lby_gen = x[:, 7]
    Lbz_gen = x[:, 8]
    Lbm_gen = x[:, 9]
    Nax_pred = y_pred[:, 0]
    Nay_pred = y_pred[:, 1]
    Naz_pred = y_pred[:, 2]
    Nam_pred = tf.zeros_like(Nax_pred)
    Nbx_pred = METx - Nax_pred
    Nby_pred = METy - Nay_pred
    Nbz_pred = y_pred[:, 3]
    Nbm_pred = tf.zeros_like(Nbx_pred)
    _, _, _, Wam_pred = data.add_fourvectors(
        Nax_pred, Nay_pred, Naz_pred, Nam_pred, Lax_gen, Lay_gen, Laz_gen, Lam_gen)
    _, _, _, Wbm_pred = data.add_fourvectors(
        Nbx_pred, Nby_pred, Nbz_pred, Nbm_pred, Lbx_gen, Lby_gen, Lbz_gen, Lbm_gen)
    return tf.stack([Wam_pred, Wbm_pred], axis=1)


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('dataset', choices=definitions.DATASETS)
    parser.add_argument('target')

    dataset = 'Wlnu'
    target = 'nu'
    pad_features = 'Wm_gen'
    seed = 5
    set_seed(seed)
    hp_rv = {'num_layers': randint(1, 5),
             'num_units': StepUniform(start=10, num=20, step=10),
             'learning_rate': LogUniform(loc=-5, scale=4, base=10, discrete=False),
             'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
             'epochs': randint(10, 201),
             'dropout': StepUniform(start=0.0, num=2, step=0.5)}
    dataset = WlnuDataset(
        definitions.FEATURES['Wlnu'], targets=target, pad_features=pad_features)
    model = Model_V1(dataset, wlnu.Wm_loss, seed=seed)
    trainer = BaseTrainer(model, dataset, concat_input=True)
    trainer.random_search(20, hp_rv)


if __name__ == '__main__':
    main()

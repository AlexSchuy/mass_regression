"""Training

These routines handle training the ML models that are studied (quantum and classical for comparison).
"""


from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from scipy.stats.distributions import randint

import definitions
import training.wlnu as wlnu
from training import data, train
from training.loguniform import LogUniform
from training.models.model_v1_factory import Model_V1_Factory
from training.steploguniform import StepLogUniform
from training.stepuniform import StepUniform
from training.wlnu import WlnuDataset, WlnuTrainer
from training.constant import Constant

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

    parser = ArgumentParser()
    parser.add_argument('dataset', choices=definitions.DATASETS)
    parser.add_argument('target')

    dataset = 'Wlnu'
    target_name = 'nu'
    pad_features = 'Wm_gen'
    seed = 5
    set_seed(seed)
    hp_rv = {'num_layers': randint(1, 5),
             'num_units': StepUniform(start=10, num=20, step=10),
             'learning_rate': LogUniform(loc=-5, scale=4, base=10, discrete=False),
             'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
             'epochs': Constant(1000),
             'dropout': StepUniform(start=0.0, num=2, step=0.5)}
    dataset = WlnuDataset(
        definitions.FEATURES['Wlnu'], target_name=target_name, pad_features=pad_features)
    model_factory = Model_V1_Factory(dataset, wlnu.WmLoss, seed=seed)
    trainer = WlnuTrainer(model_factory, dataset, delta_callback=False, early_stopping=True)
    trainer.random_search(20, hp_rv)


if __name__ == '__main__':
    main()

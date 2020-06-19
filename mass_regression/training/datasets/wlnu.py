import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.distributions import randint

import definitions
import training.data as data
from training.base.base_dataset import BaseDataset
from training.base.base_parser import BaseParser
from training.base.base_trainer import BaseTrainer
from training.distributions.constant import Constant
from training.base.base_loss import BaseLoss
from training.distributions.loguniform import LogUniform
from training.models.model_v1_factory import Model_V1_Factory
from training.distributions.steploguniform import StepLogUniform
from training.distributions.stepuniform import StepUniform
from training.callbacks.delta_callback import DeltaCallback

mse = tf.keras.losses.MeanSquaredError()


def df_calc_Wm(df, y_pred):
    x = df[definitions.FEATURES['Wlnu']].values
    return calc_Wm(x, y_pred)


def calc_Wm(x, y_pred):
    METx = x[:, 0]
    METy = x[:, 1]
    Lx_reco = x[:, 2]
    Ly_reco = x[:, 3]
    Lz_reco = x[:, 4]
    Lm_reco = x[:, 5]
    Nax_reco = METx
    Nay_reco = METy
    Naz_pred = y_pred[:, 0]
    Nam_pred = tf.zeros_like(Nax_reco)
    _, _, _, Wm_pred = data.add_fourvectors(
        Nax_reco, Nay_reco, Naz_pred, Nam_pred, Lx_reco, Ly_reco, Lz_reco, Lm_reco)
    return Wm_pred


def NUz_loss(x, x_pad, y_true, y_pred, dataset):
    return mse(y_true, y_pred)


def Wm_loss(x, x_pad, y_true, y_pred, dataset):
    Wm_pred = dataset.scale_x_pad(calc_Wm(dataset.unscale_x(x), dataset.unscale_y(y_pred)))
    Wm_true = x_pad[:, 0]
    return mse(Wm_true, Wm_pred)


class WmLoss(BaseLoss):
    name = 'Wm_loss'

    def __init__(self, x, x_pad, dataset):
        super().__init__(x, x_pad, dataset, name=self.name)

    def loss_fn(self, y_true, y_pred):
        return Wm_loss(self.x, self.x_pad, y_true, y_pred, self.dataset)


class NUzLoss(BaseLoss):
    name = 'NUz_loss'

    def __init__(self, x, x_pad, dataset):
        super().__init__(x, x_pad, dataset, name=self.name)

    def loss_fn(self, y_true, y_pred):
        return NUz_loss(self.x, self.x_pad, y_true, y_pred, self.dataset)


class WlnuDataset(BaseDataset):
    def __init__(self, features=definitions.FEATURES['Wlnu'], target_name='nu', pad_features='Wm_gen'):
        super().__init__(definitions.SAMPLES_DIR / 'Wlnu',
                         features, target_name, pad_features, name='Wlnu')

    @property
    def targets(self):
        return definitions.TARGETS['Wlnu'][self.target_name]

    def calculate_tree(self, x, y_pred):
        tree = pd.DataFrame()
        if self.target_name == 'nu':
            tree['NUz'] = y_pred
            tree['Wm'] = calc_Wm(x, y_pred)
        else:
            raise NotImplementedError()
        return tree


class WlnuTrainer(BaseTrainer):
    def __init__(self, model_factory, dataset, log_dir=None, early_stopping=False, delta_callback=True):
        super().__init__(model_factory, dataset, log_dir, early_stopping, callbacks=None)
        if delta_callback:
            self.callbacks.append(DeltaCallback(dataset, self.log_dir / 'deltas'))


class WlnuParser(BaseParser):
    def __init__(self, parser, subparsers):
        super().__init__(parser, subparsers, name='wlnu')
        subparser = subparsers.add_parser('wlnu')
        subparser.add_argument('target_name', choices=definitions.TARGETS['Wlnu'].keys())
        subparser.add_argument('loss', choices=['wm', 'nuz'])
        subparser.add_argument('--delta_callback', action='store_true')
        subparser.add_argument('--no_early_stopping', action='store_true')

    def parse(self, args):
        super().parse(args)
        pad_features = 'Wm_gen'
        if args.loss == 'wm':
            loss = WmLoss
        else:
            loss = NUzLoss
        dataset = WlnuDataset(
            definitions.FEATURES['Wlnu'], target_name=args.target_name, pad_features=pad_features)
        model_factory = Model_V1_Factory(dataset, loss, seed=args.seed)
        trainer = WlnuTrainer(model_factory, dataset, delta_callback=args.delta_callback,
                              early_stopping=not args.no_early_stopping)
        if not args.single_run:
            hp_rv = {'num_layers': randint(1, 5),
                     'num_units': StepUniform(start=10, num=20, step=10),
                     'learning_rate': LogUniform(loc=-5, scale=4, base=10, discrete=False),
                     'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
                     'epochs': Constant(200),
                     'dropout': StepUniform(start=0.0, num=2, step=0.5)}
            trainer.random_search(args.n, hp_rv, hot_start=True)
        else:
            hparams = {"num_layers": 2, "num_units": 150.0, "learning_rate": 0.00037964080309329335,
                       "batch_size": 256, "epochs": 200, "dropout": 0.0, "run_num": 13, "val_loss": 0.08129070078134537}
            run_dir = trainer.log_dir / 'single_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            trainer.train_test_model(hparams, run_dir)


def main():
    wlnu_dataset = WlnuDataset(pad_features='Wm_gen')
    df_train = wlnu_dataset.train(split=False)
    x = df_train[['METx', 'METy', 'Lx_reco',
                  'Ly_reco', 'Lz_reco', 'Lm_reco']].values
    y = df_train[['NUz_gen']].values
    Wm_pred = calc_Wm(x, y)
    print((df_train['Wm_gen'].values - Wm_pred) / df_train['Wm_gen'].values)

    Wm_pred = df_calc_Wm(df_train, df_train[['NUz_gen']].values)
    print((df_train['Wm_gen'].values - Wm_pred) / df_train['Wm_gen'].values)


if __name__ == '__main__':
    main()

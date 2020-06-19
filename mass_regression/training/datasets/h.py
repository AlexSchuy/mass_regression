import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.distributions import randint

import definitions
import training.data as data
from training.base.base_dataset import BaseDataset
from training.base.base_loss import BaseLoss
from training.base.base_parser import BaseParser
from training.base.base_trainer import BaseTrainer
from training.callbacks.delta_callback import DeltaCallback
from training.distributions.constant import Constant
from training.distributions.loguniform import LogUniform
from training.distributions.steploguniform import StepLogUniform
from training.distributions.stepuniform import StepUniform
from training.models.model_v1_factory import Model_V1_Factory

mse = tf.keras.losses.MeanSquaredError()


def df_calc_Wm(df, y_pred):
    x = df[definitions.FEATURES['H']].values
    return calc_Wm(x, y_pred)


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


def calc_Hm(x, y_pred):
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
    Wax_pred, Way_pred, Waz_pred, Wam_pred = data.add_fourvectors(
        Nax_pred, Nay_pred, Naz_pred, Nam_pred, Lax_gen, Lay_gen, Laz_gen, Lam_gen)
    Wbx_pred, Wby_pred, Wbz_pred, Wbm_pred = data.add_fourvectors(
        Nbx_pred, Nby_pred, Nbz_pred, Nbm_pred, Lbx_gen, Lby_gen, Lbz_gen, Lbm_gen)
    _, _, _, Hm_pred = data.add_fourvectors(Wax_pred, Way_pred, Waz_pred, Wam_pred,
                                            Wbx_pred, Wby_pred, Wbz_pred, Wbm_pred)
    return Hm_pred


def NUz_loss(x, x_pad, y_true, y_pred, dataset):
    return mse(y_true, y_pred)


def Wm_loss(x, x_pad, y_true, y_pred, dataset):
    Wm_pred = dataset.scale_x_pad(calc_Wm(dataset.unscale_x(x), dataset.unscale_y(y_pred)))
    Wm_true = x_pad
    return mse(Wm_true, Wm_pred)


def H_loss(x, x_pad, y_true, y_pred, dataset):
    H_pred = dataset.scale_x_pad(calc_Hm(dataset.unscale_x(x), dataset.unscale_y(y_pred)))
    H_true = x_pad
    return mse(H_true, H_pred)


class HLoss(BaseLoss):
    name = 'H_loss'

    def __init__(self, x, x_pad, dataset):
        super().__init__(x, x_pad, dataset, name=self.name)

    def loss_fn(self, y_true, y_pred):
        return H_loss(self.x, self.x_pad, y_true, y_pred, self.dataset)


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


class HiggsDataset(BaseDataset):
    def __init__(self, mass, features=definitions.FEATURES['H'], target_name='nu', pad_features=['Wam_gen', 'Wbm_gen']):
        super().__init__(definitions.SAMPLES_DIR / f'H{mass}',
                         features, target_name, pad_features, name=f'H{mass}')

    @property
    def targets(self):
        return definitions.TARGETS['H'][self.target_name]

    @property
    def tree_gen(self):
        gen_features = ['Nax_gen', 'Nay_gen', 'Naz_gen', 'Nbz_gen', 'Wam_gen', 'Wbm_gen', 'Hm_gen']
        train_df = self.train(split=False)
        return train_df[gen_features].rename(columns={k: k.replace('_gen', '') for k in gen_features})

    def calculate_tree(self, x, y_pred):
        tree = pd.DataFrame()
        if self.target_name == 'nu':
            tree['Nax'] = y_pred[:, 0]
            tree['Nay'] = y_pred[:, 1]
            tree['Naz'] = y_pred[:, 2]
            tree['Nbz'] = y_pred[:, 3]
            Wm = calc_Wm(x, y_pred)
            tree['Wam'] = Wm[:, 0]
            tree['Wbm'] = Wm[:, 1]
            tree['Hm'] = calc_Hm(x, y_pred)
        else:
            raise NotImplementedError()
        return tree


class HiggsTrainer(BaseTrainer):
    def __init__(self, model_factory, dataset, log_dir=None, early_stopping=False, delta_callback=True):
        super().__init__(model_factory, dataset, log_dir, early_stopping, callbacks=None)
        if delta_callback:
            self.callbacks.append(DeltaCallback(dataset, self.log_dir / 'deltas'))


class HiggsParser(BaseParser):
    def __init__(self, parser, subparsers):
        super().__init__(parser, subparsers, name='higgs')
        subparser = subparsers.add_parser('higgs')
        subparser.add_argument('mass', type=int, choices=[125, 400, 750, 1000, 1500])
        subparser.add_argument('target_name', choices=definitions.TARGETS['H'].keys())
        subparser.add_argument('loss', choices=['h', 'wm', 'nuz'])
        subparser.add_argument('--delta_callback', action='store_true')
        subparser.add_argument('--no_early_stopping', action='store_true')

    def parse(self, args):
        super().parse(args)
        if args.loss == 'wm':
            pad_features = ['Wam_gen', 'Wbm_gen']
            loss = WmLoss
        elif args.loss == 'nuz':
            pad_features = ['Wam_gen', 'Wbm_gen']
            loss = NUzLoss
        elif args.loss == 'h':
            pad_features = ['Hm_gen']
            loss = HLoss
        dataset = HiggsDataset(args.mass, definitions.FEATURES['H'], target_name=args.target_name, pad_features=pad_features)
        model_factory = Model_V1_Factory(dataset, loss, seed=args.seed)
        trainer = HiggsTrainer(model_factory, dataset, delta_callback=args.delta_callback,
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
    higgs_dataset = HiggsDataset(mass=125)
    df_train = higgs_dataset.train(split=False)
    x = df_train[definitions.FEATURES['H']].values
    x_pad = df_train[higgs_dataset.pad_features].values
    y = df_train[higgs_dataset.targets].values
    Wm_pred = calc_Wm(x, y)
    print((Wm_pred - x_pad) / x_pad)

    Wm_pred = df_calc_Wm(df_train, y)
    print((Wm_pred - x_pad) / x_pad)

    higgs_dataset = HiggsDataset(mass=400)
    df_train = higgs_dataset.train(split=False)
    x = df_train[definitions.FEATURES['H']].values
    x_pad = df_train[higgs_dataset.pad_features].values
    y = df_train[higgs_dataset.targets].values
    Wm_pred = calc_Wm(x, y)
    print((Wm_pred - x_pad) / x_pad)

    Wm_pred = df_calc_Wm(df_train, y)
    print((Wm_pred - x_pad) / x_pad)


if __name__ == '__main__':
    main()

import numpy as np
import tensorflow as tf
from scipy.stats.distributions import randint

import data
import definitions
from base_dataset import BaseDataset
from base_parser import BaseParser
from base_trainer import BaseTrainer
from custom_loss import CustomLoss
from training.constant import Constant
from training.loguniform import LogUniform
from training.models.model_v1_factory import Model_V1_Factory
from training.steploguniform import StepLogUniform
from training.stepuniform import StepUniform

mse = tf.keras.losses.MeanSquaredError()


def df_calc_Wm(df, NUz_reco):
    x = df[definitions.FEATURES['Wlnu']].values
    y_pred = NUz_reco.values
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


class WmLoss(CustomLoss):
    name = 'Wm_loss'

    def __init__(self, x, x_pad, dataset):
        super().__init__(x, x_pad, dataset, name=self.name)

    def loss_fn(self, y_true, y_pred):
        return Wm_loss(self.x, self.x_pad, y_true, y_pred, self.dataset)


class NUzLoss(CustomLoss):
    name = 'NUz_loss'

    def __init__(self, x, x_pad, dataset):
        super().__init__(x, x_pad, dataset, name=self.name)

    def loss_fn(self, y_true, y_pred):
        return NUz_loss(self.x, self.x_pad, y_true, y_pred, self.dataset)


class WlnuDataset(BaseDataset):
    def __init__(self, features=definitions.FEATURES['Wlnu'], target_name='nu', pad_features=None):
        super().__init__(definitions.SAMPLES_DIR / 'Wlnu',
                         features, target_name, pad_features, name='Wlnu')

    @property
    def targets(self):
        return definitions.TARGETS['Wlnu'][self.target_name]


class DeltaCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir):
        self.dataset = dataset
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        epochs = self.params['epochs']
        self.delta_Wm_train = np.zeros((epochs, self.dataset.num_training_samples))
        self.delta_NUz_train = np.zeros((epochs, self.dataset.num_training_samples))
        self.delta_Wm_val = np.zeros((epochs, self.dataset.num_val_samples))
        self.delta_NUz_val = np.zeros((epochs, self.dataset.num_val_samples))

    def on_train_end(self, logs=None):
        def _save(file_name, array):
            np.save(self.log_dir / file_name, array)
        _save('delta_Wm_train.npy', self.delta_Wm_train)
        _save('delta_NUz_train.npy', self.delta_NUz_train)
        _save('delta_Wm_val.npy', self.delta_Wm_val)
        _save('delta_NUz_val.npy', self.delta_NUz_val)

    def on_epoch_end(self, epoch, logs=None):
        def calc_delta(x, x_pad, y):
            y_pred = self.dataset.unscale_y(
                self.model.predict((x, x_pad, y)))
            Wm_pred = calc_Wm(self.dataset.unscale_x(x), y_pred)
            Wm_true = self.dataset.unscale_x_pad(x_pad[:, 0])
            return Wm_pred - Wm_true, (y_pred - self.dataset.unscale_y(y))[:, 0]
        x_train, x_pad_train, y_train = self.dataset.train()
        self.delta_Wm_train[epoch, :], self.delta_NUz_train[epoch, :] = calc_delta(x_train.values, x_pad_train.values, y_train.values)
        x_val, x_pad_val, y_val = self.dataset.val()
        self.delta_Wm_val[epoch, :], self.delta_NUz_val[epoch, :] = calc_delta(x_val.values, x_pad_val.values, y_val.values)


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

    Wm_pred = df_calc_Wm(df_train, df_train[['NUz_gen']])
    print((df_train['Wm_gen'].values - Wm_pred) / df_train['Wm_gen'].values)


if __name__ == '__main__':
    main()

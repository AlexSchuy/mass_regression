import numpy as np
import tensorflow as tf

import data
import definitions
from base_dataset import BaseDataset
from base_trainer import BaseTrainer
from custom_loss import CustomLoss

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


def Nz_loss(x, x_pad, y_true, y_pred, dataset):
    raise NotImplementedError()


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


class NzLoss(CustomLoss):
    def __init__(self, x, x_pad, dataset):
        super().__init__(x, x_pad, dataset, name='Nz_loss')

    def loss_fn(self, y_true, y_pred):
        return Nz_loss(self.x, self.x_pad, y_true, y_pred, self.dataset)


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
        num_samples = self.dataset.num_training_samples
        self.delta_Wm_train = np.zeros((epochs, num_samples))
        self.delta_NUz_train = np.zeros((epochs, num_samples))
        self.delta_Wm_val = np.zeros((epochs, num_samples))
        self.delta_NUz_val = np.zeros((epochs, num_samples))

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
                self.model.predict(self.dataset.scale_x(x)))
            Wm_pred = calc_Wm(x, y_pred)
            Wm_true = x_pad[:, 0]
            return Wm_pred - Wm_true, y_pred - y
        x_train, x_pad_train, y_train = self.dataset.train(scale_x=False, scale_y=False, scale_x_pad=False)
        self.delta_Wm_train[epoch, :], self.delta_NUz_train[epoch, :] = calc_delta(x_train, x_pad_train, y_train)
        x_val, x_pad_val, y_val = self.dataset.val(scale_x=False, scale_y=False, scale_x_pad=False)
        self.delta_Wm_val[epoch, :], self.delta_NUz_val[epoch, :] = calc_delta(x_val, x_pad_val, y_val)


class WlnuTrainer(BaseTrainer):
    def __init__(self, model_factory, dataset, log_dir=None, early_stopping=False, delta_callback=True):
        super().__init__(model_factory, dataset, log_dir, early_stopping, callbacks=None)
        if delta_callback:
            self.callbacks.append(DeltaCallback(dataset, self.log_dir / 'deltas'))
        


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

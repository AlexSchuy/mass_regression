import numpy as np
import tensorflow as tf

import data
import definitions
from base_dataset import BaseDataset

mse = tf.keras.losses.MeanSquaredError()


def df_calc_Wm(df, NUz_reco):
    NUE_reco = (df['NUx_reco']**2 + df['NUy_reco']**2 + NUz_reco**2)**0.5
    LE_reco = (df['Lx_reco']**2 + df['Ly_reco']**2 +
               df['Lz_reco']**2 + df['Lm_reco']**2)**0.5
    WE_reco = NUE_reco + LE_reco
    Wz_reco = df['Lz_reco'] + NUz_reco
    Wm_reco = (WE_reco**2 - df['Wx_reco']**2 -
               df['Wy_reco']**2 - Wz_reco**2)**0.5
    return Wm_reco


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


def Wm_loss(x, x_pad, y_true, y_pred, dataset):
    Wm_pred = dataset.scale_x_pad(
        calc_Wm(dataset.unscale_x(x), dataset.unscale_y(y_pred)))
    Wm_true = x_pad[:, 0]
    return mse(Wm_true, Wm_pred)


class WlnuDataset(BaseDataset):
    def __init__(self, features=definitions.FEATURES['Wlnu'], targets='nu', pad_features=None):
        super().__init__(definitions.SAMPLES_DIR / 'Wlnu',
                         features, targets, pad_features, name='Wlnu')


class WlnuCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir):
        self.dataset = dataset
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        epochs = self.params['epochs']
        num_samples = self.dataset.num_training_samples
        self.delta_Wm = np.zeros((epochs, num_samples))
        self.delta_NUz = np.zeros((epochs, num_samples))

    def on_train_end(self, logs=None):
        def _save(file_name, array):
            np.save(self.log_dir / file_name, array)
        _save('delta_Wm.npy', self.delta_Wm)
        _save('delta_NUz.npy', self.delta_NUz)

    def on_epoch_end(self, epoch, logs=None):
        x, x_pad, y_true = self.dataset.train(scale_x=False, scale_y=False)
        y_pred = self.dataset.unscale_y(self.model.predict(x))
        Wm_pred = calc_Wm(x, y_pred)
        Wm_true = x_pad[:, 0]
        self.delta_Wm[epoch, :] = (Wm_pred - Wm_true)
        self.delta_NUz[epoch, :] = (y_pred - y_true)


def main():
    wlnu_dataset = WlnuDataset(pad_features='Wm_gen')
    df_train, _, _ = wlnu_dataset.train(split=False)
    x = tf.constant(
        df_train[['METx', 'METy', 'Lx_reco', 'Ly_reco', 'Lz_reco', 'Lm_reco']].values)
    y = tf.constant(df_train[['NUz_gen']].values)
    Wm_pred = calc_Wm(x, y)
    print(tf.constant(df_train['Wm_gen'].values) - Wm_pred)


if __name__ == '__main__':
    main()

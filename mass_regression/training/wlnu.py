import tensorflow as tf

import data
import definitions
from base_dataset import BaseDataset

mse = tf.keras.losses.MeanSquaredError()


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
    def __init__(self, features, targets, pad_features=None):
        super().__init__(definitions.SAMPLES_DIR / 'Wlnu', features, targets, pad_features)


def main():
    df_train, _, _ = data.get_datasets(
        dataset='Wlnu', target='W', x_y_split=False)
    x = tf.constant(
        df_train[['METx', 'METy', 'Lx_reco', 'Ly_reco', 'Lz_reco', 'Lm_reco']].values)
    y = tf.constant(df_train[['NUz_gen']].values)
    Wm_pred = calc_Wm(x, y)
    print(tf.constant(df_train['Wm_gen'].values) - Wm_pred)


if __name__ == '__main__':
    main()

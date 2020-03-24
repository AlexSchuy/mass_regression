import os

import numpy as np
import pandas as pd
import tensorflow as tf

import definitions


def calc_E(px, py, pz, m):
    return (px**2 + py**2 + pz**2 + m**2)**0.5


def add_fourvectors(px1, py1, pz1, m1, px2, py2, pz2, m2):

    E1 = calc_E(px1, py1, pz1, m1)
    E2 = calc_E(px2, py2, pz2, m2)
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    E = E1 + E2
    m = (E**2 - px**2 - py**2 - pz**2)**0.5
    return px, py, pz, m


def add_to_df(p1, prefix1, p2, prefix2, p, df):
    p_x, p_y, p_z, p_m = add_fourvectors(df[f'{p1}x_{prefix1}'], df[f'{p1}y_{prefix1}'], df[f'{p1}z_{prefix1}'],
                                         df[f'{p1}m_{prefix1}'], df[f'{p2}x_{prefix2}'], df[f'{p2}y_{prefix2}'], df[f'{p2}z_{prefix2}'], df[f'{p2}m_{prefix2}'])
    df[f'{p}x_pred'] = p_x
    df[f'{p}y_pred'] = p_y
    df[f'{p}z_pred'] = p_z
    df[f'{p}m_pred'] = p_m


def calc_H125(df):
    df['Nbx_pred'] = df['METx'] - df['Nax_pred']
    df['Nby_pred'] = df['METy'] - df['Nay_pred']
    add_to_df('La', 'gen', 'Na', 'pred', 'Wa', df)
    add_to_df('Lb', 'gen', 'Nb', 'pred', 'Wb', df)
    add_to_df('Wa', 'pred', 'Wb', 'pred', 'H', df)


def calc_Wm(df, NUz_reco):
    NUE_reco = (df['NUx_reco']**2 + df['NUy_reco']**2 + NUz_reco**2)**0.5
    LE_reco = (df['Lx_reco']**2 + df['Ly_reco']**2 +
               df['Lz_reco']**2 + df['Lm_reco']**2)**0.5
    WE_reco = NUE_reco + LE_reco
    Wz_reco = df['Lz_reco'] + NUz_reco
    Wm_reco = (WE_reco**2 - df['Wx_reco']**2 -
               df['Wy_reco']**2 - Wz_reco**2)**0.5
    return Wm_reco


def get_jigsaw(dataset='Wlnu', target='W'):
    data_path = definitions.SAMPLES_DIR / dataset

    def get(t):
        path = data_path / f'{t}.pkl'
        data = pd.read_pickle(path)
        y = data[definitions.JIGSAW_TARGETS[dataset][target]]
        return y
    return get('train'), get('val'), get('test')


def get_datasets(dataset='Wlnu', target='W', scale_x=False, scale_y=False, x_y_split=True, pad_target=None):
    if (scale_x or scale_y) and not x_y_split:
        raise ValueError('Cannot scale data that is not being x/y split.')
    data_path = definitions.SAMPLES_DIR / dataset

    def get_data(t):
        path = data_path / f'{t}.pkl'
        data = pd.read_pickle(path)
        if not x_y_split:
            return data
        else:
            x = data[definitions.FEATURES[dataset]]
            targets = definitions.TARGETS[dataset][target]
            if pad_target:
                targets = targets + definitions.PAD_TARGETS[dataset][pad_target]
            y = data[targets]
            return x, y

    if x_y_split:
        x_train, y_train = get_data('train')
        x_val, y_val = get_data('val')
        x_test, y_test = get_data('test')

        def scale(X, mean, std):
            return (X - mean) / std
        if scale_x:
            x_mean = np.mean(x_train).values
            x_std = np.std(x_train).values
            print(f'x_mean = {x_mean}')
            print(f'x_std = {x_std}')
            x_train = scale(x_train, x_mean, x_std)
            x_val = scale(x_val, x_mean, x_std)
            x_test = scale(x_test, x_mean, x_std)
        if scale_y:
            y_mean = np.mean(y_train).values
            y_std = np.std(y_train).values
            print(f'y_mean = {y_mean}')
            print(f'y_std = {y_std}')
            y_train = scale(y_train, y_mean, y_std)
            y_val = scale(y_val, y_mean, y_std)
            y_test = scale(y_test, y_mean, y_std)

        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        df_train = get_data('train')
        df_val = get_data('val')
        df_test = get_data('test')
        return df_train, df_val, df_test


def get_num_targets(dataset, target):
    return len(definitions.TARGETS[dataset][target])

def get_num_pad_targets(dataset, pad_target):
    return len(definitions.PAD_TARGETS[dataset][pad_target])

def get_scale_funcs(dataset, target, pad_target):
    _, padded_y, _, _, _, _ = get_datasets(dataset=dataset, target=target, pad_target=pad_target)
    num_targets = get_num_targets(dataset, target)
    y = padded_y.iloc[:, :num_targets]
    y_pad = padded_y.iloc[:, num_targets:]
    y_mean = np.mean(y).values
    y_std = np.std(y).values
    y_pad_mean = np.mean(y_pad).values
    y_pad_std = np.std(y_pad).values
    
    def scale_y_pad(Y_pad):
        return (Y_pad - y_pad_mean) / y_pad_std

    def unscale_y(Y):
        return Y * y_std + y_mean

    return scale_y_pad, unscale_y

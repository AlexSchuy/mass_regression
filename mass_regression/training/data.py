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

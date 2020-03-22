import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

import definitions
from common import utils, validate


def get_jigsaw(dataset='Wlnu', target='W'):
    data_path = definitions.SAMPLES_DIR / dataset

    def get(t):
        path = data_path / f'{t}.pkl'
        data = pd.read_pickle(path)
        y = data[definitions.JIGSAW_TARGETS[dataset][target]]
        return y
    return get('train'), get('val'), get('test')


def get_datasets(dataset='Wlnu', target='W', scale=False):

    data_path = definitions.SAMPLES_DIR / dataset

    def get_x_y(t):
        path = data_path / f'{t}.pkl'
        data = pd.read_pickle(path)
        x = data[definitions.FEATURES[dataset]]
        y = data[definitions.TARGETS[dataset][target]]
        return x, y

    x_train, y_train = get_x_y('train')
    x_val, y_val = get_x_y('val')
    x_test, y_test = get_x_y('test')

    if scale:
        mean = np.mean(x_train.values)
        std = np.std(x_train.values)

        def scaled(X):
            return (X - mean) / std
        x_train = scaled(x_train)
        x_val = scaled(x_val)
        x_test = scaled(x_test)
        return x_train, y_train, x_val, y_val, x_test, y_test

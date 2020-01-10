import os

import numpy as np
import pandas as pd

import tensorflow as tf
from common import utils, validate
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


def get_train_test_datasets(n=-1, train_frac=0.8, seed=10598, dataset='Wlnu', full_dataset=False):

    if dataset == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

    n = train_size + test_size

    if full_dataset:
        if dataset == 'Wlnu':
            X, y, df = load_Wlnu(n, full_dataset=full_dataset)
        if dataset == 'Wlnu_Wm_label':
            X, y, df = load_Wlnu(
                n, full_dataset=full_dataset, use_Wm_labels=True)
        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X, y, df, random_state=seed, train_size=train_size, test_size=test_size)
        return X_train, y_train, df_train, X_test, y_test, df_test
    else:
        if dataset == 'Wlnu':
            X, y = load_Wlnu(n, full_dataset=full_dataset, use_Wm_labels=True)
        if dataset == 'Wlnu_Wm_label':
            X, y = load_Wlnu(n, full_dataset=full_dataset)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=seed, train_size=train_size, test_size=test_size)
        return X_train, y_train, X_test, y_test


def load_Wlnu(n, full_dataset=False, use_Wm_labels=False):
    project_path = utils.get_project_path()
    filepath = os.path.join(project_path, 'samples', 'converted', 'Wlnu.pkl')
    df = pd.read_pickle(filepath)
    df = df.iloc[:n]
    if use_Wm_labels:
        label_names = ['Wm_gen']
    else:
        label_names = ['NUz_gen']
    feature_names = ['Lx_reco', 'Ly_reco',
                     'Lz_reco', 'Lm_reco', 'METx', 'METy']
    features = df[feature_names]
    labels = df[label_names]
    labels = np.ravel(labels)
    if full_dataset:
        return features, labels, df
    return features, labels

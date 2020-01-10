"""Training

These routines handle training the ML models that are studied (quantum and classical for comparison).
"""

import argparse
import logging
import os
import time
from configparser import ConfigParser

import numpy as np
from scipy.stats import randint, uniform
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from common import utils, validate
from training.kerasregressor import KerasRegressor
from training.data import get_train_test_datasets
from training.loguniform import LogUniform
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.metrics import r2_score, make_scorer

def make_pipeline(model, model_scaler):
    model_steps = [('model_scaler', model_scaler), ('model', model)]
    pipeline = Pipeline(steps=model_steps)
    return pipeline


def make_sklearn_nn(hidden_layers=(100,), seed=0):

    nn = MLPRegressor(hidden_layers, random_state=seed)
    pipeline = make_pipeline(nn, StandardScaler())

    # Apply a random search to tune hyperparameters.

    param_dists = {}
    param_dists['model__alpha'] = LogUniform(-6, 3)
    param_dists['model__max_iter'] = randint(100, 1000)
    cv = RandomizedSearchCV(pipeline, param_dists,
                            n_iter=10, n_jobs=-1, iid=False)

    return cv


def build_model(dataset, hparams, num_outputs):
    dataset.batch(hparams['batch_size'])

    model = tf.keras.models.Sequential()
    model.add(Flatten())
    for _ in range(hparams['num_layers']):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams['num_units'], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams['dropout']),
        tf.keras.layers.Dense(num_outputs)
    ])

def make_keras_model(hidden_layers=(100,), seed=0):
    nn = KerasRegressor(build_fn=keras_build_fn)
    pipeline = make_pipeline(nn, StandardScaler())

    param_dists = {}
    param_dists['model__alpha'] = LogUniform(-6, 5)
    param_dists['model__epochs'] = randint(100, 10000)
    param_dists['model__batch_size'] = [8, 16, 32, 64, 128, 256]
    cv = RandomizedSearchCV(pipeline, param_dists, n_iter=10, n_jobs=-1, iid=False, scoring=make_scorer(r2_score))
    return cv

def make_model(model_name, hidden_layers, seed):
    validate.model_name(model_name)
    if model_name == 'sklearn_nn':
        return make_sklearn_nn(hidden_layers, seed=seed)
    elif model_name == 'keras':
        return make_keras_model(hidden_layers, seed=seed)
    else:
        raise NotImplementedError()


def train_model(model_name, train_size, test_size, seed, dataset, hidden_layers):
    validate.model_name(model_name)
    assert train_size > 0, f'train_size must be greater than 0, but is "{train_size}".'
    assert test_size > 0, f'test_size must be greater than 0, but is "{test_size}".'
    if type(hidden_layers) == str:
        hidden_layers = utils.str_to_list(hidden_layers, type_func=int)
    # Train the model.
    X_train, y_train, X_test, y_test = get_train_test_datasets(
        train_size, test_size, seed=seed, dataset=dataset)
    model = make_model(model_name, hidden_layers, seed=seed)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start
    print(f'Total training time: {training_time} s')

    result = {}
    # Add scores to the results for convenience.
    start = time.time()
    result['training_score'] = model.score(X_train, y_train)
    result['testing_score'] = model.score(X_test, y_test)
    testing_time = time.time() - start
    print(f'Total testing time: {testing_time} s')

    # Add timing to results.
    result['training_time'] = training_time
    result['testing_time'] = testing_time

    return (model, result)


def add_default_settings(config):
    defaults = {'train_size': 90000,
                'test_size': 1000, 'seed': 1, 'dataset': 'Wlnu_Wm_label'}
    for k, v in defaults.items():
        if k not in config:
            config[k] = v

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
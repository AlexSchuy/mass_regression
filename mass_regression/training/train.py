"""Training

These routines handle training the ML models that are studied (quantum and classical for comparison).
"""
import os
import shutil

import altair as alt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.distributions import randint
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import (MeanAbsoluteError,
                                      MeanAbsolutePercentageError,
                                      RootMeanSquaredError)
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import definitions
from training import data, train
from training.loguniform import LogUniform
from training.steploguniform import StepLogUniform
from training.stepuniform import StepUniform


def preprocess(x, mean, std):
    return (x - mean) / std


def build_v1_model(hparams, input_shape):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    for _ in range(hparams['num_layers']):
        model.add(layers.Dense(units=hparams['num_units'],
                               activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hparams['learning_rate']),
        loss='mean_squared_error',
        metrics=[MeanAbsolutePercentageError(), MeanAbsoluteError(), RootMeanSquaredError()])
    return model


def build_v2_model(hparams, input_shape, output_shape):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    for _ in range(hparams['num_layers']):
        model.add(layers.Dense(units=hparams['num_units'],
                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(layers.Dropout(rate=hparams['dropout']))
    model.add(layers.Dense(output_shape[0], dtype='float32'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hparams['learning_rate']),
        loss='mean_squared_error',
        metrics=[MeanAbsolutePercentageError(), MeanAbsoluteError(), RootMeanSquaredError()])
    return model


def train_test_model(build_fn, x, y, x_val, y_val, hparams, log_dir):
    mean = np.mean(x)
    std = np.std(x)
    x_val = preprocess(x_val, mean, std)
    model = build_fn(hparams, input_shape=x.shape[1:], output_shape=y.shape[1:])
    if 'epochs' in hparams:
        epochs = hparams['epochs']
    else:
        epochs = 10
    if 'batch_size' in hparams:
        batch_size = hparams['batch_size']
    else:
        batch_size = 32
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val),
              callbacks=[tf.keras.callbacks.TensorBoard(str(log_dir)), hp.KerasCallback(str(log_dir), hparams)])
    loss, mape, mae, rmse = model.evaluate(x_val, y_val)
    return model, loss, mape, mae, rmse


def random_search(build_fn, x, y, x_val, y_val, n, hp_rv, log_dir):
    best_model = None
    best_loss = float('Inf')
    for i in range(n):
        run_name = f'run{i}'
        run_dir = log_dir / run_name
        with tf.summary.create_file_writer(str(run_dir)).as_default():
            hparams = {k: v.rvs() for k, v in hp_rv.items()}
            hparams['run_num'] = i
            hp.hparams(hparams)
            model, loss, mape, mae, rmse = train_test_model(
                build_fn, x, y, x_val, y_val, hparams, run_dir)
            tf.summary.scalar('mean absolute percentage error', mape, step=1)
            tf.summary.scalar('mean absolute error', mae, step=1)
            tf.summary.scalar('root mean squared error', rmse, step=1)
            if loss < best_loss:
                best_model = model
                best_loss = loss
                best_model.save(str(log_dir / 'best_model.h5'))


def main():
    dataset = 'H125'
    target = 'nu'
    model_version = 'v2'

    log_dir = definitions.LOG_DIR / dataset / f'fix-{target}-{model_version}'
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True)
    hp_rv = {'num_layers': randint(1, 4),
             'num_units': StepUniform(start=10, num=20, step=10),
             'learning_rate': LogUniform(loc=-5, scale=4, base=10, discrete=False),
             'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
             'epochs': randint(10, 101),
             'dropout': StepUniform(start=0.0, num=2, step=0.5)}
    print(log_dir)

    x_train, y_train, x_val, y_val, x_test, y_test = data.get_datasets(
        dataset=dataset, target=target, scale_x=True, scale_y=True)
    train.random_search(build_fn=build_v2_model, x=x_train, y=y_train,
                        x_val=x_val, y_val=y_val, n=60, hp_rv=hp_rv, log_dir=log_dir)


if __name__ == '__main__':
    main()

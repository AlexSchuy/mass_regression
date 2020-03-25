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
from tensorflow.keras.losses import MSE
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
    model.add(layers.Dense(units=hparams['num_units'],
                           activation='relu', input_shape=input_shape))
    for _ in range(hparams['num_layers'] - 1):
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
    model.add(layers.Dense(units=hparams['num_units'],
                           activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape))
    model.add(layers.Dropout(rate=hparams['dropout']))
    for _ in range(hparams['num_layers'] - 1):
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


def calc_Wm(x, y_pred):
    METx = x[:, 0]
    METy = x[:, 1]
    Lax_gen = x[:, 2]
    Lay_gen = x[:, 3]
    Laz_gen = x[:, 4]
    Lam_gen = x[:, 5]
    Lbx_gen = x[:, 6]
    Lby_gen = x[:, 7]
    Lbz_gen = x[:, 8]
    Lbm_gen = x[:, 9]
    Nax_pred = y_pred[:, 0]
    Nay_pred = y_pred[:, 1]
    Naz_pred = y_pred[:, 2]
    Nam_pred = tf.zeros_like(Nax_pred)
    Nbx_pred = METx - Nax_pred
    Nby_pred = METy - Nay_pred
    Nbz_pred = y_pred[:, 3]
    Nbm_pred = tf.zeros_like(Nbx_pred)
    _, _, _, Wam_pred = data.add_fourvectors(
        Nax_pred, Nay_pred, Naz_pred, Nam_pred, Lax_gen, Lay_gen, Laz_gen, Lam_gen)
    _, _, _, Wbm_pred = data.add_fourvectors(
        Nbx_pred, Nby_pred, Nbz_pred, Nbm_pred, Lbx_gen, Lby_gen, Lbz_gen, Lbm_gen)
    return tf.stack([Wam_pred, Wbm_pred])


def make_mixed_loss(x, mix_weight, dataset, target, pad_target):
    if dataset == 'H125' and target == 'nu' and pad_target == 'W':
        calc_pad = calc_Wm
    else:
        raise NotImplementedError()
    scale_y_pad, unscale_y = data.get_scale_funcs(dataset, target, pad_target)

    num_targets = data.get_num_targets(dataset, target)

    def mixed_loss(padded_y_true, padded_y_pred):
        y_true = padded_y_true[:, :num_targets]
        y_true_pad = padded_y_true[:, num_targets:]
        y_pred = padded_y_pred[:, :num_targets]
        y_pred_pad = scale_y_pad(calc_pad(x, unscale_y(y_pred)))
        return MSE(y_true, y_pred) + mix_weight * MSE(y_true_pad, y_pred_pad)
    return mixed_loss


def make_unpadded_MAPE_metric(dataset, target):
    num_targets = data.get_num_targets(dataset, target)

    def unpadded_MAPE(padded_y_true, padded_y_pred):
        y_true = padded_y_true[:, :num_targets]
        y_pred = padded_y_pred[:, :num_targets]
        return tf.keras.losses.MAPE(y_true, y_pred)
    return unpadded_MAPE


def make_unpadded_MAE_metric(dataset, target):
    num_targets = data.get_num_targets(dataset, target)

    def unpadded_MAE(padded_y_true, padded_y_pred):
        y_true = padded_y_true[:, :num_targets]
        y_pred = padded_y_pred[:, :num_targets]
        return tf.keras.losses.MAE(y_true, y_pred)
    return unpadded_MAE


def make_unpadded_RMSE_metric(dataset, target):
    num_targets = data.get_num_targets(dataset, target)

    def unpadded_RMSE(padded_y_true, padded_y_pred):
        y_true = padded_y_true[:, :num_targets]
        y_pred = padded_y_pred[:, :num_targets]
        return tf.keras.losses.MSE(y_true, y_pred)**(1/2)
    return unpadded_RMSE


def make_mixed_model_build_fn(dataset, target, pad_target, mix_weight):
    num_pad_targets = data.get_num_pad_targets(dataset, pad_target)
    num_targets = data.get_num_targets(dataset, target)

    def build_mixed_model(hparams, input_shape, output_shape):
        model = keras.Sequential()
        model.add(layers.Dense(units=hparams['num_units'],
                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape))
        model.add(layers.Dropout(rate=hparams['dropout']))
        for _ in range(hparams['num_layers'] - 1):
            model.add(layers.Dense(units=hparams['num_units'],
                                   activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(layers.Dropout(rate=hparams['dropout']))
        model.add(layers.Dense(num_targets, dtype='float32'))
        model.add(layers.Lambda(lambda x: tf.pad(
            x, [[0, 0], [0, num_pad_targets]])))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hparams['learning_rate']),
            loss=make_mixed_loss(model.input, mix_weight,
                                 dataset, target, pad_target),
            metrics=[make_unpadded_MAPE_metric(dataset, target), make_unpadded_MAE_metric(dataset, target), make_unpadded_RMSE_metric(dataset, target)])
        return model
    return build_mixed_model


def train_test_model(build_fn, x, y, x_val, y_val, hparams, log_dir):
    model = build_fn(
        hparams, input_shape=x.shape[1:], output_shape=y.shape[1:])
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


def get_mixed_log_dir(dataset, target, pad_target, model_version, mix_weight):
    return definitions.LOG_DIR / dataset / f'{target}-{pad_target}-{model_version}-w{mix_weight}'


def run_mixed(dataset, target, pad_target, model_version):
    for mix_weight in (0.0, 1.0, 10.0):
        log_dir = get_mixed_log_dir(
            dataset, target, pad_target, model_version, mix_weight)
        shutil.rmtree(log_dir, ignore_errors=True)
        log_dir.mkdir(parents=True)
        hp_rv = {'num_layers': randint(1, 4),
                 'num_units': StepUniform(start=10, num=20, step=10),
                 'learning_rate': LogUniform(loc=-5, scale=4, base=10, discrete=False),
                 'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
                 'epochs': randint(10, 301),
                 'dropout': StepUniform(start=0.0, num=2, step=0.5)}
        print(log_dir)

        x_train, y_train, x_val, y_val, _, _ = data.get_datasets(
            dataset=dataset, target=target, scale_x=True, scale_y=True, pad_target=pad_target)
        train.random_search(build_fn=make_mixed_model_build_fn(dataset, target, pad_target, mix_weight), x=x_train, y=y_train,
                            x_val=x_val, y_val=y_val, n=60, hp_rv=hp_rv, log_dir=log_dir)


def main():
    dataset = 'H125'
    target = 'nu'
    pad_target = 'W'
    model_version = 'mixed_v1'

    run_mixed(dataset, target, pad_target, model_version)


if __name__ == '__main__':
    main()

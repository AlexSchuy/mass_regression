"""Training

These routines handle training the ML models that are studied (quantum and classical for comparison).
"""
import math
import os
import shutil
import time

import altair as alt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats.distributions import randint
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers  # pylint: disable=import-error
from tensorflow.keras.losses import MSE  # pylint: disable=import-error
from tensorflow.keras.metrics import \
    MeanAbsoluteError  # pylint: disable=import-error
from tensorflow.keras.metrics import \
    MeanAbsolutePercentageError  # pylint: disable=import-error
from tensorflow.keras.metrics import \
    RootMeanSquaredError  # pylint: disable=import-error
from tensorflow.keras.mixed_precision import experimental as mixed_precision  # pylint: disable=import-error

import definitions
from training import data, train
from training.loguniform import LogUniform
from training.steploguniform import StepLogUniform
from training.stepuniform import StepUniform


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
    return tf.stack([Wam_pred, Wbm_pred], axis=1)


class MixedLoss():

    def __init__(self, mix_weight, dataset, target, pad_target, name='MixedLoss', mixed=True):
        if dataset == 'H125' and target == 'nu' and pad_target == 'W':
            self.calc_pad = calc_Wm
        else:
            raise NotImplementedError()
        scale_y_pad, unscale_y = data.get_scale_funcs(
            dataset, target, pad_target)
        self.num_targets = data.get_num_targets(dataset, target)
        self.scale_y_pad = scale_y_pad
        self.unscale_y = unscale_y
        self.mix_weight = mix_weight
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mixed = mixed  # testing parameter

    def __call__(self, x, padded_y_true, padded_y_pred):
        y_true = padded_y_true[:, :self.num_targets]
        y_pred = padded_y_pred[:, :self.num_targets]
        if not self.mixed:
            return self.mse(y_true, y_pred)
        else:
            y_true_pad = padded_y_true[:, self.num_targets:]
            y_pred_pad = self.scale_y_pad(
                self.calc_pad(x, self.unscale_y(y_pred)))
            return self.mse(y_true, y_pred) + self.mix_weight * self.mse(y_true_pad, y_pred_pad)


class MixedLossV2():
    def __init__(self, x, mix_weight, dataset, target, pad_target, name='MixedLossV2', mixed=True):
        if dataset == 'H125' and target == 'nu' and pad_target == 'W':
            self.calc_pad = calc_Wm
        else:
            raise NotImplementedError()
        scale_y_pad, unscale_y = data.get_scale_funcs(
            dataset, target, pad_target)
        self.x = x
        self.num_targets = data.get_num_targets(dataset, target)
        self.scale_y_pad = scale_y_pad
        self.unscale_y = unscale_y
        self.mix_weight = mix_weight
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mixed = mixed  # testing parameter

    def __call__(self, padded_y_true, padded_y_pred):
        y_true = padded_y_true[:, :self.num_targets]
        y_pred = padded_y_pred[:, :self.num_targets]
        if not self.mixed:
            return self.mse(y_true, y_pred)
        else:
            y_true_pad = padded_y_true[:, self.num_targets:]
            y_pred_pad = self.scale_y_pad(
                self.calc_pad(self.x, self.unscale_y(y_pred)))
            return self.mse(y_true, y_pred) + self.mix_weight * self.mse(y_true_pad, y_pred_pad)


class paddedMAPE(tf.keras.metrics.MeanAbsolutePercentageError):
    def __init__(self, dataset, target, name='mape', **kwargs):
        super(paddedMAPE, self).__init__(name=name, **kwargs)
        self.num_targets = data.get_num_targets(dataset, target)

    def update_state(self, padded_y_true, padded_y_pred, sample_weight=None):
        y_true = padded_y_true[:, :self.num_targets]
        y_pred = padded_y_pred[:, :self.num_targets]
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return super().result()


class paddedMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, dataset, target, name='mae', **kwargs):
        super(paddedMAE, self).__init__(name=name, **kwargs)
        self.num_targets = data.get_num_targets(dataset, target)

    def update_state(self, padded_y_true, padded_y_pred, sample_weight=None):
        y_true = padded_y_true[:, :self.num_targets]
        y_pred = padded_y_pred[:, :self.num_targets]
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return super().result()


class paddedRMSE(tf.keras.metrics.RootMeanSquaredError):
    def __init__(self, dataset, target, name='rmse', **kwargs):
        super(paddedRMSE, self).__init__(name=name, **kwargs)
        self.num_targets = data.get_num_targets(dataset, target)

    def update_state(self, padded_y_true, padded_y_pred, sample_weight=None):
        y_true = padded_y_true[:, :self.num_targets]
        y_pred = padded_y_pred[:, :self.num_targets]
        super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return super().result()


def preprocess(x, mean, std):
    return (x - mean) / std


def build_v1_model(hparams, input_shape, seed=None):
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


def build_unmixed_model(hparams, input_shape, output_shape, seed=None):
    x = tf.keras.layers.Input(shape=input_shape)
    i = layers.Dense(units=hparams['num_units'],
                     activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape)(x)
    i = layers.Dropout(rate=hparams['dropout'], seed=seed)(i)
    for _ in range(hparams['num_layers'] - 1):
        i = layers.Dense(units=hparams['num_units'],
                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(i)
        i = layers.Dropout(rate=hparams['dropout'], seed=seed)(i)
    y_pred = layers.Dense(output_shape[0], dtype='float32')(i)
    model = tf.keras.models.Model(inputs=x, outputs=y_pred)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hparams['learning_rate']),
        loss='mean_squared_error',
        metrics=[MeanAbsolutePercentageError(), MeanAbsoluteError(), RootMeanSquaredError()])
    print(model.summary())
    return model


def make_mixed_model_build_fn(dataset, target, pad_target, mix_weight):
    num_targets = data.get_num_targets(dataset, target)

    def build_mixed_model(hparams, input_shape, output_shape, seed=None):
        x = tf.keras.layers.Input(shape=input_shape)
        i = layers.Dense(units=hparams['num_units'],
                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape)(x)
        i = layers.Dropout(rate=hparams['dropout'], seed=seed)(i)
        for _ in range(hparams['num_layers'] - 1):
            i = layers.Dense(units=hparams['num_units'],
                             activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(i)
            i = layers.Dropout(rate=hparams['dropout'], seed=seed)(i)
        y_pred = layers.Dense(num_targets, dtype='float32')(i)
        model = tf.keras.models.Model(inputs=x, outputs=y_pred)
        model.compile()
        print(model.summary())
        return model
    return build_mixed_model


def make_mixed_model_build_fn_v2(dataset, target, pad_target, mix_weight, mixed=True):
    num_targets = data.get_num_targets(dataset, target)
    num_pad_targets = data.get_num_pad_targets(dataset, pad_target)

    def build_mixed_model(hparams, input_shape, output_shape, seed=None):
        x = tf.keras.layers.Input(shape=input_shape)
        i = layers.Dense(units=hparams['num_units'],
                         activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape)(x)
        i = layers.Dropout(rate=hparams['dropout'], seed=seed)(i)
        for _ in range(hparams['num_layers'] - 1):
            i = layers.Dense(units=hparams['num_units'],
                             activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(i)
            i = layers.Dropout(rate=hparams['dropout'], seed=seed)(i)
        y_pred = layers.Dense(num_targets, dtype='float32')(i)
        padded_y_pred = layers.Lambda(
            lambda x: tf.pad(x, [[0, 0], [0, num_pad_targets]]))(y_pred)
        padded_y_true = tf.keras.layers.Input(num_targets + num_pad_targets)
        model = tf.keras.models.Model(
            inputs=[x, padded_y_true], outputs=padded_y_pred)
        model.add_loss(MixedLossV2(x, mix_weight, dataset, target,
                                   pad_target, mixed=mixed)(padded_y_true, padded_y_pred))

        model.compile(optimizer=tf.keras.optimizers.Adam(
            hparams['learning_rate']))
        print(model.summary())
        return model
    return build_mixed_model


def custom_fit(model, loss_fn, optimizer, metrics_fn, x, y, epochs, batch_size, x_val, y_val, callbacks):
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

    num_steps = math.ceil(x.shape[0] / batch_size)
    batch_train_metrics = metrics_fn()
    epoch_train_metrics = metrics_fn()
    val_metrics = metrics_fn()

    on_train_batch_begin_logs = {'batch': 0, 'size': batch_size}
    on_train_batch_end_logs = {'size': batch_size, 'num_steps': 1}
    on_epoch_end_logs = {}

    params = {'batch_size': batch_size, 'epochs': epochs, 'steps': num_steps, 'samples': x.shape[0], 'verbose': 1, 'do_validation': True, 'metrics': [
        'loss'] + [m.name for m in epoch_train_metrics] + [f'val_{m.name}' for m in val_metrics]}

    for callback in callbacks:
        callback.set_model(model)
        callback.set_params(params)
        callback.on_train_begin()

    for epoch in range(epochs):
        on_epoch_end_logs['loss'] = 0.0
        on_epoch_end_logs['val_loss'] = 0.0
        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            on_train_batch_begin_logs['batch'] = step
            for callback in callbacks:
                callback.on_train_batch_begin(step, on_train_batch_begin_logs)

            with tf.GradientTape() as tape:
                y_batch_pred = model(x_batch_train, training=True)
                loss_batch = loss_fn(
                    x_batch_train, y_batch_train, y_batch_pred)
            grads = tape.gradient(loss_batch, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            on_train_batch_end_logs['loss'] = loss_batch.numpy()
            on_epoch_end_logs['loss'] += loss_batch.numpy()

            for train_metric in batch_train_metrics:
                train_metric(y_batch_train, y_batch_pred)
                on_train_batch_end_logs[train_metric.name] = train_metric.result(
                ).numpy()
                train_metric.reset_states()

            for train_metric in epoch_train_metrics:
                train_metric(y_batch_train, y_batch_pred)

            for callback in callbacks:
                callback.on_train_batch_end(step, on_train_batch_end_logs)

        for train_metric in epoch_train_metrics:
            result = train_metric.result().numpy()
            on_epoch_end_logs[train_metric.name] = result

        for x_batch_val, y_batch_val in val_dataset:
            y_batch_pred = model(x_batch_val)
            on_epoch_end_logs['val_loss'] += loss_fn(
                x_batch_val, y_batch_val, y_batch_pred)
            for val_metric in val_metrics:
                val_metric(y_batch_val, y_batch_pred)
        for val_metric in val_metrics:
            result = val_metric.result().numpy()
            on_epoch_end_logs[f'val_{val_metric.name}'] = result

        for callback in callbacks:
            callback.on_epoch_end(epoch, on_epoch_end_logs)

    for callback in callbacks:
        callback.on_train_end()

    y_val_pred = model(x_val)
    val_loss = float(loss_fn(x_val, y_val, y_val_pred))
    print(f'Validation loss: {val_loss}')

    return val_loss


def train_test_model(build_fn, x, y, x_val, y_val, hparams, log_dir, loss_fn, optimizer_fn, metrics_fn, concat_input=False):
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
    if loss_fn is not None:
        assert optimizer_fn is not None
        assert metrics_fn is not None
        optimizer = optimizer_fn(hparams)
        val_loss = custom_fit(model, loss_fn, optimizer, metrics_fn, x, y,
                              epochs, batch_size, x_val, y_val, callbacks=[tf.keras.callbacks.TensorBoard(str(log_dir)), hp.KerasCallback(str(log_dir), hparams), tf.keras.callbacks.ProgbarLogger()])
    else:
        assert optimizer_fn is None
        assert metrics_fn is None
        if not concat_input:
            model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[
                tf.keras.callbacks.TensorBoard(str(log_dir)), hp.KerasCallback(str(log_dir), hparams)])
        else:
            model.fit([x, y], epochs=epochs, batch_size=batch_size, callbacks=[
                tf.keras.callbacks.TensorBoard(str(log_dir)), hp.KerasCallback(str(log_dir), hparams)])
        val_loss, _, _, _ = model.evaluate(x_val, y_val)

    return model, val_loss


def random_search(build_fn, x, y, x_val, y_val, n, hp_rv, log_dir, loss_fn=None, optimizer_fn=None, metrics_fn=None, concat_input=False):
    best_model = None
    best_loss = float('Inf')
    for i in range(n):
        run_name = f'run{i}'
        run_dir = log_dir / run_name
        with tf.summary.create_file_writer(str(run_dir)).as_default():
            hparams = {k: v.rvs() for k, v in hp_rv.items()}
            hparams['run_num'] = i
            hp.hparams(hparams)
            model, loss = train_test_model(
                build_fn, x, y, x_val, y_val, hparams, run_dir, loss_fn, optimizer_fn, metrics_fn, concat_input=concat_input)
            if loss < best_loss:
                best_model = model
                best_loss = loss
                best_model.save(str(log_dir / 'best_model.h5'))


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_unmixed_log_dir(dataset, target, model_version):
    return definitions.LOG_DIR / dataset / f'{target}-{model_version}'


def run_unmixed(dataset, target, model_version, seed=None):
    set_seed(seed)
    log_dir = get_unmixed_log_dir(dataset, target, model_version)
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True)
    print(log_dir)
    hp_rv = {'num_layers': randint(1, 3),
             'num_units': StepUniform(start=10, num=20, step=10),
             'learning_rate': LogUniform(loc=-5, scale=4, base=10, discrete=False),
             'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
             'epochs': randint(10, 201),
             'dropout': StepUniform(start=0.0, num=2, step=0.5)}
    x_train, y_train, x_val, y_val, _, _ = data.get_datasets(
        dataset=dataset, target=target, scale_x=True, scale_y=True)
    if model_version == 'v1':
        build_fn = build_v1_model
    elif model_version == 'v2':
        build_fn = build_unmixed_model
    elif model_version == 'v3':
        pass
    else:
        raise NotImplementedError(
            f'Unsupported model_version = {model_version}')
    train.random_search(build_fn=build_fn, x=x_train.to_numpy(), y=y_train.to_numpy(
    ), x_val=x_val.to_numpy(), y_val=y_val.to_numpy(), n=20, hp_rv=hp_rv, log_dir=log_dir)


def get_mixed_log_dir(dataset, target, pad_target, model_version, mix_weight):
    return definitions.LOG_DIR / dataset / f'{target}-{pad_target}-mixed_{model_version}-w{mix_weight}'


def run_mixed(dataset, target, pad_target, model_version, seed=None, mixed=True):
    set_seed(seed)
    for mix_weight in (0.0, 1.0, 10.0):
        log_dir = get_mixed_log_dir(
            dataset, target, pad_target, model_version, mix_weight)
        shutil.rmtree(log_dir, ignore_errors=True)
        log_dir.mkdir(parents=True)
        print(log_dir)
        hp_rv = {'num_layers': randint(1, 3),
                 'num_units': StepUniform(start=10, num=20, step=10),
                 'learning_rate': LogUniform(loc=-5, scale=4, base=10, discrete=False),
                 'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
                 'epochs': randint(10, 201),
                 'dropout': StepUniform(start=0.0, num=2, step=0.5)}

        x_train, y_train, x_val, y_val, _, _ = data.get_datasets(
            dataset=dataset, target=target, scale_x=True, scale_y=True, pad_target=pad_target)
        if model_version == 'v1':
            build_fn = make_mixed_model_build_fn(
                dataset, target, pad_target, mix_weight)
            loss_fn = MixedLoss(mix_weight, dataset, target,
                                pad_target, mixed=mixed)
            def optimizer_fn(a): return tf.keras.optimizers.Adam(
                a['learning_rate'])
            def metrics_fn(): return [paddedMAE(dataset, target), paddedMAPE(
                dataset, target), paddedRMSE(dataset, target)]
            concat_input = False
        elif model_version == 'v2':
            build_fn = make_mixed_model_build_fn_v2(
                dataset, target, pad_target, mix_weight, mixed=mixed)
            loss_fn = None
            optimizer_fn = None
            metrics_fn = None
            concat_input = True
        else:
            raise NotImplementedError(
                f'Unsupported model_version = {model_version}')
        train.random_search(build_fn=build_fn,
                            loss_fn=loss_fn, optimizer_fn=optimizer_fn, metrics_fn=metrics_fn,
                            x=x_train.to_numpy(), y=y_train.to_numpy(), x_val=x_val.to_numpy(), y_val=y_val.to_numpy(),
                            n=1, hp_rv=hp_rv, log_dir=log_dir, concat_input=concat_input)


def main():
    dataset = 'H125'
    target = 'nu'
    pad_target = 'W'
    unmixed_model_version = 'v2'
    mixed_model_version = 'v2'
    seed = 5
    mixed = True
    use_mixed = True # debug variable to compare timing using custom loss function with (mixed=true) and without (mixed=false) W_mass calculations.

    if not mixed:
        run_unmixed(dataset, target, unmixed_model_version, seed)
    else:
        run_mixed(dataset, target, pad_target,
                  mixed_model_version, seed, mixed=use_mixed)


if __name__ == '__main__':
    main()

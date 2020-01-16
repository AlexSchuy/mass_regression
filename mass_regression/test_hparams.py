""" A simple helloworld example
Different workflows are shown here.
"""
import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import definitions
from training import train
from training.loguniform import LogUniform
from training.stepuniform import StepUniform
from training.steploguniform import StepLogUniform
from scipy.stats.distributions import randint

(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.

x = x[:10000]
y = y[:10000]

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

"""Basic case:
- We define a `build_model` function
- It returns a compiled model
- It uses hyperparameters defined on the fly
"""


def build_model(hparams):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for _ in range(hparams['num_layers']):
        model.add(layers.Dense(units=hparams['num_units'],
                               activation='relu'))
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax', dtype='float32'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hparams['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def train_test_model(hparams, log_dir):
    model = build_model(hparams)
    model.fit(x, y, epochs=5, batch_size=128, validation_data=(val_x, val_y),
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir), hp.KerasCallback(log_dir, hparams)])
    _, accuracy = model.evaluate(val_x, val_y)
    return accuracy


def grid_search():
    HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete(list(range(2, 5))))
    HP_NUM_UNITS = hp.HParam(
        'num_units', hp.Discrete(list(range(1024, 8096, 1024))))
    HP_LEARNING_RATE = hp.HParam(
        'learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))

    log_dir = definitions.ROOT_DIR / 'logs' / 'hparam_tuning'

    with tf.summary.create_file_writer(str(log_dir)).as_default():
        hp.hparams_config(
            hparams=[HP_NUM_LAYERS, HP_NUM_UNITS, HP_LEARNING_RATE],
            metrics=[hp.Metric('accuracy', display_name='Accuracy')]
        )

    session_num = 0
    for num_layers in HP_NUM_LAYERS.domain.values:
        for num_units in HP_NUM_UNITS.domain.values:
            for learning_rate in HP_LEARNING_RATE.domain.values:
                run_name = f'run{session_num}'
                run_dir = log_dir / run_name
                hparams = {'num_layers': num_layers,
                           'num_units': num_units, 'learning_rate': learning_rate}
                accuracy = train_test_model(hparams, str(run_dir))
                session_num += 1
                with tf.summary.create_file_writer(str(run_dir)).as_default():
                    tf.summary.scalar('accuracy', accuracy, step=1)


def random_search():
    log_dir = definitions.LOG_DIR / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    hp_rv = {'num_layers': randint(2, 5),
             'num_units': StepUniform(start=128, num=10, step=128),
             'learning_rate': LogUniform(loc=-4, scale=2, base=10, discrete=False),
             'batch_size': StepLogUniform(start=5, num=4, step=1, base=2),
             'epochs': randint(10, 100)}
    train.random_search(build_fn=build_model, x=x, y=y, val_x=val_x, val_y=val_y, n=10, hp_rv=hp_rv, log_dir=log_dir)


random_search()

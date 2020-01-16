""" A simple helloworld example
Different workflows are shown here.
"""
import datetime
import pathlib

from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters
from kerastuner.tuners import RandomSearch
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import definitions

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 512, 32),
                               activation='relu'))
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax', dtype='float32'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


class MyTuner(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Int(
            'batch_size', 32, 256, step=32)
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 100)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)


def random_search():
    (x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
    x = x.astype('float32') / 255.
    val_x = val_x.astype('float32') / 255.

    x = x[:10000]
    y = y[:10000]

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_dir = definitions.LOG_DIR / timestamp
    log_dir.mkdir()

    tuner = MyTuner(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=3,
        directory=log_dir,
        project_name=timestamp,
        overwrite=True)

    tuner.search_space_summary()

    tuner.search(x=x,
                 y=y,
                 validation_data=(val_x, val_y))

    tuner.results_summary()

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()


random_search()

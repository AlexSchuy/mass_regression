""" A simple helloworld example
Different workflows are shown here.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorboard.plugins.hparams import api as hp
import definitions

(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.

x = x[:10000]
y = y[:10000]


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
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hparams['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def train_test_model(hparams, log_dir):
    model = build_model(hparams)
    model.fit(x, y, epochs=10, validation_data=(val_x, val_y),
              callbacks=[tf.keras.callbacks.TensorBoard(log_dir), hp.KerasCallback(log_dir, hparams)])
    _, accuracy = model.evaluate(val_x, val_y)
    return accuracy


HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete(list(range(2, 11))))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete(list(range(32, 513, 32))))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))

log_dir = definitions.ROOT_DIR / 'logs' / 'hparam_tuning'

with tf.summary.create_file_writer(str(log_dir)).as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_LEARNING_RATE],
        metrics=[hp.Metric('accuracy', display_name='Accuracy')]
    )

session_num = 0
for num_layers in HP_NUM_LAYERS.domain.values:
    for num_units in HP_NUM_UNITS.domain.values:
        for learning_rate in HP_LEARNING_RATE.domain.values:
            run_name = f'run{session_num}'
            run_dir = log_dir / run_name
            hparams = {'num_layers': num_layers, 'num_units': num_units, 'learning_rate': learning_rate}
            train_test_model(hparams, str(run_dir))
            session_num += 1

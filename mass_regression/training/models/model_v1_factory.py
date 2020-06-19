
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # pylint: disable=import-error

from training.base.base_loss import BaseLoss
from training.models.base_model_factory import BaseModelFactory



class Model_V1_Factory(BaseModelFactory):

    def __init__(self, dataset, loss, seed=0):
        self.loss = loss
        super().__init__(dataset, seed, name=f'model_v1-{loss.name}')

    def build(self, hparams):
        x = tf.keras.layers.Input(shape=self.dataset.num_features)
        x_pad = tf.keras.layers.Input(shape=self.dataset.num_pad_features)
        i = layers.Dense(units=hparams['num_units'], activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        i = layers.Dropout(rate=hparams['dropout'], seed=self.seed)(i)
        for _ in range(hparams['num_layers'] - 1):
            i = layers.Dense(units=hparams['num_units'], activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.01))(i)
            i = layers.Dropout(rate=hparams['dropout'], seed=self.seed)(i)
        y_pred = layers.Dense(self.dataset.num_targets, dtype='float32')(i)
        y_true = layers.Input(self.dataset.num_targets)
        model = tf.keras.models.Model(inputs=[x, x_pad, y_true], outputs=y_pred)
        model.add_loss(self.loss(x, x_pad, self.dataset)(y_true, y_pred))
        model.compile(optimizer=keras.optimizers.Adam(hparams['learning_rate']))
        print(model.summary())
        return model

    def load(self, run_dir):
        with next(run_dir.glob('*.json')).open() as f:
            hparams = json.load(f)
        model = self.build(hparams)
        model.load_weights(str(run_dir / 'model'))
        return hparams, model

    @property
    def concatenated_input(self):
        return True


from training.models.base_model import BaseModel
import tensorflow as tf
from training.custom_loss import CustomLoss
from tensorflow import keras
from tensorflow.keras import layers  # pylint: disable=import-error

class Model_V1(BaseModel):

    def __init__(self, dataset, loss_fn, seed=0):
        self.loss_fn = loss_fn
        super().__init__(dataset, seed)

    def build(self, hparams):
        x = tf.keras.layers.Input(shape=self.dataset.num_features)
        x_pad = tf.keras.layers.Input(shape=self.dataset.num_pad_features)
        i = layers.Dense(units=hparams['num_units'],
                        activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=self.dataset.num_features)(x)
        i = layers.Dropout(rate=hparams['dropout'], seed=self.seed)(i)
        for _ in range(hparams['num_layers'] - 1):
            i = layers.Dense(units=hparams['num_units'],
                            activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(i)
            i = layers.Dropout(rate=hparams['dropout'], seed=self.seed)(i)
        y_pred = layers.Dense(self.dataset.num_targets, dtype='float32')(i)
        y_true = layers.Input(self.dataset.num_targets)
        model = tf.keras.models.Model(inputs=[x, x_pad, y_true], outputs=y_pred)
        model.add_loss(CustomLoss(x, x_pad, self.dataset, self.loss_fn)(y_true, y_pred))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hparams['learning_rate']))
        print(model.summary())
        return model

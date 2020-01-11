import copy
import io

import h5py

import tensorflow as tf


class KerasRegressor(tf.keras.wrappers.scikit_learn.KerasRegressor):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasRegressor serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    """

    def __getstate__(self):
        state = self.__dict__
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio) as file:
                model.save(file)
                # tf.keras.models.save_model(model, file, save_format="h5")
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def __setstate__(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio) as file:
                state["model"] = tf.keras.models.load_model(file)
        self.__dict__ = state
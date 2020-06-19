import tensorflow as tf 
import numpy as np

class DeltaCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir):
        self.dataset = dataset
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        epochs = self.params['epochs']
        self.delta = {}
        self.delta_Wm_train = np.zeros((epochs, self.dataset.num_training_samples))
        self.delta_NUz_train = np.zeros((epochs, self.dataset.num_training_samples))
        self.delta_Wm_val = np.zeros((epochs, self.dataset.num_val_samples))
        self.delta_NUz_val = np.zeros((epochs, self.dataset.num_val_samples))

    def on_train_end(self, logs=None):
        def _save(file_name, array):
            np.save(self.log_dir / file_name, array)
        _save('delta_Wm_train.npy', self.delta_Wm_train)
        _save('delta_NUz_train.npy', self.delta_NUz_train)
        _save('delta_Wm_val.npy', self.delta_Wm_val)
        _save('delta_NUz_val.npy', self.delta_NUz_val)

    def on_epoch_end(self, epoch, logs=None):
        def calc_delta(get_data, suffix):
            x, x_pad, y = get_data()
            y_pred = self.dataset.unscale_y(
                self.model.predict((x, x_pad, y)))
            tree_pred = self.dataset.calculate_tree(self.dataset.unscale_x(x), y_pred)
            tree_gen = self.dataset.tree_gen
            for variable in tree_pred:
                delta = tree_pred[variable] - tree_gen[variable]
                self.delta[f'{variable}_{suffix}'] = delta
            return tree_pred - tree_gen
        calc_delta(self.dataset.train, 'train')
        calc_delta(self.dataset.val, 'val')
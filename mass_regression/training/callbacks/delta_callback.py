import tensorflow as tf 
import numpy as np

class DeltaCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir):
        self.dataset = dataset
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
       
        self.delta = {}

    def on_train_end(self, logs=None):
        def _save(file_name, array):
            np.save(self.log_dir / file_name, array)
        for variable in self.delta:
            _save(f'delta_{variable}.npy', self.delta[variable])

    def on_epoch_end(self, epoch, logs=None):
        def calc_delta(get_data, suffix):
            x, x_pad, y = get_data()
            y_pred = self.dataset.unscale_y(
                self.model.predict((x, x_pad, y)))
            tree_pred = self.dataset.calculate_tree(self.dataset.unscale_x(x), y_pred)
            tree_gen = self.dataset.tree_gen(partition=suffix)
            for variable in tree_pred:
                if epoch == 0:
                    if suffix == 'train':
                        num_samples = self.dataset.num_training_samples
                    elif suffix == 'val':
                        num_samples = self.dataset.num_val_samples
                    epochs = self.params['epochs']
                    self.delta[f'{variable}_{suffix}'] = np.zeros((epochs, num_samples))
                delta = tree_pred[variable] - tree_gen[variable]
                self.delta[f'{variable}_{suffix}'][epoch, :] = delta
            return tree_pred - tree_gen
        calc_delta(self.dataset.train, 'train')
        calc_delta(self.dataset.val, 'val')
import tensorflow as tf


class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    def __init__(self, dataset, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        self.dataset = dataset

    def get_monitor_value(self, logs):
        x_val, x_pad_val, y_val = self.dataset.val()
        val_loss = self.model.evaluate((x_val, x_pad_val, y_val))
        print(f'val_loss = {val_loss}')
        return val_loss

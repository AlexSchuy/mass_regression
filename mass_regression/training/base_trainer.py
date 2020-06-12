import json

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


class BaseTrainer():

    def __init__(self, model, dataset, log_dir, concat_input=False):
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir
        self.concat_input = concat_input

    def random_search(self):
        best_model = None
        best_loss = float('Inf')
        for i in range(n):
            run_name = f'run{i}'
            run_dir = self.log_dir / run_name
            with tf.summary.create_file_writer(str(run_dir)).as_default():
                hparams = {k: v.rvs() for k, v in hp_rv.items()}
                hparams['run_num'] = i
                hp.hparams(hparams)
                model, loss = self.train_test_model(hparams)
                if loss < best_loss:
                    best_model = model
                    best_loss = loss
                    best_model.save_weights(
                        str(self.log_dir / 'best_model'), save_format='tf')
                    with (self.log_dir / 'best_hparams.json').open('w') as f:
                        json.dump(hparams, f)

    def train_test_model(self, hparams):
        model = self.model.build(hparams)
        if 'epochs' in hparams:
            epochs = hparams['epochs']
        else:
            epochs = 10
        if 'batch_size' in hparams:
            batch_size = hparams['batch_size']
        else:
            batch_size = 32
        x, y = self.dataset.train()
        x_val, y_val = self.dataset.val()
        if not self.concat_input:
            model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=[
                tf.keras.callbacks.TensorBoard(str(self.log_dir)), hp.KerasCallback(str(self.log_dir), hparams)])
        else:
            model.fit([x, y], epochs=epochs, batch_size=batch_size, callbacks=[
                tf.keras.callbacks.TensorBoard(str(self.log_dir)), hp.KerasCallback(str(self.log_dir), hparams)])
        val_loss = model.evaluate([x_val, y_val])

        return model, val_loss

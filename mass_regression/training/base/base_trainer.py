import json

import tensorflow as tf
from sklearn.utils import shuffle
from tensorboard.plugins.hparams import api as hp

import definitions
from training.callbacks.early_stopping import EarlyStopping


class BaseTrainer():

    def __init__(self, model_factory, dataset, log_dir=None, early_stopping=False, callbacks=None):
        self.model_factory = model_factory
        self.dataset = dataset
        if log_dir is None:
            log_dir = f'{self.dataset.name}/{self.dataset.target_name}/{self.model_factory.name}'
        self.log_dir = definitions.LOG_DIR / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = []
        if callbacks is not None:
            self.callbacks.extend(callbacks)
        if early_stopping:
            self.callbacks.append(EarlyStopping(dataset=dataset, patience=10, restore_best_weights=True))

    def random_search(self, n, hp_rv, hot_start=False):
        save_dir = self.log_dir / 'best'
        save_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float('inf')
        if hot_start:
            hparams_path = save_dir / 'best_hparams.json'
            if hparams_path.exists():
                with (save_dir / 'best_hparams.json').open('r') as f:
                    hparams = json.load(f)
                if 'best_loss' in hparams:
                    best_loss = hparams['best_loss']
        for i in range(n):
            run_name = f'run{i}'
            run_dir = self.log_dir / run_name
            with tf.summary.create_file_writer(str(run_dir)).as_default():  # pylint: disable=not-context-manager
                hparams = {k: v.rvs() for k, v in hp_rv.items()}
                hparams['run_num'] = i
                hp.hparams(hparams)
                model, loss = self.train_test_model(hparams, run_dir)
                if loss < best_loss:
                    best_loss = loss
                    model.save_weights(
                        str(save_dir / 'best_model'), save_format='tf')
                    with (save_dir / 'best_hparams.json').open('w') as f:
                        json.dump(hparams, f)

    def train_test_model(self, hparams, run_dir):
        model = self.model_factory.build(hparams)
        if 'epochs' in hparams:
            epochs = hparams['epochs']
        else:
            epochs = 10
        if 'batch_size' in hparams:
            batch_size = hparams['batch_size']
        else:
            batch_size = 32
        x, x_pad, y = self.dataset.train()
        x, x_pad, y = shuffle(x, x_pad, y)
        x_val, x_pad_val, y_val = self.dataset.val()
        callbacks = self.callbacks + [
            tf.keras.callbacks.TensorBoard(str(self.log_dir)), hp.KerasCallback(str(self.log_dir), hparams)]
        if not self.model_factory.concatenated_input:
            assert x_pad is None and x_pad_val is None
            model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=(
                x_val, y_val), callbacks=callbacks)
        else:
            model.fit((x, x_pad, y), epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        val_loss = model.evaluate((x_val, x_pad_val, y_val))
        hparams['val_loss'] = val_loss
        model.save_weights(str(run_dir / 'model'), save_format='tf')
        with (run_dir / 'hparams.json').open('w') as f:
            json.dump(hparams, f)
        return model, val_loss

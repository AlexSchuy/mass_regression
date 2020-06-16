import numpy as np
import pandas as pd

import definitions


class BaseDataset():
    def __init__(self, data_path, features, target_name, pad_features=None, name='Base'):
        self.name = name
        self.features = features
        self.target_name = target_name
        if type(pad_features) is str:
            pad_features = [pad_features]
        self.pad_features = pad_features
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self._load(data_path)

    @property 
    def targets(self):
        raise NotImplementedError()

    @property
    def num_features(self):
        return len(self.features)

    @property
    def num_targets(self):
        return len(self.targets)
    
    @property
    def num_pad_features(self):
        if self.pad_features is not None:
            return len(self.pad_features)
        else:
            return 0

    @staticmethod
    def _scale(A, mean, std):
        return (A - mean) / std

    @staticmethod
    def _unscale(A, mean, std):
        return std * A + mean

    def scale_x(self, x):
        return self._scale(x, self.x_mean, self.x_std)

    def unscale_x(self, x):
        return self._unscale(x, self.x_mean, self.x_std)

    def scale_y(self, y):
        return self._scale(y, self.y_mean, self.y_std)

    def unscale_y(self, y):
        return self._unscale(y, self.y_mean, self.y_std)

    def scale_x_pad(self, x_pad):
        return self._scale(x_pad, self.x_pad_mean, self.x_pad_std)

    def unscale_x_pad(self, x_pad):
        return self._unscale(x_pad, self.x_pad_mean, self.x_pad_std)

    def _split(self, df, scale_x=False, scale_y=False, scale_x_pad=False):
        x = df[self.features]
        if scale_x:
            x = self.scale_x(x)
        y = df[self.targets]
        if scale_y:
            y = self.scale_y(y)
        if self.pad_features is not None:
            x_pad = df[self.pad_features]
            if scale_x_pad:
                x_pad = self.scale_x_pad(x_pad)
            return x, x_pad, y
        else:
            return x, None, y

    def _load(self, data_path):
        def get_data(t):
            path = data_path / f'{t}.pkl'
            data = pd.read_pickle(path)
            return data

        self.df_train = get_data('train')
        x_train, x_pad_train, y_train = self._split(self.df_train)
        self.num_training_samples = y_train.shape[0]
        self.x_mean = np.mean(x_train).values
        self.x_std = np.std(x_train).values
        self.y_mean = np.mean(y_train).values
        self.y_std = np.std(y_train).values
        if x_pad_train is not None:
            self.x_pad_mean = np.mean(x_pad_train).values
            self.x_pad_std = np.std(x_pad_train).values
        self.df_val = get_data('val')
        self.df_test = get_data('test')

    def _get(self, df, split=True, scale_x=True, scale_y=True, scale_x_pad=True):
        if split:
            return self._split(df, scale_x, scale_y, scale_x_pad)
        else:
            return df

    def train(self, split=True, scale_x=True, scale_y=True, scale_x_pad=True):
        return self._get(self.df_train, split, scale_x, scale_y, scale_x_pad)

    def val(self, split=True, scale_x=True, scale_y=True, scale_x_pad=True):
        return self._get(self.df_val, split, scale_x, scale_y, scale_x_pad)

    def test(self, split=True, scale_x=True, scale_y=True, scale_x_pad=True):
        return self._get(self.df_test, split, scale_x, scale_y, scale_x_pad)
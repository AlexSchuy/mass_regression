import logging

import pandas as pd
import torch
from omegaconf import DictConfig


def to_tensor(df: pd.DataFrame):
    return torch.from_numpy(df.to_numpy())  # pylint: disable=not-callable


def calc_E(px, py, pz, m):
    return (px**2 + py**2 + pz**2 + m**2)**0.5


def add_fourvectors(px1, py1, pz1, m1, px2, py2, pz2, m2):
    E1 = calc_E(px1, py1, pz1, m1)
    E2 = calc_E(px2, py2, pz2, m2)
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    E = E1 + E2
    m2 = E**2 - px**2 - py**2 - pz**2
    neg_mask = m2 < 0.0
    if neg_mask.any():
        logging.warning('Invalid masses calculated!')
        m2[neg_mask] *= -1.0
    m = m2**0.5
    return px, py, pz, m


class StandardScaler():
    """Standardize data by removing the mean and scaling to
    unit variance.  This object can be used as a transform
    in PyTorch data loaders.

    Args:
        mean (FloatTensor): The mean value for each feature in the data.
        scale (FloatTensor): Per-feature relative scaling.
    """

    def __init__(self, mean=None, scale=None):
        self.mean_ = mean
        self.scale_ = scale

    def fit(self, sample):
        """Set the mean and scale values based on the sample data.
        """
        self.mean_ = sample.mean(0)
        self.scale_ = sample.std(0, unbiased=False)
        self.scale_[self.scale_ < 1e-7] = 1e-7
        logging.info(f'mean = {self.mean_}; std = {self.scale_}')
        return self

    def __call__(self, sample):
        return (sample - self.mean_)/self.scale_

    def inverse_transform(self, sample):
        """Scale the data back to the original representation
        """
        return sample * self.scale_ + self.mean_

    @property
    def mean(self):
        return self.mean_

    @property
    def std(self):
        return self.scale_

    def to(self, device):
        self.mean_ = self.mean_.to(device)
        self.scale_ = self.scale_.to(device)
        return self


def init_transforms(fit_transforms, feature_mean=None, feature_std=None, output_mean=None, output_std=None, target_mean=None, target_std=None):
    if fit_transforms:
        feature_transform = StandardScaler()
        output_transform = StandardScaler()
        target_transform = StandardScaler()
    else:
        feature_transform = StandardScaler(feature_mean, feature_std)
        output_transform = StandardScaler(output_mean, output_std)
        target_transform = StandardScaler(target_mean, target_std)

    return feature_transform, output_transform, target_transform

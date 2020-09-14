import torch
import logging

class StandardScaler():
    """Standardize data by removing the mean and scaling to
    unit variance.  This object can be used as a transform
    in PyTorch data loaders.

    Args:
        mean (FloatTensor): The mean value for each feature in the data.
        scale (FloatTensor): Per-feature relative scaling.
    """

    def __init__(self, mean=None, scale=None):
        if mean is not None:
            mean = torch.FloatTensor(mean)
        if scale is not None:
            scale = torch.FloatTensor(scale)
        self.mean_ = mean
        self.scale_ = scale

    def fit(self, sample):
        """Set the mean and scale values based on the sample data.
        """
        self.mean_ = sample.mean(0, keepdim=True)
        self.scale_ = sample.std(0, unbiased=False, keepdim=True)
        if self.scale_ < 1e-7:
            self.scale_ = 1e-7
        logging.info(f'mean = {self.mean_}; std = {self.scale_}')
        return self

    def __call__(self, sample):
        return (sample - self.mean_)/self.scale_

    def inverse_transform(self, sample):
        """Scale the data back to the original representation
        """
        return sample * self.scale_ + self.mean_
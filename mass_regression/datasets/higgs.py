import logging
import random
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import torch
import uproot
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from uproot_methods.classes.TLorentzVector import TLorentzVector

from utils import to_tensor, StandardScaler
import hydra
from omegaconf import DictConfig

from typing import List


class HiggsDataset(Dataset):
    def __init__(self, events: pd.DataFrame, features: List[str], outputs: List[str], targets: List[str], attributes: List[str], feature_transform=None, output_transform=None, target_transform=None):
        super().__init__()
        self.events = events
        self.features = features
        self.outputs = outputs
        self.targets = targets
        self.attributes = attributes
        self.feature_transform = feature_transform
        self.output_transform = output_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        features = to_tensor(self.events.loc[index, self.features])
        outputs = to_tensor(self.events.loc[index, self.outputs])
        targets = to_tensor(self.events.loc[index, self.targets])
        attributes = to_tensor(self.events.loc[index, self.attributes])

        features = self.feature_transform(features)
        outputs = self.output_transform(outputs)
        targets = self.target_transform(targets)

        return features, outputs, targets, attributes


class HiggsDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: Union[str, Path], features: List[str], outputs: List[str], targets: List[str], attributes: List[str], training_masses: Union[List[int], str] = 'all', testing_masses: Union[List[int], str] = 'all', seed: int = 37, data_url: str = 'https://cernbox.cern.ch/index.php/s/1gP5w7skUWzdhlU/download', download: bool = False, num_workers: int = 8, scale: bool = True, event_frac: float = 1.0, train_frac: float = 0.8, test_frac: float = 0.1, feature_transform=None, output_transform=None, target_transform=None, fit_transforms=False):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.training_masses = training_masses
        self.testing_masses = testing_masses
        self.features = features
        self.outputs = outputs
        self.targets = targets
        self.attributes = attributes

        if type(data_dir) is str:
            data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir = data_dir
        self.data_path = data_dir / f'h_ww_lnulnu.pkl'
        self.root_path = data_dir / 'h_ww_lnulnu.root'
        self.data_url = data_url
        self._download = download

        self._validate_fracs(event_frac, train_frac, test_frac)
        self.event_frac = event_frac
        self.train_frac = train_frac
        self.test_frac = test_frac

        self.feature_transform = feature_transform
        self.output_transform = output_transform
        self.target_transform = target_transform

        if fit_transforms:
            self.prepare_data()
            self.setup('fit')
            self.feature_transform.fit(
                to_tensor(self.train_dataset.events[self.features]))
            self.output_transform.fit(
                to_tensor(self.train_dataset.events[self.outputs]))
            self.target_transform.fit(
                to_tensor(self.train_dataset.events[self.targets]))

    def is_downloaded(self) -> bool:
        return self.root_path.exists()

    def is_extracted(self) -> bool:
        return self.data_path.exists()

    def download(self) -> Path:
        try:
            logging.info(
                f'Downloading data to {self.data_dir} (this may take a few minutes).')
            with requests.get(self.data_url, allow_redirects=True, stream=True) as r:
                r.raise_for_status()
                with self.root_path.open(mode='wb+') as f:
                    pbar = tqdm(total=int(r.headers['content-length']))
                    for chunk in r.iter_content(chunk_size=1024**2):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            logging.info('download complete.')
        except:
            logging.error(
                f'Download failed! Please download manually from {self.data_url} to {self.root_path}.')
            raise

    def extract(self) -> None:
        logging.info(f'Extracting data to {self.data_path}.')
        f = uproot.open(self.root_path)
        events = f['event;1']
        df = pd.DataFrame()
        for k, v in tqdm(events.items(), total=len(events.keys())):
            name = k.decode('ascii')
            values = v.array()
            if type(values[0]) is TLorentzVector:
                x = np.zeros(values.shape)
                y = np.zeros(values.shape)
                z = np.zeros(values.shape)
                E = np.zeros(values.shape)
                m = np.zeros(values.shape)
                for i, vec in enumerate(values):
                    x[i] = vec.x
                    y[i] = vec.y
                    z[i] = vec.z
                    E[i] = vec.t
                    try:
                        m[i] = vec.mass
                    except:
                        m[i] = 0.0
                df[f'{name}x'] = x
                df[f'{name}y'] = y
                df[f'{name}z'] = z
                df[f'{name}E'] = E
                df[f'{name}m'] = m
            else:
                df[name] = values
        df = df.astype('float32')
        df.to_pickle(self.data_path)

    def prepare_data(self) -> None:
        if not self.is_extracted():
            logging.info(f'dataset not found at {self.data_path}.')
            if not self.is_downloaded():
                logging.info(
                    f'downloaded root dataset not found at {self.root_path}.')
                if self._download:
                    self.download()
                else:
                    logging.error('download=false, aborting!')
                    raise RuntimeError()
            self.extract()

    def setup(self, stage=None) -> None:
        if self.seed is None:
            self.seed = torch.initial_seed() % (2**32 - 1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Split events for each mass point into train/val/test.
        events = pd.read_pickle(self.data_path)
        all_masses = events['H_Mass'].unique()
        train_events = pd.DataFrame()
        val_events = pd.DataFrame()
        test_events = pd.DataFrame()
        for mass in all_masses:
            events_subset = events[events['H_Mass'] == mass]
            train_subset, val_subset, test_subset = self.train_val_test_split(
                events_subset)
            train_events = pd.concat(
                (train_events, train_subset), ignore_index=True)
            val_events = pd.concat((val_events, val_subset), ignore_index=True)
            test_events = pd.concat(
                (test_events, test_subset), ignore_index=True)

        # Select relevant masses.
        if self.training_masses == 'all':
            self.training_masses = all_masses
        if self.testing_masses == 'all':
            self.testing_masses = all_masses
        train_events = train_events[train_events['H_Mass'].isin(
            self.training_masses)].sample(frac=1).reset_index(drop=True)
        val_events = val_events[val_events['H_Mass'].isin(
            self.testing_masses)].sample(frac=1).reset_index(drop=True)
        test_events = test_events[test_events['H_Mass'].isin(
            self.testing_masses)].sample(frac=1).reset_index(drop=True)

        # Create datasets for given stage.
        if stage == 'fit' or stage is None:
            self.train_dataset = HiggsDataset(train_events, self.features, self.outputs, self.targets,
                                              self.attributes, self.feature_transform, self.output_transform, self.target_transform)
            self.val_dataset = HiggsDataset(val_events, self.features, self.outputs, self.targets,
                                            self.attributes, self.feature_transform, self.output_transform, self.target_transform)
        if stage == 'test' or stage is None:
            self.test_dataset = HiggsDataset(test_events, self.features, self.outputs, self.targets,
                                             self.attributes, self.feature_transform, self.output_transform, self.target_transform)

    def dataloader(self, dataset: HiggsDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)

    def train_val_test_split(self, events):
        num_events = int(self.event_frac * len(events))
        events = events[:num_events]
        num_train_events = int(self.train_frac * num_events)
        num_test_events = int(self.test_frac * num_events)
        num_val_events = num_events - num_train_events - num_test_events

        train_events = events[:num_train_events]
        val_events = events[num_train_events:-num_test_events]
        test_events = events[-num_test_events:]

        return train_events, val_events, test_events

    def _validate_fracs(self, event_frac, train_frac, test_frac):
        fracs = [event_frac, train_frac, test_frac]
        assert all(0.0 <= f <= 1.0 for f in fracs)
        assert train_frac + test_frac <= 1.0


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    feature_transform, output_transform, target_transform = hydra.utils.instantiate(
        cfg.transforms)
    datamodule = hydra.utils.instantiate(
        cfg.dataset, targets=cfg.dataset_criterion.targets, feature_transform=feature_transform, output_transform=output_transform, target_transform=target_transform)

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

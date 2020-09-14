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

import definitions
from utils import to_tensor


class HiggsDataset(Dataset):
    def __init__(self, events: pd.DataFrame, mass: float, features: list(str), outputs: list(str), attributes: list(str), features_transform=None, output_transform=None):
        self.events = events
        self.mass = mass
        self.features = features
        self.outputs = outputs
        self.attributes = attributes
        self.features_transform = features_transform
        self.output_transform = output_transform

    def __getitem__(self, index):
        features = to_tensor(self.events[index, self.features])
        outputs = to_tensor(self.events[index, self.outputs])
        attributes = to_tensor(self.events[index, self.attributes])

        features = self.features_transform(features)
        outputs = self.output_transform(features)
        return features, outputs, attributes

        


class HiggsDataModule(pl.LightningDataModule):
    def __init__(self, mass: int, batch_size: int, data_dir: Path, features: list(str), targets: list(str), attributes: list(str), seed: int = 37, data_url: str = 'https://cernbox.cern.ch/index.php/s/QC7N0Kdjs09qjYS', download: bool = False, num_workers: int = 8, scale: bool = True, num_events: Union[int, float] = 1.0, num_train_events: Union[int, float] = 0.8, num_test_events: Union[int, float] = 0.1, feat_transform=None, target_transform=None, fit_transforms=False):
        self.mass = mass
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir
        self.features = features
        self.targets = targets
        self.attributes = attributes
        self.data_path = data_dir / f'H{mass}'
        self.root_path = data_dir / 'higgs.root'
        self.data_url = data_url
        self._download = download
        self.num_workers = num_workers

        self._validate_num(num_events, num_train_events, num_test_events)
        self.num_events = num_events
        self.num_train_events = num_train_events
        self.num_test_events = num_test_events

        self.feat_transform = feat_transform
        self.target_transform = target_transform

        if fit_transforms:
            self.prepare_data()
            self.setup('fit')
            self.feat_transform.fit(to_tensor(self.train_dataset.events[self.features]))
            self.target_transform.fit(to_tensor(self.train_dataset.events[self.targets]))

        data_dir.mkdir(exist_ok=True, parents=True)

    def is_downloaded(self) -> bool:
        return self.root_path.exists()

    def is_extracted(self) -> bool:
        return self.data_path.exists()

    def download(self) -> Path:
        logging.info(
            f'Downloading data to {self.data_dir} (this may take a few minutes).')
        with requests.get(self.data_url, allow_redirects=True, stream=True) as r:
            r.raise_for_status()
            with self.root_path.open(mode='wb') as f:
                pbar = tqdm(total=int(r.headers['Content-Length']))
                for chunk in r.iter_content(chunk_size=1024**2):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        logging.info('download complete.')

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

        events = pd.read_pickle(self.data_path)
        if type(self.num_events) is float:
            self.num_events = int(self.num_events * len(events))
        events = events[:self.num_events]
        if type(self.num_train_events) is float:
            self.num_train_events = int(self.num_train_events * self.num_events)
        assert self.num_train_events <= self.num_events
        if type(self.num_test_events) is float:
            self.num_test_events = int(self.num_test_events * self.num_events)
        self.num_val_events = self.num_events - self.num_train_events - self.num_test_events
        assert self.num_val_events >= 0

        if stage == 'fit' or stage is None:
            train_events = events[:self.num_train_events]
            logging.debug(f'num training events={len(train_events)}')
            val_events = events[self.num_train_events:-self.num_test_events]
            self.train_dataset = HiggsDataset(train_events, self.mass, self.features,
                                              self.targets, self.attributes, self.feat_transform, self.target_transform)
            self.val_dataset = HiggsDataset(val_events, self.mass, self.features,
                                            self.targets, self.attributes, self.feat_transform, self.target_transform)
        if stage == 'test' or stage is None:
            test_events = events[-self.num_test_events:]
            logging.debug(f'num test events={len(test_events)}')
            self.test_dataset = HiggsDataset(test_events, self.mass, self.features,
                                             self.targets, self.attributes, self.feat_transform, self.target_transform)

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

    def _validate_num(self, *args):
        for num in args:
            if type(num) is float:
                assert 0.0 < num <= 1.0
            elif type(num) is int:
                assert num >= 0

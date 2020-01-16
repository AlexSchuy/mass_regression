import argparse
import os

import numpy as np
import pandas as pd
import progressbar
import tensorflow as tf

import definitions
from common import utils
from sklearn.model_selection import train_test_split

def calc_num_events(filepath, lines_per_event):
    num_lines = 1
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            num_lines += 1
    return num_lines // lines_per_event


def convert_H125(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
        text = text.replace('VIS', '')
    with open(filepath, 'w') as f:
        f.write(text)
    num_events = calc_num_events(filepath, lines_per_event=18)
    columns = []
    particles = ['H', 'Wa', 'Wb', 'La', 'Na', 'Lb', 'Nb']
    variables = ['x', 'y', 'z', 'm']
    types = ['gen', 'reco']
    columns = [
        f'{p}{v}_{t}' for p in particles for t in types for v in variables]
    columns = columns + ['METx', 'METy']
    data = pd.DataFrame(columns=columns, index=np.arange(
        num_events), dtype=np.float)
    current_type = 'gen'
    event = -1
    with progressbar.ProgressBar(max_value=num_events, redirect_stdout=True) as progbar:
        with open(filepath, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('EVENT'):
                    event += 1
                    progbar.update(event)
                elif line.startswith('MET'):
                    split = line.split()
                    data.loc[event, 'METx'] = float(split[1])
                    data.loc[event, 'METy'] = float(split[2])
                else:
                    processed_line = False
                    for p in particles:
                        if line.startswith(p):
                            split = line.split()
                            for i, v in enumerate(variables):
                                data.loc[event, f'{p}{v}_{current_type}'] = float(
                                    split[i+1])
                            processed_line = True
                            break
                    if not processed_line:
                        for t in types:
                            if line.lower().startswith(t):
                                current_type = t
                                processed_line = True
                                break
                    if not processed_line:
                        raise RuntimeError(f'Malformed input line: {line}')
                line = f.readline()

    assert event + 1 == num_events
    return data


def convert_Wlnu(filepath):
    num_events = calc_num_events(filepath, lines_per_event=10)
    columns = []
    particles = ['W', 'L', 'NU']
    variables = ['x', 'y', 'z', 'm']
    types = ['gen', 'reco']
    columns = [
        f'{p}{v}_{t}' for p in particles for t in types for v in variables]
    columns = columns + ['METx', 'METy']
    data = pd.DataFrame(columns=columns, index=np.arange(
        num_events), dtype=np.float)
    current_type = 'gen'
    event = -1
    with progressbar.ProgressBar(max_value=num_events, redirect_stdout=True) as progbar:
        with open(filepath, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('EVENT'):
                    event += 1
                    progbar.update(event)
                elif line.startswith('MET'):
                    split = line.split()
                    data.loc[event, 'METx'] = float(split[1])
                    data.loc[event, 'METy'] = float(split[2])
                else:
                    processed_line = False
                    for p in particles:
                        if line.startswith(p):
                            split = line.split()
                            for i, v in enumerate(variables):
                                data.loc[event, f'{p}{v}_{current_type}'] = float(
                                    split[i+1])
                            processed_line = True
                            break
                    if not processed_line:
                        for t in types:
                            if line.lower().startswith(t):
                                current_type = t
                                processed_line = True
                                break
                    if not processed_line:
                        raise RuntimeError(f'Malformed input line: {line}')
                line = f.readline()

    assert event + 1 == num_events
    return data


def train_val_test_split(df, n_train, n_val, n_test, seed=10856):
    train, test = train_test_split(df, train_size=n_train + n_val, test_size=n_test, random_state=seed)
    train, val = train_test_split(train, train_size=n_train, test_size=n_val, random_state=seed)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        'Convert raw samples to appropriate data format for training.')
    parser.add_argument('dataset', choices=definitions.DATASETS)
    args = parser.parse_args()
    dataset = args.dataset
    dataset_dir = definitions.SAMPLES_DIR / dataset
    pickle_path = dataset_dir / 'df.pkl'
    input_filepath = dataset_dir / 'raw.dat'
    if dataset == 'Wlnu':
        if not pickle_path.exists():
            df = convert_Wlnu(input_filepath)
            df.to_pickle(pickle_path)
        n_train = 80000
        n_val = 10000
        n_test = 10000
    elif dataset == 'H125':
        if not pickle_path.exists():
            df = convert_H125(input_filepath)
            df.to_pickle(pickle_path)
        n_train = 80000
        n_val = 10000
        n_test = 10000
    else:
        raise NotImplementedError('Unknown dataset!')
    if pickle_path.exists():
        df = pd.read_pickle(pickle_path)
    train_path = dataset_dir / 'train.pkl'
    val_path = dataset_dir / 'val.pkl'
    test_path = dataset_dir / 'test.pkl'
    train, val, test = train_val_test_split(df, n_train,  n_val, n_test)

    for path, dataset in zip((train_path, val_path, test_path), (train, val, test)):
        dataset.to_pickle(path)


if __name__ == '__main__':
    main()

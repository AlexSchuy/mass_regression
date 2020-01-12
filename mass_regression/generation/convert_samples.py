import argparse
import os

import numpy as np
import pandas as pd
import progressbar
import tensorflow as tf

import definitions
from common import utils


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


def train_dev_test_split(df, n_train, n_dev, n_test):
    dataset = tf.data.Dataset.from_tensor_slices(df.values)
    dataset.shuffle(n_train + n_dev + n_test, seed=10856)
    train_dataset = dataset.take(n_train)
    dev_dataset = dataset.skip(n_train).take(n_dev)
    test_dataset = dataset.skip(n_train + n_dev)
    return train_dataset, dev_dataset, test_dataset


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
        n_dev = 10000
        n_test = 10000
    elif dataset == 'H125':
        if not pickle_path.exists():
            df = convert_H125(input_filepath)
            df.to_pickle(pickle_path)
        n_train = 80000
        n_dev = 10000
        n_test = 10000
    else:
        raise NotImplementedError('Unknown dataset!')
    if pickle_path.exists():
        df = pd.read_pickle(pickle_path)
    train_path = dataset_dir / 'train.tfrecords'
    dev_path = dataset_dir / 'dev.tfrecords'
    test_path = dataset_dir / 'test.tfrecords'
    train, dev, test = train_dev_test_split(df, n_train, n_dev, n_test)

    for path, dataset in zip((train_path, dev_path, test_path), (train, dev, test)):
        dataset = dataset.map(tf.io.serialize_tensor)
        writer = tf.data.experimental.TFRecordWriter(str(path))
        writer.write(dataset)


if __name__ == '__main__':
    main()

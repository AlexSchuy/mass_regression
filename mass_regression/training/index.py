"""Contains methods to manipulate the index for runs (see training/run.py)."""


import functools
import operator
import os

import numpy as np
import yaml

from common import utils


def get_index_path():
    return os.path.join(utils.get_results_path(), 'index.yml')


def load():
    index_path = get_index_path()
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = yaml.full_load(f)
    else:
        index = {}
    return index


def get_run_number(config):
    index = load()
    for run_number in index:
        matches = True
        for name, value in index[run_number].items():
            if name not in config or config[name] != value:
                matches = False
        if matches:
            return run_number
    return -1


def get_config(run_number):
    index = load()
    if run_number not in index:
        return None
    else:
        return index[run_number]


def add(run_number, config):
    assert get_run_number(config) == -1, f'{config} already present in index.'
    assert get_config(
        run_number) is None, f'{run_number} already present in index.'
    assert all(not type(v) == list for k, v in config.items()
               ), 'Lists cannot be stored as elements in the index.'
    index = load()
    index[run_number] = config
    with open(get_index_path(), 'w+') as f:
        yaml.dump(index, f)


def remove(run_number, config):
    assert get_run_number(
        config) == run_number, f'({run_number}, {config}) is not present in index.'
    index = load()
    del index[run_number]
    with open(get_index_path(), 'w') as f:
        yaml.dump(index, f)

import definitions
import numpy as np
import tensorflow as tf


class BaseParser():
    def __init__(self, parser, subparsers, name='base_parser'):
        self.name = name
        self.subparsers = subparsers
        parser.add_argument('--seed', default=5, type=int)
        parser.add_argument('--single_run', action='store_true')
        parser.add_argument('--n', type=int)


    def parse(self, args):
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
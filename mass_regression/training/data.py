import os

import numpy as np
import pandas as pd
import tensorflow as tf

import definitions


def calc_E(px, py, pz, m):
    return (px**2 + py**2 + pz**2 + m**2)**0.5


def add_fourvectors(px1, py1, pz1, m1, px2, py2, pz2, m2):

    E1 = calc_E(px1, py1, pz1, m1)
    E2 = calc_E(px2, py2, pz2, m2)
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    E = E1 + E2
    m = (E**2 - px**2 - py**2 - pz**2)**0.5
    return px, py, pz, m
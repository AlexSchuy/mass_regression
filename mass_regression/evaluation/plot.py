import argparse
import os
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
from training.run import Run

def calc_Wm(NUz_reco, df):
    NUE_reco = (df['NUx_reco']**2 + df['NUy_reco']**2 + NUz_reco**2)**0.5
    LE_reco = (df['Lx_reco']**2 + df['Ly_reco']**2 + df['Lz_reco']**2 + df['Lm_reco']**2)**0.5
    WE_reco = NUE_reco + LE_reco
    Wz_reco = df['Lz_reco'] + NUz_reco
    Wm_reco = (WE_reco**2 - df['Wx_reco']**2 - df['Wy_reco']**2 - Wz_reco**2)**0.5
    return Wm_reco
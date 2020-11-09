import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

import pandas as pd


class Plotter():
    def __init__(self, data_path):
        self.events = pd.read_pickle(data_path)

    def diff_df(self, events=None, particle='H', feature='m'):
        if events is None:
            events = self.events
        nn_feature = f'{particle}_Pred{feature}'
        truth_feature = f'{particle}_Gen{feature}'
        jigsaw_feature = f'{particle}{feature}'
        display_name = f'{particle}{feature}'

        df = pd.DataFrame()
        df['nn'] = events[nn_feature] - events[truth_feature]
        df['jigsaw'] = events[jigsaw_feature] - events[truth_feature]

        diff_df = pd.DataFrame()
        source = ['nn']*events.shape[0] + ['jigsaw']*events.shape[0]
        diff_df['source'] = source
        diff_df[display_name] = pd.concat((df['nn'], df['jigsaw']), ignore_index=True)
        
        return diff_df

    def error_plot(self, events=None, particle='H', feature='m'):
        diff_df = self.diff_df(events, particle, feature)
        display_name = f'{particle}{feature}'
        fig = px.histogram(diff_df, x=display_name, color='source', barmode='overlay')
        return fig
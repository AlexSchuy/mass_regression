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


    def mass_plot(self, hm_plot_range, dataset=None):
        test_df = self.events
        bins = 60
        plot_range = (0, 150)

        Wam_gen, edges = np.histogram(test_df['Wam_gen'], bins=bins, range=plot_range)
        Wbm_gen, edges = np.histogram(test_df['Wbm_gen'], bins=bins, range=plot_range)
        Hm_gen, hm_edges = np.histogram(test_df['Hm_gen'], bins=bins, range=hm_plot_range)
        Wam_reco, _ = np.histogram(test_df['Wam_reco'], bins=bins, range=plot_range)
        Wbm_reco, _ = np.histogram(test_df['Wbm_reco'], bins=bins, range=plot_range)
        Hm_reco, _ = np.histogram(test_df['Hm_reco'], bins=bins, range=hm_plot_range)
        Wam_pred, _ = np.histogram(test_df['Wam_pred'], bins=bins, range=plot_range)
        Wbm_pred, _ = np.histogram(test_df['Wbm_pred'], bins=bins, range=plot_range)
        Hm_pred, _ = np.histogram(test_df['Hm_pred'], bins=bins, range=hm_plot_range)

        fig = make_subplots(rows=1, cols=3, column_titles=['Wa mass', 'Wb mass', 'Hm mass'])

        fig.add_trace(go.Scatter(x=edges, y=Wam_gen, name='truth', showlegend=True, legendgroup='truth', line_color='blue'), row=1, col=1)
        fig.add_trace(go.Scatter(x=edges, y=Wam_reco, name='jigsaw', showlegend=True, legendgroup='jigsaw', line_color='red'), row=1, col=1)
        fig.add_trace(go.Scatter(x=edges, y=Wam_pred, name='nn', showlegend=True, legendgroup='nn', line_color='green'), row=1, col=1)

        fig.add_trace(go.Scatter(x=edges, y=Wbm_gen, name='truth', showlegend=False, legendgroup='truth', line_color='blue'), row=1, col=2)
        fig.add_trace(go.Scatter(x=edges, y=Wbm_reco, name='jigsaw', showlegend=False, legendgroup='jigsaw', line_color='red'), row=1, col=2)
        fig.add_trace(go.Scatter(x=edges, y=Wbm_pred, name='nn', showlegend=False, legendgroup='nn', line_color='green'), row=1, col=2)

        fig.add_trace(go.Scatter(x=hm_edges, y=Hm_gen, name='truth', showlegend=False, legendgroup='truth', line_color='blue'), row=1, col=3)
        fig.add_trace(go.Scatter(x=hm_edges, y=Hm_reco, name='jigsaw', showlegend=False, legendgroup='jigsaw', line_color='red'), row=1, col=3)
        fig.add_trace(go.Scatter(x=hm_edges, y=Hm_pred, name='nn', showlegend=False, legendgroup='nn', line_color='green'), row=1, col=3)

        fig.update_xaxes(title_text='W mass [GeV]', row=1, col=1)
        fig.update_xaxes(title_text='W mass [GeV]', row=1, col=2)
        fig.update_xaxes(title_text='H mass [GeV]', row=1, col=3)
        fig.update_yaxes(title_text='counts', type='log')
        fig.update_yaxes(rangemode='tozero')
        fig.update_layout(width=1200, height=600, yaxis_rangemode='tozero')
        return fig

def main():
    plotter_mixed = Plotter(mass=[125, 400, 750], target_name='nu', loss_name='hm', log_dir=definitions.LOG_DIR / 'H125_400_750' / 'nu' / 'model_v1-Hm_loss')


if __name__ == '__main__':
    main()

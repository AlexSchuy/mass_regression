import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from training.datasets.h import (HiggsDataset, HmLoss, NuLoss, WmLoss,
                                 df_calc_Hm, df_calc_Wm)
from training.models.model_v1_factory import Model_V1_Factory


class Plotter():
    def __init__(self, mass, target_name, loss_name, log_dir):
        self.run_dir = log_dir / 'single_run'
        self.target_name = target_name
        if loss_name == 'wm':
            pad_features = ['Wam_gen', 'Wbm_gen']
            loss = WmLoss
        elif loss_name == 'nu':
            pad_features = ['Wam_gen', 'Wbm_gen']
            loss = NuLoss
        elif loss_name == 'hm':
            pad_features = ['Hm_gen']
            loss = HmLoss
        self.dataset = HiggsDataset(mass=mass, target_name=target_name, pad_features=pad_features)
        self.hparams, self.model = Model_V1_Factory(self.dataset, loss).load(self.run_dir)
        for t in ('train', 'val', 'test'):
            self.load_data(t)
        self.load_deltas(log_dir)

    def load_deltas(self, log_dir):
        self.deltas_dir = log_dir / 'deltas'
        self.deltas = {}
        for delta_path in self.deltas_dir.glob('*.npy'):
            self.deltas[delta_path.stem.replace('delta_', '')] = np.load(str(delta_path))

    def load_data(self, t, dataset=None):
        if dataset is None:
            dataset = self.dataset
        if t == 'train':
            df = dataset.train(split=False)
            x, x_pad, y = dataset.train()
        elif t == 'val':
            df = dataset.val(split=False)
            x, x_pad, y = dataset.val()
        elif t == 'test':
            df = dataset.test(split=False)
            x, x_pad, y = dataset.test()
        y_pred = dataset.unscale_y(self.model.predict((x, x_pad, y)))
        df['Nax_pred'] = y_pred[:, 0]
        df['Nay_pred'] = y_pred[:, 1]
        df['Naz_pred'] = y_pred[:, 2]
        df['Nbz_pred'] = y_pred[:, 3]
        Wm = df_calc_Wm(df, y_pred)
        df['Wam_pred'] = Wm[:, 0]
        df['Wbm_pred'] = Wm[:, 1]
        df['Hm_pred'] = df_calc_Hm(df, y_pred)
        return df

    def delta_plot(self, epochs, mean):
        test_df = self.load_data('test')
        variables = ['Nax', 'Nay', 'Naz', 'Nbz', 'Wam', 'Wbm', 'Hm']
        fig = make_subplots(rows=2, cols=4, subplot_titles=variables)

        def add(data, row, col, name, line_color, showlegend=True):
            if mean:
                y = np.mean(data[:epochs, :], axis=1)
            else:
                y = np.std(data[:epochs, :], axis=1)
            x = np.arange(epochs)
            fig.add_trace(go.Scatter(x=x, y=y, name=name, legendgroup=name, showlegend=showlegend,
                                     line_color=line_color, opacity=0.5), row=row, col=col)
            fig.add_trace(go.Scatter(x=x, y=signal.savgol_filter(
                y, 53, 3), name=f'{name} (smoothed)', legendgroup=name, showlegend=showlegend, line_color=line_color, line_dash='dash'), row=row, col=col)

        def add_pair(var, row, col, **kwargs):
            add(self.deltas[f'{var}_train'], row, col, line_color='Crimson', name='training', **kwargs)
            add(self.deltas[f'{var}_val'], row, col, line_color='MediumPurple', name='val', **kwargs)

        def add_jigsaw(var, row, col, showlegend=True, **kwargs):
            x = np.arange(epochs)
            if mean:
                y = (test_df[f'{var}_reco'] - test_df[f'{var}_gen']).mean()
            else:
                y = (test_df[f'{var}_reco'] - test_df[f'{var}_gen']).std()
            y = np.full_like(x, y, dtype=np.double)
            fig.add_trace(go.Scatter(showlegend=showlegend, name='jigsaw', line_color='Black',
                                     line_dash='dashdot', x=x, y=y), row=row, col=col)

        for i, var in enumerate(variables):
            showlegend = i == 0
            row = i // 4 + 1
            col = i % 4 + 1
            add_pair(var, row, col, showlegend=showlegend)
            add_jigsaw(var, row, col, showlegend=showlegend)
    	
        fig.update_yaxes(rangemode='tozero')
        fig.update_layout(yaxis_rangemode='tozero')
        return fig

    def training_plot(self, epochs, delta):
        test_df = self.load_data('test')
        colors = plotly.colors.qualitative.Vivid
        variables = ['Nax', 'Nay', 'Naz', 'Nbz', 'Wam', 'Wbm', 'Hm']
        bins = {k: 60 for k in variables}
        plot_range = {'Nax': (-100, 100), 'Nay': (-100, 100), 'Naz': (-300, 300), 'Nbz': (-300, 300), 'Wam': (-300, 300), 'Wbm': (-300, 300), 'Hm': (-300, 300)}
        fig = make_subplots(rows=2, cols=4, subplot_titles=variables)

        def add(data, bins, p_range, row, col, showlegend=False, **kwargs):
            y, x = np.histogram(data, density=True, bins=bins, range=p_range)
            fig.add_trace(go.Scatter(x=x, y=y, showlegend=showlegend, visible=False, **kwargs), row=row, col=col)

        def add_pair(var, bins, p_range, row, col, **kwargs):
            if delta:
                add(self.deltas[f'{var}_train'][epoch], bins, p_range, row, col, line_color=colors[0], legendgroup='training', name='training', **kwargs)
                add(self.deltas[f'{var}_val'][epoch], bins, p_range, row, col, line_color=colors[1], legendgroup='val', name='val', **kwargs)
            else:
                add(self.deltas[f'{var}_train'][epoch] + self.train_df[f'{var}_gen'], bins, p_range, row, col, line_color=colors[0], legendgroup='training', name='training', **kwargs)
                add(self.deltas[f'{var}_val'][epoch] + self.val_df[f'{var}_gen'], bins, p_range, row, col, line_color=colors[1], legendgroup='val', name='val', **kwargs)  

        steps = []

        for i, var in enumerate(variables):
            showlegend = i == 0
            row = i // 4 + 1
            col = i % 4 + 1
            if delta:
                data = test_df[f'{var}_reco'] - test_df[f'{var}_gen']
            else:
                data = test_df[f'{var}_reco']
            add(data, bins[var], plot_range[var], row, col, showlegend=showlegend, line_color=colors[2], legendgroup='jigsaw', name='jigsaw')

        for epoch in range(epochs):
            for i, var in enumerate(variables):
                showlegend = i == 0
                row = i // 4 + 1
                col = i % 4 + 1
                add_pair(var, bins=bins[var], p_range=plot_range[var], row=row, col=col, showlegend=showlegend)
            n = 2 * len(variables)
            step = {'method': 'update', 'args': [{'visible': n//2 * [True] + [False] * epochs * n}]}
            step['args'][0]['visible'][n//2 + n*epoch:n*(epoch+1)] = n * [True]
            steps.append(step)

        sliders = [{'active': 0, 'currentvalue': {'prefix': 'epoch: '}, 'pad': {'t': 50}, 'steps': steps}]

        for i in range(n//2 + n):
            fig.data[i].visible = True


        fig.update_yaxes(title_text='Counts')
        if delta:
            fig.update_xaxes(title_text='Δ [GeV]')
        else:
            fig.update_xaxes(title_text='[GeV]')
        fig.update_yaxes(rangemode='tozero')
        fig.update_layout(width=1500, height=1000, yaxis_rangemode='tozero', sliders=sliders)
        return fig

    def mass_plot(self, hm_plot_range, dataset=None):
        test_df = self.load_data('test', dataset=dataset)
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
    import definitions
    plotter_hm = Plotter(mass=125, target_name='nu', loss_name='hm',
                         log_dir=definitions.LOG_DIR / 'H125' / 'nu' / 'model_v1-Hm_loss')


if __name__ == '__main__':
    main()

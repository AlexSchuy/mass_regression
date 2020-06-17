import numpy as np

from training.models.model_v1_factory import Model_V1_Factory
from training.wlnu import NUzLoss, WlnuDataset, WmLoss, df_calc_Wm


class Plotter():
    def __init__(self, target_name, loss_name, log_dir):
        self.run_dir = log_dir / 'single_run'
        self.target_name = target_name
        self.dataset = WlnuDataset(target_name=target_name)
        self.df = self.dataset.test(split=False)
        if loss_name == 'wm':
            loss = WmLoss
        elif loss_name == 'nuz':
            loss = NUzLoss
        self.hparams, self.model = Model_V1_Factory(self.dataset, loss).load(self.run_dir)
        x, x_pad, y = self.dataset.test()
        self.y_pred = self.dataset.unscale_y(self.model.predict((x, x_pad, y)))
        self.df['NUz_pred'] = self.y_pred
        self.df['Wm_pred'] = df_calc_Wm(self.df, self.y_pred)
        self.load_deltas(log_dir)

    def load_deltas(self, log_dir):
        self.deltas_dir = log_dir / 'deltas'
        self.delta_NUz_train = np.load(self.deltas_dir / 'delta_NUz_train.npy')
        self.delta_NUz_val = np.load(self.deltas_dir / 'delta_NUz_val.npy')
        self.delta_Wm_train = np.load(self.deltas_dir / 'delta_Wm_train.npy')
        self.delta_Wm_val = np.load(self.deltas_dir / 'delta_Wm_val.npy')

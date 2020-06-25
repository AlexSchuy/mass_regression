import numpy as np

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

    def load_data(self, t):
        if t == 'train':
            self.train_df = self.dataset.train(split=False)
            df = self.train_df
            x, x_pad, y = self.dataset.train()
        elif t == 'val':
            self.val_df = self.dataset.val(split=False)
            df = self.val_df
            x, x_pad, y = self.dataset.val()
        elif t == 'test':
            self.test_df = self.dataset.test(split=False)
            df = self.test_df
            x, x_pad, y = self.dataset.test()
        y_pred = self.dataset.unscale_y(self.model.predict((x, x_pad, y)))
        df['Nax_pred'] = y_pred[:, 0]
        df['Nay_pred'] = y_pred[:, 1]
        df['Naz_pred'] = y_pred[:, 2]
        df['Nbz_pred'] = y_pred[:, 3]
        Wm = df_calc_Wm(df, y_pred)
        df['Wam_pred'] = Wm[:, 0]
        df['Wbm_pred'] = Wm[:, 1]
        df['Hm_pred'] = df_calc_Hm(df, y_pred)


def main():
    import definitions
    plotter_hm = Plotter(mass=125, target_name='nu', loss_name='hm', log_dir=definitions.LOG_DIR / 'H125' / 'nu' / 'model_v1-Hm_loss')


if __name__ == '__main__':
    main()
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

def plot_prediction(run):
    
    def plot(y_target, y_pred, y_jigsaw, figname, axlabel):
        y_pred = y_pred - y_target
        y_jigsaw = y_jigsaw - y_target
        sns.distplot(y_pred, color='red', label='nn', axlabel=axlabel)
        sns.distplot(y_jigsaw, color='blue', label='jigsaw', axlabel=axlabel)
        plt.legend()
        plt.savefig(os.path.join(run.run_path, f'{figname}.png'))
        plt.clf()
    X_train, y_train, df_train, X_test, y_test, df_test = run.get_train_test_datasets(full_dataset=True)
    y_jigsaw_train = df_train['NUz_reco']
    y_jigsaw_test = df_test['NUz_reco']
    axlabel = 'NUz_reco - NUz_gen [GeV]'
    plot(y_train, run.model.predict(X_train), y_jigsaw_train, 'NUz_train', axlabel)
    plot(y_test, run.model.predict(X_test), y_jigsaw_test, 'NUz_test', axlabel)

    y_train = df_train['Wm_gen']
    y_test = df_test['Wm_gen']
    axlabel = 'Wm_reco - Wm_gen [GeV]'
    plot(y_train, calc_Wm(run.model.predict(X_train), df_train), calc_Wm(y_jigsaw_train, df_train), 'Wm_train', axlabel)
    plot(y_test, calc_Wm(run.model.predict(X_test), df_test), calc_Wm(y_jigsaw_test, df_test), 'Wm_test', axlabel)
    

def main():
    parser = argparse.ArgumentParser(
        description='Display performance metrics for the given run.')
    parser.add_argument(
        '--run', help='The run number corresponding to the run that should be evaluated. By default, the most recent run is used.')

    args = parser.parse_args()

    if args.run is None:
        run = Run.most_recent()
    else:
        run = Run(args.run)

    plot_prediction(run)


if __name__ == '__main__':
    main()

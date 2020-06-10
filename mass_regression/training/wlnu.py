import tensorflow as tf

import data


def calc_Wm(x, y_pred):
    METx = x[:, 0]
    METy = x[:, 1]
    Lx_gen = x[:, 2]
    Ly_gen = x[:, 3]
    Lz_gen = x[:, 4]
    Lm_gen = x[:, 5]
    Nax = METx
    Nay = METy
    Naz_pred = y_pred[:, 0]
    Nam_pred = tf.zeros_like(Nax)
    _, _, _, Wm_pred = data.add_fourvectors(
        Nax, Nay, Naz_pred, Nam_pred, Lx_gen, Ly_gen, Lz_gen, Lm_gen)
    return Wm_pred

def main():
    df_train, _, _ = data.get_datasets(
        dataset='Wlnu', target='W', x_y_split=False)
    x = tf.constant(
        df_train[['METx', 'METy', 'Lx_reco', 'Ly_reco', 'Lz_reco', 'Lm_reco']].values)
    y = tf.constant(df_train[['NUz_gen']].values)
    Wm_pred = calc_Wm(x, y)
    print(tf.constant(df_train['Wm_gen'].values) - Wm_pred)


if __name__ == '__main__':
    main()

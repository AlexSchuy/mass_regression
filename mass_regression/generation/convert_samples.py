import os

import numpy as np
import pandas as pd
import progressbar

from common import utils


def calc_num_events(filepath, lines_per_event):
    num_lines = 1
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            num_lines += 1
    return num_lines // lines_per_event


def convert_Wlnu(filepath, output_filepath):
    num_events = calc_num_events(filepath, lines_per_event=10)
    data = pd.DataFrame(columns=['Lx_gen', 'Ly_gen', 'Lz_gen', 'Lm_gen', 'Lx_reco', 'Ly_reco', 'Lz_reco', 'Lm_reco', 'METx', 'METy', 'NUx_gen', 'NUy_gen',
                                 'NUz_gen', 'NUm_gen', 'NUx_reco', 'NUy_reco', 'NUz_reco', 'NUm_gen', 'Wm_gen', 'Wm_reco'], index=np.arange(num_events), dtype=np.float)
    event = -1
    gen = True
    with progressbar.ProgressBar(max_value=num_events, redirect_stdout=True) as p:
        with open(filepath, 'r') as f:
            line = f.readline()
            while line:
                if line.startswith('EVENT'):
                    event += 1
                    p.update(event)
                elif line.startswith('W') and not gen:
                    split = line.split()
                    data.loc[event, 'Wx_reco'] = float(split[1])
                    data.loc[event, 'Wy_reco'] = float(split[2])
                    data.loc[event, 'Wz_reco'] = float(split[3])
                    data.loc[event, 'Wm_reco'] = float(split[4])
                elif line.startswith('W') and gen:
                    split = line.split()
                    data.loc[event, 'Wx_gen'] = float(split[1])
                    data.loc[event, 'Wy_gen'] = float(split[2])
                    data.loc[event, 'Wz_gen'] = float(split[3])
                    data.loc[event, 'Wm_gen'] = float(split[4])
                elif line.startswith('L') and not gen:
                    split = line.split()
                    data.loc[event, 'Lx_reco'] = float(split[1])
                    data.loc[event, 'Ly_reco'] = float(split[2])
                    data.loc[event, 'Lz_reco'] = float(split[3])
                    data.loc[event, 'Lm_reco'] = float(split[4])
                elif line.startswith('L') and gen:
                    split = line.split()
                    data.loc[event, 'Lx_gen'] = float(split[1])
                    data.loc[event, 'Ly_gen'] = float(split[2])
                    data.loc[event, 'Lz_gen'] = float(split[3])
                    data.loc[event, 'Lm_gen'] = float(split[4])
                elif line.startswith('NU') and not gen:
                    split = line.split()
                    data.loc[event, 'NUx_reco'] = float(split[1])
                    data.loc[event, 'NUy_reco'] = float(split[2])
                    data.loc[event, 'NUz_reco'] = float(split[3])
                    data.loc[event, 'NUm_reco'] = float(split[4])
                elif line.startswith('NU') and gen:
                    split = line.split()
                    data.loc[event, 'NUx_gen'] = float(split[1])
                    data.loc[event, 'NUy_gen'] = float(split[2])
                    data.loc[event, 'NUz_gen'] = float(split[3])
                    data.loc[event, 'NUm_gen'] = float(split[4])
                elif line.startswith('MET'):
                    split = line.split()
                    data.loc[event, 'METx'] = float(split[1])
                    data.loc[event, 'METy'] = float(split[2])
                elif line.startswith('GEN'):
                    gen = True
                elif line.startswith('RECO'):
                    gen = False
                line = f.readline()
    data.to_pickle(output_filepath)


def main():
    project_path = utils.get_project_path()
    filepath = os.path.join(project_path, 'samples', 'raw', 'output_Wlnu.dat')
    output_filepath = os.path.join(
        project_path, 'samples', 'converted', 'Wlnu.pkl')
    convert_Wlnu(filepath, output_filepath)


if __name__ == '__main__':
    main()

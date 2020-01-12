from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
SAMPLES_DIR = ROOT_DIR / 'samples'

DATASETS = ['Wlnu', 'H125']
TARGETS = {'Wlnu': {'nu': ['NUz_gen'], 'W': ['Wm_gen']},
           'H125': {'nu': ['Na_gen', 'Nb_gen'], 'W': ['Wam_gen', 'Wbm_gen'], 'H': ['H_gen']}}
FEATURES = {'Wlnu': ['METx', 'METy'] + [f'{p}{v}_reco' for p in ['L'] for v in ['x', 'y', 'z', 'm']],
            'H125': ['METx', 'METy'] + [f'{p}{v}_reco' for p in ['La', 'Lb'] for v in ['x', 'y', 'z', 'm']]}

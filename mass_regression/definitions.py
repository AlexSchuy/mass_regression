from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
SAMPLES_DIR = ROOT_DIR / 'samples'
LOG_DIR = ROOT_DIR / 'logs'

DATASETS = ['Wlnu', 'H125']
TARGETS = {'Wlnu': {'nu': ['NUz_gen'], 'W': ['Wm_gen']},
           'H125': {'nu': ['Na_gen', 'Nb_gen'], 'W': ['Wam_gen', 'Wbm_gen'], 'H': ['Hm_gen']}}
JIGSAW_TARGETS = {'Wlnu': {'nu': ['NUz_reco'], 'W': ['Wm_reco']},
           'H125': {'nu': ['Na_reco', 'Nb_reco'], 'W': ['Wam_reco', 'Wbm_reco'], 'H': ['Hm_reco']}}
FEATURES = {'Wlnu': ['METx', 'METy'] + [f'{p}{v}_reco' for p in ['L'] for v in ['x', 'y', 'z', 'm']],
            'H125': ['METx', 'METy'] + [f'{p}{v}_reco' for p in ['La', 'Lb'] for v in ['x', 'y', 'z', 'm']]}

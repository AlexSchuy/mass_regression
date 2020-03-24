from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
SAMPLES_DIR = ROOT_DIR / 'samples'
LOG_DIR = ROOT_DIR / 'logs'

DATASETS = ['Wlnu', 'H125']
TARGETS = {'Wlnu': {'nu': ['NUz_gen'], 'W': ['Wm_gen']},
           'H125': {'nu': ['Nax_gen', 'Nay_gen', 'Naz_gen', 'Nbz_gen'], 'W': ['Wam_gen', 'Wbm_gen'], 'H': ['Hm_gen'], 'nuW': ['Nax_gen', 'Nay_gen', 'Wam_gen', 'Wbm_gen']}}
JIGSAW_TARGETS = {'Wlnu': {'nu': ['NUz_reco'], 'W': ['Wm_reco']},
           'H125': {'nu': ['Nax_reco', 'Nay_reco', 'Naz_reco', 'Nbz_reco'], 'W': ['Wam_reco', 'Wbm_reco'], 'H': ['Hm_reco'], 'nuW': ['Nax_reco', 'Nay_reco', 'Wam_reco', 'Wbm_reco']}}
FEATURES = {'Wlnu': ['METx', 'METy'] + [f'{p}{v}_reco' for p in ['L'] for v in ['x', 'y', 'z', 'm']],
            'H125': ['METx', 'METy'] + [f'{p}{v}_reco' for p in ['La', 'Lb'] for v in ['x', 'y', 'z', 'm']]}

PAD_TARGETS = {'H125': {'W': ['Wam_gen', 'Wbm_gen']}}
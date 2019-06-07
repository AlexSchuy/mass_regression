MODEL_NAMES = ['sklearn_nn']
DATASETS = ['Wlnu', 'Wlnu_Wm_label']

def model_name(name):
    assert name in MODEL_NAMES, f'model_name must be one of {MODEL_NAMES} but was {name}'

def dataset(name):
    assert name in DATASETS, f'dataset must be one of {DATASETS} but was {name}'

import os

from common import utils
from sklearn.externals import joblib
from training.data import get_train_test_datasets


def save_model(model_name, model, result, run_path):
    result_path = os.path.join(run_path, 'result.joblib')
    model_path = os.path.join(run_path, 'model.joblib')
    joblib.dump(model, model_path)
    joblib.dump(result, result_path)


def load_model(run_path):
    result_path = os.path.join(run_path, 'result.joblib')
    model_name, features, train_size, test_size, seed = utils.load_run_config_settings(
        run_path)
    model_path = os.path.join(run_path, 'model.joblib')
    model = joblib.load(model_path)
    result = joblib.load(result_path)
    return (model, result)

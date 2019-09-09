import argparse

import pandas as pd
from sklearn.metrics import r2_score

from training.run import Run



def print_metrics(run):

    print(f'Printing metrics for run at {run.run_path}.')

    X_train, y_train, df_train, X_test, y_test, df_test = run.get_train_test_datasets(
        full_dataset=True)
    y_jigsaw = df_test['NUz_reco']

    if 'testing_score' not in run.result:
        testing_score = run.model.score(X_test, y_test)
        run.result['testing_score'] = testing_score
        run.save()
    else:
        testing_score = run.result['testing_score']
    print(f'\tTesting score = {testing_score}')

    if 'training_score' not in run.result:
        training_score = run.model.score(X_train, y_train)
        run.result['training_score'] = training_score
        run.save()
    else:
        training_score = run.result['training_score']
    print(f'\tTraining score = {training_score}')
    print(f'\tTraining time = {run.result["training_time"]}')

    if 'jigsaw_score' not in run.result:
        jigsaw_score = r2_score(y_test, y_jigsaw)
    print(f'\tJigsaw score = {jigsaw_score}')

    if 'cv_results' not in run.result:
        cv_results = pd.DataFrame(run.model.cv_results_)
        run.result['cv_results'] = pd.DataFrame(run.model.cv_results_)
        run.save()
    else:
        cv_results = run.result['cv_results']
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(cv_results)


def main():
    parser = argparse.ArgumentParser(
        description='Display performance metrics for the given run.')
    parser.add_argument(
        '--run', help='The run number corresponding to the run that should be evaluated. By default, the most recent run is used.', type=int)

    args = parser.parse_args()

    if args.run is None:
        run = Run.most_recent()
    else:
        run = Run(args.run)

    print_metrics(run)


if __name__ == '__main__':
    main()

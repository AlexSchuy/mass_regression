PROJECTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
export PYTHONPATH="${PYTHONPATH}:${PROJECTPATH}/mass_regression"
pipenv shell

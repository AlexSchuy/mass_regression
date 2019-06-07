"""Utility Routines

These routines handle common utility functions for the other modules, such as
get common paths and handling config files.
"""
import os


def get_source_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_path():
    return os.path.dirname(get_source_path())


def get_results_path():
    return os.path.join(get_project_path(), 'results')


def get_run_path(run_number):
    return os.path.join(get_results_path(), run_number)


def list_to_str(l):
    s = ''
    for item in l[:-1]:
        s += str(item) + ';'
    s += str(l[-1])
    return s


def str_to_list(s, type_func=str):
    return [type_func(i) for i in s.split(';')]

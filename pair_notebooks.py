import os

from jupytext import cli

MAX_LINE = 5


def is_python_file(file):
    return file.endswith('.py')


def has_jupytext_metadata(dir_path, file, max_line):
    with open(os.path.join(dir_path, file)) as f:
        for i in range(max_line):
            if 'jupytext:' in f.readline():
                return True
    return False


"""
This is a utility scripts that search recursively for all the python percent scripts with jupyter metadata
and sets formats 'ipynb,py:percent' for those scripts. This operation results paring scripts with notebook files
"""
if __name__ == '__main__':
    rootdir = os.path.join(os.path.dirname(__file__), '../')
    for dir_path, _, files in os.walk(rootdir):
        for file in files:
            if is_python_file(file) and has_jupytext_metadata(dir_path, file, MAX_LINE):
                path = os.path.join(dir_path, file)
                norm_path = os.path.normpath(path)
                cli.jupytext(['--set-formats', 'ipynb,py:percent', norm_path])

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='mass_regression',
      version='0.0.1',
      author='Alex Schuy',
      author_email='alexjschuy@gmail.com',
      description='implementation of missing mass regression.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages())

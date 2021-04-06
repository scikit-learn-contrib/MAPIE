.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.com/simai-ml/MAPIE.svg?branch=master
.. _Travis: https://travis-ci.com/github/simai-ml/MAPIE

.. .. |AppVeyor| image:: 
.. .. _AppVeyor: 

.. |Codecov| image:: https://app.codecov.io/gh/simai-ml/MAPIE/branch/master/graph/badge.svg
.. _Codecov: https://app.codecov.io/gh/simai-ml/MAPIE/branch/master

.. .. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. .. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mapie/badge/?version=latest
.. _ReadTheDocs: https://mapie.readthedocs.io/en/latest/?badge=latest


MAPIE - Model Agnostic Prediction Interval Estimator
============================================================

**MAPIE** is a module that allows you to easily estimate prediction intervals using your favourite sklearn-compatible regressor.


Install Conda
=============

- :code:`curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh`
- :code:`source Miniconda3-latest-MacOSX-x86_64.sh`
- :code:`source ~/.bashrc`
- :code:`rm Miniconda3-latest-MacOSX-x86_64.sh`


Create virtual environment
==========================

- :code:`conda env create -f environment.yml`
- :code:`conda activate mapie`
- :code:`python -m ipykernel install --user --name=mapie`

Create html documentation from rst files
========================================

- :code:`pip install -e .`
- :code:`cd doc/`
- :code:`make clean`
- :code:`make html`
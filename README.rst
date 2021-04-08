.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.com/simai-ml/MAPIE.svg?branch=master
.. _Travis: https://travis-ci.com/github/simai-ml/MAPIE

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/mapie/notreadyyet

.. |Codecov| image:: https://codecov.io/gh/simai-ml/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1
.. _Codecov: https://codecov.io/gh/simai-ml/MAPIE

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/mapie/project-template/tree/master/notreadyyet

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mapie/badge/?version=latest
.. _ReadTheDocs: https://mapie.readthedocs.io/en/latest/?badge=latest


MAPIE - Model Agnostic Prediction Interval Estimator
============================================================

**MAPIE** allows you to easily estimate prediction intervals using your favourite sklearn-compatible regressor.
The documentation can be found `here <https://en.wikipedia.org/wiki/Uncertainty_quantification>`_.

Install Conda
=============

- :code:`wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh`
- :code:`chmod +x miniconda.sh`
- :code:`bash miniconda.sh -b -p $HOME/miniconda`
- :code:`export PATH=$HOME/miniconda/bin:$PATH`
- :code:`source $HOME/miniconda/etc/profile.d/conda.sh`
- :code:`conda update --yes conda`


Create virtual environment
==========================

- :code:`conda env create -f environment_dev.yml`
- :code:`conda activate mapie`
- :code:`python -m ipykernel install --user --name=mapie`

Create html documentation from rst files
========================================

- :code:`pip install -e .`
- :code:`cd doc/`
- :code:`make clean`
- :code:`make html`
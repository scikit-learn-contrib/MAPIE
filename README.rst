.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_ |License|_ |PythonVersion|_ |PyPi|_ |Anaconda|_

.. |Travis| image:: https://travis-ci.com/simai-ml/MAPIE.svg?branch=master
.. _Travis: https://travis-ci.com/simai-ml/MAPIE

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/github/MAPIE
.. _AppVeyor: https://ci.appveyor.com/project/gmartinonQM/mapie

.. |Codecov| image:: https://codecov.io/gh/simai-ml/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1
.. _Codecov: https://codecov.io/gh/simai-ml/MAPIE

.. |CircleCI| image:: https://circleci.com/gh/simai-ml/MAPIE.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/simai-ml/MAPIE

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mapie/badge
.. _ReadTheDocs: https://mapie.readthedocs.io/en/latest

.. |License| image:: https://img.shields.io/github/license/simai-ml/MAPIE
.. _Licence: https://github.com/simai-ml/MAPIE/blob/master/LICENSE

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/mapie
.. _PythonVersion: https://pypi.org/project/mapie/

.. |PyPi| image:: https://img.shields.io/pypi/v/mapie
.. _PyPi: https://pypi.org/project/mapie/

.. |Anaconda| image:: https://anaconda.org/conda-forge/mapie/badges/version.svg
.. _Anaconda: https://anaconda.org/conda-forge/mapie


MAPIE - Model Agnostic Prediction Interval Estimator
====================================================

**MAPIE** allows you to easily estimate prediction intervals using your favourite sklearn-compatible regressor.
The documentation can be found `here <https://mapie.readthedocs.io/en/latest/>`_.


🛠 Installation
===============

Install via `pip`:

.. code:: python

    pip install mapie

To install directly from the github repository :

.. code:: python

    pip install git+https://github.com/simai-ml/MAPIE


⚡️ Quickstart
============

Let us start with a basic regression problem. 
Here, we generate one-dimensional noisy data that we fit with a linear model.

.. code:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression

    regressor = LinearRegression()
    X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)

Since MAPIE is compliant with the standard scikit-learn API, we follow the standard
sequential ``fit`` and ``predict`` process  like any scikit-learn regressor.

.. code:: python

    from mapie.estimators import MapieRegressor
    mapie = MapieRegressor(regressor)
    mapie.fit(X, y)
    X_pi = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_preds = mapie.predict(X_pi)


MAPIE returns a ``np.ndarray`` of shape (3, n_sample) giving the predictions,
as well as the lower and upper bounds of the prediction intervals for the target quantile.
The estimated prediction interval can be easily plotted as follows.

.. code:: python
    
    from matplotlib import pyplot as plt
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X_pi, y_preds[:, 0], color='C1')
    plt.fill_between(X_pi.ravel(), y_preds[:, 1], y_preds[:, 2], alpha=0.3)
    plt.show()


.. image:: doc/images/quickstart_1.png
    :width: 400
    :align: center


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
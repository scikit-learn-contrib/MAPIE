.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_ |License|_ |PythonVersion|_ |PyPi|_

.. |Travis| image:: https://travis-ci.com/simai-ml/MAPIE.svg?branch=master
.. _Travis: https://travis-ci.com/simai-ml/MAPIE

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/js4d7km6ckr801nj/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/gmartinonQM/mapie

.. |Codecov| image:: https://codecov.io/gh/simai-ml/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1
.. _Codecov: https://codecov.io/gh/simai-ml/MAPIE

.. |CircleCI| image:: https://circleci.com/gh/simai-ml/MAPIE.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/simai-ml/MAPIE

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mapie/badge
.. _ReadTheDocs: https://mapie.readthedocs.io/en/latest

.. |License| image:: https://img.shields.io/github/license/simai-ml/MAPIE
.. _License: https://github.com/simai-ml/MAPIE/blob/master/LICENSE

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/mapie
.. _PythonVersion: https://pypi.org/project/mapie/

.. |PyPi| image:: https://img.shields.io/pypi/v/mapie
.. _PyPi: https://pypi.org/project/mapie/


.. image:: https://github.com/simai-ml/MAPIE/raw/master/doc/images/mapie_logo_nobg_cut.png
    :width: 400
    :align: center



MAPIE - Model Agnostic Prediction Interval Estimator
====================================================

**MAPIE** allows you to easily estimate prediction intervals on single-output data using your favourite scikit-learn-compatible regressor.

Prediction intervals output by **MAPIE** encompass both aleatoric and epistemic uncertainty and are backed by strong theoretical guarantees [1].

🔗 Requirements
===============

Python 3.7+

**MAPIE** stands on the shoulders of giants.

Its only internal dependency is `scikit-learn <https://scikit-learn.org/stable/>`_.


🛠 Installation
===============

Install via `pip`:

.. code:: python

    pip install mapie

To install directly from the github repository :

.. code:: python

    pip install git+https://github.com/simai-ml/MAPIE


⚡️ Quickstart
==============

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
We set two values for alpha to estimate prediction intervals at approximately one
and two standard deviations from the mean.

.. code:: python

    from mapie.estimators import MapieRegressor
    alpha = [0.05, 0.32]
    mapie = MapieRegressor(regressor, alpha=alpha)
    mapie.fit(X, y)
    y_preds = mapie.predict(X)



MAPIE returns a ``np.ndarray`` of shape (n_samples, 3, len(alpha)) giving the predictions,
as well as the lower and upper bounds of the prediction intervals for the target quantile
for each desired alpha value.
The estimated prediction intervals can then be plotted as follows. 

.. code:: python
    
    from matplotlib import pyplot as plt
    from mapie.metrics import coverage_score
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X, y_preds[:, 0, 0], color="C1")
    order = np.argsort(X[:, 0])
    plt.plot(X[order], y_preds[order][:, 1, 1], color="C1", ls="--")
    plt.plot(X[order], y_preds[order][:, 2, 1], color="C1", ls="--")
    plt.fill_between(X[order].ravel(), y_preds[:, 1, 0][order].ravel(), y_preds[:, 2, 0][order].ravel(), alpha=0.2)
    coverage_scores = [coverage_score(y, y_preds[:, 1, i], y_preds[:, 2, i]) for i, _ in enumerate(alpha)]
    plt.title(
        f"Target and effective coverages for alpha={alpha[0]:.2f}: ({1-alpha[0]:.3f}, {coverage_scores[0]:.3f})\n" +
        f"Target and effective coverages for alpha={alpha[1]:.2f}: ({1-alpha[1]:.3f}, {coverage_scores[1]:.3f})"
    )
    plt.show()

The title of the plot compares the target coverages with the effective coverages.
The target coverage, or the confidence interval, is the fraction of true labels lying in the
prediction intervals that we aim to obtain for a given dataset.
It is given by the alpha parameter defined in ``MapieRegressor``, here equal to 0.05 and 0.32,
thus giving target coverages of 0.95 and 0.68.
The effective coverage is the actual fraction of true labels lying in the prediction intervals.


.. image:: https://github.com/simai-ml/MAPIE/raw/master/doc/images/quickstart_1.png
    :width: 400
    :align: center


📘 Documentation
================

The documentation can be found `on this link <https://mapie.readthedocs.io/en/latest/>`_.
It contains the following sections:

- `Quickstart <https://mapie.readthedocs.io/en/latest/quick_start.html>`_
- `Theoretical description <https://mapie.readthedocs.io/en/latest/theoretical_description.html>`_
- `Tutorial <https://mapie.readthedocs.io/en/latest/tutorial.html>`_
- `API <https://mapie.readthedocs.io/en/latest/api.html>`_
- `Examples <https://mapie.readthedocs.io/en/latest/auto_examples/index.html>`_


📝 Contributing
===============

You are welcome to propose and contribute new ideas.
We encourage you to `open an issue <https://github.com/simai-ml/MAPIE/issues>`_ so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.
For more information on the contribution process, please go `here <CONTRIBUTING.rst>`_.


🤝  Affiliations
================

MAPIE has been developed through a collaboration between Quantmetry, Michelin, and ENS Paris-Saclay
with the financial support from Région Ile de France.

|Quantmetry|_ |Michelin|_ |ENS|_ |IledeFrance|_ 

.. |Quantmetry| image:: https://www.quantmetry.com/wp-content/uploads/2020/08/08-Logo-quant-Texte-noir.svg
    :width: 150
.. _Quantmetry: https://www.quantmetry.com/

.. |Michelin| image:: https://www.michelin.com/wp-content/themes/michelin/public/img/michelin-logo-en.svg
    :width: 100
.. _Michelin: https://www.michelin.com/en/

.. |ENS| image:: https://file.diplomeo-static.com/file/00/00/01/34/13434.svg
    :width: 100
.. _ENS: https://ens-paris-saclay.fr/en

.. |IledeFrance| image:: https://www.iledefrance.fr/themes/custom/portail_idf/logo.svg
    :width: 100
.. _IledeFrance: https://www.iledefrance.fr/


🔍  References
==============

MAPIE methods are based on the work by `Foygel-Barber et al. (2020) <https://www.stat.uchicago.edu/~rina/jackknife.html>`_.

[1] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani.
Predictive inference with the jackknife+. Ann. Statist., 49(1):486–507, 022021

📝 License
==========

MAPIE is free and open-source software licensed under the `3-clause BSD license <https://github.com/simai-ml/MAPIE/blob/master/LICENSE>`_.

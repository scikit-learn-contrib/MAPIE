.. -*- mode: rst -*-

|GitHubActions|_ |Codecov|_ |ReadTheDocs|_ |License|_ |PythonVersion|_ |PyPi|_ |Conda|_ |Release|_ |Commits|_ |DOI|_

.. |GitHubActions| image:: https://github.com/scikit-learn-contrib/MAPIE/actions/workflows/test.yml/badge.svg
.. _GitHubActions: https://github.com/scikit-learn-contrib/MAPIE/actions

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/MAPIE/branch/master/graph/badge.svg?token=F2S6KYH4V1
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/MAPIE

.. |ReadTheDocs| image:: https://readthedocs.org/projects/mapie/badge
.. _ReadTheDocs: https://mapie.readthedocs.io/en/latest

.. |License| image:: https://img.shields.io/github/license/simai-ml/MAPIE
.. _License: https://github.com/scikit-learn-contrib/MAPIE/blob/master/LICENSE

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/mapie
.. _PythonVersion: https://pypi.org/project/mapie/

.. |PyPi| image:: https://img.shields.io/pypi/v/mapie
.. _PyPi: https://pypi.org/project/mapie/

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/mapie
.. _Conda: https://anaconda.org/conda-forge/mapie

.. |Release| image:: https://img.shields.io/github/v/release/scikit-learn-contrib/mapie
.. _Release: https://github.com/scikit-learn-contrib/MAPIE/releases

.. |Commits| image:: https://img.shields.io/github/commits-since/scikit-learn-contrib/mapie/latest/master
.. _Commits: https://github.com/scikit-learn-contrib/MAPIE/commits/master

.. |DOI| image:: https://img.shields.io/badge/10.48550/arXiv.2207.12274-B31B1B.svg
.. _DOI: https://arxiv.org/abs/2207.12274

.. image:: https://github.com/simai-ml/MAPIE/raw/master/doc/images/mapie_logo_nobg_cut.png
    :width: 400
    :align: center



MAPIE - Model Agnostic Prediction Interval Estimator
====================================================

**MAPIE** is an open-source Python library for quantifying uncertainties and controlling the risks of machine learning models.
It is a scikit-learn-contrib project that allows you to:

- Easily **compute conformal prediction intervals** (or prediction sets) with controlled (or guaranteed) marginal coverage rate
  for regression [3,4,8], classification (binary and multi-class) [5-7] and time series [9].
- Easily **control risks** of more complex tasks such as multi-label classification,
  semantic segmentation in computer vision (probabilistic guarantees on recall, precision, ...) [10-12].
- Easily **wrap any model (scikit-learn, tensorflow, pytorch, ...) with, if needed, a scikit-learn-compatible wrapper**
  for the purposes just mentioned.

Here's a quick instantiation of MAPIE models for regression and classification problems related to uncertainty quantification
(more details in the Quickstart section):

.. code:: python

    # Uncertainty quantification for regression problem
    from mapie.regression import MapieRegressor
    mapie_regressor = MapieRegressor(estimator=regressor, method='plus', cv=5)

.. code:: python

    # Uncertainty quantification for classification problem
    from mapie.classification import MapieClassifier
    mapie_classifier = MapieClassifier(estimator=classifier, method='score', cv=5)

Implemented methods in **MAPIE** respect three fundamental pillars:

- They are **model and use case agnostic**, 
- They possess **theoretical guarantees** under minimal assumptions on the data and the model,
- They are based on **peer-reviewed algorithms** and respect programming standards.

**MAPIE** relies notably on the field of *Conformal Prediction* and *Distribution-Free Inference*.


🔗 Requirements
===============

- **MAPIE** runs on Python 3.7+.
- **MAPIE** stands on the shoulders of giants. Its only internal dependencies are `scikit-learn <https://scikit-learn.org/stable/>`_ and `numpy=>1.21 <https://numpy.org/>`_.


🛠 Installation
===============

**MAPIE** can be installed in different ways:

.. code:: sh

    $ pip install mapie  # installation via `pip`
    $ conda install -c conda-forge mapie  # or via `conda`
    $ pip install git+https://github.com/scikit-learn-contrib/MAPIE  # or directly from the github repository


⚡ Quickstart
=============

Here we propose two basic uncertainty quantification problems for regression and classification tasks with scikit-learn.

As **MAPIE** is compatible with the standard scikit-learn API, you can see that with just these few lines of code:

- How easy it is **to wrap your favorite scikit-learn-compatible model** around your model.
- How easy it is **to follow the standard sequential** ``fit`` and ``predict`` process like any scikit-learn estimator.

.. code:: python

    # Uncertainty quantification for regression problem
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    from mapie.regression import MapieRegressor


    X, y = make_regression(n_samples=500, n_features=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    regressor = LinearRegression()

    mapie_regressor = MapieRegressor(estimator=regressor, method='plus', cv=5)

    mapie_regressor = mapie_regressor.fit(X_train, y_train)
    y_pred, y_pis = mapie_regressor.predict(X_test, alpha=[0.05, 0.32])

.. code:: python

    # Uncertainty quantification for classification problem
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    from mapie.classification import MapieClassifier


    X, y = make_blobs(n_samples=500, n_features=2, centers=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    classifier = LogisticRegression()

    mapie_classifier = MapieClassifier(estimator=classifier, method='score', cv=5)

    mapie_classifier = mapie_classifier.fit(X_train, y_train)
    y_pred, y_pis = mapie_classifier.predict(X_test, alpha=[0.05, 0.32])


📘 Documentation
================

The full documentation can be found `on this link <https://mapie.readthedocs.io/en/latest/>`_.


📝 Contributing
===============

You are welcome to propose and contribute new ideas.
We encourage you to `open an issue <https://github.com/simai-ml/MAPIE/issues>`_ so that we can align on the work to be done.
It is generally a good idea to have a quick discussion before opening a pull request that is potentially out-of-scope.
For more information on the contribution process, please go `here <CONTRIBUTING.rst>`_.


🤝  Affiliations
================

MAPIE has been developed through a collaboration between Quantmetry, Michelin, ENS Paris-Saclay,
and with the financial support from Région Ile de France and Confiance.ai.

|Quantmetry|_ |Michelin|_ |ENS|_ |Confiance.ai|_  |IledeFrance|_ 

.. |Quantmetry| image:: https://www.quantmetry.com/wp-content/uploads/2020/08/08-Logo-quant-Texte-noir.svg
    :width: 150
.. _Quantmetry: https://www.quantmetry.com/

.. |Michelin| image:: https://www.michelin.com/wp-content/themes/michelin/public/img/michelin-logo-en.svg
    :width: 100
.. _Michelin: https://www.michelin.com/en/

.. |ENS| image:: https://file.diplomeo-static.com/file/00/00/01/34/13434.svg
    :width: 100
.. _ENS: https://ens-paris-saclay.fr/en

.. |Confiance.ai| image:: https://pbs.twimg.com/profile_images/1443838558549258264/EvWlv1Vq_400x400.jpg
    :width: 100
.. _Confiance.ai: https://www.confiance.ai/

.. |IledeFrance| image:: https://www.iledefrance.fr/themes/custom/portail_idf/logo.svg
    :width: 100
.. _IledeFrance: https://www.iledefrance.fr/


🔍  References
==============

[1] Vovk, Vladimir, Alexander Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer Nature, 2022.

[2] Angelopoulos, Anastasios N., and Stephen Bates. "Conformal prediction: A gentle introduction." Foundations and Trends® in Machine Learning 16.4 (2023): 494-591.

[3] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. "Predictive inference with the jackknife+." Ann. Statist., 49(1):486–507, (2021).

[4] Kim, Byol, Chen Xu, and Rina Barber. "Predictive inference is free with the jackknife+-after-bootstrap." Advances in Neural Information Processing Systems 33 (2020): 4138-4149.

[5] Sadinle, Mauricio, Jing Lei, and Larry Wasserman. "Least ambiguous set-valued classifiers with bounded error levels." Journal of the American Statistical Association 114.525 (2019): 223-234.

[6] Romano, Yaniv, Matteo Sesia, and Emmanuel Candes. "Classification with valid and adaptive coverage." Advances in Neural Information Processing Systems 33 (2020): 3581-3591.

[7] Angelopoulos, Anastasios, et al. "Uncertainty sets for image classifiers using conformal prediction." International Conference on Learning Representations (2021).

[8] Romano, Yaniv, Evan Patterson, and Emmanuel Candes. "Conformalized quantile regression." Advances in neural information processing systems 32 (2019).

[9] Xu, Chen, and Yao Xie. "Conformal prediction interval for dynamic time-series." International Conference on Machine Learning. PMLR, (2021).

[10] Bates, Stephen, et al. "Distribution-free, risk-controlling prediction sets." Journal of the ACM (JACM) 68.6 (2021): 1-34.

[11] Angelopoulos, Anastasios N., Stephen, Bates, Adam, Fisch, Lihua, Lei, and Tal, Schuster. "Conformal Risk Control." (2022).

[12] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al. "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022).


📝 License
==========

MAPIE is free and open-source software licensed under the `3-clause BSD license <https://github.com/simai-ml/MAPIE/blob/master/LICENSE>`_.

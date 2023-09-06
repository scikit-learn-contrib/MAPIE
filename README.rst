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

Quantifying the uncertainties and controlling the risks of ML model predictions is of crucial importance
for developing and deploying reliable artificial intelligence (AI) systems. Uncertainty quantification (UQ)
involves all the stakeholders who develop and use AI models.

**MAPIE** is an open-source Python library hosted on scikit-learn-contrib project that allows you to:

- easily **estimate conformal prediction intervals** (or prediction sets) given a degree of confidence or risk
  for single-output regression, binary and multi-class classification settings [3-9].
- easily **control risks** (such as coverage, recall or any other non-monotone risk) by estimating
  relevant prediction sets for multi-label classification and beyond [10-12].
- easily **wrap your favorite scikit-learn-compatible model** for the purposes just mentioned.

Here's a quick instantiation of MAPIE models for regression and classification problems related to uncertainty quantification
(more details in the Quickstart section):

.. code:: python

    from mapie.regression import MapieRegressor
    mapie_regressor = MapieRegressor(estimator=regressor, method='plus', cv=5)

.. code:: python

    from mapie.classification import MapieClassifier
    mapie_classifier = MapieClassifier(estimator=classifier, method='score', cv=5)

**MAPIE** has been designed to respect three fundamental pillars:

- Implemented methods are **model and use case agnostic** in order to address all relevant use cases tackled in industry.
- Implemented methods must have **strong theoretical guarantees** on the marginal coverage of the estimated uncertainties
  with as little assumption on the data or the model as possible.
- Implemented methods follow **state-of-the-art trends** that respect programming standards in order to develop trustworthy AI systems.

Importantly, **MAPIE** contributes to the wide diffusion of the attractive **Conformal Prediction** (CP) framework for regression
and classification settings that is model and use case agnostic with mathematical guarantees on the marginal coverages on the prediction sets
with few assumptions (distribution-free and data-exchangeability assumptions) [1-2].

Prediction sets output by **MAPIE** encompass both aleatoric and epistemic uncertainties
and are backed by strong theoretical guarantees using a variety of conformal prediction methods [3-9]
and a variety of conformal risk control methods [10-12].


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

- how easy it is **to wrap your favorite scikit-learn-compatible model** around your model.
- how easy it is **to follow the standard sequential** ``fit`` and ``predict`` process like any scikit-learn estimator.

.. code:: python

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

🔎 Further Explanations
=======================

Let us start with a basic regression problem. 
Here, we generate one-dimensional noisy data that we fit with a linear model.

.. code:: python

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    regressor = LinearRegression()
    X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

Since MAPIE is compliant with the standard scikit-learn API, we follow the standard
sequential ``fit`` and ``predict`` process  like any scikit-learn regressor.
We set two values for alpha to estimate prediction intervals at approximately one
and two standard deviations from the mean.

.. code:: python

    from mapie.regression import MapieRegressor

    mapie_regressor = MapieRegressor(regressor)
    mapie_regressor.fit(X_train, y_train)

    alpha = [0.05, 0.32]
    y_pred, y_pis = mapie_regressor.predict(X_test, alpha=alpha)

MAPIE returns a ``np.ndarray`` of shape ``(n_samples, 3, len(alpha))`` giving the predictions,
as well as the lower and upper bounds of the prediction intervals for the target quantile
for each desired alpha value.

You can compute the coverage of your prediction intervals.

.. code:: python
    
    from mapie.metrics import regression_coverage_score_v2

    coverage_scores = regression_coverage_score_v2(y_test, y_pis)

The estimated prediction intervals can then be plotted as follows. 

.. code:: python

    from matplotlib import pyplot as plt

    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, y, alpha=0.3)
    plt.plot(X_test, y_pred, color="C1")
    order = np.argsort(X_test[:, 0])
    plt.plot(X_test[order], y_pis[order][:, 0, 1], color="C1", ls="--")
    plt.plot(X_test[order], y_pis[order][:, 1, 1], color="C1", ls="--")
    plt.fill_between(
        X_test[order].ravel(),
        y_pis[order][:, 0, 0].ravel(),
        y_pis[order][:, 1, 0].ravel(),
        alpha=0.2
    )
    plt.title(
        f"Target and effective coverages for "
        f"alpha={alpha[0]:.2f}: ({1-alpha[0]:.3f}, {coverage_scores[0]:.3f})\n"
        f"Target and effective coverages for "
        f"alpha={alpha[1]:.2f}: ({1-alpha[1]:.3f}, {coverage_scores[1]:.3f})"
    )
    plt.show()

The title of the plot compares the target coverages with the effective coverages.
The target coverage, or the confidence interval, is the fraction of true labels lying in the
prediction intervals that we aim to obtain for a given dataset.
It is given by the alpha parameter defined in ``MapieRegressor``, here equal to 0.05 and 0.32,
thus giving target coverages of 0.95 and 0.68.
The effective coverage is the actual fraction of true labels lying in the prediction intervals.

.. image:: https://github.com/scikit-learn-contrib/MAPIE/raw/master/doc/images/quickstart_1.png
    :width: 400
    :align: center

Similarly, it's possible to do the same for a basic classification problem.

.. code:: python

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split

    classifier = LogisticRegression()
    X, y = make_blobs(n_samples=500, n_features=2, centers=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

.. code:: python

    from mapie.classification import MapieClassifier

    mapie_classifier = MapieClassifier(estimator=classifier, method='score', cv=5)
    mapie_classifier = mapie_classifier.fit(X_train, y_train)

    alpha = [0.05, 0.32]
    y_pred, y_pis = mapie_classifier.predict(X_test, alpha=alpha)

.. code:: python

    from mapie.metrics import classification_coverage_score_v2

    coverage_scores = classification_coverage_score_v2(y_test, y_pis)

.. code:: python

    from matplotlib import pyplot as plt

    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
    step = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    X_test_mesh = np.stack([xx.ravel(), yy.ravel()], axis=1)

    y_pis = mapie_classifier.predict(X_test_mesh, alpha=alpha)[1][:,:,0]

    plt.scatter(
        X_test_mesh[:, 0], X_test_mesh[:, 1],
        c=np.ravel_multi_index(y_pis.T, (2,2,2)),
        marker='.', s=10, alpha=0.2
    )
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab20c')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(
        f"Target and effective coverages for "
        f"alpha={alpha[0]:.2f}: ({1-alpha[0]:.3f}, {coverage_scores[0]:.3f})"
    )
    plt.show()

.. image:: https://github.com/scikit-learn-contrib/MAPIE/raw/master/doc/images/quickstart_2.png
    :width: 400
    :align: center

📘 Documentation
================

The full documentation can be found `on this link <https://mapie.readthedocs.io/en/latest/>`_.

**How does MAPIE work?** 

It is basically based on two types of techniques:

**Cross conformal predictions**

- Conformity scores on the whole training set obtained by cross-validation,
- Perturbed models generated during the cross-validation.

**MAPIE** then combines all these elements in a way that provides prediction intervals on new data with strong theoretical guarantees [3-4].

.. image:: https://github.com/simai-ml/MAPIE/raw/master/doc/images/mapie_internals_regression.png
    :width: 300
    :align: center

**Split conformal predictions**

- Construction of a conformity score
- Calibration of the conformity score on a calibration set not seen by the model during training

**MAPIE** then uses the calibrated conformity scores to estimate sets of labels associated with the desired coverage on new data with strong theoretical guarantees [5-6-7].

.. image:: https://github.com/simai-ml/MAPIE/raw/master/doc/images/mapie_internals_classification.png
    :width: 300
    :align: center



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

MAPIE methods belong to the field of conformal inference.

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
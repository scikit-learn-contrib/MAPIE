from __future__ import annotations
from typing import Optional, Union, Tuple, Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from ._typing import ArrayLike
from .utils import (
    check_null_weight,
    check_n_features_in,
    check_alpha,
    check_alpha_and_n_samples,
    check_n_jobs,
    check_verbose
)


class MapieClassifier (BaseEstimator, ClassifierMixin):  # type: ignore
    """
    Prediction sets for classification.

    This class implements several conformal prediction strategies for
    estimating prediction sets for classification. Instead of giving a
    single predicted label, the idea is to give a set of predicted labels
    (or prediction sets) which come with mathematically guaranteed coverages.

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default None.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "score", based on the the scores
          (i.e. 1 minus the softmax score of the true label)
          on the calibration set.
        - "cumulated_score", based on the sum of the softmax outputs of the
          labels until the true label is reached, on the calibration set
          (to be implemented).

          By default "score".

    cv: Optional[Union[float, str]]
        The cross-validation strategy for computing scores :

        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          to calibrate the predictions through the score computation.
          At prediction time, quantiles of these scores are used to estimate
          prediction sets.

        By default ``prefit``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        At this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for ‘unset’ that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    verbose : int, optional
        The verbosity level, used with joblib for multiprocessing.
        At this moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    single_estimator_ : sklearn.ClassifierMixin
        Estimator fitted on the whole training set.

    n_features_in_: int
        Number of features passed to the fit method.

    n_samples_val_: Union[int, List[int]]
        Number of samples passed to the fit method.

    scores_ : np.ndarray of shape (n_samples_train)
        The scores used to calibrate the prediction sets.

    quantiles_ : np.ndarray of shape (n_alpha)
        The quantiles estimated from ``scores_`` and alpha values.

    References
    ----------
    Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels",
    Journal of the American Statistical Association, 114, 2019.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie.classification import MapieClassifier
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> clf = GaussianNB().fit(X_toy, y_toy)
    >>> mapie = MapieClassifier(estimator=clf, cv="prefit").fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.2)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False False]
     [ True False False]
     [ True False False]
     [ True  True False]
     [False  True False]
     [False  True  True]
     [False False  True]
     [False False  True]
     [False False  True]]
    """

    valid_methods_ = [
        "score"
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "score",
        cv: Optional[Union[int, str, BaseCrossValidator]] = "prefit",
        n_jobs: Optional[int] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        if self.method not in self.valid_methods_:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'score'."
            )

        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)

    def _check_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
    ) -> ClassifierMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LogisticRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        estimator : Optional[ClassifierMixin], optional
            Estimator to check, by default ``None``

        Returns
        -------
        ClassifierMixin
            The estimator itself or a default ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit, predict, nor predict_proba methods.

        NotFittedError
            If the estimator is not fitted and ``cv`` attribute is "prefit".
        """
        if estimator is None:
            return LogisticRegression(multi_class="multinomial").fit(X, y)
        if (
                not hasattr(estimator, "fit")
                and not hasattr(estimator, "predict")
                and not hasattr(estimator, 'predict_proba')
        ):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )
        if self.cv == "prefit":
            if isinstance(self.estimator, Pipeline):
                check_is_fitted(self.estimator[-1])
            else:
                check_is_fitted(self.estimator)
        return estimator

    def _check_cv(
        self,
        cv: Optional[Union[int, str, BaseCrossValidator]] = None
    ) -> Optional[Union[float, str]]:
        """
        Check if cross-validator is ``None`` or ``"prefit"``.
        Else raise error.

        Parameters
        ----------
        cv : Optional[Union[int, str, BaseCrossValidator]], optional
            Cross-validator to check, by default ``None``.

        Returns
        -------
        Optional[Union[float, str]]
            'prefit' or None.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv is None:
            return "prefit"
        if cv == "prefit":
            return cv
        raise ValueError(
            "Invalid cv argument."
            "Allowed value is 'prefit'."
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None
    ) -> MapieClassifier:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no prediction sets.

            By default None.

        Returns
        -------
        MapieClassifier
            The model itself.
        """
        # Checks
        self._check_parameters()
        cv = self._check_cv(self.cv)
        estimator = self._check_estimator(X, y, self.estimator)
        X, y = check_X_y(
            X, y, force_all_finite=False, dtype=["float64", "int", "object"]
        )
        assert type_of_target(y) == "multiclass"
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        # Work
        self.single_estimator_ = estimator
        y_pred = self.single_estimator_.predict_proba(X)
        self.n_samples_val_ = X.shape[0]
        self.scores_ = np.take_along_axis(
            y_pred, y.reshape(-1, 1), axis=1
        )
        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prediction prediction sets on new samples based on target confidence
        interval.
        Prediction sets for a given ``alpha`` are deduced from :
        - quantiles of softmax scores (score method)

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``np.ndarray`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            sets.
            ``alpha`` is the complement of the target coverage level.
            By default ``None``.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]

        - np.ndarray of shape (n_samples,) if alpha is None.

        - Tuple[np.ndarray, np.ndarray] of shapes
        (n_samples,) and (n_samples, n_classes, n_alpha) if alpha is not None.
        """
        # Checks
        alpha_ = check_alpha(alpha)
        check_is_fitted(
            self,
            [
                "single_estimator_",
                "scores_",
                "n_features_in_",
                "n_samples_val_"
            ]
        )
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)
        y_pred_proba = self.single_estimator_.predict_proba(X)
        n = self.n_samples_val_
        if alpha_ is None:
            return np.array(y_pred)
        else:
            check_alpha_and_n_samples(alpha_, n)
            self.quantiles_ = np.stack([
                np.quantile(
                    self.scores_,
                    ((n + 1) * (_alpha)) / n,
                    interpolation="lower"
                ) for _alpha in alpha_
            ])
            prediction_sets = np.stack([
                y_pred_proba > (quantile)
                for quantile in self.quantiles_
            ], axis=2)
            return y_pred, prediction_sets

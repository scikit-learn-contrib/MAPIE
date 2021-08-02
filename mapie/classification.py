from __future__ import annotations
from typing import Optional, Union, Iterable

import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ._typing import ArrayLike
from .utils import check_null_weight, fit_estimator
from .utils import check_n_features_in, check_alpha


class MapieClassifier (BaseEstimator, ClassifierMixin):  # type: ignore
    """
    Prediction intervals for classification
    using conformal predictions (in development).

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit and predict methods), by default None.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "naive", based on training set scores

        By default "naive".

    cv: Optional[str]
        The cross-validation strategy for computing scores :

        - ``None``, to use mapie for fitting.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          for computing scores only.
          At prediction time, quantiles of these scores are used to provide
          a prediction interval with fixed width.
          The user has to take care manually that data for model fitting.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for ‘unset’ that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    verbose : int, optional
        The verbosity level, used with joblib for multiprocessing.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    single_estimator_ : sklearn.ClassifierMixin
        Estimator fit on the whole training set.

    n_features_in_: int
        Number of features passed to the fit method.

    n_samples_in_train_:int
        Number of samples passed to the fit method.

    References
    ----------


    Examples
    --------

    """
    valid_methods_ = [
        "naive"
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "naive",
        cv: Optional[str] = None,
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
            Is parameters are not valid.
        """
        if self.method not in self.valid_methods_:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'naive'."
            )

        if not isinstance(self.n_jobs, (int, type(None))):
            raise ValueError(
                "Invalid n_jobs argument. Must be an integer."
            )

        if self.n_jobs == 0:
            raise ValueError(
                "Invalid n_jobs argument. Must be different than 0."
            )

        if not isinstance(self.verbose, int):
            raise ValueError(
                "Invalid verbose argument. Must be an integer."
            )

        if self.verbose < 0:
            raise ValueError(
                "Invalid verbose argument. Must be non-negative."
            )

    def _check_estimator(
        self,
        estimator: Optional[ClassifierMixin] = None
    ) -> ClassifierMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LogisticRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
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
            and has no fit nor predict methods.

        NotFittedError
            If the estimator is not fitted and ``cv`` attribute is "prefit".
        """
        if estimator is None:
            return LogisticRegression()
        if not hasattr(estimator, "fit") and not hasattr(estimator, "predict"):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a regressor with fit and predict methods."
            )
        if self.cv == "prefit":
            if isinstance(self.estimator, Pipeline):
                check_is_fitted(self.estimator[-1])
            else:
                check_is_fitted(self.estimator)
        return estimator

    def _check_cv(
        self,
        cv: Optional[str] = None
    ) -> Union[str, None]:
        """
        Check if cross-validator is
        ``None`` or ``"prefit"``.
        Else raise error.

        Parameters
        ----------
        cv : Optional[str],by default ``None``

        Returns
        -------
        Union[str, None]
            'prefit' or None.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if (cv is None or cv == "prefit"):
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
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            By default None.

        Returns
        -------
        MapieClassifier
            The model itself.
        """
        # Checks
        self._check_parameters()
        cv = self._check_cv(self.cv)
        estimator = self._check_estimator(self.estimator)
        X, y = check_X_y(
            X, y, force_all_finite=False, dtype=["float64", "int", "object"]
        )
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        self.n_samples_in_train_ = X.shape[0]

        # Work
        if cv == "prefit":
            self.single_estimator_ = estimator
        else:
            self.single_estimator_ = fit_estimator(
                clone(estimator), X, y, sample_weight
            )
        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None
    ) -> np.ndarray:
        """

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``np.ndarray`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.
            By default ``None``.

        Returns
        -------

        - np.ndarray of shape (n_samples,) if alpha is None

        """
        # Checks
        check_alpha(alpha)
        check_is_fitted(
            self,
            [
                "single_estimator_",
                "n_features_in_",
                "n_samples_in_train_"
            ]
        )
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = np.array(self.single_estimator_.predict(X))
        return y_pred

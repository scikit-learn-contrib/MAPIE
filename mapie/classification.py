from __future__ import annotations
from typing import Optional, Union, Iterable, Any, cast

import numpy as np
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline

from ._typing import ArrayLike
from .utils import check_null_weight, fit_estimator


class MapieClassifier (BaseEstimator, ClassifierMixin):  # type: ignore
    """
    TO BE CONTINUED

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit and predict methods), by default None.
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

    method: str, optional
        Method to choose for prediction interval estimates.
        Choose among:

        - "naive", based on training set scores

        By default "naive".

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing residuals.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: ``sklearn.model_selection.LeaveOneOut()``
          (jackknife variants) or ``sklearn.model_selection.KFold()``
          (cross-validation variants)
        - ``"prefit"``, assumes that ``estimator`` has been fitted already,
          and the ``method`` parameter is ignored.
          All data provided in the ``fit`` method is then used
          for computing residuals only.
          At prediction time, quantiles of these residuals are used to provide
          a prediction interval with fixed width.
          The user has to take care manually that data for model fitting and
          residual estimate are disjoint.

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

    ensemble: bool, optional
        Determines how to return the predictions.
        If ``False``, returns the predictions from the single estimator
        trained on the full training dataset.
        If ``True``, returns the median of the prediction intervals computed
        from the out-of-folds models.
        The Jackknife+ interval can be interpreted as an interval
        around the median prediction,
        and is guaranteed to lie inside the interval,
        unlike the single estimator predictions.

        By default ``False``.

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
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        ensemble: bool = False,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.ensemble = ensemble
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

        if not isinstance(self.ensemble, bool):
            raise ValueError(
                "Invalid ensemble argument. Must be a boolean."
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
        estimator : Optional[RegressorMixin], optional
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
        cv: Optional[Union[int, str, BaseCrossValidator]] = None
    ) -> Union[str, BaseCrossValidator]:
        """
        Check if cross-validator is
        ``None``, ``int``, ``"prefit"``, ``KFold`` or ``LeaveOneOut``.
        Return a ``LeaveOneOut`` instance if integer equal to -1.
        Return a ``KFold`` instance if integer superior or equal to 2.
        Return a ``KFold`` instance if ``None``.
        Else raise error.

        Parameters
        ----------
        cv : Optional[Union[int, str, BaseCrossValidator]], optional
            Cross-validator to check, by default ``None``

        Returns
        -------
        Union[str, BaseCrossValidator]
            The cross-validator itself or a default ``KFold`` instance.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv is None:
            return KFold(n_splits=5)
        if isinstance(cv, int):
            if cv == -1:
                return LeaveOneOut()
            if cv >= 2:
                return KFold(n_splits=cv)
        if (
            isinstance(cv, KFold)
            or isinstance(cv, LeaveOneOut)
            or cv == "prefit"
        ):
            return cv
        raise ValueError(
            "Invalid cv argument. "
            "Allowed values are None, -1, int >= 2, 'prefit', "
            "KFold or LeaveOneOut."
        )

    def _check_alpha(
        self,
        alpha: Optional[Union[float, Iterable[float]]] = None
    ) -> Optional[np.ndarray]:
        """
        Check alpha and prepare it as a np.ndarray

        Parameters
        ----------
        alpha : Union[float, Iterable[float]]
        Can be a float, a list of floats, or a np.ndarray of floats.
        Between 0 and 1, represent the uncertainty of the confidence interval.
        Lower alpha produce larger (more conservative) prediction intervals.
        alpha is the complement of the target coverage level.
        Only used at prediction time. By default 0.1.

        Returns
        -------
        np.ndarray
            Prepared alpha.

        Raises
        ------
        ValueError
            If alpha is not a float or an Iterable of floats between 0 and 1.
        """
        if alpha is None:
            return alpha
        if isinstance(alpha, float):
            alpha_np = np.array([alpha])
        elif isinstance(alpha, Iterable):
            alpha_np = np.array(alpha)
        else:
            raise ValueError(
                "Invalid alpha. Allowed values are float or Iterable."
            )
        if len(alpha_np.shape) != 1:
            raise ValueError(
                "Invalid alpha. "
                "Please provide a one-dimensional list of values."
            )
        if alpha_np.dtype.type not in [np.float64, np.float32]:
            raise ValueError(
                "Invalid alpha. Allowed values are Iterable of floats."
            )
        if np.any((alpha_np <= 0) | (alpha_np >= 1)):
            raise ValueError(
                "Invalid alpha. Allowed values are between 0 and 1."
            )
        return alpha_np

    def _check_n_features_in(
        self,
        X: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
    ) -> int:
        """
        Check the expected number of training features.
        In general it is simply the number of columns in the data.
        If ``cv=="prefit"`` however,
        it can be deduced from the estimator's ``n_features_in_`` attribute.
        These two values absolutely must coincide.

        Parameters
        ----------
        estimator : ClassifierMixin
            Backend estimator of MAPIE.
        X : ArrayLike of shape (n_samples, n_features)
            Data passed into the ``fit`` method.

        Returns
        -------
        int
            Expected number of training features.

        Raises
        ------
        ValueError
            If there is an inconsistency between the shape of the dataset
            and the one expected by the estimator.
        """
        n_features_in: int = X.shape[1]
        if self.cv == "prefit" and hasattr(estimator, "n_features_in_"):
            if cast(Any, estimator).n_features_in_ != n_features_in:
                raise ValueError(
                    "Invalid mismatch between "
                    "X.shape and estimator.n_features_in_."
                )
        return n_features_in

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None
    ) -> MapieClassifier:
        """

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

        """
        # Checks
        self._check_parameters()
        cv = self._check_cv(self.cv)
        estimator = self._check_estimator(self.estimator)
        X, y = check_X_y(
            X, y, force_all_finite=False, dtype=["float64", "int", "object"]
        )
        self.n_features_in_ = self._check_n_features_in(X, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        self.n_samples_in_train_ = X.shape[0]

        # Work
        if cv == "prefit":
            self.single_estimator_ = estimator
            y_pred= self.single_estimator_.predict(X)
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
        check_is_fitted(
            self,
            [
                "single_estimator_",
                "n_features_in_",
                "n_samples_in_train_"
            ]
        )
        alpha_ = self._check_alpha(alpha)
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = np.array(self.single_estimator_.predict(X))
        return y_pred

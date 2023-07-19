from typing import Optional, Union, Tuple

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import (check_is_fitted,
                                      check_random_state,
                                      indexable)

from mapie._machine_precision import EPSILON
from mapie._typing import ArrayLike, NDArray

from mapie.conformity_scores import ConformityScore


class AbsoluteConformityScore(ConformityScore):
    """
    Absolute conformity score.

    The signed conformity score = y - y_pred.
    The conformity score is symmetrical.

    This is appropriate when the confidence interval is symmetrical and
    its range is approximatively the same over the range of predicted values.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(sym=True, consistency_check=True)

    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the predicted values
        and the observed ones, from the following formula:
        signed conformity score = y - y_pred
        """
        return np.subtract(y, y_pred)

    def get_estimation_distribution(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = y - y_pred
        <=> y = y_pred + signed conformity score

        ``conformity_scores`` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        return np.add(y_pred, conformity_scores)


class GammaConformityScore(ConformityScore):
    """
    Gamma conformity score.

    The signed conformity score = (y - y_pred) / y_pred.
    The conformity score is not symmetrical.

    This is appropriate when the confidence interval is not symmetrical and
    its range depends on the predicted values. Like the Gamma distribution,
    its support is limited to strictly positive reals.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(sym=False, consistency_check=False, eps=EPSILON)

    def _check_observed_data(
        self,
        y: ArrayLike,
    ) -> None:
        if not self._all_strictly_positive(y):
            raise ValueError(
                f"At least one of the observed target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be strictly positive, "
                "in conformity with the Gamma distribution support."
            )

    def _check_predicted_data(
        self,
        y_pred: ArrayLike,
    ) -> None:
        if not self._all_strictly_positive(y_pred):
            raise ValueError(
                f"At least one of the predicted target is negative "
                f"which is incompatible with {self.__class__.__name__}. "
                "All values must be strictly positive, "
                "in conformity with the Gamma distribution support."
            )

    @staticmethod
    def _all_strictly_positive(
        y: ArrayLike,
    ) -> bool:
        return not np.any(np.less_equal(y, 0))

    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike,
    ) -> NDArray:
        """
        Compute the signed conformity scores from the observed values
        and the predicted ones, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        """
        self._check_observed_data(y)
        self._check_predicted_data(y_pred)
        return np.divide(np.subtract(y, y_pred), y_pred)

    def get_estimation_distribution(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        signed conformity score = (y - y_pred) / y_pred
        <=> y = y_pred * (1 + signed conformity score)

        ``conformity_scores`` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        self._check_predicted_data(y_pred)
        return np.multiply(y_pred, np.add(1, conformity_scores))


class ConformalResidualFittingScore(ConformityScore):
    """
    ConformalResidualFittingScore (CRF) score.

    The signed conformity score = (y - y_pred) / r_pred. r_pred being the
    predicted residual (|y - y_pred|) of the base estimator.
    It is calculated by a model that learns to predict these residuals.
    The learning is done with the log of the residual and we use the
    exponential of the prediction to avoid negative values.

    The conformity score is symmetrical and allows the calculation of adaptive
    prediction intervals (taking X into account). It is possible to use it
    only with split and prefit methods (not with cross methods).

    Warning : if the estimator provided is not fitted a subset of the
    calibration data will be used to fit the model (50% by default).

    Parameters
    ----------
    residual_estimator: Optional[RegressorMixin]
        The model that learns to predict the residuals of the base estimator.
        It can be any regressor with scikit-learn API (i.e. with ``fit``
        and ``predict`` methods).
        If ``None``, estimator defaults to a ``LinearRegression`` instance.

    prefit: bool
        Specify if the ``residual_estimator` is already fitted or not.
        By default ``False``.

    split_size: Optional[Union[int, float]]
        The proportion of data that is used to fit the ``residual_estimator``.
        By default 0.5.

    random_state: Optional[Union[int, np.random.RandomState]]
        Pseudo random number used for random sampling.
        Pass an int for reproducible output across multiple function calls.
        By default ``None``.
    """

    def __init__(
        self,
        residual_estimator: Optional[RegressorMixin] = None,
        prefit: bool = False,
        split_size: Optional[Union[int, float]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ) -> None:
        super().__init__(sym=True, consistency_check=False)
        self.prefit = prefit
        self.residual_estimator = residual_estimator
        self.split_size = split_size
        self.random_state = random_state

    def _check_estimator(
        self, estimator: Optional[RegressorMixin] = None
    ) -> RegressorMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LinearRegression`` instance if necessary.
        If the ``prefit`` attribute is ``True``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator: Optional[RegressorMixin]
            Estimator to check, by default ``None``.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``LinearRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no ``fit`` nor ``predict`` methods.

        NotFittedError
            If the estimator is not fitted
            and ``prefit`` attribute is ``True``.
        """
        if estimator is None:
            return LinearRegression()
        else:
            if not (hasattr(estimator, "fit") and
                    hasattr(estimator, "predict")):
                raise ValueError(
                    "Invalid estimator. "
                    "Please provide a regressor with fit and predict methods."
                )
            if self.prefit:
                if isinstance(estimator, Pipeline):
                    check_is_fitted(estimator[-1])
                else:
                    check_is_fitted(estimator)
            return estimator

    def _check_parameters(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike
    ) -> Tuple[NDArray, NDArray, NDArray, RegressorMixin]:
        """
        Checks all the parameters of the class. Raises an error if the
        parameter are not well defined.

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y: ArrayLike
            Target values.

        y_pred: ArrayLike
            Predicted targets.

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, RegressorMixin]
        Well initiated and typed :
            - X
            - y
            - y_pred
            - residual_estimator
        """
        residual_estimator = self._check_estimator(
            self.residual_estimator
        )
        check_random_state(self.random_state)
        if self.split_size is None:
            self.split_size = 0.5
        X, y, y_pred = indexable(X, y, y_pred)
        X = np.array(X)
        y = np.array(y)
        y_pred = np.array(y_pred)
        return X, y, y_pred, residual_estimator

    def _fit_residual_estimator(
        self,
        residual_estimator_: RegressorMixin,
        X: NDArray,
        y: NDArray,
        y_pred: NDArray,
        calres_indexes: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Fit the residual estimator and returns the indexes used for the
        training of the base estimator and those needed for the calibration.

        Parameters
        ----------
        X: NDArray
            All the observed values used in the general fit.

        y: NDArray
            All the observed targets used in the general fit.

        y_pred: NDArray
            Predicted targets.

        calres_indexes: NDArray
            Indexes used for the training of the estimator and the calibration.

        Returns
        -------
        Tuple[NDArray, NDArray]
            - indexes needed for the calibration.
            - indexes used for the training of the base estimator.
        """
        (X_res_indexes,
         X_cal_indexes,
         y_res_indexes,
         y_cal_indexes) = train_test_split(
            calres_indexes,
            calres_indexes,
            test_size=self.split_size,
            random_state=self.random_state,
        )

        residuals = np.abs(np.subtract(
            y[y_res_indexes],
            y_pred[y_res_indexes]
        ))
        residual_estimator_targets = np.log(np.maximum(
            residuals,
            np.full(residuals.shape, self.eps)
        ))

        residual_estimator_ = residual_estimator_.fit(
            X[X_res_indexes],
            residual_estimator_targets
        )

        cal_index = X_cal_indexes
        train_index = list(set(np.arange(y_pred.shape[0])) - set(cal_index))

        self.residual_estimator_ = residual_estimator_

        return cal_index, np.array(train_index)

    def get_signed_conformity_scores(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: ArrayLike
    ) -> NDArray:
        """
        Computes the signed conformity score = (y - y_pred) / r_pred.
        r_pred being the predicted residual (y - y_pred) of the estimator.
        It is calculated by a model (``residual_estimator_``) that learns
        to predict this residual.

        The learning is done with the log of the residual and later we
        use the exponential of the prediction to avoid negative values.
        """
        X, y, y_pred, self.residual_estimator_ = self._check_parameters(
            X, y, y_pred
        )
        calres_indexes = np.argwhere(
                            np.logical_not(np.isnan(y_pred))
                        ).reshape((-1, ))
        if not self.prefit:
            cal_indexes, train_indexes = self._fit_residual_estimator(
                clone(self.residual_estimator_), X, y, y_pred, calres_indexes
            )
        else:
            cal_indexes = calres_indexes
            train_indexes = np.argwhere(np.isnan(y_pred)).reshape((-1,))

        normalizer = np.maximum(
            np.exp(self.residual_estimator_.predict(X[cal_indexes])),
            self.eps
        )
        signed_conformity_scores = np.divide(
            np.abs(np.subtract(y[cal_indexes], y_pred[cal_indexes])),
            normalizer
        )

        # reconstruct array with nan and conformity scores
        complete_signed_cs = np.zeros(y_pred.shape)
        complete_signed_cs[cal_indexes] = signed_conformity_scores
        complete_signed_cs[train_indexes] = np.full(
            (train_indexes.shape[0],),
            np.nan
        )
        return signed_conformity_scores

    def get_estimation_distribution(
        self,
        X: ArrayLike,
        y_pred: ArrayLike,
        conformity_scores: ArrayLike
    ) -> NDArray:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        `y_pred + conformity_scores * r_pred``.

        The learning has been done with the log of the residual so we use the
        exponential of the prediction to avoid negative values.

        ``conformity_scores`` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        r_pred = np.exp(self.residual_estimator_.predict(X))
        return np.add(y_pred, np.multiply(conformity_scores, r_pred))

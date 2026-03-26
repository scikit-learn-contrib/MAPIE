from typing import Any, Optional, Protocol, Tuple, Union, cast, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_random_state, indexable

from mapie.conformity_scores.regression import BaseFitRegressionScore
from mapie.conformity_scores.bounds.utils import Trainer


@runtime_checkable
class CovarianceEstimator(Protocol):
    def fit(
        self, X: NDArray, y: NDArray, y_pred: Optional[NDArray], **kwargs: Any
    ) -> Any: ...

    def predict(self, X: NDArray) -> NDArray: ...

    def get_distribution(self, X: NDArray) -> Tuple[NDArray, NDArray]: ...

    def get_covariance_matrix(self, X: NDArray) -> NDArray: ...


class MultivariateResidualNormalisedScore(BaseFitRegressionScore):
    r"""
    Multivariate Residual Normalised score.

    The conformity score = $\Vert\Sigma(X)^{-1/2}(Y-f(X))\Vert_2$. Sigma(X) being the
    predicted covariance of the base estimator.
    It is calculated by a model that learns to predict the covariance of these residuals.
    The learning is done by minimizing the NLL induced by a Gaussian model. For large dimension,
    we encourage to use the low_rank approximation to learn Sigma(X) = Diagonal(X) + R(X)R(X)^T
    where R() is low rank. To do so, use the ``init_trainer(self, input_dim, output_dim; mode="low_rank")`` function

    The conformity score allows the calculation of prediction sets with improved conditional coverage properties.
    (taking X into account). It is possible to use it only with split and prefit methods (not with cross methods).

    Warning : if the estimator provided is not fitted a subset of the
    calibration data will be used to fit the model (80% by default).

    References
    ----------
    [1] Braun, S., E. Berta, M. I. Jordan, and F. Bach.
    "Multivariate Standardized Residuals for Conformal Prediction."
    arXiv preprint arXiv:2507.20941 2025.

    Parameters
    ----------
    covariance_estimator_: Optional[RegressorMixin]
        The model that learns to predict the residuals of the base estimator.
        It can be any regressor with functions :
                - ``fit``
                - ``get_distribution``
                - ``get_covariance_matrix``
        If ``None``, estimator defaults to a ``Trainer`` instance.

    prefit: bool
        Specify if the ``covariance_estimator_`` is already fitted or not.
        By default ``False``.

    split_size: Optional[Union[int, float]]
        The proportion of data that is used to fit the ``covariance_estimator_``.
        By default it is the default value of
        ``sklearn.model_selection.train_test_split`` ie 0.25.

    random_state: Optional[Union[int, np.random.RandomState]]
        Pseudo random number used for random sampling.
        Pass an int for reproducible output across multiple function calls.
        By default ``None``.
    """

    def __init__(
        self,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        prefit: bool = False,
        split_size: Optional[Union[int, float]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        sym: bool = False,
        consistency_check: bool = False,
        **kwargs,
    ) -> None:
        """
        Initializes the Multivariate Residual Normalised Score estimator.

        Parameters
        ----------
        covariance_estimator : Optional[CovarianceEstimator], optional
            The model that learns to predict the covariance of the residuals.
            Must implement `fit`, `get_distribution`, and `get_covariance_matrix`.
            If None, defaults to a Trainer instance, by default None.
        prefit : bool, optional
            Specify if the `covariance_estimator` is already fitted, by default False.
        split_size : Optional[Union[int, float]], optional
            The proportion of data used to fit the `covariance_estimator`.
            Defaults to 0.25 (sklearn's default train_test_split behavior), by default None.
        random_state : Optional[Union[int, np.random.RandomState]], optional
            Pseudo-random number seed for reproducible sampling, by default None.
        sym : bool, optional
            Specifies if the conformity score should be symmetric, by default False.
        consistency_check : bool, optional
            Whether to perform consistency checks on the base estimator, by default False.
        """
        super().__init__(sym=sym, consistency_check=consistency_check)
        self.prefit = prefit
        self.covariance_estimator_ = covariance_estimator
        self.split_size = split_size
        self.random_state = random_state
        self.kwargs = kwargs
        self.is_fitted = False
        self.multi_output = True

    def _check_estimator(self, estimator: Optional[Any] = None) -> Any:
        """
        Checks if the estimator is `None` and returns a default `Trainer` instance if necessary.

        If the `prefit` attribute is True, it implicitly assumes the estimator
        is already fitted. Otherwise, it verifies the presence of required methods.

        Parameters
        ----------
        estimator : Optional[Any], optional
            The covariance estimator to check, by default None.

        Returns
        -------
        Any
            The validated estimator itself or a newly instantiated default `Trainer`.

        Raises
        ------
        ValueError
            If the provided estimator lacks `fit`, `get_distribution`, or `get_covariance_matrix` methods.
        """
        if estimator is None:
            return Trainer(self.input_dim, self.output_dim, **self.kwargs)
        else:
            if not (
                hasattr(estimator, "fit")
                and hasattr(estimator, "get_distribution")
                and hasattr(estimator, "get_covariance_matrix")
            ):
                raise ValueError(
                    "Invalid estimator. "
                    "Please provide a regressor with fit, get_distribution and get_covariance_matrix methods."
                )
            return estimator

    def _check_parameters(
        self, X: ArrayLike, y: ArrayLike, y_pred: Optional[ArrayLike] = None
    ) -> Tuple[
        NDArray, NDArray, Optional[NDArray], Any, Union[int, np.random.RandomState]
    ]:
        """
        Checks and validates all incoming parameters and target matrices.

        Parameters
        ----------
        X : ArrayLike
            Observed input features.
        y : ArrayLike
            Target values.
        y_pred : Optional[ArrayLike], optional
            Predicted target values from the base estimator, by default None.

        Returns
        -------
        Tuple[NDArray, NDArray, Optional[NDArray], Any, Union[int, np.random.RandomState]]
            A tuple containing the validated and properly typed:
            - X array
            - y array
            - y_pred array (or None)
            - covariance_estimator_
            - random_state instance
        """
        if y_pred is not None:
            X_, y_, y_pred_ = indexable(X, y, y_pred)
        else:
            X_, y_ = indexable(X, y)
            y_pred_ = None
        assert y_.ndim == 2, (
            "Multivariate Residual Normalised Score method only supports multivariate targets."
        )
        self.input_dim = X_.shape[-1]
        self.output_dim = y_.shape[-1]
        covariance_estimator_ = self._check_estimator(self.covariance_estimator_)
        random_state = check_random_state(self.random_state)
        return (
            cast(NDArray, X_),
            cast(NDArray, y_),
            cast(NDArray, y_pred_),
            covariance_estimator_,
            random_state,
        )

    def _fit_covariance_estimator(
        self,
        X: NDArray,
        y: NDArray,
        y_pred: Optional[NDArray] = None,
        **kwargs,
    ) -> Any:
        """
        Fits the residual covariance estimator on the provided data.

        Parameters
        ----------
        X : NDArray
            The observed input features used to train the covariance estimator.
        y : NDArray
            The target values used to train the covariance estimator.
        y : Optional[NDArray]
            The predicted values. If not None, the model learns the residuals,
            otherwise if learns the center and the covariance matrix.

        Returns
        -------
        CovarianceEstimator
            The newly fitted covariance estimator.
        """
        assert self.covariance_estimator_ is not None
        self.covariance_estimator_.fit(X, y, y_pred, **kwargs)

        return self.covariance_estimator_

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        y_pred: Optional[NDArray] = None,
        **kwargs,
    ) -> "MultivariateResidualNormalisedScore":
        """
        Fits the residual covariance estimator on the provided data.

        Parameters
        ----------
        X : NDArray
            The observed input features used to train the covariance estimator.
        y : NDArray
            The target values (or residuals) used to train the covariance estimator.

        Returns
        -------
        MultivariateResidualNormalisedScore
            The newly fitted score.
        """
        (X, y, y_pred, self.covariance_estimator_, _) = self._check_parameters(
            X, y, y_pred
        )

        self._fit_covariance_estimator(X, y, y_pred, **kwargs)
        self.is_fitted = True
        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Returns the predictions y_pred for the associated X values.

        Parameters
        ----------
        X : ArrayLike
            The input feature values.

        Returns
        -------
        NDArray
            An array y_pred of the prediction of the model.
        """
        assert self.covariance_estimator_ is not None
        X_array = cast(NDArray, np.asarray(X))
        return self.covariance_estimator_.predict(X_array)

    def get_signed_conformity_scores(
        self,
        y: ArrayLike,
        y_pred: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> NDArray:
        r"""
        Computes the multivariate standardized conformity score:
        $ \Vert \Sigma_{pred}^{-1/2}(y - y_{pred}) \Vert_2 $

        $\Sigma_{pred}$ is the predicted covariance matrix of the residuals, calculated
        by minimizing the NLL of a Gaussian model.

        Parameters
        ----------
        y : ArrayLike
            The true observed target values.
        y_pred : Optional[ArrayLike], optional
            Predicted target values. If None, the base model predictions are learned automatically.
        X : Optional[ArrayLike], optional
            The input feature values corresponding to the targets. Required for this method.

        Returns
        -------
        NDArray
            An array of conformity scores representing the Euclidean norm of the standardized residuals.

        Raises
        ------
        ValueError
            If `X` is None, or if `y_pred` contains NaN values.
        """
        X: Optional[ArrayLike] = kwargs.get("X", None)
        if X is None:
            raise ValueError(
                "Additional parameters must be provided for the method to "
                + "work (here `X` is missing)."
            )
        X = cast(ArrayLike, X)

        (X, y, y_pred, self.covariance_estimator_, random_state) = (
            self._check_parameters(X, y, y_pred)
        )

        if y_pred is not None and np.isnan(y_pred).any():
            raise ValueError("y_pred contains NaN values.")
        if y_pred is None:
            y_pred = self.predict(X)

        full_indexes = np.arange(len(y))

        if not self.is_fitted:
            raise ValueError("This score needs to be learned first.")

        cal_indexes = full_indexes
        X_cal = _safe_indexing(X, cal_indexes)
        y_cal = _safe_indexing(y, cal_indexes)

        y_pred_cal = _safe_indexing(y_pred, cal_indexes)
        Sigma_pred = self.covariance_estimator_.get_covariance_matrix(X_cal)
        conformity_scores = self._get_standardized_score(y_cal, y_pred_cal, Sigma_pred)

        return conformity_scores

    @staticmethod
    def _get_standardized_score(
        y: NDArray, y_pred: NDArray, Sigma_pred: NDArray
    ) -> NDArray:
        r"""
        Computes the standardized Euclidean distance $ \Vert \Sigma_{pred}^{-1/2}(y - y_{pred}) \Vert_2 $.

        Parameters
        ----------
        y : NDArray
            True target values of shape (n, d).
        y_pred : NDArray
            Predicted target values of shape (n, d).
        Sigma_pred : NDArray
            Predicted covariance matrices of shape (n, d, d).

        Returns
        -------
        NDArray
            An array of shape (n,) containing the Euclidean norm of the standardized residuals.
        """
        residuals = y - y_pred
        vals, vecs = np.linalg.eigh(Sigma_pred)
        inv_sqrt_vals = 1.0 / np.sqrt(vals)
        Sigma_inv_half = (vecs * inv_sqrt_vals[:, np.newaxis, :]) @ vecs.transpose(
            0, 2, 1
        )
        standardized_trainiduals = Sigma_inv_half @ residuals[..., np.newaxis]
        return np.linalg.norm(standardized_trainiduals.squeeze(-1), axis=1)

    def get_estimation_distribution(
        self,
        y_pred: Optional[ArrayLike],
        conformity_scores: ArrayLike,
        X: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Any:
        """
        Compute samples of the estimation distribution from the predicted
        values and the conformity scores, from the following formula:
        ``y_pred + conformity_scores * r_pred``.

        The learning has been done with the log of the residual so we use the
        exponential of the prediction to avoid negative values.

        ``conformity_scores`` can be either the conformity scores or
        the quantile of the conformity scores.
        """
        if X is None:
            raise ValueError(
                "Additional parameters must be provided for the method to "
                + "work (here `X` is missing)."
            )
        assert self.covariance_estimator_ is not None
        X_array = cast(NDArray, np.asarray(X))
        return self.covariance_estimator_.get_distribution(X_array)
        # if X is None:
        #     raise ValueError(
        #         "Additional parameters must be provided for the method to "
        #         + "work (here `X` is missing)."
        #     )

        # X = cast(ArrayLike, X)
        # if y_pred is not None:
        #     Sigma_X = self.covariance_estimator_.get_covariance_matrix(X)
        # else:
        #     y_pred, Sigma_X = self.covariance_estimator_.get_distribution(X)

        # return y_pred, Sigma_X

    # def get_estimation_covariance(
    #     self,
    #     X: ArrayLike,
    # ) -> Tuple[NDArray, NDArray]:
    #     _, Sigma = self.get_estimation_distribution(X)
    #     return Sigma

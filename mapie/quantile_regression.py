from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union, cast

from joblib import Parallel, delayed
import numpy as np
import numpy.ma as ma
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    _num_samples,
    _check_y,
)

from ._typing import ArrayLike, NDArray
from .aggregation_functions import aggregate_all, phi2D
from .subsample import Subsample
from .utils import (
    check_cv,
    check_alpha,
    check_alpha_and_n_samples,
    check_n_features_in,
    check_n_jobs,
    check_nan_in_aposteriori_prediction,
    check_null_weight,
    check_verbose,
    fit_estimator,
)
from ._compatibility import np_quantile
from .regression import MapieRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


class MapieQuantileRegressor(MapieRegressor):
    fit_attributes = [
        "list_estimators",
        "list_y_preds"
    ]

    def __init__(
        self,
        estimator: RegressorMixin = GradientBoostingRegressor(),
        method: str = "quantile",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        alpha: float = 0.1
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.agg_function = agg_function
        self.verbose = verbose
        self.alpha = alpha
        self.valid_methods_.append("quantile")

    def _check_alpha(
        self,
        alpha: ArrayLike,
    ) -> ArrayLike:
        if len(alpha) != 2:
            raise ValueError(
                "We need two values for the alphas"
            )
        alpha[0] = cast(Optional[NDArray], check_alpha(alpha[0]))[0]
        alpha[1] = cast(Optional[NDArray], check_alpha(alpha[1]))[0]
        if (alpha[0] >= alpha[1]):
            raise ValueError(
                "First alpha has to be stricly smaller than the second alpha"
            )
        return alpha

    def _fit_and_predict_oof_model(
        self,
        estimator: RegressorMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        val_index: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        alpha: Optional[Union[float, ArrayLike]] = None,
    ) -> Tuple[Union[RegressorMixin, Tuple[RegressorMixin]],
               Union[NDArray, Tuple[NDArray]], ArrayLike]:
        """
        Fit a single out-of-fold model on a given training set and
        perform predictions on a test set.

        Parameters
        ----------
        estimator : RegressorMixin
            Estimator to train.

        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
            Input labels.

        train_index : ArrayLike of shape (n_samples_train)
            Training data indices.

        val_index : ArrayLike of shape (n_samples_val)
            Validation data indices.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default ``None``.

        Returns
        -------
        Tuple[RegressorMixin, NDArray, ArrayLike]

        - [0]: RegressorMixin, fitted estimator
        - [1]: NDArray of shape (n_samples_val,),
          estimator predictions on the validation fold.
        - [3]: ArrayLike of shape (n_samples_val,),
          validation data indices.
        """
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        X_val = _safe_indexing(X, val_index)
        return_val_index = val_index
        sample_weight_train = None
        return_estimator = fit_estimator(estimator, X_train, y_train,
                                         sample_weight_train)
        return_y_pred = np.array([])
        if _num_samples(X_val) > 0:
            if sample_weight is not None:
                sample_weight_train = _safe_indexing(sample_weight,
                                                     train_index)
            if (isinstance(alpha, list)):
                estimators = []
                y_preds = []
                alpha_copy = alpha.copy()
                alpha_copy.append(0.5)
                for item in alpha_copy:
                    estimator_cloned = clone(estimator)
                    estimator_cloned.alpha = item
                    estimator_cloned_ = fit_estimator(
                            estimator_cloned,
                            X_train,
                            y_train,
                            sample_weight_train)
                    estimators.append(estimator_cloned_)
                    y_pred = estimator_cloned_.predict(X_val)
                    y_preds.append(y_pred)
                return_estimator = tuple(estimators)
                return_y_pred = tuple(y_preds)
                return_val_index = tuple([val_index, val_index, val_index])
            else:
                return_estimator = fit_estimator(
                                estimator,
                                X_train,
                                y_train,
                                sample_weight_train
                            )
                return_y_pred = return_estimator.predict(X_val)
        return return_estimator, return_y_pred, return_val_index

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieQuantileRegressor:
        """
        Fit estimator and compute residuals used for prediction intervals.
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold residuals are stored under
        the ``conformity_scores_`` attribute.

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

            By default ``None``.

        Returns
        -------
        MapieRegressor
            The model itself.
        """
        # Checks
        self._check_parameters()
        cv = check_cv(self.cv)
        estimator = self._check_estimator(self.estimator)
        self.alpha = [self.alpha/2, (1-(self.alpha/2))]
        alpha = self._check_alpha(self.alpha)
        alpha_copy = alpha.copy()
        alpha_copy.append(0.5)
        agg_function = self._check_agg_function(self.agg_function)
        X, y = indexable(X, y)
        y = _check_y(y)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)
        n_samples = _num_samples(y)

        self.single_estimator_ = fit_estimator(
            clone(estimator), X, y, sample_weight
        )

        # Initialization
        self.estimators_: List[RegressorMixin] = []

        # Work
        if cv == "simple":
            X_fit, X_calib, y_fit, y_calib = train_test_split(
                X,
                y,
                test_size=0.5,
                random_state=42
            )
            list_estimators = []
            list_y_preds_calib = []
            for item in alpha_copy:
                estimator_cloned = clone(estimator)
                estimator_cloned.alpha = item
                estimator_cloned_ = fit_estimator(
                    estimator_cloned, X_fit, y_fit, sample_weight
                )
                list_estimators.append(estimator_cloned_)
                list_y_preds_calib.append(estimator_cloned_.predict(X_calib))
                self.y_calib = y_calib
        else:
            cv = cast(BaseCrossValidator, cv)
            num_splits = cv.get_n_splits(X, y)
            self.k_ = np.full(
                shape=(n_samples, num_splits),
                fill_value=np.nan,
                dtype=float,
            )
            pred_matrix = np.full(
                shape=(n_samples, num_splits),
                fill_value=np.nan,
                dtype=float,
            )
            outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_and_predict_oof_model)(
                    clone(estimator),
                    X,
                    y,
                    train_index,
                    val_index,
                    sample_weight,
                    alpha_copy
                )
                for train_index, val_index in cv.split(X)
            )
            self.estimators_, predictions, val_indices = map(
                list, zip(*outputs)
            )
            list_estimators = []
            list_y_preds_calib = []
            self.list_k = []
            for x in np.arange(3):
                est = []
                for j, _ in enumerate(self.estimators_):
                    est.append(self.estimators_[j][x])
                list_estimators.append(est)
                pred_matrix_copy = pred_matrix.copy()
                self.k_copy_ = self.k_.copy()
                for i, val_ind in enumerate(val_indices):
                    pred_matrix_copy[val_ind, i] = np.array(predictions[i][x])
                    self.k_copy_[val_ind, i] = 1
                check_nan_in_aposteriori_prediction(pred_matrix_copy)
                y_pred = aggregate_all(agg_function, pred_matrix_copy)
                list_y_preds_calib.append(y_pred)
                self.list_k.append(self.k_copy_)

        self.list_y_preds_calib = list_y_preds_calib
        self.list_estimators = list_estimators
        self.conformity_scores_ = np.max(
            [self.list_y_preds_calib[0]-self.y_calib,
             self.y_calib-self.list_y_preds_calib[1]],
            axis=0
            )
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        symmetry: Optional[bool] = True,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from either

        - quantiles of residuals (naive and base methods),
        - quantiles of (predictions +/- residuals) (plus method),
        - quantiles of (max/min(predictions) +/- residuals) (minmax method).

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Boolean determining whether the predictions are ensembled or not.
            If False, predictions are those of the model trained on the whole
            training set.
            If True, predictions from perturbed models are aggregated by
            the aggregation function specified in the ``agg_function``
            attribute.

            If cv is ``"prefit"``, ``ensemble`` is ignored.

            By default ``False``.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represents the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is None.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples,) and (n_samples, 2, n_alpha) if alpha is not None.

            - [:, 0, :]: Lower bound of the prediction interval.
            - [:, 1, :]: Upper bound of the prediction interval.
        """
        # Checks
        # check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha = cast(Optional[NDArray], check_alpha(self.alpha[0]*2))
        y_pred = self.single_estimator_.predict(X)
        n = len(self.conformity_scores_)

        if alpha is None:
            return np.array(y_pred)

        alpha_np = cast(NDArray, alpha)
        check_alpha_and_n_samples(alpha_np, n)

        self.conformity_scores_ = np.full(
                shape=(3, n),
                fill_value=self.conformity_scores_
            )

        y_preds = np.full(
                shape=(3, X.shape[0]),
                fill_value=y_pred
            )
        n = len(self.y_calib)
        q = (1-(alpha_np/2))*(1+(1/n))

        self.conformity_scores_[0] = self.list_y_preds_calib[0]-self.y_calib
        self.conformity_scores_[1] = self.y_calib-self.list_y_preds_calib[1]
        self.conformity_scores_[2] = np.max(
                [self.conformity_scores_[0],
                 self.conformity_scores_[1]],
                axis=0)

        y_preds[0] = self.list_estimators[0].predict(X)
        y_preds[1] = self.list_estimators[1].predict(X)
        y_preds[2] = self.list_estimators[2].predict(X)
        if symmetry is True:
                q = (1-(alpha_np))*(1+(1/n))
                quantile = np.full(2, np_quantile(
                    self.conformity_scores_[2], q, method="higher"
                ))
        else:
                quantile = np.full(shape=2, fill_value=np.nan)
                quantile[0] = np_quantile(
                    self.conformity_scores_[0], q, method="higher"
                )
                quantile[1] = np_quantile(
                    self.conformity_scores_[1], q, method="higher"
                )
        y_pred_low = y_preds[0][:, np.newaxis] - quantile[0]
        y_pred_up = y_preds[1][:, np.newaxis] + quantile[1]
        print("q mapie ", q)
        print("quantile mapie ", quantile)
        return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)
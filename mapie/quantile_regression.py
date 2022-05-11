from __future__ import annotations
import warnings
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    # _num_samples,
    _check_y,
)

from ._typing import ArrayLike, NDArray
from .utils import (
    check_alpha,
    check_alpha_and_n_samples,
    check_n_features_in,
    # check_n_jobs,
    # check_nan_in_aposteriori_prediction, <-- check what happens
    # when you also include this!
    check_null_weight,
    # check_verbose,
    fit_estimator,
)
from ._compatibility import np_quantile
from .regression import MapieRegressor


class MapieQuantileRegressor(MapieRegressor):
    valid_methods_ = ["quantile"]
    valid_agg_functions_ = [None, "median", "mean"]
    fit_attributes = [
        "list_estimators",
        "list_y_preds_calib",
        "conformity_scores_",
        "y_calib"
    ]

    link_estimator_quantile = {
        "GradientBoostingRegressor": ["loss", "alpha"],
        "QuantileRegressor": ["quantile", "quantile"]
    }

    def __init__(
        self,
        estimator: RegressorMixin = None,
        method: str = "quantile",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        n_jobs: Optional[int] = None,
        agg_function: Optional[str] = "mean",
        verbose: int = 0,
        alpha: float = 0.2
    ) -> None:
        self.alpha = alpha
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
            agg_function=agg_function,
            verbose=verbose,
        )
        if n_jobs is not None:
            raise NotImplementedError

    def _check_alpha(
        self,
        alpha: float,
    ) -> ArrayLike:
        if isinstance(alpha, float) is False:
            raise ValueError(
                "Invalid alpha. Allowed values are floats."
            )
        if ((alpha >= 0.5) or (alpha <= 0)):
            raise ValueError(
                "The alpha value has to be lower than 0.5."
                "Recall that this is a two sided value."
            )
        # So we don't need these right?
        alpha = [alpha/2, 1 - alpha/2]
        alpha[0] = cast(Optional[NDArray], check_alpha(alpha[0]))[0]
        alpha[1] = cast(Optional[NDArray], check_alpha(alpha[1]))[0]
        return alpha

    def _check_estimator(
        self, estimator: Optional[RegressorMixin] = None
    ) -> RegressorMixin:
        if estimator is None:
            return QuantileRegressor(solver="highs")
        if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a regressor with fit and predict methods."
            )
        name_estimator = estimator.__class__.__name__
        if name_estimator != "QuantileRegressor":
            if name_estimator in self.link_estimator_quantile:
                param_estimator = estimator.get_params()
                if (
                    self.link_estimator_quantile[name_estimator][0]
                    in param_estimator
                ):
                    if param_estimator[
                        self.link_estimator_quantile[
                            name_estimator][0]] != "quantile":
                        raise ValueError(
                            "You need to set the loss/metric"
                            "value to quantile."
                        )
                    else:
                        if (
                            self.link_estimator_quantile[name_estimator][1]
                            in param_estimator
                        ):
                            pass
                        else:
                            # Not sure how to test for the next two
                            raise ValueError(
                                "The matching parameter for the quantile"
                                "value of this method does not exist."
                            )
                else:
                    raise ValueError(
                        "The value specified as link to the quantile"
                        "method does not exist."
                    )
            else:
                raise ValueError(
                    "We cannot find your method to have a link"
                    "to a quantile method."
                )
        return estimator

    def _check_size_calib_set(self, y, alpha) -> None:
        value = 2/((len(y)/2)+1)
        if alpha < value:
            raise ValueError(
                "The calibration set is too small."
            )

    def _check_validity_results(
        self,
        y_preds: NDArray,
        y_pred_low: ArrayLike,
        y_pred_up: ArrayLike
    ) -> None:
        """_summary_

        Parameters
        ----------
        y_preds : _type_
            _description_
        y_pred_low : _type_
            _description_
        y_pred_up : _type_
            _description_

        Raises
        ------
        Warning
            If the aggregated predictions of any training sample would be nan.
        Examples
        --------
        >>> import warnings
        >>> warnings.filterwarnings("error")
        >>> import numpy as np
        >>> from mapie.quantile_regression import _check_validity_results
        >>> y_preds = np.array([[1, 2, 3],[3, 4, 5],[2, 3, 4]])
        >>> y_pred_low = np.array([4, 3, 2])
        >>> y_pred_up = np.array([4, 4, 4])
        >>> try:
        ...     _check_validity_results(y_preds, y_pred_low, y_pred_up)
        ... except Exception as exception:
        ...     print(exception)
        ...
        WARNING: Not correct.
        """
        if (
            (y_pred_low >= y_pred_up).all() or
                (y_preds[2] <= y_pred_low).all() or
                (y_preds[2] >= y_pred_up).all()):
            warnings.warn(
                "WARNING: Not correct."
            )

    def _check_cv(
        self,
        cv: Optional[Union[int, str, BaseCrossValidator]] = None
    ) -> Union[str, BaseCrossValidator]:
        if cv is None:
            cv == "split"
            return cv
        if cv == "split":
            return cv
        else:
            raise ValueError(
                "Invalid cv method."
            )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieQuantileRegressor:
        # Checks
        self._check_parameters()
        cv = self._check_cv(self.cv)
        estimator = self._check_estimator(self.estimator)
        alpha = self._check_alpha(self.alpha)
        self._check_size_calib_set(y, self.alpha)
        alpha_copy = alpha.copy()
        alpha_copy.append(0.5)
        # agg_function = self._check_agg_function(self.agg_function)
        X, y = indexable(X, y)
        y = _check_y(y)
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)
        y = cast(NDArray, y)
        # n_samples = _num_samples(y)

        # Initialization
        self.estimators_: List[RegressorMixin] = []

        # Work
        if sample_weight is None:
            (X_fit, X_calib,
                y_fit, y_calib) = train_test_split(
                X,
                y,
                test_size=0.5,
                random_state=42
            )
            sample_weight_fit = sample_weight
        else:
            (X_fit, X_calib,
                y_fit, y_calib,
                sample_weight_fit, sample_weight_calib) = train_test_split(
                X,
                y,
                sample_weight,
                test_size=0.5,
                random_state=42
            )
        list_estimators = []
        list_y_preds_calib = []
        name_estimator = estimator.__class__.__name__
        for item in alpha_copy:
            estimator_cloned = clone(estimator)
            params = {
                self.link_estimator_quantile[name_estimator][1]: item
            }
            estimator_cloned.set_params(**params)
            estimator_cloned_ = fit_estimator(
                estimator_cloned, X_fit, y_fit, sample_weight_fit
            )
            list_estimators.append(estimator_cloned_)
            y_pred_calib = estimator_cloned_.predict(X_calib)
            # check_nan_in_aposteriori_prediction(y_pred_calib)
            list_y_preds_calib.append(y_pred_calib)
            self.y_calib = y_calib

        self.list_y_preds_calib = list_y_preds_calib
        self.list_estimators = list_estimators

        self.conformity_scores_ = np.full(
                shape=(3, len(y_calib)),
                fill_value=np.nan
            )
        self.conformity_scores_[0] = self.list_y_preds_calib[0]-self.y_calib
        self.conformity_scores_[1] = self.y_calib-self.list_y_preds_calib[1]
        self.conformity_scores_[2] = np.max(
                [self.conformity_scores_[0],
                 self.conformity_scores_[1]],
                axis=0)
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        symmetry: Optional[bool] = True,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:

        # Checks
        check_is_fitted(self, self.fit_attributes)
        self._check_ensemble(ensemble)
        alpha = cast([NDArray], check_alpha(self.alpha))
        alpha_np = cast(NDArray, alpha)
        n = len(self.y_calib)
        check_alpha_and_n_samples(alpha_np, n)

        y_preds = np.full(
                shape=(3, X.shape[0]),
                fill_value=np.nan
            )
        n = len(self.y_calib)
        q = (1-(alpha_np/2))*(1+(1/n))

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
        self._check_validity_results(y_preds, y_pred_low, y_pred_up)
        return y_preds[2], np.stack([y_pred_low, y_pred_up], axis=1)

from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union, cast

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor
# from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import (
    indexable,
    check_is_fitted,
    # _num_samples,
    _check_y,
)
from typing_extensions import TypedDict

from ._typing import ArrayLike, NDArray
from .utils import (
    # check_alpha,
    # check_alpha_and_n_samples,
    # check_n_features_in,
    # check_n_jobs,
    check_nan_in_aposteriori_prediction,
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
        "estimators_",
        "y_calib_pred",
        "conformity_scores_"
    ]

    Params = TypedDict(
        "Params",
        {
            "loss_name": str,
            "alpha_name": str
        },
    )
    quantile_estimator_params = {
        "GradientBoostingRegressor": Params(
            loss_name="loss",
            alpha_name="alpha"
            ),
        "QuantileRegressor": Params(
            loss_name="quantile",
            alpha_name="quantile"
            ),
        "LGBMRegressor": Params(
            loss_name="objective",
            alpha_name="alpha"
            )
    }

    def __init__(
        self,
        estimator: RegressorMixin = None,
        method: str = "quantile",
        cv: Optional[str] = None,
        alpha: float = 0.2
    ) -> None:
        self.alpha = alpha
        super().__init__(
            estimator=estimator,
            method=method,
            cv=cv
        )

    def _check_alpha(
        self,
        alpha: float,
    ) -> ArrayLike:
        if alpha is None:
            alpha = 0.2
        if isinstance(alpha, float):
            if np.any(np.logical_or(alpha <= 0, alpha > 0.5)):
                raise ValueError(
                    "Invalid alpha. Allowed values are between 0 and 0.5."
                )
            else:
                alpha_np = np.array([alpha / 2, 1 - alpha / 2, 0.5])
        else:
            raise ValueError(
                "Invalid alpha. Allowed values are float."
            )
        return alpha_np

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
        if name_estimator == "QuantileRegressor":
            return estimator
        else:
            if name_estimator in self.quantile_estimator_params:
                param_estimator = estimator.get_params()
                if (
                    self.quantile_estimator_params[name_estimator]["loss_name"]
                    in param_estimator
                ):
                    if param_estimator[
                        self.quantile_estimator_params[
                            name_estimator]["loss_name"]] != "quantile":
                        raise ValueError(
                            "You need to set the loss/metric of your base"
                            + " model to quantile."
                        )
                    else:
                        if (
                            self.quantile_estimator_params[
                                name_estimator
                                ]["alpha_name"]
                            in param_estimator
                        ):
                            return estimator
                        else:
                            # Not sure how to test for the next two
                            raise ValueError(
                                "The matching parameter alpha_name for"
                                + " estimator does not exist."
                            )
                else:
                    raise ValueError(
                        "The matching parameter loss_name for"
                        + " estimator does not exist."
                    )
            else:
                raise ValueError(
                    "The base model does not seem to be accepted "
                    + " by MapieQuantileRegressor."
                )
        return estimator

    def _check_size_calib_set(self, y, alpha) -> None:
        value = 2/((len(y)/2)+1)
        if alpha < value:
            raise ValueError(
                "The calibration set is too small."
            )

    def _check_lower_upper_bounds(
        self,
        y_preds: NDArray,
        y_pred_low: ArrayLike,
        y_pred_up: ArrayLike
    ) -> None:
        """_summary_

        Parameters
        ----------
        y_pred : _type_
            All the predictions at quantile:
            alpha/2, (1 - alpha/2), 0.5.
        y_pred_low : _type_
            Final lower bound prediction with additional quantile
            value added
        y_pred_up : _type_
            Final upper bound prediction with additional quantile
            value added

        Raises
        ------
        Warning
            If the prediction at alpha/2 are greater than the predictions
            at the median and check that predictions at (1- alpha/2) are
            smaller than predictions at the median.
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
        WARNING: The addition of the extra value has made it that:
        The final prediction values obtained are now not
        all lower for quantile: alpha/2 compared to (1 - alpha/2)
        or some predictions from quantile: alpha/2 are greater
        than those of 0.5 or some predictions from quantile:
        (1 - alpha/2) are lower than those of 0.5.
        """
        if np.any(np.logical_or(
            y_preds[0] >= y_preds[1],
            y_preds[2] <= y_preds[0],
            y_preds[2] >= y_preds[1]
            )
        ):
            warnings.warn(
                "WARNING: The initial prediction values obtained are now not "
                "all lower for quantile: alpha/2 compared to (1 - alpha/2) "
                "or some predictions from quantile: alpha/2 are greater "
                "than those of 0.5 or some predictions from quantile: "
                "(1 - alpha/2) are lower than those of 0.5."

            )
        elif np.any(np.logical_or(
            y_pred_low >= y_pred_up,
            y_preds[2] <= y_pred_low,
            y_preds[2] >= y_pred_up
            )
        ):
            warnings.warn(
                "WARNING: The addition of the extra value has made it that:"
                "The final prediction values obtained are now not "
                "all lower for quantile: alpha/2 compared to (1 - alpha/2) "
                "or some predictions from quantile: alpha/2 are greater "
                "than those of 0.5 or some predictions from quantile: "
                "(1 - alpha/2) are lower than those of 0.5."
            )

    def _check_cv(
        self,
        cv: Optional[str] = None
    ) -> str:
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
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_calib: ArrayLike,
        y_calib: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieQuantileRegressor:
        # Checks
        self._check_parameters()
        estimator = self._check_estimator(self.estimator)
        alpha = self._check_alpha(self.alpha)
        self._check_size_calib_set(y_calib, self.alpha)
        X_train, y_train = indexable(X_train, y_train)
        X_calib, y_calib = indexable(X_calib, y_calib)
        y_train = _check_y(y_train)
        self.n = len(y_train)
        y_calib = _check_y(y_calib)
        sample_weight, X_train, y_train = check_null_weight(
            sample_weight,
            X_train,
            y_train
        )
        y_train = cast(NDArray, y_train)

        # Initialization
        self.estimators_ = {}

        # Work
        self.y_calib_pred = np.full(
            shape=(3, len(y_calib)),
            fill_value=np.nan
        )
        name_estimator = estimator.__class__.__name__
        for i, alpha_ in enumerate(alpha):
            estimator_cloned = clone(estimator)
            params = {
                self.quantile_estimator_params[
                    name_estimator
                    ]["alpha_name"]: alpha_
            }
            estimator_cloned.set_params(**params)
            estimator_cloned_ = fit_estimator(
                estimator_cloned, X_train, y_train, sample_weight
            )
            self.estimators_[alpha_] = estimator_cloned_
            y_pred_calib = estimator_cloned_.predict(X_calib)
            check_nan_in_aposteriori_prediction(y_pred_calib)
            self.y_calib_pred[i] = y_pred_calib

        self.conformity_scores_ = np.full(
                shape=(3, len(y_calib)),
                fill_value=np.nan
            )
        self.conformity_scores_[0] = self.y_calib_pred[0]-y_calib
        self.conformity_scores_[1] = y_calib-self.y_calib_pred[1]
        self.conformity_scores_[2] = np.max(
                [
                    self.conformity_scores_[0],
                    self.conformity_scores_[1]
                ], axis=0
            )
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
        alpha = self._check_alpha(self.alpha)
        n = self.n
        q = (1-(self.alpha/2))*(1+(1/n))

        y_preds = np.full(
            shape=(3, X.shape[0]),
            fill_value=np.nan
        )
        for i, alpha_ in enumerate(alpha):
            y_preds[i] = self.estimators_[alpha_].predict(X)
        check_nan_in_aposteriori_prediction(y_preds)
        if symmetry:
            q = (1-(self.alpha))*(1+(1/n))
            quantile = np.full(
                2,
                np_quantile(
                    self.conformity_scores_[2], q, method="higher"
                )
            )
        else:
            quantile = np.array(
                [
                    np_quantile(
                        self.conformity_scores_[0], q, method="higher"
                    ),
                    np_quantile(
                        self.conformity_scores_[0], q, method="higher"
                    )
                ]
            )
        y_pred_low = y_preds[0] - quantile[0]
        y_pred_up = y_preds[1] + quantile[1]
        self._check_lower_upper_bounds(y_preds, y_pred_low, y_pred_up)
        return y_preds[2], np.stack([y_pred_low, y_pred_up], axis=1)

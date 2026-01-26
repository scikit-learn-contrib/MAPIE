from __future__ import annotations

import warnings
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.utils import check_valid_ltt_params_index

from .methods import ltt_procedure
from .risks import (
    BinaryClassificationRisk,
    abstention_rate,
    accuracy,
    false_positive_rate,
    precision,
    negative_predictive_value,
    positive_predictive_value,
    predicted_positive_fraction,
    recall,
)

Risk_str = Literal[
    "precision",
    "recall",
    "accuracy",
    "fpr",
    "predicted_positive_fraction",
    "positive_predictive_value",
    "negative_predictive_value",
    "abstention_rate",
]
Risk = Union[
    BinaryClassificationRisk,
    Risk_str,
    List[BinaryClassificationRisk],
    List[Risk_str],
    List[Union[BinaryClassificationRisk, Risk_str]],
]


class BinaryClassificationController:
    """
    Controls the risk or performance of a binary classifier.

    BinaryClassificationController finds the decision thresholds of a binary classifier
    that statistically guarantee a risk to be below a target level
    (the risk is "controlled").
    It can be used to control a performance metric as well, such as the precision.
    In that case, the thresholds guarantee that the performance is above a target level.

    Usage:

    1. Instantiate a BinaryClassificationController, providing the predict_proba method
       of your binary classifier
    2. Call the calibrate method to find the thresholds
    3. Use the predict method to predict using the best threshold

    Note: for a given model, calibration dataset, target level, and confidence level,
    there may not be any threshold controlling the risk.

    Parameters
    ----------
    predict_function : Callable[[ArrayLike], NDArray]
        predict_proba method of a fitted binary classifier.
        Its output signature must be of shape (len(X), 2).

        Or, in the general case of multi-dimensional parameters (thresholds),
        a function that takes (X, \*params) and outputs 0 or 1. This can be useful to e.g.,
        ensemble multiple binary classifiers with different thresholds for each classifier.
        In that case, `predict_params` must be provided.

    risk : Union[BinaryClassificationRisk, str, List[BinaryClassificationRisk, str]]
        The risk or performance metric to control.
        Valid options:

        - An existing risk defined in `mapie.risk_control` accessible through
          its string equivalent: "precision", "recall", "accuracy",
          "fpr" for false positive rate, or "predicted_positive_fraction".
        - A custom instance of BinaryClassificationRisk object

        Can be a list of risks in the case of multi risk control.

    target_level : Union[float, List[float]]
        The maximum risk level (or minimum performance level). Must be between 0 and 1.
        Can be a list of target levels in the case of multi risk control (length should
        match the length of the risks list).

    confidence_level : float, default=0.9
        The confidence level with which the risk (or performance) is controlled.
        Must be between 0 and 1. See the documentation for detailed explanations.

    best_predict_param_choice : Union["auto", BinaryClassificationRisk, str],
        default="auto"
        How to select the best threshold from the valid thresholds that control the risk
        (or performance). The BinaryClassificationController will try to minimize
        (or maximize) a secondary objective.
        Valid options:

        - "auto" (default). For mono risk defined in mapie.risk_control, an automatic choice is made.
          For multi risk, we use the first risk in the list.
        - An existing risk defined in `mapie.risk_control` accessible through

          its string equivalent: "precision", "recall", "accuracy",
          "fpr" for false positive rate, or "predicted_positive_fraction".
        - A custom instance of BinaryClassificationRisk object

    list_predict_params : NDArray, default=np.linspace(0, 0.99, 100)
        The set of parameters (noted λ in [1]) to consider for controlling the risk (or performance).
        When `predict_function` is a `predict_proba` method, the shape is (n_params,)
        and the parameter values are used to threshold the probabilities.
        When `predict_function` is a general function with multi-dimensional parameters (λ) that outputs 0 or 1,
        the shape is (n_params, params_dim).
        Note that performance is degraded when `len(predict_params)` is large as it is used by the Bonferroni correction [1].

    Attributes
    ----------
    valid_predict_params : NDArray
        The valid thresholds that control the risk (or performance).
        Use the calibrate method to compute these.

    best_predict_param : Optional[Union[float, Tuple[float, ...]]]
        The best threshold that control the risk (or performance). It is a tuple if multi-dimensional
        parameters are used.
        Use the calibrate method to compute it.

    p_values : NDArray
        P-values associated with each tested parameter in `list_predict_params`.
        In the multi-risk setting, the value corresponds to the maximum over the tested risks.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from mapie.risk_control import BinaryClassificationController, precision

    >>> X, y = make_classification(
    ...     n_features=2,
    ...     n_redundant=0,
    ...     n_informative=2,
    ...     n_clusters_per_class=1,
    ...     n_classes=2,
    ...     random_state=42,
    ...     class_sep=2.0
    ... )
    >>> X_train, X_temp, y_train, y_temp = train_test_split(
    ...     X, y, test_size=0.4, random_state=42
    ... )
    >>> X_calib, X_test, y_calib, y_test = train_test_split(
    ...     X_temp, y_temp, test_size=0.1, random_state=42
    ... )

    >>> clf = LogisticRegression().fit(X_train, y_train)

    >>> controller = BinaryClassificationController(
    ...     predict_function=clf.predict_proba,
    ...     risk=precision,
    ...     target_level=0.6
    ... )

    >>> predictions = controller.calibrate(X_calib, y_calib).predict(X_test)

    References
    ----------
    [1] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al.
    "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022)
    """

    _best_predict_param_choice_map = {
        precision: recall,
        recall: precision,
        accuracy: accuracy,
        false_positive_rate: recall,
    }

    risk_choice_map = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "fpr": false_positive_rate,
        "predicted_positive_fraction": predicted_positive_fraction,
        "positive_predictive_value": positive_predictive_value,
        "negative_predictive_value": negative_predictive_value,
        "abstention_rate": abstention_rate,
    }

    def __init__(
        self,
        predict_function: Callable[[ArrayLike], NDArray],
        risk: Risk,
        target_level: Union[float, List[float]],
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[
            Literal["auto"], Risk_str, BinaryClassificationRisk
        ] = "auto",
        list_predict_params: NDArray = np.linspace(0, 0.99, 100),
    ):
        self.is_multi_risk = self._check_if_multi_risk_control(risk, target_level)
        self._predict_function = predict_function
        risk_list = risk if isinstance(risk, list) else [risk]
        try:
            self._risk = [
                BinaryClassificationController.risk_choice_map[risk]
                if isinstance(risk, str)
                else risk
                for risk in risk_list
            ]
        except KeyError as e:
            raise ValueError(
                "When risk is provided as a string, it must be one of: "
                f"{list(BinaryClassificationController.risk_choice_map.keys())}"
            ) from e
        target_level_list = (
            target_level if isinstance(target_level, list) else [target_level]
        )
        self._alpha = self._convert_target_level_to_alpha(target_level_list)
        self._delta = 1 - confidence_level

        self._best_predict_param_choice = self._set_best_predict_param_choice(
            best_predict_param_choice
        )

        self._predict_params = list_predict_params
        self.is_multi_dimensional_param = self._check_if_multi_dimensional_param(
            self._predict_params
        )

        self.valid_predict_params: NDArray = np.array([])
        self.best_predict_param: Optional[Union[float, Tuple[float, ...]]] = None
        self.p_values: Optional[NDArray] = None

    # All subfunctions are unit-tested. To avoid having to write
    # tests just to make sure those subfunctions are called,
    # we don't include .calibrate in the coverage report
    def calibrate(  # pragma: no cover
        self, X_calibrate: ArrayLike, y_calibrate: ArrayLike
    ) -> BinaryClassificationController:
        """
        Calibrate the BinaryClassificationController.
        Sets attributes valid_predict_params and best_predict_param (if the risk
        or performance can be controlled at the target level).

        Parameters
        ----------
        X_calibrate : ArrayLike
            Features of the calibration set.

        y_calibrate : ArrayLike
            Binary labels of the calibration set.

        Returns
        -------
        BinaryClassificationController
            The calibrated controller instance.
        """
        y_calibrate_ = np.asarray(y_calibrate, dtype=int)

        predictions_per_param = self._get_predictions_per_param(
            X_calibrate, self._predict_params, is_calibration_step=True
        )

        risk_values, eff_sample_sizes = self._get_risk_values_and_eff_sample_sizes(
            y_calibrate_, predictions_per_param, self._risk
        )
        (valid_index, p_values) = ltt_procedure(
            risk_values,
            np.expand_dims(self._alpha, axis=1),
            self._delta,
            eff_sample_sizes,
            True,
        )
        valid_params_index = valid_index[0]

        self.valid_predict_params = self._predict_params[valid_params_index]

        check_valid_ltt_params_index(
            predict_params=self._predict_params, valid_index=self.valid_predict_params
        )

        if len(self.valid_predict_params) == 0:
            self.best_predict_param = None
        else:
            self._set_best_predict_param(
                y_calibrate_,
                predictions_per_param,
                valid_params_index,
            )

        self.p_values = p_values

        return self

    def predict(self, X_test: ArrayLike) -> NDArray:
        """
        Predict using predict_function at the best threshold.

        Parameters
        ----------
        X_test : ArrayLike
            Features

        Returns
        -------
        NDArray
            NDArray of shape (n_samples,)

        Raises
        ------
        ValueError
            If the method .calibrate was not called,
            or if no valid thresholds were found during calibration.
        """
        if self.best_predict_param is None:
            raise ValueError(
                "Cannot predict. "
                "Either you forgot to calibrate the controller first, "
                "or calibration was not successful."
            )
        return self._get_predictions_per_param(
            X_test,
            np.array([self.best_predict_param]),
        )[0]

    def _set_best_predict_param_choice(
        self,
        best_predict_param_choice: Union[
            Literal["auto"], Risk_str, BinaryClassificationRisk
        ] = "auto",
    ) -> BinaryClassificationRisk:
        if best_predict_param_choice == "auto":
            if self.is_multi_risk:
                # when multi risk, we minimize the first risk in the list
                return self._risk[0]
            else:
                try:
                    return self._best_predict_param_choice_map[self._risk[0]]
                except KeyError:
                    raise ValueError(
                        "When best_predict_param_choice is 'auto', "
                        "risk must be one of the risks defined in mapie.risk_control"
                        "(e.g. precision, accuracy, false_positive_rate)."
                    )
        if isinstance(best_predict_param_choice, str):
            return BinaryClassificationController.risk_choice_map[
                best_predict_param_choice
            ]
        if isinstance(best_predict_param_choice, BinaryClassificationRisk):
            return best_predict_param_choice

        raise TypeError(
            f"Got object of type {type(best_predict_param_choice)}. "
            "best_predict_param_choice must be either 'auto', "
            "a BinaryClassificationRisk instance, "
            "or a risk name (str) among those defined in mapie.risk_control "
            "(e.g. 'precision', 'accuracy', 'false_positive_rate')."
        )

    def _set_best_predict_param(
        self,
        y_calibrate_: NDArray,
        predictions_per_param: NDArray,
        valid_params_index: List[Any],
    ):
        secondary_risks_per_param, _ = self._get_risk_values_and_eff_sample_sizes(
            y_calibrate_,
            predictions_per_param[valid_params_index],
            [self._best_predict_param_choice],
        )

        self.best_predict_param = self.valid_predict_params[
            np.argmin(secondary_risks_per_param)
        ]
        if isinstance(self.best_predict_param, np.ndarray):
            self.best_predict_param = tuple(self.best_predict_param.tolist())

    @staticmethod
    def _get_risk_values_and_eff_sample_sizes(
        y_true: NDArray,
        predictions_per_param: NDArray,
        risks: List[BinaryClassificationRisk],
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the values of risks and effective sample sizes for multiple risks
        and for multiple parameter values.
        Returns arrays with shape (n_risks, n_params).
        """

        risks_values_and_eff_sizes = np.array(
            [
                [
                    risk.get_value_and_effective_sample_size(y_true, predictions)
                    for predictions in predictions_per_param
                ]
                for risk in risks
            ]
        )

        risk_values = risks_values_and_eff_sizes[:, :, 0]
        effective_sample_sizes = risks_values_and_eff_sizes[:, :, 1]

        return risk_values, effective_sample_sizes

    def _get_predictions_per_param(
        self, X: ArrayLike, params: NDArray, is_calibration_step=False
    ) -> NDArray:
        """Returns y_pred of shape (n_params, n_samples)"""
        n_params = len(params)
        n_samples = len(np.asarray(X))
        if self.is_multi_dimensional_param:
            y_pred: NDArray[np.float_] = np.empty((n_params, n_samples), dtype=float)
            for i in range(n_params):
                y_pred[i] = self._predict_function(X, *params[i])
            if is_calibration_step:
                self._check_predictions(y_pred)
        else:
            try:
                predictions_proba = self._predict_function(X)[:, 1]
            except TypeError as e:
                if "object is not callable" in str(e):
                    raise TypeError(
                        "Error when calling the predict_function. "
                        "Maybe you provided a binary classifier to the "
                        "predict_function parameter of the BinaryClassificationController. "
                        "You should provide your classifier's predict_proba method instead."
                    ) from e
                else:
                    raise
            except IndexError as e:
                if "array is 1-dimensional, but 2 were indexed" in str(e):
                    raise IndexError(
                        "Error when calling the predict_function. "
                        "Maybe the predict function you provided returns only the "
                        "probability of the positive class. "
                        "You should provide a predict function that returns the "
                        "probabilities of both classes, like scikit-learn estimators."
                    ) from e
                else:
                    raise
            if is_calibration_step:
                self._check_predictions(predictions_proba)
            y_pred = (predictions_proba[:, np.newaxis] >= params).T.astype(int)
        return y_pred

    def _convert_target_level_to_alpha(self, target_level: List[float]) -> NDArray:
        alpha = []
        for risk, target in zip(self._risk, target_level):
            if risk.higher_is_better:
                alpha.append(1 - target)
            else:
                alpha.append(target)
        return np.array(alpha)

    @staticmethod
    def _check_if_multi_risk_control(
        risk: Risk,
        target_level: Union[float, List[float]],
    ) -> bool:
        """
        Check if we are in a multi risk setting and if inputs types are correct.
        """
        if (
            isinstance(risk, list)
            and isinstance(target_level, list)
            and len(risk) == len(target_level)
            and len(risk) > 0
        ):
            if len(risk) == 1:
                return False
            else:
                return True
        elif (
            isinstance(risk, BinaryClassificationRisk) or isinstance(risk, str)
        ) and isinstance(target_level, float):
            return False
        else:
            raise ValueError(
                "If you provide a list of risks, you must provide "
                "a list of target levels of the same length and vice versa. "
                "If you provide a single BinaryClassificationRisk risk, "
                "you must provide a single float target level."
            )

    @staticmethod
    def _check_if_multi_dimensional_param(
        predict_params: NDArray,
    ) -> bool:
        """
        Check if the the parameters (the λ) are multi-dimensional.
        """
        if predict_params.ndim == 1:
            return False
        elif predict_params.ndim == 2:
            return True
        else:
            raise ValueError(
                "predict_params must be a 1D array of shape (n_params,) for one-dimensional parameters, "
                "or a 2D array of shape (n_params, params_dim) for multi-dimensional parameters "
                "(params_dim=1 is allowed for the case when a one-dimensional parameter is not used as a threshold)."
            )

    def _check_predictions(self, predictions_per_param: NDArray) -> None:
        """
        Checks if predictions are probabilities for one-dimensional parameters,
        or binary predictions for multi-dimensional parameters.
        """
        if (
            not self.is_multi_dimensional_param
            and np.logical_or(
                predictions_per_param == 0, predictions_per_param == 1
            ).all()
        ):
            warnings.warn(
                "All predictions are either 0 or 1 while the parameters are one-dimensional. "
                "Make sure that the provided predict_function is a "
                "predict_proba method or a function that outputs probabilities.",
            )

        if (
            self.is_multi_dimensional_param
            and not np.logical_or.reduce(
                (
                    predictions_per_param == 0,
                    predictions_per_param == 1,
                    np.isnan(predictions_per_param),
                )
            ).all()
        ):
            raise ValueError(
                "The provided predict_function with multi-dimensional "
                "parameters must return binary predictions (0, 1, np.nan)."
            )

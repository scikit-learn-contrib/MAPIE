from __future__ import annotations

import warnings
from typing import Callable, List, Literal, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.risk_control.fwer_control import (
    FWER_METHODS,
    FWERProcedure,
)

from ._base_controller import _BaseLTTController
from .risks import (
    BinaryRisk,
    BinaryRiskLike,
    BinaryRiskNames,
    _best_predict_param_choice_map,
    binary_risk_choice_map,
)


class BinaryClassificationController(_BaseLTTController):
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
        a function that takes (X, \\*params) and outputs 0 or 1. This can be useful to e.g.,
        ensemble multiple binary classifiers with different thresholds for each classifier.
        In that case, `predict_params` must be provided.

    risk : Union[BinaryRisk, str, List[BinaryRisk, str]]
        The risk or performance metric to control.
        Valid options:

        - An existing risk defined in `mapie.risk_control` accessible through
          its string equivalent: "precision", "recall", "accuracy",
          "fpr" for false positive rate, or "predicted_positive_fraction".
        - A custom instance of BinaryRisk object

        Can be a list of risks in the case of multi risk control.

    target_level : Union[float, List[float]]
        The maximum risk level (or minimum performance level). Must be between 0 and 1.
        Can be a list of target levels in the case of multi risk control (length should
        match the length of the risks list).

    confidence_level : float, default=0.9
        The confidence level with which the risk (or performance) is controlled.
        Must be between 0 and 1. See the documentation for detailed explanations.

    best_predict_param_choice : Union["auto", BinaryRisk, str],
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
        - A custom instance of BinaryRisk object

    list_predict_params : NDArray, default=np.linspace(0, 0.99, 100)
        The set of parameters (noted λ in [1]) to consider for controlling the risk (or performance).
        When `predict_function` is a `predict_proba` method, the shape is (n_params,)
        and the parameter values are used to threshold the probabilities.
        When `predict_function` is a general function with multi-dimensional parameters (λ) that outputs 0 or 1,
        the shape is (n_params, params_dim).
        Note that performance is degraded when `len(predict_params)` is large as it is used by the Bonferroni correction [1].

    fwer_method : {"bonferroni", "bonferroni_holm", "fixed_sequence", "split_fixed_sequence"} or FWERProcedure instance, default="bonferroni_holm"
        Method used to control the family-wise error rate (FWER).

        Supported methods:
        - `"bonferroni"` : Classical Bonferroni correction.
        It is valid in all settings but can be conservative, especially when the number of tested parameters is large.
        - `"fixed_sequence"` : Fixed Sequence Testing (FST) with a single start.
        However, users can use multi-start by instantiating `FWERFixedSequenceTesting` with any desired number
        of starts and passing the instance to control_fwer.
        - `"bonferroni_holm"` : Sequential Graphical Testing corresponding
        to the Bonferroni–Holm procedure. This is the default method and is suitable for general settings.
        - `"split_fixed_sequence"` : Split Fixed Sequence Testing (SFST).

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

    def __init__(
        self,
        predict_function: Callable[[ArrayLike], NDArray],
        risk: BinaryRiskLike,
        target_level: Union[float, List[float]],
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[
            Literal["auto"], BinaryRiskNames, BinaryRisk
        ] = "auto",
        list_predict_params: NDArray = np.linspace(0, 0.99, 100),
        fwer_method: Union[FWER_METHODS, FWERProcedure] = "bonferroni_holm",
    ):
        self.is_multi_risk = self._check_if_multi_risk_control(risk, target_level)
        risk_list = risk if isinstance(risk, list) else [risk]
        try:
            risks = [
                binary_risk_choice_map[risk] if isinstance(risk, str) else risk
                for risk in risk_list
            ]
        except KeyError as e:
            raise ValueError(
                "When risk is provided as a string, it must be one of: "
                f"{list(binary_risk_choice_map.keys())}"
            ) from e
        target_level_list = (
            target_level if isinstance(target_level, list) else [target_level]
        )
        self._init_ltt(
            predict_function,
            risks,
            target_level_list,
            confidence_level,
            best_predict_param_choice,
            list_predict_params,
            fwer_method,
        )

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

        Notes
        -----
        When using `fwer_method="split_fixed_sequence"`,
        the learning step must be performed separately on independent data:

        1. bcc.learn_fixed_sequence_order(X_learn, y_learn)
        2. bcc.calibrate(X_calibrate, y_calibrate)

        Using the same data for both steps would invalidate guarantees.
        """
        self._calibrate(X_calibrate, y_calibrate)
        return self

    def _set_best_predict_param_choice(
        self,
        best_predict_param_choice: Union[
            Literal["auto"], BinaryRiskNames, BinaryRisk
        ] = "auto",
    ) -> BinaryRisk:
        if best_predict_param_choice == "auto":
            first_risk = cast(BinaryRisk, self._risk[0])
            if self.is_multi_risk:
                # when multi risk, we minimize the first risk in the list
                return first_risk
            else:
                try:
                    return _best_predict_param_choice_map[first_risk]
                except KeyError:
                    raise ValueError(
                        "When best_predict_param_choice is 'auto', "
                        "risk must be one of the risks defined in mapie.risk_control"
                        "(e.g. precision, accuracy, false_positive_rate)."
                    )
        if isinstance(best_predict_param_choice, str):
            return binary_risk_choice_map[best_predict_param_choice]
        if isinstance(best_predict_param_choice, BinaryRisk):
            return best_predict_param_choice

        raise TypeError(
            f"Got object of type {type(best_predict_param_choice)}. "
            "best_predict_param_choice must be either 'auto', "
            "a BinaryRisk instance, "
            "or a risk name (str) among those defined in mapie.risk_control "
            "(e.g. 'precision', 'accuracy', 'false_positive_rate')."
        )

    def _get_predictions_per_param_mono_param(
        self, X, params, is_calibration_step=False
    ):
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

    def _check_predictions(self, predictions_per_param: NDArray) -> None:
        """
        Checks if predictions are probabilities for one-dimensional parameters,
        or binary predictions for multi-dimensional parameters.
        """
        if self.is_multi_dimensional_param:
            if not np.logical_or.reduce(
                (
                    predictions_per_param == 0,
                    predictions_per_param == 1,
                    np.isnan(predictions_per_param),
                )
            ).all():
                raise ValueError(
                    "The provided predict_function with multi-dimensional "
                    "parameters must return binary predictions (0, 1, np.nan)."
                )
        elif np.logical_or(
            predictions_per_param == 0, predictions_per_param == 1
        ).all():
            warnings.warn(
                "All predictions are either 0 or 1 while the parameters are one-dimensional. "
                "Make sure that the provided predict_function is a "
                "predict_proba method or a function that outputs probabilities.",
            )

    @staticmethod
    def _check_if_multi_risk_control(
        risk: BinaryRiskLike,
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
            return len(risk) > 1
        elif (isinstance(risk, BinaryRisk) or isinstance(risk, str)) and isinstance(
            target_level, float
        ):
            return False
        else:
            raise ValueError(
                "If you provide a list of risks, you must provide "
                "a list of target levels of the same length and vice versa. "
                "If you provide a single BinaryRisk risk, "
                "you must provide a single float target level."
            )

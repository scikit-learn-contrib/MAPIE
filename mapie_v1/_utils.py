import copy
from typing import Union, Tuple, cast, Optional, Iterable
from collections.abc import Iterable as IterableType

from mapie._typing import ArrayLike, NDArray
from sklearn.model_selection import BaseCrossValidator
from decimal import Decimal


def transform_confidence_level_to_alpha_list(
    confidence_level: Union[float, Iterable[float]]
) -> Iterable[float]:
    if isinstance(confidence_level, IterableType):
        confidence_levels = confidence_level
    else:
        confidence_levels = [confidence_level]
    return [
        transform_confidence_level_to_alpha(confidence_level)
        for confidence_level in confidence_levels
    ]


def transform_confidence_level_to_alpha(
    confidence_level: float,
) -> float:
    # Using decimals to avoid weird-looking float approximations
    # when computing alpha = 1 - confidence_level
    # Such approximations arise even with simple confidence levels like 0.9
    confidence_level_decimal = Decimal(str(confidence_level))
    alpha_decimal = Decimal("1") - confidence_level_decimal
    return float(alpha_decimal)


def check_if_param_in_allowed_values(
    param: str, param_name: str, allowed_values: list
) -> None:
    if param not in allowed_values:
        raise ValueError(
            f"'{param}' option not valid for parameter '{param_name}'"
            f"Available options are: {allowed_values}"
        )


def check_cv_not_string(cv: Union[int, str, BaseCrossValidator]) -> None:
    if isinstance(cv, str):
        raise ValueError(
            "'cv' string options not available in MAPIE >= v1"
        )


def cast_point_predictions_to_ndarray(
    point_predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> NDArray:
    return cast(NDArray, point_predictions)


def cast_predictions_to_ndarray_tuple(
    predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> Tuple[NDArray, NDArray]:
    return cast(Tuple[NDArray, NDArray], predictions)


def prepare_params(params: Union[dict, None]) -> dict:
    return copy.deepcopy(params) if params else {}


def prepare_fit_params_and_sample_weight(
    fit_params: Union[dict, None]
) -> Tuple[dict, Optional[ArrayLike]]:
    fit_params_ = prepare_params(fit_params)
    sample_weight = fit_params_.pop("sample_weight", None)
    return fit_params_, sample_weight


def raise_error_if_previous_method_not_called(
    current_method_name: str,
    previous_method_name: str,
    was_previous_method_called: bool,
) -> None:
    if not was_previous_method_called:
        raise ValueError(
            f"Incorrect method order: call {previous_method_name} "
            f"before calling {current_method_name}."
        )


def raise_error_if_method_already_called(
    method_name: str,
    was_method_called: bool,
) -> None:
    if was_method_called:
        raise ValueError(
            f"{method_name} method already called. "
            f"MAPIE does not currently support calling {method_name} several times."
        )


def raise_error_if_fit_called_in_prefit_mode(
    is_mode_prefit: bool,
) -> None:
    if is_mode_prefit:
        raise ValueError(
            "The fit method must be skipped when the prefit parameter is set to True. "
            "Use the conformalize method directly after instanciation."
        )

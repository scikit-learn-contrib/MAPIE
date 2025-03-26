import copy
from typing import Union, Tuple, cast, Optional, Iterable
from collections.abc import Iterable as IterableType

from mapie._typing import ArrayLike, NDArray
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import train_test_split
from decimal import Decimal
from math import isclose


def transform_confidence_level_to_alpha(
    confidence_level: float,
) -> float:
    # Using decimals to avoid weird-looking float approximations
    # when computing alpha = 1 - confidence_level
    # Such approximations arise even with simple confidence levels like 0.9
    confidence_level_decimal = Decimal(str(confidence_level))
    alpha_decimal = Decimal("1") - confidence_level_decimal
    return float(alpha_decimal)


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


# Could be replaced by using sklearn _validate_params (ideally wrapping it)
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
            "'cv' string options not available in MAPIE >= v1.0.0"
        )


def cast_point_predictions_to_ndarray(
    point_predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> NDArray:
    if isinstance(point_predictions, tuple):
        raise TypeError(
            "Developer error: use this function to cast point predictions only, "
            "not points + intervals."
        )
    return cast(NDArray, point_predictions)


def cast_predictions_to_ndarray_tuple(
    predictions: Union[NDArray, Tuple[NDArray, NDArray]]
) -> Tuple[NDArray, NDArray]:
    if not isinstance(predictions, tuple):
        raise TypeError(
            "Developer error: use this function to cast predictions containing points "
            "and intervals, not points only."
        )
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


def train_conformalize_test_split(
    X: NDArray,
    y: NDArray,
    train_size: Union[float, int],
    conformalize_size: Union[float, int],
    test_size: Union[float, int],
    random_state: int = None,
    shuffle: bool = True,
    stratify: list = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Split arrays or matrices into train, conformalize and test subsets.

    Quick utility similar to sklearn.model_selection.train_test_split
    for splitting data into 3 sets in one line.

    We advise to give the major part of the data points to the train set
    and at least 200 data points to the conformalize set.

    Parameters
    ----------
    X : indexable with same type and length / shape[0] than "y"
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    y : indexable with same type and length / shape[0] than "X"
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    train_size : float or int
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples.

    conformalize_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    test_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    X_train, X_conformalize, X_test, y_train, y_conformalize, y_test :
        6 array-like splits of inputs.
        output types are the same as the input types.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> from mapie_v1.utils import train_conformalize_test_split
    >>> X, y = np.arange(10).reshape((5, 2)), range(5)
    >>> X
    array([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
    >>> list(y)
    [0, 1, 2, 3, 4]
    >>> (
    ...     X_train, X_conformalize, X_test,
    ...     y_train, y_conformalize, y_test
    ... ) = train_conformalize_test_split(
    ...     X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
    ... )
    >>> X_train
    array([[8, 9],
        [0, 1],
        [6, 7]])
    >>> X_conformalize
    array([[2, 3]])
    >>> X_test
    array([[4, 5]])
    >>> y_train
    [4, 0, 3]
    >>> y_conformalize
    [1]
    >>> y_test
    [2]
    """

    _check_train_conf_test_proportions(
        train_size, conformalize_size, test_size, len(X)
    )

    X_train, X_conformalize_test, y_train, y_conformalize_test = train_test_split(
        X, y,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle, stratify=stratify
    )

    if isinstance(train_size, float):
        test_size_after_split = test_size/(1-train_size)
    else:
        test_size_after_split = test_size

    X_conformalize, X_test, y_conformalize, y_test = train_test_split(
        X_conformalize_test, y_conformalize_test,
        test_size=test_size_after_split,
        random_state=random_state,
        shuffle=shuffle, stratify=stratify
    )

    return X_train, X_conformalize, X_test, y_train, y_conformalize, y_test


def _check_train_conf_test_proportions(
    train_size: Union[float, int],
    conformalize_size: Union[float, int],
    test_size: Union[float, int],
    dataset_size: int,
) -> Tuple[float, float]:

    count_input_proportions = sum([test_size, train_size, conformalize_size])

    if isinstance(train_size, float) and \
            isinstance(conformalize_size, float) and \
            isinstance(test_size, float):
        if not isclose(1, count_input_proportions):
            raise ValueError("When using floats, train_size + \
                             conformalize_size + test_size must be equal to 1.")

    elif isinstance(train_size, int) and \
            isinstance(conformalize_size, int) and \
            isinstance(test_size, int):
        if count_input_proportions != dataset_size:
            raise ValueError("When using integers, train_size + \
                             conformalize_size + test_size must be equal \
                             to the size of the input data.")

    else:
        raise TypeError(
            "train_size, conformalize_size and test_size \
                should be either all int or all float.")

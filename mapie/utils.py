from typing import Union, Tuple

from sklearn.utils.validation import _check_sample_weight
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator

from ._typing import ArrayLike


def get_n_features_in(
    cv: Union[str, BaseCrossValidator],
    estimator: RegressorMixin,
    X: ArrayLike
) -> int:
    """
    Get the expected number of training features.
    In general it is simply the number of columns of the data.
    If ``cv=="prefit"`` however, it can be deduced from the estimator's ``n_features_in_`` attribute.

    Parameters
    ----------
    cv : Union[str, BaseCrossValidator]
        Cross-validator.
    estimator : RegressorMixin
        Backend estimator of MAPE.
    X : ArrayLike of shape (n_samples, n_features)
        Data passed into the ``fit`` method.

    Returns
    -------
    int
        Expected number of training features.
    """
    n_features_in = X.shape[1]
    if cv == "prefit" and hasattr(estimator, "n_features_in_"):
        n_features_in = estimator.n_features_in_
    return n_features_in


def check_null_weight(
    sample_weight: ArrayLike,
    X: ArrayLike,
    y: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Check sample weights and remove samples with null sample weights.

    Parameters
    ----------
    sample_weight : ArrayLike
        Sample weights.
    X : ArrayLike of shape (n_samples, n_features)
        Training samples.
    y : ArrayLike of shape (n_samples,)
        Training labels.

    Returns
    -------
    sample_weight : ArrayLike of shape (n_samples,)
        Non-null sample weights.
    X : ArrayLike of shape (n_samples, n_features)
        Training samples with non-null weights.
    y : ArrayLike of shape (n_samples,)
        Training labels with non-null weights.
    """
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)
        non_null_weight = sample_weight != 0
        X, y = X[non_null_weight, :], y[non_null_weight]
        sample_weight = sample_weight[non_null_weight]
    return sample_weight, X, y


def fit_estimator(
    estimator: RegressorMixin,
    X: ArrayLike,
    y: ArrayLike,
    supports_sw: bool,
    sample_weight: ArrayLike
) -> RegressorMixin:
    """
    Fit an estimator on training data by distinguishing two cases:
    - the estimator supports sample weights and sample weights are provided.
    - the estimator does not support samples weights or samples weights are not provided

    Parameters
    ----------
    estimator : RegressorMixin
        Estimator to train.

    X : ArrayLike of shape (n_samples, n_features)
        Input data.

    y : ArrayLike of shape (n_samples,)
        Input labels.

    supports_sw : bool
        Whether or not estimator supports sample weights.

    sample_weight : ArrayLike of shape (n_samples,)
        Sample weights. If None, then samples are equally weighted. By default None.

    Returns
    -------
    RegressorMixin
        Fitted estimator.
    """
    if sample_weight is not None and supports_sw:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator

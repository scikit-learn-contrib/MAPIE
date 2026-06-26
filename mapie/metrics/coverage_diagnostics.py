import importlib
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.utils import column_or_1d

from mapie.utils import (
    _check_array_inf,
    _check_array_nan,
    _check_array_shape_classification,
    _check_array_shape_regression,
    _check_arrays_length,
)


CoverageMetric = Literal[
    "covgap",
    "ert",
    "fsc",
    "hsic",
    "pearson",
    "ssc",
    "wsc",
]
CoverageTask = Literal["classification", "regression"]

_METRIC_CLASSES = {
    "covgap": "CovGap",
    "ert": "ERT",
    "fsc": "FSC",
    "hsic": "HSIC",
    "pearson": "PearsonCorrelation",
    "ssc": "SSC",
    "wsc": "WSC",
}


class _DefaultERTClassifier:
    def __init__(
        self,
        max_iter: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.classes_: NDArray = np.array([0, 1])
        self.constant_proba_: Optional[float] = None
        self.estimator_: Optional[LogisticRegression] = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        **fit_kwargs: Any,
    ) -> "_DefaultERTClassifier":
        fit_kwargs.pop("X_val", None)
        fit_kwargs.pop("y_val", None)
        X_array = self._to_numpy(X)
        y_array = column_or_1d(self._to_numpy(y)).astype(int)
        classes = np.unique(y_array)
        self.classes_ = np.array([0, 1])

        if classes.shape[0] == 1:
            self.constant_proba_ = float(classes[0])
            self.estimator_ = None
            return self

        self.constant_proba_ = None
        self.estimator_ = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.estimator_.fit(X_array, y_array, **fit_kwargs)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray:
        X_array = self._to_numpy(X)
        if self.constant_proba_ is not None:
            proba = np.full(X_array.shape[0], self.constant_proba_)
            return cast(NDArray, np.column_stack((1 - proba, proba)))
        if self.estimator_ is None:  # pragma: no cover
            raise ValueError("The classifier must be fitted before prediction.")
        return cast(NDArray, self.estimator_.predict_proba(X_array))

    def predict(self, X: ArrayLike) -> NDArray:
        return cast(NDArray, self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @staticmethod
    def _to_numpy(array: ArrayLike) -> NDArray:
        if hasattr(array, "to_numpy"):
            return cast(NDArray, array.to_numpy())
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        return cast(NDArray, np.asarray(array))


def covmetrics_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: CoverageMetric,
    *,
    task: CoverageTask = "regression",
    alpha: Optional[Union[float, ArrayLike]] = None,
    groups: Optional[ArrayLike] = None,
    X: Optional[ArrayLike] = None,
    delta: Optional[float] = None,
    weighted: bool = False,
    number_max_groups: int = 10,
    M: int = 1000,
    random_state: int = 42,
    max_number_samples: int = 5000,
    sigma_x: float = 1,
    sigma_y: float = 1,
    model_cls: Optional[type] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    loss: Optional[Callable[..., Any]] = None,
    n_splits: int = 5,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> NDArray:
    """
    Evaluate a covmetrics diagnostic on MAPIE prediction outputs.

    The function converts MAPIE regression intervals or classification prediction
    sets into the binary coverage indicators expected by covmetrics, then
    evaluates one score per confidence level.

    Parameters
    ----------
    y_true : ArrayLike of shape (n_samples,)
        True labels or regression targets.
    y_pred : ArrayLike
        For regression, prediction intervals of shape
        ``(n_samples, 2, n_confidence_level)`` or ``(n_samples, 2)``.
        For classification, prediction sets of shape
        ``(n_samples, n_classes, n_confidence_level)`` or
        ``(n_samples, n_classes)``.
    metric : {"covgap", "ert", "fsc", "hsic", "pearson", "ssc", "wsc"}
        Covmetrics diagnostic to evaluate.
    task : {"classification", "regression"}, default="regression"
        Type of MAPIE prediction output passed to ``y_pred``.
    alpha : float or ArrayLike, optional
        Target miscoverage. Required by ``"covgap"``, ``"ert"``, and ``"ssc"``.
        If several confidence levels are evaluated, this can be a scalar or one
        value per confidence level. For ``"ert"``, this can also be one value
        per sample or an array of shape ``(n_samples, n_confidence_level)``.
    groups : ArrayLike of shape (n_samples,), optional
        Integer group labels. Required by ``"covgap"`` and ``"fsc"``.
    X : ArrayLike of shape (n_samples, n_features), optional
        Features. Required by ``"ert"`` and ``"wsc"``.
    delta : float, optional
        Slab size parameter. Required by ``"wsc"``.
    weighted : bool, default=False
        Whether ``"covgap"`` and ``"ssc"`` use weighted group gaps.
    number_max_groups : int, default=10
        Maximum number of groups used by covmetrics ``"ssc"``.
    M : int, default=1000
        Number of random directions used by covmetrics ``"wsc"``.
    random_state : int, default=42
        Random state passed to covmetrics ``"ert"`` and ``"wsc"``.
    max_number_samples : int, default=5000
        Maximum sample count used by covmetrics ``"hsic"``.
    sigma_x : float, default=1
        First kernel bandwidth used by covmetrics ``"hsic"``.
    sigma_y : float, default=1
        Second kernel bandwidth used by covmetrics ``"hsic"``.
    model_cls : type, optional
        Classifier class used by covmetrics ``"ert"``. If omitted, MAPIE uses
        a logistic-regression classifier that only depends on scikit-learn.
    model_kwargs : dict, optional
        Keyword arguments used to instantiate ``model_cls`` for ``"ert"``.
    loss : callable, optional
        Loss function passed to covmetrics ``"ert"``.
    n_splits : int, default=5
        Number of folds used by covmetrics ``"ert"``.
    fit_kwargs : dict, optional
        Keyword arguments passed to the ``"ert"`` classifier fit method.

    Returns
    -------
    NDArray of shape (n_confidence_level,)
        Diagnostic score for each confidence level.

    Examples
    --------
    This example requires the optional ``conditional`` dependencies.

    >>> import numpy as np
    >>> from mapie.metrics.coverage_diagnostics import covmetrics_score
    >>> y_true = np.array([0.5, 1.5, 2.0, 7.0])
    >>> y_intervals = np.array([[0, 1], [1, 2], [1, 3], [4, 6]])
    >>> covmetrics_score(y_true, y_intervals, "ssc", alpha=0.25)  # doctest: +SKIP
    array([0.25])
    """
    coverage, sizes = _prediction_outputs_to_coverage_and_sizes(
        y_true,
        y_pred,
        task,
    )
    n_confidence_levels = coverage.shape[1]

    scores = [
        _evaluate_metric_for_confidence_level(
            metric=metric,
            coverage=coverage[:, confidence_level],
            sizes=sizes[:, confidence_level],
            confidence_level=confidence_level,
            n_confidence_levels=n_confidence_levels,
            alpha=alpha,
            groups=groups,
            X=X,
            delta=delta,
            weighted=weighted,
            number_max_groups=number_max_groups,
            M=M,
            random_state=random_state,
            max_number_samples=max_number_samples,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            loss=loss,
            n_splits=n_splits,
            fit_kwargs=fit_kwargs,
        )
        for confidence_level in range(n_confidence_levels)
    ]
    return cast(NDArray, np.asarray(scores, dtype=float))


def _prediction_outputs_to_coverage_and_sizes(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    task: CoverageTask,
) -> Tuple[NDArray, NDArray]:
    if task == "regression":
        return _regression_coverage_and_sizes(y_true, y_pred)
    if task == "classification":
        return _classification_coverage_and_sizes(y_true, y_pred)
    raise ValueError("task must be 'regression' or 'classification'.")


def _regression_coverage_and_sizes(
    y_true: ArrayLike,
    y_intervals: ArrayLike,
) -> Tuple[NDArray, NDArray]:
    y_true_array = np.asarray(y_true)
    y_intervals_array = np.asarray(y_intervals, dtype=float)

    _check_arrays_length(y_true_array, y_intervals_array)
    _check_array_nan(y_true_array)
    _check_array_inf(y_true_array)
    _check_array_nan(y_intervals_array)
    _check_array_inf(y_intervals_array)

    y_intervals_array = _check_array_shape_regression(
        y_true_array,
        y_intervals_array,
    )
    if y_true_array.ndim != 2:
        y_true_array = np.expand_dims(column_or_1d(y_true_array), axis=1)

    coverage = np.logical_and(
        np.less_equal(y_intervals_array[:, 0, :], y_true_array),
        np.greater_equal(y_intervals_array[:, 1, :], y_true_array),
    )
    sizes = np.abs(y_intervals_array[:, 1, :] - y_intervals_array[:, 0, :])
    return coverage.astype(int), cast(NDArray, sizes)


def _classification_coverage_and_sizes(
    y_true: ArrayLike,
    y_pred_set: ArrayLike,
) -> Tuple[NDArray, NDArray]:
    y_true_array = np.asarray(y_true)
    y_pred_set_array = np.asarray(y_pred_set, dtype=bool)

    _check_arrays_length(y_true_array, y_pred_set_array)
    _check_array_nan(y_true_array)
    _check_array_inf(y_true_array)
    _check_array_nan(y_pred_set_array)
    _check_array_inf(y_pred_set_array)

    y_pred_set_array = _check_array_shape_classification(
        y_true_array,
        y_pred_set_array,
    )
    if y_true_array.ndim != 2:
        y_true_array = np.expand_dims(column_or_1d(y_true_array), axis=1)

    y_true_array = np.expand_dims(y_true_array, axis=1)
    coverage = np.take_along_axis(
        y_pred_set_array,
        y_true_array,
        axis=1,
    )[:, 0, :]
    sizes = np.sum(y_pred_set_array, axis=1)
    return coverage.astype(int), cast(NDArray, sizes)


def _evaluate_metric_for_confidence_level(
    *,
    metric: CoverageMetric,
    coverage: NDArray,
    sizes: NDArray,
    confidence_level: int,
    n_confidence_levels: int,
    alpha: Optional[Union[float, ArrayLike]],
    groups: Optional[ArrayLike],
    X: Optional[ArrayLike],
    delta: Optional[float],
    weighted: bool,
    number_max_groups: int,
    M: int,
    random_state: int,
    max_number_samples: int,
    sigma_x: float,
    sigma_y: float,
    model_cls: Optional[type],
    model_kwargs: Optional[Dict[str, Any]],
    loss: Optional[Callable[..., Any]],
    n_splits: int,
    fit_kwargs: Optional[Dict[str, Any]],
) -> float:
    metric_class = _get_covmetric_class(metric)
    coverage = column_or_1d(coverage).astype(int)
    sizes = column_or_1d(sizes).astype(float)

    if metric in ("covgap", "ert", "ssc"):
        alpha_value = _alpha_for_confidence_level(
            alpha,
            confidence_level,
            n_confidence_levels,
            len(coverage),
            allow_per_sample=metric == "ert",
        )

    if metric in ("covgap", "fsc"):
        groups_array = _required_1d_array(groups, "groups")
        if metric == "covgap":
            return float(
                metric_class(alpha_value).evaluate(
                    groups_array,
                    coverage,
                    alpha=alpha_value,
                    weighted=weighted,
                )
            )
        return float(metric_class().evaluate(groups_array, coverage))

    if metric == "ssc":
        return float(
            metric_class(alpha_value).evaluate(
                sizes,
                coverage,
                alpha=alpha_value,
                number_max_groups=number_max_groups,
                weighted=weighted,
            )
        )

    if metric == "wsc":
        if delta is None:
            raise ValueError("delta is required when metric='wsc'.")
        X_array = _required_2d_array(X, "X")
        return float(
            metric_class(delta).evaluate(
                X_array,
                coverage,
                delta=delta,
                M=M,
                seed=random_state,
            )
        )

    if metric == "pearson":
        return float(metric_class().evaluate(sizes, coverage))

    if metric == "hsic":
        return float(
            metric_class(sigma_x=sigma_x, sigma_y=sigma_y).evaluate(
                sizes,
                coverage,
                max_number_samples=max_number_samples,
            )
        )

    if metric == "ert":
        X_array = _required_2d_array(X, "X")
        ert_model_cls = model_cls or _DefaultERTClassifier
        estimator = metric_class(model_cls=ert_model_cls, **(model_kwargs or {}))
        return float(
            estimator.evaluate(
                X_array,
                coverage,
                alpha_value,
                n_splits=n_splits,
                random_state=random_state,
                loss=loss,
                **(fit_kwargs or {}),
            )
        )

    raise ValueError(  # pragma: no cover
        "metric must be one of 'covgap', 'ert', 'fsc', 'hsic', "
        "'pearson', 'ssc', or 'wsc'."
    )


def _alpha_for_confidence_level(
    alpha: Optional[Union[float, ArrayLike]],
    confidence_level: int,
    n_confidence_levels: int,
    n_samples: int,
    *,
    allow_per_sample: bool,
) -> Union[float, NDArray]:
    if alpha is None:
        raise ValueError("alpha is required for this metric.")

    if np.isscalar(alpha):
        return float(cast(float, alpha))

    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.ndim == 1:
        if alpha_array.shape[0] == n_confidence_levels:
            return float(alpha_array[confidence_level])
        if allow_per_sample and alpha_array.shape[0] == n_samples:
            return cast(NDArray, alpha_array)
    if allow_per_sample and alpha_array.shape == (n_samples, n_confidence_levels):
        return cast(NDArray, alpha_array[:, confidence_level])

    raise ValueError(
        "alpha must be a scalar, have one value per confidence level, "
        "or, for metric='ert', have shape "
        "(n_samples, n_confidence_level)."
    )


def _required_1d_array(value: Optional[ArrayLike], name: str) -> NDArray:
    if value is None:
        raise ValueError(f"{name} is required for this metric.")
    return cast(NDArray, column_or_1d(np.asarray(value)))


def _required_2d_array(value: Optional[ArrayLike], name: str) -> NDArray:
    if value is None:
        raise ValueError(f"{name} is required for this metric.")
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return cast(NDArray, array)


def _get_covmetric_class(metric: CoverageMetric) -> type:
    try:
        class_name = _METRIC_CLASSES[metric]
    except KeyError as exc:
        raise ValueError(
            "metric must be one of 'covgap', 'ert', 'fsc', 'hsic', "
            "'pearson', 'ssc', or 'wsc'."
        ) from exc
    return cast(type, getattr(_load_covmetrics(), class_name))


def _load_covmetrics() -> Any:
    try:
        return importlib.import_module("covmetrics")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "MAPIE's covmetrics wrapper requires the optional 'covmetrics' "
            "dependency. Install MAPIE with the 'conditional' extra to use "
            "this wrapper."
        ) from exc

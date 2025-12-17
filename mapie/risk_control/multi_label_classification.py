from __future__ import annotations

import warnings
from itertools import chain
from typing import Callable, Iterable, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_y, _num_samples, indexable

from mapie.utils import (
    _check_alpha,
    _check_n_jobs,
    _check_verbose,
    check_is_fitted,
    check_valid_ltt_params_index,
)

from .methods import (
    find_best_predict_param,
    find_precision_best_predict_param,
    get_r_hat_plus,
    ltt_procedure,
)
from .risks import precision, recall


class MultiLabelClassificationController:
    """
    Prediction sets for multilabel-classification.

    This class implements two conformal prediction methods for
    estimating prediction sets for multilabel-classification.
    It guarantees (under the hypothesis of exchangeability) that
    a risk is at least 1 - alpha (alpha is a user-specified parameter).
    For now, we consider the recall as risk.

    Parameters
    ----------
    predict_function : Callable[[ArrayLike], Union[list[NDArray], NDArray]]
        predict_proba method of a fitted multi-label classifier.
        It can return either:
        - a list of arrays of length n_classes where each array is of shape
          (n_samples, 2) with probabilities of the negative and positive class
          (as output by ``MultiOutputClassifier``), or
        - an ndarray of shape (n_samples, n_classes) or (n_samples, n_classes, 2)
          containing positive probabilities, or positive and negative probabilities
          (assuming last dimension is [neg, pos]).

    risk : str
        The risk metric to control ("precision" or "recall").
        The selected risk determines which conformal prediction methods are valid:
            - "precision" implies that method must be "ltt"
            - "recall" implies that method can be "crc" (default) or "rcps"

    method : Optional[str]
        Method to use for the prediction . If `risk` is
        "recall", the method can be either "crc" (default) or "rcps".
        If `risk` is "precision", the method used is "ltt".
        If ``None``, the default is "crc" for recall and "ltt" for precision.

    target_level : Optional[Union[float, Iterable[float]]]
        The minimum performance level for the metric. Must be between 0 and 1.
        Can be a float or any iterable of floats.
        By default ``0.9``.

    confidence_level : Optional[float]
        Can be a float, or ``None``. If using method="rcps" or method="ltt"
        (precision control), then it cannot be set to ``None`` and must lie in
        (0, 1). Between 0 and 1, the level of certainty at which we compute
        the Upper Confidence Bound of the average risk. Higher ``confidence_level``
        produce larger (more conservative) prediction sets. By default ``None``.

    rcps_bound : Optional[Union[str, ``None``]]
        Method used to compute the Upper Confidence Bound of the
        average risk. Only necessary with the RCPS method. If provided when
        using CRC or LTT it is ignored and a warning is raised. By default ``None``.
    predict_params : Optional[ArrayLike]
        Array of parameters (thresholds λ) to consider for controlling the risk.
        Defaults to np.arange(0, 1, 0.01). Length is used to set
        ``n_predict_params``.


    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        For this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        "None" is a marker for `unset` that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        to evaluate quantiles and prediction sets.
        Pass an int for reproducible output across multiple function calls.

        By default ``1``.

    verbose : int, optional
        The verbosity level, used with joblib for parallel processing.
        For the moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods. Either CRC or RCPS
    valid_bounds: List[Union[str, ``None``]]
        List of all valid bounds computation for RCPS only.

    n_predict_params: int
        Number of thresholds on which we compute the risk.

    predict_params: NDArray
        Array of parameters (noted λ in [3]) to consider for controlling the risk.

    risks : ArrayLike of shape (n_samples_cal, n_predict_params)
        The risk for each observation for each threshold

    r_hat : ArrayLike of shape (n_predict_params)
        Average risk for each predict_param

    r_hat_plus: ArrayLike of shape (n_predict_params)
        Upper confidence bound for each predict_param, computed
        with different bounds. Only relevant when
        method="rcps".

    best_predict_param: NDArray of shape (n_alpha)
        Optimal threshold for a given alpha.

    valid_index: List[List[Any]]
        List of list of all index that satisfy fwer controlling.
        This attribute is computed when the user wants to
        control precision score.
        Only relevant when risk="precision" as it uses
        learn then test (ltt) procedure.
        Contains n_alpha lists.

    valid_predict_params: List[List[Any]]
        List of list of all thresholds that satisfy fwer controlling.
        This attribute is computed when the user wants to
        control precision score.
        Only relevant when risk="precision" as it uses
        learn then test (ltt) procedure.
        Contains n_alpha lists.

     sigma_init : Optional[float]
        First variance in the sigma_hat array. The default
        value is the same as in the paper implementation [1].

    References
    ----------
    [1] Stephen Bates, Anastasios Angelopoulos, Lihua Lei, Jitendra Malik,
    and Michael I. Jordan. Distribution-free, risk-controlling prediction
    sets. CoRR, abs/2101.02703, 2021.
    URL https://arxiv.org/abs/2101.02703

    [2] Angelopoulos, Anastasios N., Stephen, Bates, Adam, Fisch, Lihua,
    Lei, and Tal, Schuster. "Conformal Risk Control." (2022).

    [3] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from mapie.risk_control import MultiLabelClassificationController
    >>> X_toy = np.arange(4).reshape(-1, 1)
    >>> y_toy = np.stack([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]])
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    >>> mapie_clf = MultiLabelClassificationController(predict_function=clf.predict_proba, target_level=0.7).calibrate(X_toy, y_toy)
    >>> y_pi_mapie = mapie_clf.predict(X_toy)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False  True]
     [ True False  True]
     [False  True False]
     [False  True False]]
    """

    risk_choice_map = {
        "precision": precision,
        "recall": recall,
    }

    valid_methods_by_metric_ = {"precision": ["ltt"], "recall": ["crc", "rcps"]}
    valid_methods = list(chain(*valid_methods_by_metric_.values()))
    valid_metric_ = list(valid_methods_by_metric_.keys())
    valid_bounds_ = ["hoeffding", "bernstein", "wsr", None]
    fit_attributes = ["risks"]
    sigma_init = 0.25  # Value given in the paper [1]
    cal_size = 0.3

    def __init__(
        self,
        predict_function: Callable[[ArrayLike], Union[list[NDArray], NDArray]],
        risk: str = "recall",
        method: Optional[str] = None,
        target_level: Union[float, Iterable[float]] = 0.9,
        confidence_level: Optional[float] = None,
        rcps_bound: Optional[Union[str, None]] = None,
        predict_params: ArrayLike = np.arange(0, 1, 0.01),
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0,
    ) -> None:
        self._predict_function = predict_function
        self._risk_name = risk
        self._risk = self._check_and_convert_risk(risk)
        self.method = method
        self._check_method()

        alpha = []
        for target in (
            target_level if isinstance(target_level, Iterable) else [target_level]
        ):
            assert self._risk.higher_is_better, (
                "Current implemented risks (precision and recall) are defined such that "
                "'higher is better'. The 'lower is better' case is not implemented."
            )
            alpha.append(1 - target)  # for higher is better only

        self._alpha = np.array(_check_alpha(alpha))

        self._check_confidence_level(confidence_level)
        self._delta = 1 - confidence_level if confidence_level is not None else None

        self._check_bound(rcps_bound)
        self._rcps_bound = rcps_bound

        self.predict_params = np.asarray(predict_params)
        self.n_predict_params = len(self.predict_params)

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self._check_parameters()

        self._is_fitted = False

    @property
    def is_fitted(self):
        """Returns True if the controller is fitted"""
        return self._is_fitted

    def _check_and_convert_risk(self, risk):
        """Check and convert risk parameter."""

        if risk not in self.risk_choice_map:
            raise ValueError(
                f"risk must be one of: {list(self.risk_choice_map.keys())}"
            )

        return self.risk_choice_map[risk]

    def _check_parameters(self) -> None:
        """
        Check n_jobs, verbose, and random_states.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        _check_n_jobs(self.n_jobs)
        _check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_method(self) -> None:
        """
        Check that the specified method is valid

        Raises
        ------
        ValueError
            Raise error if the name of the method is not
            in self.valid_methods_
        """
        valid_methods = self.valid_methods_by_metric_[self._risk_name]
        if self.method is None:
            self.method = valid_methods[0]
            return

        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid method '{self.method}' for risk '{self._risk_name}'. "
                f"Valid methods are: {valid_methods}."
            )

    def _check_all_labelled(self, y: NDArray) -> None:
        """
        Check that all observations have at least
        one label

        Parameters
        ----------
        y : NDArray of shape (n_samples, n_classes)
            Labels of the observations.

        Raises
        ------
        ValueError
            Raise error if at least one observation
            has no label.
        """
        if not (y.sum(axis=tuple(range(1, y.ndim))) > 0).all():
            raise ValueError(
                "Invalid y. All observations should contain at least one label."
            )

    def _check_confidence_level(self, confidence_level: Optional[float]):
        """
        Check that confidence_level is not ``None`` when the
        method is RCPS or LTT and that it is between 0 and 1.

        Parameters
        ----------
        confidence_level : float
            Probability with which we control the risk. The higher
            the probability, the more conservative the algorithm will be.

        Raises
        ------
        ValueError
            If confidence_level is ``None`` and method requires it
            (RCPS or LTT) or if confidence_level is not in (0, 1).

        Warning
            If confidence_level is not ``None`` and method is CRC
            (because it will be ignored).
        """
        if (not isinstance(confidence_level, float)) and (confidence_level is not None):
            raise ValueError(
                f"Invalid confidence_level. confidence_level must be a float, not a {type(confidence_level)}"
            )
        if (self.method == "rcps") or (self.method == "ltt"):
            if confidence_level is None:
                raise ValueError(
                    "Invalid confidence_level. "
                    f"confidence_level cannot be ``None`` when using method '{self.method}'."
                )
            elif (confidence_level <= 0) or (confidence_level >= 1):
                raise ValueError(
                    "Invalid confidence_level. confidence_level must be in (0, 1)"
                )
        if (self.method == "crc") and (confidence_level is not None):
            warnings.warn(
                "WARNING: you are using method 'crc', hence "
                "even if confidence_level is not ``None``, it will be ignored."
            )

    def _check_compute_risks_first_call(self) -> bool:
        """
        Check that this is the first time compute_risks
        or calibrate is called.

        Returns
        -------
        bool
            True if it is the first time, else False.
        """
        return not hasattr(self, "_risks")

    def _check_bound(self, bound: Optional[str]):
        """
        Check the value of the bound.
        """
        if bound not in self.valid_bounds_:
            raise ValueError(
                "bound must be in ['hoeffding', 'bernstein', 'wsr', ``None``]"
            )
        elif (bound is not None) and (self.method == "crc"):
            warnings.warn(
                "WARNING: you are using crc method, hence "
                + "even if the bound is not ``None``, it won't be"
                + "taken into account."
            )
        elif (bound is not None) and (self.method == "ltt"):
            warnings.warn(
                "WARNING: you are using ltt method hence "
                + "even if bound is not ``None``, it won't be"
                + "taken into account."
            )

    def _transform_pred_proba(
        self, y_pred_proba: Union[Sequence[NDArray], NDArray]
    ) -> NDArray:
        """Transform predict_function outputs to shape (n_samples, n_classes, 1)
        containing positive-class probabilities.

        - If a list of arrays is provided (e.g., MultiOutputClassifier), each
          array is expected to be of shape (n_samples, 2); we take the positive
          class column.
        - If an ndarray is provided, it can be of shape (n_samples, n_classes)
          containing positive-class probabilities, or
          (n_samples, n_classes, 2) containing both class probabilities, with
          last dim [negative, positive].
        """
        if isinstance(y_pred_proba, np.ndarray):
            if y_pred_proba.ndim == 3:
                # assume last dim is [neg, pos], keep positive class
                y_pred_pos = y_pred_proba[..., 1]
            elif y_pred_proba.ndim == 2:
                # already positive-class probabilities
                y_pred_pos = y_pred_proba
            else:
                raise ValueError(
                    "When predict_proba returns an ndarray, it must have 2 or 3 "
                    "dimensions: (n_samples, n_classes) or (n_samples, n_classes, 2)."
                )
        else:
            # list of length n_classes with (n_samples, 2) arrays
            y_pred_pos = np.stack([proba[:, 1] for proba in y_pred_proba], axis=1)

        return np.expand_dims(y_pred_pos, axis=2)

    def compute_risks(
        self,
        X: ArrayLike,
        y: ArrayLike,
        _refit: Optional[bool] = False,
    ) -> MultiLabelClassificationController:
        """
        Fit the base estimator or use the fitted base estimator on
        batch data to compute risks. All the computed risks will be concatenated each
        time the compute_risks method is called.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : NDArray of shape (n_samples, n_classes)
            Training labels.

        _refit: bool
            Whether or not refit from scratch.

            By default False

        Returns
        -------
        MultiLabelClassificationController
            The model itself.
        """
        # Checks
        first_call = self._check_compute_risks_first_call()

        X, y = indexable(X, y)

        y = cast(NDArray, y)
        X = cast(NDArray, X)

        self._check_all_labelled(y)
        self.n_samples_ = _num_samples(X)

        # Compute risks
        y_pred_proba = self._predict_function(X)
        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)

        n_lambdas = len(self.predict_params)
        n_samples = len(y_pred_proba_array)

        y_pred_proba_array_repeat = np.repeat(y_pred_proba_array, n_lambdas, axis=2)
        y_pred = (y_pred_proba_array_repeat > self.predict_params).astype(int)

        risk = np.zeros((n_samples, n_lambdas))
        for index_sample in range(n_samples):
            for index_lambda in range(n_lambdas):
                risk[index_sample, index_lambda], _ = (
                    self._risk.get_value_and_effective_sample_size(
                        y[index_sample, :], y_pred[index_sample, :, index_lambda]
                    )
                )

        if first_call or _refit:
            self._risks = risk
        else:
            self._risks = np.vstack((self._risks, risk))

        return self

    def compute_best_predict_param(self) -> MultiLabelClassificationController:
        """
        Compute optimal predict_params based on the computed risks.
        """
        if self._risk_name == "precision":
            self.n_obs = len(self._risks)
            self.r_hat = self._risks.mean(axis=0)
            self.valid_index, _ = ltt_procedure(
                np.expand_dims(self.r_hat, axis=0),
                np.expand_dims(self._alpha, axis=0),
                cast(float, self._delta),
                np.expand_dims(np.array([self.n_obs]), axis=0),
            )
            self.valid_predict_params = []
            for index_list in self.valid_index:
                self.valid_predict_params.append(self.predict_params[index_list])
            check_valid_ltt_params_index(
                predict_params=self.predict_params,
                valid_index=self.valid_index,
                alpha=self._alpha,
            )
            self.best_predict_param, _ = find_precision_best_predict_param(
                self.r_hat, self.valid_index, self.predict_params
            )
        elif self._risk_name == "recall":
            self.r_hat, self.r_hat_plus = get_r_hat_plus(
                self._risks,
                self.predict_params,
                self.method,
                self._rcps_bound,
                self._delta,
                self.sigma_init,
            )
            self.best_predict_param = find_best_predict_param(
                self.predict_params, self.r_hat_plus, self._alpha
            )
        else:
            raise NotImplementedError(
                "risk not implemented. Only 'precision' and 'recall' are currently supported."
            )
        self._is_fitted = True

        return self

    def calibrate(
        self, X: ArrayLike, y: ArrayLike
    ) -> MultiLabelClassificationController:
        """
         Use the fitted base estimator to compute risks and predict_params.
         Note that for high dimensional data, you can instead use the compute_risks
         method to compute risks batch by batch, followed by compute_best_predict_param.

         Parameters
         ----------
         X: ArrayLike of shape (n_samples, n_features)
             Training data.

         y: NDArray of shape (n_samples, n_classes)
             Training labels.

         Returns
         -------
        MultiLabelClassificationController
             The model itself.
        """

        self.compute_risks(X, y, _refit=True)
        self.compute_best_predict_param()

        return self

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Prediction sets on new samples based on the target risk level.
        Prediction sets for a given ``alpha`` are deduced from the computed
        risks.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)

        Returns
        -------
        NDArray of shape (n_samples, n_classes, n_alpha)
        """

        check_is_fitted(self)

        # Estimate prediction sets
        y_pred_proba = self._predict_function(X)
        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)

        y_pred_proba_array = np.repeat(y_pred_proba_array, len(self._alpha), axis=2)
        y_pred_proba_array = (
            y_pred_proba_array > self.best_predict_param[np.newaxis, np.newaxis, :]
        )
        return y_pred_proba_array

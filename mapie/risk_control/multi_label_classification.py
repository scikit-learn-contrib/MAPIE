from __future__ import annotations

import warnings
from itertools import chain
from typing import Callable, Iterable, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_y, _num_samples, indexable

from mapie.utils import (
    _check_alpha,
    _check_n_jobs,
    _check_verbose,
    check_is_fitted,
)

from .methods import (
    find_best_predict_param,
    find_precision_best_predict_param,
    get_r_hat_plus,
    ltt_procedure,
)
from .risks import compute_risk_precision, compute_risk_recall


class MultiLabelClassificationController(BaseEstimator, ClassifierMixin):
    """
    Prediction sets for multilabel-classification.

    This class implements two conformal prediction methods for
    estimating prediction sets for multilabel-classification.
    It guarantees (under the hypothesis of exchangeability) that
    a risk is at least 1 - alpha (alpha is a user-specified parameter).
    For now, we consider the recall as risk.

    Parameters
    ----------
    predict_function : Callable[[ArrayLike], NDArray]
        predict_proba method of a fitted multi-label classifier.
        It should return a list of arrays where the length of the list is n_classes
        and each array is of shape (n_samples, 2) corresponding to the
        probabilities of the negative and positive class for each label.

    metric_control : Optional[str]
        Metric to control. Either "recall" or "precision".
        By default ``recall``.

    method : Optional[str]
        Method to use for the prediction sets. If `metric_control` is
        "recall", then the method can be either "crc" (default) or "rcps".
        If `metric_control` is "precision", then the method used to control
        the precision is "ltt".

    target_level : Optional[Union[float, Iterable[float]]]
        The minimum performance level for the metric. Must be between 0 and 1.
        Can be a float or a list of floats.
        By default ``0.9``.

    confidence_level : Optional[float]
        Can be a float, or ``None``. If using method="rcps", then it
        can not be set to ``None``.
        Between 0 and 1, the level of certainty at which we compute
        the Upper Confidence Bound of the average risk.
        Higher ``confidence_level`` produce larger (more conservative) prediction
        sets.
        By default ``None``.

    rcps_bound : Optional[Union[str, ``None``]]
        Method used to compute the Upper Confidence Bound of the
        average risk. Only necessary with the RCPS method.
        By default ``None``.


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
        Only relevant when metric_control="precision" as it uses
        learn then test (ltt) procedure.
        Contains n_alpha lists.

    valid_predict_params: List[List[Any]]
        List of list of all thresholds that satisfy fwer controlling.
        This attribute is computed when the user wants to
        control precision score.
        Only relevant when metric_control="precision" as it uses
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

    valid_methods_by_metric_ = {"precision": ["ltt"], "recall": ["rcps", "crc"]}
    valid_methods = list(chain(*valid_methods_by_metric_.values()))
    valid_metric_ = list(valid_methods_by_metric_.keys())
    valid_bounds_ = ["hoeffding", "bernstein", "wsr", None]
    predict_params = np.arange(0, 1, 0.01)
    n_predict_params = len(predict_params)
    fit_attributes = ["risks"]
    sigma_init = 0.25  # Value given in the paper [1]
    cal_size = 0.3

    def __init__(
        self,
        predict_function: Callable[[ArrayLike], Union[list[NDArray], NDArray]],
        metric_control: Optional[str] = "recall",
        method: Optional[str] = None,
        target_level: Union[float, Iterable[float]] = 0.9,
        confidence_level: Optional[float] = None,
        rcps_bound: Optional[Union[str, None]] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0,
    ) -> None:
        self._predict_function = predict_function
        self.metric_control = metric_control
        self.method = method
        self._check_metric_control()
        self._check_method()

        alpha = []
        for target in (
            target_level if isinstance(target_level, Iterable) else [target_level]
        ):
            alpha.append(1 - target)  # higher is better for precision/recall
        self._alpha = np.array(_check_alpha(alpha))

        self._check_confidence_level(confidence_level)
        self._delta = 1 - confidence_level if confidence_level is not None else None

        self._check_bound(rcps_bound)
        self._rcps_bound = rcps_bound

        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self._check_parameters()

        self._is_fitted = False

    @property
    def is_fitted(self):
        """Returns True if the controller is fitted"""
        return self._is_fitted

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
        self.method = cast(str, self.method)
        self.metric_control = cast(str, self.metric_control)

        if self.method not in self.valid_methods_by_metric_[self.metric_control]:
            raise ValueError(
                "Invalid method for metric: "
                + "You are controlling "
                + self.metric_control
                + " and you are using invalid method: "
                + self.method
                + ". Use instead: "
                + "".join(self.valid_methods_by_metric_[self.metric_control])
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
        if not (y.sum(axis=1) > 0).all():
            raise ValueError(
                "Invalid y. All observations should contain at least one label."
            )

    def _check_confidence_level(self, confidence_level: Optional[float]):
        """
        Check that confidence_level is not ``None`` when the
        method is RCPS and that it is between 0 and 1.

        Parameters
        ----------
        confidence_level : float
            Probability with which we control the risk. The higher
            the probability, the more conservative the algorithm will be.

        Raises
        ------
        ValueError
            If confidence_level is ``None`` and method is RCPS or
            if confidence_level is not in [0, 1] and method
            is RCPS.
        Warning
            If confidence_level is not ``None`` and method is CRC
        """
        if (not isinstance(confidence_level, float)) and (confidence_level is not None):
            raise ValueError(
                f"Invalid confidence_level. confidence_level must be a float, not a {type(confidence_level)}"
            )
        if (self.method == "rcps") or (self.method == "ltt"):
            if confidence_level is None:
                raise ValueError(
                    "Invalid confidence_level. "
                    "confidence_level cannot be ``None`` when controlling "
                    "Recall with RCPS or Precision with LTT"
                )
            elif (confidence_level <= 0) or (confidence_level >= 1):
                raise ValueError(
                    "Invalid confidence_level. confidence_level must be in ]0, 1["
                )
        if (self.method == "crc") and (confidence_level is not None):
            warnings.warn(
                "WARNING: you are using crc method, hence "
                + "even if the confidence_level is not ``None``, it won't be"
                + "taken into account"
            )

    def _check_valid_index(self, alpha: NDArray):
        """
        Check if the valid index is empty.
        If it is, we should warn the user that for the alpha value
        and delta level chosen, LTT will return no results.
        The user must be less inclined to take risks or
        must choose a higher alpha value.
        """
        for i in range(len(self.valid_index)):
            if self.valid_index[i] == []:
                warnings.warn(
                    "Warning: LTT method has returned an empty sequence"
                    + " for alpha="
                    + str(alpha[i])
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
        return not hasattr(self, "risks")

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

    def _check_metric_control(self):
        """
        Check that the metrics to control are valid
        (can be a string or list of string.)
        """
        if self.metric_control not in self.valid_metric_:
            raise ValueError(
                "Invalid metric. "
                "Allowed scores must be in the following list "
                + ", ".join(self.valid_metric_)
            )

        if self.method is None:
            if self.metric_control == "recall":
                self.method = "crc"
            else:  # self.metric_control == "precision"
                self.method = "ltt"

    def _transform_pred_proba(
        self, y_pred_proba: Union[Sequence[NDArray], NDArray]
    ) -> NDArray:
        """If the output of the predict_proba is a list of arrays (output of
        the ``predict_proba`` of ``MultiOutputClassifier``) we transform it
        into an array of shape (n_samples, n_classes, 1), otherwise, we add
        one dimension at the end.

        Parameters
        ----------
        y_pred_proba : Union[List, NDArray]
            Output of the multi-label classifier.

        Returns
        -------
        NDArray of shape (n_samples, n_classes, 1)
            Output of the model ready for risk computation.
        """
        if isinstance(y_pred_proba, np.ndarray):
            y_pred_proba_array = y_pred_proba
        else:
            y_pred_proba_stacked = np.stack(y_pred_proba, axis=0)[:, :, 1]
            y_pred_proba_array = np.moveaxis(y_pred_proba_stacked, 0, -1)

        return np.expand_dims(y_pred_proba_array, axis=2)

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
        _check_y(y, multi_output=True)

        y = cast(NDArray, y)
        X = cast(NDArray, X)

        self._check_all_labelled(y)
        self.n_samples_ = _num_samples(X)

        # Compute risks
        y_pred_proba = self._predict_function(X)
        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)

        if self.metric_control == "recall":
            risk = compute_risk_recall(self.predict_params, y_pred_proba_array, y)
        else:  # self.metric_control == "precision"
            risk = compute_risk_precision(self.predict_params, y_pred_proba_array, y)

        if first_call or _refit:
            self.risks = risk
        else:
            self.risks = np.vstack((self.risks, risk))

        return self

    def compute_best_predict_param(self) -> MultiLabelClassificationController:
        """
        Compute optimal predict_params based on the computed risks.
        """
        if self.metric_control == "precision":
            self.n_obs = len(self.risks)
            self.r_hat = self.risks.mean(axis=0)
            self.valid_index, _ = ltt_procedure(
                np.expand_dims(self.r_hat, axis=0),
                np.expand_dims(self._alpha, axis=0),
                cast(float, self._delta),
                np.expand_dims(np.array([self.n_obs]), axis=0),
            )
            self.valid_predict_params = []
            for index_list in self.valid_index:
                self.valid_predict_params.append(self.predict_params[index_list])
            self._check_valid_index(self._alpha)
            self.best_predict_param, _ = find_precision_best_predict_param(
                self.r_hat, self.valid_index, self.predict_params
            )
        else:
            self.r_hat, self.r_hat_plus = get_r_hat_plus(
                self.risks,
                self.predict_params,
                self.method,
                self._rcps_bound,
                self._delta,
                self.sigma_init,
            )
            self.best_predict_param = find_best_predict_param(
                self.predict_params, self.r_hat_plus, self._alpha
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

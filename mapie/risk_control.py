from __future__ import annotations

import warnings
from itertools import chain
from typing import (
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    Callable,
    Literal,
    List, Any,
)

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from numpy.typing import ArrayLike, NDArray
from .control_risk.crc_rcps import find_lambda_star, get_r_hat_plus
from .control_risk.ltt import find_lambda_control_star, ltt_procedure
from .control_risk.risks import compute_risk_precision, compute_risk_recall
from .utils import _check_alpha, _check_n_jobs, _check_verbose


class PrecisionRecallController(BaseEstimator, ClassifierMixin):
    """
    Prediction sets for multilabel-classification.

    This class implements two conformal prediction methods for
    estimating prediction sets for multilabel-classification.
    It guarantees (under the hypothesis of exchangeability) that
    a risk is at least 1 - alpha (alpha is a user-specified parameter).
    For now, we consider the recall as risk.

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any fitted multi-label classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods).
        If ``None``, estimator by default is a sklearn LogisticRegression
        instance.

         by default ``None``

    metric_control : Optional[str]
        Metric to control. Either "recall" or "precision".
        By default ``recall``.

    method : Optional[str]
        Method to use for the prediction sets. If `metric_control` is
        "recall", then the method can be either "crc" or "rcps".
        If `metric_control` is "precision", then the method used to control
        the precision is "ltt".
        If `metric_control` is "recall" the default method is "crc".

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
    valid_methods: List[Union[str, ``None``]]
        List of all valid bounds computation for RCPS only.
    single_estimator_ : sklearn.ClassifierMixin
        Estimator fitted on the whole training set.

    n_lambdas: int
        Number of thresholds on which we compute the risk.

    lambdas: NDArray
        Array with all the values of lambda.

    risks : ArrayLike of shape (n_samples_cal, n_lambdas)
        The risk for each observation for each threshold

    r_hat : ArrayLike of shape (n_lambdas)
        Average risk for each lambda

    r_hat_plus: ArrayLike of shape (n_lambdas)
        Upper confidence bound for each lambda, computed
        with different bounds (see predict). Only relevant when
        method="rcps".

    lambdas_star: ArrayLike of shape (n_lambdas)
        Optimal threshold for a given alpha.

    valid_index: List[List[Any]]
        List of list of all index that satisfy fwer controlling.
        This attribute is computed when the user wants to
        control precision score.
        Only relevant when metric_control="precision" as it uses
        learn then test (ltt) procedure.
        Contains n_alpha lists (see predict).

     sigma_init : Optional[float]
        First variance in the sigma_hat array. The default
        value is the same as in the paper implementation [1].

    References
    ----------
    [1] Lihua Lei Jitendra Malik Stephen Bates, Anastasios Angelopoulos,
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
    >>> from mapie.risk_control import PrecisionRecallController
    >>> X_toy = np.arange(4).reshape(-1, 1)
    >>> y_toy = np.stack([[1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0]])
    >>> clf = MultiOutputClassifier(LogisticRegression()).fit(X_toy, y_toy)
    >>> mapie = PrecisionRecallController(estimator=clf).fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.3)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False  True]
     [ True False  True]
     [False  True False]
     [False  True False]]
    """
    valid_methods_by_metric_ = {
        "precision": ["ltt"],
        "recall": ["rcps", "crc"]
    }
    valid_methods = list(chain(*valid_methods_by_metric_.values()))
    valid_metric_ = list(valid_methods_by_metric_.keys())
    valid_bounds_ = ["hoeffding", "bernstein", "wsr", None]
    lambdas = np.arange(0, 1, 0.01)
    n_lambdas = len(lambdas)
    fit_attributes = [
        "single_estimator_",
        "risks"
    ]
    sigma_init = 0.25  # Value given in the paper [1]
    cal_size = .3

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        metric_control: Optional[str] = 'recall',
        method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.metric_control = metric_control
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

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

        if self.method not in self.valid_methods_by_metric_[
            self.metric_control
        ]:
            raise ValueError(
                "Invalid method for metric: "
                + "You are controlling " + self.metric_control
                + " and you are using invalid method: " + self.method
                + ". Use instead: " + "".join(self.valid_methods_by_metric_[
                    self.metric_control]
                )
            )

    def _check_all_labelled(self, y: NDArray) -> None:
        """
        Check that all observations have at least
        one label

        Parameters
        ----------
        y : NDArray of shape (n_samples, n_labels)
            Labels of the observations.

        Raises
        ------
        ValueError
            Raise error if at least one observation
            has no label.
        """
        if not (y.sum(axis=1) > 0).all():
            raise ValueError(
                "Invalid y. "
                "All observations should contain at "
                "least one label."
            )

    def _check_delta(self, delta: Optional[float]):
        """
        Check that delta is not ``None`` when the
        method is RCPS and that it is between 0 and 1.

        Parameters
        ----------
        delta : float
            Probability with which we control the risk. The higher
            the probability, the more conservative the algorithm will be.

        Raises
        ------
        ValueError
            If delta is ``None`` and method is RCSP or
            if delta is not in [0, 1] and method
            is RCPS.
        Warning
            If delta is not ``None`` and method is CRC
        """
        if (not isinstance(delta, float)) and (delta is not None):
            raise ValueError(
                "Invalid delta. "
                f"delta must be a float, not a {type(delta)}"
            )
        if (self.method == "rcps") or (self.method == "ltt"):
            if delta is None:
                raise ValueError(
                    "Invalid delta. "
                    "delta cannot be ``None`` when controlling "
                    "Recall with RCPS or Precision with LTT"
                )
            elif ((delta <= 0) or (delta >= 1)):
                raise ValueError(
                    "Invalid delta. "
                    "delta must be in ]0, 1["
                )
        if (self.method == "crc") and (delta is not None):
            warnings.warn(
                "WARNING: you are using crc method, hence "
                + "even if the delta is not ``None``, it won't be"
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
                    + " for alpha=" + str(alpha[i])
                )

    def _check_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
        _refit: Optional[bool] = False,
    ) -> ClassifierMixin:
        """
        Check the estimator value. If it is ``None``,
        it returns a multi-output ``LogisticRegression``
        instance if necessary.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        estimator : Optional[ClassifierMixin], optional
            Estimator to check, by default ``None``

        _refit : Optional[bool]
            Whether or not the user is using fit (True) or
            partial_fit (False).

            By default False


        Returns
        -------
        ClassifierMixin
            The estimator itself or a default multi-output
            ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit, predict, nor predict_proba methods.

        NotFittedError
            If the estimator is not fitted.

        Warning
            If estimator is then to warn about the split of the
            data between train and calibration
        """
        if (estimator is None) and (not _refit):
            raise ValueError(
                "Invalid estimator with partial_fit. "
                "If the estimator is ``None`` you can not "
                "use partial_fit."
            )
        if (estimator is None) and (_refit):
            estimator = MultiOutputClassifier(
                LogisticRegression()
            )
            X_train, X_calib, y_train, y_calib = train_test_split(
                X,
                y,
                test_size=self.calib_size,
                random_state=self.random_state,
            )
            estimator.fit(X_train, y_train)
            warnings.warn(
                "WARNING: To avoid overfitting, X has been split"
                + "into X_train and X_calib. The calibration will only"
                + "be done on X_calib"
            )
            return estimator, X_calib, y_calib

        if isinstance(estimator, Pipeline):
            est = estimator[-1]
        else:
            est = estimator
        if (
            not hasattr(est, "fit")
            or not hasattr(est, "predict")
            or not hasattr(est, "predict_proba")
        ):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )
        check_is_fitted(est)
        return estimator, X, y

    def _check_partial_fit_first_call(self) -> bool:
        """
        Check that this is the first time partial_fit
        or fit is called.

        Returns
        -------
        bool
            True if it is the first time, else False.
        """
        return not hasattr(self, "risks")

    def _check_bound(self, bound: Optional[str]):
        """
        Check the value of the bound.

        Parameters
        ----------
        bound : Optional[str]
            Bound defined in the predict.

        Raises
        ------
        AttributeError
            If bound is not in ["hoeffding", "bernstein", "wsr", ``None``]

        Warning
            If bound is not ``None``and method is CRC
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
        self,
        y_pred_proba: Union[Sequence[NDArray], NDArray]
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
            y_pred_proba_stacked = np.stack(
                y_pred_proba,  # type: ignore
                axis=0
            )[:, :, 1]
            y_pred_proba_array = np.moveaxis(y_pred_proba_stacked, 0, -1)

        return np.expand_dims(y_pred_proba_array, axis=2)

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        _refit: Optional[bool] = False,
    ) -> PrecisionRecallController:
        """
        Fit the base estimator or use the fitted base estimator on
        batch data. All the computed risks will be concatenated each
        time the partial_fit method is called.

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
        PrecisionRecallController
            The model itself.
        """
        # Checks
        first_call = self._check_partial_fit_first_call()
        self._check_parameters()
        self._check_metric_control()
        self._check_method()

        X, y = indexable(X, y)
        _check_y(y, multi_output=True)
        estimator, X, y = self._check_estimator(
            X, y, self.estimator,
            _refit
        )

        y = cast(NDArray, y)
        X = cast(NDArray, X)

        self._check_all_labelled(y)
        self.n_samples_ = _num_samples(X)

        # Work
        if first_call or _refit:
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
            self.theta_ = X.shape[1]

            if self.metric_control == "recall":
                self.risks = compute_risk_recall(
                    self.lambdas, y_pred_proba_array, y
                )
            else:  # self.metric_control == "precision"
                self.risks = compute_risk_precision(
                    self.lambdas, y_pred_proba_array, y
                )
        else:
            if X.shape[1] != self.theta_:
                msg = "Number of features %d does not match previous data %d."
                raise ValueError(msg % (X.shape[1], self.theta_))
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
            if self.metric_control == "recall":
                partial_risk = compute_risk_recall(
                    self.lambdas,
                    y_pred_proba_array,
                    y
                )
            else:  # self.metric_control == "precision"
                partial_risk = compute_risk_precision(
                    self.lambdas,
                    y_pred_proba_array,
                    y
                )
            self.risks = np.concatenate([self.risks, partial_risk], axis=0)

        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        calib_size: Optional[float] = .3
    ) -> PrecisionRecallController:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: NDArray of shape (n_samples, n_classes)
            Training labels.

        calib_size: Optional[float]
            Size of the calibration dataset with respect to X if the
            given model is ``None`` need to fit a LogisticRegression.

            By default .3

        Returns
        -------
        PrecisionRecallController
            The model itself.
        """
        self.calib_size = calib_size
        return self.partial_fit(X, y, _refit=True)

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        delta: Optional[float] = None,
        bound: Optional[Union[str, None]] = None
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Prediction sets on new samples based on target confidence
        interval.
        Prediction sets for a given ``alpha`` are deduced from the computed
        risks.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)

        alpha : Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            sets.
            ``alpha`` is the complement of the target coverage level.
            By default ``None``.

        delta : Optional[float]
            Can be a float, or ``None``. If using method="rcps", then it
            can not be set to ``None``.
            Between 0 and 1, the level of certainty at which we compute
            the Upper Confidence Bound of the average risk.
            Lower ``delta`` produce larger (more conservative) prediction
            sets.
            By default ``None``.

        bound : Optional[Union[str, ``None``]]
            Method used to compute the Upper Confidence Bound of the
            average risk. Only necessary with the RCPS method.
            By default ``None``.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is ``None``.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples, n_classes) and (n_samples, n_classes, n_alpha)
        if alpha is not ``None``.
        """

        self._check_delta(delta)
        self._check_bound(bound)
        alpha = cast(Optional[NDArray], _check_alpha(alpha))
        check_is_fitted(self, self.fit_attributes)

        # Estimate prediction sets
        y_pred = self.single_estimator_.predict(X)

        if alpha is None:
            return np.array(y_pred)

        alpha_np = cast(NDArray, alpha)

        y_pred_proba = self.single_estimator_.predict_proba(X)

        y_pred_proba_array = self._transform_pred_proba(y_pred_proba)
        y_pred_proba_array = np.repeat(
            y_pred_proba_array,
            len(alpha_np),
            axis=2
        )
        if self.metric_control == 'precision':
            self.n_obs = len(self.risks)
            self.r_hat = self.risks.mean(axis=0)
            self.valid_index = ltt_procedure(
                self.r_hat, alpha_np, cast(float, delta), self.n_obs
            )
            self._check_valid_index(alpha_np)
            self.lambdas_star, self.r_star = find_lambda_control_star(
                self.r_hat, self.valid_index, self.lambdas
            )
            y_pred_proba_array = (
                y_pred_proba_array >
                np.array(self.lambdas_star)[np.newaxis, np.newaxis, :]
            )

        else:
            self.r_hat, self.r_hat_plus = get_r_hat_plus(
                self.risks, self.lambdas, self.method,
                bound, delta, self.sigma_init
            )
            self.lambdas_star = find_lambda_star(
                self.lambdas, self.r_hat_plus, alpha_np
            )
            y_pred_proba_array = (
                y_pred_proba_array >
                self.lambdas_star[np.newaxis, np.newaxis, :]
            )
        return y_pred, y_pred_proba_array


class BinaryClassificationRisk:
    """
    Define a risk (or a performance metric) to be used with the
    BinaryClassificationController. Predefined instances are implemented,
    see :data:`mapie.risk_control.precision`, :data:`mapie.risk_control.recall`,
    :data:`mapie.risk_control.accuracy` and
    :data:`mapie.risk_control.false_positive_rate`.

    Here, a binary classification risk (or performance) is defined by an occurrence and
    a condition. Let's take the example of precision. Precision is the sum of true
    positives over the total number of predicted positives. In other words, precision is
    the average of correct predictions (occurrence) given that those predictions
    are positive (condition). Programmatically,
    ``precision = (sum(y_pred == y_true) if y_pred == 1)/sum(y_pred == 1)``.
    Because precision is a performance metric rather than a risk, `higher_is_better`
    must be set to `True`. See the implementation of `precision` in mapie.risk_control.

    Note: any risk or performance metric that can be defined as
    ``sum(occurrence if condition) / sum(condition)`` can be theoretically controlled
    with the BinaryClassificationController, thanks to the LearnThenTest framework [1]
    and the binary Hoeffding-Bentkus p-values implemented in MAPIE.

    Note: by definition, the value of the risk (or performance metric) here is always
    between 0 and 1.

    Parameters
    ----------
    risk_occurrence : Callable[[int, int], bool]
        A function defining the occurrence of the risk for a given sample.
        Must take y_true and y_pred as input and return a boolean.

    risk_condition : Callable[[int, int], bool]
        A function defining the condition of the risk for a given sample,
        Must take y_true and y_pred as input and return a boolean.

    higher_is_better : bool
        Whether this BinaryClassificationRisk instance is a risk
        (higher_is_better=False) or a performance metric (higher_is_better=True).

    Attributes
    ----------
    higher_is_better : bool
        See params.

    References
    ----------
    [1] Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al.
    "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022)
    """

    def __init__(
        self,
        risk_occurrence: Callable[[int, int], bool],
        risk_condition: Callable[[int, int], bool],
        higher_is_better: bool,
    ):
        self._risk_occurrence = risk_occurrence
        self._risk_condition = risk_condition
        self.higher_is_better = higher_is_better

    def get_value_and_effective_sample_size(
        self,
        y_true: NDArray,
        y_pred: NDArray,
    ) -> Tuple[float, int]:
        """
        Computes the value of a risk given an array of ground
        truth labels and the corresponding predictions. Also returns the number of
        samples used to compute that value.

        That number can be different from the total number of samples. For example, in
        the case of precision, only the samples with positive predictions are used.

        In the case of a performance metric, this function returns 1 - perf_value.

        Parameters
        ----------
        y_true : NDArray
            NDArray of ground truth labels, of shape (n_samples,), with values in {0, 1}

        y_pred : NDArray
            NDArray of predictions, of shape (n_samples,), with values in {0, 1}

        Returns
        -------
        Tuple[float, int]
            A tuple containing the value of the risk between 0 and 1,
            and the number of effective samples used to compute that value
            (between 1 and n_samples).

            In the case of a performance metric, this function returns 1 - perf_value.

            If the risk is not defined (condition never met), the value is set to 1,
            and the number of effective samples is set to -1.
        """
        risk_occurrences = np.array([
            self._risk_occurrence(y_true_i, y_pred_i)
            for y_true_i, y_pred_i in zip(y_true, y_pred)
        ])
        risk_conditions = np.array([
            self._risk_condition(y_true_i, y_pred_i)
            for y_true_i, y_pred_i in zip(y_true, y_pred)
        ])
        effective_sample_size = len(y_true) - np.sum(~risk_conditions)
        # Casting needed for MyPy with Python 3.9
        effective_sample_size_int = cast(int, effective_sample_size)
        if effective_sample_size_int != 0:
            risk_sum: int = np.sum(risk_occurrences[risk_conditions])
            risk_value = risk_sum / effective_sample_size_int
        else:
            # In this case, the corresponding lambda shouldn't be considered valid.
            # In the current LTT implementation, providing n_obs=-1 will result
            # in an infinite p_value, effectively invaliding the lambda
            risk_value, effective_sample_size_int = 1, -1
        if self.higher_is_better:
            risk_value = 1 - risk_value
        return risk_value, effective_sample_size_int


precision = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: y_pred == 1,
    higher_is_better=True,
)

accuracy = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: True,
    higher_is_better=True,
)

recall = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == y_true,
    risk_condition=lambda y_true, y_pred: y_true == 1,
    higher_is_better=True,
)

false_positive_rate = BinaryClassificationRisk(
    risk_occurrence=lambda y_true, y_pred: y_pred == 1,
    risk_condition=lambda y_true, y_pred: y_true == 0,
    higher_is_better=False,
)


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
        Its output signature must be of shape (len(X), 2)

    risk : Union[BinaryClassificationRisk, List[BinaryClassificationRisk]]
        The risk or performance metric to control.
        Valid options:

        - An existing risk defined in `mapie.risk_control` (e.g. precision, recall,
          accuracy, false_positive_rate)
        - A custom instance of BinaryClassificationRisk object

        Can be a list of risks in the case of multi risk control.

    target_level : Union[float, List[float]]
        The maximum risk level (or minimum performance level). Must be between 0 and 1.
        Can be a list of target levels in the case of multi risk control (length should
        match the length of the risks list).

    confidence_level : float, default=0.9
        The confidence level with which the risk (or performance) is controlled.
        Must be between 0 and 1. See the documentation for detailed explanations.

    best_predict_param_choice : Union["auto", BinaryClassificationRisk], default="auto"
        How to select the best threshold from the valid thresholds that control the risk
        (or performance). The BinaryClassificationController will try to minimize
        (or maximize) a secondary objective.
        Valid options:

        - "auto" (default)
        - An existing risk defined in `mapie.risk_control` (e.g. precision, recall,
          accuracy, false_positive_rate)
        - A custom instance of BinaryClassificationRisk object

    Attributes
    ----------
    valid_predict_params : NDArray
        The valid thresholds that control the risk (or performance).
        Use the calibrate method to compute these.

    best_predict_param : Optional[float]
        The best threshold that control the risk (or performance).
        Use the calibrate method to compute it.

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
    Angelopoulos, Anastasios N., Stephen, Bates, Emmanuel J. Candès, et al.
    "Learn Then Test: Calibrating Predictive Algorithms to Achieve Risk Control." (2022)
    """
    _best_predict_param_choice_map = {
        precision: recall,
        recall: precision,
        accuracy: accuracy,
        false_positive_rate: recall,
    }

    def __init__(
        self,
        predict_function: Callable[[ArrayLike], NDArray],
        risk: Union[BinaryClassificationRisk, List[BinaryClassificationRisk]],
        target_level: Union[float, List[float]],
        confidence_level: float = 0.9,
        best_predict_param_choice: Union[
            Literal["auto"], BinaryClassificationRisk] = "auto",
    ):
        self.is_multi_risk = self._check_if_multi_risk_control(risk, target_level)
        self._predict_function = predict_function
        self._risk = risk
        self._alpha = self._convert_target_level_to_alpha(target_level)
        self._delta = 1 - confidence_level

        self._best_predict_param_choice = self._set_best_predict_param_choice(
            best_predict_param_choice
        )

        self._predict_params: NDArray = np.linspace(0, 0.99, 100)

        self.valid_predict_params: NDArray = np.array([])
        self.best_predict_param: Optional[float] = None

    # All subfunctions are unit-tested. To avoid having to write
    # tests just to make sure those subfunctions are called,
    # we don't include .calibrate in the coverage report
    def calibrate(  # pragma: no cover
        self,
        X_calibrate: ArrayLike,
        y_calibrate: ArrayLike
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
            X_calibrate,
            self._predict_params
        )

        risk_values, eff_sample_sizes = self._get_risk_values_and_eff_sample_sizes(
            y_calibrate_,
            predictions_per_param,
            self._risk if isinstance(self._risk, list) else [self._risk]
        )

        valid_params_index = ltt_procedure(
            risk_values,
            np.array([self._alpha]),
            self._delta,
            eff_sample_sizes,
            True,
        )[0]

        self.valid_predict_params = self._predict_params[valid_params_index]

        if len(self.valid_predict_params) == 0:
            self._set_risk_not_controlled()
        else:
            self._set_best_predict_param(
                y_calibrate_,
                predictions_per_param,
                valid_params_index,
            )
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
                "either calibration was not successful."
            )
        return self._get_predictions_per_param(
            X_test,
            np.array([self.best_predict_param]),
        )[0]

    def _set_best_predict_param_choice(
        self,
        best_predict_param_choice: Union[
            Literal["auto"], BinaryClassificationRisk] = "auto",
    ) -> BinaryClassificationRisk:
        if best_predict_param_choice == "auto":
            if self.is_multi_risk:
                # when multi risk, we minimize the first risk in the list
                return self._risk[0]  # type: ignore
            else:
                try:
                    return self._best_predict_param_choice_map[
                        self._risk  # type: ignore
                    ]
                except KeyError:
                    raise ValueError(
                        "When best_predict_param_choice is 'auto', "
                        "risk must be one of the risks defined in mapie.risk_control"
                        "(e.g. precision, accuracy, false_positive_rate)."
                    )
        else:
            return best_predict_param_choice

    def _set_risk_not_controlled(self) -> None:
        self.best_predict_param = None
        warnings.warn(
            "No predict parameters were found to control the risk at the given "
            "target and confidence levels. "
            "Try using a larger calibration set or a better model.",
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
                [self._best_predict_param_choice]
            )

        self.best_predict_param = self.valid_predict_params[
            np.argmin(secondary_risks_per_param)
        ]

    @staticmethod
    def _get_risk_values_and_eff_sample_sizes(
        y_true: NDArray,
        predictions_per_param: NDArray,
        risks: List[BinaryClassificationRisk],
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the values of risks and effective sample sizes for multiple risks
        and for multiple parameter values.
        Returns a tuple of two arrays with shape (n_risks, n_params).
        """
        risks_values_and_eff_sizes = np.array([
            [risk.get_value_and_effective_sample_size(y_true, predictions)
             for predictions in predictions_per_param]
            for risk in risks
        ])

        risk_values = risks_values_and_eff_sizes[:, :, 0]
        effective_sample_sizes = risks_values_and_eff_sizes[:, :, 1]

        return risk_values, effective_sample_sizes

    def _get_predictions_per_param(self, X: ArrayLike, params: NDArray) -> NDArray:
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
        return (predictions_proba[:, np.newaxis] >= params).T.astype(int)

    def _convert_target_level_to_alpha(self, target_level):
        if self.is_multi_risk:
            alpha = []
            for risk, target in zip(self._risk, target_level):
                if risk.higher_is_better:
                    alpha.append(1 - target)
                else:
                    alpha.append(target)
        else:
            if self._risk.higher_is_better:
                alpha = 1 - target_level
            else:
                alpha = target_level
        return alpha

    @staticmethod
    def _check_if_multi_risk_control(  # TODO what about lists of len 1
        risk: Union[BinaryClassificationRisk, List[BinaryClassificationRisk]],
        target_level: Union[float, List[float]],
    ) -> bool:
        """
        Check if we are in a multi risk setting and if inputs types are correct.
        """
        if (
            isinstance(risk, list) and isinstance(target_level, list)
            and len(risk) == len(target_level)
        ):
            return True
        elif (
            isinstance(risk, BinaryClassificationRisk)
            and isinstance(target_level, float)
        ):
            return False
        else:
            raise ValueError(
                "If you provide a list of risks, you must provide "
                "a list of target levels of the same length and vice versa. "
                "If you provide a single BinaryClassificationRisk risk, "
                "you must provide a single float target level."
            )

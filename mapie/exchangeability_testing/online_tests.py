from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from mapie.classification import CrossConformalClassifier, SplitConformalClassifier
from mapie.regression import (
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    SplitConformalRegressor,
)

MapieEstimator = Union[
    SplitConformalClassifier,
    SplitConformalRegressor,
]


class OnlineMartingaleTest:
    """
    Online test of exchangeability based on conformal p-values and test martingales.

    OnlineMartingaleTest sequentially monitors whether newly observed labeled data
    remain exchangeable with respect to a reference stream, using conformity
    scores, conformal p-values, and a martingale-based evidence process.

    At each update, the class:

    1. computes conformity scores from observed features and labels,
    2. converts these scores into conformal p-values using past scores,
    3. updates a martingale statistic from the p-values,
    4. monitors whether the observed stream provides evidence against exchangeability.

    Two martingale constructions are currently supported:

    - ``"jumper_martingale"``: a simple and robust betting martingale based on
    a finite set of experts.
    - ``"plugin_martingale"``: a plug-in martingale using an estimated density
    of past p-values.

    The null hypothesis is that the sequence of observations is exchangeable.
    Large martingale values provide evidence against exchangeability.

    Parameters
    ----------
    mapie_estimator : Optional[MapieEstimator], default=None
        MAPIE estimator used to compute predictions and non-conformity
        scores. Supported estimators are
        :class:`SplitConformalClassifier`,
        and :class:`SplitConformalRegressor`.
        If ``None``, a default
        :class:`SplitConformalClassifier` or
        :class:`SplitConformalRegressor` is built
        when needed.
        If the estimator is not fitted or not provided, it will be fitted on a
        slice of the data in order to compute non-conformity scores.

    task : Optional[Literal["classification", "regression"]], default=None
        Task type. If ``None``, the task is inferred from `y`.

    test_method : {"jumper_martingale", "plugin_martingale"}, default="jumper_martingale"
        Martingale construction used to aggregate evidence across p-values.
        To compare both methods in parallel, instantiate two
        ``OnlineMartingaleTest`` objects with different ``test_method`` values
        and update them on the same stream.

    test_level : float, default=0.05
        Level used to test the hypothesis that the dataset is exchangeable.
        The probability that the test gives a false positive is at most
        `test_level` (type I error).

    warn : bool, default=True
        Whether to raise a warning when exchangeability is rejected.
        The warning is issued at most once per instance.

    jump_size : float, default=0.01
        Mixing parameter used by the jumper martingale.
        Ignored when ``test_method="plugin_martingale"``.

    burn_in : int, default=100
        Minimum sample size required before the ``is_exchangeable``
        property is allowed to return a non-``None`` decision.

    random_state : Optional[int], default=None
        Random seed used for random tie-breaking and density estimation.

    Attributes
    ----------
    pvalue_history : list of float
        History of conformal p-values observed so far.

    conformity_score_history : list of float
        History of conformity scores observed so far.

    martingale_value_history : list of float
        History of martingale values after each update.

    current_martingale_value : float
        Current value of the martingale process.

    Examples
    --------
    >>> RANDOM_STATE = 7
    >>> mapie_estimator = SplitConformalRegressor(prefit=False)
    >>> omt = OnlineMartingaleTest(
    ...     mapie_estimator=mapie_estimator,
    ...     task="regression",
    ...     random_state=0,
    ...     burn_in=1,
    ... )
    >>> rng = np.random.default_rng(RANDOM_STATE)
    >>> X = np.linspace(0.1, 0.9, 2400).reshape(-1, 1)
    >>> y = 3.0 * X.ravel() + rng.normal(scale=0.1, size=X.shape[0])
    >>> omt = omt.update(X, y)
    >>> omt.is_exchangeable is True
    True

    Notes
    -----
    The class is designed for sequential monitoring. It can be initialized on a
    reference labeled dataset using ``update`` and then updated online whenever
    new labels become available.

    The martingale provides a valid sequential test against exchangeability when
    the p-values are valid under the null hypothesis.

    References
    ----------
    .. [1] Angelopoulos, Barber, Bates (2026).
        "Theoretical Foundations of Conformal Prediction".
        arXiv preprint arXiv:2411.11824.
    .. [2] Vovk, Gammerman, Shafer (2005).
        "Algorithmic Learning in a Random World".
        Boston, MA: Springer US. Section 7.1, page 169.
    .. [3] Fedorova, Gammerman, Nouretdinov, Vovk (2012).
        "Plug-in Martingales for Testing Exchangeability on-line".
        In Proceedings of the 29th ICML. Algorithm 1, page 3.
    """

    def __init__(
        self,
        mapie_estimator: Optional[MapieEstimator] = None,
        task: Optional[Literal["classification", "regression"]] = None,
        test_method: Literal[
            "jumper_martingale", "plugin_martingale"
        ] = "jumper_martingale",
        test_level: float = 0.05,
        warn: bool = True,
        jump_size: float = 0.01,
        burn_in: int = 100,
        random_state: Optional[np.int_] = None,
    ):
        """
        Initialize the online martingale test.

        Parameters
        ----------
        mapie_estimator : Optional[MapieEstimator], default=None
            MAPIE estimator used to compute predictions and non-conformity
            scores. Supported estimators are
            :class:`SplitConformalClassifier`,
            and :class:`SplitConformalRegressor`.
            If ``None``, a default
            :class:`SplitConformalClassifier` or
            :class:`SplitConformalRegressor` is built
            when needed.
            If the estimator is not fitted or not provided, it will be fitted on a
            slice of the data in order to compute non-conformity scores.

        task : Optional[Literal["classification", "regression"]], default=None
            Task type. If ``None``, the task is inferred from `y`.

        test_method : {"jumper_martingale", "plugin_martingale"}, default="jumper_martingale"
            Martingale construction used to aggregate evidence from conformal p-values.
            "jumper_martingale" is more stable and less tuning-sensitive.
            "plugin_martingale" is more adaptive but sensitive to density estimation.
            To monitor both methods, use two instances and update both online.

        test_level : float, default=0.05
            Level used to test the hypothesis that the dataset is exchangeable.
            The probability that the test gives a false positive is at most
            `test_level` (type I error).

        warn : bool, default=True
            Whether to raise a warning when exchangeability is rejected.

        jump_size : float, default=0.01
            Mixing parameter for the jumper martingale, controlling expert diversity.
            Must lie in (0, 1). Ignored when test_method="plugin_martingale".

        burn_in : int, default=100
            Minimum number of observations required before is_exchangeable returns
            a non-None decision.

        random_state : Optional[int], default=None
            Random seed used for randomization (e.g., tie-breaking in p-value computation).

        Raises
        ------
        ValueError
            If test_level is not in (0, 1), if test_method is not supported,
            or if jump_size is not in (0, 1).

        See Also
        --------
        update : Update the test with new observations.
        is_exchangeable : Get current exchangeability decision.
        summary : Get diagnostic summary of the test state.

        References
        ----------
        .. [1] Angelopoulos, Barber, Bates (2026).
            "Theoretical Foundations of Conformal Prediction".
            arXiv preprint arXiv:2411.11824.
        .. [2] Vovk, Gammerman, Shafer (2005).
            "Algorithmic Learning in a Random World".
            Boston, MA: Springer US. Section 7.1, page 169.
        .. [3] Fedorova, Gammerman, Nouretdinov, Vovk (2012).
            "Plug-in Martingales for Testing Exchangeability on-line".
            In Proceedings of the 29th ICML. Algorithm 1, page 3.
        """
        if not 0.0 < test_level < 1.0:
            raise ValueError("test_level must lie in (0, 1).")

        if test_method not in {"jumper_martingale", "plugin_martingale"}:
            raise ValueError(
                "test_method must be one of {'jumper_martingale', 'plugin_martingale'}."
            )

        if not 0.0 < jump_size < 1.0:
            raise ValueError("jump_size must lie in (0, 1).")

        self.mapie_estimator = self._prepare_estimator(mapie_estimator)
        self.task = task
        self.test_method = test_method
        self.test_level = test_level
        self.warn = warn

        self.jump_size = jump_size
        self.burn_in = burn_in
        self.rng = np.random.default_rng(random_state)

        self.pvalue_history: list[float] = []
        self.conformity_score_history: list[float] = []
        self.martingale_value_history: list[float] = []
        self.current_martingale_value = 1.0

        self._warning_already_raised = False

        self._jumper_expert_grid = np.array([-1.0, 0.0, 1.0], dtype=float)
        self._jumper_wealth_by_expert: NDArray[np.floating] = np.full(
            3, 1.0 / 3.0, dtype=float
        )

    @staticmethod
    def _prepare_estimator(
        mapie_estimator: Optional[MapieEstimator],
    ) -> Optional[MapieEstimator]:
        """Copy an estimator to avoid modifying the original estimator
        and clear conformalization state to allow calling conformalize method."""
        if mapie_estimator is None:
            return None

        if isinstance(
            mapie_estimator,
            (
                CrossConformalClassifier,
                CrossConformalRegressor,
                JackknifeAfterBootstrapRegressor,
            ),
        ):
            raise ValueError(
                "Cross conformal and jackknife-after-bootstrap estimators are not "
                "supported in permutation tests because they mix the data."
            )

        estimator_copy = deepcopy(mapie_estimator)

        if hasattr(estimator_copy, "_is_conformalized"):
            estimator_copy._is_conformalized = False
        if hasattr(estimator_copy, "_predict_params"):
            estimator_copy._predict_params = {}

        for attr_name in ("conformity_scores_", "quantiles_"):
            if hasattr(estimator_copy, attr_name):
                delattr(estimator_copy, attr_name)

        for inner_estimator_name in ("_mapie_regressor", "_mapie_classifier"):
            if hasattr(estimator_copy, inner_estimator_name):
                inner_estimator = getattr(estimator_copy, inner_estimator_name)
                for attr_name in ("conformity_scores_", "quantiles_"):
                    if hasattr(inner_estimator, attr_name):
                        delattr(inner_estimator, attr_name)

        return estimator_copy

    def _infer_task(self, y: NDArray) -> Literal["classification", "regression"]:
        """Infer whether the current data should use classification scores."""
        if self.mapie_estimator is not None:
            if hasattr(self.mapie_estimator, "_mapie_classifier"):
                return "classification"
            if hasattr(self.mapie_estimator, "_mapie_regressor"):
                return "regression"
            raise ValueError(
                "Unable to infer the task from the provided MAPIE estimator."
            )

        type_of_target_y = type_of_target(y)
        if "multiclass" in type_of_target_y or "binary" in type_of_target_y:
            return "classification"
        if "continuous" in type_of_target_y:
            return "regression"
        raise ValueError("Unknown type of target, please manually set the task type.")

    def _initiate_estimator(self) -> OnlineMartingaleTest:
        """Initiate a default MAPIE estimator based on the task type."""
        if self.task == "classification":
            self.mapie_estimator = SplitConformalClassifier(prefit=False)
        elif self.task == "regression":
            self.mapie_estimator = SplitConformalRegressor(prefit=False)
        else:
            raise ValueError("Unknown task type.")
        return self

    def _compute_non_conformity_scores(self, X: NDArray, y: NDArray) -> NDArray:
        """Compute non-conformity scores from features and labels.

        Parameters
        ----------
        X : NDArray
            Feature matrix.
        y : NDArray
            Target values.
        Returns
        -------
        NDArray
            Non-conformity scores associated with ``(X, y)``.
        """

        if self.task is None:
            self.task = self._infer_task(y)

        if self.mapie_estimator is None:
            self._initiate_estimator()

        assert self.mapie_estimator is not None

        if not self.mapie_estimator._is_fitted:
            X_train, X, y_train, y = train_test_split(
                X,
                y,
                test_size=0.7,
                shuffle=False,
            )
            warnings.warn(
                "The provided MAPIE estimator is not fitted."
                "Fitting it on a slice of the data to compute non-conformity scores."
                f"{X_train.shape[0]} observations will be used to fit the estimator."
            )
            self.mapie_estimator.fit(X_train, y_train)

        self.mapie_estimator.conformalize(X, y)  # compute scores internally

        if self.task == "classification":
            self.mapie_estimator = cast(SplitConformalClassifier, self.mapie_estimator)
            scores = cast(
                NDArray,
                self.mapie_estimator._mapie_classifier.conformity_scores_,
            )
        else:
            self.mapie_estimator = cast(SplitConformalRegressor, self.mapie_estimator)
            scores = cast(
                NDArray,
                self.mapie_estimator._mapie_regressor.conformity_scores_,
            )

        return scores

    @property
    def reject_threshold(self) -> float:
        """
        Return the martingale rejection threshold.

        Returns
        -------
        float
            Rejection threshold equal to ``1 / test_level``.
            Exchangeability is rejected when the martingale exceeds this threshold.
        """
        return 1.0 / self.test_level

    @property
    def is_exchangeable(self) -> Optional[bool]:
        """
        Return the current exchangeability decision based on the martingale process.

        The decision is based on the trajectory of martingale values compared to the
        rejection threshold (``1 / test_level``). The interpretation is:

        - ``False``: Exchangeability is rejected when the martingale exceeds the
          rejection threshold at least once.
        - ``True``: Failure to reject exchangeability when the martingale remains
          below the significance level (``test_level``) throughout the history.
        - ``None``: The test is currently inconclusive because insufficient
          observations have been processed (fewer than
          ``burn_in``).

        Returns
        -------
        Optional[bool]
            Exchangeability decision, or ``None`` if inconclusive.

        Notes
        -----
        This implementation uses a persistent stopping-rule interpretation:
        once the martingale has crossed the rejection threshold at any time,
        the decision remains ``False`` thereafter, even if the martingale later decreases.

        Therefore, ``True`` should be interpreted as "no rejection so far",
        not as evidence in favor of exchangeability.

        See Also
        --------
        reject_threshold : The rejection threshold for the martingale.
        """
        if len(self.pvalue_history) < self.burn_in:
            return None
        # The value 1 is arbitrary
        # The idea is to reject if there are some values above the rejection threshold, cf. [1], Figure 8.2, page 153.
        # Here, we choose we reject if there are 1 or more values above the rejection threshold
        # This can be tuned to be more or less conservative.
        if (
            len([x for x in self.martingale_value_history if x > self.reject_threshold])
            >= 1
            or self.current_martingale_value > self.reject_threshold
        ):
            return False
        else:
            return True

    def compute_p_value(
        self,
        current_conformity_score: float,
        conformity_score_history: NDArray,
    ) -> float:
        r"""
        Compute the conformal p-value associated with a new conformity score.

        The p-value is computed using only the past conformity scores, according
        to the empirical conformal formula:

        .. math::

            p_t = \frac{1 + \#\{i : s_i > s_t\} + U \cdot \#\{i : s_i = s_t\}}{n + 1}

        where :math:`s_t` is the current conformity score, :math:`s_i` are past
        scores, :math:`U \sim \mathrm{Uniform}(0, 1)` is a random tie-breaker, and
        :math:`n` is the number of past observations.

        Parameters
        ----------
        current_conformity_score : float
            Conformity score of the current observation.

        conformity_score_history : NDArray
            Array of past conformity scores used as reference.

        Returns
        -------
        float
            Conformal p-value in ``[0, 1]`` associated with the current score.
            Under the null hypothesis of exchangeability, this p-value is uniformly
            distributed on ``[0, 1]``.

        Notes
        -----
        When no past observations are available, a uniform random p-value is returned.
        Tie-breaking via random uniform sampling ensures valid p-values even when
        conformity scores have ties.

        References
        ----------
        .. [1] Angelopoulos, Barber, Bates (2026).
            "Theoretical Foundations of Conformal Prediction".
            arXiv preprint arXiv:2411.11824.
        .. [3] Fedorova, Gammerman, Nouretdinov, Vovk (2012).
            "Plug-in Martingales for Testing Exchangeability on-line".
            In Proceedings of the 29th ICML. Algorithm 1, page 3.
        """
        history = np.asarray(conformity_score_history, dtype=float)
        n = len(history)
        u = self.rng.uniform()

        if n == 0:
            return float(self.rng.uniform())

        n_greater: int = int(np.sum(history > current_conformity_score))
        n_equal: int = int(np.sum(history == current_conformity_score))

        return float((1.0 + n_greater + u * n_equal) / (n + 1.0))

    def _estimate_pvalues_density(self, pvalue: float) -> float:
        """
        Estimate the density of p-values on the unit interval using reflected KDE.

        This method computes a probability density estimate for p-values using
        reflected kernel density estimation (KDE) with reflection at the boundaries
        to reduce edge bias. The density is normalized over ``[0, 1]`` and
        regularized toward the uniform distribution to account for estimation
        uncertainty in small samples.

        This density estimate is used by the plug-in martingale update.

        Raises
        ------
        ValueError
            If ``pvalue`` is not in ``[0, 1]``.

        Parameters
        ----------
        pvalue : float
            Evaluation point in ``[0, 1]``.

        Returns
        -------
        float
            Estimated density value at ``pvalue``.

        Notes
        -----
        - The method applies Silverman's bandwidth selector for kernel density
          estimation.
        - A regularization factor that decreases with sample size is applied to
          smoothly transition from uniform density (small samples) to the KDE
          estimate (large samples).
        - Numerical stability is ensured through finite-value checks at multiple
          stages.

        References
        ----------
        .. [3] Fedorova, Gammerman, Nouretdinov, Vovk (2012).
            "Plug-in Martingales for Testing Exchangeability on-line".
            In Proceedings of the 29th ICML. Algorithm 1, page 3.
        """
        if not (isinstance(pvalue, float) and 0.0 <= pvalue <= 1.0):
            raise ValueError("pvalue must lie in [0, 1].")
        p_array = np.asarray(self.pvalue_history, dtype=float)
        reflected = np.concatenate([-p_array, p_array, 2.0 - p_array])

        try:
            kde = gaussian_kde(reflected, bw_method="silverman")
        except Exception:
            return 1.0

        grid = np.linspace(0.0, 1.0, 200)
        density_vals = kde(grid)
        normalization_val = trapezoid(density_vals, grid)

        if normalization_val <= 0 or not np.isfinite(normalization_val):
            return 1.0

        density = float(kde([pvalue])[0] / normalization_val)

        if not np.isfinite(density):
            return 1.0

        # Sample-size dependent regularization towards uniform density to control against estimation errors in small samples
        regularization_strength = min(1.0, self.burn_in / len(self.pvalue_history))
        density = (
            1.0 - regularization_strength
        ) * density + regularization_strength * 1.0

        return density

    def update_simple_jumper_martingale(self, pvalue: float) -> float:
        """
        Update the simple jumper martingale with a new p-value.

        The simple jumper martingale maintains a mixture of betting experts and
        updates their wealth sequentially according to the incoming p-values.

        Parameters
        ----------
        pvalue : float
            New conformal p-value in ``[0, 1]``.

        Returns
        -------
        float
            Updated martingale value.

        Raises
        ------
        ValueError
            If ``pvalue`` does not lie in ``[0, 1]``.

        Notes
        -----
        This martingale is generally more stable and less tuning-sensitive than the
        plug-in martingale, making it a suitable default choice in practice.

        References
        ----------
        .. [2] Vovk, Gammerman, Shafer (2005).
            "Algorithmic Learning in a Random World".
            Boston, MA: Springer US. Section 7.1, page 169.
        """
        if not (0.0 <= pvalue <= 1.0):
            raise ValueError("pvalue must lie in [0, 1].")

        m_prev = float(np.sum(self._jumper_wealth_by_expert))

        mixed_wealth = (1.0 - self.jump_size) * self._jumper_wealth_by_expert + (
            self.jump_size / 3.0
        ) * m_prev

        betting_multipliers = 1.0 + self._jumper_expert_grid * (pvalue - 0.5)
        self._jumper_wealth_by_expert = mixed_wealth * betting_multipliers

        self.current_martingale_value = float(np.sum(self._jumper_wealth_by_expert))
        self.martingale_value_history.append(self.current_martingale_value)

        return self.current_martingale_value

    def update_plugin_martingale(self, pvalue: float) -> float:
        """
        Update the plug-in martingale with a new p-value.

        The plug-in martingale multiplies the current martingale value by an estimate
        of the p-value density evaluated at the new p-value.

        Parameters
        ----------
        pvalue : float
            New conformal p-value in ``[0, 1]``.

        Returns
        -------
        float
            Updated martingale value.

        Raises
        ------
        ValueError
            If ``pvalue`` does not lie in ``[0, 1]``.

        Notes
        -----
        The plug-in martingale can be more adaptive than the jumper martingale,
        but is also more sensitive to density estimation choices and warm-up size.

        References
        ----------
        .. [3] Fedorova, Gammerman, Nouretdinov, Vovk (2012).
            "Plug-in Martingales for Testing Exchangeability on-line".
            In Proceedings of the 29th ICML. Algorithm 1, page 3.
        """
        rho_hat = self._estimate_pvalues_density(pvalue)
        self.current_martingale_value *= rho_hat
        self.martingale_value_history.append(self.current_martingale_value)
        return self.current_martingale_value

    @staticmethod
    def _to_1d_array(values: NDArray) -> NDArray:
        """
        Convert input values to a one-dimensional NumPy array.

        Parameters
        ----------
        values : NDArray
            Input array-like object.

        Returns
        -------
        NDArray
            Flattened one-dimensional NumPy array.
        """
        values = np.asarray(values)
        if values.ndim == 0:
            values = values.reshape(1)
        return values.reshape(-1)

    def update(
        self,
        X: NDArray,
        y: NDArray,
    ) -> OnlineMartingaleTest:
        """
        Update the online martingale test with newly labeled observations.

        This method computes conformity scores from the provided features and
        labels using the MAPIE conformalizer, converts them into conformal
        p-values using past history, updates the selected martingale, and
        appends the new observations to the internal state.

        Parameters
        ----------
        X : NDArray
            Feature matrix associated with the new observations.

        y : NDArray
            True labels associated with the new observations.

        Returns
        -------
        OnlineMartingaleTest
            Updated instance.

        Warns
        -----
        UserWarning
            If exchangeability is rejected and ``warn=True``.

        Notes
        -----
        This method can be used both to initialize the test on a labeled reference
        set and to update it online as new labels become available.
        """
        y = self._to_1d_array(y)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must have the same number of rows. "
                f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
            )

        scores = self._compute_non_conformity_scores(X, y)
        scores = self._to_1d_array(scores).astype(float)

        for current_score in scores:
            pvalue = self.compute_p_value(
                current_conformity_score=current_score,
                conformity_score_history=np.asarray(
                    self.conformity_score_history, dtype=float
                ),
            )

            if self.test_method == "jumper_martingale":
                self.update_simple_jumper_martingale(pvalue)
            elif self.test_method == "plugin_martingale":
                self.update_plugin_martingale(pvalue)
            else:
                raise ValueError(f"Unsupported test method: {self.test_method}")

            self.conformity_score_history.append(float(current_score))
            self.pvalue_history.append(float(pvalue))

        if (
            self.is_exchangeable is False
            and self.warn
            and not self._warning_already_raised
        ):
            n_crossings = int(
                np.sum(
                    np.asarray(self.martingale_value_history) > self.reject_threshold
                )
            )
            warnings.warn(
                "The online martingale test has rejected exchangeability. "
                f"The martingale exceeded the rejection threshold "
                f"{n_crossings} time(s), with threshold = {self.reject_threshold:.3g}.",
                UserWarning,
            )
            self._warning_already_raised = True

        return self

    def summary(self) -> dict:
        """
        Summarize the current state of the online martingale test.

        Returns
        -------
        dict
            Dictionary containing the current martingale value, exchangeability
            decision, rejection threshold, summary statistics of the martingale
            trajectory, and stopping-time information.

        Notes
        -----
        The returned summary is intended for diagnostics and monitoring.
        It does not modify the internal state of the test.

        The reported ``stopping_time`` is the first index at which the martingale
        exceeds the rejection threshold. If the martingale never exceeds the
        threshold, stopping_time is the index of the last martingale value.
        """
        martingale_values = np.asarray(self.martingale_value_history, dtype=float)

        if martingale_values.size == 0:
            return {
                "test_method": self.test_method,
                "burn_in": self.burn_in,
                "test_level": self.test_level,
                "is_exchangeable": self.is_exchangeable,
                "stopping_time": None,
                "martingale_value_at_decision": None,
                "last_martingale_value": float(self.current_martingale_value),
                "martingale_statistics": {
                    "min": None,
                    "q025": None,
                    "q25": None,
                    "median": None,
                    "mean": None,
                    "q75": None,
                    "q975": None,
                    "max": None,
                },
            }

        quantiles = np.asarray(
            np.quantile(
                martingale_values,
                [0.0, 0.025, 0.25, 0.5, 0.75, 0.975, 1.0],
            ),
            dtype=float,
        )

        above_threshold = martingale_values > self.reject_threshold

        # Find the first index where a value exceeds the rejection threshold
        threshold_crossing_indices = np.flatnonzero(above_threshold)

        if threshold_crossing_indices.size > 0:
            stopping_time = int(threshold_crossing_indices[0]) + 1
        else:
            stopping_time = int(martingale_values.size)

        return {
            "test_method": self.test_method,
            "burn_in": self.burn_in,
            "test_level": self.test_level,
            "is_exchangeable": self.is_exchangeable,
            "stopping_time": stopping_time,
            "martingale_value_at_decision": float(martingale_values[stopping_time - 1]),
            "last_martingale_value": float(self.current_martingale_value),
            "martingale_statistics": {
                "min": float(quantiles[0]),
                "q025": float(quantiles[1]),
                "q25": float(quantiles[2]),
                "median": float(quantiles[3]),
                "mean": float(np.mean(martingale_values)),
                "q75": float(quantiles[4]),
                "q975": float(quantiles[5]),
                "max": float(quantiles[6]),
            },
        }

from __future__ import annotations

import warnings
from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde


class OnlineMartingaleTest:
    """
    Online test of exchangeability based on conformal p-values and test martingales.

    OnlineMartingaleTest sequentially monitors whether newly observed labeled data
    remain exchangeable with respect to a reference stream, using non-conformity
    scores, conformal p-values, and a martingale-based evidence process.

    At each update, the class:

    1. computes non-conformity scores from observed labels and model predictions,
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
    non_conformity_score_function : Callable[[NDArray, NDArray, Optional[NDArray]], NDArray]
        Function used to compute non-conformity scores from observed labels,
        model predictions, and optionally covariates.
        It must accept ``(y_true, y_pred, X)`` and return an array-like of
        non-conformity scores of shape ``(n_samples,)``.

    test_method : {"jumper_martingale", "plugin_martingale"}, default="jumper_martingale"
        Martingale construction used to aggregate evidence across p-values.

    confidence_level : float, default=0.95
        Confidence level used to define the rejection threshold.
        Must lie in ``(0, 1)``.
        The corresponding test level is ``alpha_level = 1 - confidence_level``.

    raise_warning : bool, default=True
        Whether to raise a warning when exchangeability is rejected.

    jump_size : float, default=0.01
        Mixing parameter used by the jumper martingale.
        Ignored when ``test_method="plugin_martingale"``.

    min_history_for_pvalue : int, default=50
        Minimum number of past non-conformity scores required before computing
        empirical conformal p-values from history.
        Before this threshold, a neutral fallback value is used.

    min_history_for_density : int, default=50
        Minimum number of past p-values required before estimating a p-value
        density for the plug-in martingale.
        Before this threshold, a neutral density is used.

    min_history_to_decide : int, default=200
        Minimum number of p-values required before the ``is_exchangeable``
        property is allowed to return a non-``None`` decision.

    Attributes
    ----------
    pvalue_history : list of float
        History of conformal p-values observed so far.

    non_conformity_score_history : list of float
        History of non-conformity scores observed so far.

    martingale_value_history : list of float
        History of martingale values after each update.

    current_martingale_value : float
        Current value of the martingale process.

    Examples
    --------
    >>> etod = OnlineMartingaleTest(
    ...     non_conformity_score_function=binary_prob_nonconformity,
    ...     test_method="jumper_martingale",
    ...     confidence_level=0.95,
    ... )
    >>> etod.update(y_test, y_pred, X_test)
    >>> etod.is_exchangeable

    Notes
    -----
    The class is designed for sequential monitoring. It can be initialized on a
    reference labeled dataset using ``update`` and then updated online whenever
    new labels become available.

    The martingale provides a valid sequential test against exchangeability when
    the p-values are valid under the null hypothesis.
    """

    def __init__(
        self,
        non_conformity_score_function: Callable[
            [NDArray, NDArray, Optional[NDArray]], NDArray
        ],
        test_method: Literal[
            "jumper_martingale", "plugin_martingale"
        ] = "jumper_martingale",
        confidence_level: float = 0.95,
        raise_warning: bool = True,
        jump_size: float = 0.01,
        min_history_for_pvalue: int = 50,
        min_history_for_density: int = 50,
        min_history_to_decide: int = 100,
    ):
        """
        Initialize the online martingale test.

        Parameters
        ----------
        non_conformity_score_function : Callable[[NDArray, NDArray, Optional[NDArray]], NDArray]
            Function used to compute non-conformity scores from observed labels,
            predictions, and optionally features.

        test_method : {"jumper_martingale", "plugin_martingale"}, default="jumper_martingale"
            Martingale construction used to aggregate evidence from conformal p-values.

        confidence_level : float, default=0.95
            Confidence level used to define the rejection threshold.
            Must lie in ``(0, 1)``.

        raise_warning : bool, default=True
            Whether to raise a warning when the martingale rejects exchangeability.

        jump_size : float, default=0.01
            Jump parameter of the jumper martingale.
            Must lie in ``[0, 1]``.

        min_history_for_pvalue : int, default=50
            Minimum number of past non-conformity scores required before computing
            empirical p-values from history.

        min_history_for_density : int, default=50
            Minimum number of past p-values required before estimating a p-value
            density for the plug-in martingale.

        min_history_to_decide : int, default=200
            Minimum number of observations required before returning a non-``None``
            value from ``is_exchangeable``.

        Raises
        ------
        ValueError
            If ``confidence_level`` is not in ``(0, 1)``,
            if ``test_method`` is unsupported,
            or if ``jump_size`` is not in ``[0, 1]``.

        References
        ----------
        - Angelopoulos, Barber, Bates (2026),
        "Theoretical Foundations of Conformal Prediction",
        Definition 3.8.
        - Vovk, Gammerman, Shafer (2005),
        "Algorithmic Learning in a Random World",
        Section 7.1, page 169.
        - Fedorova, Gammerman, Nouretdinov, Vovk (2012),
        "Plug-in Martingales for Testing Exchangeability on-line",
        In precedings of the 29th ICML, 2012,
        Algorithm 1, page 3.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1).")

        if test_method not in {"jumper_martingale", "plugin_martingale"}:
            raise ValueError(
                "test_method must be one of {'jumper_martingale', 'plugin_martingale'}."
            )

        if not 0.0 <= jump_size <= 1.0:
            raise ValueError("jump_size must lie in [0, 1].")

        self.non_conformity_score_function = non_conformity_score_function
        self.test_method = test_method
        self.confidence_level = confidence_level
        self.raise_warning = raise_warning

        self.jump_size = jump_size
        self.min_history_for_pvalue = min_history_for_pvalue
        self.min_history_for_density = min_history_for_density
        self.min_history_to_decide = min_history_to_decide

        self.pvalue_history: list[float] = []
        self.non_conformity_score_history: list[float] = []
        self.martingale_value_history: list[float] = []
        self.current_martingale_value = 1.0

        self._warning_already_raised = False

        self._jumper_expert_grid = np.array([-1.0, 0.0, 1.0], dtype=float)
        self._jumper_wealth_by_expert = np.full(3, 1.0 / 3.0, dtype=float)

    @property
    def alpha_level(self) -> float:
        """
        Return the test level associated with the confidence level.

        Returns
        -------
        float
            Test level ``alpha_level = 1 - confidence_level``.
        """
        return 1.0 - self.confidence_level

    @property
    def reject_threshold(self) -> float:
        """
        Return the martingale rejection threshold.

        Returns
        -------
        float
            Rejection threshold equal to ``1 / alpha_level``.
            Exchangeability is rejected when the martingale exceeds this threshold.
        """
        return 1.0 / self.alpha_level

    @property
    def is_exchangeable(self) -> Optional[bool]:
        """
        Return the current exchangeability decision.

        The returned value has the following interpretation:

        - ``False``: exchangeability is rejected.
        - ``True``: failure to reject exchangeability.
        - ``None``: the test is currently inconclusive.

        Returns
        -------
        Optional[bool]
            Exchangeability decision.

        Notes
        -----
        This property is based solely on the martingale value and does not constitute
        evidence in favor of exchangeability in a strict hypothesis-testing sense.
        Therefore, ``True`` should be interpreted as "failure to reject
        exchangeability", not as proof of exchangeability.
        """
        if len(self.pvalue_history) < self.min_history_to_decide:
            return None

        if self.current_martingale_value > self.reject_threshold:
            return False

        if self.current_martingale_value < 1.0:
            return True

        return None

    def compute_p_value(
        self,
        current_non_conformity_score: float,
        non_conformity_score_history: NDArray,
    ) -> float:
        """
        Compute the conformal p-value associated with a new non-conformity score.

        The p-value is computed using only the past non-conformity scores, according
        to the empirical conformal formula:

        ``p_t = (1 + #{s_i >= s_t, i < t}) / (t + 1)``

        where ``s_t`` is the current score.

        Parameters
        ----------
        current_non_conformity_score : float
            Non-conformity score of the current observation.

        non_conformity_score_history : NDArray
            Array of past non-conformity scores used as reference.

        Returns
        -------
        float
            Conformal p-value associated with the current score.

        References
        ----------
        Angelopoulos, Barber, Bates (2026),
        "Theoretical Foundations of Conformal Prediction",
        Definition 3.8.
        """
        history = np.asarray(non_conformity_score_history, dtype=float)
        n = len(history)

        if n < self.min_history_for_pvalue:
            return 0.5

        n_geq = np.sum(history >= current_non_conformity_score)
        return float((1.0 + n_geq) / (n + 1.0))

    def _estimate_pvalues_density(self, pvalue: float) -> float:
        """
        Estimate the density of p-values on the unit interval.

        The density is estimated using reflected kernel density estimation (KDE)
        to reduce boundary bias near 0 and 1, and normalized over ``[0, 1]``.

        This estimate is used by the plug-in martingale.

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
        When insufficient p-value history is available, a neutral density equal to
        1 is returned, corresponding to the uniform distribution on ``[0, 1]``.

        References
        ----------
        Fedorova, Gammerman, Nouretdinov, Vovk (2012),
        "Plug-in Martingales for Testing Exchangeability on-line",
        In precedings of the 29th ICML, 2012.
        """
        if not 0.0 <= pvalue <= 1.0:
            return 0.0

        if len(self.pvalue_history) < self.min_history_for_density:
            return 1.0

        p_array = np.asarray(self.pvalue_history, dtype=float)

        if np.allclose(p_array, p_array[0]):
            return 1.0

        reflected = np.concatenate([-p_array, p_array, 2.0 - p_array])

        try:
            kde = gaussian_kde(reflected, bw_method="silverman")
        except Exception:
            return 1.0

        grid = np.linspace(0.0, 1.0, 200)
        density_vals = kde(grid)
        normalization_val = np.trapezoid(density_vals, grid)

        if normalization_val <= 0 or not np.isfinite(normalization_val):
            return 1.0

        density = float(kde([pvalue])[0] / normalization_val)

        if not np.isfinite(density):
            return 1.0

        return float(np.clip(density, 1e-3, 10.0))

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
        Vovk, Gammerman, Shafer (2005),
        "Algorithmic Learning in a Random World",
        Section 7.1, page 169.
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
        Fedorova, Gammerman, Nouretdinov, Vovk (2012),
        "Plug-in Martingales for Testing Exchangeability on-line",
        In precedings of the 29th International Conference on Machine Learning (ICML 2012),
        Algorithm 1, page 3.
        """
        if not (0.0 <= pvalue <= 1.0):
            raise ValueError("pvalue must lie in [0, 1].")

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
        y_true: NDArray,
        y_pred: NDArray,
        X: Optional[NDArray] = None,
    ) -> OnlineMartingaleTest:
        """
        Update the online martingale test with newly labeled observations.

        This method computes non-conformity scores from the provided labels and
        predictions, converts them into conformal p-values using past history,
        updates the selected martingale, and appends the new observations to the
        internal state.

        Parameters
        ----------
        y_true : NDArray
            True labels associated with the new observations.

        y_pred : NDArray
            Model predictions associated with the new observations.
            For binary classification, this is typically the predicted probability
            of the positive class.

        X : Optional[NDArray], default=None
            Optional features associated with the new observations.
            Used only if required by the non-conformity score function.

        Returns
        -------
        OnlineMartingaleTest
            Updated instance.

        Warns
        -----
        UserWarning
            If exchangeability is rejected and ``raise_warning=True``.

        Notes
        -----
        This method can be used both to initialize the test on a labeled reference
        set and to update it online as new labels become available.
        """
        scores = self.non_conformity_score_function(y_true, y_pred, X)
        scores = self._to_1d_array(scores).astype(float)

        for current_score in scores:
            pvalue = self.compute_p_value(
                current_non_conformity_score=current_score,
                non_conformity_score_history=np.asarray(
                    self.non_conformity_score_history, dtype=float
                ),
            )

            if self.test_method == "jumper_martingale":
                self.update_simple_jumper_martingale(pvalue)
            elif self.test_method == "plugin_martingale":
                self.update_plugin_martingale(pvalue)
            else:
                raise ValueError(f"Unsupported test method: {self.test_method}")

            self.non_conformity_score_history.append(float(current_score))
            self.pvalue_history.append(float(pvalue))

        if (
            self.is_exchangeable is False
            and self.raise_warning
            and not self._warning_already_raised
        ):
            warnings.warn(
                "The online martingale test has rejected exchangeability. "
                f"Martingale value = {self.current_martingale_value:.3g} "
                f"exceeds threshold = {self.reject_threshold:.3g}.",
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
            decision, number of processed observations, rejection threshold,
            summary statistics of the martingale trajectory, and stopping-time
            information when available.

        Notes
        -----
        The returned summary is intended for diagnostics and monitoring.
        It does not modify the internal state of the test.
        """
        martingale_values = np.asarray(self.martingale_value_history, dtype=float)

        if martingale_values.size == 0:
            return {
                "test_method": self.test_method,
                "confidence_level": self.confidence_level,
                "alpha_level": self.alpha_level,
                "reject_threshold": self.reject_threshold,
                "current_martingale_value": float(self.current_martingale_value),
                "is_exchangeable": self.is_exchangeable,
                "n_non_conformity_scores": len(self.non_conformity_score_history),
                "n_pvalues": len(self.pvalue_history),
                "stopping_time": None,
                "first_rejection_index": None,
                "first_acceptance_index": None,
            }

        quantiles = np.quantile(
            martingale_values, [0.0, 0.025, 0.25, 0.5, 0.75, 0.975, 1.0]
        )

        first_rejection_index = next(
            (
                i + 1
                for i, value in enumerate(martingale_values)
                if value > self.reject_threshold
            ),
            None,
        )
        first_acceptance_index = next(
            (
                i + 1
                for i, value in enumerate(martingale_values)
                if value < self.alpha_level
            ),
            None,
        )

        if first_rejection_index is not None and first_acceptance_index is not None:
            stopping_time = min(first_rejection_index, first_acceptance_index)
        else:
            stopping_time = first_rejection_index or first_acceptance_index

        return {
            "test_method": self.test_method,
            "confidence_level": self.confidence_level,
            "alpha_level": self.alpha_level,
            "reject_threshold": self.reject_threshold,
            "current_martingale_value": float(self.current_martingale_value),
            "is_exchangeable": self.is_exchangeable,
            "n_non_conformity_scores": len(self.non_conformity_score_history),
            "n_pvalues": len(self.pvalue_history),
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
            "stopping_time": stopping_time,
            "first_rejection_index": first_rejection_index,
            "first_acceptance_index": first_acceptance_index,
            "last_observation_index": int(martingale_values.size),
        }

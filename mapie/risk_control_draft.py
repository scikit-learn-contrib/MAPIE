from typing import Any, List, Tuple, cast, Optional, Union

import numpy as np
from scipy.stats import binom
from sklearn.pipeline import Pipeline
from numpy._typing import ArrayLike, NDArray
from sklearn.utils import check_random_state

from mapie.utils import _check_n_jobs, _check_verbose


class BinaryClassificationController:
    # TODO : test that this is working with a sklearn pipeline
    """
    fitted_classifier: Any
    Any object that provides predict and predict_proba methods.
    """

    def __init__(
        self,
        fitted_binary_classifier: Any,
        metric: "str",
        target_level: float,
        confidence_level: float = 0.9,
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ):
        _check_n_jobs(n_jobs)
        _check_verbose(verbose)
        check_random_state(random_state)

        self._classifier = fitted_binary_classifier
        self._alpha = 1-target_level
        self._delta = 1-confidence_level
        self._n_jobs = n_jobs # TODO : use this in the class
        self._random_state = random_state
        self._verbose = verbose # TODO : use this in the class

        self._thresholds: NDArray = np.arange(0, 1, 0.01)

        self.valid_thresholds: Optional[NDArray] = None
        self.best_threshold: Optional[float] = None

    def calibrate(self, X_calibrate: ArrayLike, y_calibrate: ArrayLike):
        X_cal = cast(NDArray, X_calibrate)
        y_cal = cast(NDArray, y_calibrate)

        predictions_proba = self._classifier.predict_proba(X_cal)[:,
        1]  # TODO : use _transform_pred_proba or a function adapted to binary

        risk_per_threshold = 1 - self._compute_precision(predictions_proba, y_cal)

        # TODO : remove the following, only relevant for upskilling day
        self.precision_per_threshold = 1 - risk_per_threshold

        valid_thresholds_index, _ = ltt_procedure(
            risk_per_threshold,
            np.array([self._alpha]),
            self._delta,
            len(y_cal),
            True
        )
        self.valid_thresholds = self._thresholds[valid_thresholds_index[0]]
        if len(self.valid_thresholds) == 0:
            raise ValueError("No valid thresholds found")

        # Minimum in case of precision control only
        self.best_threshold = min(self.valid_thresholds)


    def predict(self, X_test: ArrayLike):
        predictions_proba = self._classifier.predict_proba(X_test)[:, 1]
        return (predictions_proba >= self.best_threshold).astype(int)


    def _compute_precision(self, predictions_proba, y_cal):
        """
        Compute the precision for each threshold.
        """
        predictions_per_threshold = (
            predictions_proba[:, np.newaxis] >= self._thresholds
        ).astype(int)

        true_positives = np.sum(
            (predictions_per_threshold == 1) & (y_cal[:, np.newaxis] == 1),
            axis=0
        )
        false_positives = np.sum(
            (predictions_per_threshold == 1) & (y_cal[:, np.newaxis] == 0),
            axis=0
        )

        positive_predictions = true_positives + false_positives

        # Avoid division by zero
        precision_per_threshold = np.ones_like(self._thresholds, dtype=float)
        nonzero_mask = positive_predictions > 0
        precision_per_threshold[nonzero_mask] = (
            true_positives[nonzero_mask] / positive_predictions[nonzero_mask]
            )

        return precision_per_threshold


def ltt_procedure(
    r_hat: NDArray,
    alpha_np: NDArray,
    delta: Optional[float],
    n_obs: int,
    binary: bool = False # TODO : probably should pass p_values fonction instead
) -> Tuple[List[List[Any]], NDArray]:
    """
    Apply the Learn-Then-Test procedure for risk control.
    Note that we will do a multiple test for ``r_hat`` that are
    less than level ``alpha_np``.
    The procedure follows the instructions in [1]:
        - Calculate p-values for each lambdas descretized
        - Apply a family wise error rate algorithm,
        here Bonferonni correction
        - Return the index lambdas that give you the control
        at alpha level

    Parameters
    ----------
    r_hat: NDArray of shape (n_lambdas, ).
        Empirical risk with respect
        to the lambdas.
        Here lambdas are thresholds that impact decision making,
        therefore empirical risk.

    alpha_np: NDArray of shape (n_alpha, ).
        Contains the different alphas control level.
        The empirical risk should be less than alpha with
        probability 1-delta.

    delta: float.
        Probability of not controlling empirical risk.
        Correspond to proportion of failure we don't
        want to exceed.

    Returns
    -------
    valid_index: List[List[Any]].
        Contain the valid index that satisfy fwer control
        for each alpha (length aren't the same for each alpha).

    p_values: NDArray of shape (n_lambda, n_alpha).
        Contains the values of p_value for different alpha.

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".
    """
    if delta is None:
        raise ValueError(
            "Invalid delta: delta cannot be None while"
            + " controlling precision with LTT. "
        )
    p_values = compute_hoeffdding_bentkus_p_value(r_hat, n_obs, alpha_np, binary)
    N = len(p_values)
    valid_index = []
    for i in range(len(alpha_np)):
        l_index = np.where(p_values[:, i] <= delta/N)[0].tolist()
        valid_index.append(l_index)
    return valid_index, p_values # TODO : p_values is not used, we could remove it
    # Or return corrected p_values

def compute_hoeffdding_bentkus_p_value(
    r_hat: NDArray,
    n_obs: int,
    alpha: Union[float, NDArray],
    binary: bool = False
) -> NDArray:
    """
    The method computes the p_values according to
    the Hoeffding_Bentkus inequality for each
    alpha.
    We return the minimum between the Hoeffding and
    Bentkus p-values (Note that it depends on
    scipy.stats). The p_value is introduced in
    learn then test paper [1].

    Parameters
    ----------
    r_hat: NDArray of shape (n_lambdas, )
        Empirical risk with respect
        to the lambdas.
        Here lambdas are thresholds that impact decision
        making and therefore empirical risk.

    n_obs: int.
        Correspond to the number of observations in
        dataset.

    alpha: Union[float, Iterable[float]].
        Contains the different alphas control level.
        The empirical risk must be less than alpha.
        If it is a iterable, it is a NDArray of shape
        (n_alpha, ).

    Returns
    -------
    hb_p_values: NDArray of shape (n_lambda, n_alpha).

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".
    """
    # Should we cast again? We're deep in the code here, should have been done earlier.
    alpha_np = cast(NDArray, alpha)
    alpha_np = alpha_np[:, np.newaxis]
    r_hat_repeat = np.repeat(
        np.expand_dims(r_hat, axis=1),
        len(alpha_np),
        axis=1
    )
    alpha_repeat = np.repeat(
        alpha_np.reshape(1, -1),
        len(r_hat),
        axis=0
    )
    hoeffding_p_value = np.exp(
        -n_obs * _h1(
            np.where(  # TODO : shouldn't we use np.minimum ?
                r_hat_repeat > alpha_repeat,
                alpha_repeat,
                r_hat_repeat
            ),
            alpha_repeat
        )
    )
    factor = 1 if binary else np.e
    bentkus_p_value = factor * binom.cdf(
        np.ceil(n_obs * r_hat_repeat), n_obs, alpha_repeat
    )
    hb_p_value = np.where(  # TODO : shouldn't we use np.minimum ?
        bentkus_p_value > hoeffding_p_value,
        hoeffding_p_value,
        bentkus_p_value
    )
    return hb_p_value


def _h1(
    r_hats: NDArray,
    alphas: NDArray
) -> NDArray:
    """
    This function allow us to compute
    the tighter version of hoeffding inequality.
    This function is then used in the
    hoeffding_bentkus_p_value function for the
    computation of p-values.

    Parameters
    ----------
    r_hats: NDArray of shape (n_lambdas, n_alpha).
        Empirical risk with respect
        to the lambdas.
        Here lambdas are thresholds that impact decision
        making and therefore empirical risk.
        The value table has an extended dimension of
        shape (n_lambda, n_alpha).

    alphas: NDArray of shape (n_lambdas, n_alpha).
        Contains the different alphas control level.
        In other words, empirical risk must be less
        than each alpha in alphas.
        The value table has an extended dimension of
        shape (n_lambda, n_alpha).

    Returns
    -------
    NDArray of shape a(n_lambdas, n_alpha).
    """
    elt1 = r_hats * np.log(r_hats/alphas)
    elt2 = (1-r_hats) * np.log((1-r_hats)/(1-alphas))
    return elt1 + elt2

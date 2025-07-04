from typing import Any, List, Optional, Tuple, Union

import numpy as np
from numpy._typing import ArrayLike, NDArray
from scipy.stats import binom
from sklearn.utils import check_random_state

from mapie.utils import _check_n_jobs, _check_verbose

# General TODOs:
# TODO: maybe use type float instead of float32?
# TODO : in calibration and prediction,
#  use _transform_pred_proba or a function adapted to binary
# to get the probabilities depending on the classifier


class BinaryClassificationController:
    # TODO : test that this is working with a sklearn pipeline
    # TODO : test that this is working with a pandas dataframes
    """
    Controller for the calibration of our binary classifier.

    Parameters
    ----------
    fitted_binary_classifier: Any
        Any object that provides `predict_proba` and `predict` methods.

    metric: str
        The performance metric we want to control (ex: "precision")

    target_level: float
        The target performance level we want to achieve (ex: 0.8)

    confidence_level: float
        The maximum acceptable probability of the precision falling below the
        target precision level (ex: 0.8)

    Attributes
    ----------
    precision_per_threshold: NDArray
        Precision of the binary classifier on the calibration set for each
        threshold from self._thresholds.

    valid_threshold: NDArray
        Thresholds that meet the target precision with the desired confidence.

    best_threshold: float
        Valid threshold that maximizes the recall, i.e. the smallest valid
        threshold.
    """

    def __init__(
        self,
        fitted_binary_classifier: Any,
        metric: str,
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
        self._alpha = 1 - target_level
        self._delta = 1 - confidence_level
        self._n_jobs = n_jobs  # TODO : use this in the class or delete
        self._random_state = random_state  # TODO : use this in the class or delete
        self._verbose = verbose  # TODO : use this in the class or delete

        self._thresholds: NDArray[np.float32] = np.arange(0, 1, 0.01)
        # TODO: add a _is_calibrated attribute to check at prediction time

        self.valid_thresholds: Optional[NDArray[np.float32]] = None
        self.best_threshold: Optional[float] = None

    def calibrate(self, X_calibrate: ArrayLike, y_calibrate: ArrayLike) -> None:
        """
        Find the threshold that statistically guarantees the desired precision
        level while maximizing the recall.

        Parameters
        ----------
        X_calibrate: ArrayLike
            Features of the calibration set.

        y_calibrate: ArrayLike
            True labels of the calibration set.

        Raises
        ------
        ValueError
            If no thresholds that meet the target precision with the desired
            confidence level are found.
        """
        y_calibrate_ = np.asarray(y_calibrate)

        predictions_proba = self._classifier.predict_proba(X_calibrate)[:, 1]

        risk_per_threshold = 1 - self._compute_precision(
            predictions_proba, y_calibrate_
        )

        valid_thresholds_index, _ = ltt_procedure(
            risk_per_threshold,
            np.array([self._alpha]),
            self._delta,
            len(y_calibrate_),
            True,
        )
        self.valid_thresholds = self._thresholds[valid_thresholds_index[0]]
        if len(self.valid_thresholds) == 0:
            # TODO: just warn, and raise error at prediction if no valid thresholds
            raise ValueError("No valid thresholds found")

        # Minimum in case of precision control only
        self.best_threshold = min(self.valid_thresholds)

    def predict(self, X_test: ArrayLike) -> NDArray:
        """
        Predict binary labels on the test set, using the best threshold found
        during calibration.

        Parameters
        ----------
        X_test: ArrayLike
            Features of the test set.

        Returns
        -------
        ArrayLike
            Predicted labels (0 or 1) for each sample in the test set.
        """
        predictions_proba = self._classifier.predict_proba(X_test)[:, 1]
        return (predictions_proba >= self.best_threshold).astype(int)

    def _compute_precision(
        self, predictions_proba: NDArray[np.float32], y_cal: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Compute the precision for each threshold.
        """
        predictions_per_threshold = (
            predictions_proba[:, np.newaxis] >= self._thresholds
        ).astype(int)

        true_positives = np.sum(
            (predictions_per_threshold == 1) & (y_cal[:, np.newaxis] == 1),
            axis=0,
        )
        false_positives = np.sum(
            (predictions_per_threshold == 1) & (y_cal[:, np.newaxis] == 0),
            axis=0,
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
    r_hat: NDArray[np.float32],
    alpha_np: NDArray[np.float32],
    delta: Optional[float],
    n_obs: int,
    binary: bool = False,  # TODO: maybe should pass p_values fonction instead
) -> Tuple[List[List[Any]], NDArray[np.float32]]:
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
    r_hat: NDArray[np.float32],
    n_obs: int,
    alpha: Union[float, NDArray[np.float32]],
    binary: bool = False,
) -> NDArray[np.float32]:
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
    # TODO: We shouldn't have to transform alpha to a nparray so deep in the code
    alpha_np = np.asarray(alpha)
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
    r_hats: NDArray[np.float32], alphas: NDArray[np.float32]
) -> NDArray[np.float32]:
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
    elt1 = np.zeros_like(r_hats, dtype=float)

    # Compute only where r_hats != 0 to avoid log(0)
    # TODO: check Angelopoulos implementation
    mask = r_hats != 0
    elt1[mask] = r_hats[mask] * np.log(r_hats[mask] / alphas[mask])
    elt2 = (1 - r_hats) * np.log((1 - r_hats) / (1 - alphas))
    return elt1 + elt2

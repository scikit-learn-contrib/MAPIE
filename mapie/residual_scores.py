import sys

import numpy as np


class ResidualScore:
    def __init__(self, sym: bool, eps: float = 10 * sys.float_info.epsilon):
        self.sym = sym
        self.eps = eps

    def get_signed_residual_scores(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        return np.array([])

    def get_residual_scores(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        residuals = self.get_signed_residual_scores(y, y_pred)
        if self.sym:
            residuals = np.abs(residuals)
        return residuals

    def get_observed_value(
        self, y_pred: np.ndarray, residual_scores: np.ndarray
    ) -> np.ndarray:
        return np.array([])

    def check_consistency(self, y: np.ndarray, y_pred: np.ndarray):
        residual_scores = self.get_signed_residual_scores(y, y_pred)
        if (
            np.abs(self.get_observed_value(y_pred, residual_scores) - y) > self.eps
        ).any():
            raise ValueError(
                "The two functions get_residual_score and get_observed_value "
                "of the ResidualScore class are not consistent. "
                "The following equation must be verified: "
                "self.get_observed_value(y_pred, self.get_residual_score(y, y_pred)) == y."  # noqa: E501
            )


class AbsoluteResidualScore(ResidualScore):
    def __init__(self, eps: float = 10 * sys.float_info.epsilon):
        ResidualScore.__init__(self, True, eps)

    def get_signed_residual_scores(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        return y - y_pred

    def get_observed_value(
        self, y_pred: np.ndarray, residual_scores: np.ndarray
    ) -> np.ndarray:
        return y_pred + residual_scores


class GammaResidualScore(ResidualScore):
    def __init__(self, eps: float = 10 * sys.float_info.epsilon):
        ResidualScore.__init__(self, False, eps)

    def get_signed_residual_scores(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        return (y - y_pred) / y_pred

    def get_observed_value(
        self, y_pred: np.ndarray, residual_scores: np.ndarray
    ) -> np.ndarray:
        return y_pred * (1 + residual_scores)

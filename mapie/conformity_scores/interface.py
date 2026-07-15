from abc import ABCMeta, abstractmethod
from typing import Optional

from sklearn.base import BaseEstimator

from numpy.typing import NDArray


class BaseConformityScore(metaclass=ABCMeta):
    """
    Base class for conformity scores.

    This class should not be used directly. Use derived classes instead.
    """

    def __init__(self) -> None:
        pass

    def set_external_attributes(self, **kwargs) -> None:
        """
        Set attributes that are not provided by the user.

        Must be overloaded by subclasses if necessary to add more attributes,
        particularly when the attributes are known after the object has been
        instantiated.
        """

    def set_ref_predictor(self, predictor: BaseEstimator):
        """
        Set the reference predictor.

        Parameters
        ----------
        predictor: BaseEstimator
            Reference predictor.
        """
        self.predictor = predictor

    def split_data(
        self,
        X: NDArray,
        y: NDArray,
        y_enc: NDArray,
        sample_weight: Optional[NDArray] = None,
        groups: Optional[NDArray] = None,
    ):
        """
        Split data. Keeps part of the data for the calibration estimator
        (separate from the calibration data).

        Parameters
        ----------
        *args: Tuple of NDArray

        Returns
        -------
        Tuple of NDArray
            Split data for training and calibration.
        """
        self.n_samples_ = len(X)
        return X, y, y_enc, sample_weight, groups

    @abstractmethod
    def get_conformity_scores(self, y: NDArray, y_pred: NDArray, **kwargs) -> NDArray:
        """
        Placeholder for `get_conformity_scores`.
        Subclasses should implement this method!

        Compute the sample conformity scores given the predicted and
        observed targets.

        Parameters
        ----------
        y: NDArray of shape (n_samples,)
            Observed target values.

        y_pred: NDArray of shape (n_samples,)
            Predicted target values.

        Returns
        -------
        NDArray of shape (n_samples,)
            Conformity scores.
        """

    @abstractmethod
    def predict_set(self, X: NDArray, alpha_np: NDArray, **kwargs):
        """
        Compute the prediction sets on new samples based on the uncertainty of
        the target confidence set.

        Parameters:
        -----------
        X: NDArray of shape (n_samples,)
            The input data or samples for prediction.

        alpha_np: NDArray of shape (n_alpha, )
            Represents the uncertainty of the confidence set to produce.

        **kwargs: dict
            Additional keyword arguments.

        Returns:
        --------
        result
            The prediction sets for each sample and each alpha level.
            The output structure depends on the subclass.
        """

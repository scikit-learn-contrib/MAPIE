from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from sklearn.base import ClassifierMixin

from mapie._typing import ArrayLike, NDArray


class EnsembleEstimator(ClassifierMixin, metaclass=ABCMeta):
    """
    This class implements methods to handle the training and usage of the
    estimator. This estimator can be unique or composed by cross validated
    estimators.
    """

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_enc: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params
    ) -> EnsembleEstimator:
        """
        Fit the base estimator under the ``single_estimator_`` attribute.
        Fit all cross-validated estimator clones
        and rearrange them into a list, the ``estimators_`` attribute.
        Out-of-fold conformity scores are stored under
        the ``conformity_scores_`` attribute.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Input data.

        y: ArrayLike of shape (n_samples,)
            Input labels.

        y_enc: ArrayLike
            Target values as normalized encodings.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
            By default ``None``.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        EnsembleClassifier
            The estimator fitted.
        """

    @abstractmethod
    def predict(
        self,
        X: ArrayLike,
        alpha_np: ArrayLike = [],
        agg_scores: Any = None
    ) -> NDArray:
        """
        Predict target from X. It also computes the prediction per train sample
        for each test sample according to ``self.method``.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        alpha_np: ArrayLike of shape (n_alphas)
            Level of confidences.

        agg_scores: Optional[str]
            How to aggregate the scores output by the estimators on test data
            if a cross-validation strategy is used

        Returns
        -------
        NDArray
            Predictions of shape
            (n_samples, n_classes)
        """

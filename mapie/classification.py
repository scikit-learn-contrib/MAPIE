from __future__ import annotations
from typing import Optional, Union, Tuple, Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer

from ._typing import ArrayLike
from ._machine_precision import EPSILON
from .utils import (
    check_null_weight,
    check_n_features_in,
    check_alpha,
    check_alpha_and_n_samples,
    check_n_jobs,
    check_verbose
)


class MapieClassifier(BaseEstimator, ClassifierMixin):  # type: ignore
    """
    Prediction sets for classification.

    This class implements several conformal prediction strategies for
    estimating prediction sets for classification. Instead of giving a
    single predicted label, the idea is to give a set of predicted labels
    (or prediction sets) which come with mathematically guaranteed coverages.

    Parameters
    ----------
    estimator : Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default None.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    method: Optional[str]
        Method to choose for prediction interval estimates.
        Choose among:

        - "score", based on the the scores
          (i.e. 1 minus the softmax score of the true label)
          on the calibration set.
        - "cumulated_score", based on the sum of the softmax outputs of the
          labels until the true label is reached, on the calibration set.

          By default "score".

    cv: Optional[str]
        The cross-validation strategy for computing scores :

        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          to calibrate the predictions through the score computation.
          At prediction time, quantiles of these scores are used to estimate
          prediction sets.

        By default ``prefit``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        At this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for ‘unset’ that will be interpreted as ``n_jobs=1``
        (sequential execution).

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        for evaluation quantiles and prediction sets in cumulated_score.
        Pass an int for reproducible output across multiple function calls.

        By default ```0``.

    verbose : int, optional
        The verbosity level, used with joblib for multiprocessing.
        At this moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    valid_methods: List[str]
        List of all valid methods.

    single_estimator_ : sklearn.ClassifierMixin
        Estimator fitted on the whole training set.

    n_features_in_: int
        Number of features passed to the fit method.

    n_samples_val_: Union[int, List[int]]
        Number of samples passed to the fit method.

    conformity_scores_ : np.ndarray of shape (n_samples_train)
        The conformity scores used to calibrate the prediction sets.

    quantiles_ : np.ndarray of shape (n_alpha)
        The quantiles estimated from ``conformity_scores_`` and alpha values.

    References
    ----------
    Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels",
    Journal of the American Statistical Association, 114, 2019.

    Yaniv Romano, Matteo Sesia and Emmanuel J. Candès.
    "Classification with Valid and Adaptive Coverage."
    NeurIPS 202 (spotlight).

    Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan
    and Jitendra Malik.
    "Uncertainty Sets for Image Classifiers using Conformal Prediction."
    International Conference on Learning Representations 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie.classification import MapieClassifier
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> clf = GaussianNB().fit(X_toy, y_toy)
    >>> mapie = MapieClassifier(estimator=clf, cv="prefit").fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.2)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False False]
     [ True False False]
     [ True False False]
     [ True  True False]
     [False  True False]
     [False  True  True]
     [False False  True]
     [False False  True]
     [False False  True]]
    """

    valid_methods_ = ["score", "cumulated_score"]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "score",
        cv: Optional[str] = "prefit",
        n_jobs: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        if self.method not in self.valid_methods_:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'score' or 'cumulated_score'."
            )
        check_n_jobs(self.n_jobs)
        check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _check_estimator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        estimator: Optional[ClassifierMixin] = None,
    ) -> ClassifierMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LogisticRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        estimator : Optional[ClassifierMixin], optional
            Estimator to check, by default ``None``

        Returns
        -------
        ClassifierMixin
            The estimator itself or a default ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no fit, predict, nor predict_proba methods.

        NotFittedError
            If the estimator is not fitted and ``cv`` attribute is "prefit".
        """
        if estimator is None:
            return LogisticRegression(multi_class="multinomial").fit(X, y)
        if (
            not hasattr(estimator, "fit")
            and not hasattr(estimator, "predict")
            and not hasattr(estimator, "predict_proba")
        ):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )
        if self.cv == "prefit":
            if isinstance(self.estimator, Pipeline):
                check_is_fitted(self.estimator[-1])
            else:
                check_is_fitted(self.estimator)
        return estimator

    def _check_cv(
        self, cv: Optional[Union[int, str, BaseCrossValidator]] = None
    ) -> Optional[Union[float, str]]:
        """
        Check if cross-validator is ``None`` or ``"prefit"``.
        Else raise error.

        Parameters
        ----------
        cv : Optional[Union[int, str, BaseCrossValidator]], optional
            Cross-validator to check, by default ``None``.

        Returns
        -------
        Optional[Union[float, str]]
            'prefit' or None.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv is None:
            return "prefit"
        if cv == "prefit":
            return cv
        raise ValueError("Invalid cv argument." "Allowed value is 'prefit'.")

    def _check_include_last_label(
        self,
        include_last_label: Optional[Union[bool, str]]
    ) -> Optional[Union[bool, str]]:
        """
        Check if include_last_label is a boolean or a string.
        Else raise error.

        Parameters
        ----------
        include_last_label : Optional[Union[bool, str]]
            Whether or not to include last label in
            prediction sets for the "cumulated_score" method. Choose among:

            - False, does not include label whose cumulated score is just over
             the quantile.
            - True, includes label whose cumulated score is just over the
            quantile, unless there is only one label in the prediction set.
            - "randomized", randomly includes label whose cumulated score is
            just over the quantile based on the comparison of a uniform number
            and the difference between the cumulated score of the last label
            and the quantile.

        Returns
        -------
        Optional[Union[bool, str]]

        Raises
        ------
        ValueError
            "Invalid include_last_label argument. "
            "Should be a boolean or 'randomized'."
        """
        if (
            (not isinstance(include_last_label, bool)) and
            (not include_last_label == "randomized")
        ):
            raise ValueError(
                "Invalid include_last_label argument. "
                "Should be a boolean or 'randomized'."
            )
        else:
            return include_last_label

    def _check_proba_normalized(
        self,
        y_pred_proba: ArrayLike
    ) -> Optional[ArrayLike]:
        """
        Check if, for all the observations, the sum of
        the probabilities is equal to one.

        Parameters
        ----------
        y_pred_proba : ArrayLike of shape (n_samples, n_classes)
            Softmax output of a model.

        Returns
        -------
        Optional[ArrayLike] of shape (n_samples, n_classes)
            Softmax output of a model if the scores all sum
            to one.

        Raises
        ------
            ValueError
            If the sum of the scores is not equal to one.
        """
        np.testing.assert_allclose(
            np.sum(y_pred_proba, axis=1),
            1,
            err_msg="The sum of the scores is not equal to one."
        )
        return y_pred_proba

    def _get_last_index_included(
        self,
        y_pred_proba_cumsum: ArrayLike,
        include_last_label: Optional[Union[bool, str]]
    ) -> ArrayLike:
        """
        Return the index of the last included sorted probability
        depending if we included the first label over the quantile
        or not.

        Parameters
        ----------
        y_pred_proba_cumsum : ArrayLike of shape (n_samples, n_classes)
            Cumsumed probabilities in the original order.
        include_last_label : Union[bool, str]
            Whether or not include the last label. If 'randomized',
            the last label is included.

        Returns
        -------
        Optional[ArrayLike] of shape (n_samples, n_classes)
            Index of the last included sorted probability.
        """
        if (
            (include_last_label is True) or
            (include_last_label == 'randomized')
        ):
            y_pred_index_last = np.stack(
                [
                    np.argmin(
                        np.ma.masked_less(
                            y_pred_proba_cumsum,
                            quantile
                        ),
                        axis=1
                    )
                    for quantile in self.quantiles_
                ], axis=1
            )
        elif (include_last_label is False):
            y_pred_index_last = np.stack(
                [
                    np.argmax(
                        np.ma.masked_where(
                            y_pred_proba_cumsum > np.maximum(
                                quantile,
                                np.min(y_pred_proba_cumsum, axis=1) + EPSILON
                            ).reshape(-1, 1),
                            y_pred_proba_cumsum
                        ),
                        axis=1
                    )
                    for quantile in self.quantiles_
                ], axis=1
            )
        else:
            raise ValueError(
                "Invalid include_last_label argument. "
                "Should be a boolean or 'randomized'."
            )

        return y_pred_index_last

    def _add_random_tie_breaking(
        self,
        prediction_sets: ArrayLike,
        y_pred_index_last: ArrayLike,
        y_pred_proba_cumsum: ArrayLike,
        y_pred_proba_last: ArrayLike
    ) -> ArrayLike:
        """
        Randomly remove last label from prediction set based on the
        comparison between a random number and the difference between
        cumulated score of the last included label and the quantile.

        Parameters
        ----------
        prediction_sets : ArrayLike of shape (n_samples, n_classes, n_alpha)
            Prediction set for each observation and each alpha.
        y_pred_index_last : ArrayLike of shape (n_samples, n_alpha)
            Index of the last included label.
        y_pred_proba_cumsum : ArrayLike of shape (n_samples, n_classes)
            Cumsumed probability of the model in the original order.
        y_pred_proba_last : ArrayLike of shape (n_samples, n_alpha)
            Last included probability.

        Returns
        -------
        ArrayLike of shape (n_samples, n_classes, n_alpha)
            Updated version of prediction_sets with randomly removed
            labels.
        """
        # filter sorting probabilities with kept labels
        y_proba_last_cumsumed = np.stack(
            [
                np.squeeze(
                    np.take_along_axis(
                        y_pred_proba_cumsum,
                        y_pred_index_last[:, iq].reshape(-1, 1),
                        axis=1
                    )
                )
                for iq, _ in enumerate(self.quantiles_)
            ], axis=1
        )
        # compute V parameter from Romano+(2020)
        vs = np.stack(
            [
                (
                    y_proba_last_cumsumed[:, iq]
                    - quantile
                ) / y_pred_proba_last[:, iq]
                for iq, quantile in enumerate(self.quantiles_)
            ], axis=1,
        )
        # get random numbers for each observation and alpha value
        random_state = check_random_state(self.random_state)
        us = random_state.uniform(size=prediction_sets.shape[0])
        # remove last label from comparison between uniform number and V
        vs_less_than_us = vs < us[:, np.newaxis]
        np.put_along_axis(
            prediction_sets,
            y_pred_index_last[:, np.newaxis, :],
            vs_less_than_us[:, np.newaxis, :],
            axis=1
        )
        return prediction_sets

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> MapieClassifier:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.

        y : ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no prediction sets.

            By default None.

        Returns
        -------
        MapieClassifier
            The model itself.
        """
        # Checks
        self._check_parameters()
        cv = self._check_cv(self.cv)
        estimator = self._check_estimator(X, y, self.estimator)

        X, y = check_X_y(
            X, y, force_all_finite=False, dtype=["float64", "int", "object"]
        )
        assert type_of_target(y) == "multiclass"
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        # Work
        self.single_estimator_ = estimator
        y_pred_proba = self.single_estimator_.predict_proba(X)
        y_pred_proba = self._check_proba_normalized(y_pred_proba)
        self.n_samples_val_ = X.shape[0]
        if self.method == "score":
            self.conformity_scores_ = np.take_along_axis(
                1 - y_pred_proba, y.reshape(-1, 1), axis=1
            )
        elif self.method == "cumulated_score":
            encoder = LabelBinarizer().fit(y)
            y_true = encoder.transform(y)
            index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
            y_pred_proba_sorted = np.take_along_axis(
                y_pred_proba, index_sorted, axis=1
            )
            y_true_sorted = np.take_along_axis(y_true, index_sorted, axis=1)
            y_pred_proba_sorted_cumsum = np.cumsum(y_pred_proba_sorted, axis=1)
            cutoff = encoder.inverse_transform(y_true_sorted)
            self.conformity_scores_ = np.take_along_axis(
                y_pred_proba_sorted_cumsum, cutoff.reshape(-1, 1), axis=1
            )
            y_proba_true = np.take_along_axis(
                y_pred_proba, y.reshape(-1, 1), axis=1
            )
            random_state = check_random_state(self.random_state)
            u = random_state.uniform(size=len(y_pred_proba)).reshape(-1, 1)
            self.conformity_scores_ -= u*y_proba_true

        else:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'score' or 'cumulated_score'."
            )

        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        include_last_label: Optional[Union[bool, str]] = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prediction prediction sets on new samples based on target confidence
        interval.
        Prediction sets for a given ``alpha`` are deduced from :

        - quantiles of softmax scores ("score" method)
        - quantiles of cumulated scores ("cumulated_score" method)

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Test data.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``np.ndarray`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            sets.
            ``alpha`` is the complement of the target coverage level.
            By default ``None``.

        include_last_label: Optional[Union[bool, str]]
            Whether or not to include last label in
            prediction sets for the "cumulated_score" method. Choose among:

            - False, does not include label whose cumulated score is just over
             the quantile.
            - True, includes label whose cumulated score is just over the
            quantile, unless there is only one label in the prediction set.
            - "randomized", randomly includes label whose cumulated score is
            just over the quantile based on the comparison of a uniform number
            and the difference between the cumulated score of the last label
            and the quantile.
            By default ``True``.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]

        - np.ndarray of shape (n_samples,) if alpha is None.

        - Tuple[np.ndarray, np.ndarray] of shapes
        (n_samples,) and (n_samples, n_classes, n_alpha) if alpha is not None.
        """
        # Checks
        include_last_label = self._check_include_last_label(include_last_label)
        alpha_ = check_alpha(alpha)
        check_is_fitted(
            self,
            [
                "single_estimator_",
                "conformity_scores_",
                "n_features_in_",
                "n_samples_val_",
            ],
        )
        X = check_array(X, force_all_finite=False, dtype=["float64", "object"])
        y_pred = self.single_estimator_.predict(X)
        y_pred_proba = self.single_estimator_.predict_proba(X)
        y_pred_proba = self._check_proba_normalized(y_pred_proba)
        n = self.n_samples_val_
        if alpha_ is None:
            return np.array(y_pred)
        else:
            check_alpha_and_n_samples(alpha_, n)
            self.quantiles_ = np.stack([
                np.quantile(
                    self.conformity_scores_,
                    ((n + 1) * (1 - _alpha)) / n,
                    interpolation="higher"
                ) for _alpha in alpha_
            ])
            if self.method == "score":
                prediction_sets = np.stack(
                    [
                        y_pred_proba > 1 - quantile
                        for quantile in self.quantiles_
                    ],
                    axis=2,
                )
            elif self.method == "cumulated_score":
                # sort labels by decreasing probability
                index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
                # sort probabilities by decreasing order
                y_pred_proba_sorted = np.take_along_axis(
                    y_pred_proba, index_sorted, axis=1
                )
                # get sorted cumulated score
                y_pred_proba_sorted_cumsum = np.cumsum(
                    y_pred_proba_sorted, axis=1
                )
                # get cumulated score at their original position
                y_pred_proba_cumsum = np.take_along_axis(
                    y_pred_proba_sorted_cumsum,
                    np.argsort(index_sorted),
                    axis=1
                )
                # get index of the last included label
                y_pred_index_last = self._get_last_index_included(
                    y_pred_proba_cumsum,
                    include_last_label
                )
                # get the probability of the last included label
                y_pred_proba_last = np.stack(
                    [
                        np.squeeze(
                            np.take_along_axis(
                                y_pred_proba,
                                y_pred_index_last[:, iq].reshape(-1, 1),
                                axis=1
                            )
                        )
                        for iq, _ in enumerate(self.quantiles_)
                    ], axis=1
                )
                # get the prediction set by taking all probabilities above the
                # last one
                prediction_sets = np.stack(
                    [
                        np.ma.masked_greater_equal(
                            y_pred_proba,
                            y_pred_proba_last[:, iq].reshape(-1, 1) - EPSILON
                        ).mask
                        for iq, _ in enumerate(self.quantiles_)
                    ], axis=2
                )
                # remove last label randomly
                if include_last_label == 'randomized':
                    prediction_sets = self._add_random_tie_breaking(
                        prediction_sets,
                        y_pred_index_last,
                        y_pred_proba_cumsum,
                        y_pred_proba_last
                    )
            else:
                raise ValueError(
                    "Invalid method. "
                    "Allowed values are 'score' or 'cumulated_score'."
                )

            return y_pred, prediction_sets

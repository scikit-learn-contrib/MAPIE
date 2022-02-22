from __future__ import annotations
from typing import Optional, Union, Tuple, Iterable, List

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state, _safe_indexing
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import label_binarize

from ._typing import ArrayLike
from ._machine_precision import EPSILON
from .utils import (
    check_cv,
    check_null_weight,
    check_n_features_in,
    check_alpha,
    check_alpha_and_n_samples,
    check_n_jobs,
    check_verbose,
    check_input_is_image,
    fit_estimator
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

        - "naive", sum of the probabilities until the 1-alpha thresold.

        - "score", based on the the scores
          (i.e. 1 minus the softmax score of the true label)
          on the calibration set.

        - "cumulated_score", based on the sum of the softmax outputs of the
          labels until the true label is reached, on the calibration set.

        - "top_k", based on the sorted index of the probability of the true
          label in the softmax outputs, on the calibration set. In case two
          probabilities are equal, both are taken, thus, the size of some
          prediction sets may be different from the others.

        By default "score".

    cv: Optional[str]
        The cross-validation strategy for computing scores :

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
            - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
            - ``sklearn.model_selection.KFold`` (cross-validation)
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          to calibrate the predictions through the score computation.
          At prediction time, quantiles of these scores are used to estimate
          prediction sets.

        By default ``None``.

    n_jobs: Optional[int]
        Number of jobs for parallel processing using joblib
        via the "locky" backend.
        At this moment, parallel processing is disabled.
        If ``-1`` all CPUs are used.
        If ``1`` is given, no parallel computing code is used at all,
        which is useful for debugging.
        For n_jobs below ``-1``, ``(n_cpus + 1 + n_jobs)`` are used.
        None is a marker for `unset` that will be interpreted as ``n_jobs=1``
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

    conformity_scores_ : ArrayLike of shape (n_samples_train)
        The conformity scores used to calibrate the prediction sets.

    quantiles_ : ArrayLike of shape (n_alpha)
        The quantiles estimated from ``conformity_scores_`` and alpha values.

    References
    ----------
    Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.",
    Journal of the American Statistical Association, 114, 2019.

    Yaniv Romano, Matteo Sesia and Emmanuel J. Candès.
    "Classification with Valid and Adaptive Coverage."
    NeurIPS 202 (spotlight) 2020.

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

    valid_methods_ = ["naive", "score", "cumulated_score", "top_k"]
    fit_attributes = [
        "single_estimator_",
        "estimators_",
        "n_features_in_",
        "n_samples_val_",
        "conformity_scores_"
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        method: str = "score",
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
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
            if not self.image_input:
                return LogisticRegression(multi_class="multinomial").fit(X, y)
            else:
                raise ValueError(
                    "Default LogisticRegression's input can't be an image."
                    "Please provide a proper model."
                )
        if isinstance(estimator, Pipeline):
            est = estimator[-1]
        else:
            est = estimator
        if (
            not hasattr(est, "fit")
            and not hasattr(est, "predict")
            and not hasattr(est, "predict_proba")
        ):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )
        if self.cv == "prefit":
            check_is_fitted(est)
            if not hasattr(est, "classes_"):
                raise AttributeError(
                    "Invalid classifier. "
                    "Fitted classifier does not contain "
                    "'classes_' attribute."
                )
        return estimator

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
        y_pred_proba: ArrayLike,
        axis: Optional[int] = 1
    ) -> Optional[ArrayLike]:
        """
        Check if, for all the observations, the sum of
        the probabilities is equal to one.

        Parameters
        ----------
        y_pred_proba : ArrayLike of shape
            (n_samples, n_classes) or
            (n_samples, n_train_samples, n_classes)
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
            np.sum(y_pred_proba, axis=axis),
            1,
            err_msg="The sum of the scores is not equal to one.",
            rtol=1e-5
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
        # get cumsumed probabilities up to last retained label
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
                ) / np.squeeze(y_pred_proba_last[:, :, iq])
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

    def _fix_number_of_classes(
        self,
        n_classes_training: ArrayLike,
        y_proba: ArrayLike
    ) -> ArrayLike:
        """
        Fix shape of y_proba of validation set if number of classes
        of the training set used for cross-validation is different than
        number of classes of the original dataset y.

        Parameters
        ----------
        n_classes_training : ArrayLike
            Classes of the training set.
        y_proba : ArrayLike
            Probabilities of the validation set.

        Returns
        -------
        ArrayLike
            Probabilities with the right number of classes.
        """
        y_pred_full = np.zeros(
            shape=(len(y_proba), self.n_classes_)
        )
        y_index = np.tile(n_classes_training, (len(y_proba), 1))
        np.put_along_axis(
            y_pred_full,
            y_index,
            y_proba,
            axis=1
        )
        return y_pred_full

    def _predict_oof_model(
        self,
        estimator: ClassifierMixin,
        X: ArrayLike,
    ) -> ArrayLike:
        """
        Predict probabilities of a test set from a fitted estimator.

        Parameters
        ----------
        estimator : ClassifierMixin
            Fitted estimator.
        X : ArrayLike
            Test set.

        Returns
        -------
        ArrayLike
            Predicted probabilities.
        """
        y_pred_proba = estimator.predict_proba(X)
        # we enforce y_pred_proba to contain all labels included y
        if len(estimator.classes_) != self.n_classes_:
            y_pred_proba = self._fix_number_of_classes(
                estimator.classes_,
                y_pred_proba
            )
        return y_pred_proba

    def _fit_and_predict_oof_model(
        self,
        estimator: ClassifierMixin,
        X: ArrayLike,
        y: ArrayLike,
        train_index: ArrayLike,
        val_index: ArrayLike,
        k: int,
        sample_weight: Optional[ArrayLike] = None,
    ) -> Tuple[ClassifierMixin, ArrayLike, ArrayLike, ArrayLike]:
        """
        Fit a single out-of-fold model on a given training set and
        perform predictions on a test set.

        Parameters
        ----------
        estimator : ClassifierMixin
            Estimator to train.

        X : ArrayLike of shape (n_samples, n_features)
            Input data.

        y : ArrayLike of shape (n_samples,)
            Input labels.

        train_index : np.ndarray of shape (n_samples_train)
            Training data indices.

        val_index : np.ndarray of shape (n_samples_val)
            Validation data indices.

        k : int
            Split identification number.

        sample_weight : Optional[ArrayLike] of shape (n_samples,)
            Sample weights. If None, then samples are equally weighted.
            By default None.

        Returns
        -------
        Tuple[ClassifierMixin, ArrayLike, ArrayLike, ArrayLike]

        - [0]: Fitted estimator
        - [1]: Estimator predictions on the validation fold,
          of shape (n_samples_val,)
        - [2]: Identification number of the validation fold,
          of shape (n_samples_val,)
        - [3]: Validation data indices,
          of shape (n_samples_val,).

        """
        X_train = _safe_indexing(X, train_index)
        y_train = _safe_indexing(y, train_index)
        X_val = _safe_indexing(X, val_index)
        y_val = _safe_indexing(y, val_index)

        if sample_weight is None:
            estimator = fit_estimator(estimator, X_train, y_train)
        else:
            estimator = fit_estimator(
                estimator, X_train, y_train, sample_weight[train_index]
            )
        if X_val.shape[0] > 0:
            y_pred_proba = self._predict_oof_model(
                estimator, X_val,
            )
        else:
            y_pred_proba = np.array([])
        val_id = np.full_like(y_val, k, dtype=int)
        return estimator, y_pred_proba, val_id, val_index

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        image_input: Optional[bool] = False,
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

        image_input: Optional[bool] = False
            Whether or not the X input is an image. If True, you must provide
            a model that accepts image as input (e.g., a Neural Network). All
            Scikit-learn classifiers only accept two-dimensional inputs.

            By default False.

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
        self.image_input = image_input
        self._check_parameters()
        cv = check_cv(self.cv)
        estimator = self._check_estimator(X, y, self.estimator)

        if self.image_input:
            check_input_is_image(X)
        # X, y = check_X_y(
        #     X, y, force_all_finite=False, ensure_2d=self.image_input,
        #     allow_nd=self.image_input, dtype=["float64", "int", "object"]
        # )
        assert type_of_target(y) == "multiclass"
        self.n_classes_ = len(set(y))
        self.n_features_in_ = check_n_features_in(X, cv, estimator)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        # Initialization
        self.estimators_: List[ClassifierMixin] = []
        self.k_ = np.empty_like(y, dtype=int)
        self.n_samples_val_ = X.shape[0]

        # Work

        if cv == "prefit":
            self.single_estimator_ = estimator
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba = self._check_proba_normalized(y_pred_proba)

        else:
            self.single_estimator_ = fit_estimator(
                clone(estimator), X, y, sample_weight
            )
            y_pred_proba = np.empty((len(y), len(np.unique(y))), dtype=float)
            outputs = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._fit_and_predict_oof_model)(
                    clone(estimator),
                    X,
                    y,
                    train_index,
                    val_index,
                    k,
                    sample_weight,
                )
                for k, (train_index, val_index) in enumerate(cv.split(X))
            )
            self.estimators_, predictions, val_ids, val_indices = map(
                list, zip(*outputs)
            )
            predictions, val_ids, val_indices = map(
                np.concatenate, (predictions, val_ids, val_indices)
            )
            self.k_[val_indices] = val_ids
            y_pred_proba[val_indices] = predictions

        if self.method == "naive":
            self.conformity_scores_ = np.empty(y_pred_proba.shape)
        elif self.method == "score":
            self.conformity_scores_ = np.take_along_axis(
                1 - y_pred_proba, np.ravel(y).reshape(-1, 1), axis=1
            )
        elif self.method == "cumulated_score":
            y_true = label_binarize(y=y, classes=estimator.classes_)
            index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
            y_pred_proba_sorted = np.take_along_axis(
                y_pred_proba, index_sorted, axis=1
            )
            y_true_sorted = np.take_along_axis(y_true, index_sorted, axis=1)
            y_pred_proba_sorted_cumsum = np.cumsum(y_pred_proba_sorted, axis=1)
            cutoff = np.argmax(y_true_sorted, axis=1)
            self.conformity_scores_ = np.take_along_axis(
                y_pred_proba_sorted_cumsum, cutoff.reshape(-1, 1), axis=1
            )
            y_proba_true = np.take_along_axis(
                y_pred_proba, np.ravel(y).reshape(-1, 1), axis=1
            )
            random_state = check_random_state(self.random_state)
            u = random_state.uniform(size=len(y_pred_proba)).reshape(-1, 1)
            self.conformity_scores_ -= u*y_proba_true
        elif self.method == "top_k":
            # Here we reorder the labels by decreasing probability
            # and get the position of each label from decreasing probability
            index = np.argsort(
                np.fliplr(np.argsort(y_pred_proba, axis=1))
            )
            self.conformity_scores_ = np.take_along_axis(
                index,
                np.ravel(y).reshape(-1, 1),
                axis=1
            )

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
        agg_scores: Optional[str] = "mean"
    ) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
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
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
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
              just over the quantile based on the comparison of a uniform
              number and the difference between the cumulated score of
              the last label and the quantile.

            By default ``True``.

        agg_scores: Optional[str]

            How to aggregate the scores output by the estimators on test data
            if a cross-validation strategy is used. Choose among:

            - "mean", take the mean of scores.
            - "crossval", compare the scores between al training data and each
              test point for each label to estimate if the label must be
              included in the prediction set. Follows algorithm 2 of
              Romano+2020.

            By default "mean".

        Returns
        -------
        Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]

        - ArrayLike of shape (n_samples,) if alpha is None.

        - Tuple[ArrayLike, ArrayLike] of shapes
        (n_samples,) and (n_samples, n_classes, n_alpha) if alpha is not None.
        """
        # Checks
        cv = check_cv(self.cv)
        include_last_label = self._check_include_last_label(include_last_label)
        alpha_ = check_alpha(alpha)
        check_is_fitted(self, self.fit_attributes)
        if self.image_input:
            check_input_is_image(X)
        # X = check_array(
        #     X,
        #     force_all_finite=False,
        #     ensure_2d=self.image_input,
        #     allow_nd=self.image_input,
        #     dtype=["float64", "object"]
        # )

        # Estimate probabilities from estimator(s)
        y_pred = self.single_estimator_.predict(X)
        if cv == "prefit":
            y_pred_proba = self.single_estimator_.predict_proba(X)
            y_pred_proba = self._check_proba_normalized(y_pred_proba)
        else:
            y_pred_proba_k = np.asarray(
                Parallel(
                    n_jobs=self.n_jobs, verbose=self.verbose
                )(
                    delayed(self._predict_oof_model)(estimator, X)
                    for estimator in self.estimators_
                )
            )
            if agg_scores == "crossval":
                y_pred_proba = y_pred_proba_k[self.k_]
                y_pred_proba = self._check_proba_normalized(
                    y_pred_proba, axis=2
                )
            elif agg_scores == "mean":
                y_pred_proba = np.mean(y_pred_proba_k, axis=0)
                y_pred_proba = self._check_proba_normalized(
                    y_pred_proba, axis=1
                )
            else:
                raise ValueError("Invalid 'agg_scores' argument.")

        # Estimate prediction sets
        n = self.n_samples_val_
        if alpha_ is None:
            return np.array(y_pred)

        else:
            # Choice of the quantile
            check_alpha_and_n_samples(alpha_, n)
            if self.method == "naive":
                self.quantiles_ = 1 - alpha_
            else:
                if (cv == "prefit") or (agg_scores in ["mean"]):
                    self.quantiles_ = np.stack([
                        np.quantile(
                            self.conformity_scores_,
                            ((n + 1) * (1 - _alpha)) / n,
                            interpolation="higher"
                        ) for _alpha in alpha_
                    ])
                else:
                    self.quantiles_ = (n + 1) * (1 - alpha_)

            # Build prediction sets
            if self.method == "score":
                if (cv == "prefit") or (agg_scores == "mean"):
                    prediction_sets = np.stack(
                        [
                            y_pred_proba > 1 - quantile
                            for quantile in self.quantiles_
                        ],
                        axis=2,
                    )
                else:
                    y_comp = (
                        1 - y_pred_proba < self.conformity_scores_[:, None]
                    ).sum(axis=0)
                    prediction_sets = np.stack(
                        [
                            y_comp > _alpha * (n - 1)
                            for _alpha in alpha_
                        ], axis=2
                    )

            elif self.method in ["cumulated_score", "naive"]:
                if (cv == "prefit") or (agg_scores in ["mean"]):
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
                            np.take_along_axis(
                                y_pred_proba,
                                y_pred_index_last[:, iq].reshape(-1, 1),
                                axis=1
                            )
                            for iq, _ in enumerate(self.quantiles_)
                        ], axis=2
                    )
                    # get the prediction set by taking all probabilities
                    # above the last one
                    prediction_sets = np.stack(
                        [
                            y_pred_proba >= (
                                y_pred_proba_last[:, :, iq] - EPSILON
                            )
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
                        '"cumulated_score" method with '
                        'agg_scores = "crossval" is not implemented yet.'
                    )
            elif self.method == "top_k":
                index_sorted = np.fliplr(np.argsort(y_pred_proba, axis=1))
                y_pred_index_last = np.stack(
                    [
                        index_sorted[:, quantile]
                        for quantile in self.quantiles_
                    ], axis=1
                )
                y_pred_proba_last = np.stack(
                    [
                        np.take_along_axis(
                            y_pred_proba,
                            y_pred_index_last[:, iq].reshape(-1, 1),
                            axis=1
                        )
                        for iq, _ in enumerate(self.quantiles_)
                    ], axis=2
                )
                prediction_sets = np.stack(
                    [
                        y_pred_proba >= y_pred_proba_last[:, :, iq] - EPSILON
                        for iq, _ in enumerate(self.quantiles_)
                    ], axis=2
                )
            else:
                raise ValueError(
                    "Invalid method. "
                    "Allowed values are 'score' or 'cumulated_score'."
                )

            return y_pred, prediction_sets

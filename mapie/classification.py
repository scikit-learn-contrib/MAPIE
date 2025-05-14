from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional, Tuple, Union, cast
from typing_extensions import Self

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, check_is_fitted, indexable)

from numpy.typing import ArrayLike, NDArray

from mapie.conformity_scores import BaseClassificationScore
from mapie.conformity_scores.sets.raps import RAPSConformityScore
from mapie.conformity_scores.utils import (
    check_classification_conformity_score,
    check_target, check_and_select_conformity_score,
)
from mapie.estimator.classifier import EnsembleClassifier
from mapie.utils import (_check_alpha, _check_alpha_and_n_samples, _check_cv,
                         _check_estimator_classification, _check_n_features_in,
                         _check_n_jobs, _check_null_weight, _check_predict_params,
                         _check_verbose)
from mapie.utils import (
    _transform_confidence_level_to_alpha_list,
    _raise_error_if_fit_called_in_prefit_mode,
    _raise_error_if_method_already_called,
    _prepare_params,
    _raise_error_if_previous_method_not_called,
    _cast_predictions_to_ndarray_tuple,
    _cast_point_predictions_to_ndarray,
    _check_cv_not_string,
    _prepare_fit_params_and_sample_weight,
)


class SplitConformalClassifier:
    """
    Computes prediction sets using the split conformal classification technique:

    1. The ``fit`` method (optional) fits the base classifier to the training data.
    2. The ``conformalize`` method estimates the uncertainty of the base classifier by
       computing conformity scores on the conformity set.
    3. The ``predict_set`` method predicts labels and sets of labels.

    Parameters
    ----------
    estimator : ClassifierMixin, default=LogisticRegression()
        The base classifier used to predict labels.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction sets, indicating the
        desired coverage probability of the prediction sets. If a float is
        provided, it represents a single confidence level. If a list, multiple
        prediction sets for each specified confidence level are returned.

    conformity_score : Union[str, BaseClassificationScore], default="lac"
        The method used to compute conformity scores.

        Valid options:

        - "lac"
        - "top_k"
        - "aps"
        - "raps"
        - Any subclass of BaseClassificationScore

        A custom score function inheriting from BaseClassificationScore may also
        be provided.

        See :ref:`theoretical_description_classification`.

    prefit : bool, default=False
        If True, the base classifier must be fitted, and the ``fit``
        method must be skipped.

        If False, the base classifier will be fitted during the ``fit`` method.

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level.
        Higher values increase the output details.

    Examples
    --------
    >>> from mapie.classification import SplitConformalClassifier
    >>> from mapie.utils import train_conformalize_test_split
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.neighbors import KNeighborsClassifier

    >>> X, y = make_classification(n_samples=500)
    >>> (
    ...     X_train, X_conformalize, X_test,
    ...     y_train, y_conformalize, y_test
    ... ) = train_conformalize_test_split(
    ...     X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
    ... )

    >>> mapie_classifier = SplitConformalClassifier(
    ...     estimator=KNeighborsClassifier(),
    ...     confidence_level=0.95,
    ...     prefit=False,
    ... ).fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)

    >>> predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        prefit: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        self._estimator = estimator
        self._alphas = _transform_confidence_level_to_alpha_list(
            confidence_level
        )
        self._conformity_score = check_and_select_conformity_score(
            conformity_score,
            BaseClassificationScore
        )
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False

        # Note to developers: to implement this v1 class without touching the
        # v0 backend, we're for now using a hack. We always set cv="prefit",
        # and we fit the estimator if needed. See the .fit method below.
        self._mapie_classifier = _MapieClassifier(
            estimator=self._estimator,
            cv="prefit",
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=self._conformity_score,
            random_state=random_state,
        )
        self._predict_params: dict = {}

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits the base classifier to the training data.

        Parameters
        ----------
        X_train : ArrayLike
            Training data features.

        y_train : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Parameters to pass to the ``fit`` method of the base classifier.

        Returns
        -------
        Self
            The fitted SplitConformalClassifier instance.
        """
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)

        cloned_estimator = clone(self._estimator)
        fit_params_ = _prepare_params(fit_params)
        cloned_estimator.fit(X_train, y_train, **fit_params_)
        self._mapie_classifier.estimator = cloned_estimator

        self._is_fitted = True
        return self

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Estimates the uncertainty of the base classifier by computing
        conformity scores on the conformity set.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Features of the conformity set.

        y_conformalize : ArrayLike
            Targets of the conformity set.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` and ``predict_proba`` methods
            of the base classifier. These parameters will also be used in the
            ``predict_set`` and ``predict`` methods of this SplitConformalClassifier.

        Returns
        -------
        Self
            The conformalized SplitConformalClassifier instance.
        """
        _raise_error_if_previous_method_not_called(
            "conformalize",
            "fit",
            self._is_fitted,
        )
        _raise_error_if_method_already_called(
            "conformalize",
            self._is_conformalized,
        )

        self._predict_params = _prepare_params(predict_params)
        self._mapie_classifier.fit(
            X_conformalize,
            y_conformalize,
            predict_params=self._predict_params,
        )

        self._is_conformalized = True
        return self

    def predict_set(
        self,
        X: ArrayLike,
        conformity_score_params: Optional[dict] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        For each sample in X, predicts a label (using the base classifier),
        and a set of labels.

        If several confidence levels were provided during initialisation, several
        sets will be predicted for each sample. See the return signature.

        Parameters
        ----------
        X : ArrayLike
            Features

        conformity_score_params : Optional[dict], default=None
            Parameters specific to conformity scores, used at prediction time.

            The only example for now is ``include_last_label``, available for `aps`
            and `raps` conformity scores. For detailed information on
            ``include_last_label``, see the docstring of
            :meth:`conformity_scores.sets.aps.APSConformityScore.get_prediction_sets`.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction labels, of shape ``(n_samples,)``
            - Prediction sets, of shape ``(n_samples, n_class, n_confidence_levels)``
        """
        _raise_error_if_previous_method_not_called(
            "predict_set",
            "conformalize",
            self._is_conformalized,
        )
        conformity_score_params_ = _prepare_params(conformity_score_params)
        predictions = self._mapie_classifier.predict(
            X,
            alpha=self._alphas,
            include_last_label=conformity_score_params_.get("include_last_label", True),
            **self._predict_params,
        )
        return _cast_predictions_to_ndarray_tuple(predictions)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        For each sample in X, returns the predicted label by the base classifier.

        Parameters
        ----------
        X : ArrayLike
            Features

        Returns
        -------
        NDArray
            Array of predicted labels, with shape ``(n_samples,)``.
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "conformalize",
            self._is_conformalized,
        )
        predictions = self._mapie_classifier.predict(
            X,
            alpha=None,
            **self._predict_params,
        )
        return _cast_point_predictions_to_ndarray(predictions)


class CrossConformalClassifier:
    """
    Computes prediction sets using the cross conformal classification technique:

    1. The ``fit_conformalize`` method estimates the uncertainty of the base classifier
       in a cross-validation style. It fits the base classifier on folds of the dataset
       and computes conformity scores on the out-of-fold data.
    2. The ``predict_set`` method predicts labels and sets of labels.

    Parameters
    ----------
    estimator : ClassifierMixin, default=LogisticRegression()
        The base classifier used to predict labels.

    confidence_level : Union[float, List[float]], default=0.9
        The confidence level(s) for the prediction sets, indicating the
        desired coverage probability of the prediction sets. If a float is
        provided, it represents a single confidence level. If a list, multiple
        prediction sets for each specified confidence level are returned.

    conformity_score : Union[str, BaseClassificationScore], default="lac"
        The method used to compute conformity scores.
        Valid options:

        - "lac"
        - "aps"
        - Any subclass of BaseClassificationScore

        A custom score function inheriting from BaseClassificationScore may also
        be provided.

        See :ref:`theoretical_description_classification`.

    cv : Union[int, BaseCrossValidator], default=5
        The cross-validator used to compute conformity scores.
        Valid options:

        - integer, to specify the number of folds
        - any ``sklearn.model_selection.BaseCrossValidator`` suitable for
          classification, or a custom cross-validator inheriting from it.

        Main variants in the cross conformal setting are:

        - ``sklearn.model_selection.KFold`` (vanilla cross conformal)
        - ``sklearn.model_selection.LeaveOneOut`` (jackknife)

    n_jobs : Optional[int], default=None
        The number of jobs to run in parallel when applicable.

    verbose : int, default=0
        Controls the verbosity level. Higher values increase the
        output details.

    random_state : Optional[Union[int, np.random.RandomState]], default=None
        A seed or random state instance to ensure reproducibility in any random
        operations within the classifier.

    Examples
    --------
    >>> from mapie.classification import CrossConformalClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.neighbors import KNeighborsClassifier

    >>> X_full, y_full = make_classification(n_samples=500)
    >>> X, X_test, y, y_test = train_test_split(X_full, y_full)

    >>> mapie_classifier = CrossConformalClassifier(
    ...     estimator=KNeighborsClassifier(),
    ...     confidence_level=0.95,
    ...     cv=10
    ... ).fit_conformalize(X, y)

    >>> predicted_labels, predicted_sets = mapie_classifier.predict_set(X_test)
    """

    def __init__(
        self,
        estimator: ClassifierMixin = LogisticRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        cv: Union[int, BaseCrossValidator] = 5,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        _check_cv_not_string(cv)

        self._mapie_classifier = _MapieClassifier(
            estimator=estimator,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            conformity_score=check_and_select_conformity_score(
                conformity_score,
                BaseClassificationScore,
            ),
            random_state=random_state,
        )

        self._alphas = _transform_confidence_level_to_alpha_list(
            confidence_level
        )
        self.is_fitted_and_conformalized = False

        self._predict_params: dict = {}

    def fit_conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        fit_params: Optional[dict] = None,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Estimates the uncertainty of the base classifier in a cross-validation style:
        fits the base classifier on different folds of the dataset
        and computes conformity scores on the corresponding out-of-fold data.

        Parameters
        ----------
        X : ArrayLike
            Features

        y : ArrayLike
            Targets

        groups: Optional[ArrayLike] of shape (n_samples,), default=None
            Groups to pass to the cross-validator.

        fit_params : Optional[dict], default=None
            Parameters to pass to the ``fit`` method of the base classifier.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` and ``predict_proba`` methods
            of the base classifier. These parameters will also be used in the
            ``predict_set`` and ``predict`` methods of this CrossConformalClassifier.

        Returns
        -------
        Self
            This CrossConformalClassifier instance, fitted and conformalized.
        """
        _raise_error_if_method_already_called(
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        fit_params_, sample_weight = _prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._predict_params = _prepare_params(predict_params)
        self._mapie_classifier.fit(
            X=X,
            y=y,
            sample_weight=sample_weight,
            groups=groups,
            fit_params=fit_params_,
            predict_params=self._predict_params
        )

        self.is_fitted_and_conformalized = True
        return self

    def predict_set(
        self,
        X: ArrayLike,
        conformity_score_params: Optional[dict] = None,
        agg_scores: str = "mean",
    ) -> Tuple[NDArray, NDArray]:
        """
        For each sample in X, predicts a label (using the base classifier),
        and a set of labels.

        If several confidence levels were provided during initialisation, several
        sets will be predicted for each sample. See the return signature.

        Parameters
        ----------
        X : ArrayLike
            Features

        conformity_score_params : Optional[dict], default=None
            Parameters specific to conformity scores, used at prediction time.

            The only example for now is ``include_last_label``, available for `aps`
            and `raps` conformity scores. For detailed information on
            ``include_last_label``, see the docstring of
            :meth:`conformity_scores.sets.aps.APSConformityScore.get_prediction_sets`.

        agg_scores : str, default="mean"
            How to aggregate conformity scores.

            Each classifier fitted on different folds of the dataset is used to produce
            conformity scores on the test data. The agg_score parameter allows to
            control how those scores are aggregated. Valid options:

            - "mean", takes the mean of scores.
            - "crossval", compares the scores between all training data and each
              test point for each label to estimate if the label must be
              included in the prediction set. Follows algorithm 2 of
              Classification with Valid and Adaptive Coverage (Romano+2020).

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction labels, of shape ``(n_samples,)``
            - Prediction sets, of shape ``(n_samples, n_class, n_confidence_levels)``
        """
        _raise_error_if_previous_method_not_called(
            "predict_set",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        conformity_score_params_ = _prepare_params(conformity_score_params)
        predictions = self._mapie_classifier.predict(
            X,
            alpha=self._alphas,
            include_last_label=conformity_score_params_.get("include_last_label", True),
            agg_scores=agg_scores,
            **self._predict_params,
        )
        return _cast_predictions_to_ndarray_tuple(predictions)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        For each sample in X, returns the predicted label by the base classifier.

        Parameters
        ----------
        X : ArrayLike
            Features

        Returns
        -------
        NDArray
            Array of predicted labels, with shape ``(n_samples,)``.
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )
        predictions = self._mapie_classifier.predict(
            X, alpha=None, **self._predict_params,
        )
        return _cast_point_predictions_to_ndarray(predictions)


class _MapieClassifier(ClassifierMixin, BaseEstimator):
    """
    Note to users: _MapieClassifier is now private, and may change at any time.
    Please use CrossConformalClassifier or CrossConformalClassifier instead.
    See the v1 migration guide for more information.

    Prediction sets for classification.

    This class implements several conformal prediction strategies for
    estimating prediction sets for classification. Instead of giving a
    single predicted label, the idea is to give a set of predicted labels
    (or prediction sets) which come with mathematically guaranteed coverages.

    Parameters
    ----------
    estimator: Optional[ClassifierMixin]
        Any classifier with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default None.
        If ``None``, estimator defaults to a ``LogisticRegression`` instance.

    cv: Optional[Union[int, str, BaseCrossValidator]]
        The cross-validation strategy for computing scores.
        It directly drives the distinction between jackknife and cv variants.
        Choose among:

        - ``None``, to use the default 5-fold cross-validation
        - integer, to specify the number of folds.
          If equal to -1, equivalent to
          ``sklearn.model_selection.LeaveOneOut()``.
        - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
          Main variants are:
          - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
          - ``sklearn.model_selection.KFold`` (cross-validation)
        - ``"split"``, does not involve cross-validation but a division
          of the data into training and calibration subsets. The splitter
          used is the following: ``sklearn.model_selection.ShuffleSplit``.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          to calibrate the predictions through the score computation.
          At prediction time, quantiles of these scores are used to estimate
          prediction sets.

        By default ``None``.

    test_size: Optional[Union[int, float]]
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, it will be set to 0.1.

        If cv is not ``"split"``, ``test_size`` is ignored.

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

    conformity_score: BaseClassificationScore
        Score function that handle all that is related to conformity scores.

        By default ``None``.

    random_state: Optional[Union[int, RandomState]]
        Pseudo random number generator state used for random uniform sampling
        for evaluation quantiles and prediction sets.
        Pass an int for reproducible output across multiple function calls.

        By default ``None``.

    verbose: int, optional
        The verbosity level, used with joblib for multiprocessing.
        At this moment, parallel processing is disabled.
        The frequency of the messages increases with the verbosity level.
        If it more than ``10``, all iterations are reported.
        Above ``50``, the output is sent to stdout.

        By default ``0``.

    Attributes
    ----------
    estimator_: EnsembleClassifier
        Sklearn estimator that handle all that is related to the estimator.

    conformity_score_function_: BaseClassificationScore
        Score function that handle all that is related to conformity scores.

    n_features_in_: int
        Number of features passed to the fit method.

    conformity_scores_: ArrayLike of shape (n_samples_train)
        The conformity scores used to calibrate the prediction sets.

    quantiles_: ArrayLike of shape (n_alpha)
        The quantiles estimated from ``conformity_scores_`` and alpha values.

    label_encoder_: LabelEncoder
        Label encoder used to encode the labels.

    References
    ----------
    [1] Mauricio Sadinle, Jing Lei, and Larry Wasserman.
    "Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.",
    Journal of the American Statistical Association, 114, 2019.

    [2] Yaniv Romano, Matteo Sesia and Emmanuel J. Candès.
    "Classification with Valid and Adaptive Coverage."
    NeurIPS 202 (spotlight) 2020.

    [3] Anastasios Nikolas Angelopoulos, Stephen Bates, Michael Jordan
    and Jitendra Malik.
    "Uncertainty Sets for Image Classifiers using Conformal Prediction."
    International Conference on Learning Representations 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie.classification import _MapieClassifier
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> clf = GaussianNB().fit(X_toy, y_toy)
    >>> mapie = _MapieClassifier(estimator=clf, cv="prefit").fit(X_toy, y_toy)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.2)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False False]
     [ True False False]
     [ True  True False]
     [ True  True False]
     [False  True False]
     [False  True  True]
     [False  True  True]
     [False False  True]
     [False False  True]]
    """

    fit_attributes = [
        "estimator_",
        "n_features_in_",
        "conformity_scores_",
        "conformity_score_function_",
        "classes_",
        "label_encoder_"
    ]

    def __init__(
        self,
        estimator: Optional[ClassifierMixin] = None,
        cv: Optional[Union[int, str, BaseCrossValidator]] = None,
        test_size: Optional[Union[int, float]] = None,
        n_jobs: Optional[int] = None,
        conformity_score: Optional[BaseClassificationScore] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        verbose: int = 0
    ) -> None:
        self.estimator = estimator
        self.cv = cv
        self.test_size = test_size
        self.n_jobs = n_jobs
        self.conformity_score = conformity_score
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
        _check_n_jobs(self.n_jobs)
        _check_verbose(self.verbose)
        check_random_state(self.random_state)

    def _get_classes_info(
            self, estimator: ClassifierMixin, y: NDArray
    ) -> Tuple[int, NDArray]:
        """
        Compute the number of classes and the classes values
        according to either the pre-trained model or to the
        values in y.

        Parameters
        ----------
        estimator: ClassifierMixin
            Estimator pre-fitted or not.

        y: NDArray
            Values to predict.

        Returns
        -------
        Tuple[int, NDArray]
            The number of unique classes and their unique
            values.

        Raises
        ------
        ValueError
            If `cv="prefit"` and that classes in `y` are not included into
            `estimator.classes_`.

        Warning
            If number of calibration labels is lower than number of labels
            for training (in prefit setting)
        """
        n_unique_y_labels = len(np.unique(y))
        if self.cv == "prefit":
            classes = estimator.classes_
            n_classes = len(np.unique(classes))
            if not set(np.unique(y)).issubset(classes):
                raise ValueError(
                    "Values in y do not matched values in estimator.classes_."
                    + " Check that you are not adding any new label"
                )
            if n_classes > n_unique_y_labels:
                warnings.warn(
                    "WARNING: your conformity dataset has less labels"
                    + " than your training dataset (training"
                    + f" has {n_classes} unique labels while"
                    + f" conformity have {n_unique_y_labels} unique labels"
                )

        else:
            n_classes = n_unique_y_labels
            classes = np.unique(y)

        return n_classes, classes

    def _get_label_encoder(self) -> LabelEncoder:
        """
        Construct the label encoder with respect to the classes values.

        Returns
        -------
        LabelEncoder
        """
        return LabelEncoder().fit(self.classes_)

    def _check_fit_parameter(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ):
        """
        Perform several checks on class parameters.
        """
        self._check_parameters()
        cv = _check_cv(
            self.cv, test_size=self.test_size, random_state=self.random_state
        )
        X, y = indexable(X, y)
        y = _check_y(y)

        sample_weight = cast(Optional[NDArray], sample_weight)
        groups = cast(Optional[NDArray], groups)
        sample_weight, X, y = _check_null_weight(sample_weight, X, y)

        y = cast(NDArray, y)

        estimator = _check_estimator_classification(X, y, cv, self.estimator)
        self.n_features_in_ = _check_n_features_in(X, cv, estimator)

        self.n_classes_, self.classes_ = self._get_classes_info(estimator, y)
        self.label_encoder_ = self._get_label_encoder()
        y_enc = self.label_encoder_.transform(y)

        cs_estimator = check_classification_conformity_score(self.conformity_score)
        cs_estimator.set_external_attributes(
            classes=self.classes_,
            label_encoder=self.label_encoder_,
            random_state=self.random_state
        )
        if (
            isinstance(cs_estimator, RAPSConformityScore) and
            not (
                self.cv in ["split", "prefit"] or
                isinstance(self.cv, BaseShuffleSplit)
            )
        ):
            raise ValueError(
                "RAPS conformity score can only be used "
                "with SplitConformalClassifier."
            )

        # Cast
        X, y_enc, y = cast(NDArray, X), cast(NDArray, y_enc), cast(NDArray, y)
        sample_weight = cast(NDArray, sample_weight)
        groups = cast(NDArray, groups)

        X, y, y_enc, sample_weight, groups = \
            cs_estimator.split_data(X, y, y_enc, sample_weight, groups)
        self.n_samples_ = cs_estimator.n_samples_

        check_target(cs_estimator, y)

        return (
            estimator, cs_estimator, cv, X, y, y_enc, sample_weight, groups
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any
    ) -> _MapieClassifier:
        """
        Fit the base estimator or use the fitted base estimator.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: NDArray of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If None, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no prediction sets.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        kwargs : dict
            Additional fit and predict parameters.

        Returns
        -------
        _MapieClassifier
            The model itself.
        """
        fit_params = kwargs.pop('fit_params', {})
        predict_params = kwargs.pop('predict_params', {})

        if len(predict_params) > 0:
            self._predict_params = True
        else:
            self._predict_params = False

        # Checks
        (estimator,
         self.conformity_score_function_,
         cv,
         X,
         y,
         y_enc,
         sample_weight,
         groups) = self._check_fit_parameter(
            X, y, sample_weight, groups
        )

        # Cast
        X, y_enc, y = cast(NDArray, X), cast(NDArray, y_enc), cast(NDArray, y)
        sample_weight = cast(NDArray, sample_weight)
        groups = cast(NDArray, groups)

        # Work
        self.estimator_ = EnsembleClassifier(
            estimator,
            self.n_classes_,
            cv,
            self.n_jobs,
            self.test_size,
            self.verbose,
        )
        # Fit the prediction function
        self.estimator_ = self.estimator_.fit(
            X, y, y_enc=y_enc, sample_weight=sample_weight, groups=groups,
            **fit_params
        )

        # Predict on calibration data
        y_pred_proba, y, y_enc = self.estimator_.predict_proba_calib(
            X, y, y_enc, groups, **predict_params
        )

        # Compute the conformity scores
        self.conformity_score_function_.set_ref_predictor(self.estimator_)
        self.conformity_scores_ = \
            self.conformity_score_function_.get_conformity_scores(
                y, y_pred_proba, y_enc=y_enc, X=X,
                sample_weight=sample_weight, groups=groups
            )
        return self

    def predict(
        self,
        X: ArrayLike,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        include_last_label: Optional[Union[bool, str]] = True,
        agg_scores: Optional[str] = "mean",
        **predict_params
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Prediction and prediction sets on new samples based on target
        confidence interval.
        Prediction sets for a given ``alpha`` are deduced from:

        - quantiles of softmax scores (``"lac"`` method)
        - quantiles of cumulated scores (``"aps"`` method)

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        alpha: Optional[Union[float, Iterable[float]]]
            Can be a float, a list of floats, or a ``ArrayLike`` of floats.
            Between 0 and 1, represent the uncertainty of the confidence
            interval.
            Lower ``alpha`` produce larger (more conservative) prediction sets.
            ``alpha`` is the complement of the target coverage level.

            By default ``None``.

        include_last_label: Optional[Union[bool, str]]
            Whether or not to include last label in
            prediction sets for the "aps" method. Choose among:

            - False, does not include label whose cumulated score is just over
              the quantile.
            - True, includes label whose cumulated score is just over the
              quantile, unless there is only one label in the prediction set.
            - "randomized", randomly includes label whose cumulated score is
              just over the quantile based on the comparison of a uniform
              number and the difference between the cumulated score of
              the last label and the quantile.

            When set to ``True`` or ``False``, it may result in a coverage
            higher than ``1 - alpha`` (because contrary to the "randomized"
            setting, none of these methods create empty prediction sets). See
            [2] and [3] for more details.

            By default ``True``.

        agg_scores: Optional[str]

            How to aggregate the scores output by the estimators on test data
            if a cross-validation strategy is used. Choose among:

            - "mean", take the mean of scores.
            - "crossval", compare the scores between all training data and each
              test point for each label to estimate if the label must be
              included in the prediction set. Follows algorithm 2 of
              Romano+2020.

            By default "mean".

        predict_params : dict
            Additional predict parameters.

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]

        - NDArray of shape (n_samples,) if alpha is None.

        - Tuple[NDArray, NDArray] of shapes
        (n_samples,) and (n_samples, n_classes, n_alpha) if alpha is not None.
        """
        # Checks

        if hasattr(self, '_predict_params'):
            _check_predict_params(
                self._predict_params,
                predict_params, self.cv
            )

        check_is_fitted(self, self.fit_attributes)
        alpha = cast(Optional[NDArray], _check_alpha(alpha))

        # Estimate predictions
        y_pred = self.estimator_.single_estimator_.predict(X, **predict_params)
        if alpha is None:
            return y_pred

        # Estimate of probabilities from estimator(s)
        # In all cases: len(y_pred_proba.shape) == 3
        # with  (n_test, n_classes, n_alpha or n_train_samples)
        n = len(self.conformity_scores_)
        alpha_np = cast(NDArray, alpha)
        _check_alpha_and_n_samples(alpha_np, n)

        # Estimate prediction sets
        prediction_sets = self.conformity_score_function_.predict_set(
            X, alpha_np,
            estimator=self.estimator_,
            conformity_scores=self.conformity_scores_,
            include_last_label=include_last_label,
            agg_scores=agg_scores,
        )

        self.quantiles_ = self.conformity_score_function_.quantiles_

        return y_pred, prediction_sets

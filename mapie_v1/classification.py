from __future__ import annotations

from typing import Optional, Union, Tuple, Iterable
from typing_extensions import Self

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.linear_model import LogisticRegression

from numpy.typing import ArrayLike, NDArray
from mapie.classification import MapieClassifier
from mapie.conformity_scores import BaseClassificationScore
from mapie_v1.utils import (
    transform_confidence_level_to_alpha_list,
    prepare_params,
    cast_predictions_to_ndarray_tuple,
    cast_point_predictions_to_ndarray,
    raise_error_if_previous_method_not_called,
    raise_error_if_method_already_called,
    raise_error_if_fit_called_in_prefit_mode,
    check_cv_not_string,
    prepare_fit_params_and_sample_weight,
)
from mapie_v1.conformity_scores._utils import check_and_select_conformity_score


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
    >>> from mapie_v1.classification import SplitConformalClassifier
    >>> from mapie_v1.utils import train_conformalize_test_split
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
        self._alphas = transform_confidence_level_to_alpha_list(
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
        self._mapie_classifier = MapieClassifier(
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
        raise_error_if_fit_called_in_prefit_mode(self._prefit)
        raise_error_if_method_already_called("fit", self._is_fitted)

        cloned_estimator = clone(self._estimator)
        fit_params_ = prepare_params(fit_params)
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
        raise_error_if_previous_method_not_called(
            "conformalize",
            "fit",
            self._is_fitted,
        )
        raise_error_if_method_already_called(
            "conformalize",
            self._is_conformalized,
        )

        self._predict_params = prepare_params(predict_params)
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
        For each sample in X, returns the predicted label and a set of labels.

        If several confidence levels were provided during initialisation, several
        sets will be predicted for each sample. See the return signature.

        Parameters
        ----------
        X : ArrayLike
            Features

        conformity_score_params : dict, default=None
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
        raise_error_if_previous_method_not_called(
            "predict_set",
            "conformalize",
            self._is_conformalized,
        )
        conformity_score_params_ = prepare_params(conformity_score_params)
        predictions = self._mapie_classifier.predict(
            X,
            alpha=self._alphas,
            include_last_label=conformity_score_params_.get("include_last_label", True),
            **self._predict_params,
        )
        return cast_predictions_to_ndarray_tuple(predictions)

    def predict(self, X: ArrayLike) -> NDArray:
        """
        For each sample in X, returns the predicted label

        Parameters
        ----------
        X : ArrayLike
            Features

        Returns
        -------
        NDArray
            Array of predicted labels, with shape (n_samples,).
        """
        raise_error_if_previous_method_not_called(
            "predict",
            "conformalize",
            self._is_conformalized,
        )
        predictions = self._mapie_classifier.predict(
            X,
            alpha=None,
            **self._predict_params,
        )
        return cast_point_predictions_to_ndarray(predictions)


class CrossConformalClassifier:
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
        """
        All except raps & top-k
        """
        check_cv_not_string(cv)

        self._mapie_classifier = MapieClassifier(
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

        self._alphas = transform_confidence_level_to_alpha_list(
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
        raise_error_if_method_already_called(
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        fit_params_, sample_weight = prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._predict_params = prepare_params(predict_params)
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
        Shape: (n, ), (n, n_class, n_confidence_level)
        """
        raise_error_if_previous_method_not_called(
            "predict_set",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )

        conformity_score_params_ = prepare_params(conformity_score_params)
        predictions = self._mapie_classifier.predict(
            X,
            alpha=self._alphas,
            include_last_label=conformity_score_params_.get("include_last_label", True),
            agg_scores=agg_scores,
            **self._predict_params,
        )
        return cast_predictions_to_ndarray_tuple(predictions)

    def predict(self, X: ArrayLike) -> NDArray:
        raise_error_if_previous_method_not_called(
            "predict",
            "fit_conformalize",
            self.is_fitted_and_conformalized,
        )
        predictions = self._mapie_classifier.predict(
            X, alpha=None, **self._predict_params,
        )
        return cast_point_predictions_to_ndarray(predictions)

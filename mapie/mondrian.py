from copy import deepcopy
from typing import Iterable, Optional, Tuple, Union, cast

import numpy as np
from sklearn.utils.validation import _check_y, check_is_fitted, indexable

from mapie.calibration import MapieCalibrator
from mapie.classification import MapieClassifier
from mapie.conformity_scores import (
    AbsoluteConformityScore,
    APSConformityScore,
    GammaConformityScore,
    LACConformityScore,
    NaiveConformityScore,
    TopKConformityScore
)
from mapie.multi_label_classification import MapieMultiLabelClassifier
from mapie.regression import MapieRegressor
from mapie.utils import check_alpha
from mapie._typing import ArrayLike, NDArray


class Mondrian:
    """Mondrian is a method that allows to make  perform conformal predictions
    for disjoints groups of individuals.
    The Mondrian method is implemented in the Mondrian class. It takes as
    input a MapieClassifier, MapieRegressor or MapieMultiLabelClassifier
    estimator and fits a model for each group of individuals. The Mondrian
    class can then be used to run a conformal prediction procedure for each
    of these groups and hence achieve marginal coverage on each of them.

    Parameters
    ----------
    mapie_estimator : Union[MapieClassifier, MapieRegressor,
                            MapieMultiLabelClassifier]
        The estimator for which the Mondrian method will be applied. The
        estimator must be used with cv='prefit' and the conformity score must
        be one of the following:
        - For MapieClassifier: 'lac', 'score', 'cumulated_score',
        'aps' or 'topk'
        - For MapieRegressor: 'gamma', 'absolute' or 'aps'

    Attributes
    ----------
    unique_groups : NDArray
        The unique groups of individuals for which the estimator was fitted
    mapie_estimators : Dict
        A dictionary containing the fitted conformal estimator for each group
        of individuals

    References
    ----------
    Vladimir Vovk, David Lindsay, Ilia Nouretdinov, and Alex Gammerman.
    Mondrian confidence machine.
    Technical report, Royal Holloway University of London, 2003

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie.classification import MapieClassifier
    >>> X_toy = np.arange(9).reshape(-1, 1)
    >>> y_toy = np.stack([0, 0, 1, 0, 1, 2, 1, 2, 2])
    >>> groups = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    >>> clf = GaussianNB().fit(X_toy, y_toy)
    >>> mapie = Mondrian(MapieClassifier(estimator=clf, cv="prefit")).fit(
    ...     X_toy, y_toy, groups)
    >>> _, y_pi_mapie = mapie.predict(X_toy, alpha=0.4, groups=groups)
    >>> print(y_pi_mapie[:, :, 0])
    [[ True False False]
    [ True False False]
    [ True  True False]
    [ True  True False]
    [False  True False]
    [False  True  True]
    [False False  True]
    [False False  True]
    [False False  True]]
    """

    allowed_estimators = (
        MapieClassifier,
        MapieRegressor,
        MapieMultiLabelClassifier,
        MapieCalibrator
    )
    allowed_classification_ncs_str = [
        "lac", "score", "cumulated_score", "aps", "topk"
    ]
    allowed_classification_ncs_class = (
        LACConformityScore, NaiveConformityScore, APSConformityScore,
        TopKConformityScore
    )
    allowed_regression_ncs = (
        GammaConformityScore, AbsoluteConformityScore, APSConformityScore
    )
    fit_attributes = [
        "unique_groups",
        "mapie_estimators"
    ]

    def __init__(
        self, mapie_estimator: Union[
            MapieCalibrator,
            MapieClassifier,
            MapieRegressor,
            MapieMultiLabelClassifier
        ]
    ):
        self.mapie_estimator = mapie_estimator

    def _check_mapie_classifier(self):
        """
        Check that the underlying Mapie estimator uses cv='prefit'

        Raises
        ------
        ValueError
            If the underlying Mapie estimator does not use cv='prefit'
            if the Mondrian method is not used with a MapieMultiLabelClassifier
        NotFittedError
            If the underlying Mapie estimator is not fitted and is the Mondrian
            method is used with a MapieMultiLabelClassifier
        """
        if not isinstance(self.mapie_estimator, MapieMultiLabelClassifier):
            if not self.mapie_estimator.cv == "prefit":
                raise ValueError(
                    "Mondrian can only be used if the underlying Mapie" +
                    "estimator uses cv='prefit'."
                )
        else:
            check_is_fitted(self.mapie_estimator.estimator)

    def _check_groups_fit(self, X: NDArray, groups: NDArray):
        """Check that each group is defined by an integer and check that there
        are at least 2 individuals per group

        Parameters
        ----------
        X : NDArray of shape (n_samples, n_features)
            The input data
        groups : NDArray of shape (n_samples,)

        Raises
        ------
        ValueError
            If the groups are not defined by integers
            If there is less than 2 individuals per group
            If the number of individuals in the groups is not equal to the
            number of rows in X
        """
        if not np.issubdtype(groups.dtype, np.integer):
            raise ValueError("The groups must be defined by integers")
        _, counts = np.unique(groups, return_counts=True)
        if np.min(counts) < 2:
            raise ValueError("There must be at least 2 individuals per group")
        if len(groups) != X.shape[0]:
            raise ValueError(
                "The number of individuals in the groups must be equal" +
                " to the number of rows in X"
            )

    def _check_groups_predict(self, X: NDArray, groups: ArrayLike) -> NDArray:
        """Check that there is no new group in the prediction and that
        the number of individuals in the groups is equal to the number of
        rows in X

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data
        groups : ArrayLike of shape (n_samples,)

        returns
        -------
        groups : NDArray of shape (n_samples,)
            Groups of individuals

        Raises
        ------
        ValueError
            If there is a new group in the prediction
            If the number of individuals in the groups is not equal to the
            number of rows in X
        """
        groups = cast(NDArray, np.array(groups))
        if not np.all(np.isin(groups, self.unique_groups)):
            raise ValueError("There is a new group in the prediction")
        if len(groups) != X.shape[0]:
            raise ValueError("The number of individuals in the groups must " +
                             "be equal to the number of rows in X")
        return groups

    def _check_estimator(self):
        """
        Check that the estimator is in the allowed_estimators

        Raises
        ------
        ValueError
            If the estimator is not in the allowed_estimators
        """
        if not isinstance(self.mapie_estimator, self.allowed_estimators):
            raise ValueError(
                "The estimator must be a MapieClassifier, MapieRegressor or" +
                " MapieMultiLabelClassifier"
            )

    def _check_confomity_score(self):
        """
        Check that the conformity score is in allowed_classification_ncs_str
        or allowed_classification_ncs_class if the estimator is MapieClassifier
        or in the allowed_regression_ncs if the estimator is a MapieRegressor

        Raises
        ------
        ValueError
            If conformity score is not in the allowed_classification_ncs_str
            or allowed_classification_ncs_class if the estimator is a
            MapieClassifier or in the allowed_regression_ncs if the estimator
            is a MapieRegressor
        """
        if isinstance(self.mapie_estimator, MapieClassifier):
            if self.mapie_estimator.conformity_score is not None:
                if self.mapie_estimator.conformity_score not in \
                   self.allowed_classification_ncs_class:
                    raise ValueError(
                        "The conformity score for the MapieClassifier must" +
                        f" be one of {self.allowed_classification_ncs_class}"
                    )
            if self.mapie_estimator.method is not None:
                if self.mapie_estimator.method not in \
                   self.allowed_classification_ncs_str:
                    raise ValueError(
                        "The conformity score for the MapieClassifier must " +
                        f"be one of {self.allowed_classification_ncs_str}"
                    )
        elif isinstance(self.mapie_estimator, MapieRegressor):
            if self.mapie_estimator.conformity_score is not None:
                if self.mapie_estimator.conformity_score not in\
                      self.allowed_regression_ncs:
                    raise ValueError(
                        "The conformity score for the MapieRegressor must " +
                        f"be one of {self.allowed_regression_ncs}"
                    )

    def _check_fit_parameters(
            self, X: ArrayLike, y: ArrayLike, groups: ArrayLike
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Perform checks on the input data, groups and the estimator

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data
        y : ArrayLike of shape (n_samples,) or (n_samples, n_outputs)
            The target values
        groups : ArrayLike of shape (n_samples,)
            The groups of individuals

        Returns
        -------
        X : NDArray of shape (n_samples, n_features)
            The input data
        y : NDArray of shape (n_samples,) or (n_samples, n_outputs)
            The target values
        groups : NDArray of shape (n_samples,)
        """
        self._check_estimator()
        self._check_mapie_classifier()
        self._check_confomity_score()
        X, y = indexable(X, y)
        if isinstance(self.mapie_estimator, MapieMultiLabelClassifier):
            y = _check_y(y, multi_output=True)
        else:
            y = _check_y(y)
        X = cast(NDArray, X)
        y = cast(NDArray, y)
        groups = cast(NDArray, np.array(groups))
        self._check_groups_fit(X, groups)

        return X, y, groups

    def _check_is_topk_calibrator(self):
        """
        Check that the predict_proba method can only be used with a
        MapieCalibrator estimator
        """
        if not isinstance(self.mapie_estimator, MapieCalibrator):
            raise ValueError(
                "The predict_proba method can only be used with a " +
                "MapieCalibrator estimator"
            )

    def _check_not_topk_calibrator(self):
        """
        Check that the predict method can only be used with a MapieCalibrator
        estimator
        """
        if isinstance(self.mapie_estimator, MapieCalibrator):
            raise ValueError(
                "The predict method can only be used with a MapieClassifier," +
                "MapieRegressor or MapieMultiLabelClassifier estimator"
            )

    def fit(self, X: ArrayLike, y: ArrayLike, groups: ArrayLike, **kwargs):
        """
        Fit the Mondrian method

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data
        y : ArrayLike of shape (n_samples,) or (n_samples, n_outputs)
            The target values
        groups : ArrayLike of shape (n_samples,)
            The groups of individuals
        **kwargs
            Additional keyword arguments to pass to the estimator's fit method
            that may be specific to the Mapie estimator used
        """

        X, y, groups = self._check_fit_parameters(X, y, groups)
        self.unique_groups = np.unique(groups)
        self.mapie_estimators = {}

        for group in self.unique_groups:
            mapie_group_estimator = deepcopy(self.mapie_estimator)
            indices_groups = np.argwhere(groups == group)[:, 0]
            X_g, y_g = X[indices_groups], y[indices_groups]
            mapie_group_estimator.fit(X_g, y_g, **kwargs)
            self.mapie_estimators[group] = mapie_group_estimator
        return self

    def predict(
            self, X: ArrayLike, groups: ArrayLike,
            alpha: Optional[Union[float, Iterable[float]]] = None,
            **kwargs
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Perform conformal prediction for each group of individuals

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data
        groups : ArrayLike of shape (n_samples,)
            The groups of individuals
        alpha : float or Iterable[float], optional
            The desired coverage level(s) for each group.

            By default None.
        **kwargs
            Additional keyword arguments to pass to the estimator's predict
            method that may be specific to the Mapie estimator used

        Returns
        -------
        y_pred : NDArray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted values
        y_pss : NDArray of shape (n_samples, n_outputs, n_alpha)
        """

        check_is_fitted(self, self.fit_attributes)
        self._check_not_topk_calibrator()
        X = indexable(X)
        X = cast(NDArray, X)
        groups = self._check_groups_predict(X, groups)
        if alpha is None:
            return self.mapie_estimator.predict(X, **kwargs)
        else:
            alpha_np = cast(NDArray, check_alpha(alpha))
            unique_groups = np.unique(groups)
            for i, group in enumerate(unique_groups):
                indices_groups = np.argwhere(groups == group)[:, 0]
                X_g = X[indices_groups]
                y_pred_g, y_pss_g = self.mapie_estimators[group].predict(
                    X_g, alpha=alpha_np, **kwargs
                )
                if i == 0:
                    if len(y_pred_g.shape) == 1:
                        y_pred = np.empty((X.shape[0],))
                    else:
                        y_pred = np.empty((X.shape[0], y_pred_g.shape[1]))
                    y_pss = np.empty(
                        (X.shape[0], y_pss_g.shape[1], len(alpha_np))
                    )
                y_pred[indices_groups] = y_pred_g
                y_pss[indices_groups] = y_pss_g

            return y_pred, y_pss

    def predict_proba(
            self, X: ArrayLike, groups: ArrayLike, **kwargs
    ) -> NDArray:
        """
        Perform top-label calibration for each group of individuals

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            The input data
        groups : ArrayLike of shape (n_samples,)
            The groups of individuals
        **kwargs
            Additional keyword arguments to pass to the estimator's
            predict_proba method that may be specific to the Mapie estimator
            used

        Returns
        -------
        y_pred_proba : NDArray of shape (n_samples, n_classes)
            The calibrated predicted probabilities
        """
        self._check_is_topk_calibrator()
        X = indexable(X)
        X = cast(NDArray, X)
        unique_groups = np.unique(groups)
        y_pred_proba = np.empty(
            (X.shape[0], len(self.mapie_estimator.estimator.classes_))
        )
        for group in unique_groups:
            indices_groups = np.argwhere(groups == group)[:, 0]
            X_g = X[indices_groups]
            y_pred_proba_g = self.mapie_estimators[group].predict_proba(
                X_g, **kwargs
            )
            y_pred_proba[indices_groups] = y_pred_proba_g

        return y_pred_proba
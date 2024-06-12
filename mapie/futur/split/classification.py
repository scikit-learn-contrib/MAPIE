from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.calibrators.ccp import check_calibrator
from mapie.futur.split.base import SplitMapie, Calibrator
from mapie.conformity_scores import ConformityScore
from mapie.conformity_scores.classification_scores import LAC


class SplitMapieClassifier(SplitMapie):
    """
    This class implements an adaptative conformal prediction method proposed by
    Gibbs et al. (2023) in "Conformal Prediction With Conditional Guarantees".
    This method works with a ``"split"`` approach which requires a separate
    calibration phase. The ``fit`` method automatically split the data into
    two disjoint sets to train the predictor and the calibrator. You can call
    ``fit_estimator`` and ``fit_calibrator`` to do the two step one after the
    other. You will have to make sure that data used in the two methods,
    for training and calibration are disjoint, to guarantee the expected
    ``1-alpha`` coverage.

    Parameters
    ----------
    predictor: Optional[RegressorMixin]
        Any regressor from scikit-learn API.
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, ``predictor`` defaults to a ``LinearRegressor`` instance.

        By default ``"None"``.

    calibrator: Optional[CCP]
        A ``CCP`` instance used to estimate the conformity scores.

        If ``None``, use as default a ``GaussianCCP`` instance.
        See the examples and the documentation to build a ``CCP``
        adaptated to your dataset and constraints.

        By default ``None``.

    cv: Optional[Union[int, str, ShuffleSplit, PredefinedSplit]]
        The splitting strategy for computing conformity scores.
        Choose among:

        - Any splitter (``ShuffleSplit`` or ``PredefinedSplit``)
        with ``n_splits=1``.
        - ``"prefit"``, assumes that ``predictor`` has been fitted already.
          All data provided in the ``calibrate`` method is then used
          for the calibration.
          The user has to take care manually that data used for model fitting
          and calibration (the data given in the ``calibrate`` method)
          are disjoint.
        - ``"split"`` or ``None``: divide the data into training and
          calibration subsets (using the default ``calib_size``=0.3).
          The splitter used is the following:
            ``sklearn.model_selection.ShuffleSplit`` with ``n_splits=1``.

        By default ``None``.

    conformity_score: Optional[ConformityScore]
        ConformityScore instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteConformityScore`` symetrical
        conformity score
        - Any ``ConformityScore`` class

        By default ``None``.

    alpha: Optional[float]
        Between ``0.0`` and ``1.0``, represents the risk level of the
        confidence interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default ``None``

    random_state: Optional[int]
        Integer used to set the numpy seed, to get reproducible calibration
        results.
        If ``None``, the prediction intervals will be stochastics, and will
        change if you refit the calibration (even if no arguments have change).

        WARNING: If ``random_state``is not ``None``, ``np.random.seed`` will
        be changed, which will reset the seed for all the other random
        number generators. It may have an impact on the rest of your code.

        By default ``None``.

    Attributes
    ----------
    beta_up_: Tuple[NDArray, bool]
        Calibration fitting results, used to build the upper bound of the
        prediction intervals.
        beta_up[0]: Array of shape (calibrator.n_out, )
        beta_up[1]: Whether the optimization process converged or not
                    (the coverage is not garantied if the optimization fail)

    beta_low_: Tuple[NDArray, bool]
        Same as beta_up, but for the lower bound

    References
    ----------
    Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
    "Conformal Prediction With Conditional Guarantees", 2023

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.futur import SplitMapieClassifier
    >>> np.random.seed(1)
    >>> X_train = np.arange(0,400,2).reshape(-1, 1)
    >>> y_train = np.array([0]*50 + [1]*50 + [2]*50 + [3]*50)
    >>> mapie_reg = SplitMapieClassifier(alpha=0.1, random_state=1)
    >>> mapie_reg = mapie_reg.fit(X_train, y_train)
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    >>> print(np.round(y_pred[[0, 40, 80, 120]], 2))
    [0 0 1 2]
    >>> print(np.round(y_pis[[0, 40, 80, 120], :, 0], 2))
    [[1. 0. 0. 0.]
     [1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]]
    """
    def _check_estimator_fit_predict_predict_proba(
        self, estimator: Union[RegressorMixin, ClassifierMixin]
    ) -> None:
        """
        Check that the estimator has a fit and precict method.

        Parameters
        ----------
        estimator: Union[RegressorMixin, ClassifierMixin]
            Estimator to train.

        Raises
        ------
        ValueError
            If the estimator does not have a fit or predict or predict_proba
            attribute.
        """
        if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")
                and hasattr(estimator, "predict_proba")):
            raise ValueError(
                "Invalid estimator. "
                "Please provide a classifier with fit,"
                "predict, and predict_proba methods."
            )

    def _check_estimator_classification(
        self,
        estimator: Optional[ClassifierMixin] = None,
        cv: Optional[Union[str, BaseCrossValidator, BaseShuffleSplit]] = None,
    ) -> RegressorMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LogisticRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator: Optional[ClassifierMixin]
            Estimator to check, by default ``None``.

        Returns
        -------
        ClassifierMixin
            The estimator itself or a default ``LogisticRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no ``fit`` nor ``predict`` nor ``predict_proba`` methods.

        NotFittedError
            If the estimator is not fitted
            and ``cv`` attribute is ``"prefit"``.
        """
        if estimator is None:
            estimator = LogisticRegression(multi_class="multinomial")

        if isinstance(estimator, Pipeline):
            est = estimator[-1]
        else:
            est = estimator
        self._check_estimator_fit_predict_predict_proba(est)

        if cv == "prefit":
            check_is_fitted(est)
            if not hasattr(est, "classes_"):
                raise AttributeError(
                    "Invalid classifier. "
                    "Fitted classifier does not contain "
                    "'classes_' attribute."
                )
        return est

    def _check_fit_parameters(self) -> RegressorMixin:
        """
        Check and replace default value of ``predictor`` and ``cv`` arguments.
        Copy the ``predictor`` in ``predictor_`` attribute if ``cv="prefit"``.
        """
        self.cv = self._check_cv(self.cv)
        predictor = self._check_estimator_classification(self.predictor,
                                                         self.cv)
        return predictor

    def _check_calib_conformity_score(
        self, conformity_score: Optional[ConformityScore], sym: bool
    ):
        if not sym:
            raise ValueError("`sym` argument should be set to `True`"
                             "in classification")
        if conformity_score is None:
            return LAC()
        elif isinstance(conformity_score, ConformityScore):
            return conformity_score
        else:
            raise ValueError(
                "Invalid conformity_score argument.\n"
                "Must be None or a ConformityScore instance."
            )

    def _check_calibrate_parameters(self) -> Calibrator:
        """
        Check and replace default ``conformity_score``, ``alpha`` and
        ``calibrator`` arguments.
        """
        self.conformity_score_ = self._check_calib_conformity_score(
            self.conformity_score, self.default_sym_
        )
        calibrator = check_calibrator(self.calibrator)
        self._check_alpha(self.alpha)
        return calibrator

    def predict_score(
        self, predictor: RegressorMixin, X: ArrayLike
    ) -> NDArray:
        """
        Compute conformity scores

        Parameters
        ----------
        predictor : RegressorMixin
            Prediction
        X: ArrayLike
            Observed values.

        Returns
        -------
        NDArray of shape (n_samples, n_classes)
            Predicted probas
        """
        return predictor.predict_proba(X)

    def predict_cs(
        self,
        X: ArrayLike,
        y: ArrayLike,
        y_pred: NDArray,
    ) -> NDArray:
        """
        Compute conformity scores

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y: ArrayLike
            Observed Target

        y_pred: NDArray
            Predicted target.

        Returns
        -------
        NDArray
            Conformity scores on observed data
        """
        return self.conformity_score_.get_conformity_scores(X, y, y_pred)

    def predict_bounds(
        self,
        X: ArrayLike,
        y_pred: NDArray,
        z: Optional[ArrayLike] = None,
    ) -> NDArray:
        """
        Compute conformity scores

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y_pred: 2D NDArray
            Observed Target

        z: ArrayLike
            Exogenous variables

        Returns
        -------
        NDArray
            Prediction sets, as a 3D array of shape (n_samples, n_classes, 1)
            (because we only have 1 alpha value)
        """
        # Classification conformity scores always have ``sym=True``, so
        # the calibrator_.predict result is a 2D array with
        # column 1 = -1 * column 2, So the true values are in res[:, 1]
        conformity_score_pred = self.calibrator_.predict(X, y_pred, z)

        y_pred_set = self.conformity_score_.get_estimation_distribution(
            X, y_pred, conformity_score_pred[:, 1]
        )

        return y_pred_set[:, :, np.newaxis]

    def predict_best(self, y_pred: NDArray) -> NDArray:
        """
        Compute the prediction

        Parameters
        ----------
        y_pred: NDArray
            Prediction scores (can be the prediction, the probas, ...)

        Returns
        -------
        NDArray
            best predictions
        """
        return np.argmax(y_pred, axis=1)

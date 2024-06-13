from __future__ import annotations

from typing import List, Optional, Tuple, Union
import inspect
import numpy as np
from sklearn.base import RegressorMixin

from mapie._typing import ArrayLike, NDArray
from mapie.calibrators.ccp import check_calibrator, CCPCalibrator
from mapie.conformity_scores import ConformityScore
from mapie.futur.split.base import CCP, Calibrator
from mapie.utils import (check_lower_upper_bounds, check_estimator_regression,
                         check_conformity_score)
from sklearn.model_selection import BaseCrossValidator, BaseShuffleSplit
from sklearn.pipeline import Pipeline


class SplitMapieRegressor(CCP):
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
    >>> from mapie.regression import SplitMapieRegressor
    >>> np.random.seed(1)
    >>> X_train = np.arange(0,400, 2).reshape(-1, 1)
    >>> y_train = 2*X_train[:,0] + np.random.rand(len(X_train))
    >>> mapie_reg = SplitMapieRegressor(alpha=0.1, random_state=1)
    >>> mapie_reg = mapie_reg.fit(X_train, y_train)
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    >>> print(np.round(y_pred[:5], 2))
    [ 0.46  4.46  8.46 12.46 16.46]
    >>> print(np.round(y_pis[:5,:, 0], 2))
    [[-0.23  1.15]
     [ 3.77  5.15]
     [ 7.76  9.16]
     [11.76 13.16]
     [15.76 17.16]]
    """
    def __init__(
        self,
        predictor: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        calibrator: Optional[CCPCalibrator] = None,
        cv: Optional[
            Union[str, BaseCrossValidator, BaseShuffleSplit]
        ] = None,
        alpha: Optional[float] = None,
        conformity_score: Optional[ConformityScore] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.random_state = random_state
        self.cv = cv
        self.predictor = predictor
        self.conformity_score = conformity_score
        self.calibrator = calibrator
        self.alpha = alpha

    def _check_fit_parameters(self) -> RegressorMixin:
        """
        Check and replace default value of ``predictor`` and ``cv`` arguments.
        Copy the ``predictor`` in ``predictor_`` attribute if ``cv="prefit"``.
        """
        self.cv = self._check_cv(self.cv)
        predictor = check_estimator_regression(self.predictor, self.cv)
        return predictor

    def _check_calibrate_parameters(self) -> Tuple[
        ConformityScore, Calibrator
    ]:
        """
        Check and replace default ``conformity_score``, ``alpha`` and
        ``calibrator`` arguments.
        """
        conformity_score_ = check_conformity_score(
            self.conformity_score, self.default_sym_
        )
        calibrator = check_calibrator(self.calibrator)
        self.sym = conformity_score_.sym
        self._check_alpha(self.alpha)
        return conformity_score_, calibrator

    def predict_score(
        self, X: ArrayLike
    ) -> NDArray:
        """
        Compute conformity scores

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        Returns
        -------
        NDArray of shape (n_samples, )
            predictions
        """
        return self.predictor_.predict(X)

    def predict_bounds(
        self,
        X: ArrayLike,
        y_pred: NDArray,
        **predict_kwargs,
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
            Bounds, as a 3D array of shape (n_samples, 2, 1)
            (because we only have 1 alpha value)
        """
        predict_kwargs = self.get_method_arguments(
            self.calibrator_.predict, inspect.currentframe(), predict_kwargs,
            ["X"]
        )
        conformity_score_pred = self.calibrator_.predict(X, **predict_kwargs)

        y_pred_low = self.conformity_score_.get_estimation_distribution(
            X, y_pred[:, np.newaxis], conformity_score_pred[:, [0]]
        )
        y_pred_up = self.conformity_score_.get_estimation_distribution(
            X, y_pred[:, np.newaxis], conformity_score_pred[:, [1]]
        )

        check_lower_upper_bounds(y_pred_low, y_pred_up, y_pred)

        return np.stack([y_pred_low, y_pred_up], axis=1)

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
        return y_pred

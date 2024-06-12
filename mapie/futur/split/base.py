from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union, Dict, cast
import warnings

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import (BaseCrossValidator, BaseShuffleSplit,
                                     ShuffleSplit, PredefinedSplit)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import ConformityScore
from mapie.calibrators.ccp import CCP
from mapie.calibrators import Calibrator
from mapie.utils import fit_estimator, _safe_sample


class SplitMapie(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """
    This class implements an adaptative conformal prediction method proposed by
    Gibbs et al. (2023) in "Conformal Prediction With Conditional Guarantees".
    This method works with a ``"split"`` approach which requires a separate
    calibration phase. The ``fit`` method automatically split the data into
    two disjoint sets to train the predictor and the calibrator. You can call
    ``fit_predictor`` and ``fit_calibrator`` to do the two step one after the
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

    References
    ----------
    Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
    "Conformal Prediction With Conditional Guarantees", 2023
    """

    default_sym_ = True
    fit_attributes = ["predictor_"]
    calib_attributes = ["calibrator_"]

    def __init__(
        self,
        predictor: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        calibrator: Optional[CCP] = None,
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

    @abstractmethod
    def _check_fit_parameters(self) -> RegressorMixin:
        """
        Check and replace default value of ``predictor`` and ``cv`` arguments.
        """

    @abstractmethod
    def _check_calibrate_parameters(self) -> Calibrator:
        """
        Check and replace default ``conformity_score``, ``alpha`` and
        ``calibrator`` arguments.
        """

    def _check_cv(
        self,
        cv: Optional[Union[str, BaseCrossValidator, BaseShuffleSplit]] = None,
        test_size: Optional[Union[int, float]] = None,
    ) -> Union[str, BaseCrossValidator, BaseShuffleSplit]:
        """
        Check if ``cv`` is ``None``, ``"prefit"``, ``"split"``,
        or ``BaseShuffleSplit``/``BaseCrossValidator`` with ``n_splits``=1.
        Return a ``ShuffleSplit`` instance ``n_splits``=1
        if ``None`` or ``"split"``.
        Else raise error.

        Parameters
        ----------
        cv: Optional[Union[str, BaseCrossValidator, BaseShuffleSplit]]
            Cross-validator to check, by default ``None``.

        test_size: float
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split.
            If cv is not ``"split"``, ``test_size`` is ignored.

            By default ``None``.

        Returns
        -------
        Union[str, PredefinedSplit, ShuffleSplit]
            The cast `cv` parameter.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if cv is None or cv == "split":
            return ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=self.random_state
            )
        elif (isinstance(cv, (PredefinedSplit, ShuffleSplit))
              and cv.get_n_splits() == 1):
            return cv
        elif cv == "prefit":
            return cv
        else:
            raise ValueError(
                "Invalid cv argument.  Allowed values are None, 'prefit', "
                "'split' or a ShuffleSplit/PredefinedSplit object with "
                "``n_splits=1``."
            )

    def _check_alpha(self, alpha: Optional[float] = None) -> None:
        """
        Check alpha

        Parameters
        ----------
        alpha: Optional[float]
            Can be a float between 0 and 1, represent the uncertainty
            of the confidence interval. Lower alpha produce
            larger (more conservative) prediction intervals.
            alpha is the complement of the target coverage level.

        Raises
        ------
        ValueError
            If alpha is not ``None`` or a float between 0 and 1.
        """
        if alpha is None:
            return
        if isinstance(alpha, float):
            alpha = alpha
        else:
            raise ValueError(
                "Invalid alpha. Allowed values are float."
            )

        if alpha < 0 or alpha > 1:
            raise ValueError("Invalid alpha. "
                             "Allowed values are between 0 and 1.")

    def fit_predictor(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> SplitMapie:
        """
        Fit the predictor if ``cv`` argument is not ``"prefit"``

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        **fit_params: dict
            Additional fit parameters for the predictor.

        Returns
        -------
        SplitMapie
            self
        """
        predictor = self._check_fit_parameters()

        if self.cv != 'prefit':
            self.cv = cast(BaseCrossValidator, self.cv)

            train_index, _ = list(self.cv.split(X, y, groups))[0]

            (
                X_train, y_train, sample_weight_train
            ) = _safe_sample(X, y, sample_weight, train_index)

            self.predictor_ = fit_estimator(
                predictor, X_train, y_train,
                sample_weight=sample_weight_train, **fit_params
            )
        else:
            self.predictor_ = predictor
        return self

    def fit_calibrator(
        self,
        X: ArrayLike,
        y: ArrayLike,
        z: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **optim_kwargs,
    ) -> SplitMapie:
        """
        Calibrate with (``X``, ``y`` and ``z``)
        and the new value ``alpha`` value, if not ``None``

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``

        alpha: Optional[float]
            Between ``0.0`` and ``1.0``, represents the risk level of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            If ``None``, the calibration will be done using the ``alpha``value
            set in the initialisation. Else, the new value will overwrite the
            old one.

            By default ``None``

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        optim_kwargs: Dict
            Other argument, used in sklear.optimize.minimize

        Returns
        -------
        SplitMapie
            self
        """
        self._check_fit_parameters()
        calibrator = self._check_calibrate_parameters()
        check_is_fitted(self, self.fit_attributes)

        # Get calibration set
        if self.cv != 'prefit':
            self.cv = cast(BaseCrossValidator, self.cv)

            _, calib_index = list(self.cv.split(X, y, groups))[0]
            (
                X_calib, y_calib, sample_weight_calib
            ) = _safe_sample(X, y, sample_weight, calib_index)

            if z is not None:
                (
                    z_calib, _, _
                ) = _safe_sample(z, y, sample_weight, calib_index)
            else:
                z_calib = None
        else:
            X_calib, y_calib, z_calib = X, y, z
            sample_weight_calib = cast(Optional[NDArray], sample_weight)

        if alpha is not None and self.alpha != alpha:
            self._check_alpha(alpha)
            self.alpha = alpha
            warnings.warn(f"WARNING: The old value of alpha ({self.alpha}) "
                          f"has been overwritten by the new one ({alpha}).")

        if self.alpha is None:
            warnings.warn("No calibration is done, because alpha is None.")
            return self

        # Compute conformity scores
        y_pred_calib = self.predict_score(self.predictor_, X_calib)

        calib_conformity_scores = self.predict_cs(
            X_calib, y_calib, y_pred_calib
        )

        # Fit the calibrator
        self.calibrator_ = calibrator.fit(
            X_calib, y_pred_calib, z_calib, calib_conformity_scores,
            self.alpha, self.conformity_score_.sym, sample_weight_calib,
            self.random_state, **optim_kwargs,
        )

        return self

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        z: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        fit_params: Optional[Dict] = None,
        calib_params: Optional[Dict] = None,
    ) -> SplitMapie:
        """
        Fit the predictor (if ``cv`` is not ``"prefit"``)
        and fit the calibration.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

            By default ``None``

        alpha: Optional[float]
            Between ``0.0`` and ``1.0``, represents the risk level of the
            confidence interval.
            Lower ``alpha`` produce larger (more conservative) prediction
            intervals.
            ``alpha`` is the complement of the target coverage level.

            If ``None``, the calibration will be done using the ``alpha``value
            set in the initialisation. Else, the new value will overwrite the
            old one.

            By default ``None``

        sample_weight: Optional[ArrayLike] of shape (n_samples,)
            Sample weights for fitting the out-of-fold models.
            If ``None``, then samples are equally weighted.
            If some weights are null,
            their corresponding observations are removed
            before the fitting process and hence have no residuals.
            If weights are non-uniform, residuals are still uniformly weighted.
            Note that the sample weight defined are only for the training, not
            for the calibration procedure.

            By default ``None``.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

        fit_params: dict
            Additional fit parameters for the predictor, used as kwargs.

        calib_params: dict
            Additional fit parameters for the calibrator, used as kwargs.

        Returns
        -------
        SplitMapie
            self
        """
        self.fit_predictor(X, y, sample_weight, groups,
                           **(fit_params if fit_params is not None else {}))
        self.fit_calibrator(X, y, z, alpha, sample_weight, groups,
                            **(calib_params
                               if calib_params is not None else {}))
        return self

    def predict(
        self,
        X: ArrayLike,
        z: Optional[ArrayLike] = None,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        The prediction interval is computed

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        z: Optional[ArrayLike] of shape (n_calib_samples, n_exog_features)
            Exogenous variables

        Returns
        -------
        Union[NDArray, Tuple[NDArray, NDArray]]
            - NDArray of shape (n_samples,) if ``alpha`` is ``None``.
            - Tuple[NDArray, NDArray] of shapes (n_samples,) and
              (n_samples, 2, n_alpha) if ``alpha`` is not ``None``.
                - [:, 0, :]: Lower bound of the prediction interval.
                - [:, 1, :]: Upper bound of the prediction interval.
        """
        check_is_fitted(self, self.fit_attributes)
        y_pred = self.predict_score(self.predictor_, X)

        if self.alpha is None:
            return y_pred

        check_is_fitted(self, self.calib_attributes)

        y_bounds = self.predict_bounds(X, y_pred, z)

        return self.predict_best(y_pred), y_bounds

    @abstractmethod
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
        NDArray
            Scores (usually ``y_pred`` in regression and ``y_pred_proba``
            in classification)
        """

    @abstractmethod
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

    @abstractmethod
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
            Predicted scores (target)

        z: ArrayLike
            Exogenous variables

        Returns
        -------
        NDArray
            Bounds (or prediction set in classification), as a 2D array
        """

    @abstractmethod
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

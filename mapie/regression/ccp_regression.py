from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from scipy.optimize import minimize
from sklearn.base import RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (BaseCrossValidator, BaseShuffleSplit,
                                     ShuffleSplit)
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import _check_y, check_is_fitted, indexable

from mapie._typing import ArrayLike, NDArray
from mapie.conformity_scores import ConformityScore
from .utils.ccp_phi_function import PhiFunction
from mapie.utils import (check_conformity_score, check_estimator_fit_predict,
                         check_lower_upper_bounds, check_null_weight,
                         fit_estimator)


class MapieCCPRegressor():
    """
    This class implements Conformal Prediction With Conditional Guarantees
    method as proposed by Gibbs et al. (2023) to make conformal predictions.
    The only valid cross-val strategy is the "split" approach.

    Parameters
    ----------
    estimator: Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``QuantileRegressor`` instance.

        By default ``"None"``.

    phi: Optional[PhiFunction]
        The phi function used to estimate the conformity scores

        If ``None``, use the default PhiFunction(lambda X: np.ones(len(X))).
        It will result in a constant interval prediction (basic split method).
        See the examples and the documentation to build a PhiFunction
        adaptated to your dataset and constraints.

        By default ``None``.

    cv: Optional[Union[int, str, BaseCrossValidator, BaseShuffleSplit]]
        The cross-validation strategy for computing conformity scores.
        The method only works with a "split" approach.
        Choose among:

        - Any ``sklearn.model_selection.BaseCrossValidator``
          with ``n_splits``=1.
        - ``"split"`` or ``None``, divide the data into training and
          calibration subsets (using the default ``calib_size``=0.3).
          The splitter used is the following:
            ``sklearn.model_selection.ShuffleSplit`` with ``n_splits``=1.
        - ``"prefit"``, assumes that ``estimator`` has been fitted already.
          All data provided in the ``fit`` method is then used
          for the calibration.
          The user has to take care manually that data for model fitting and
          calibration (the data given in the ``fit`` method) are disjoint.

        Note: You can choose the calibration indexes
        with sklearn.model_selection.PredefinedSplit(test_fold),
        where test_fold[i] = 1 (or any not negative integer)
        if the row should be in the calibration set,
        -1 otherwise (if it should be used for training).

        By default ``None``.

    conformity_score: Optional[ConformityScore]
        ConformityScore instance.
        It defines the link between the observed values, the predicted ones
        and the conformity scores. For instance, the default ``None`` value
        correspondonds to a conformity score which assumes
        y_obs = y_pred + conformity_score.

        - ``None``, to use the default ``AbsoluteConformityScore`` conformity
          score
        - Any ``ConformityScore`` class

        By default ``None``.

    alpha: float
        Between ``0.0`` and ``1.0``, represents the risk level of the
        confidence interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default 0.1


    random_state: Optional[int]
        Pseudo random number generator state used for random sampling.
        Pass an int for reproducible output across multiple function calls.

        By default ``None``.

    Attributes
    ----------
    beta_up: Tuple[NDArray, bool]
        Calibration fitting results, used to build the upper bound of the
        prediction intervals.
        beta_up[0]: Array of shape (phi.n_out, )
        beta_up[1]: Whether the optimization process converged or not
                    (the coverage is not garantied if the optimization fail)

    beta_low: Tuple[NDArray, bool]
        Same as beta_up, but for the lower bound

    References
    ----------
    Isaac Gibbs and John J. Cherian and Emmanuel J. Candès.
    "Conformal Prediction With Conditional Guarantees", 2023

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression import MapieCCPRegressor
    >>> X_train = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y_train = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> mapie_reg = MapieCCPRegressor(alpha=0.1, random_state=42)
    >>> mapie_reg.fit_calibrate(
    ...     X_train,
    ...     y_train,
    ... )
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    >>> print(y_pis[:,:, 0])
    [[ 5.    5.8 ]
     [ 6.85  7.65]
     [ 8.7   9.5 ]
     [10.55 11.35]
     [12.4  13.2 ]
     [14.25 15.05]]
    >>> print(y_pred)
    [ 5.4   7.25  9.1  10.95 12.8  14.65]
    """

    default_sym_ = True

    def __init__(
        self,
        estimator: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        phi: Optional[PhiFunction] = None,
        cv: Optional[
            Union[str, BaseCrossValidator, BaseShuffleSplit]
        ] = "split",
        alpha: float = 0.1,
        conformity_score: Optional[ConformityScore] = None,
        random_state: Optional[int] = None,
    ) -> None:

        self.random_state = random_state
        self.cv = self._check_cv(
            cv, random_state=self.random_state
        )
        self.estimator = self._check_estimator(estimator)
        self.conformity_score_ = check_conformity_score(
            conformity_score, self.default_sym_
        )

        if phi is None:
            self.phi = PhiFunction(lambda X: np.ones(len(X)))
        else:
            self.phi = cast(PhiFunction, phi)

        self.alpha = cast(float, self._check_alpha(alpha))
        self.beta_up: Optional[Tuple[NDArray, bool]] = None
        self.beta_low: Optional[Tuple[NDArray, bool]] = None

    def _check_cv(
        self,
        cv: Optional[Union[str, BaseCrossValidator, BaseShuffleSplit]] = None,
        test_size: float = 0.3,
        random_state: Optional[int] = None,
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

        random_state: Optional[int]
            Pseudo random number generator state used for random uniform
            sampling for evaluation quantiles and prediction sets.
            Pass an int for reproducible output across multiple function calls.

            By default ```None``.

        Returns
        -------
        Union[str, BaseCrossValidator, BaseShuffleSplit]
            The cast `cv` parameter.

        Raises
        ------
        ValueError
            If the cross-validator is not valid.
        """
        if random_state is None:
            random_seeds = cast(list, np.random.get_state())[1]
            random_state = np.random.choice(random_seeds)
        if cv is None:
            return ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
        elif isinstance(cv, (BaseCrossValidator, BaseShuffleSplit)):
            try:
                if hasattr(cv, "get_n_splits") and cv.get_n_splits() != 1:
                    raise ValueError(
                        "Invalid cv argument. "
                        "Allowed values are a BaseCrossValidator or "
                        "BaseShuffleSplit object with ``n_splits``=1. "
                        f"Got `n_splits`={cv.get_n_splits()}."
                    )
                return cv
            except (ValueError, TypeError):
                raise ValueError(
                        "Invalid cv argument. "
                        "Allowed values are a BaseCrossValidator or "
                        "BaseShuffleSplit object with ``n_splits``=1."
                    )
        elif cv == "prefit":
            return cv
        elif cv == "split":
            return ShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
        else:
            raise ValueError(
                "Invalid cv argument. "
                "Allowed values are None, 'prefit', 'split' "
                "or a BaseCrossValidator/BaseShuffleSplit "
                "object with ``n_splits``=1."
            )

    def _check_alpha(
        self,
        alpha: Optional[float] = None
    ) -> float:
        """
        Check alpha

        Parameters
        ----------
        alpha: float
            Can be a float between 0 and 1, represent the uncertainty
            of the confidence interval. Lower alpha produce
            larger (more conservative) prediction intervals.
            alpha is the complement of the target coverage level.

        Returns
        -------
        float
            Valid alpha.

        Raises
        ------
        ValueError
            If alpha is not a float between 0 and 1.

        """
        if isinstance(alpha, float):
            alpha = alpha
        else:
            raise ValueError(
                "Invalid alpha. Allowed values are float."
            )

        if alpha < 0 or alpha > 1:
            raise ValueError("Invalid alpha. "
                             "Allowed values are between 0 and 1.")
        return alpha

    def _check_estimator(
        self, estimator: Optional[RegressorMixin] = None
    ) -> RegressorMixin:
        """
        Check if estimator is ``None``,
        and returns a ``LinearRegression`` instance if necessary.
        If the ``cv`` attribute is ``"prefit"``,
        check if estimator is indeed already fitted.

        Parameters
        ----------
        estimator: Optional[RegressorMixin]
            Estimator to check, by default ``None``.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``LinearRegression`` instance.

        Raises
        ------
        ValueError
            If the estimator is not ``None``
            and has no ``fit`` nor ``predict`` methods.

        NotFittedError
            If the estimator is not fitted
            and ``cv`` attribute is ``"prefit"``.
        """
        if estimator is None:
            return LinearRegression()
        else:
            check_estimator_fit_predict(estimator)
            if self.cv == "prefit":
                try:
                    if isinstance(estimator, Pipeline):
                        check_is_fitted(estimator[-1])
                    else:
                        check_is_fitted(estimator)
                except NotFittedError as exc:
                    raise NotFittedError(
                        "You are using cv='prefit' with an estimator "
                        "which is not fitted yet.\n"
                        "Fit the estimator first, or change the "
                        "cv argument value."
                    ) from exc
            return estimator

    def _check_fit_parameters(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ):
        """
        Validate sample_weight

        Parameters
        ----------
        X: ArrayLike
            Observed values.

        y: ArrayLike
            Target values.

        sample_weight: Optional[NDArray] of shape (n_samples,)
            Non-null sample weights.

        Returns
        -------
        NDArray
            The X observed values.

        NDArray
            the y target values.

        Optional[NDArray] of shape (n_samples,)
            Validated Non-null sample weights.

        """
        # Checking

        X, y = indexable(X, y)
        y = _check_y(y)
        sample_weight, X, y = check_null_weight(sample_weight, X, y)

        X = cast(NDArray, X)
        y = cast(NDArray, y)
        sample_weight = cast(Optional[NDArray], sample_weight)

        return (
            X, y,
            sample_weight
        )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **fit_params,
    ) -> None:
        """
        Fit the estimator.

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
            Additional fit parameters for the estimator.

        """

        if self.cv != 'prefit':
            train_index = list(
                cast(BaseCrossValidator, self.cv).split(X, y, groups)
            )[0][0]
            X_train = cast(NDArray, _safe_indexing(X, train_index))
            y_train = cast(NDArray, _safe_indexing(y, train_index))

            if sample_weight is not None:
                sample_weight_train = cast(
                    NDArray, _safe_indexing(sample_weight, train_index)
                )
            else:
                sample_weight_train = None

            (X_train,
                y_train,
                sample_weight_train) = self._check_fit_parameters(
                X_train, y_train, sample_weight_train
            )
            fit_estimator(self.estimator, X_train, y_train,
                          sample_weight=sample_weight_train, **fit_params)

        else:
            warnings.warn("WARNING: As cv='prefit', the estimator will not "
                          "be fitted again. You can directly call the"
                          "calibrate method.")

    def calibrate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Calibrate with (``X``, ``y`` and ``z``)
        and the new value ``alpha`` value, if not ``None``

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Training data.

        y: ArrayLike of shape (n_samples,)
            Training labels.

        groups: Optional[ArrayLike] of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

            By default ``None``.

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

        """
        if self.cv != 'prefit':
            try:
                if isinstance(self.estimator, Pipeline):
                    check_is_fitted(self.estimator[-1])
                else:
                    check_is_fitted(self.estimator)
            except NotFittedError as exc:
                raise NotFittedError("As you are using an estimator which is "
                                     "not fitted yet, you need to call the "
                                     "fit method before calibrate.") from exc

            calib_index = list(
                cast(BaseCrossValidator, self.cv).split(X, y, groups)
            )[0][1]
            X_calib = cast(NDArray, _safe_indexing(X, calib_index))
            y_calib = cast(NDArray, _safe_indexing(y, calib_index))
            if z is not None:
                z_calib = cast(NDArray, _safe_indexing(z, calib_index))
            else:
                z_calib = None
        else:
            X_calib = cast(NDArray, X)
            y_calib = cast(NDArray, y)
            if z is not None:
                z_calib = cast(NDArray, z)
            else:
                z_calib = None

        if alpha is not None:
            if self.alpha != alpha:
                self.alpha = self._check_alpha(alpha)
                warnings.warn(f"WARNING: The old value of alpha "
                              f"({self.alpha}) has been overwritten "
                              f"by the new one ({alpha}).")

        y_pred_calib = self.estimator.predict(X_calib)

        calib_conformity_scores = \
            self.conformity_score_.get_conformity_scores(
                X_calib, y_calib, y_pred_calib
            )

        if self.conformity_score_.sym:
            alpha_low = 1 - self.alpha
            alpha_up = 1 - self.alpha
        else:
            alpha_low = self.alpha / 2
            alpha_up = 1 - self.alpha / 2

        def l_alpha(alpha, X, S):
            return np.where(S >= X, (1 - alpha) * (S - X), alpha * (X - S))

        def sum_of_losses(beta, phi_x, S, alpha):
            return np.sum(l_alpha(alpha, phi_x.dot(beta), S))

        phi_x = self.phi(
            X_calib,
            cast(NDArray, y_pred_calib),
            cast(NDArray, z_calib),
        )

        if np.any(np.all(phi_x == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation "
                          "phi(X, y_pred, z) is full of zeros. "
                          "It will result in a prediction interval of zero "
                          "width. Consider changing the PhiFunction "
                          "definintion.\n"
                          "Fix: Use `marginal_guarantee`=True in PhiFunction")

        not_nan_index = np.where(~np.isnan(calib_conformity_scores))[0]
        # Some conf. score values may be nan (ex: with ResidualNormalisedScore)

        if self.random_state is None:
            warnings.warn("WARNING: The method implemented in "
                          "MapieCCPRegressor has a stochastic behavior. "
                          "To have reproductible results, use a integer "
                          "`random_state` value in the MapieCCPRegressor "
                          "initialisation.")
        else:
            np.random.seed(self.random_state)

        optimal_beta_up = minimize(
            sum_of_losses, np.random.normal(0, 1, self.phi.n_out),
            args=(
                phi_x[not_nan_index, :],
                calib_conformity_scores[not_nan_index],
                1-alpha_up
                )
            )

        if not self.conformity_score_.sym:
            optimal_beta_low = minimize(
                sum_of_losses, np.random.normal(0, 1, self.phi.n_out),
                args=(
                    phi_x[not_nan_index, :],
                    calib_conformity_scores[not_nan_index],
                    1-alpha_low
                )
            )
        else:
            optimal_beta_low = optimal_beta_up

        if not optimal_beta_up.success:
            warnings.warn(
                "WARNING: The optimization process for the upper bound with "
                f"alpha={self.alpha} failed with the following error: \n"
                f"{optimal_beta_low.message}\n"
                "The returned prediction interval may be inaccurate."
            )
        if (not self.conformity_score_.sym
           and not optimal_beta_low.success):
            warnings.warn(
                "WARNING: The optimization process for the lower bound with "
                f"alpha={self.alpha} failed with the following error: \n"
                f"{optimal_beta_low.message}\n"
                "The returned prediction interval may be inaccurate."
            )

        self.beta_up = (cast(NDArray, optimal_beta_up.x),
                        cast(bool, optimal_beta_up.success))
        self.beta_low = (cast(NDArray, optimal_beta_low.x),
                         cast(bool, optimal_beta_low.success))

    def fit_calibrate(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        z: Optional[ArrayLike] = None,
        alpha: Optional[float] = None,
        **fit_params,
    ) -> None:
        """
        Fit the estimator and the calibration.

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

        **fit_params: dict
            Additional fit parameters for the estimator.

        """
        self.fit(X, y, sample_weight, groups, **fit_params)
        self.calibrate(X, y, groups, z, alpha)

    def predict(
        self,
        X: ArrayLike,
        z: Optional[ArrayLike] = None,
    ) -> Tuple[NDArray, NDArray]:
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
        Tuple[NDArray, NDArray]
            Tuple[NDArray, NDArray] of shapes (n_samples,)
            and (n_samples, 2, 1).
              - [:, 0, 0]: Lower bound of the prediction interval.
              - [:, 1, 0]: Upper bound of the prediction interval.
        """

        if self.beta_low is None or self.beta_up is None:
            raise NotFittedError(
                "The calibration method has not been fitted yet.\n"
                "You must call the calibrate method before predict."
            )

        y_pred = self.estimator.predict(X)

        X = cast(NDArray, X)
        y_pred = cast(NDArray, y_pred)
        z = cast(NDArray, z)

        phi_x = self.phi(X, y_pred, z)
        if np.any(np.all(phi_x == 0, axis=1)):
            warnings.warn("WARNING: At least one row of the transformation"
                          "phi(X, y_pred, z) is full of zeros."
                          "It will result in a prediction interval of zero"
                          "width. Consider changing the PhiFunction"
                          "definintion. \n"
                          "Fix: Use `marginal_guarantee`=True in PhiFunction")

        signed = -1 if self.conformity_score_.sym else 1

        y_pred_low = self.conformity_score_.get_estimation_distribution(
            X, y_pred[:, np.newaxis],
            phi_x.dot(signed * self.beta_low[0][:, np.newaxis])
        )
        y_pred_up = self.conformity_score_.get_estimation_distribution(
            X, y_pred[:, np.newaxis],
            phi_x.dot(self.beta_up[0][:, np.newaxis])
        )

        check_lower_upper_bounds(y_pred_low, y_pred_up, y_pred)

        return y_pred, np.stack([y_pred_low, y_pred_up], axis=1)

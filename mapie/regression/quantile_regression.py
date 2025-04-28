from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union, cast, Any
from typing_extensions import Self

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from sklearn.utils.validation import (_check_y, _num_samples, check_is_fitted,
                                      indexable)

from numpy.typing import ArrayLike, NDArray
from mapie.utils import (_check_alpha_and_n_samples,
                         _check_defined_variables_predict_cqr,
                         _check_estimator_fit_predict, _check_lower_upper_bounds,
                         _check_null_weight, _fit_estimator)

from .regression import _MapieRegressor
from mapie.utils import (
    _cast_predictions_to_ndarray_tuple,
    _prepare_params,
    _prepare_fit_params_and_sample_weight,
    _raise_error_if_previous_method_not_called,
    _raise_error_if_method_already_called,
    _raise_error_if_fit_called_in_prefit_mode, _transform_confidence_level_to_alpha,
)


class ConformalizedQuantileRegressor:
    """
    Computes prediction intervals using the conformalized quantile regression technique:

    1. The ``fit`` method fits three models to the training data using the provided
       regressor: a model to predict the target, and models to predict upper
       and lower quantiles around the target.
    2. The ``conformalize`` method estimates the uncertainty of the quantile models
       using the conformity set.
    3. The ``predict_interval`` computes prediction points and intervals.

    Parameters
    ----------
    estimator : Union[``RegressorMixin``, ``Pipeline``, \
``List[Union[RegressorMixin, Pipeline]]``]
        The regressor used to predict points and quantiles.

        When ``prefit=False`` (default), a single regressor that supports the quantile
        loss must be passed. Valid options:

        - ``sklearn.linear_model.QuantileRegressor``
        - ``sklearn.ensemble.GradientBoostingRegressor``
        - ``sklearn.ensemble.HistGradientBoostingRegressor``
        - ``lightgbm.LGBMRegressor``

        When ``prefit=True``, a list of three fitted quantile regressors predicting the
        lower, upper, and median quantiles must be passed (in that order).
        These quantiles must be:

        - ``lower quantile = (1 - confidence_level) / 2``
        - ``upper quantile = (1 + confidence_level) / 2``
        - ``median quantile = 0.5``

    confidence_level : float default=0.9
        The confidence level for the prediction intervals, indicating the
        desired coverage probability of the prediction intervals.

    prefit : bool, default=False
        If True, three fitted quantile regressors must be provided, and the ``fit``
        method must be skipped.

        If False, the three regressors will be fitted during the ``fit`` method.

    Examples
    --------
    >>> from mapie.regression import ConformalizedQuantileRegressor
    >>> from mapie.utils import train_conformalize_test_split
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import QuantileRegressor

    >>> X, y = make_regression(n_samples=500, n_features=2, noise=1.0)
    >>> (
    ...     X_train, X_conformalize, X_test,
    ...     y_train, y_conformalize, y_test
    ... ) = train_conformalize_test_split(
    ...     X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=1
    ... )

    >>> mapie_regressor = ConformalizedQuantileRegressor(
    ...     estimator=QuantileRegressor(),
    ...     confidence_level=0.95,
    ... ).fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)

    >>> predicted_points, predicted_intervals = mapie_regressor.predict_interval(X_test)
    """

    def __init__(
        self,
        estimator: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        confidence_level: float = 0.9,
        prefit: bool = False,
    ) -> None:
        self._alpha = _transform_confidence_level_to_alpha(confidence_level)
        self._prefit = prefit
        self._is_fitted = prefit
        self._is_conformalized = False

        self._mapie_quantile_regressor = _MapieQuantileRegressor(
            estimator=estimator,
            method="quantile",
            cv="prefit" if prefit else "split",
            alpha=self._alpha,
        )

        self._sample_weight: Optional[ArrayLike] = None
        self._predict_params: dict = {}

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        fit_params: Optional[dict] = None,
    ) -> Self:
        """
        Fits three models using the regressor provided at initialisation:

        - a model to predict the target
        - a model to predict the upper quantile of the target
        - a model to predict the lower quantile of the target

        Parameters
        ----------
        X_train : ArrayLike
            Training data features.

        y_train : ArrayLike
            Training data targets.

        fit_params : Optional[dict], default=None
            Parameters to pass to the ``fit`` method of the regressors.

        Returns
        -------
        Self
            The fitted ConformalizedQuantileRegressor instance.
        """
        _raise_error_if_fit_called_in_prefit_mode(self._prefit)
        _raise_error_if_method_already_called("fit", self._is_fitted)

        fit_params_, self._sample_weight = _prepare_fit_params_and_sample_weight(
            fit_params
        )
        self._mapie_quantile_regressor._initialize_fit_conformalize()
        self._mapie_quantile_regressor._fit_estimators(
            X=X_train,
            y=y_train,
            sample_weight=self._sample_weight,
            **fit_params_,
        )

        self._is_fitted = True
        return self

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> Self:
        """
        Estimates the uncertainty of the quantile regressors by computing
        conformity scores on the conformity set.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Features of the conformity set.

        y_conformalize : ArrayLike
            Targets of the conformity set.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` method of the regressors.
            These parameters will also be used in the ``predict_interval``
            and ``predict`` methods of this SplitConformalRegressor.

        Returns
        -------
        Self
            The ConformalizedQuantileRegressor instance.
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
        self._mapie_quantile_regressor.conformalize(
            X_conformalize,
            y_conformalize,
            **self._predict_params
        )

        self._is_conformalized = True
        return self

    def predict_interval(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
        symmetric_correction: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predicts points (using the base regressor) and intervals.

        The returned NDArray containing the prediction intervals is of shape
        (n_samples, 2, 1). The third dimension is unnecessary, but kept for consistency
        with the other conformal regression methods available in MAPIE.

        Parameters
        ----------
        X : ArrayLike
            Features

        minimize_interval_width : bool, default=False
            If True, attempts to minimize the intervals width.

        allow_infinite_bounds : bool, default=False
            If True, allows prediction intervals with infinite bounds.

        symmetric_correction : bool, default=False
            To produce prediction intervals, the conformalized quantile regression
            technique corrects the predictions of the upper and lower quantile
            regressors by adding a constant.

            If ``symmetric_correction`` is set to ``False`` , this constant is different
            for the upper and the lower quantile predictions. If set to ``True``,
            this constant is the same for both.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape ``(n_samples,)``
            - Prediction intervals, of shape ``(n_samples, 2, 1)``
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval",
            "conformalize",
            self._is_conformalized,
        )

        predictions = self._mapie_quantile_regressor.predict(
            X,
            optimize_beta=minimize_interval_width,
            allow_infinite_bounds=allow_infinite_bounds,
            symmetry=symmetric_correction,
            **self._predict_params
        )
        return _cast_predictions_to_ndarray_tuple(predictions)

    def predict(
        self,
        X: ArrayLike,
    ) -> NDArray:
        """
        Predicts points.

        Parameters
        ----------
        X : ArrayLike
            Features

        Returns
        -------
        NDArray
            Array of point predictions with shape ``(n_samples,)``.
        """
        _raise_error_if_previous_method_not_called(
            "predict",
            "conformalize",
            self._is_conformalized,
        )

        estimator = self._mapie_quantile_regressor
        predictions, _ = estimator.predict(X, **self._predict_params)
        return predictions


class _MapieQuantileRegressor(_MapieRegressor):
    """
    Note to users: _MapieQuantileRegressor is now private, and may change at any time.
    Please use ConformalizedQuantileRegressor instead.
    See the v1 migration guide for more information.

    This class implements the conformalized quantile regression strategy
    as proposed by Romano et al. (2019) to make conformal predictions.

    Parameters
    ----------
    estimator : Optional[RegressorMixin]
        Any regressor with scikit-learn API
        (i.e. with ``fit`` and ``predict`` methods).
        If ``None``, estimator defaults to a ``QuantileRegressor`` instance.

        By default ``"None"``.

    method: str
        Method to choose for prediction, in this case, the only valid method
        is the ``"quantile"`` method.

        By default ``"quantile"``.

    cv: Optional[str]
        The cross-validation strategy for computing conformity scores.
        In theory a split method is implemented as it is needed to provide
        both a training and calibration set.

        By default ``None``.

    alpha: float
        Between ``0.0`` and ``1.0``, represents the risk level of the
        confidence interval.
        Lower ``alpha`` produce larger (more conservative) prediction
        intervals.
        ``alpha`` is the complement of the target coverage level.

        By default ``0.1``.

    Attributes
    ----------
    valid_methods_: List[str]
        List of all valid methods.

    single_estimator_: RegressorMixin
        Estimator fitted on the whole training set.

    estimators_: List[RegressorMixin]
        - [0]: Estimator with quantile value of alpha/2
        - [1]: Estimator with quantile value of 1 - alpha/2
        - [2]: Estimator with quantile value of 0.5

    conformity_scores_: NDArray of shape (n_samples_train, 3)
        Conformity scores between ``y_calib`` and ``y_pred``.

        - [:, 0]: for ``y_calib`` coming from prediction estimator
          with quantile of alpha/2
        - [:, 1]: for ``y_calib`` coming from prediction estimator
          with quantile of 1 - alpha/2
        - [:, 2]: maximum of those first two scores

    n_calib_samples: int
        Number of samples in the calibration dataset.

    References
    ----------
    Yaniv Romano, Evan Patterson and Emmanuel J. Candès.
    "Conformalized Quantile Regression"
    Advances in neural information processing systems 32 (2019).

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.regression.quantile_regression import _MapieQuantileRegressor
    >>> X_train = np.array([[0], [1], [2], [3], [4], [5]])
    >>> y_train = np.array([5, 7.5, 9.5, 10.5, 12.5, 15])
    >>> X_calib = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_calib = np.array([5, 7, 9, 4, 8, 1, 5, 7.5, 9.5, 12])
    >>> mapie_reg = _MapieQuantileRegressor().fit(
    ...     X_train,
    ...     y_train,
    ...     X_calib=X_calib,
    ...     y_calib=y_calib
    ... )
    >>> y_pred, y_pis = mapie_reg.predict(X_train)
    >>> print(y_pis[:, :, 0])
    [[-8.16666667 19.        ]
     [-6.33333333 20.83333333]
     [-4.5        22.66666667]
     [-2.66666667 24.5       ]
     [-0.83333333 26.33333333]
     [ 1.         28.16666667]]
    >>> print(y_pred)
    [ 5.  7.  9. 11. 13. 15.]
    """
    valid_methods_ = ["quantile"]
    fit_attributes = [
        "estimators_",
        "conformity_scores_",
        "n_calib_samples",
    ]

    quantile_estimator_params = {
        "GradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "alpha"
        },
        "QuantileRegressor": {
            "loss_name": "quantile",
            "alpha_name": "quantile"
        },
        "HistGradientBoostingRegressor": {
            "loss_name": "loss",
            "alpha_name": "quantile"
        },
        "LGBMRegressor": {
            "loss_name": "objective",
            "alpha_name": "alpha"
        },
    }

    def __init__(
        self,
        estimator: Optional[
            Union[
                RegressorMixin,
                Pipeline,
                List[Union[RegressorMixin, Pipeline]]
            ]
        ] = None,
        method: str = "quantile",
        cv: Optional[str] = None,
        alpha: float = 0.1,
    ) -> None:
        super().__init__(
            estimator=estimator,
            method=method,
        )
        self.cv = cv
        self.alpha = alpha

    def _check_alpha(
        self,
        alpha: float = 0.1,
    ) -> NDArray:
        """
        Perform several checks on the alpha value and changes it from
        a float to an ArrayLike.

        Parameters
        ----------
        alpha : float
            Can only be a float value between ``0.0`` and ``1.0``.
            Represent the risk level of the confidence interval.
            Lower alpha produce larger (more conservative) prediction
            intervals. Alpha is the complement of the target coverage level.

            By default ``0.1``.

        Returns
        -------
        ArrayLike
            An ArrayLike of three values:

            - [0]: alpha value of alpha/2
            - [1]: alpha value of of 1 - alpha/2
            - [2]: alpha value of 0.5

        Raises
        ------
        ValueError
            If alpha is not a float.

        ValueError
            If the value of ``alpha`` is not between ``0.0`` and ``1.0``.
        """
        if isinstance(alpha, float):
            if np.any(np.logical_or(alpha <= 0, alpha >= 1.0)):
                raise ValueError(
                    "Invalid confidence_level. "
                    "Allowed values are between 0.0 and 1.0."
                )
            else:
                alpha_np = np.array([alpha / 2, 1 - alpha / 2, 0.5])
        else:
            raise ValueError(
                "Invalid confidence_level. Allowed values are float."
            )
        return alpha_np

    def _check_estimator(
        self,
        estimator: Optional[Union[RegressorMixin, Pipeline]] = None,
    ) -> Union[RegressorMixin, Pipeline]:
        """
        Perform several checks on the estimator to check if it has
        all the required specifications to be used with this methodology.
        The estimators that can be used in _MapieQuantileRegressor need to
        have a ``fit`` and ``predict`` attribute, but also need to allow
        a quantile loss and therefore also setting a quantile value.
        Note that there is a ``TypedDict`` to check which methods allow for
        quantile regression.

        Parameters
        ----------
        estimator : Optional[RegressorMixin], optional
            Estimator to check, by default ``None``.

        Returns
        -------
        RegressorMixin
            The estimator itself or a default ``QuantileRegressor`` instance
            with ``solver`` set to "highs".

        Raises
        ------
        ValueError
            If the estimator implements ``fit`` or ``predict`` methods.

        ValueError
            We check if it's a known estimator that does quantile regression
            according to the dictionnary set quantile_estimator_params.
            This dictionnary will need to be updated with the latest new
            available estimators.

        ValueError
            The estimator does not have the ``"loss_name"`` in its parameters
            and therefore can not be used as an estimator.

        ValueError
            There is no quantile ``"loss_name"`` and therefore this estimator
            can not be used as a ``_MapieQuantileRegressor``.

        ValueError
            The parameter to set the alpha value does not exist in this
            estimator and therefore we cannot use it.
        """
        if estimator is None:
            return QuantileRegressor(
                solver="highs-ds",
                alpha=0.0,
            )
        _check_estimator_fit_predict(estimator)
        if isinstance(estimator, Pipeline):
            self._check_estimator(estimator[-1])
            return estimator
        else:
            name_estimator = estimator.__class__.__name__
            if name_estimator == "QuantileRegressor":
                return estimator
            else:
                if name_estimator in self.quantile_estimator_params:
                    param_estimator = estimator.get_params()
                    loss_name, alpha_name = self.quantile_estimator_params[
                        name_estimator
                    ].values()
                    if loss_name in param_estimator:
                        if param_estimator[loss_name] != "quantile":
                            raise ValueError(
                                "You need to set the loss/objective argument"
                                + " of your base model to ``quantile``."
                            )
                        else:
                            if alpha_name in param_estimator:
                                return estimator
                            else:
                                raise ValueError(
                                    "The matching parameter `alpha_name` for"
                                    " estimator does not exist. "
                                    "Make sure you set it when initializing "
                                    "your estimator."
                                )
                    else:
                        raise ValueError(
                            "The matching parameter `loss_name` for"
                            + " estimator does not exist."
                        )
                else:
                    raise ValueError(
                        "The base model is not supported. \n"
                        "Give a base model among: \n"
                        f"{self.quantile_estimator_params.keys()} "
                        "Or, add your base model to"
                        + " ``quantile_estimator_params``."
                    )

    def _check_cv(
        self,
        cv: Optional[str] = None
    ) -> str:
        """
        Check if cv argument is ``None``, ``"split"`` or ``"prefit"``.

        Parameters
        ----------
        cv : Optional[str], optional
           cv to check, by default ``None``.

        Returns
        -------
        str
            cv itself or a default ``"split"``.

        Raises
        ------
        ValueError
            Raises an error if the cv is anything else but the method
            ``"split"`` or ``"prefit"``.
            Only the split method has been implemented.
        """
        if cv is None:
            return "split"
        if cv in ("split", "prefit"):
            return cv
        else:
            raise ValueError(
                "Invalid cv method, only valid method is ``split``."
            )

    def _train_calib_split(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState, None]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]
    ]:
        if sample_weight is None:
            X_train, X_calib, y_train, y_calib = train_test_split(
                X,
                y,
                test_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify
            )
            sample_weight_train = sample_weight
        else:
            (
                X_train,
                X_calib,
                y_train,
                y_calib,
                sample_weight_train,
                _,
            ) = train_test_split(
                X,
                y,
                sample_weight,
                test_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify
            )
        return X_train, y_train, X_calib, y_calib, sample_weight_train

    def _check_prefit_params(
        self,
        estimator: List[Union[RegressorMixin, Pipeline]],
    ) -> None:
        """
        Check the parameters set for the specific case of prefit
        estimators.

        Parameters
        ----------
        estimator : List[Union[RegressorMixin, Pipeline]]
            List of three prefitted estimators that should have
            pre-defined quantile levels of alpha/2, 1 - alpha/2 and 0.5.

        Raises
        ------
        ValueError
            If a non-iterable variable is provided for estimator.

        ValueError
            If less or more than three models are defined.

        Warning
            If the alpha is defined, warns the user that it must be set
            accordingly with the prefit estimators.
        """
        if isinstance(estimator, Iterable) is False:
            raise ValueError(
                "Estimator for prefit must be an iterable object."
            )
        if len(estimator) == 3:
            for est in estimator:
                _check_estimator_fit_predict(est)
                check_is_fitted(est)
        else:
            raise ValueError(
                "You need to have provided 3 different estimators, they"
                " need to be preset with alpha values"
                "(alpha = 1 - confidence_level)"
                "in the following order [alpha/2, 1 - alpha/2, 0.5]."
            )

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        X_calib: Optional[ArrayLike] = None,
        y_calib: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
        **fit_params,
    ) -> _MapieQuantileRegressor:
        """
        Fit estimator and compute residuals used for prediction intervals.
        All the clones of the estimators for different quantile values are
        stored in order alpha/2, 1 - alpha/2, 0.5 in the ``estimators_``
        attribute. Residuals for the first two estimators and the maximum
        of residuals among these residuals are stored in the
        ``conformity_scores_`` attribute.

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
            Always ignored, exists for compatibility.

        X_calib: Optional[ArrayLike] of shape (n_calib_samples, n_features)
            Calibration data.

        y_calib: Optional[ArrayLike] of shape (n_calib_samples,)
            Calibration labels.

        calib_size: Optional[float]
            If ``X_calib`` and ``y_calib`` are not defined,
            then the calibration dataset is created with the split
            defined by ``calib_size``.

        random_state: Optional[Union[int, np.random.RandomState]], default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Controls the shuffling applied to the data before applying the
            split.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

            By default ``None``.

        shuffle: bool, default=True
            For the ``sklearn.model_selection.train_test_split`` documentation.
            Whether or not to shuffle the data before splitting.
            If ``shuffle=False`` then stratify must be None.

            By default ``True``.

        stratify: array-like, default=None
            For the ``sklearn.model_selection.train_test_split`` documentation.
            If not ``None``, data is split in a stratified fashion, using this
            as the class labels.
            Read more in the :ref:`User Guide <stratification>`.

            By default ``None``.

        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        _MapieQuantileRegressor
             The model itself.
        """
        self._initialize_fit_conformalize()

        if self.cv == "prefit":
            X_calib, y_calib = X, y
        else:
            result = self._prepare_train_calib(
                X=X,
                y=y,
                sample_weight=sample_weight,
                groups=groups,
                X_calib=X_calib,
                y_calib=y_calib,
                calib_size=calib_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify,
            )
            X_train, y_train, X_calib, y_calib, sample_weight = result
            self._fit_estimators(
                X=X_train,
                y=y_train,
                sample_weight=sample_weight,
                **fit_params
            )

        self.conformalize(X_calib, y_calib)

        return self

    def _initialize_fit_conformalize(self) -> None:
        self.cv = self._check_cv(cast(str, self.cv))
        self.alpha_np = self._check_alpha(self.alpha)
        self.estimators_: List[RegressorMixin] = []

    def _initialize_and_check_prefit_estimators(self) -> None:
        estimator = cast(List, self.estimator)
        self._check_prefit_params(estimator)
        self.estimators_ = list(estimator)
        self.single_estimator_ = self.estimators_[2]

    def _prepare_train_calib(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        X_calib: Optional[ArrayLike] = None,
        y_calib: Optional[ArrayLike] = None,
        calib_size: Optional[float] = 0.3,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        shuffle: Optional[bool] = True,
        stratify: Optional[ArrayLike] = None,
    ) -> Tuple[
        ArrayLike, ArrayLike, ArrayLike, ArrayLike, Optional[ArrayLike]
    ]:
        """
        Handles the preparation of training and calibration datasets,
        including validation and splitting.
        Returns: X_train, y_train, X_calib, y_calib, sample_weight_train
        """
        self._check_parameters()
        random_state = check_random_state(random_state)
        X, y = indexable(X, y)

        if X_calib is None or y_calib is None:
            return self._train_calib_split(
                X,
                y,
                sample_weight,
                calib_size,
                random_state,
                shuffle,
                stratify
            )
        else:
            return X, y, X_calib, y_calib, sample_weight

    # Second function: Handles estimator fitting
    def _fit_estimators(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        **fit_params
    ) -> None:
        """
        Fits the estimators with provided training data
        and stores them in self.estimators_.
        """
        checked_estimator = self._check_estimator(self.estimator)

        X, y = indexable(X, y)
        y = _check_y(y)

        sample_weight, X, y = _check_null_weight(
            sample_weight, X, y
        )

        if isinstance(checked_estimator, Pipeline):
            estimator = checked_estimator[-1]
        else:
            estimator = checked_estimator

        name_estimator = estimator.__class__.__name__
        alpha_name = self.quantile_estimator_params[name_estimator][
            "alpha_name"
        ]

        for i, alpha_ in enumerate(self.alpha_np):
            cloned_estimator_ = clone(checked_estimator)
            params = {alpha_name: alpha_}
            if isinstance(checked_estimator, Pipeline):
                cloned_estimator_[-1].set_params(**params)
            else:
                cloned_estimator_.set_params(**params)

            self.estimators_.append(
                _fit_estimator(
                    cloned_estimator_,
                    X,
                    y,
                    sample_weight,
                    **fit_params,
                )
            )

        self.single_estimator_ = self.estimators_[2]

    def conformalize(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        # Parameter groups kept for compliance with superclass _MapieRegressor
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> _MapieRegressor:
        if self.cv == "prefit":
            self._initialize_and_check_prefit_estimators()

        X_calib, y_calib = cast(ArrayLike, X), cast(ArrayLike, y)
        X_calib, y_calib = indexable(X_calib, y_calib)
        y_calib = _check_y(y_calib)

        self.n_calib_samples = _num_samples(y_calib)
        _check_alpha_and_n_samples(self.alpha, self.n_calib_samples)

        y_calib_preds = np.full(
                shape=(3, self.n_calib_samples),
                fill_value=np.nan
            )

        for i, est in enumerate(self.estimators_):
            y_calib_preds[i] = est.predict(X_calib, **kwargs).ravel()

        self.conformity_scores_ = np.full(
                shape=(3, self.n_calib_samples),
                fill_value=np.nan
            )

        self.conformity_scores_[0] = y_calib_preds[0] - y_calib
        self.conformity_scores_[1] = y_calib - y_calib_preds[1]
        self.conformity_scores_[2] = np.max(
            [
                self.conformity_scores_[0],
                self.conformity_scores_[1]
            ], axis=0
        )
        return self

    def predict(
        self,
        X: ArrayLike,
        ensemble: bool = False,
        alpha: Optional[Union[float, Iterable[float]]] = None,
        optimize_beta: bool = False,
        allow_infinite_bounds: bool = False,
        symmetry: Optional[bool] = True,
        **predict_params,
    ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Predict target on new samples with confidence intervals.
        Residuals from the training set and predictions from the model clones
        are central to the computation.
        Prediction Intervals for a given ``alpha`` are deduced from the
        quantile regression at the alpha values: alpha/2, 1 - (alpha/2)
        while adding a constant based uppon their residuals.

        Parameters
        ----------
        X: ArrayLike of shape (n_samples, n_features)
            Test data.

        ensemble: bool
            Ensemble has not been defined in predict and therefore should
            will not have any effects in this method.

        alpha: Optional[Union[float, Iterable[float]]]
            For ``MapieQuantileRegresor`` the alpha has to be defined
            directly in initial arguments of the class.

        symmetry: Optional[bool]
            Deciding factor to whether to find the quantile value for
            each residuals separatly or to use the maximum of the two
            combined.

        predict_params : dict
            Additional predict parameters.

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
        _check_defined_variables_predict_cqr(ensemble, alpha)
        alpha = self.alpha if symmetry else self.alpha/2
        _check_alpha_and_n_samples(alpha, self.n_calib_samples)

        n = self.n_calib_samples
        q = (1 - (alpha)) * (1 + (1 / n))

        y_preds = np.full(
            shape=(3, _num_samples(X)),
            fill_value=np.nan,
            dtype=float,
        )
        for i, est in enumerate(self.estimators_):
            y_preds[i] = est.predict(X, **predict_params)
        _check_lower_upper_bounds(y_preds[0], y_preds[1], y_preds[2])
        if symmetry:
            quantile = np.full(
                2,
                np.quantile(
                    self.conformity_scores_[2], q, method="higher"
                )
            )
        else:
            quantile = np.array(
                [
                    np.quantile(
                        self.conformity_scores_[0], q, method="higher"
                    ),
                    np.quantile(
                        self.conformity_scores_[1], q, method="higher"
                    )
                ]
            )
        y_pred_low = y_preds[0][:, np.newaxis] - quantile[0]
        y_pred_up = y_preds[1][:, np.newaxis] + quantile[1]
        _check_lower_upper_bounds(y_pred_low, y_pred_up, y_preds[2])
        return y_preds[2], np.stack([y_pred_low, y_pred_up], axis=1)

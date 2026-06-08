import warnings
from inspect import signature
from typing import Any, List, Tuple

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline

from mapie.classification import (
    CrossConformalClassifier,
    SplitConformalClassifier,
    _MapieClassifier,
)
from mapie.regression.quantile_regression import (
    ConformalizedQuantileRegressor,
    _MapieQuantileRegressor,
)
from mapie.regression.regression import (
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    SplitConformalRegressor,
    _MapieRegressor,
)
from mapie.utils import NotFittedError, check_sklearn_user_model_is_fitted

RANDOM_STATE = 1


@pytest.fixture(scope="module")
def dataset_regression():
    X, y = make_regression(
        n_samples=500, n_features=2, noise=1.0, random_state=RANDOM_STATE
    )
    X_train, X_conf_test, y_train, y_conf_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )
    X_conformalize, X_test, y_conformalize, y_test = train_test_split(
        X_conf_test, y_conf_test, random_state=RANDOM_STATE
    )
    return X_train, X_conformalize, X_test, y_train, y_conformalize, y_test


@pytest.fixture(scope="module")
def dataset_classification():
    X, y = make_classification(
        n_samples=500,
        n_informative=5,
        n_classes=4,
        random_state=RANDOM_STATE,
    )
    X_train, X_conf_test, y_train, y_conf_test = train_test_split(
        X, y, random_state=RANDOM_STATE
    )
    X_conformalize, X_test, y_conformalize, y_test = train_test_split(
        X_conf_test, y_conf_test, random_state=RANDOM_STATE
    )
    return X_train, X_conformalize, X_test, y_train, y_conformalize, y_test


def test_scr_same_predictions_prefit_not_prefit(dataset_regression) -> None:
    X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = (
        dataset_regression
    )
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    scr_prefit = SplitConformalRegressor(estimator=regressor, prefit=True)
    scr_prefit.conformalize(X_conformalize, y_conformalize)
    predictions_scr_prefit = scr_prefit.predict_interval(X_test)

    scr_not_prefit = SplitConformalRegressor(estimator=LinearRegression(), prefit=False)
    scr_not_prefit.fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)
    predictions_scr_not_prefit = scr_not_prefit.predict_interval(X_test)
    np.testing.assert_equal(predictions_scr_prefit, predictions_scr_not_prefit)


@pytest.mark.parametrize(
    "split_technique,predict_method,dataset,estimator_class",
    [
        (
            SplitConformalRegressor,
            "predict_interval",
            "dataset_regression",
            DummyRegressor,
        ),
        (
            ConformalizedQuantileRegressor,
            "predict_interval",
            "dataset_regression",
            QuantileRegressor,
        ),
        (
            SplitConformalClassifier,
            "predict_set",
            "dataset_classification",
            DummyClassifier,
        ),
    ],
)
class TestWrongMethodsOrderRaisesErrorForSplitTechniques:
    def test_with_prefit_false(
        self, split_technique, predict_method, dataset, estimator_class, request
    ):
        dataset = request.getfixturevalue(dataset)
        X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
        estimator = estimator_class()
        technique = split_technique(estimator=estimator, prefit=False)

        with pytest.raises(ValueError, match=r"call fit before calling conformalize"):
            technique.conformalize(X_conformalize, y_conformalize)

        technique.fit(X_train, y_train)

        with pytest.raises(ValueError, match=r"fit method already called"):
            technique.fit(X_train, y_train)
        with pytest.raises(
            ValueError, match=r"call conformalize before calling predict"
        ):
            technique.predict(X_test)

        with pytest.raises(
            ValueError, match=f"call conformalize before calling {predict_method}"
        ):
            getattr(technique, predict_method)(X_test)

        technique.conformalize(X_conformalize, y_conformalize)

        with pytest.raises(ValueError, match=r"conformalize method already called"):
            technique.conformalize(X_conformalize, y_conformalize)

    def test_with_prefit_true(
        self, split_technique, predict_method, dataset, estimator_class, request
    ):
        dataset = request.getfixturevalue(dataset)
        X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
        estimator = estimator_class()
        estimator.fit(X_train, y_train)

        if split_technique == ConformalizedQuantileRegressor:
            technique = split_technique(estimator=[estimator] * 3, prefit=True)
        else:
            technique = split_technique(estimator=estimator, prefit=True)

        with pytest.raises(ValueError, match=r"The fit method must be skipped"):
            technique.fit(X_train, y_train)
        with pytest.raises(
            ValueError, match=r"call conformalize before calling predict"
        ):
            technique.predict(X_test)

        with pytest.raises(
            ValueError, match=f"call conformalize before calling {predict_method}"
        ):
            getattr(technique, predict_method)(X_test)

        technique.conformalize(X_conformalize, y_conformalize)

        with pytest.raises(ValueError, match=r"conformalize method already called"):
            technique.conformalize(X_conformalize, y_conformalize)


@pytest.mark.parametrize(
    "cross_technique,predict_method,dataset,estimator_class",
    [
        (
            CrossConformalRegressor,
            "predict_interval",
            "dataset_regression",
            DummyRegressor,
        ),
        (
            JackknifeAfterBootstrapRegressor,
            "predict_interval",
            "dataset_regression",
            DummyRegressor,
        ),
        (
            CrossConformalClassifier,
            "predict_set",
            "dataset_classification",
            DummyClassifier,
        ),
    ],
)
class TestWrongMethodsOrderRaisesErrorForCrossTechniques:
    def test_wrong_methods_order(
        self, cross_technique, predict_method, dataset, estimator_class, request
    ):
        dataset = request.getfixturevalue(dataset)
        X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
        technique = cross_technique(estimator=estimator_class())

        with pytest.raises(
            ValueError, match=r"call fit_conformalize before calling predict"
        ):
            technique.predict(X_test)
        with pytest.raises(
            ValueError, match=f"call fit_conformalize before calling {predict_method}"
        ):
            getattr(technique, predict_method)(X_test)

        technique.fit_conformalize(X_conformalize, y_conformalize)

        with pytest.warns(UserWarning, match=r"fit_conformalize was already called"):
            technique.fit_conformalize(X_conformalize, y_conformalize)


def test_cross_conformal_regressor_rejects_subsample():
    """Test that CrossConformalRegressor raises an error when cv=Subsample().

    Users should use JackknifeAfterBootstrapRegressor for bootstrap-based
    conformal prediction.  See https://github.com/scikit-learn-contrib/MAPIE/issues/924
    """
    from mapie.subsample import Subsample

    with pytest.raises(
        ValueError,
        match=r".*Subsample.*JackknifeAfterBootstrapRegressor.*",
    ):
        CrossConformalRegressor(cv=Subsample())


X_toy = np.arange(18).reshape(-1, 1)
y_toy = np.array([0, 0, 1, 0, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1, 2, 1, 2, 2])


def MapieSimpleEstimators() -> List[BaseEstimator]:
    return [_MapieRegressor, _MapieClassifier]


def MapieEstimators() -> List[BaseEstimator]:
    return [_MapieRegressor, _MapieClassifier, _MapieQuantileRegressor]


def MapieDefaultEstimators() -> List[BaseEstimator]:
    return [
        (_MapieRegressor, LinearRegression),
        (_MapieClassifier, LogisticRegression),
    ]


def MapieTestEstimators() -> List[BaseEstimator]:
    return [
        (_MapieRegressor, LinearRegression()),
        (_MapieRegressor, make_pipeline(LinearRegression())),
        (_MapieClassifier, LogisticRegression()),
        (_MapieClassifier, make_pipeline(LogisticRegression())),
    ]


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_initialized(MapieEstimator: BaseEstimator) -> None:
    """Test that initialization does not crash."""
    MapieEstimator()


@pytest.mark.parametrize("MapieEstimator", MapieEstimators())
def test_default_parameters(MapieEstimator: BaseEstimator) -> None:
    """Test default values of input parameters."""
    mapie_estimator = MapieEstimator()
    assert mapie_estimator.estimator is None
    assert mapie_estimator.cv is None
    assert mapie_estimator.verbose == 0
    assert mapie_estimator.n_jobs is None


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_fit(MapieEstimator: BaseEstimator) -> None:
    """Test that fit raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that fit-predict raises no errors."""
    mapie_estimator = MapieEstimator()
    mapie_estimator.fit(X_toy, y_toy)
    mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_no_fit_predict(MapieEstimator: BaseEstimator) -> None:
    """Test that predict before fit raises errors."""
    mapie_estimator = MapieEstimator()
    with pytest.raises(NotFittedError):
        mapie_estimator.predict(X_toy)


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_default_alpha(MapieEstimator: BaseEstimator) -> None:
    """Test default alpha."""
    mapie_estimator = MapieEstimator()
    assert signature(mapie_estimator.predict).parameters["alpha"].default is None


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_estimator(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """Test that None estimator defaults to expected value."""
    MapieEstimator, DefaultEstimator = pack
    mapie_estimator = MapieEstimator(estimator=None)
    mapie_estimator.fit(X_toy, y_toy)
    if isinstance(mapie_estimator, _MapieClassifier):
        assert isinstance(
            mapie_estimator.estimator_.single_estimator_, DefaultEstimator
        )
    if isinstance(mapie_estimator, _MapieRegressor):
        assert isinstance(
            mapie_estimator.estimator_.single_estimator_, DefaultEstimator
        )


@pytest.mark.parametrize("estimator", [0, "a", KFold(), ["a", "b"]])
@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
def test_invalid_estimator(MapieEstimator: BaseEstimator, estimator: Any) -> None:
    """Test that invalid estimators raise errors."""
    mapie_estimator = MapieEstimator(estimator=estimator)
    with pytest.raises(ValueError, match=r".*Invalid estimator.*"):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.filterwarnings("ignore:Estimator does not appear fitted.*:UserWarning")
@pytest.mark.parametrize("pack", MapieTestEstimators())
def test_invalid_prefit_estimator(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """Test that non-fitted estimator with prefit cv raise errors."""
    MapieEstimator, estimator = pack
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    with pytest.raises(
        (AttributeError, ValueError),
        match=r".*(does not contain 'classes_'|is not fitted).*",
    ):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieTestEstimators())
def test_valid_prefit_estimator(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """Test that fitted estimators with prefit cv raise no errors."""
    MapieEstimator, estimator = pack
    estimator.fit(X_toy, y_toy)
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    mapie_estimator.fit(X_toy, y_toy)
    check_sklearn_user_model_is_fitted(mapie_estimator)
    assert mapie_estimator.n_features_in_ == 1


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
@pytest.mark.parametrize("cv", [-3.14, -2, 0, 1, "cv", LinearRegression(), [1, 2]])
def test_invalid_cv(MapieEstimator: BaseEstimator, cv: Any) -> None:
    """Test that invalid cv raise errors."""
    if MapieEstimator is _MapieClassifier and isinstance(cv, str):
        with pytest.raises(ValueError, match=r'.*it must be equal to "prefit".*'):
            MapieEstimator(cv=cv)
    else:
        mapie_estimator = MapieEstimator(cv=cv)
        with pytest.raises(ValueError, match=r".*Invalid cv.*"):
            mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_alpha_results(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """
    Test that alpha set to `None` in MapieEstimator gives same predictions
    as base estimator.
    """
    MapieEstimator, DefaultEstimator = pack
    estimator = DefaultEstimator()
    estimator.fit(X_toy, y_toy)
    y_pred_expected = estimator.predict(X_toy)
    mapie_estimator = MapieEstimator(estimator=estimator, cv="prefit")
    mapie_estimator.fit(X_toy, y_toy)
    y_pred = mapie_estimator.predict(X_toy)
    np.testing.assert_allclose(y_pred_expected, y_pred)


class TestCrossConformalRegressorReset:
    def test_reset_clears_state(self, dataset_regression) -> None:
        _, X_conformalize, _, _, y_conformalize, _ = dataset_regression
        technique = CrossConformalRegressor(estimator=DummyRegressor())
        technique.fit_conformalize(X_conformalize, y_conformalize)
        assert technique.is_fitted_and_conformalized

        returned = technique.reset()
        assert returned is technique
        assert not technique.is_fitted_and_conformalized
        assert technique._predict_params == {}

    def test_explicit_reset_then_refit_does_not_warn(self, dataset_regression) -> None:
        _, X_conformalize, _, _, y_conformalize, _ = dataset_regression
        technique = CrossConformalRegressor(estimator=DummyRegressor())
        technique.fit_conformalize(X_conformalize, y_conformalize)
        technique.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            technique.fit_conformalize(X_conformalize, y_conformalize)
        assert technique.is_fitted_and_conformalized

    def test_refit_produces_calibration_from_second_dataset(
        self, dataset_regression
    ) -> None:
        _, X_conformalize, X_test, _, y_conformalize, _ = dataset_regression
        scale_a, scale_b = 0.5, 50.0

        refit_technique = CrossConformalRegressor(
            estimator=DummyRegressor(), random_state=RANDOM_STATE
        )
        refit_technique.fit_conformalize(X_conformalize, y_conformalize * scale_a)
        with pytest.warns(UserWarning):
            refit_technique.fit_conformalize(X_conformalize, y_conformalize * scale_b)
        _, intervals_refit = refit_technique.predict_interval(X_test)

        reference_technique = CrossConformalRegressor(
            estimator=DummyRegressor(), random_state=RANDOM_STATE
        )
        reference_technique.fit_conformalize(X_conformalize, y_conformalize * scale_b)
        _, intervals_reference = reference_technique.predict_interval(X_test)

        np.testing.assert_allclose(intervals_refit, intervals_reference)


class TestJackknifeAfterBootstrapRegressorReset:
    def test_reset_clears_state(self, dataset_regression) -> None:
        _, X_conformalize, _, _, y_conformalize, _ = dataset_regression
        technique = JackknifeAfterBootstrapRegressor(estimator=DummyRegressor())
        technique.fit_conformalize(X_conformalize, y_conformalize)
        assert technique.is_fitted_and_conformalized

        returned = technique.reset()
        assert returned is technique
        assert not technique.is_fitted_and_conformalized
        assert technique._predict_params == {}

    def test_explicit_reset_then_refit_does_not_warn(self, dataset_regression) -> None:
        _, X_conformalize, _, _, y_conformalize, _ = dataset_regression
        technique = JackknifeAfterBootstrapRegressor(estimator=DummyRegressor())
        technique.fit_conformalize(X_conformalize, y_conformalize)
        technique.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            technique.fit_conformalize(X_conformalize, y_conformalize)
        assert technique.is_fitted_and_conformalized

    def test_refit_produces_calibration_from_second_dataset(
        self, dataset_regression
    ) -> None:
        _, X_conformalize, X_test, _, y_conformalize, _ = dataset_regression
        scale_a, scale_b = 0.5, 50.0

        refit_technique = JackknifeAfterBootstrapRegressor(
            estimator=DummyRegressor(), random_state=RANDOM_STATE
        )
        refit_technique.fit_conformalize(X_conformalize, y_conformalize * scale_a)
        with pytest.warns(UserWarning):
            refit_technique.fit_conformalize(X_conformalize, y_conformalize * scale_b)
        _, intervals_refit = refit_technique.predict_interval(X_test)

        reference_technique = JackknifeAfterBootstrapRegressor(
            estimator=DummyRegressor(), random_state=RANDOM_STATE
        )
        reference_technique.fit_conformalize(X_conformalize, y_conformalize * scale_b)
        _, intervals_reference = reference_technique.predict_interval(X_test)

        np.testing.assert_allclose(intervals_refit, intervals_reference, rtol=0.05)


class TestDeprecatedAggregatePointPredictionsRenaming:
    """The `aggregate_predictions` (CrossConformalRegressor) and `ensemble`
    (JackknifeAfterBootstrapRegressor) arguments were renamed to
    `aggregate_point_predictions`. The old names still work but emit a
    `FutureWarning`."""

    @staticmethod
    def _assert_same_output(deprecated, new) -> None:
        # `predict_interval` returns a (points, intervals) tuple, `predict` an array.
        if isinstance(new, tuple):
            for deprecated_array, new_array in zip(deprecated, new):
                np.testing.assert_array_equal(deprecated_array, new_array)
        else:
            np.testing.assert_array_equal(deprecated, new)

    @pytest.mark.parametrize("predict_method", ["predict", "predict_interval"])
    def test_cross_conformal_aggregate_predictions_deprecated(
        self, dataset_regression, predict_method
    ) -> None:
        _, X_conformalize, X_test, _, y_conformalize, _ = dataset_regression
        technique = CrossConformalRegressor(estimator=DummyRegressor())
        technique.fit_conformalize(X_conformalize, y_conformalize)

        with pytest.warns(FutureWarning, match=r"aggregate_predictions.*deprecated"):
            deprecated = getattr(technique, predict_method)(
                X_test, aggregate_predictions="median"
            )
        new = getattr(technique, predict_method)(
            X_test, aggregate_point_predictions="median"
        )
        self._assert_same_output(deprecated, new)

    @pytest.mark.parametrize("predict_method", ["predict", "predict_interval"])
    def test_jackknife_ensemble_deprecated(
        self, dataset_regression, predict_method
    ) -> None:
        _, X_conformalize, X_test, _, y_conformalize, _ = dataset_regression
        technique = JackknifeAfterBootstrapRegressor(estimator=DummyRegressor())
        technique.fit_conformalize(X_conformalize, y_conformalize)

        with pytest.warns(FutureWarning, match=r"ensemble.*deprecated"):
            deprecated = getattr(technique, predict_method)(X_test, ensemble=False)
        new = getattr(technique, predict_method)(
            X_test, aggregate_point_predictions=False
        )
        self._assert_same_output(deprecated, new)

    @pytest.mark.parametrize(
        "technique_class, predict_method",
        [
            (CrossConformalRegressor, "predict"),
            (CrossConformalRegressor, "predict_interval"),
            (JackknifeAfterBootstrapRegressor, "predict"),
            (JackknifeAfterBootstrapRegressor, "predict_interval"),
        ],
    )
    def test_new_name_does_not_warn(
        self, dataset_regression, technique_class, predict_method
    ) -> None:
        _, X_conformalize, X_test, _, y_conformalize, _ = dataset_regression
        technique = technique_class(estimator=DummyRegressor())
        technique.fit_conformalize(X_conformalize, y_conformalize)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            getattr(technique, predict_method)(X_test)


class TestCrossConformalClassifierReset:
    def test_reset_clears_state(self, dataset_classification) -> None:
        _, X_conformalize, _, _, y_conformalize, _ = dataset_classification
        technique = CrossConformalClassifier(estimator=DummyClassifier())
        technique.fit_conformalize(X_conformalize, y_conformalize)
        assert technique.is_fitted_and_conformalized

        returned = technique.reset()
        assert returned is technique
        assert not technique.is_fitted_and_conformalized
        assert technique._predict_params == {}

    def test_explicit_reset_then_refit_does_not_warn(
        self, dataset_classification
    ) -> None:
        _, X_conformalize, _, _, y_conformalize, _ = dataset_classification
        technique = CrossConformalClassifier(estimator=DummyClassifier())
        technique.fit_conformalize(X_conformalize, y_conformalize)
        technique.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            technique.fit_conformalize(X_conformalize, y_conformalize)
        assert technique.is_fitted_and_conformalized

    def test_refit_produces_calibration_from_second_dataset(
        self, dataset_classification
    ) -> None:
        _, X_conformalize, X_test, _, y_conformalize, _ = dataset_classification
        rng = np.random.RandomState(RANDOM_STATE)
        y_perm = rng.permutation(y_conformalize)

        refit_technique = CrossConformalClassifier(
            estimator=DummyClassifier(), random_state=RANDOM_STATE
        )
        refit_technique.fit_conformalize(X_conformalize, y_conformalize)
        with pytest.warns(UserWarning):
            refit_technique.fit_conformalize(X_conformalize, y_perm)
        _, sets_refit = refit_technique.predict_set(X_test)

        reference_technique = CrossConformalClassifier(
            estimator=DummyClassifier(), random_state=RANDOM_STATE
        )
        reference_technique.fit_conformalize(X_conformalize, y_perm)
        _, sets_reference = reference_technique.predict_set(X_test)

        np.testing.assert_array_equal(sets_refit, sets_reference)

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
from mapie.utils import NotFittedError, check_user_model_is_fitted

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

        with pytest.raises(ValueError, match=r"fit_conformalize method already called"):
            technique.fit_conformalize(X_conformalize, y_conformalize)


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
def test_default_sample_weight(MapieEstimator: BaseEstimator) -> None:
    """Test default sample weights."""
    mapie_estimator = MapieEstimator()
    assert signature(mapie_estimator.fit).parameters["sample_weight"].default is None


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
    check_user_model_is_fitted(mapie_estimator)
    assert mapie_estimator.n_features_in_ == 1


@pytest.mark.parametrize("MapieEstimator", MapieSimpleEstimators())
@pytest.mark.parametrize("cv", [-3.14, -2, 0, 1, "cv", LinearRegression(), [1, 2]])
def test_invalid_cv(MapieEstimator: BaseEstimator, cv: Any) -> None:
    """Test that invalid cv raise errors."""
    mapie_estimator = MapieEstimator(cv=cv)
    with pytest.raises(ValueError, match=r".*Invalid cv.*"):
        mapie_estimator.fit(X_toy, y_toy)


@pytest.mark.parametrize("pack", MapieDefaultEstimators())
def test_none_alpha_results(pack: Tuple[BaseEstimator, BaseEstimator]) -> None:
    """
    Test that alpha set to ``None`` in MapieEstimator gives same predictions
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

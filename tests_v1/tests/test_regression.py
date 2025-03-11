import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from mapie_v1.regression import (
    SplitConformalRegressor,
    CrossConformalRegressor,
    ConformalizedQuantileRegressor, JackknifeAfterBootstrapRegressor,
)

RANDOM_STATE = 1


@pytest.fixture(scope="module")
def dataset():
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


def test_scr_same_predictions_prefit_not_prefit(dataset) -> None:
    X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    scr_prefit = SplitConformalRegressor(estimator=regressor, prefit=True)
    scr_prefit.conformalize(X_conformalize, y_conformalize)
    predictions_scr_prefit = scr_prefit.predict_interval(X_test)

    scr_not_prefit = SplitConformalRegressor(estimator=LinearRegression(), prefit=False)
    scr_not_prefit.fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)
    predictions_scr_not_prefit = scr_not_prefit.predict_interval(X_test)
    np.testing.assert_equal(predictions_scr_prefit, predictions_scr_not_prefit)


class TestWrongMethodsOrderRaisesError:
    @pytest.mark.parametrize(
        "split_technique",
        [SplitConformalRegressor, ConformalizedQuantileRegressor]
    )
    def test_split_technique_with_prefit_false(self, split_technique, dataset):
        X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
        if split_technique == ConformalizedQuantileRegressor:
            estimator = QuantileRegressor()
        else:
            estimator = DummyRegressor()
        technique = split_technique(estimator=estimator, prefit=False)

        with pytest.raises(ValueError, match=r"call fit before calling conformalize"):
            technique.conformalize(
                X_conformalize,
                y_conformalize
            )

        technique.fit(X_train, y_train)

        with pytest.raises(ValueError, match=r"fit method already called"):
            technique.fit(X_train, y_train)
        with pytest.raises(
            ValueError,
            match=r"call conformalize before calling predict"
        ):
            technique.predict(X_test)
        with pytest.raises(
            ValueError,
            match=r"call conformalize before calling predict_interval"
        ):
            technique.predict_interval(X_test)

        technique.conformalize(X_conformalize, y_conformalize)

        with pytest.raises(ValueError, match=r"conformalize method already called"):
            technique.conformalize(X_conformalize, y_conformalize)

    @pytest.mark.parametrize(
        "split_technique",
        [SplitConformalRegressor, ConformalizedQuantileRegressor]
    )
    def test_split_technique_with_prefit_true(self, split_technique, dataset):
        X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
        estimator = DummyRegressor()
        estimator.fit(X_train, y_train)
        if split_technique == ConformalizedQuantileRegressor:
            technique = split_technique(estimator=[estimator] * 3, prefit=True)
        else:
            technique = split_technique(estimator=estimator, prefit=True)

        with pytest.raises(ValueError, match=r"The fit method must be skipped"):
            technique.fit(X_train, y_train)
        with pytest.raises(
            ValueError,
            match=r"call conformalize before calling predict"
        ):
            technique.predict(X_test)
        with pytest.raises(
            ValueError,
            match=r"call conformalize before calling predict_interval"
        ):
            technique.predict_interval(X_test)

        technique.conformalize(X_conformalize, y_conformalize)

        with pytest.raises(ValueError, match=r"conformalize method already called"):
            technique.conformalize(X_conformalize, y_conformalize)

    @pytest.mark.parametrize(
        "cross_technique",
        [CrossConformalRegressor, JackknifeAfterBootstrapRegressor]
    )
    def test_cross_technique_with_prefit_false(self, cross_technique, dataset):
        X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
        technique = cross_technique(estimator=DummyRegressor())

        with pytest.raises(
            ValueError,
            match=r"call fit_conformalize before calling predict"
        ):
            technique.predict(X_test)
        with pytest.raises(
            ValueError,
            match=r"call fit_conformalize before calling predict_interval"
        ):
            technique.predict_interval(X_test)

        technique.fit_conformalize(X_conformalize, y_conformalize)

        with pytest.raises(ValueError, match=r"fit_conformalize method already called"):
            technique.fit_conformalize(X_conformalize, y_conformalize)

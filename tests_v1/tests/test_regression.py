import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mapie_v1.regression import SplitConformalRegressor

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


@pytest.fixture
def predictions_scr_prefit(dataset):
    X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    scr_prefit = SplitConformalRegressor(estimator=regressor, prefit=True)
    scr_prefit.conformalize(X_conformalize, y_conformalize)
    return scr_prefit.predict_interval(X_test)


@pytest.fixture
def predictions_scr_not_prefit(dataset):
    X_train, X_conformalize, X_test, y_train, y_conformalize, y_test = dataset
    scr_not_prefit = SplitConformalRegressor(estimator=LinearRegression(), prefit=False)
    scr_not_prefit.fit(X_train, y_train).conformalize(X_conformalize, y_conformalize)
    return scr_not_prefit.predict_interval(X_test)


def test_scr_same_intervals_prefit_not_prefit(
    predictions_scr_prefit, predictions_scr_not_prefit
) -> None:
    intervals_scr_prefit = predictions_scr_prefit[1]
    intervals_scr_not_prefit = predictions_scr_not_prefit[1]
    np.testing.assert_equal(intervals_scr_prefit, intervals_scr_not_prefit)


def test_scr_same_predictions_prefit_not_prefit(
    predictions_scr_prefit, predictions_scr_not_prefit
) -> None:
    predictions_scr_prefit = predictions_scr_prefit[0]
    predictions_scr_not_prefit = predictions_scr_not_prefit[0]
    np.testing.assert_equal(predictions_scr_prefit, predictions_scr_not_prefit)

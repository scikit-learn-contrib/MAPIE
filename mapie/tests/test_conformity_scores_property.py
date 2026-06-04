import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
)
from sklearn.model_selection import train_test_split

from mapie.classification import (
    CrossConformalClassifier,
    SplitConformalClassifier,
)
from mapie.regression import (
    ConformalizedQuantileRegressor,
    CrossConformalRegressor,
    JackknifeAfterBootstrapRegressor,
    SplitConformalRegressor,
)
from mapie.regression.time_series_regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

RANDOM_STATE = 42


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200, n_features=4, n_classes=2, random_state=RANDOM_STATE
    )
    return train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE)


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=200, n_features=3, noise=10.0, random_state=RANDOM_STATE
    )
    return train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE)


class TestConformityScoresProperty:
    def test_split_conformal_classifier(self, classification_data):
        X_train, X_calib, y_train, y_calib = classification_data
        estimator = SplitConformalClassifier(
            estimator=LogisticRegression(), prefit=False
        )
        estimator.fit(X_train, y_train).conformalize(X_calib, y_calib)

        assert isinstance(estimator.conformity_scores, np.ndarray)
        assert (
            estimator.conformity_scores
            is estimator._mapie_classifier.conformity_scores_
        )

    def test_cross_conformal_classifier(self, classification_data):
        X_train, X_calib, y_train, y_calib = classification_data
        X, y = np.concatenate([X_train, X_calib]), np.concatenate([y_train, y_calib])
        estimator = CrossConformalClassifier(estimator=LogisticRegression(), cv=5)
        estimator.fit_conformalize(X, y)

        assert isinstance(estimator.conformity_scores, np.ndarray)
        assert (
            estimator.conformity_scores
            is estimator._mapie_classifier.conformity_scores_
        )

    def test_split_conformal_regressor(self, regression_data):
        X_train, X_calib, y_train, y_calib = regression_data
        estimator = SplitConformalRegressor(estimator=LinearRegression(), prefit=False)
        estimator.fit(X_train, y_train).conformalize(X_calib, y_calib)

        assert isinstance(estimator.conformity_scores, np.ndarray)
        assert (
            estimator.conformity_scores is estimator._mapie_regressor.conformity_scores_
        )

    def test_cross_conformal_regressor(self, regression_data):
        X_train, X_calib, y_train, y_calib = regression_data
        X, y = np.concatenate([X_train, X_calib]), np.concatenate([y_train, y_calib])
        estimator = CrossConformalRegressor(estimator=LinearRegression(), cv=5)
        estimator.fit_conformalize(X, y)

        assert isinstance(estimator.conformity_scores, np.ndarray)
        assert (
            estimator.conformity_scores is estimator._mapie_regressor.conformity_scores_
        )

    def test_jackknife_after_bootstrap_regressor(self, regression_data):
        X_train, X_calib, y_train, y_calib = regression_data
        X, y = np.concatenate([X_train, X_calib]), np.concatenate([y_train, y_calib])
        estimator = JackknifeAfterBootstrapRegressor(
            estimator=LinearRegression(), resampling=10
        )
        estimator.fit_conformalize(X, y)

        assert isinstance(estimator.conformity_scores, np.ndarray)
        assert (
            estimator.conformity_scores is estimator._mapie_regressor.conformity_scores_
        )

    def test_conformalized_quantile_regressor(self, regression_data):
        X_train, X_calib, y_train, y_calib = regression_data
        estimator = ConformalizedQuantileRegressor(
            estimator=QuantileRegressor(solver="highs"),
            confidence_level=0.9,
            prefit=False,
        )
        estimator.fit(X_train, y_train).conformalize(X_calib, y_calib)

        assert isinstance(estimator.conformity_scores, np.ndarray)
        assert estimator.conformity_scores.shape[0] == 3
        assert (
            estimator.conformity_scores
            is estimator._mapie_quantile_regressor.conformity_scores_
        )

    def test_time_series_regressor(self, regression_data):
        X_train, X_calib, y_train, y_calib = regression_data
        X, y = np.concatenate([X_train, X_calib]), np.concatenate([y_train, y_calib])
        estimator = TimeSeriesRegressor(
            estimator=LinearRegression(),
            method="enbpi",
            cv=BlockBootstrap(n_resamplings=10, length=20, random_state=RANDOM_STATE),
        )
        estimator.fit(X, y)

        assert isinstance(estimator.conformity_scores, np.ndarray)
        assert estimator.conformity_scores is estimator.conformity_scores_


class TestConformityScoresPropertyRaisesBeforeComputation:
    def test_split_conformal_classifier(self):
        estimator = SplitConformalClassifier(
            estimator=LogisticRegression(), prefit=False
        )
        with pytest.raises(ValueError):
            estimator.conformity_scores

    def test_cross_conformal_classifier(self):
        estimator = CrossConformalClassifier(estimator=LogisticRegression(), cv=5)
        with pytest.raises(ValueError):
            estimator.conformity_scores

    def test_split_conformal_regressor(self):
        estimator = SplitConformalRegressor(estimator=LinearRegression(), prefit=False)
        with pytest.raises(ValueError):
            estimator.conformity_scores

    def test_cross_conformal_regressor(self):
        estimator = CrossConformalRegressor(estimator=LinearRegression(), cv=5)
        with pytest.raises(ValueError):
            estimator.conformity_scores

    def test_jackknife_after_bootstrap_regressor(self):
        estimator = JackknifeAfterBootstrapRegressor(
            estimator=LinearRegression(), resampling=10
        )
        with pytest.raises(ValueError):
            estimator.conformity_scores

    def test_conformalized_quantile_regressor(self):
        estimator = ConformalizedQuantileRegressor(
            estimator=QuantileRegressor(solver="highs"),
            confidence_level=0.9,
            prefit=False,
        )
        with pytest.raises(ValueError):
            estimator.conformity_scores

    def test_time_series_regressor(self):
        estimator = TimeSeriesRegressor(
            estimator=LinearRegression(),
            method="enbpi",
            cv=BlockBootstrap(n_resamplings=10, length=20, random_state=RANDOM_STATE),
        )
        with pytest.raises(ValueError):
            estimator.conformity_scores

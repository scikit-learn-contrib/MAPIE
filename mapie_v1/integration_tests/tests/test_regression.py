import numpy as np
from sklearn.model_selection import train_test_split

from mapie_v1.regression import SplitConformalRegressor
from mapie.tests.test_regression import X_toy, y_toy
from mapiev0.regression import MapieRegressor as MapieRegressorV0  # noqa


def test_dummy():
    test_size = 0.5
    alpha = 0.5
    confidence_level = 1 - alpha
    random_state = 42

    v0 = MapieRegressorV0(cv="split", test_size=test_size, random_state=random_state)
    v0.fit(X_toy, y_toy)
    v0_preds = v0.predict(X_toy)
    v0_pred_intervals = v0.predict(X_toy, alpha=alpha)

    X_train, y_train, X_conf, y_conf = train_test_split(
        X_toy, y_toy, test_size=test_size, random_state=random_state
    )
    v1 = SplitConformalRegressor(confidence_level=confidence_level, random_state=random_state)
    v1.fit_conformalize(X_train, y_train, X_conf, y_conf)
    v1_preds = v1.predict(X_toy)
    v1_pred_intervals = v1.predict_set(X_toy)
    np.testing.assert_array_equal(v1_preds, v0_preds)
    np.testing.assert_array_equal(v1_pred_intervals, v0_pred_intervals)

import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from sklpredinterv import PredictionInterval


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.fixture
def toy_data():
    X = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 2, 3, 4, 5]
    return X, y


def test_predinterv_input(data):
    """Test default values of input parameters."""
    pireg = PredictionInterval(DummyRegressor())
    assert pireg.method == "jackknife_plus"
    assert pireg.alpha == 0.1
    assert pireg.n_splits == 10
    assert pireg.shuffle
    assert pireg.return_pred == "single"
    assert pireg.random_state is None


class TestAttributes:
    @pytest.mark.parametrize(
        "method",
        ["naive", "jackknife", "jackknife_plus", "cv", "cv_plus", "cv_minmax"]
    )
    def test_shared_attributes(self, data, method):
        """Test class attributes shared by all PI methods."""
        pireg = PredictionInterval(DummyRegressor(), method=method)
        pireg.fit(*data)
        assert hasattr(pireg, 'single_estimator_')
        assert hasattr(pireg, 'n_')

    @pytest.mark.parametrize("method", ["naive", "jackknife", "cv"])
    def test_quantile_attribute(self, data, method):
        """Test quantile attribute."""
        pireg = PredictionInterval(DummyRegressor(), method=method)
        pireg.fit(*data)
        assert hasattr(pireg, 'quantile_')
        assert (pireg.quantile_ >= 0) & (pireg.quantile_ < data[0].shape[0])

    @pytest.mark.parametrize(
        "method",
        ["jackknife", "jackknife_plus", "cv", "cv_plus", "cv_minmax"]
    )
    def test_estimators_attribute(self, data, method):
        """Test class attributes shared by jackknife and CV methods."""
        pireg = PredictionInterval(DummyRegressor(), method=method)
        pireg.fit(*data)
        assert hasattr(pireg, 'estimators_')
        assert hasattr(pireg, 'residuals_split_')
        assert hasattr(pireg, 'y_train_pred_split_')

    @pytest.mark.parametrize(
        "method",
        ["cv", "cv_plus", "cv_minmax"]
    )
    def test_cv_attributes(self, data, method):
        """Test class attributes shared by CV methods."""
        pireg = PredictionInterval(DummyRegressor(), method=method)
        pireg.fit(*data)
        assert hasattr(pireg, 'val_fold_ids_')


def test_predinterv_notfitted(data):
    X, y = data
    pireg = PredictionInterval(DummyRegressor())
    msg = ("This PredictionInterval instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this estimator.")
    with pytest.raises(NotFittedError, match=msg):
        pireg.predict(X)


def test_predinterv_invalidmethod(data):
    pireg = PredictionInterval(DummyRegressor(), method="dummy")
    msg = ("Invalid method.")
    with pytest.raises(ValueError, match=msg):
        pireg.fit(*data)


def test_predinterv_outputshape(data):
    X, y = data
    pireg = PredictionInterval(DummyRegressor())
    pireg.fit(X, y)
    assert pireg.predict(X).shape[0] == X.shape[0]


def test_results(toy_data):
    X, y = data



# def test_predictioninterval_est(data, baseest)
#     est.fit(*data)
#     assert hasattr(est, "single_estimator_")



# def test_template_estimator(data):
#     est = TemplateEstimator()
#     assert est.demo_param == 'demo_param'

#     est.fit(*data)
#     assert hasattr(est, 'is_fitted_')

#     X = data[0]
#     y_pred = est.predict(X)
#     assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


# def test_template_transformer_error(data):
#     X, y = data
#     trans = TemplateTransformer()
#     trans.fit(X)
#     with pytest.raises(ValueError, match="Shape of input is different"):
#         X_diff_size = np.ones((10, X.shape[1] + 1))
#         trans.transform(X_diff_size)


# def test_template_transformer(data):
#     X, y = data
#     trans = TemplateTransformer()
#     assert trans.demo_param == 'demo'

#     trans.fit(X)
#     assert trans.n_features_ == X.shape[1]

#     X_trans = trans.transform(X)
#     assert_allclose(X_trans, np.sqrt(X))

#     X_trans = trans.fit_transform(X)
#     assert_allclose(X_trans, np.sqrt(X))



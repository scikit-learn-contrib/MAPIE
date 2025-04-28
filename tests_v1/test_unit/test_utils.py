import numpy as np
import pytest
from sklearn.datasets import make_regression

from mapie.utils import (
    _prepare_params,
    _prepare_fit_params_and_sample_weight,
    _transform_confidence_level_to_alpha_list,
    _transform_confidence_level_to_alpha,
    _check_if_param_in_allowed_values,
    _check_cv_not_string,
    _cast_point_predictions_to_ndarray,
    _cast_predictions_to_ndarray_tuple,
    _raise_error_if_previous_method_not_called,
    _raise_error_if_method_already_called,
    _raise_error_if_fit_called_in_prefit_mode,
    train_conformalize_test_split
)
from unittest.mock import patch


RANDOM_STATE = 1


@pytest.fixture(scope="module")
def dataset():
    X, y = make_regression(
        n_samples=100, n_features=2, noise=1.0, random_state=RANDOM_STATE
    )
    return X, y


class TestTrainConformalizeTestSplit:

    def test_error_sum_int_is_not_dataset_size(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError):
            train_conformalize_test_split(
                X, y, train_size=1, conformalize_size=1,
                test_size=1, random_state=RANDOM_STATE
            )

    def test_error_sum_float_is_not_1(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError):
            train_conformalize_test_split(
                X, y, train_size=0.5, conformalize_size=0.5,
                test_size=0.5, random_state=RANDOM_STATE
            )

    def test_error_sizes_are_int_and_float(self, dataset):
        X, y = dataset
        with pytest.raises(TypeError):
            train_conformalize_test_split(
                X, y, train_size=5, conformalize_size=0.5,
                test_size=0.5, random_state=RANDOM_STATE
            )

    def test_3_floats(self, dataset):
        X, y = dataset
        (
            X_train, X_conformalize, X_test, y_train, y_conformalize, y_test
        ) = train_conformalize_test_split(
            X, y, train_size=0.6, conformalize_size=0.2,
            test_size=0.2, random_state=RANDOM_STATE
        )
        assert len(X_train) == 60
        assert len(X_conformalize) == 20
        assert len(X_test) == 20

    def test_3_ints(self, dataset):
        X, y = dataset
        (
            X_train, X_conformalize, X_test, y_train, y_conformalize, y_test
        ) = train_conformalize_test_split(
            X, y, train_size=60, conformalize_size=20,
            test_size=20, random_state=RANDOM_STATE
        )
        assert len(X_train) == 60
        assert len(X_conformalize) == 20
        assert len(X_test) == 20

    def test_random_state(self, dataset):
        X, y = dataset
        (
            X_train_1, X_conformalize_1, X_test_1, y_train_1, y_conformalize_1, y_test_1
        ) = train_conformalize_test_split(
            X, y, train_size=60, conformalize_size=20,
            test_size=20, random_state=RANDOM_STATE
        )
        (
            X_train_2, X_conformalize_2, X_test_2, y_train_2, y_conformalize_2, y_test_2
        ) = train_conformalize_test_split(
            X, y, train_size=60, conformalize_size=20,
            test_size=20, random_state=RANDOM_STATE
        )
        assert np.array_equal(X_train_1, X_train_2)
        assert np.array_equal(X_conformalize_1, X_conformalize_2)
        assert np.array_equal(X_test_1, X_test_2)
        assert np.array_equal(y_train_1, y_train_2)
        assert np.array_equal(y_conformalize_1, y_conformalize_2)
        assert np.array_equal(y_test_1, y_test_2)

    def test_different_random_state(self, dataset):
        X, y = dataset
        (
            X_train_1, X_conformalize_1, X_test_1, y_train_1, y_conformalize_1, y_test_1
        ) = train_conformalize_test_split(
            X, y, train_size=60, conformalize_size=20,
            test_size=20, random_state=RANDOM_STATE
        )
        (
            X_train_2, X_conformalize_2, X_test_2, y_train_2, y_conformalize_2, y_test_2
        ) = train_conformalize_test_split(
            X, y, train_size=60, conformalize_size=20,
            test_size=20, random_state=RANDOM_STATE + 1
        )
        assert not np.array_equal(X_train_1, X_train_2)
        assert not np.array_equal(X_conformalize_1, X_conformalize_2)
        assert not np.array_equal(X_test_1, X_test_2)
        assert not np.array_equal(y_train_1, y_train_2)
        assert not np.array_equal(y_conformalize_1, y_conformalize_2)
        assert not np.array_equal(y_test_1, y_test_2)

    def test_shuffle_false(self, dataset):
        X, y = dataset
        (
            X_train, X_conformalize, X_test, y_train, y_conformalize, y_test
        ) = train_conformalize_test_split(
            X, y, train_size=60, conformalize_size=20,
            test_size=20, random_state=RANDOM_STATE, shuffle=False
        )
        assert np.array_equal(np.concatenate((y_train, y_conformalize, y_test)), y)


@pytest.fixture
def point_predictions():
    return np.array([1, 2, 3])


@pytest.fixture
def point_and_interval_predictions():
    return np.array([1, 2]), np.array([3, 4])


@pytest.mark.parametrize(
    "confidence_level, expected",
    [
        (0.9, 0.1),
        (0.7, 0.3),
        (0.999, 0.001),
    ]
)
def test_transform_confidence_level_to_alpha(confidence_level, expected):
    result = _transform_confidence_level_to_alpha(confidence_level)
    assert result == expected
    assert str(result) == str(expected)  # Ensure clean representation


class TestTransformConfidenceLevelToAlphaList:
    def test_non_list_iterable(self):
        confidence_level = (0.8, 0.7)  # Testing a non-list iterable
        assert _transform_confidence_level_to_alpha_list(confidence_level) == [0.2, 0.3]

    def test_transform_confidence_level_to_alpha_is_called(self):
        with patch(
            'mapie.utils._transform_confidence_level_to_alpha'
        ) as mock_transform_confidence_level_to_alpha:
            _transform_confidence_level_to_alpha_list([0.2, 0.3])
            mock_transform_confidence_level_to_alpha.assert_called()


class TestCheckIfParamInAllowedValues:
    def test_error(self):
        with pytest.raises(ValueError):
            _check_if_param_in_allowed_values("invalid_option", "", ["valid_option"])

    def test_ok(self):
        assert _check_if_param_in_allowed_values("valid", "", ["valid"]) is None


def test_check_cv_not_string():
    with pytest.raises(ValueError):
        _check_cv_not_string("string")


class TestCastPointPredictionsToNdarray:
    def test_error(self, point_and_interval_predictions):
        with pytest.raises(TypeError):
            _cast_point_predictions_to_ndarray(point_and_interval_predictions)

    def test_valid_ndarray(self, point_predictions):
        point_predictions = np.array([1, 2, 3])
        result = _cast_point_predictions_to_ndarray(point_predictions)
        assert result is point_predictions
        assert isinstance(result, np.ndarray)


class TestCastPredictionsToNdarrayTuple:
    def test_error(self, point_predictions):
        with pytest.raises(TypeError):
            _cast_predictions_to_ndarray_tuple(point_predictions)

    def test_valid_ndarray(self, point_and_interval_predictions):
        result = _cast_predictions_to_ndarray_tuple(point_and_interval_predictions)
        assert result is point_and_interval_predictions
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)


@pytest.mark.parametrize(
    "params, expected", [(None, {}), ({"a": 1, "b": 2}, {"a": 1, "b": 2})]
)
def test_prepare_params(params, expected):
    assert _prepare_params(params) == expected
    assert _prepare_params(params) is not params


class TestPrepareFitParamsAndSampleWeight:
    def test_uses_prepare_params(self):
        with patch('mapie.utils._prepare_params') as mock_prepare_params:
            _prepare_fit_params_and_sample_weight({"param1": 1})
            mock_prepare_params.assert_called()

    def test_with_sample_weight(self):
        fit_params = {"sample_weight": [0.1, 0.2, 0.3]}
        assert _prepare_fit_params_and_sample_weight(fit_params) == (
            {},
            [0.1, 0.2, 0.3]
        )

    def test_without_sample_weight(self):
        params = {"param1": 1}
        assert _prepare_fit_params_and_sample_weight(params) == (params, None)


class TestRaiseErrorIfPreviousMethodNotCalled:
    def test_raises_error_when_previous_method_not_called(self):
        with pytest.raises(ValueError):
            _raise_error_if_previous_method_not_called(
                "current_method", "previous_method", False
            )

    def test_does_nothing_when_previous_method_called(self):
        assert _raise_error_if_previous_method_not_called(
            "current_method", "previous_method", True
        ) is None


class TestRaiseErrorIfMethodAlreadyCalled:
    def test_raises_error_when_method_already_called(self):
        with pytest.raises(ValueError):
            _raise_error_if_method_already_called("method", True)

    def test_does_nothing_when_method_not_called(self):
        assert _raise_error_if_method_already_called("method", False) is None


class TestRaiseErrorIfFitCalledInPrefitMode:
    def test_raises_error_in_prefit_mode(self):
        with pytest.raises(ValueError):
            _raise_error_if_fit_called_in_prefit_mode(True)

    def test_does_nothing_when_not_in_prefit_mode(self):
        assert _raise_error_if_fit_called_in_prefit_mode(False) is None

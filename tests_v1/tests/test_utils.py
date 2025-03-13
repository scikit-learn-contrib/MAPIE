"""
Testing small functions.
Group tests in a class if more than one test is needed for a given function.
Focus on the function goal to define the test cases (the function name is a good hint).
Prefer black-box tests (no mocks) if possible, to avoid testing implementation details.
"""

import numpy as np
import pytest

from mapie_v1._utils import (
    prepare_params,
    prepare_fit_params_and_sample_weight,
    transform_confidence_level_to_alpha_list,
    transform_confidence_level_to_alpha,
    check_if_param_in_allowed_values,
    check_cv_not_string,
    cast_point_predictions_to_ndarray,
    cast_predictions_to_ndarray_tuple,
    raise_error_if_previous_method_not_called,
    raise_error_if_method_already_called,
    raise_error_if_fit_called_in_prefit_mode,
)
from unittest.mock import patch


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
    result = transform_confidence_level_to_alpha(confidence_level)
    assert result == expected
    assert str(result) == str(expected)  # Ensure clean representation


class TestTransformConfidenceLevelToAlphaList:
    def test_non_list_iterable(self):
        confidence_level = (0.8, 0.7)  # Testing a non-list iterable
        assert transform_confidence_level_to_alpha_list(confidence_level) == [0.2, 0.3]

    def test_transform_confidence_level_to_alpha_is_called(self):
        with patch(
            'mapie_v1._utils.transform_confidence_level_to_alpha'
        ) as mock_transform_confidence_level_to_alpha:
            transform_confidence_level_to_alpha_list([0.2, 0.3])
            mock_transform_confidence_level_to_alpha.assert_called()


class TestCheckIfParamInAllowedValues:
    def test_error(self):
        with pytest.raises(ValueError):
            check_if_param_in_allowed_values("invalid_option", "", ["valid_option"])

    def test_ok(self):
        assert check_if_param_in_allowed_values("valid", "", ["valid"]) is None


def test_check_cv_not_string():
    with pytest.raises(ValueError):
        check_cv_not_string("string")


class TestCastPointPredictionsToNdarray:
    def test_error(self, point_and_interval_predictions):
        with pytest.raises(TypeError):
            cast_point_predictions_to_ndarray(point_and_interval_predictions)

    def test_valid_ndarray(self, point_predictions):
        point_predictions = np.array([1, 2, 3])
        result = cast_point_predictions_to_ndarray(point_predictions)
        assert result is point_predictions
        assert isinstance(result, np.ndarray)


class TestCastPredictionsToNdarrayTuple:
    def test_error(self, point_predictions):
        with pytest.raises(TypeError):
            cast_predictions_to_ndarray_tuple(point_predictions)

    def test_valid_ndarray(self, point_and_interval_predictions):
        result = cast_predictions_to_ndarray_tuple(point_and_interval_predictions)
        assert result is point_and_interval_predictions
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)


@pytest.mark.parametrize(
    "params, expected", [(None, {}), ({"a": 1, "b": 2}, {"a": 1, "b": 2})]
)
def test_prepare_params(params, expected):
    assert prepare_params(params) == expected
    assert prepare_params(params) is not params


class TestPrepareFitParamsAndSampleWeight:
    def test_uses_prepare_params(self):
        with patch('mapie_v1._utils.prepare_params') as mock_prepare_params:
            prepare_fit_params_and_sample_weight({"param1": 1})
            mock_prepare_params.assert_called()

    def test_with_sample_weight(self):
        fit_params = {"sample_weight": [0.1, 0.2, 0.3]}
        assert prepare_fit_params_and_sample_weight(fit_params) == ({}, [0.1, 0.2, 0.3])

    def test_without_sample_weight(self):
        params = {"param1": 1}
        assert prepare_fit_params_and_sample_weight(params) == (params, None)


class TestRaiseErrorIfPreviousMethodNotCalled:
    def test_raises_error_when_previous_method_not_called(self):
        with pytest.raises(ValueError):
            raise_error_if_previous_method_not_called(
                "current_method", "previous_method", False
            )

    def test_does_nothing_when_previous_method_called(self):
        assert raise_error_if_previous_method_not_called(
            "current_method", "previous_method", True
        ) is None


class TestRaiseErrorIfMethodAlreadyCalled:
    def test_raises_error_when_method_already_called(self):
        with pytest.raises(ValueError):
            raise_error_if_method_already_called("method", True)

    def test_does_nothing_when_method_not_called(self):
        assert raise_error_if_method_already_called("method", False) is None


class TestRaiseErrorIfFitCalledInPrefitMode:
    def test_raises_error_in_prefit_mode(self):
        with pytest.raises(ValueError):
            raise_error_if_fit_called_in_prefit_mode(True)

    def test_does_nothing_when_not_in_prefit_mode(self):
        assert raise_error_if_fit_called_in_prefit_mode(False) is None

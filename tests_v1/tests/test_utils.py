import pytest

from mapie_v1._utils import (
    prepare_params,
    prepare_fit_params_and_sample_weight,
    transform_confidence_level_to_alpha_list,
)
from unittest.mock import patch


@pytest.mark.parametrize(
    "confidence_level, expected", [(0.9, [0.1]), ([0.1, 0.2], [0.9, 0.8])]
)
def test_transform_confidence_level_to_alpha_list(confidence_level, expected):
    assert transform_confidence_level_to_alpha_list(confidence_level) == expected


def test_prepare_params_none():
    assert prepare_params(None) == {}


def test_prepare_params_non_empty_dict():
    params = {"param1": 1, "param2": 2}
    assert prepare_params(params) == {"param1": 1, "param2": 2}


def test_prepare_params_deepcopy():
    params = {"param1": [1, 2, 3]}
    assert prepare_params(params) is not params


def test_prepare_fit_params_and_sample_weight_uses_prepare_params():
    fit_params = {"param1": 1}
    with patch('mapie_v1._utils.prepare_params') as mock_prepare_params:
        prepare_fit_params_and_sample_weight(fit_params)
        mock_prepare_params.assert_called_once_with(fit_params)


def test_prepare_fit_params_and_sample_weight_with_sample_weight():
    fit_params = {"sample_weight": [0.1, 0.2, 0.3]}
    assert prepare_fit_params_and_sample_weight(fit_params) == ({}, [0.1, 0.2, 0.3])


def test_prepare_fit_params_and_sample_weight_without_sample_weight():
    fit_params = {"param1": 1}
    assert prepare_fit_params_and_sample_weight(fit_params) == ({"param1": 1}, None)

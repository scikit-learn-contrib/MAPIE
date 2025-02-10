from mapie_v1._utils import prepare_params, prepare_fit_params_and_sample_weight
from unittest.mock import patch


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

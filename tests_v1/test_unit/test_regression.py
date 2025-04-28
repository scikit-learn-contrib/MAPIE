import pytest
from mapie.subsample import Subsample
from mapie.regression import JackknifeAfterBootstrapRegressor


class TestCheckAndConvertResamplingToCv:
    def test_with_integer(self):
        regressor = JackknifeAfterBootstrapRegressor()
        cv = regressor._check_and_convert_resampling_to_cv(50)

        assert isinstance(cv, Subsample)
        assert cv.n_resamplings == 50

    def test_with_subsample(self):
        custom_subsample = Subsample(n_resamplings=25, random_state=42)
        regressor = JackknifeAfterBootstrapRegressor()
        cv = regressor._check_and_convert_resampling_to_cv(custom_subsample)

        assert cv is custom_subsample

    def test_with_invalid_input(self):
        regressor = JackknifeAfterBootstrapRegressor()

        with pytest.raises(
            ValueError,
            match="resampling must be an integer or a Subsample instance"
        ):
            regressor._check_and_convert_resampling_to_cv("invalid_input")

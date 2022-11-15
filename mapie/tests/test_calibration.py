from mapie.calibration import MapieCalibrator


def test_default_parameters() -> None:
    """Test default values of input parameters."""
    mapie_cal = MapieCalibrator()
    assert mapie_cal.method == "top_label"
    assert mapie_cal.calibration_method is None
    assert mapie_cal.cv is None
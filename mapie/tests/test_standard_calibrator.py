from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from mapie.calibrators import StandardCalibrator
from mapie.regression import SplitCPRegressor
from mapie.conformity_scores import AbsoluteConformityScore

random_state = 1
np.random.seed(random_state)

X, y = make_regression(
    n_samples=500, n_features=10, noise=1.0, random_state=random_state
)
z = X[:, -2:]


@pytest.mark.parametrize("sym", [True, False])
def test_calibrator_fit(sym: bool) -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=StandardCalibrator(), alpha=0.1,
                             conformity_score=AbsoluteConformityScore(sym=sym))
    mapie.fit(X, y, z=z)


@pytest.mark.parametrize("sym", [True, False])
def test_calibrator_fit_predict(sym: bool) -> None:
    """Test that initialization does not crash."""
    mapie = SplitCPRegressor(calibrator=StandardCalibrator(), alpha=0.1,
                             conformity_score=AbsoluteConformityScore(sym=sym))
    mapie.fit(X, y, z=z)
    mapie.predict(X, z=z)

import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklpredinterv import TemplateEstimator
from sklpredinterv import TemplateClassifier
from sklpredinterv import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

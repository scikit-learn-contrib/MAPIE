import pytest

from mapie.conformity_scores.utils import (
    check_and_select_conformity_score,
)
from mapie.conformity_scores.regression import BaseRegressionScore
from mapie.conformity_scores.classification import BaseClassificationScore
from mapie.conformity_scores.bounds import (
    AbsoluteConformityScore,
    GammaConformityScore,
)
from mapie.conformity_scores.sets import (
    LACConformityScore,
    TopKConformityScore,
)


class TestCheckAndSelectConformityScore:

    @pytest.mark.parametrize(
        "score, score_type, expected_class", [
            (AbsoluteConformityScore(), BaseRegressionScore, AbsoluteConformityScore),
            ("gamma", BaseRegressionScore, GammaConformityScore),
            (LACConformityScore(), BaseClassificationScore, LACConformityScore),
            ("top_k", BaseClassificationScore, TopKConformityScore),
        ]
    )
    def test_with_valid_inputs(self, score, score_type, expected_class):
        result = check_and_select_conformity_score(score, score_type)
        assert isinstance(result, expected_class)

    @pytest.mark.parametrize(
        "score_type", [BaseRegressionScore, BaseClassificationScore]
    )
    def test_with_invalid_input(self, score_type):
        with pytest.raises(ValueError):
            check_and_select_conformity_score("I'm not a valid input :(", score_type)

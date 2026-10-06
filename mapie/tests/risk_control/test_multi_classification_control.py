from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie.risk_control import MultiClassificationController
from mapie.risk_control.risks import build_precision_ovr

random_state = 42
dummy_target = 0.75


def dummy_predict(X):
    return np.random.rand(1, 3)  # pragma: no cover


@pytest.fixture
def mcc_dummy():
    return MultiClassificationController(
        predict_function=dummy_predict,
        target_level=dummy_target,
        risks=build_precision_ovr([0, 1, 2]),
        risk_combination_method=1,
    )


X_toy, y_toy = make_classification(
    n_samples=300,
    n_features=5,
    n_informative=5,
    n_redundant=0,
    n_repeated=0,
    n_classes=5,
    n_clusters_per_class=1,
    flip_y=0,
    random_state=random_state,
)
clf = LogisticRegression().fit(X_toy, y_toy)


def test_notebook_workflow_split_fixed_sequence() -> None:
    """
    Reproduce a workflow for MultiClassificationController.
    Build one-vs-rest precision risks, calibrate with fwer_method="split_fixed_sequence"
    (requiring learn_fixed_sequence_order beforehand), and predict.
    """
    mapie_clf = MultiClassificationController(
        predict_function=clf.predict_proba,
        target_level=0.75,
        risks=build_precision_ovr(np.unique(y_toy)),
        confidence_level=0.9,
        list_predict_params=np.linspace(0.05, 0.9, 90),
        fwer_method="split_fixed_sequence",
        risk_combination_method=1,
    )

    X_calibrate = X_toy
    y_calibrate = y_toy

    mapie_clf.learn_fixed_sequence_order(X_calibrate, y_calibrate)
    mapie_clf.calibrate(X_toy, y_toy)

    assert mapie_clf.p_values is not None
    assert mapie_clf.valid_predict_params is not None

    predictions = mapie_clf.predict(X_toy)
    assert predictions.shape == (len(X_toy),)


class TestCheckRiskCombinationMethod:
    @pytest.mark.parametrize(
        "y_calibrate, class_label, expected_index",
        [
            (np.array([0, 1, 2, 1, 0]), 0, 0),
            (np.array([0, 1, 2, 1, 0]), 1, 1),
            (np.array([0, 1, 2, 1, 0]), 2, 2),
            # Labels not sorted/contiguous in the data: dict_class_label is
            # built from np.unique, so it is always sorted ascending.
            (np.array([5, 2, 8]), 8, 2),
            (np.array(["b", "a", "c"]), "b", 1),
        ],
    )
    def test_int_or_str_selects_correct_row(
        self,
        mcc_dummy: MultiClassificationController,
        y_calibrate: Any,
        class_label: Any,
        expected_index: int,
    ) -> None:
        """
        When risk_combination_method is a class label, the returned callable
        must select the row of the input array corresponding to that label's
        position in np.unique(y_calibrate).
        """
        selector = mcc_dummy._check_risk_combination_method(class_label, y_calibrate)

        array = np.array([[10, 20], [30, 40], [50, 60]])
        result = selector(array)

        np.testing.assert_array_equal(result, array[expected_index, :])

    @pytest.mark.parametrize(
        "y_calibrate, class_label",
        [
            (np.array([0, 1, 2]), 3),
            (np.array([0, 1, 2]), -1),
            (np.array(["a", "b", "c"]), "z"),
        ],
    )
    def test_label_not_in_y_calibrate_raises(
        self,
        mcc_dummy: MultiClassificationController,
        y_calibrate: Any,
        class_label: Any,
    ) -> None:
        """An unknown class label must raise an AssertionError."""
        with pytest.raises(AssertionError, match=r".*must be in list_class_label.*"):
            mcc_dummy._check_risk_combination_method(class_label, y_calibrate)

    def test_callable_is_returned_unchanged(
        self, mcc_dummy: MultiClassificationController
    ) -> None:
        """A callable risk_combination_method must be passed through as-is."""

        def custom_combination(array: Any) -> Any:
            return array.max(axis=0)  # pragma: no cover

        result = mcc_dummy._check_risk_combination_method(
            custom_combination, np.array([0, 1, 2])
        )

        assert result is custom_combination

    @pytest.mark.parametrize("invalid_method", [2.5, [1, 2], (1,), None, {1: 2}])
    def test_invalid_type_raises_type_error(
        self, mcc_dummy: MultiClassificationController, invalid_method: Any
    ) -> None:
        """A risk_combination_method that is neither a class label nor callable
        must raise a TypeError."""
        with pytest.raises(TypeError, match=r".*must be a class label or a callable.*"):
            mcc_dummy._check_risk_combination_method(
                invalid_method, np.array([0, 1, 2])
            )


class TestSetBestPredictParamChoice:
    def test_auto_returns_all_risks(
        self, mcc_dummy: MultiClassificationController
    ) -> None:
        """
        "auto" is the only supported value: it must return the full list of
        risks (unlike BinaryClassificationController, which picks a single
        complementary risk).
        """
        result = mcc_dummy._set_best_predict_param_choice("auto")

        assert result is mcc_dummy._risk

    def test_auto_is_set_at_init(self) -> None:
        """By default (best_predict_param_choice="auto"), the controller's
        _best_predict_param_choice attribute must be set to its risks list."""
        risks = build_precision_ovr([0, 1, 2])
        mcc = MultiClassificationController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=risks,
            risk_combination_method=1,
        )

        assert mcc._best_predict_param_choice is mcc._risk
        assert mcc._risk is risks

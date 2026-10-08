from typing import Any, Optional

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie.risk_control import MultiClassificationLTTController
from mapie.risk_control.risks import build_precision_ovr

random_state = 42
dummy_target = 0.75
_Y_INT = np.array([0, 1, 2])
_Y_STR = np.array(["a", "b", "c"])


def dummy_predict(X):
    return np.random.rand(1, 3)  # pragma: no cover


@pytest.fixture
def mcc_dummy():
    return MultiClassificationLTTController(
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


class TestSplitFixedSequence:
    def test_workflow_split_fixed_sequence(self) -> None:
        """
        Reproduce a workflow for MultiClassificationLTTController.
        calibrate() must raise a ValueError if learn_fixed_sequence_order was
        not called beforehand. Then, build one-vs-rest precision risks,
        calibrate with fwer_method="split_fixed_sequence" (requiring
        learn_fixed_sequence_order beforehand), and predict.
        """
        mapie_clf = MultiClassificationLTTController(
            predict_function=clf.predict_proba,
            target_level=0.75,
            risks=build_precision_ovr(np.unique(y_toy)),
            confidence_level=0.9,
            list_predict_params=np.linspace(0.05, 0.9, 90),
            fwer_method="split_fixed_sequence",
            risk_combination_method=1,
        )

        with pytest.raises(
            ValueError,
            match=r"You must call 'learn_fixed_sequence_order' before 'calibrate'",
        ):
            mapie_clf.calibrate(X_toy, y_toy)

        mapie_clf.learn_fixed_sequence_order(X_toy, y_toy)
        mapie_clf.calibrate(X_toy, y_toy)

        assert mapie_clf.p_values is not None
        assert mapie_clf.valid_predict_params is not None

        predictions = mapie_clf.predict(X_toy)
        assert predictions.shape == (len(X_toy),)


def _raise_value_error(X):
    raise ValueError("Luke")


def _raise_index_error(X):
    raise IndexError("I am")


def _raise_type_error(X):
    raise TypeError("Your Father")


class TestMulticlassificationControllerGetPredictionsPerParam:
    @pytest.mark.parametrize(
        "predict_function,expected_error_type,expected_error_message",
        [
            (clf, TypeError, r"Error when calling the predict_function."),
            (_raise_value_error, ValueError, "Luke"),
            (_raise_index_error, IndexError, "I am"),
            (_raise_type_error, TypeError, "Your Father"),
        ],
    )
    def test_errors(
        self,
        predict_function: Any,
        expected_error_type: Any,
        expected_error_message: str,
    ) -> None:
        """
        Passing a classifier instead of a predict_proba method raises a
        wrapped TypeError; any other error raised by predict_function is
        re-raised without modification, mirroring BinaryClassificationController's
        equivalent test.
        """
        mapie_clf = MultiClassificationLTTController(
            predict_function=predict_function,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
            risk_combination_method=1,
        )

        with pytest.raises(expected_error_type, match=expected_error_message):
            mapie_clf._get_predictions_per_param(X_toy, np.array([0.5]))


class TestCheckPredictionsMulticlassification:
    @pytest.mark.parametrize(
        "predictions,expected_error_message",
        [
            (
                np.array([0.1, 0.2, 0.3]),
                r"Maybe you provided a predict method instead of a",
            ),
            (
                np.array([[0.5, 0.6], [0.3, 0.3]]),
                r"must lie in \[0, 1\]",
            ),
        ],
    )
    def test_errors(
        self,
        mcc_dummy: MultiClassificationLTTController,
        predictions: Any,
        expected_error_message: str,
    ) -> None:
        """
        _check_predictions must raise a ValueError when
        predictions are not 2D (e.g. a predict method was provided instead of
        a predict_proba method), or when probabilities are invalid (out of
        [0, 1], or rows that don't sum to 1).
        """
        with pytest.raises(ValueError, match=expected_error_message):
            mcc_dummy._check_predictions(predictions)


def _custom_combination(array: Any) -> Any:
    return array.max(axis=0)  # pragma: no cover


class TestCheckRiskCombinationMethod:
    @pytest.mark.parametrize(
        "y_calibrate, risk_combination_method, expected_index",
        [
            (np.array([0, 1, 2, 1, 0]), 0, 0),
            (np.array([0, 1, 2, 1, 0]), 1, 1),
            (np.array([0, 1, 2, 1, 0]), 2, 2),
            # Labels not sorted/contiguous in the data: dict_class_label is
            # built from np.unique, so it is always sorted ascending.
            (np.array([5, 2, 8]), 8, 2),
            (np.array(["b", "a", "c"]), "b", 1),
            # A callable must be passed through as-is (expected_index unused).
            (np.array([0, 1, 2]), _custom_combination, None),
        ],
    )
    def test_valid_inputs(
        self,
        mcc_dummy: MultiClassificationLTTController,
        y_calibrate: Any,
        risk_combination_method: Any,
        expected_index: Optional[int],
    ) -> None:
        """
        When risk_combination_method is a class label, the returned callable
        must select the row of the input array corresponding to that label's
        position in np.unique(y_calibrate). When it is already a callable, it
        must be passed through unchanged.
        """
        result = mcc_dummy._check_risk_combination_method(
            risk_combination_method, y_calibrate
        )

        if callable(risk_combination_method):
            assert result is risk_combination_method
        else:
            array = np.array([[10, 20], [30, 40], [50, 60]])
            np.testing.assert_array_equal(result(array), array[expected_index, :])

    @pytest.mark.parametrize(
        "y_calibrate, risk_combination_method, expected_error_type, expected_error_message",
        [
            (_Y_INT, 3, AssertionError, r"must be in list_class_label"),
            (_Y_INT, -1, AssertionError, r"must be in list_class_label"),
            (_Y_STR, "z", AssertionError, r"must be in list_class_label"),
            (_Y_INT, 2.5, TypeError, r"class label or a callable"),
            (_Y_INT, [1, 2], TypeError, r"class label or a callable"),
            (_Y_INT, (1,), TypeError, r"class label or a callable"),
            (_Y_INT, None, TypeError, r"class label or a callable"),
            (_Y_INT, {1: 2}, TypeError, r"class label or a callable"),
        ],
    )
    def test_invalid_inputs_raise(
        self,
        mcc_dummy: MultiClassificationLTTController,
        y_calibrate: Any,
        risk_combination_method: Any,
        expected_error_type: Any,
        expected_error_message: str,
    ) -> None:
        """
        An unknown class label must raise an AssertionError. A
        risk_combination_method that is neither a class label nor a callable
        must raise a TypeError.
        """
        with pytest.raises(expected_error_type, match=expected_error_message):
            mcc_dummy._check_risk_combination_method(
                risk_combination_method, y_calibrate
            )


class TestSetBestPredictParamChoice:
    def test_set_best_predict_param_choice(self) -> None:
        """
        "auto" is the only supported value: it must return the full list of
        risks (unlike BinaryClassificationController, which picks a single
        complementary risk), and is set automatically at init. Anything else
        raises a NotImplementedError.
        """
        risks = build_precision_ovr([0, 1, 2])
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=risks,
            risk_combination_method=1,
        )

        assert mcc._set_best_predict_param_choice("auto") is risks
        assert mcc._best_predict_param_choice is risks

        with pytest.raises(NotImplementedError):
            mcc._set_best_predict_param_choice("not_auto")  # type: ignore[arg-type]

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.dummy import DummyClassifier

from numpy.typing import NDArray
from mapie.risk_control import (
    precision,
    recall,
    BinaryClassificationRisk, false_positive_rate,
    BinaryClassificationController, accuracy,
)

random_state = 42
dummy_single_param = np.array([0.5])
dummy_target = 0.9
dummy_X = [[0]]
dummy_y = np.array([1, 0])
dummy_predictions = np.array([[True, False]])


def fpr_func(y_true: NDArray, y_pred: NDArray) -> float:
    """Computes false positive rate."""
    tn: int = np.sum((y_true == 0) & (y_pred == 0))
    fp: int = np.sum((y_true == 0) & (y_pred == 1))
    return fp / (tn + fp)


def dummy_predict(X):
    return np.random.rand(1, 2)  # pragma: no cover


@pytest.fixture
def bcc_dummy():
    return BinaryClassificationController(
        predict_function=dummy_predict,
        risk=precision,
        target_level=dummy_target,
    )


def deterministic_predict_function(X):
    probs1 = np.array([0.2, 0.5, 0.9])
    probs0 = 1.0 - probs1
    return np.stack([probs0, probs1], axis=1)


@pytest.fixture
def bcc_deterministic():
    return BinaryClassificationController(
        predict_function=deterministic_predict_function,
        risk=precision,
        target_level=dummy_target,
    )


# The following test is voluntarily agnostic
# to the specific binary classification risk control implementation.
@pytest.mark.parametrize(
    "risk_instance, metric_func, effective_sample_func",
    [
        (precision, precision_score, lambda y_true, y_pred: np.sum(y_pred == 1)),
        (recall, recall_score, lambda y_true, y_pred: np.sum(y_true == 1)),
        (false_positive_rate, fpr_func, lambda y_true, y_pred: np.sum(y_true == 0)),
    ],
)
@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])),
        (np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1])),
    ],
)
def test_binary_classification_risk(
    risk_instance: BinaryClassificationRisk,
    metric_func,
    effective_sample_func,
    y_true,
    y_pred
):
    value, n = risk_instance.get_value_and_effective_sample_size(
        y_true, y_pred)
    effective_sample_size = effective_sample_func(y_true, y_pred)

    if effective_sample_size != 0:
        expected_value = metric_func(y_true, y_pred)
        expected_n = effective_sample_size
    else:
        expected_value = 1
        expected_n = -1
    if risk_instance.higher_is_better:
        expected_value = 1 - expected_value
    assert np.isclose(value, expected_value)
    assert n == expected_n


class TestBinaryClassificationControllerBestPredictParamChoice:
    @pytest.mark.parametrize(
        "risk_instance, expected",
        [
            (precision, recall),
            (recall, precision),
            (accuracy, accuracy),
            (false_positive_rate, recall),
        ],
    )
    def test_auto(
        self,
        risk_instance: BinaryClassificationRisk,
        expected
    ):
        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=risk_instance,
            target_level=dummy_target,
            best_predict_param_choice="auto"
        )

        result = controller._best_predict_param_choice
        assert result is expected

    def test_custom(self):
        """Test _set_best_predict_param_choice with a custom risk instance."""
        custom_risk = accuracy

        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            best_predict_param_choice=custom_risk
        )

        result = controller._set_best_predict_param_choice(custom_risk)
        assert result is custom_risk

    def test_auto_unknown_risk(self):
        """Test _set_best_predict_param_choice with 'auto' mode for unknown risk."""
        unknown_risk = deepcopy(accuracy)

        with pytest.raises(ValueError):
            BinaryClassificationController(
                predict_function=dummy_predict,
                risk=unknown_risk,
                target_level=dummy_target,
                best_predict_param_choice="auto"
            )


@pytest.mark.parametrize(
    "risk_instance,target_level,expected_alpha",
    [
        (recall, 0.6, 0.4),  # higher_is_better=True
        (false_positive_rate, 0.6, 0.6),  # higher_is_better=False
    ],
)
def test_binary_classification_controller_alpha(
    risk_instance: BinaryClassificationRisk,
    target_level: float,
    expected_alpha: float,
) -> None:
    controller = BinaryClassificationController(
        predict_function=dummy_predict,
        risk=risk_instance,
        target_level=target_level,
    )
    assert np.isclose(controller._alpha, expected_alpha)


def test_binary_classification_controller_sklearn_pipeline_with_dataframe() -> None:
    X_df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([1, 1, 0, 1], dtype=int)

    pipe = Pipeline(
        steps=[("clf", DummyClassifier(random_state=random_state))])
    pipe.fit(X_df, y)

    controller = BinaryClassificationController(
        predict_function=pipe.predict_proba,
        risk=precision,
        target_level=0.1,
        confidence_level=0.1,
    )

    controller.calibrate(X_df, y)
    controller.predict(X_df)


def test_set_risk_not_controlled(bcc_dummy):
    controller = bcc_dummy
    with pytest.warns(
        UserWarning,
        match=r"No predict parameters were found to control the risk"
    ):
        controller._set_risk_not_controlled()
    assert controller.best_predict_param is None


class TestBinaryClassificationControllerSetBestPredictParam:
    @pytest.mark.parametrize("best_predict_param_choice", ["auto", precision, recall])
    def test_only_one_param(self, best_predict_param_choice):
        """
        Expected: should always set this param
        """
        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            best_predict_param_choice=best_predict_param_choice
        )

        valid_params_index = [0]
        controller.valid_predict_params = dummy_single_param

        controller._set_best_predict_param(
            y_calibrate_=dummy_y,
            predictions_per_param=dummy_predictions,
            valid_params_index=valid_params_index
        )

        assert controller.best_predict_param == dummy_single_param[0]

    @pytest.mark.parametrize(
        "best_predict_param_choice, expected",
        [[precision, 0.5], [recall, 0.7]]
    )
    def test_correct_param_out_of_two(self, best_predict_param_choice, expected):
        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            best_predict_param_choice=best_predict_param_choice
        )

        y_calibrate = np.array([1, 1, 0])
        predictions_per_param = np.array(
            [
                [True, False, False],
                [True, True, True]
            ]
        )
        valid_params_index = [0, 1]

        controller.valid_predict_params = np.array(
            [0.5, 0.7])

        controller._set_best_predict_param(
            y_calibrate_=y_calibrate,
            predictions_per_param=predictions_per_param,
            valid_params_index=valid_params_index
        )

        assert controller.best_predict_param == expected

    def test_secondary_risk_undefined(self):
        """
        Expected: should set the param even though precision is not defined
        """
        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            best_predict_param_choice=precision
        )

        y_calibrate = np.array([1, 0])
        predictions_per_param = np.array(
            [[False, False]])  # precision undefined
        valid_params_index = [0]
        controller.valid_predict_params = dummy_single_param

        controller._set_best_predict_param(
            y_calibrate_=y_calibrate,
            predictions_per_param=predictions_per_param,
            valid_params_index=valid_params_index
        )
        assert controller.best_predict_param == dummy_single_param[0]


class TestBinaryClassificationControllerGetPredictionsPerParam:
    def test_single_parameter(self, bcc_deterministic):
        result = bcc_deterministic._get_predictions_per_param(
            X=[],
            params=dummy_single_param
        )

        expected = np.array([[False, True, True]])
        assert result.shape == (1, 3)
        assert result.dtype == int
        np.testing.assert_array_equal(result, expected)

    def test_multiple_parameters(self, bcc_deterministic):
        result = bcc_deterministic._get_predictions_per_param(
            X=[],
            params=np.array([0.0, 0.5, 0.8])
        )

        expected = np.array([
            [True, True, True],
            [False, True, True],
            [False, False, True],
        ])
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape_consistency(self):
        def predict_fn(X):
            return np.array([[0.1, 0.9], [0.7, 0.3], [0.4, 0.6]])

        controller = BinaryClassificationController(
            predict_function=predict_fn,
            risk=precision,
            target_level=dummy_target,
        )

        params = np.array([0.2, 0.5, 0.8])
        result = controller._get_predictions_per_param(X=[], params=params)

        assert result.shape == (len(params), 3)

    def test_error_passing_classifier(self):
        """
        Test when the user provides a classifier instead of a predict_proba
        method
        """
        clf = LogisticRegression().fit([[0], [1]], [0, 1])
        bcc = BinaryClassificationController(
            predict_function=clf,
            risk=precision,
            target_level=dummy_target
        )

        with pytest.raises(
            TypeError,
            match=r"Maybe you provided a binary classifier"
        ):
            bcc._get_predictions_per_param(dummy_X, dummy_single_param)

    def test_error_incorrect_predict_shape(self):
        """
        Test when the user provides a predict function that outputs only
        the positive class.
        """
        clf = LogisticRegression().fit([[0], [1]], [0, 1])

        def pred_func(X):
            return clf.predict_proba(X)[:, 0]

        bcc = BinaryClassificationController(
            predict_function=pred_func,
            risk=precision,
            target_level=dummy_target
        )

        with pytest.raises(
            IndexError,
            match=r"Maybe the predict function you provided returns only the "
                  r"probability of the positive class."
        ):
            bcc._get_predictions_per_param(dummy_X, dummy_single_param)

    @pytest.mark.parametrize(
        "error,expected_error_type,expected_error_message",
        [
            (ValueError("Hey"), ValueError, "Hey"),
            (IndexError("Gloups"), IndexError, "Gloups"),
            (TypeError("I'm hungry"), TypeError, "I'm hungry"),
        ],
    )
    def test_other_error(self, error, expected_error_type, expected_error_message):
        """Test that other errors are re-raised without modification"""

        def failing_predict_function(X):
            raise error

        bcc = BinaryClassificationController(
            predict_function=failing_predict_function,
            risk=precision,
            target_level=dummy_target
        )

        with pytest.raises(expected_error_type, match=expected_error_message):
            bcc._get_predictions_per_param(dummy_X, dummy_single_param)


class TestBinaryClassificationControllerPredict:
    def test_output_shape(self, bcc_deterministic):
        controller = bcc_deterministic
        controller.best_predict_param = 0.5
        predictions = controller.predict([])

        assert predictions.shape == (3,)
        assert predictions.dtype == int

    def test_error(self, bcc_dummy):
        controller = bcc_dummy
        controller.best_predict_param = None

        with pytest.raises(
            ValueError,
            match=r"Cannot predict"
        ):
            controller.predict(dummy_X)

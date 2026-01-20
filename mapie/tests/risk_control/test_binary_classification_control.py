from copy import deepcopy
from typing import List, Union

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mapie.risk_control import (
    BinaryClassificationController,
    BinaryClassificationRisk,
    accuracy,
    false_positive_rate,
    precision,
    recall,
)
from mapie.risk_control.binary_classification import Risk

random_state = 42
dummy_single_param = np.array([0.5])
dummy_single_param_multi_dim = np.array([[0.3, 0.7]])
dummy_grid_param_multi_dim = np.array(
    [[l1, l2] for l1 in [0.5, 0.7] for l2 in [0.2, 0.4]]
)

dummy_target = 0.9
dummy_X = [[0]]
dummy_y = np.array([1, 0])
dummy_predictions = np.array([[True, False]])
realistic_X, realistic_y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42,
)
(realistic_X_train, realistic_X_calib, realistic_y_train, realistic_y_calib) = (
    train_test_split(realistic_X, realistic_y, test_size=0.4, random_state=42)
)
realistic_clf = LogisticRegression().fit(realistic_X_train, realistic_y_train)


def fpr_func(y_true: NDArray, y_pred: NDArray) -> float:
    """Computes false positive rate."""
    tn: int = np.sum((y_true == 0) & (y_pred == 0))
    fp: int = np.sum((y_true == 0) & (y_pred == 1))
    return fp / (tn + fp)


def dummy_predict(X):
    return np.random.rand(1, 2)  # pragma: no cover


def dummy_predict_general_2d(X, param1, param2):
    return np.ones(len(X))


@pytest.fixture
def bcc_dummy():
    return BinaryClassificationController(
        predict_function=dummy_predict,
        risk=precision,
        target_level=dummy_target,
    )


@pytest.fixture
def bcc_dummy_multi_dim():
    return BinaryClassificationController(
        predict_function=dummy_predict_general_2d,
        risk=precision,
        target_level=dummy_target,
        list_predict_params=dummy_grid_param_multi_dim,
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
    y_pred,
):
    value, n = risk_instance.get_value_and_effective_sample_size(y_true, y_pred)
    effective_sample_size = effective_sample_func(y_true, y_pred)

    if effective_sample_size != 0:
        expected_value = metric_func(y_true, y_pred)
        expected_n = effective_sample_size
        if risk_instance.higher_is_better:
            expected_value = 1 - expected_value
    else:
        expected_value = 1
        expected_n = -1

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
    def test_auto(self, risk_instance: BinaryClassificationRisk, expected):
        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=risk_instance,
            target_level=dummy_target,
            best_predict_param_choice="auto",
        )

        result = controller._best_predict_param_choice
        assert result is expected

    def test_str(self):
        """Test _set_best_predict_param_choice with a string risk name."""
        str_risk = "precision"

        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            best_predict_param_choice=str_risk,
        )

        result = controller._set_best_predict_param_choice(str_risk)
        assert result is BinaryClassificationController.risk_choice_map[str_risk]

    def test_custom(self):
        """Test _set_best_predict_param_choice with a custom risk instance."""
        custom_risk = accuracy

        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            best_predict_param_choice=custom_risk,
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
                best_predict_param_choice="auto",
            )

    def test_multi_risk_auto(self):
        """Test _set_best_predict_param_choice with 'auto' mode for multiple risks."""
        first_risk = precision
        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=[first_risk, recall],
            target_level=[dummy_target, dummy_target],
            best_predict_param_choice="auto",
        )

        result = controller._best_predict_param_choice
        assert result is first_risk

    @pytest.mark.parametrize("invalid_risk_choice", [0.5, 5, [0.5, 0.7]])
    def test_invalid_type(self, invalid_risk_choice):
        """Test _set_best_predict_param_choice with an invalid type."""
        invalid_risk_choice = 0.5

        with pytest.raises(
            TypeError, match=r".*best_predict_param_choice must be either.*"
        ):
            BinaryClassificationController(
                predict_function=dummy_predict,
                risk=precision,
                target_level=dummy_target,
                best_predict_param_choice=invalid_risk_choice,
            )


@pytest.mark.parametrize(
    "risk_instance,target_level,expected_alpha",
    [
        (recall, 0.6, 0.4),  # higher_is_better=True
        (false_positive_rate, 0.6, 0.6),  # higher_is_better=False
        ([recall, false_positive_rate], [0.7, 0.8], [0.3, 0.8]),  # multi-risk
    ],
)
def test_binary_classification__convert_target_level_to_alpha(
    risk_instance: BinaryClassificationRisk,
    target_level: float,
    expected_alpha: float,
) -> None:
    controller = BinaryClassificationController(
        predict_function=dummy_predict,
        risk=risk_instance,
        target_level=target_level,
    )
    assert np.isclose(controller._alpha, expected_alpha).all()


def test_binary_classification_controller_sklearn_pipeline_with_dataframe() -> None:
    X_df = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = np.array([1, 1, 0, 1], dtype=int)

    pipe = Pipeline(steps=[("clf", DummyClassifier(random_state=random_state))])
    pipe.fit(X_df, y)

    controller = BinaryClassificationController(
        predict_function=pipe.predict_proba,
        risk=precision,
        target_level=0.1,
        confidence_level=0.1,
    )

    controller.calibrate(X_df, y).predict(X_df)


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
            best_predict_param_choice=best_predict_param_choice,
        )

        valid_params_index = [0]
        controller.valid_predict_params = dummy_single_param

        controller._set_best_predict_param(
            y_calibrate_=dummy_y,
            predictions_per_param=dummy_predictions,
            valid_params_index=valid_params_index,
        )

        assert controller.best_predict_param == dummy_single_param[0]

    @pytest.mark.parametrize(
        "best_predict_param_choice, expected", [[precision, 0.5], [recall, 0.7]]
    )
    def test_correct_param_out_of_two(self, best_predict_param_choice, expected):
        controller = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            best_predict_param_choice=best_predict_param_choice,
        )

        y_calibrate = np.array([1, 1, 0])
        predictions_per_param = np.array([[True, False, False], [True, True, True]])
        valid_params_index = [0, 1]

        controller.valid_predict_params = np.array([0.5, 0.7])

        controller._set_best_predict_param(
            y_calibrate_=y_calibrate,
            predictions_per_param=predictions_per_param,
            valid_params_index=valid_params_index,
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
            best_predict_param_choice=precision,
        )

        y_calibrate = np.array([1, 0])
        predictions_per_param = np.array([[False, False]])  # precision undefined
        valid_params_index = [0]
        controller.valid_predict_params = dummy_single_param

        controller._set_best_predict_param(
            y_calibrate_=y_calibrate,
            predictions_per_param=predictions_per_param,
            valid_params_index=valid_params_index,
        )
        assert controller.best_predict_param == dummy_single_param[0]


class TestBinaryClassificationControllerGetPredictionsPerParam:
    def test_single_parameter(self, bcc_deterministic):
        result = bcc_deterministic._get_predictions_per_param(
            X=[], params=dummy_single_param
        )

        expected = np.array([[False, True, True]])
        assert result.shape == (1, 3)
        assert result.dtype == int
        np.testing.assert_array_equal(result, expected)

    def test_single_parameter_multi_dim(self, bcc_dummy_multi_dim):
        X = [1, 2, 3]
        result = bcc_dummy_multi_dim._get_predictions_per_param(
            X=X, params=dummy_single_param_multi_dim
        )

        expected = np.array([len(X) * [True]])
        assert result.shape == (len(dummy_single_param_multi_dim), len(X))
        assert result.dtype == float
        np.testing.assert_array_equal(result, expected)

    def test_multiple_parameters(self, bcc_deterministic):
        result = bcc_deterministic._get_predictions_per_param(
            X=[], params=np.array([0.0, 0.5, 0.8])
        )
        expected = np.array(
            [
                [True, True, True],
                [False, True, True],
                [False, False, True],
            ]
        )
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_parameters_multi_dim(self, bcc_dummy_multi_dim):
        X = [1, 2, 3]
        result = bcc_dummy_multi_dim._get_predictions_per_param(
            X=X, params=dummy_grid_param_multi_dim
        )

        expected = np.array(len(dummy_grid_param_multi_dim) * [len(X) * [True]])
        assert result.shape == (len(dummy_grid_param_multi_dim), len(X))
        assert result.dtype == float
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
            predict_function=clf, risk=precision, target_level=dummy_target
        )

        with pytest.raises(TypeError, match=r"Maybe you provided a binary classifier"):
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
            predict_function=pred_func, risk=precision, target_level=dummy_target
        )

        with pytest.raises(
            IndexError,
            match=r"Maybe the predict function you provided returns only the "
            r"probability of the positive class.",
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
            target_level=dummy_target,
        )

        with pytest.raises(expected_error_type, match=expected_error_message):
            bcc._get_predictions_per_param(dummy_X, dummy_single_param)

    def test_error_multi_dim_params_dim_mismatch(self):
        """Test error raised when params_dim do not match with predict function"""
        bcc_2d = BinaryClassificationController(
            predict_function=dummy_predict_general_2d,
            risk=precision,
            target_level=dummy_target,
            list_predict_params=dummy_grid_param_multi_dim,
        )
        dummy_grid_3d = np.array(
            [[l1, l2, l3] for l1 in [0.5, 0.7] for l2 in [0.2, 0.4] for l3 in [0.1]]
        )
        with pytest.raises(
            TypeError,
            match=r"takes \d+ positional arguments but \d+ were given",
        ):
            bcc_2d._get_predictions_per_param(dummy_X, dummy_grid_3d)

    def test_error_multi_dim_non_binary_predictions(self):
        """Test error raised when predictions are not binary (0 or 1)"""

        def non_binary_predict(X, *params):
            return 0.2 * np.ones(len(X))

        bcc = BinaryClassificationController(
            predict_function=non_binary_predict,
            risk=precision,
            target_level=dummy_target,
            list_predict_params=dummy_grid_param_multi_dim,
        )
        with pytest.raises(
            ValueError,
            match=r"must return binary predictions",
        ):
            bcc.calibrate([1, 2], [0, 1])

    def test_warning_one_dim_binary_predictions(self):
        """Test warning raised when predictions are binary (0 or 1) with one-dimensional parameters"""

        def binary_predict(X):
            return np.zeros((len(X), 2))

        bcc = BinaryClassificationController(
            predict_function=binary_predict, risk=precision, target_level=dummy_target
        )
        with pytest.warns(
            UserWarning,
            match=r"All predictions are either 0 or 1 while the parameters are one-dimensional.",
        ):
            bcc.calibrate([1, 2], [0, 1])


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

        with pytest.raises(ValueError, match=r"Cannot predict"):
            controller.predict(dummy_X)


class TestCheckIfMultiRiskControl:
    @pytest.mark.parametrize(
        "risk",
        [precision, "precision"],
    )
    def test_mono_risk(
        self, bcc_deterministic: BinaryClassificationController, risk: Risk
    ):
        is_multi_risk = bcc_deterministic._check_if_multi_risk_control(
            risk, dummy_target
        )
        assert not is_multi_risk

    @pytest.mark.parametrize(
        "risk",
        [[precision], ["precision"]],
    )
    def test_mono_risk_list(
        self, bcc_deterministic: BinaryClassificationController, risk: Risk
    ):
        is_multi_risk = bcc_deterministic._check_if_multi_risk_control(
            risk, [dummy_target]
        )
        assert not is_multi_risk

    @pytest.mark.parametrize(
        "risk",
        [
            [precision, recall],
            ["precision", "recall"],
            [precision, "recall"],
        ],
    )
    def test_multi_risk(
        self, bcc_deterministic: BinaryClassificationController, risk: Risk
    ):
        is_multi_risk = bcc_deterministic._check_if_multi_risk_control(
            risk, [dummy_target, dummy_target]
        )
        assert is_multi_risk

    @pytest.mark.parametrize(
        "risk,target_level",
        [
            ([], []),
            ([recall, false_positive_rate], 0.6),
            (false_positive_rate, [0.6, 0.8]),
            ([recall, false_positive_rate], [0.6, 0.8, 0.7]),
        ],
    )
    def test_error_cases(self, risk: Risk, target_level: Union[List[float], float]):
        with pytest.raises(ValueError, match="If you provide a list of risks,"):
            BinaryClassificationController._check_if_multi_risk_control(
                risk, target_level
            )


@pytest.mark.parametrize(
    "risk, target_level",
    [
        ("invalid_metric", dummy_target),
        (["precision", "false_positive_rate", "invalid_metric"], 3 * [dummy_target]),
        ([precision, "sensitivity"], 2 * [dummy_target]),
    ],
)
def test_invalid_risk_str_raises_error(
    risk: Risk, target_level: Union[List[float], float]
):
    with pytest.raises(ValueError, match="When risk is provided as a string,"):
        BinaryClassificationController(
            predict_function=deterministic_predict_function,
            risk=risk,
            target_level=target_level,
        )


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        (np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0])),
        (np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0])),
        (np.array([0, 0, 0, 0]), np.array([0, 1, 0, 1])),
    ],
)
def test_get_risk_values_and_eff_sample_sizes(y_true: NDArray, y_pred: NDArray):
    risk_list = [precision, recall, false_positive_rate]
    bcc = BinaryClassificationController(
        predict_function=deterministic_predict_function,
        risk=risk_list,
        target_level=[dummy_target] * len(risk_list),
    )
    all_values, all_n = bcc._get_risk_values_and_eff_sample_sizes(
        y_true, y_pred[np.newaxis, :], risk_list
    )

    for i, risk in enumerate(risk_list):
        value, n = risk.get_value_and_effective_sample_size(y_true, y_pred)
        assert np.isclose(all_values[i], value)
        assert all_n[i] == n


@pytest.mark.parametrize(
    "risks_1, targets_1, risks_2, targets_2",
    [
        # Lists of one risk and target should be equivalent to a single risk
        # and target.
        (
            [precision],
            [0.7],
            precision,
            0.7,
        ),
        # Lists with two identical risks and targets should be equivalent to a
        # single risk and target.
        (
            [precision, precision],
            [0.7, 0.7],
            precision,
            0.7,
        ),
        # Lists of multiple risks and targets should be equivalent
        # when order is swapped.
        (
            [precision, recall],
            [0.65, 0.6],
            [recall, precision],
            [0.6, 0.65],
        ),
        # Lists of identical risks with different targets should be equivalent to a
        # single risk with the most strict target.
        (
            [precision, precision],
            [0.7, 0.6],
            precision,
            0.7,
        ),
        # Lists with many risks should not pose an issue.
        (
            10 * [precision],
            10 * [0.7],
            precision,
            0.7,
        ),
        # Lists of multiple risks and targets
        # which mix str and BinaryClassificationRisk.
        (
            ["precision", "recall"],
            [0.65, 0.6],
            [precision, recall],
            [0.65, 0.6],
        ),
    ],
)
def test_functional_multi_risk(
    risks_1: List[BinaryClassificationRisk],
    targets_1: List[float],
    risks_2: Union[List[BinaryClassificationRisk], BinaryClassificationRisk],
    targets_2: Union[List[float], float],
):
    """
    Functional tests for multi-risk binary classification controller.
    Test cases where two different combinations of risks and targets should lead
    to the same result.
    """
    bcc_1 = BinaryClassificationController(
        predict_function=realistic_clf.predict_proba,
        risk=risks_1,
        target_level=targets_1,
    )
    bcc_1.calibrate(realistic_X_calib, realistic_y_calib)

    bcc_2 = BinaryClassificationController(
        predict_function=realistic_clf.predict_proba,
        risk=risks_2,
        target_level=targets_2,
    )
    bcc_2.calibrate(realistic_X_calib, realistic_y_calib)

    # check that both controllers found valid parameters
    assert len(bcc_1.valid_predict_params) > 1 and len(bcc_2.valid_predict_params) > 1
    assert bcc_1.best_predict_param is not None and bcc_2.best_predict_param is not None

    # check that both controllers found the same valid parameters and best param
    assert np.isclose(bcc_1.valid_predict_params, bcc_2.valid_predict_params).all()
    assert np.isclose(bcc_1.best_predict_param, bcc_2.best_predict_param)


def test_functional_multi_risk_vs_twice_mono_risk():
    """
    Functional test comparing multi-risk calibration to two separate
    mono-risk calibrations.
    """
    risks = [precision, recall]
    targets = [0.65, 0.7]

    bcc_multi = BinaryClassificationController(
        predict_function=realistic_clf.predict_proba,
        risk=risks,
        target_level=targets,
    )
    bcc_multi.calibrate(realistic_X_calib, realistic_y_calib)

    valid_predict_params_mono = []
    for risk, target in zip(risks, targets):
        bcc_mono = BinaryClassificationController(
            predict_function=realistic_clf.predict_proba,
            risk=risk,
            target_level=target,
        )
        bcc_mono.calibrate(realistic_X_calib, realistic_y_calib)
        valid_predict_params_mono.append(bcc_mono.valid_predict_params)

    # check that multi-risk controller found valid parameters
    assert len(bcc_multi.valid_predict_params) > 1
    assert bcc_multi.best_predict_param is not None

    # check that multi-risk valid parameters set is the intersection of mono-risk ones
    assert np.isclose(
        bcc_multi.valid_predict_params,
        np.intersect1d(valid_predict_params_mono[0], valid_predict_params_mono[1]),
    ).all()


class TestCheckIfMultiDimensionalParam:
    def test_mono_dimensional_param_default(self):
        """Test mono dimensional with default predict_params"""
        bcc = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
        )
        assert not bcc.is_multi_dimensional_param

    def test_mono_dimensional_param_custom(self):
        """Test mono dimensional with predict_params as 1D array"""
        predict_params = np.linspace(0, 0.99, 3)
        bcc = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            list_predict_params=predict_params,
        )
        assert not bcc.is_multi_dimensional_param

    def test_mono_dimensional_2d_param(self):
        """Test mono dimensional with predict_params as 2D array"""
        predict_params = np.linspace(0, 0.99, 3)
        predict_params = predict_params[:, np.newaxis]
        bcc = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            list_predict_params=predict_params,
        )
        assert bcc.is_multi_dimensional_param

    def test_multi_dimensional(self):
        """Test multi dimensional with predict_params as 2D array"""
        bcc = BinaryClassificationController(
            predict_function=dummy_predict,
            risk=precision,
            target_level=dummy_target,
            list_predict_params=dummy_grid_param_multi_dim,
        )
        assert bcc.is_multi_dimensional_param

    def test_multi_dimensional_error(self):
        """Test multi dimensional with predict_params as 3D array"""
        lambda_vals = np.linspace(0, 1, 3)
        lambda1_grid, lambda2_grid = np.meshgrid(lambda_vals, lambda_vals)
        predict_params = np.column_stack((lambda1_grid, lambda2_grid))
        predict_params = predict_params[:, :, np.newaxis]
        with pytest.raises(ValueError, match="predict_params must be a 1D array of"):
            BinaryClassificationController(
                predict_function=dummy_predict,
                risk=precision,
                target_level=dummy_target,
                list_predict_params=predict_params,
            )


def test_functional_multi_dimensional_params():
    """
    Functional test for multi-dimensional parameters BinaryClassificationController.
    """

    def realistic_general_predict(X, param1, param2):
        probs = realistic_clf.predict_proba(X)[:, 1]
        return ((probs >= param1) & (probs <= param2)).astype(int)

    grid_param_multi_dim = np.array(
        [[l1, l2] for l1 in [0.5, 0.6, 0.7] for l2 in [0.8, 0.9, 1.0]]
    )
    bcc_multi_dim = BinaryClassificationController(
        predict_function=realistic_general_predict,
        risk=precision,
        target_level=0.6,
        list_predict_params=grid_param_multi_dim,
    )
    bcc_multi_dim.calibrate(realistic_X_calib, realistic_y_calib)

    # check that controller found valid parameters
    assert len(bcc_multi_dim.valid_predict_params) > 1
    assert bcc_multi_dim.best_predict_param is not None
    assert isinstance(bcc_multi_dim.best_predict_param, tuple)
    assert len(bcc_multi_dim.best_predict_param) == 2


def test_functional_multi_dimensional_params_multi_risk():
    """
    Functional test for multi-dimensional parameters BinaryClassificationController with multiple risks.
    """

    def realistic_general_predict(X, param1, param2):
        probs = realistic_clf.predict_proba(X)[:, 1]
        return ((probs >= param1) & (probs <= param2)).astype(int)

    grid_param_multi_dim = np.array(
        [[l1, l2] for l1 in [0.5, 0.6, 0.7] for l2 in [0.8, 0.9, 1.0]]
    )
    bcc_multi_dim = BinaryClassificationController(
        predict_function=realistic_general_predict,
        risk=[precision, recall],
        target_level=[0.65, 0.6],
        list_predict_params=grid_param_multi_dim,
    )
    bcc_multi_dim.calibrate(realistic_X_calib, realistic_y_calib)

    # check that controller found valid parameters
    assert len(bcc_multi_dim.valid_predict_params) > 1
    assert bcc_multi_dim.best_predict_param is not None
    assert isinstance(bcc_multi_dim.best_predict_param, tuple)
    assert len(bcc_multi_dim.best_predict_param) == 2

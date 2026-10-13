from typing import Any

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie.risk_control import MultiClassificationLTTController
from mapie.risk_control.risks import build_abstention_rate_ovr, build_precision_ovr

random_state = 42
dummy_target = 0.75
_Y_INT = np.array([0, 1, 2])
_Y_STR = np.array(["a", "b", "c"])


def dummy_predict(X):
    return np.random.rand(1, 3)  # pragma: no cover


def dummy_predict_multi_dim(X, param1, param2):
    return np.full(len(X), param1 + param2)  # pragma: no cover


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

_CLASS_LABELS_STR = np.array(["a", "b", "c", "d", "e"])
y_toy_str = _CLASS_LABELS_STR[y_toy]


def predict_multi_dim_str(X, min_proba, min_margin):
    """
    predict_function for multi-dimensional params: it must return labels
    directly (not probabilities), here applying a confidence threshold
    (min_proba) and a margin threshold (min_margin) on top of clf's
    predict_proba, abstaining (np.nan) when not confident enough.
    """
    probas = clf.predict_proba(X)
    sorted_probas = np.sort(probas, axis=1)
    max_proba = sorted_probas[:, -1]
    margin = max_proba - sorted_probas[:, -2]
    predicted_label = _CLASS_LABELS_STR[probas.argmax(axis=1)].astype(object)
    is_confident = (max_proba >= min_proba) & (margin >= min_margin)
    return np.where(is_confident, predicted_label, np.nan)


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
            list_predict_params=np.linspace(0.9, 0.05, 90),
            fwer_method="split_fixed_sequence",
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


class TestMultiDimensionalStrClassesWorkflow:
    def test_workflow_multi_dimensional_str_classes(self) -> None:
        """
        Reproduce a full workflow combining multi-dimensional predict params
        (predict_function returns labels directly, bypassing
        default_agg_class_pba) with string class labels: build one-vs-rest
        abstention rate risks from string classes, calibrate, then predict.
        """
        params = np.array(
            [
                [min_proba, min_margin]
                for min_proba in [0.2, 0.3, 0.4, 0.5, 0.6]
                for min_margin in [0.0, 0.02, 0.05, 0.1]
            ]
        )
        mapie_clf = MultiClassificationLTTController(
            predict_function=predict_multi_dim_str,
            target_level=0.5,
            risks=build_abstention_rate_ovr(np.unique(y_toy_str)),
            confidence_level=0.9,
            list_predict_params=params,
        )
        assert mapie_clf.is_multi_dimensional_param

        mapie_clf.calibrate(X_toy, y_toy_str)

        assert len(mapie_clf.valid_predict_params) > 0
        assert isinstance(mapie_clf.best_predict_param, tuple)

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
        )

        with pytest.raises(expected_error_type, match=expected_error_message):
            mapie_clf._get_predictions_per_param(X_toy, np.array([0.5]))

    def test_multi_dimensional_param(self) -> None:
        """
        When list_predict_params is 2D, is_multi_dimensional_param is True
        and _get_predictions_per_param must delegate to
        _get_predictions_per_param_multi_params, which calls
        predict_function(X, *params[i]) for each row of params directly
        (predict_function already returns labels, not probabilities).
        """
        params = np.array([[0.1, 0.2], [0.3, 0.4]])
        mapie_clf = MultiClassificationLTTController(
            predict_function=dummy_predict_multi_dim,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
            list_predict_params=params,
        )
        assert mapie_clf.is_multi_dimensional_param

        y_pred = mapie_clf._get_predictions_per_param(
            X_toy, params, is_calibration_step=True
        )

        expected = np.array(
            [np.full(len(X_toy), 0.1 + 0.2), np.full(len(X_toy), 0.3 + 0.4)]
        )
        np.testing.assert_array_equal(y_pred, expected)


class TestSetBestPredictParamChoice:
    def test_set_best_predict_param_choice(self) -> None:
        """
        "auto" is the only supported value: it must return the first risk in
        the list (used to minimize the secondary risk when selecting the best
        predict param among valid ones), and is set automatically at init.
        Anything else raises a NotImplementedError.
        """
        risks = build_precision_ovr([0, 1, 2])
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=risks,
        )

        assert mcc._set_best_predict_param_choice("auto") is risks[0]
        assert mcc._best_predict_param_choice is risks[0]

        # Passing a ClassSpecificRisk instance directly must return it as-is.
        assert mcc._set_best_predict_param_choice(risks[0]) is risks[0]

        with pytest.raises(NotImplementedError):
            mcc._set_best_predict_param_choice("not_auto")  # type: ignore[arg-type]


class TestCheckPredictionsType:
    def test_check_predictions_type(self) -> None:
        """
        No error must be raised when predictions and y_calibrate share the
        same "is numeric" status, whether both are numeric labels or both
        are string labels. A TypeError must be raised when predictions are
        numeric but y_calibrate is made of string labels (or vice versa).
        """
        MultiClassificationLTTController._check_predictions_type(_Y_INT, _Y_INT)
        MultiClassificationLTTController._check_predictions_type(_Y_STR, _Y_STR)

        with pytest.raises(
            TypeError,
            match=r"predictions type .* is different from y_calibrate type",
        ):
            MultiClassificationLTTController._check_predictions_type(_Y_INT, _Y_STR)


class TestSetBestPredictParam:
    @pytest.mark.parametrize(
        "valid_predict_params, expected_best_predict_param",
        [
            # mono-dimensional: valid_predict_params[best_index] is a numpy
            # scalar, stored as a float.
            (np.array([0.3, 0.7]), 0.7),
            # multi-dimensional: valid_predict_params[best_index] is a 1D
            # array, stored as a tuple.
            (np.array([[0.3, 0.1], [0.7, 0.2]]), (0.7, 0.2)),
        ],
    )
    def test_set_best_predict_param(
        self, valid_predict_params, expected_best_predict_param
    ) -> None:
        """
        _set_best_predict_param must select, among the valid params, the one
        minimizing the secondary risk (here the first risk in `risks`, i.e.
        precision for class 0), and store it as a float for a
        mono-dimensional param, or as a tuple for a multi-dimensional one.
        """
        risks = build_precision_ovr([0, 1])
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=risks,
        )
        mcc.valid_predict_params = valid_predict_params

        y_calibrate_ = np.array([0, 1, 0, 1])
        # param 0: precision for class 0 is 1/2; param 1: precision for
        # class 0 is 1/1 (perfect), so param 1 has the lowest secondary risk
        # (1 - precision) and must be selected.
        predictions_per_param = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 0, 1],
            ]
        )
        valid_params_index = [0, 1]
        risk_values = np.array([[0.1, 0.2], [0.3, 0.4]])

        mcc._set_best_predict_param(
            y_calibrate_, predictions_per_param, valid_params_index, risk_values
        )

        assert mcc.best_predict_param == expected_best_predict_param


class TestCheckPredictions:
    def test_check_predictions(self) -> None:
        """
        _check_predictions must raise a ValueError when predict_function's
        output is not 2D (e.g. a predict method was provided instead of a
        predict_proba method), and also when its output rows are not valid
        probabilities (do not lie in [0, 1] or do not sum to 1, e.g. a
        decision_function or other scoring method was provided instead).
        """
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
        )
        with pytest.raises(ValueError, match=r"provided a predict method instead of a"):
            mcc._check_predictions(np.array([0.1, 0.2, 0.3]))

        invalid_probas = np.array([[0.5, 0.6], [0.2, 0.2]])
        with pytest.raises(ValueError, match=r"must be valid probabilities"):
            mcc._check_predictions(invalid_probas)


class TestDefaultAggClassPba:
    def test_classes_none_uses_predicted_index(self) -> None:
        """
        Before calibrate() has run, _classes_ is None: default_agg_class_pba
        must fall back to using the predicted index itself as the label.
        """
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
        )
        assert mcc._classes_ is None

        params = np.array([0.3])
        predictions_proba = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        y_pred = mcc.default_agg_class_pba(params, predictions_proba)

        np.testing.assert_array_equal(y_pred, np.array([[1, 0]]))

    def test_string_classes_are_used_as_labels(self) -> None:
        """
        When _classes_ holds non-numeric (e.g. string) labels,
        default_agg_class_pba must index into them to return the predicted
        class label, rather than the predicted index.
        """
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
        )
        mcc._classes_ = np.array(["a", "b", "c"])

        params = np.array([0.3])
        predictions_proba = np.array([[0.1, 0.7, 0.2]])
        y_pred = mcc.default_agg_class_pba(params, predictions_proba)

        assert y_pred[0, 0] == "b"

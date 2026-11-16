from typing import Any, List, Union, cast

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie.risk_control import MultiClassificationLTTController, abstention_rate
from mapie.risk_control.risks import build_precision_ovr

random_state = 42
dummy_target = [0.75, 0.75, 0.75]
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
            target_level=[0.7] * len(np.unique(y_toy)),
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
        _predict_pba_function) with string class labels: control the abstention
        rate with string classes, calibrate, then predict.
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
            target_level=[0.5],
            risks=[abstention_rate],
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


def predict_mono_dim_direct_labels(X, params):
    """
    predict_function for mono-dimensional params: it must return labels
    directly (not probabilities), applying a confidence threshold on top of
    clf's predict_proba, vectorized over params like _predict_pba_function.
    """
    probas = clf.predict_proba(X)
    max_proba = probas.max(axis=1)
    predicted_label = probas.argmax(axis=1)
    is_confident = max_proba[np.newaxis, :] >= np.asarray(params)[:, np.newaxis]
    return np.where(is_confident, predicted_label[np.newaxis, :], np.nan)


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

    def test_mono_dimensional_param_direct_labels(self) -> None:
        """
        When list_predict_params is 1D and predict_function requires a
        "params" argument besides X, _is_predict_proba_method must catch the
        resulting "positional argument" TypeError and return False, so that
        _get_predictions_per_param_mono_param calls predict_function(X,
        params) directly instead of going through _predict_pba_function.
        """
        params = np.array([0.3, 0.9])
        mapie_clf = MultiClassificationLTTController(
            predict_function=predict_mono_dim_direct_labels,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
            list_predict_params=params,
        )
        assert not mapie_clf.is_multi_dimensional_param

        y_pred = mapie_clf._get_predictions_per_param(X_toy, params)

        assert y_pred.shape == (len(params), len(X_toy))


class TestSetBestPredictParamChoice:
    def test_set_best_predict_param_choice(self) -> None:
        """
        "auto" is the only supported value: it must return the first risk in
        the list (used to minimize the secondary risk when selecting the best
        predict param among valid ones), and is set automatically at init.
        Anything else raises a TypeError.
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

        with pytest.raises(
            TypeError, match=r"best_predict_param_choice must be either 'auto'"
        ):
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
            target_level=[0.75, 0.75],
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


class TestPredictPbaFunction:
    def test_classes_none_uses_predicted_index(self) -> None:
        """
        Before calibrate() has run, _classes_ is None: _predict_pba_function
        must fall back to using the predicted index itself as the label.
        """
        predictions_proba = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        mcc = MultiClassificationLTTController(
            predict_function=lambda X: predictions_proba,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
        )
        assert mcc._classes_ is None

        params = np.array([0.3])
        y_pred = mcc._predict_pba_function(X_toy, params)

        np.testing.assert_array_equal(y_pred, np.array([[1, 0]]))

    def test_string_classes_are_used_as_labels(self) -> None:
        """
        When _classes_ holds non-numeric (e.g. string) labels,
        _predict_pba_function must index into them to return the predicted
        class label, rather than the predicted index.
        """
        predictions_proba = np.array([[0.1, 0.7, 0.2]])
        mcc = MultiClassificationLTTController(
            predict_function=lambda X: predictions_proba,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
        )
        mcc._classes_ = np.array(["a", "b", "c"])

        params = np.array([0.3])
        y_pred = mcc._predict_pba_function(X_toy, params)

        assert y_pred[0, 0] == "b"


class TestCheckIfMultiRiskControl:
    @pytest.mark.parametrize(
        "n_risks, expected_is_multi_risk", [(1, False), (2, True), (3, True)]
    )
    def test_is_multi_risk(self, n_risks, expected_is_multi_risk) -> None:
        """
        A single risk must not be considered as multi risk, so that
        fwer_method="fixed_sequence" is accepted.
        """
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=[0.75] * n_risks,
            risks=build_precision_ovr(list(range(n_risks))),
        )
        assert mcc.is_multi_risk is expected_is_multi_risk

    @pytest.mark.parametrize(
        "risks, target_level",
        [
            (build_precision_ovr([0, 1, 2]), 0.75),
            (build_precision_ovr([0, 1, 2]), [0.75, 0.75]),
            (build_precision_ovr([0, 1, 2])[0], [0.75]),
            ([], []),
        ],
    )
    def test_errors(self, risks, target_level) -> None:
        with pytest.raises(ValueError, match=r"same non-zero length"):
            MultiClassificationLTTController(
                predict_function=dummy_predict,
                target_level=target_level,
                risks=risks,
            )

    def test_fixed_sequence_single_risk_only(self) -> None:
        MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=[0.75],
            risks=build_precision_ovr([0]),
            fwer_method="fixed_sequence",
        )
        with pytest.raises(ValueError, match=r"Fixed sequence testing cannot be used"):
            MultiClassificationLTTController(
                predict_function=dummy_predict,
                target_level=dummy_target,
                risks=build_precision_ovr([0, 1, 2]),
                fwer_method="fixed_sequence",
            )


class TestClasses:
    def test_classes_set_at_init(self) -> None:
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
            classes=[0, 1, 2],
        )
        np.testing.assert_array_equal(mcc._classes_, np.array([0, 1, 2]))

    @pytest.mark.parametrize("classes", [[0], [0, 1, 1], [[0, 1], [1, 2]]])
    def test_invalid_classes(self, classes) -> None:
        with pytest.raises(ValueError, match=r"at least two distinct class labels"):
            MultiClassificationLTTController(
                predict_function=dummy_predict,
                target_level=dummy_target,
                risks=build_precision_ovr([0, 1, 2]),
                classes=classes,
            )

    def test_check_labels_in_classes(self) -> None:
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
            classes=[0, 1, 2],
        )
        mcc._check_labels_in_classes(np.array([0, 2, 2]))
        with pytest.raises(ValueError, match=r"not in classes: \[3\]"):
            mcc._check_labels_in_classes(np.array([0, 3]))

    def test_number_of_columns_must_match_classes(self) -> None:
        predictions_proba = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        mcc = MultiClassificationLTTController(
            predict_function=lambda X: predictions_proba,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
            classes=[0, 1],
        )
        with pytest.raises(ValueError, match=r"returns 3 columns but there are 2"):
            mcc._predict_pba_function(X_toy, np.array([0.3]))

    def test_calibrate_with_class_missing_from_calibration_set(self) -> None:
        """
        When classes is provided, calibrate must not infer the classes from y:
        if class 0 is absent from the calibration set, inferring them would
        shift the columns of predict_proba and mislabel the predictions.
        """
        labels = np.array([0, 1, 2])
        mask = y_toy % 3 != 0
        X_calib, y_calib = X_toy[mask], labels[y_toy[mask] % 3]
        clf_3 = LogisticRegression().fit(X_toy, labels[y_toy % 3])
        mcc = MultiClassificationLTTController(
            predict_function=clf_3.predict_proba,
            target_level=[0.5, 0.5],
            risks=build_precision_ovr([1, 2]),
            classes=clf_3.classes_,
        )
        mcc.calibrate(X_calib, y_calib)
        np.testing.assert_array_equal(mcc._classes_, clf_3.classes_)
        y_pred = mcc._predict_pba_function(X_toy, np.array([0.0]))
        assert 0 in y_pred


class TestPrepareY:
    def test_infers_classes_from_y(self) -> None:
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
        )
        y_ = mcc._prepare_y(["b", "a", "c", "a"])
        np.testing.assert_array_equal(y_, np.array(["b", "a", "c", "a"]))
        np.testing.assert_array_equal(mcc._classes_, np.array(["a", "b", "c"]))

    def test_keeps_given_classes(self) -> None:
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=dummy_target,
            risks=build_precision_ovr([0, 1, 2]),
            classes=[0, 1, 2],
        )
        mcc._prepare_y([1, 2])
        np.testing.assert_array_equal(mcc._classes_, np.array([0, 1, 2]))
        with pytest.raises(ValueError, match=r"not in classes"):
            mcc._prepare_y([1, 5])


class TestLearnFixedSequenceOrder:
    @pytest.mark.parametrize("labels", [[1, 2, 3], ["a", "b", "c"]])
    def test_sequence_does_not_depend_on_label_names(self, labels) -> None:
        """
        The learned sequence must be the same whatever the names of the
        classes: integer labels not starting at 0 and string labels used to
        be mislabelled or to raise an error.
        """
        X_3, y_3 = make_classification(
            n_samples=300,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            flip_y=0,
            random_state=0,
        )
        sequences = []
        for class_labels in ([0, 1, 2], labels):
            y_labels = np.asarray(class_labels)[y_3]
            clf_3 = LogisticRegression().fit(X_3, y_labels)
            mcc = MultiClassificationLTTController(
                predict_function=clf_3.predict_proba,
                target_level=[0.7] * 3,
                risks=build_precision_ovr(cast(List[Union[int, str]], class_labels)),
                list_predict_params=np.linspace(0.9, 0.05, 90),
                fwer_method="split_fixed_sequence",
            )
            mcc.learn_fixed_sequence_order(X_3, y_labels)
            sequences.append(mcc._learned_fixed_sequence)

        assert sequences[0] is not None and len(sequences[0]) > 1
        np.testing.assert_array_equal(sequences[0], sequences[1])


class TestBinaryRiskInMulticlass:
    def test_abstention_rate_as_single_risk(self) -> None:
        """
        A class-agnostic BinaryRisk such as abstention_rate can be controlled
        on its own by the multiclass controller, with string labels too.
        """
        mcc = MultiClassificationLTTController(
            predict_function=clf.predict_proba,
            target_level=[0.5],
            risks=[abstention_rate],
            list_predict_params=np.linspace(0.05, 0.9, 30),
        )
        assert not mcc.is_multi_risk

        mcc.calibrate(X_toy, y_toy_str)

        assert len(mcc.valid_predict_params) > 0
        assert mcc.predict(X_toy).shape == (len(X_toy),)

    def test_abstention_rate_with_class_specific_risks(self) -> None:
        risks = [abstention_rate, *build_precision_ovr([0, 1, 2])]
        mcc = MultiClassificationLTTController(
            predict_function=dummy_predict,
            target_level=[0.5, 0.75, 0.75, 0.75],
            risks=risks,
        )
        assert mcc.is_multi_risk
        # a BinaryRisk is accepted as best_predict_param_choice too
        assert mcc._set_best_predict_param_choice(abstention_rate) is abstention_rate

from copy import deepcopy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
    make_regression
)
from sklearn.multioutput import MultiOutputClassifier

from mapie.calibration import MapieCalibrator
from mapie.classification import MapieClassifier
from mapie.conformity_scores import (
    AbsoluteConformityScore,
    APSConformityScore,
    GammaConformityScore,
    LACConformityScore,
    TopKConformityScore
)
from mapie.mondrian import Mondrian
from mapie.multi_label_classification import MapieMultiLabelClassifier
from mapie.regression import MapieRegressor

VALID_MAPIE_ESTIMATORS_NAMES = [
    "calibration",
    "classif_score",
    "classif_lac",
    "classif_aps",
    "classif_cumulated_score",
    "classif_topk",
    "classif_lac_conformity",
    "classif_aps_conformity",
    "classif_topk_conformity",
    "multi_label_recall_crc",
    "multi_label_recall_rcps",
    "multi_label_precision_ltt",
    "regression_absolute_conformity",
    "regression_gamma_conformity",
]

VALID_MAPIE_ESTIMATORS = {
    "calibration": {
        "estimator": MapieCalibrator,
        "task": "calibration",
        "kwargs": {"method": "top_label"}
    },
    "classif_score": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"method": "score"}
    },
    "classif_lac": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"method": "lac"}
    },
    "classif_aps": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"method": "aps"}
    },
    "classif_cumulated_score": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"method": "cumulated_score"}
    },
    "classif_topk": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"method": "topk"}
    },
    "classif_lac_conformity": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"conformity_score": LACConformityScore()}
    },
    "classif_aps_conformity": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"conformity_score": APSConformityScore()}
    },
    "classif_topk_conformity": {
        "estimator": MapieClassifier,
        "task": "classification",
        "kwargs": {"conformity_score": TopKConformityScore()}
    },
    "multi_label_recall_crc": {
        "estimator": MapieMultiLabelClassifier,
        "task": "multilabel_classification",
        "kwargs": {"metric_control": "recall", "method": "crc"}
    },
    "multi_label_recall_rcps": {
        "estimator": MapieMultiLabelClassifier,
        "task": "multilabel_classification",
        "kwargs": {"metric_control": "recall", "method": "rcps"},
        "predict_kargs": {"delta": 0.01}
    },
    "multi_label_precision_ltt": {
        "estimator": MapieMultiLabelClassifier,
        "task": "multilabel_classification",
        "kwargs": {"metric_control": "precision", "method": "ltt"},
        "predict_kargs": {"delta": 0.01}
    },
    "regression_absolute_conformity": {
        "estimator": MapieRegressor,
        "task": "regression",
        "kwargs": {"conformity_score": AbsoluteConformityScore()}
    },
    "regression_gamma_conformity": {
        "estimator": MapieRegressor,
        "task": "regression",
        "kwargs": {"conformity_score": GammaConformityScore()}
    },
}

TOY_DATASETS = {
    "calibration": make_classification(
        n_samples=1000, n_features=5, n_informative=5,
        n_redundant=0, n_classes=10
    ),
    "classification": make_classification(
        n_samples=1000, n_features=5, n_informative=5,
        n_redundant=0, n_classes=10
    ),
    "multilabel_classification": make_multilabel_classification(
        n_samples=1000, n_features=5, n_classes=5, allow_unlabeled=False
    ),
    "regression": make_regression(
        n_samples=1000, n_features=5, n_informative=5
    )

}

ML_MODELS = {
    "calibration": LogisticRegression(),
    "classification": LogisticRegression(),
    "multilabel_classification": MultiOutputClassifier(
                LogisticRegression(multi_class="multinomial")
            ),
    "regression": LinearRegression(),
}


@pytest.mark.parametrize("mapie_estimator_name", VALID_MAPIE_ESTIMATORS_NAMES)
def test_valid_estimators_dont_fail(mapie_estimator_name):
    task_dict = VALID_MAPIE_ESTIMATORS[mapie_estimator_name]
    mapie_estimator = task_dict["estimator"]
    mapie_kwargs = task_dict["kwargs"]
    task = task_dict["task"]
    x, y = TOY_DATASETS[task]
    ml_model = ML_MODELS[task]
    groups = np.random.choice(10, len(x))
    model = clone(ml_model)
    model.fit(x, y)
    mapie_inst = deepcopy(mapie_estimator)
    if not isinstance(mapie_inst(), MapieMultiLabelClassifier):
        mondrian_cp = Mondrian(
            mapie_estimator=mapie_inst(estimator=model, cv="prefit")
        )
    else:
        mondrian_cp = Mondrian(
            mapie_estimator=mapie_inst(estimator=model, **mapie_kwargs),
        )
    if task == "multilabel_classification":
        mondrian_cp.fit(x, y, groups=groups)
        if mapie_estimator_name in [
            "multi_label_recall_rcps", "multi_label_precision_ltt"
        ]:
            mondrian_cp.predict(
                x, groups=groups, alpha=.2, **task_dict["predict_kargs"]
            )
        else:
            mondrian_cp.predict(x, groups=groups, alpha=.2)
    elif task == "calibration":
        mondrian_cp.fit(x, y, groups=groups, **mapie_kwargs)
        mondrian_cp.predict_proba(x, groups=groups)
    else:
        mondrian_cp.fit(x, y, groups=groups, **mapie_kwargs)
        mondrian_cp.predict(x, groups=groups, alpha=.2)

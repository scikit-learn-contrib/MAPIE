from .binary_classification import BinaryClassificationController
from .fwer_control import (
    FWERBonferroniCorrection,
    FWERBonferroniHolm,
    FWERFixedSequenceTesting,
    FWERProcedure,
    control_fwer,
)
from .multi_label_classification import MultiLabelClassificationController
from .risks import (
    BinaryRisk,
    BinaryClassificationRisk,
    ContinuousRisk,
    RiskLoss,
    abstention_rate,
    accuracy,
    false_positive_rate,
    mae,
    mean_absolute_error,
    mean_squared_error,
    miscoverage_loss,
    mse,
    negative_predictive_value,
    positive_predictive_value,
    precision,
    predicted_positive_fraction,
    recall,
    recall_loss,
)
from .semantic_segmentation import SemanticSegmentationController

# ``ConditionalExpectedRiskController`` depends on PyTorch (the ``conditional``
# extra). It is imported lazily so that importing ``mapie.risk_control`` does not
# require PyTorch; the import (and a helpful error if PyTorch is missing) only
# fires when the class is actually accessed.
_LAZY_IMPORTS = {
    "ConditionalExpectedRiskController": (
        "mapie.risk_control.adaptive_conformal_risk_control"
    ),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module = import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConditionalExpectedRiskController",
    "miscoverage_loss",
    "recall_loss",
    "MultiLabelClassificationController",
    "SemanticSegmentationController",
    "BinaryClassificationController",
    "BinaryRisk",
    "BinaryClassificationRisk",
    "ContinuousRisk",
    "RiskLoss",
    "accuracy",
    "false_positive_rate",
    "mae",
    "mean_absolute_error",
    "mean_squared_error",
    "mse",
    "precision",
    "recall",
    "predicted_positive_fraction",
    "positive_predictive_value",
    "negative_predictive_value",
    "abstention_rate",
    "control_fwer",
    "FWERProcedure",
    "FWERBonferroniHolm",
    "FWERFixedSequenceTesting",
    "FWERBonferroniCorrection",
]

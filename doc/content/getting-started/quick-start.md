# Quick Start with MAPIE

MAPIE adds statistical guarantees to the outputs of machine learning models.
The two most common starting points are:

- **conformal prediction**, which returns prediction intervals for regression
  or prediction sets for classification;
- **risk control**, which selects a decision rule that controls a metric such
  as precision or recall.

Conformal prediction works with scikit-learn-compatible estimators. Risk
controllers accept a fitted model's prediction function, so they can also be
used with models from other frameworks.

## 1. Installation

=== "pip"

    ```bash
    pip install mapie
    ```

=== "conda"

    ```bash
    conda install -c conda-forge mapie
    ```

=== "From GitHub"

    ```bash
    pip install git+https://github.com/scikit-learn-contrib/MAPIE
    ```

MAPIE requires Python 3.9 or later, NumPy 1.23 or later, and scikit-learn 1.4
or later.

!!! warning "Notebook users"
    After installing, upgrading, or downgrading MAPIE in Jupyter, Colab, or
    Kaggle, restart the kernel before importing MAPIE. Otherwise, Python may
    continue using an already-imported version from `sys.modules`.

## 2. Regression: Prediction Intervals

The split-conformal workflow uses separate training, conformalization, and test
sets. Fit any compatible regression model as usual, then give the fitted model
to MAPIE. MAPIE measures its errors on the conformalization set without
retraining it.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split

X, y = make_regression(
    n_samples=1_000,
    n_features=5,
    noise=10.0,
    random_state=42,
)
X_train, X_conf, X_test, y_train, y_conf, y_test = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.6,
        conformalize_size=0.2,
        test_size=0.2,
        random_state=42,
    )
)

regressor = Ridge().fit(X_train, y_train)

mapie_regressor = SplitConformalRegressor(
    estimator=regressor,
    confidence_level=0.9,
    prefit=True,
)
mapie_regressor.conformalize(X_conf, y_conf)

y_pred, y_intervals = mapie_regressor.predict_interval(X_test)
```

`y_pred` contains the point predictions. `y_intervals` contains lower and upper
bounds for the 90% prediction intervals.

## 3. Classification: Prediction Sets

Classification follows the same workflow: fit the original classifier, pass it
to MAPIE for conformalization, and then call `predict_set`. A prediction set may
contain one or several plausible labels.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from mapie.classification import SplitConformalClassifier
from mapie.utils import train_conformalize_test_split

X, y = make_classification(
    n_samples=1_000,
    n_features=10,
    random_state=42,
)
X_train, X_conf, X_test, y_train, y_conf, y_test = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.6,
        conformalize_size=0.2,
        test_size=0.2,
        random_state=42,
    )
)

classifier = RandomForestClassifier(random_state=42).fit(X_train, y_train)

mapie_classifier = SplitConformalClassifier(
    estimator=classifier,
    confidence_level=0.9,
    conformity_score="lac",
    prefit=True,
)
mapie_classifier.conformalize(X_conf, y_conf)

y_pred, y_sets = mapie_classifier.predict_set(X_test)
```

`y_sets` indicates which labels belong to the prediction set for each test
observation.

!!! note "MAPIE can also fit the estimator"
    If the estimator is not already fitted, set `prefit=False` and call MAPIE's
    `fit(X_train, y_train)` before `conformalize`. This option is available for
    both `SplitConformalRegressor` and `SplitConformalClassifier`.

## 4. Risk Control: Guaranteed Decision Thresholds

Risk control is useful when the goal concerns a downstream decision metric
rather than interval or prediction-set coverage. The example below fits a
binary classifier and asks MAPIE to find probability thresholds that guarantee
a precision of at least 80% with 90% confidence.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mapie.risk_control import BinaryClassificationController
from mapie.utils import train_conformalize_test_split

X, y = make_classification(
    n_samples=1_500,
    n_features=10,
    n_informative=5,
    class_sep=1.5,
    random_state=42,
)
X_train, X_conf, X_test, y_train, y_conf, y_test = (
    train_conformalize_test_split(
        X,
        y,
        train_size=0.6,
        conformalize_size=0.2,
        test_size=0.2,
        random_state=42,
    )
)

classifier = LogisticRegression(max_iter=1_000).fit(X_train, y_train)

controller = BinaryClassificationController(
    predict_function=classifier.predict_proba,
    risk="precision",
    target_level=0.8,
    confidence_level=0.9,
)
controller.calibrate(X_conf, y_conf)

y_pred = controller.predict(X_test)
selected_threshold = controller.best_predict_param
```

During `calibrate`, MAPIE tests a grid of thresholds and stores those that
control the requested metric in `valid_predict_params`. It then selects one as
`best_predict_param` using a secondary objective—recall in this example—and
`predict` applies that threshold to new observations.

!!! warning "A valid threshold is not guaranteed to exist"
    If the fitted model and conformalization data cannot support the requested
    target at the chosen confidence level, `best_predict_param` is `None` and
    `predict` raises a `ValueError`. Consider collecting more conformalization
    data, lowering the target or confidence level, or improving the model.

For multi-label classification, use
`MultiLabelClassificationController`. For pixel-level prediction sets, use
`SemanticSegmentationController`.

## Next Steps

- [Choosing the right algorithm](choosing-algorithm.md) compares the available
  conformal predictors and risk controllers.
- [Conformalization set](../conformal-prediction/split-cross-conformal.md)
  explains split- and cross-conformal workflows and their trade-offs.
- [Conformal Prediction overview](../conformal-prediction/index.md)
  introduces prediction intervals, prediction sets, and coverage guarantees.
- [Risk Control overview](../risk-control/index.md) explains risk
  targets, confidence levels, and the available methods.
- [All examples](../all-examples/index.md) contains complete runnable examples
  for every MAPIE application area.

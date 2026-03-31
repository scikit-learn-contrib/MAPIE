# Quick Start with MAPIE

This package allows you to easily estimate uncertainties in both regression and classification settings.

- In **regression** settings, MAPIE provides **prediction intervals** on single-output data.
- In **classification** settings, MAPIE provides **prediction sets** on multi-class data.
- In any case, MAPIE is compatible with **any scikit-learn-compatible estimator**.

---

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

---

## 2. Regression

Let us start with a basic regression problem. Here, we generate one-dimensional noisy data that we fit with a `MLPRegressor`.

```python
from sklearn.neural_network import MLPRegressor
from mapie.regression import SplitConformalRegressor
from mapie.utils import train_conformalize_test_split
import numpy as np

# Generate toy data
np.random.seed(42)
X = np.linspace(0, 5, 500).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 500)

# Split into train / conformalize / test
X_train, X_conf, X_test, y_train, y_conf, y_test = (
    train_conformalize_test_split(X, y, test_size=0.2, conformalize_size=0.25)
)

# Fit and conformalize
model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
mapie_reg = SplitConformalRegressor(estimator=model)
mapie_reg.fit(X_train, y_train)
mapie_reg.conformalize(X_conf, y_conf)

# Predict with intervals
y_pred, y_intervals = mapie_reg.predict_interval(X_test, confidence_level=0.9)
```

---

## 3. Classification

Similarly, it's possible to do the same for a basic classification problem.

```python
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import SplitConformalClassifier
from mapie.utils import train_conformalize_test_split
from sklearn.datasets import make_classification

# Generate toy data
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# Split into train / conformalize / test
X_train, X_conf, X_test, y_train, y_conf, y_test = (
    train_conformalize_test_split(X, y, test_size=0.2, conformalize_size=0.25)
)

# Fit and conformalize
model = RandomForestClassifier(random_state=42)
mapie_clf = SplitConformalClassifier(estimator=model)
mapie_clf.fit(X_train, y_train)
mapie_clf.conformalize(X_conf, y_conf)

# Predict with sets
y_pred, y_sets = mapie_clf.predict_set(X_test, confidence_level=0.9)
```

---

## 4. Risk Control

MAPIE implements risk control methods for multilabel classification (in particular, image segmentation) and binary classification.

```python
from mapie.risk_control import BinaryClassificationController

# After training your binary classifier...
controller = BinaryClassificationController()
# See the Risk Control documentation for the full workflow.
```

[:material-arrow-right: Full risk control documentation](../theory/risk-control.md)

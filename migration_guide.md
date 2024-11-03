# Migration Guide from MAPIE v0.9 to MAPIE v1

MAPIE v1 introduces several updates, enhancements, and structural changes that simplify the API by breaking down `MapieRegressor` functionality into dedicated classes for different conformal prediction methods. This guide outlines the key differences between MAPIE v0.9 and MAPIE v1 and provides instructions for adapting your code to the new structure.

---

## 1. Overview of Class Restructuring

MAPIE v1 organizes the `MapieRegressor` functionality into specific regressor classes, each optimized for a particular type of conformal prediction:

- **`SplitConformalRegressor`**: Handles split conformal prediction.
- **`CrossConformalRegressor`**: Implements cross-validation-based conformal prediction.
- **`JackknifeAfterBootstrapRegressor`**: Supports jackknife-after-bootstrap conformal prediction.
- **`ConformalizedQuantileRegressor`**: For quantile-based conformal prediction.

This modular approach makes it easier to select and configure a specific conformal regression method. Each class includes parameters relevant to its own methodology, reducing redundancy and improving readability.

### Migration Summary of `MapieRegressor` to New Classes

In MAPIE v0.9, `MapieRegressor` managed all conformal regression methods under a single interface, which sometimes led to parameter redundancy and ambiguity. In MAPIE v1, each method-specific class includes only the parameters and methods relevant to its method.

| MAPIE v0.9 Class | MAPIE v1 Classes                                      |
|------------------|-------------------------------------------------------|
| `MapieRegressor` | `SplitConformalRegressor`, `CrossConformalRegressor`, `JackknifeAfterBootstrapRegressor`, `ConformalizedQuantileRegressor` |

---

## 2. Parameter Changes

MAPIE v1 introduces changes in parameter naming, usage, and scope to align with the new class structure.

### Key Parameter Adjustments

- **`conformity_score`**:
  - In MAPIE v1, `conformity_score` can be a `str` (e.g., `"absolute"`) or an instance of `BaseRegressionScore`.
  - This change standardizes conformity score specification across all classes, allowing for simpler configurations.

- **`confidence_level`**:
  - Replaces `alpha` from v0.9, making it more intuitive by representing the desired confidence level directly (e.g., 0.9 for 90% confidence).
  - Now initialized when declaring the model instance instead of at the `predict` stage, unlike v0.9 where it was set during prediction.
  - This parameter signifies the theoretical coverage rate for prediction intervals (the percentage of true points expected within intervals).

- **`method` Parameter**:
  - In MAPIE v1, `method` is specific to `CrossConformalRegressor` and `JackknifeAfterBootstrapRegressor`, where it determines the approach used for calculating intervals (`"base"`, `"plus"`, or `"minmax"`).
  - Removed from `SplitConformalRegressor` and `ConformalizedQuantileRegressor` for clarity.

- **`cv` Parameter**:
  - Now exclusively in `CrossConformalRegressor`, allowing integers or `BaseCrossValidator` instances.
  - The `prefit` option is no longer relevant in cross-validation contexts, as it has been removed in favor of a more streamlined approach.

- **`prefit` Parameter**:
  - For models that require a pre-trained estimator, MAPIE v1 includes this as a separate parameter in relevant classes rather than embedding it in cross-validation methods.

- **`fit_params` and `predict_params`**:
  - In MAPIE v1, `fit_params` and `predict_params` have been standardized to use dictionaries, providing a more organized and flexible way to pass additional parameters during fitting and prediction.
  - `fit_params` is a dictionary passed to the `fit` method, allowing you to specify parameters for the model training process. For example, if you need to apply sample weights, you would include `sample_weight` in `fit_params` as `fit_params={'sample_weight': sample_weight_array}`.  
  - `predict_params`: is a dictionary that allows you to pass additional parameters to the `predict` or `predict_set` methods, specifically during the `conformalize` or `predict` steps in v1. By moving `predict_params` out of `fit` (as was often done in v0.9), MAPIE v1 makes the configuration of each stage more explicit.

---

## 3. Method Changes

MAPIE v1 redefines the fit and prediction workflow to offer more flexibility and clarity, adding new methods to separate stages in the conformal prediction process.

### Method Changes Overview

- **Data Split**:
  - In MAPIE v1, a `conf_split` function has been introduced to split the data `(X, y)` into training `(X_train, y_train)`, calibration `(X_calib, y_calib)`, and test sets `(X_test, y_test)`.
  - This contrasts with v0.9, where data splitting was a two-phase process. First, data `(X, y)` was divided into training `(X_train, y_train)` and test `(X_test, y_test)` sets using `train_test_split` from `sklearn`. In the second phase, the split between training and calibration was either done manually or handled internally by `MapieRegressor`.
  - This new approach in v1 gives users more control over data splitting, making it easier to manage training, calibration, and testing phases explicitly.
  - The `CrossConformalRegressor` is an exception, where train/calibration splitting happens internally because cross-validation requires more granular control over data splits.

- **`fit` Method**:
  - In v1, `fit` only trains the model on training data, without handling calibration.
  - Additional fitting parameters, like `sample_weight`, should be included in `fit_params`, keeping this method focused on training alone.

- **`conformalize` Method**:
  - This new method performs calibration after fitting, using separate calibration data (`X_calib`, `y_calib`).
  - `predict_params` can be passed here, allowing independent control over calibration and prediction stages.

- **`predict_set` and `predict` Methods**:
  - MAPIE v1 introduces `predict_set` for interval prediction, distinguishing it clearly from `predict`, which now only provides point predictions.
  - `predict_set` returns prediction intervals (e.g., lower and upper bounds), while `predict` returns single-value predictions for each input sample.
  - The `optimize_beta` parameter, now specific to `CrossConformalRegressor`, is used to optimize prediction interval widths in certain cases.

## 4. Migration Example: MAPIE v0.9 to MAPIE v1

Below is a side-by-side example of code in MAPIE v0.9 and its equivalent in MAPIE v1 using the new modular classes and methods.

### MAPIE v0.9 Code

```python
from sklearn.linear_model import LinearRegression
from mapie.estimator import MapieRegressor
from mapie.conformity_scores import GammaConformityScore
from sklearn.model_selection import train_test_split

# Step 1: Split data
X_train, X_conf_test, y_train, y_conf_test = train_test_split(X, y, test_size=0.4)
X_conf, X_test, y_conf, y_test = train_test_split(X_conf_test, y_conf_test, test_size=0.5)

# Step 2: Train the model on the training set
prefit_model = LinearRegression().fit(X_train, y_train)

# Step 3: Initialize MapieRegressor with the prefit model and gamma conformity score
v0 = MapieRegressor(
    estimator=prefit_model,
    cv="prefit",
    conformity_score=GammaConformityScore()
)

# Step 4: Fit MAPIE on the calibration set
v0.fit(X_conf, y_conf)

# Step 5: Make predictions with confidence intervals
prediction_intervals_v0 = v0.predict(X_test, alpha=0.1)[1][:, :, 0]
prediction_points_v0 = v0.predict(X_test)
```

### Equivalent MAPIE v1 Code

```python
from sklearn.linear_model import LinearRegression
from mapie.estimator import SplitConformalRegressor
from mapie.utils import conf_split

# Step 1: Split data with conf_split (returns X_train, y_train, X_conf, y_conf, X_test, y_test)
X_train, y_train, X_conf, y_conf, X_test, y_test = conf_split(X, y)

# Step 2: Train the model on the training set
prefit_model = LinearRegression().fit(X_train, y_train)

# Step 3: Initialize SplitConformalRegressor with the prefit model, gamma conformity score, and prefit option
v1 = SplitConformalRegressor(
    estimator=prefit_model,
    confidence_level=0.9,       # equivalent to alpha=0.1 in v0.9
    conformity_score="gamma",
    prefit=True
)

# Step 4: Calibrate the model with the conformalize method on the calibration set
v1.conformalize(X_conf, y_conf)

# Step 5: Make predictions with confidence intervals
prediction_intervals_v1 = v1.predict_set(X_test)
prediction_points_v1 = v1.predict(X_test)
```
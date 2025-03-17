"""
===================================================================
Plotting CQR with symmetric argument
===================================================================


An example plot of :class:`~mapie_v1.regression.ConformalizedQuantileRegressor`
illustrating the impact of the symmetry parameter.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from mapie.metrics import regression_coverage_score
from mapie_v1.regression import ConformalizedQuantileRegressor

RANDOM_STATE = 1

##############################################################################
# We generate a synthetic data.

X, y = make_regression(
    n_samples=1000, n_features=1, noise=20, random_state=RANDOM_STATE
)

X_train_conformalize, X_test, y_train_conformalize, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

X_train, X_conformalize, y_train, y_conformalize = train_test_split(
    X_train_conformalize, y_train_conformalize, test_size=0.3, random_state=RANDOM_STATE
)

# Define confidence level
confidence_level = 0.8

# Initialize a Gradient Boosting Regressor for quantile regression
gb_reg = GradientBoostingRegressor(
    loss="quantile", alpha=0.5, random_state=RANDOM_STATE
)

# Using ConformalizedQuantileRegressor
mapie_qr = ConformalizedQuantileRegressor(
    estimator=gb_reg, confidence_level=confidence_level)
mapie_qr.fit(X_train, y_train)
mapie_qr.conformalize(X_conformalize, y_conformalize)
y_pred_sym, y_pis_sym = mapie_qr.predict_interval(X_test, symmetric_correction=True)
y_pred_asym, y_pis_asym = mapie_qr.predict_interval(X_test, symmetric_correction=False)
y_qlow = mapie_qr._mapie_quantile_regressor.estimators_[0].predict(X_test)
y_qup = mapie_qr._mapie_quantile_regressor.estimators_[1].predict(X_test)

print(f"y.shape: {y.shape}")
print(f"y_pis_sym[:, 0].shape: {y_pis_sym[:, 0].shape}")
print(f"y_pis_sym[:, 1].shape: {y_pis_sym[:, 1].shape}")
# Calculate coverage scores
coverage_score_sym = regression_coverage_score(
    y_test, y_pis_sym[:, 0], y_pis_sym[:, 1]
)
coverage_score_asym = regression_coverage_score(
    y_test, y_pis_asym[:, 0], y_pis_asym[:, 1]
)

# Sort the values for plotting
order = np.argsort(X_test[:, 0])
X_test_sorted = X_test[order]
y_pred_sym_sorted = y_pred_sym[order]
y_pis_sym_sorted = y_pis_sym[order]
y_pred_asym_sorted = y_pred_asym[order]
y_pis_asym_sorted = y_pis_asym[order]
y_qlow = y_qlow[order]
y_qup = y_qup[order]

##############################################################################
# We will plot the predictions and prediction intervals for both symmetric
# and asymmetric intervals. The line represents the predicted values, the
# dashed lines represent the prediction intervals, and the shaded area
# represents the symmetric and asymmetric prediction intervals.

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_test, y_test, alpha=0.3)
plt.plot(X_test_sorted, y_qlow, color="C1")
plt.plot(X_test_sorted, y_qup, color="C1")
plt.plot(X_test_sorted, y_pis_sym_sorted[:, 0], color="C1", ls="--")
plt.plot(X_test_sorted, y_pis_sym_sorted[:, 1], color="C1", ls="--")
plt.fill_between(
    X_test_sorted.ravel(),
    y_pis_sym_sorted[:, 0].ravel(),
    y_pis_sym_sorted[:, 1].ravel(),
    alpha=0.2,
)
plt.title(
    f"Symmetric Intervals\n"
    f"Target and effective coverages for "
    f"confidence_level={confidence_level:.2f}; coverage={coverage_score_sym:.3f})"
)

# Plot asymmetric prediction intervals
plt.subplot(1, 2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X_test, y_test, alpha=0.3)
plt.plot(X_test_sorted, y_qlow, color="C2")
plt.plot(X_test_sorted, y_qup, color="C2")
plt.plot(X_test_sorted, y_pis_asym_sorted[:, 0], color="C2", ls="--")
plt.plot(X_test_sorted, y_pis_asym_sorted[:, 1], color="C2", ls="--")
plt.fill_between(
    X_test_sorted.ravel(),
    y_pis_asym_sorted[:, 0].ravel(),
    y_pis_asym_sorted[:, 1].ravel(),
    alpha=0.2,
)
plt.title(
    f"Asymmetric Intervals\n"
    f"Target and effective coverages for "
    f"confidence_level={confidence_level:.2f}; coverage={coverage_score_sym:.3f})"
)
plt.tight_layout()
plt.show()

##############################################################################
# The symmetric intervals (`symmetry=True`) use a combined set of residuals
# for both bounds, while the asymmetric intervals use distinct residuals for
# each bound, allowing for more flexible and accurate intervals that reflect
# the heteroscedastic nature of the data. The resulting effective coverages
# demonstrate the theoretical guarantee of the target coverage level
# ``confidence_level``.

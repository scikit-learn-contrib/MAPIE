"""
======================================================
Plotting MAPIE Quantile Regressor prediction intervals
======================================================
An example plot of :class:`~mapie.quantile_regression.MapieQuantileRegressor`
illustrating the impact of the symmetry parameter.
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor

from mapie.metrics import regression_coverage_score
from mapie.quantile_regression import MapieQuantileRegressor

# Generate synthetic data
X, y = make_regression(n_samples=500, n_features=1, noise=20, random_state=59)

# Define alpha level
alpha = 0.2

# Fit a Gradient Boosting Regressor for quantile regression
quantiles = [0.1, 0.9]
gb_reg = GradientBoostingRegressor(loss="quantile", alpha=quantiles[1])
gb_reg.fit(X, y)

# MAPIE Quantile Regressor with symmetry=True
mapie_qr_sym = MapieQuantileRegressor(estimator=gb_reg, alpha=alpha)
mapie_qr_sym.fit(X, y)
y_pred_sym, y_pis_sym = mapie_qr_sym.predict(X, symmetry=True)

# MAPIE Quantile Regressor with symmetry=False
mapie_qr_asym = MapieQuantileRegressor(estimator=gb_reg, alpha=alpha)
mapie_qr_asym.fit(X, y)
y_pred_asym, y_pis_asym = mapie_qr_asym.predict(X, symmetry=False)

# Calculate coverage scores
coverage_score_sym = regression_coverage_score(y, y_pis_sym[:, 0], y_pis_sym[:, 1])
coverage_score_asym = regression_coverage_score(y, y_pis_asym[:, 0], y_pis_asym[:, 1])

# Sort the values for plotting
order = np.argsort(X[:, 0])
X_sorted = X[order]
y_pred_sym_sorted = y_pred_sym[order]
y_pis_sym_sorted = y_pis_sym[order]
y_pred_asym_sorted = y_pred_asym[order]
y_pis_asym_sorted = y_pis_asym[order]

# Plot symmetric prediction intervals
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X, y, alpha=0.3)
plt.plot(X_sorted, y_pred_sym_sorted, color="C1")
plt.plot(X_sorted, y_pis_sym_sorted[:, 0], color="C1", ls="--")
plt.plot(X_sorted, y_pis_sym_sorted[:, 1], color="C1", ls="--")
plt.fill_between(
    X_sorted.ravel(),
    y_pis_sym_sorted[:, 0].ravel(),
    y_pis_sym_sorted[:, 1].ravel(),
    alpha=0.2,
)
plt.title(
    f"Symmetric Intervals\n"
    f"Target and effective coverages for "
    f"alpha={alpha:.2f}: ({1-alpha:.3f}, {coverage_score_sym:.3f})"
)

# Plot asymmetric prediction intervals
plt.subplot(1, 2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(X, y, alpha=0.3)
plt.plot(X_sorted, y_pred_asym_sorted, color="C2")
plt.plot(X_sorted, y_pis_asym_sorted[:, 0], color="C2", ls="--")
plt.plot(X_sorted, y_pis_asym_sorted[:, 1], color="C2", ls="--")
plt.fill_between(
    X_sorted.ravel(),
    y_pis_asym_sorted[:, 0].ravel(),
    y_pis_asym_sorted[:, 1].ravel(),
    alpha=0.2,
)
plt.title(
    f"Asymmetric Intervals\n"
    f"Target and effective coverages for "
    f"alpha={alpha:.2f}: ({1-alpha:.3f}, {coverage_score_asym:.3f})"
)

plt.tight_layout()
plt.show()

# Explanation of the results
"""
The symmetric intervals (`symmetry=True`) are easier to interpret and tend to have higher 
coverage but might not adapt well to varying noise levels. The asymmetric intervals 
(`symmetry=False`) are more flexible and better capture heteroscedasticity but can appear 
more jagged.
"""

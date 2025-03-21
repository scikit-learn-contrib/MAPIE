"""
==========================================================================================
Use MAPIE on data with gamma distribution
==========================================================================================


This example uses :class:`~mapie_v1.regression.CrossConformalRegressor` to estimate
prediction intervals associated with Gamma distributed target.
The limit of the absolute residual conformity score is illustrated.

We use here the OpenML house_prices dataset:
https://www.openml.org/search?type=data&sort=runs&id=42165&status=active.

Note : OpenML is down as of 14/01/25, so we'll load the data from Kaggle instead.

The data is modelled by a Random Forest model
:class:`~sklearn.ensemble.RandomForestRegressor` with a fixed parameter set.
The prediction intervals are determined by means of the MAPIE regressor
:class:`~mapie_v1.regression.CrossConformalRegressor` considering two conformity scores:
``"absolute"`` which
considers the absolute residuals as the conformity scores and
``"gamma"`` which
considers the residuals divided by the predicted means as conformity scores.
We consider the standard CV+ resampling method.

We would like to emphasize one main limitation with this example.
With the default conformity score, the prediction intervals
are approximately equal over the range of house prices which may
be inapporpriate when the price range is wide. The Gamma conformity score
overcomes this issue by considering prediction intervals with width
proportional to the predicted mean. For low prices, the Gamma prediction
intervals are narrower than the default ones, conversely to high prices
for which the confidence intervals are higher but visually more relevant.
The empirical coverage is similar between the two conformity scores.
"""
import matplotlib.pyplot as plt
import numpy as np
import requests
import zipfile
import io
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from mapie.metrics import regression_coverage_score
from mapie_v1.regression import CrossConformalRegressor

RANDOM_STATE = 42

# Parameters
features = [
    "MS SubClass",
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Garage Area",
]
target = "SalePrice"

confidence_level = 0.95
rf_kwargs = {"n_estimators": 10, "random_state": RANDOM_STATE}
model = RandomForestRegressor(**rf_kwargs)

##############################################################################
# 1. Load dataset with a target following approximativeley a Gamma distribution
# -----------------------------------------------------------------------------
#
# We start by loading a dataset with a target following approximately
# a Gamma distribution.
# Two sub datasets are extracted: the training and test ones.

dataset_url = (
    "https://www.kaggle.com" +
    "/api/v1/datasets/download/shashanknecrothapa/ames-housing-dataset"
)
r = requests.get(dataset_url, stream=True)
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    with z.open("AmesHousing.csv") as file:
        data = pd.read_csv(file)

X = data[features]
y = data[target]

X_train_conformalize, X_test, y_train_conformalize, y_test = train_test_split(
    X[features], y, test_size=0.2, random_state=RANDOM_STATE
)

##############################################################################
# 2. Train model with two conformity scores
# -----------------------------------------
#
# Two models are trained with two different conformity score:
#
# - ``conformity_score = "absolute"`` (default
#   conformity score) is relevant for target positive as well as negative.
#   The prediction interval widths are, in this case, approximately the same
#   over the range of prediction.
#
# - ``conformity_score = "gamma"`` is relevant for target
#   following roughly a Gamma distribution. The prediction interval widths
#   scale with the predicted value.

##############################################################################
# First, train model with
# conformity_score = "absolute".
mapie = CrossConformalRegressor(
    model, confidence_level=confidence_level, conformity_score="absolute"
)
mapie.fit_conformalize(X_train_conformalize, y_train_conformalize)
y_pred_absconfscore, y_pis_absconfscore = mapie.predict_interval(
    X_test
)

coverage_absconfscore = regression_coverage_score(
    y_test, y_pis_absconfscore[:, 0, 0], y_pis_absconfscore[:, 1, 0]
)

##############################################################################
# Prepare the results for matplotlib. Get the prediction intervals and their
# corresponding widths.


def get_yerr(y_pred, y_pis):
    return np.concatenate(
        [
            np.expand_dims(y_pred, 0) - y_pis[:, 0, 0].T,
            y_pis[:, 1, 0].T - np.expand_dims(y_pred, 0),
        ],
        axis=0,
    )


yerr_absconfscore = get_yerr(y_pred_absconfscore, y_pis_absconfscore)
pred_int_width_absconfscore = (
    y_pis_absconfscore[:, 1, 0] - y_pis_absconfscore[:, 0, 0]
)

##############################################################################
# Then, train the model with:
# `conformity_score = "gamma"`.
mapie = CrossConformalRegressor(
    model, confidence_level=confidence_level, conformity_score="gamma"
)
mapie.fit_conformalize(X_train_conformalize, y_train_conformalize)
y_pred_gammaconfscore, y_pis_gammaconfscore = mapie.predict_interval(
    X_test
)

coverage_gammaconfscore = regression_coverage_score(
    y_test, y_pis_gammaconfscore[:, 0, 0], y_pis_gammaconfscore[:, 1, 0]
)

yerr_gammaconfscore = get_yerr(y_pred_gammaconfscore, y_pis_gammaconfscore)
pred_int_width_gammaconfscore = (
    y_pis_gammaconfscore[:, 1, 0] - y_pis_gammaconfscore[:, 0, 0]
)


##############################################################################
# 3. Compare the prediction intervals
# -----------------------------------
#
# Once the models have been trained, we now compare the prediction intervals
# obtained from the two conformity scores. We can see that the
# ``"absolute" ``conformity score generates
# prediction interval with almost the same width for all the predicted values.
# Conversely, the ``"gamma"`` conformity score
# yields prediction interval with width scaling with the predicted values.
#
# The choice of the conformity score depends on the problem we face.

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for img_id, y_pred, y_err, cov, class_name, int_width in zip(
    [0, 1],
    [y_pred_absconfscore, y_pred_gammaconfscore],
    [yerr_absconfscore, yerr_gammaconfscore],
    [coverage_absconfscore, coverage_gammaconfscore],
    ["AbsoluteResidualScore", "GammaResidualScore"],
    [pred_int_width_absconfscore, pred_int_width_gammaconfscore],
):
    axs[0, img_id].errorbar(
        y_test,
        y_pred,
        yerr=y_err,
        alpha=0.5,
        linestyle="None",
    )
    axs[0, img_id].scatter(y_test, y_pred, s=1, color="black")
    axs[0, img_id].plot(
        [0, max(max(y_test), max(y_pred))],
        [0, max(max(y_test), max(y_pred))],
        "-r",
    )
    axs[0, img_id].set_xlabel("Actual price [$]")
    axs[0, img_id].set_ylabel("Predicted price [$]")
    axs[0, img_id].grid()
    axs[0, img_id].set_title(f"{class_name} - coverage={cov:.0%}")

    xmin, xmax = axs[0, img_id].get_xlim()
    ymin, ymax = axs[0, img_id].get_ylim()
    axs[1, img_id].scatter(y_test, int_width, marker="+")
    axs[1, img_id].set_xlabel("Actual price [$]")
    axs[1, img_id].set_ylabel("Prediction interval width [$]")
    axs[1, img_id].grid()
    axs[1, img_id].set_xlim([xmin, xmax])
    axs[1, img_id].set_ylim([ymin, ymax])

fig.suptitle(
    f"Predicted values with the prediction intervals of level {confidence_level}"
)
plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.show()

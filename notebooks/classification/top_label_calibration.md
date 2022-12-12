---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3.10.8 ('mapie-dev')
    language: python
    name: python3
---

# Top-label calibration for outputs of ML models

The goal of this notebook is to present :class:`mapie.calibration.MapieCalibrator` by comparing it to the method presented in the paper for Top-label calibration [1].

[1] Gupta, Chirag, and Aaditya K. Ramdas. "Top-label calibration and multiclass-to-binary reductions." arXiv preprint arXiv:2107.08353 (2021).

```python
!git clone https://github.com/AIgen/df-posthoc-calibration
!cd df-posthoc-calibration && git checkout 109da93c1487cb38ee51fcac47088cdd29854999

```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-learn-contrib/MAPIE/blob/master/notebooks/classification/top_label_calibration.ipynb)



# Tutorial preparation

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from mapie.calibration import MapieCalibrator
from mapie.metrics import top_label_ece

import sys

sys.path.append("df-posthoc-calibration/")
import assessment
import calibration 

random_state = 20

%matplotlib inline

```

# 1. Creating a classification dataset

```python
centers = [(0, 3.5), (-2, 0), (2, 0)]
covs = [np.eye(2), np.eye(2)*2, np.diag([5, 1])]
x_min, x_max, y_min, y_max, step = -6, 8, -6, 8, 0.1
n_samples = 1000
n_classes = 3
np.random.seed(42)
X = np.vstack([
    np.random.multivariate_normal(center, cov, n_samples)
    for center, cov in zip(centers, covs)
])
y = np.hstack([np.full(n_samples, i) for i in range(n_classes)])
y += 1

X_train_cal, X_test, y_train_cal, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_train_cal, y_train_cal, test_size=0.25, random_state=random_state
)

```

# 2. Fitting a classifier on the training data

```python
clf = RandomForestClassifier(random_state=random_state)
clf.fit(X_train, y_train)

```

### Computing the prediction probabilities using the trained classifier

```python
preds_calib = clf.predict_proba(X_cal)
preds_test = clf.predict_proba(X_test)
arg_max_preds_test = clf.classes_[np.argmax(preds_test, axis=1)]
```

### Evaluating model

```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()
plt.show()
```

# 3. Calibration of data


### Using method from the paper [1].

```python
points_per_bin=50

# initialize recalibrator and set number of points per bins
hb = calibration.HB_toplabel(points_per_bin=points_per_bin)
hb.fit(preds_calib, y_cal)

# get histogram binning probabilities on test data
preds_test_hb = hb.predict_proba(preds_test)
```

### Using MAPIE `calibration.py`

```python
mapie_reg = MapieCalibrator(estimator=clf, cv="prefit", calibrator="isotonic")
mapie_reg.fit(X_cal, y_cal)
mapie_prob_preds = mapie_reg.predict_proba(X_test)
mapie_preds = mapie_reg.predict(X_test)

```

```python
# Verification that the same predictions are made.
np.testing.assert_array_equal(mapie_preds, clf.classes_[np.argmax(mapie_prob_preds, axis=1)])
np.testing.assert_array_equal(mapie_preds, arg_max_preds_test)
```

# 4. Evaluating the models using ECE and Reliability diagrams created in the paper.

Note that since we use different calibration methods, the results are slightly different, however, we still find similar results.

```python
# make some plots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5.5), constrained_layout=True)
fig.suptitle('Top-label reliability diagrams', fontsize=30)

assessment.toplabel_reliability_diagram(y_test, preds_test, ax=ax[0])
ax[0].set_title("Base ResNet-50 model \n (Top-label ECE = {:.3f})".format(top_label_ece(y_test, np.max(preds_test, axis=1), arg_max_preds_test)))

assessment.toplabel_reliability_diagram(y_test, preds_test_hb, arg_max_preds_test, ax=ax[1], color='g')
ax[1].set_title("ResNet-50 + histogram binning calibration\n (Top-label-ECE = {:.3f})".format(top_label_ece(y_test, preds_test_hb, arg_max_preds_test)));

assessment.toplabel_reliability_diagram(y_test, np.max(mapie_prob_preds, axis=1), arg_max_preds_test, ax=ax[2], color='g')
ax[2].set_title("ResNet-50 + MAPIE calibration (using Platt)\n (Top-label-ECE = {:.3f})".format(top_label_ece(y_test, np.max(mapie_prob_preds, axis=1), arg_max_preds_test)));
```

```python
!rm -r -f df-posthoc-calibration
```

```python

```

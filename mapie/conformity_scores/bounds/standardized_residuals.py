import warnings
from typing import Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_random_state, indexable

from mapie.conformity_scores import BaseRegressionScore
from mapie.utils import check_sklearn_user_model_is_fitted


import warnings
import os

import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
warnings.simplefilter('ignore', category=FutureWarning)

import multiprocessing as mp
import numpy as np
import sklearn
import pandas as pd
import torch

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from probmetrics.calibrators import get_calibrator

def _worker_init():
    """Restored: Forces spawned test-runner children to obey the filter."""
    import warnings
    warnings.simplefilter('ignore', category=FutureWarning)

class CheapBetterCatBoostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        m = CatBoostClassifier(iterations=1_000, learning_rate=0.08, subsample=0.9, bootstrap_type='Bernoulli',
                               max_depth=7, l2_leaf_reg=1e-5, random_strength=0.8, one_hot_max_size=15,
                               random_state=0, early_stopping_rounds=100, thread_count=1, verbose=0 if not self.verbose_ else 1)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y, verbose=False):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class BetterLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        m = LGBMClassifier(n_estimators=10_000, learning_rate=0.02, subsample=0.75, subsample_freq=1, num_leaves=50,
                           random_state=0, early_stopping_round=300, min_child_samples=40, min_child_weight=1e-7,
                           n_jobs=1, verbosity=-1 if not self.verbose_ else 1)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y, verbose=False):
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count()), initializer=_worker_init) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True).fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class BetterCatBoostClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, iterations=10000, early_stopping_rounds=300, thread_count=1, random_state=0):
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.thread_count = thread_count
        self.random_state = random_state

    def _fit_model(self, idxs):
        m = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=None,
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
            thread_count=self.thread_count,
            verbose=0 if not self.verbose_ else 1
        )
        return m.fit(
            self.X_.take(idxs[0], 0),
            self.y_[idxs[0]],
            eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]])
        )

    def fit(self, X, y, verbose=False):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(
            sklearn.model_selection.StratifiedKFold(
                n_splits=8, shuffle=True, random_state=0
            ).split(X, y)
        )
        with mp.Pool(processes=min(len(splits), mp.cpu_count())) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

class CheapLGBMClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def _fit_model(self, idxs):
        m = LGBMClassifier(n_estimators=1_000, learning_rate=0.04, subsample=0.75, subsample_freq=1, num_leaves=50,
                           random_state=0, early_stopping_round=100, min_child_samples=40, min_child_weight=1e-7,
                           n_jobs=1, verbosity=-1 if not self.verbose_ else 1)
        return m.fit(self.X_.take(idxs[0], 0), self.y_[idxs[0]], eval_set=(self.X_.take(idxs[1], 0), self.y_[idxs[1]]))

    def fit(self, X, y, verbose=False):
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        if isinstance(y, torch.Tensor) or isinstance(y, pd.DataFrame) or isinstance(y, pd.Series): y = np.asarray(y)
        self.verbose_ = verbose
        self.le_ = sklearn.preprocessing.LabelEncoder().fit(y)
        self.X_, self.y_, self.classes_ = X, self.le_.transform(y), self.le_.classes_
        splits = list(sklearn.model_selection.StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X, y))
        with mp.Pool(processes=min(len(splits), mp.cpu_count()), initializer=_worker_init) as pool:
            self.models_ = pool.map(self._fit_model, splits)
        oof_preds = np.concatenate([m.predict_proba(X.take(idxs[1], 0)) for m, idxs in zip(self.models_, splits)],
                                   axis=0)
        oof_labels = np.concatenate([y[idxs[1]] for idxs in splits], axis=0)
        self.calib_ = get_calibrator('logistic', calibrate_with_mixture=True,
                                     logistic_binary_type='quadratic').fit(oof_preds, oof_labels)
        return self

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.calib_.predict_proba(np.mean([m.predict_proba(X) for m in self.models_], axis=0))

    def predict(self, X):
        if isinstance(X, torch.Tensor) or isinstance(X, pd.DataFrame) or isinstance(X, pd.Series): X = np.asarray(X)
        return self.le_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

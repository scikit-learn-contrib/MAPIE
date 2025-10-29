import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

sklearn.set_config(enable_metadata_routing=True)
np.seterr(divide="ignore", invalid="ignore")

"""
Private module containing core Venn-ABERS implementation classes.

This module contains the internal implementation details for Venn-ABERS
calibration. Users should use VennAbersCalibrator from mapie.calibration instead.
"""


def _geo_mean(a):
    """Geometric mean calculation for Venn-ABERS."""
    return a.prod(axis=1) ** (1.0 / a.shape[1])


def calc_p0p1(p_cal, y_cal, precision=None):
    """
    Function that calculates isotonic calibration vectors
    required for Venn-ABERS calibration

    This function relies on the geometric representation of isotonic
    regression as the slope of the GCM (greatest convex minorant) of the CSD
    (cumulative sum diagram) as decribed in [1] pages 9–13 (especially Theorem 1.1).
    In particular, the function implements
    algorithms 1-4 as described in Chapter 2 in [2]


    References
    ----------
    [1] Richard E. Barlow, D. J. Bartholomew, J. M. Bremner, and H. Daniel
    Brunk. Statistical Inference under Order Restrictions: The Theory and
    Application of Isotonic Regression. Wiley, London, 1972.

    [2] Vovk, Vladimir, Ivan Petej, and Valentina Fedorova.
    "Large-scale probabilistic predictors with and without guarantees of validity."
    Advances in Neural Information Processing Systems 28 (2015).
    (arxiv version https://arxiv.org/pdf/1511.00213.pdf)


    Parameters
    ----------
    p_cal : {array-like}, shape (n_samples, 2)
    Input data for calibration consisting of calibration set probabilities

    y_cal : {array-like}, shape (n_samples,)
    Associated binary class labels.

    precision: int, default = None
    Optional number of decimal points to which
    Venn-Abers calibration probabilities p_cal are rounded to.
    Yields significantly faster computation time for larger calibration datasets.
    If None no rounding is applied.


    Returns
    ----------
    p_0 : {array-like}, shape (n_samples, )
        Precomputed vector storing values of the isotonic regression
        fitted to a sequence that contains binary class label 0

    p_1 : {array-like}, shape (n_samples, )
        Precomputed vector storing values of the isotonic regression
        fitted to a sequence
        that contains binary class label 1

    c : {array-like}, shape (n_samples, )
        Ordered set of unique calibration probabilities
    """
    if precision is not None:
        cal = np.hstack(
            (np.round(p_cal[:, 1], precision).reshape(-1, 1), y_cal.reshape(-1, 1))
        )
    else:
        cal = np.hstack((p_cal[:, 1].reshape(-1, 1), y_cal.reshape(-1, 1)))
    ix = np.argsort(cal[:, 0])
    k_sort = cal[ix, 0]
    k_label_sort = cal[ix, 1]

    c = np.unique(k_sort)
    ia = np.searchsorted(k_sort, c)

    w = np.zeros(len(c))

    w[:-1] = np.diff(ia)
    w[-1] = len(k_sort) - ia[-1]

    k_dash = len(c)
    P = np.zeros((k_dash + 2, 2))

    P[0, :] = -1

    P[2:, 0] = np.cumsum(w)
    P[2:-1, 1] = np.cumsum(k_label_sort)[(ia - 1)[1:]]
    P[-1, 1] = np.cumsum(k_label_sort)[-1]

    p1 = np.zeros((len(c) + 1, 2))
    p1[1:, 0] = c

    P1 = P[1:] + 1

    for i in range(len(p1)):
        P1[i, :] = P1[i, :] - 1

        if i == 0:
            grads = np.divide(P1[:, 1], P1[:, 0])
            grad = np.nanmin(grads)
            p1[i, 1] = grad
            c_point = 0
        else:
            imp_point = P1[c_point, 1] + (P1[i, 0] - P1[c_point, 0]) * grad

            if P1[i, 1] < imp_point:
                grads = np.divide((P1[i:, 1] - P1[i, 1]), (P1[i:, 0] - P1[i, 0]))
                if np.sum(np.isnan(np.nanmin(grads))) == 0:
                    grad = np.nanmin(grads)
                c_point = i
                p1[i, 1] = grad
            else:
                p1[i, 1] = grad

    p0 = np.zeros((len(c) + 1, 2))
    p0[1:, 0] = c

    P0 = P[1:]

    for i in range(len(p1) - 1, -1, -1):
        P0[i, 0] = P0[i, 0] + 1

        if i == len(p1) - 1:
            grads = np.divide((P0[:, 1] - P0[i, 1]), (P0[:, 0] - P0[i, 0]))
            grad = np.nanmax(grads)
            p0[i, 1] = grad
            c_point = i
        else:
            imp_point = P0[c_point, 1] + (P0[i, 0] - P0[c_point, 0]) * grad

            if P0[i, 1] < imp_point:
                grads = np.divide((P0[:, 1] - P0[i, 1]), (P0[:, 0] - P0[i, 0]))
                grads[i:] = 0
                grad = np.nanmax(grads)
                c_point = i
                p0[i, 1] = grad
            else:
                p0[i, 1] = grad
    return p0, p1, c


def calc_probs(p0, p1, c, p_test):
    """
    Function that calculates Venn-Abers multiprobability outputs and
    associated calibrated probabilities

    In particular, the function implements algorithms 5-6
    as described in Chapter 2 in [1]

    References
    ----------
    [1] Vovk, Vladimir, Ivan Petej, and Valentina Fedorova.
    "Large-scale probabilistic predictors with and without guarantees of validity."
    Advances in Neural Information Processing Systems 28 (2015).
    (arxiv version https://arxiv.org/pdf/1511.00213.pdf)


    Parameters
    ----------
    p0 : {array-like}, shape (n_samples, )
        Precomputed vector storing values of the isotonic regression
        fitted to a sequence that contains binary class label 0

    p1 : {array-like}, shape (n_samples, )
        Precomputed vector storing values of the isotonic regression
        fitted to a sequence that contains binary class label 1

    c : {array-like}, shape (n_samples, )
        Ordered set of unique calibration probabilities

    p_test : {array-like}, shape (n_samples, 2)
        An array of probability outputs which are to be calibrated


    Returns
    ----------
    p_prime : {array-like}, shape (n_samples, 2)
    Calibrated probability outputs

    p0_p1 : {array-like}, shape (n_samples, 2)
    Associated multiprobability outputs
    (as described in Section 4 in https://arxiv.org/pdf/1511.00213.pdf)
    """
    out = p_test[:, 1]
    p0_p1 = np.hstack(
        (
            p0[np.searchsorted(c, out, "right"), 1].reshape(-1, 1),
            p1[np.searchsorted(c, out, "left"), 1].reshape(-1, 1),
        )
    )

    p_prime = np.zeros((len(out), 2))
    p_prime[:, 1] = p0_p1[:, 1] / (1 - p0_p1[:, 0] + p0_p1[:, 1])
    p_prime[:, 0] = 1 - p_prime[:, 1]

    return p_prime, p0_p1


def predict_proba_prefitted_va(
    p_cal, y_cal, p_test, precision=None, va_tpe="one_vs_one"
):
    """
    Generate Venn-ABERS calibrated probabilities
    for multiclass problems using pre-fitted calibration data.

    This function performs Venn-ABERS calibration on multiclass problems
    by decomposing them into binary classification problems
    using either one-vs-one or one-vs-all strategies.
    It uses pre-computed calibration probabilities and
    labels to calibrate test probabilities.

    Parameters
    ----------
    p_cal : array-like of shape (n_cal_samples, n_classes)
        Calibration set probabilities for each class.
        These are the predicted probabilities
        from the base classifier on the calibration set.

    y_cal : array-like of shape (n_cal_samples,)
        True class labels for the calibration set.
        Should contain integer class labels.

    p_test : array-like of shape (n_test_samples, n_classes)
        Test set probabilities for each class that need to be calibrated.
        These are the predicted probabilities from the base classifier
        on the test set.

    precision : int, optional, default=None
        Number of decimal places to round calibration probabilities
        to for faster computation.
        If None, no rounding is applied.
        Useful for large calibration datasets to improve
        computational efficiency.

    va_tpe : {'one_vs_one', 'one_vs_all'}, default='one_vs_one'
        Strategy for decomposing multiclass problem into binary problems:
        - 'one_vs_one': Creates binary classifiers for each pair of classes
        - 'one_vs_all': Creates binary classifiers for each class vs all others

    Returns
    -------
    p_prime : ndarray of shape (n_test_samples, n_classes)
        Venn-ABERS calibrated probabilities for each class.
        Probabilities are normalized
        to sum to 1 across classes for each sample.

    multiclass_p0p1 : list of ndarray
        List containing the multiprobability outputs (p0, p1)
        for each binary problem.
        The structure depends on the decomposition strategy:
        - For 'one_vs_one': List of length C(n_classes, 2)
          containing arrays of shape (n_test_samples, 2) for each class pair
        - For 'one_vs_all': List of length n_classes containing arrays of shape
          (n_test_samples, 2) for each class vs rest problem

    Notes
    -----
    This function implements the Venn-ABERS calibration method as described in [1]_.
    The multiclass extension uses the approach described in [2]_ for combining
    binary calibrators.

    For 'one_vs_one' strategy, the final probabilities are computed using the
    pairwise coupling method where each class probability is the harmonic mean
    of its pairwise probabilities.

    References
    ----------
    .. [1] Vovk, Vladimir, Ivan Petej, and Valentina Fedorova. "Large-scale
           probabilistic predictors with and without guarantees of validity."
           Advances in Neural Information Processing Systems 28 (2015).

    .. [2] Hastie, Trevor, and Robert Tibshirani. "Classification by pairwise
           coupling." Advances in Neural Information Processing Systems 10 (1997).

    Examples
    --------
    >>> import numpy as np
    >>> # Calibration data
    >>> p_cal = np.array([[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.1, 0.1, 0.8]])
    >>> y_cal = np.array([0, 1, 2])
    >>> # Test data
    >>> p_test = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
    >>>
    >>> p_calibrated, p0p1 = predict_proba_prefitted_va(p_cal, y_cal, p_test)
    >>> print(p_calibrated.shape)
    (2, 3)
    """
    # Validate va_tpe parameter
    if va_tpe not in ["one_vs_one", "one_vs_all"]:
        raise ValueError(
            f"Invalid va_tpe '{va_tpe}'. "
            f"Allowed values are ['one_vs_one', 'one_vs_all']."
        )

    p_prime = None
    multiclass_p0p1 = None

    if va_tpe == "one_vs_one":
        classes = np.unique(y_cal)
        class_pairs = []
        for i in range(len(classes) - 1):
            for j in range(i + 1, len(classes)):
                class_pairs.append([classes[i], classes[j]])

        multiclass_probs = []
        multiclass_p0p1 = []
        for i, class_pair in enumerate(class_pairs):
            pairwise_indices = (y_cal == class_pair[0]) + (y_cal == class_pair[1])
            binary_cal_probs = p_cal[:, class_pair][pairwise_indices] / np.sum(
                p_cal[:, class_pair][pairwise_indices], axis=1
            ).reshape(-1, 1)
            binary_test_probs = p_test[:, class_pair] / np.sum(
                p_test[:, class_pair], axis=1
            ).reshape(-1, 1)
            binary_classes = y_cal[pairwise_indices] == class_pair[1]

            va = VennAbers()
            va.fit(binary_cal_probs, binary_classes, precision=precision)
            p_pr, p0_p1 = va.predict_proba(binary_test_probs)
            multiclass_probs.append(p_pr)
            multiclass_p0p1.append(p0_p1)

        p_prime = np.zeros((len(p_test), len(classes)))

        for (
            i,
            cl_id,
        ) in enumerate(classes):
            stack_i = [
                p[:, 0].reshape(-1, 1)
                for i, p in enumerate(multiclass_probs)
                if class_pairs[i][0] == cl_id
            ]
            stack_j = [
                p[:, 1].reshape(-1, 1)
                for i, p in enumerate(multiclass_probs)
                if class_pairs[i][1] == cl_id
            ]
            p_stack = stack_i + stack_j

            p_prime[:, i] = 1 / (
                np.sum(np.hstack([(1 / p) for p in p_stack]), axis=1)
                - (len(classes) - 2)
            )

    else:
        classes = np.unique(y_cal)

        multiclass_probs = []
        multiclass_p0p1 = []
        for _, class_id in enumerate(classes):
            class_indices = y_cal == class_id
            binary_cal_probs = np.zeros((len(p_cal), 2))
            binary_test_probs = np.zeros((len(p_test), 2))
            binary_cal_probs[:, 1] = p_cal[:, class_id]
            binary_cal_probs[:, 0] = 1 - binary_cal_probs[:, 1]
            binary_test_probs[:, 1] = p_test[:, class_id]
            binary_test_probs[:, 0] = 1 - binary_test_probs[:, 1]
            binary_classes = class_indices

            va = VennAbers()
            va.fit(binary_cal_probs, binary_classes, precision=precision)
            p_pr, p0_p1 = va.predict_proba(binary_test_probs)
            multiclass_probs.append(p_pr)
            multiclass_p0p1.append(p0_p1)

        p_prime = np.zeros((len(p_test), len(classes)))

        for i, _ in enumerate(classes):
            p_prime[:, i] = multiclass_probs[i][:, 1]

    p_prime = p_prime / np.sum(p_prime, axis=1).reshape(-1, 1)

    return p_prime, multiclass_p0p1


class VennAbers:
    """
    Implementation of the Venn-ABERS calibration for binary classification problems.
    Venn-ABERS calibration is a method of turning machine learning
    classification algorithms into probabilistic predictors that
    automatically enjoys a property of validity (perfect calibration) and
    is computationally efficient.
    The algorithm is described in [1].


    References
    ----------
    [1] Vovk, Vladimir, Ivan Petej, and Valentina Fedorova.
    "Large-scale probabilistic predictors with and without guarantees of validity."
    Advances in Neural Information Processing Systems 28 (2015).
    (arxiv version https://arxiv.org/pdf/1511.00213.pdf)

    .. versionadded:: 1.0


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie._venn_abers import VennAbers
    >>>
    >>> # Generate data and split into train/test
    >>> X, y = make_classification(n_samples=1000, n_classes=2, n_informative=10)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> # Further split training data into proper training and calibration sets
    >>> X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
    ...     X_train, y_train, test_size=0.2, shuffle=False
    ... )
    >>>
    >>> # Train classifier on proper training set
    >>> clf = GaussianNB()
    >>> _ = clf.fit(X_train_proper, y_train_proper)
    >>>
    >>> # Get probability predictions for calibration and test sets
    >>> p_cal = clf.predict_proba(X_cal)
    >>> p_test = clf.predict_proba(X_test)
    >>>
    >>> # Apply Venn-ABERS calibration
    >>> va = VennAbers()
    >>> va.fit(p_cal, y_cal)
    >>> p_prime, p0_p1 = va.predict_proba(p_test)
    >>>
    >>> # p_prime contains calibrated probabilities
    >>> print(p_prime.shape)
    (250, 2)
    """

    def __init__(self):
        self.p0 = None
        self.p1 = None
        self.c = None

    def fit(self, p_cal, y_cal, precision=None):
        """Fits the VennAbers calibrator to the calibration dataset

        Parameters
        ----------
        p_cal : {array-like}, shape (n_samples, 2)
            Input data for calibration consisting of calibration set probabilities

        y_cal : {array-like}, shape (n_samples,)
            Associated binary class labels.

        precision: int, default = None
            Optional number of decimal points to which Venn-Abers calibration
            probabilities p_cal are rounded to.
            Yields significantly faster computation time for larger calibration datasets
        """
        self.p0, self.p1, self.c = calc_p0p1(p_cal, y_cal, precision)

    def predict_proba(self, p_test):
        """Generates Venn-Abers probability estimates


        Parameters
        ----------
        p_test : {array-like}, shape (n_samples, 2)
            An array of probability outputs which are to be calibrated


        Returns
        ----------
        p_prime : {array-like}, shape (n_samples, 2)
            Calibrated probability outputs

        p0_p1 : {array-like}, shape (n_samples, 2)
            Associated multiprobability outputs
            (as described in Section 4 in https://arxiv.org/pdf/1511.00213.pdf)
        """
        p_prime, p0_p1 = calc_probs(self.p0, self.p1, self.c, p_test)
        return p_prime, p0_p1


class VennAbersCV:
    """
    Inductive (IVAP) or Cross (CVAP) Venn-ABERS prediction method
    for binary classification problems

    Implements the Inductive or Cross Venn-Abers calibration method
    as described in Sections 2-4 in [1]

    References
    ----------
    [1] Vovk, Vladimir, Ivan Petej, and Valentina Fedorova.
    "Large-scale probabilistic predictors with and without guarantees of validity."
    Advances in Neural Information Processing Systems 28 (2015).
    (arxiv version https://arxiv.org/pdf/1511.00213.pdf)

    Parameters
    ----------

    estimator : sci-kit learn estimator instance, default=None
        The classifier whose output need to be calibrated to provide more
        accurate `predict_proba` outputs.

    inductive : bool
        True to run the Inductive (IVAP) or False for Cross (CVAP)
        Venn-ABERS calibration

    n_splits: int, default=5
        For CVAP only, number of folds. Must be at least 2.
        Uses sklearn.model_selection.StratifiedKFold functionality
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

    cal_size : float or int, default=None
        For IVAP only, uses sklearn.model_selection.train_test_split functionality
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the proper training / calibration split.
        If int, represents the absolute number of test samples. If None, the
        value is set to the complement of the train size. If ``train_size``
        is also None, it will be set to 0.25.

    train_proper_size : float or int, default=None
        For IVAP only, if float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the proper training set split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting. For IVAP if shuffle=False
        then stratify must be None. For CVAP whether to shuffle each class's samples
        before splitting into batches

    stratify : array-like, default=None
        For IVAP only. If not None, data is split in a stratified fashion, using this as
        the class labels.

    precision: int, default = None
        Optional number of decimal points to which Venn-Abers calibration
        probabilities p_cal are rounded to.
        Yields significantly faster computation time for larger calibration datasets
    """

    def __init__(
        self,
        estimator,
        inductive,
        n_splits=None,
        cal_size=None,
        train_proper_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
        precision=None,
    ):
        self.estimator = estimator
        self.n_splits = n_splits
        self.clf_p_cal = []
        self.clf_y_cal = []
        self.inductive = inductive
        self.cal_size = cal_size
        self.train_proper_size = train_proper_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.precision = precision

    def fit(self, _x_train, _y_train, sample_weight=None):
        """Fits the IVAP or CVAP calibrator to the training set.

        Parameters
        ----------
        _x_train : {array-like}, shape (n_samples,)
            Input data for calibration consisting of training set numerical features

        _y_train : {array-like}, shape (n_samples,)
            Associated binary class labels.

        sample_weight : {array-like}, shape (n_samples,), optional
            Sample weights for fitting the estimators.
            If None, then samples are equally weighted.
        """
        if self.inductive:
            self.n_splits = 1

            # Split sample_weight along with data if provided
            if sample_weight is not None:
                x_train_proper, x_cal, y_train_proper, y_cal, sw_train, sw_cal = (
                    train_test_split(
                        _x_train,
                        _y_train,
                        sample_weight,
                        test_size=self.cal_size,
                        train_size=self.train_proper_size,
                        random_state=self.random_state,
                        shuffle=self.shuffle,
                        stratify=self.stratify,
                    )
                )
            else:
                x_train_proper, x_cal, y_train_proper, y_cal = train_test_split(
                    _x_train,
                    _y_train,
                    test_size=self.cal_size,
                    train_size=self.train_proper_size,
                    random_state=self.random_state,
                    shuffle=self.shuffle,
                    stratify=self.stratify,
                )
                sw_train = None

            # Fit estimator with sample weights if provided
            if sw_train is not None:
                self.estimator.fit(
                    x_train_proper, y_train_proper.flatten(), sample_weight=sw_train
                )
            else:
                self.estimator.fit(x_train_proper, y_train_proper.flatten())

            clf_prob = self.estimator.predict_proba(x_cal)
            self.clf_p_cal.append(clf_prob)
            self.clf_y_cal.append(y_cal)
        else:
            kf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
            for train_index, test_index in kf.split(_x_train, _y_train):
                # Extract sample weights for this fold if provided
                fold_sample_weight = None
                if sample_weight is not None:
                    fold_sample_weight = sample_weight[train_index]

                # Fit estimator with sample weights if provided
                if fold_sample_weight is not None:
                    self.estimator.fit(
                        _x_train[train_index],
                        _y_train[train_index].flatten(),
                        sample_weight=fold_sample_weight,
                    )
                else:
                    self.estimator.fit(
                        _x_train[train_index], _y_train[train_index].flatten()
                    )

                clf_prob = self.estimator.predict_proba(_x_train[test_index])
                self.clf_p_cal.append(clf_prob)
                self.clf_y_cal.append(_y_train[test_index])

    def predict_proba(self, _x_test, loss="log", p0_p1_output=False):
        """Generates Venn-ABERS calibrated probabilities.


        Parameters
        ----------
        _x_test : {array-like}, shape (n_samples,)
            Training set numerical features

        loss : str, default='log'
            Log or Brier loss. For further details of calculation
            see Section 4 in https://arxiv.org/pdf/1511.00213.pdf

        p0_p1_output: bool, default = False
            If True, function also returns p0_p1 binary probabilistic outputs

        Returns
        ----------
        p_prime: {array-like}, shape (n_samples,n_classes)
            Venn-ABERS calibrated probabilities

        p0_p1: {array-like}, default  = None
            Venn-ABERS calibrated p0 and p1 outputs (if p0_p1_output = True)
        """

        p0p1_test = []
        clf_prob_test = self.estimator.predict_proba(_x_test)
        for i in range(self.n_splits):
            va = VennAbers()
            va.fit(
                p_cal=self.clf_p_cal[i],
                y_cal=self.clf_y_cal[i],
                precision=self.precision,
            )
            _, probs = va.predict_proba(p_test=clf_prob_test)
            p0p1_test.append(probs)
        p0_stack = np.hstack([prob[:, 0].reshape(-1, 1) for prob in p0p1_test])
        p1_stack = np.hstack([prob[:, 1].reshape(-1, 1) for prob in p0p1_test])

        p_prime = np.zeros((len(_x_test), 2))

        if loss == "log":
            p_prime[:, 1] = _geo_mean(p1_stack) / (
                _geo_mean(1 - p0_stack) + _geo_mean(p1_stack)
            )
            p_prime[:, 0] = 1 - p_prime[:, 1]
        else:
            p_prime[:, 1] = (
                1
                / self.n_splits
                * (
                    np.sum(p1_stack, axis=1)
                    + 0.5 * np.sum(p0_stack**2, axis=1)
                    - 0.5 * np.sum(p1_stack**2, axis=1)
                )
            )
            p_prime[:, 0] = 1 - p_prime[:, 1]

        if p0_p1_output:
            p0_p1 = np.hstack((p0_stack, p1_stack))
            return p_prime, p0_p1
        else:
            return p_prime


class VennAbersMultiClass:
    """
    Inductive (IVAP) or Cross (CVAP) Venn-ABERS prediction method
    for multi-class classification problems

    Implements the Inductive or Cross Venn-Abers calibration method
    as described in [1]

    References
    ----------
    [1] Manokhin, Valery. "Multi-class probabilistic classification using
    inductive and cross Venn–Abers predictors." In Conformal and Probabilistic
    Prediction and Applications, pp. 228-240. PMLR, 2017.

    Parameters
    __________

    estimator : sci-kit learn estimator instance
        The classifier whose output need to be calibrated to provide more
        accurate `predict_proba` outputs.

    inductive : bool
        True to run the Inductive (IVAP) or False for Cross (CVAP)
        Venn-ABERS calibration

    n_splits: int, default=5
        For CVAP only, number of folds. Must be at least 2.
        Uses sklearn.model_selection.StratifiedKFold functionality
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

    cal_size : float or int, default=None
        For IVAP only, uses sklearn.model_selection.train_test_split functionality
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the proper training / calibration split.
        If int, represents the absolute number of test samples. If None, the
        value is set to the complement of the train size. If ``train_size``
        is also None, it will be set to 0.25.

    train_size : float or int, default=None
        For IVAP only, if float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the proper training set split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting. For IVAP if shuffle=False
        then stratify must be None.
        For CVAP whether to shuffle each class's samples
        before splitting into batches

    stratify : array-like, default=None
        For IVAP only. If not None, data is split in a stratified fashion,
        using this as the class labels.

    precision: int, default = None
        Optional number of decimal points to which Venn-Abers calibration
        probabilities p_cal are rounded to.
        Yields significantly faster computation time for larger calibration datasets
    """

    def __init__(
        self,
        estimator,
        inductive,
        n_splits=None,
        cal_size=None,
        train_proper_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
        precision=None,
    ):
        self.estimator = estimator
        self.inductive = inductive
        self.n_splits = n_splits
        self.cal_size = cal_size
        self.train_proper_size = train_proper_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.multi_class_model = []
        self.n_classes = None
        self.classes = None
        self.pairwise_id = []
        self.clf_ovo = None
        self.multiclass_cal = []
        self.multiclass_va_estimators = []
        self.multiclass_probs = []
        self.multiclass_p0p1 = []
        self.precision = precision

    def fit(self, _x_train, _y_train, sample_weight=None):
        """
        Fits the Venn-ABERS calibrator to the training set

        Parameters
        ----------
        _x_train : {array-like}, shape (n_samples,)
            Input data for calibration consisting of training set numerical features

        _y_train : {array-like}, shape (n_samples,)
            Associated binary class labels.

        sample_weight : {array-like}, shape (n_samples,), optional
            Sample weights for fitting the estimators.
            If None, then samples are equally weighted.
        """

        # integrity checks
        if not self.inductive and self.n_splits is None:
            raise Exception("For Cross Venn ABERS please provide n_splits")
        try:
            check_is_fitted(self.estimator)
        except NotFittedError:
            if (self.inductive and self.cal_size is None) and (
                self.train_proper_size is None
            ):
                raise Exception(
                    "For Inductive Venn-ABERS please provide either calibration"
                    "or proper train set size"
                )

        self.classes = np.unique(_y_train)
        self.n_classes = len(self.classes)

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                self.pairwise_id.append([self.classes[i], self.classes[j]])

        # Fit the OneVsOne classifier with sample weights if provided
        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        # Clone the estimator to avoid modifying the original
        # estimator_clone = clone(self.estimator)

        # OneVsOneClassifier will handle the estimator's preprocessing
        # (e.g., if it's a pipeline, it will apply transformations internally)
        self.clf_ovo = OneVsOneClassifier(self.estimator)
        self.clf_ovo.fit(_x_train, _y_train, **fit_params)

        for pair_id, clf_ovo_estimator in enumerate(self.clf_ovo.estimators_):
            _pairwise_indices = (_y_train == self.pairwise_id[pair_id][0]) + (
                _y_train == self.pairwise_id[pair_id][1]
            )

            # Extract sample weights for this pair if provided
            pair_sample_weight = None
            if sample_weight is not None:
                pair_sample_weight = sample_weight[_pairwise_indices]

            # pair_estimator = clone(self.estimator)

            va_cv = VennAbersCV(
                self.estimator,
                inductive=self.inductive,
                n_splits=self.n_splits,
                cal_size=self.cal_size,
                train_proper_size=self.train_proper_size,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=self.stratify,
                precision=self.precision,
            )
            va_cv.fit(
                _x_train[_pairwise_indices],
                np.array(
                    _y_train[_pairwise_indices] == self.pairwise_id[pair_id][1]
                ).reshape(-1, 1),
                sample_weight=pair_sample_weight,
            )
            self.multiclass_va_estimators.append(va_cv)

    def predict_proba(self, _x_test, loss="log", p0_p1_output=False):
        """
        Generates Venn-ABERS calibrated probabilities.

        Parameters
        ----------
        _x_test : {array-like}, shape (n_samples,)
            Training set numerical features

        loss : str, default='log'
            Log or Brier loss. For further details of calculation
            see Section 4 in https://arxiv.org/pdf/1511.00213.pdf

        p0_p1_output: bool, default = False
        If True, function also returns a set p0_p1 binary probabilistic outputs
        for each fold

        Returns
        ----------
        p_prime: {array-like}, shape (n_samples,n_classes)
            Venn-ABERS calibrated probabilities

        p0_p1: {array-like}, default  = None
        Venn-ABERS calibrated p0 and p1 outputs (if p0_p1_output = True)
        """

        self.multiclass_probs = []
        self.multiclass_p0p1 = []

        if p0_p1_output:
            for i, va_estimator in enumerate(self.multiclass_va_estimators):
                _p_prime, _p0_p1 = va_estimator.predict_proba(
                    _x_test, loss=loss, p0_p1_output=True
                )
                self.multiclass_probs.append(_p_prime)
                self.multiclass_p0p1.append(_p0_p1)
        else:
            for i, va_estimator in enumerate(self.multiclass_va_estimators):
                _p_prime = va_estimator.predict_proba(_x_test, loss=loss)
                self.multiclass_probs.append(_p_prime)

        p_prime = np.zeros((len(_x_test), self.n_classes))

        for (
            i,
            cl_id,
        ) in enumerate(self.classes):
            stack_i = [
                p[:, 0].reshape(-1, 1)
                for i, p in enumerate(self.multiclass_probs)
                if self.pairwise_id[i][0] == cl_id
            ]
            stack_j = [
                p[:, 1].reshape(-1, 1)
                for i, p in enumerate(self.multiclass_probs)
                if self.pairwise_id[i][1] == cl_id
            ]
            p_stack = stack_i + stack_j

            p_prime[:, i] = 1 / (
                np.sum(np.hstack([(1 / p) for p in p_stack]), axis=1)
                - (self.n_classes - 2)
            )

        p_prime = p_prime / np.sum(p_prime, axis=1).reshape(-1, 1)

        if p0_p1_output:
            return p_prime, self.multiclass_p0p1
        else:
            return p_prime

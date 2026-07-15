from __future__ import annotations

from functools import lru_cache, partial
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels

from mapie.classification import SplitConformalClassifier
from mapie.conformity_scores import BaseClassificationScore, BaseRegressionScore
from mapie.conformity_scores.sets.raps import RAPSConformityScore
from mapie.conformity_scores.sets.topk import TopKConformityScore
from mapie.regression import SplitConformalRegressor
from mapie.utils import (
    _prepare_params,
    _raise_error_if_previous_method_not_called,
    check_proba_normalized,
)

FUNCTION_DEFAULTS = {"kernel": None, "gamma": 1, "lambda": 1}


def _import_cvxpy():
    """Import cvxpy lazily, raising a helpful error if it is not installed.

    cvxpy is an optional dependency of MAPIE (the ``conditional`` extra), so it
    is imported only when the conditional conformal procedure is actually used.
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "cvxpy is required for ConditionalSplitConformalRegressor. "
            "Install it with: pip install mapie[conditional]"
        ) from e
    return cp


class _ConditionalConformalMixin:
    """
    Shared machinery for the conditional conformal procedure of Gibbs et al.

    Computes, for a single test point, the conditionally valid
    conformity-score cutoff S^* such that the prediction set is
    ``{y : S(x, y) <= S^*}``. Task-specific subclasses only differ in how this
    cutoff is inverted into a prediction interval (regression) or a prediction
    set (classification).
    """

    def _init_conditional(
        self,
        feature_map: Callable,
        randomize: bool,
        exact: bool,
        infinite_params: Optional[dict],
        seed: int,
    ) -> None:
        self.feature_map = feature_map
        self.randomize = randomize
        self.exact = exact
        self.infinite_params = {} if infinite_params is None else infinite_params
        self.rng = np.random.default_rng(seed=seed)

    def _conformalize_conditional(
        self, x_calib: NDArray, scores_calib: NDArray
    ) -> None:
        """
        Set up the final fitting problem for the given conformalization set:
        reduce the basis to full rank if needed and build the cvxpy problem
        used for the conditional procedure.
        """
        self.x_calib = x_calib
        self.scores_calib = np.asarray(scores_calib).ravel()
        phi_calib = self.feature_map(x_calib)

        _, s, Vt = np.linalg.svd(phi_calib, full_matrices=False)

        # Set a tolerance to decide which singular values are nonzero
        tol = 1e-10
        r = int(np.sum(s > tol))

        if r < len(s):
            self.feature_map_orig = self.feature_map
            T = Vt.T[:, :r]
            self.feature_map = lambda x: (self.feature_map_orig(x) @ T)
            phi_calib = self.feature_map(x_calib)

        self.phi_calib = phi_calib

        self.cvx_problem = setup_cvx_problem(
            self.x_calib, self.scores_calib, self.phi_calib, self.infinite_params
        )

    @lru_cache()
    def _get_calibration_solution(self, quantile: float):
        S = self.scores_calib.reshape(-1, 1)
        Phi = self.phi_calib.astype(float)
        zeros = np.zeros((Phi.shape[1],))

        bounds = np.asarray([quantile - 1, quantile])
        bounds = np.tile(bounds.reshape(1, -1), (len(S), 1))

        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, method="highs")
        primal_vars = -1 * res.eqlin.marginals.reshape(-1, 1)
        dual_vars = res.x.reshape(-1, 1)

        residuals = S - (Phi @ primal_vars)
        interpolated_pts = np.isclose(residuals, 0)

        # if I didn't converge to a solution that interpolates at least Phi.shape[1] pts,
        # I need to manually find one via a modified simplex iteration
        if interpolated_pts.sum() < Phi.shape[1]:  # pragma: no cover
            num_to_add = Phi.shape[1] - interpolated_pts.sum()
            for _ in range(num_to_add):
                candidate_pts = interpolated_pts.copy().flatten()

                # find candidate idx for interpolation, e.g., new covariate that is
                # linearly independent of the previously interpolated points
                Q, _ = np.linalg.qr(Phi[candidate_pts].T)
                projections = Phi @ Q @ Q.T
                norms = np.linalg.norm(Phi - projections, axis=1)
                candidate_idx = np.where(norms > 1e-5)[0][0]
                candidate_pts[candidate_idx] = True

                # find direction to solution that would interpolate the new point
                gamma, _, _, _ = np.linalg.lstsq(
                    Phi[candidate_pts], S[candidate_pts], rcond=None
                )
                direction = gamma.reshape(-1, 1) - primal_vars
                step_sizes = residuals / (Phi @ direction)

                # check the non-basic indices for which a step in this direction could have led to interpolation
                # e.g., those for which the step size is positive and the point is not already interpolated
                positive_indices = np.where((step_sizes > 0) & ~interpolated_pts)[0]

                # take smallest possible step that would lead to interpolation
                primal_vars += np.min(step_sizes[positive_indices]) * direction

                residuals = S - (Phi @ primal_vars)
                interpolated_pts = np.isclose(residuals, 0)

        return dual_vars, primal_vars

    def _compute_exact_cutoff(
        self, quantiles, primals, duals, phi_test, dual_threshold
    ):
        def get_current_basis(primals, duals, Phi, S, quantiles):
            interp_bools = np.logical_and(
                ~np.isclose(duals, quantiles - 1), ~np.isclose(duals, quantiles)
            )
            if np.sum(interp_bools) == Phi.shape[1]:
                return interp_bools
            else:  # pragma: no cover
                # Fallback for degenerate dual solutions: not reached on
                # well-posed problems with a full-rank basis.
                preds = (Phi @ primals).flatten()
                active_indices = np.where(interp_bools)[0]
                interp_indices = np.where(np.isclose(np.abs(S - preds), 0))[0]
                diff_indices = np.setdiff1d(interp_indices, active_indices)
                num_missing = Phi.shape[1] - np.sum(interp_bools)
                if num_missing < len(diff_indices):
                    from itertools import combinations

                    for cand_indices in combinations(diff_indices, num_missing):
                        cand_phi = Phi[np.concatenate((active_indices, cand_indices))]
                        if np.isfinite(np.linalg.cond(cand_phi)):
                            interp_bools[np.asarray(cand_indices)] = True
                            break
                else:
                    interp_bools[diff_indices] = True
                if np.sum(interp_bools) != Phi.shape[1]:
                    raise ValueError(
                        "Initial basis could not be found - retry with exact=False."
                    )
                return interp_bools

        if np.allclose(phi_test, 0):
            return np.inf if quantiles[-1] >= 0.5 else -np.inf

        basis = get_current_basis(
            primals, duals, self.phi_calib, self.scores_calib, quantiles[:-1]
        )
        S_test = phi_test @ primals

        duals = np.concatenate((duals.flatten(), [0]))
        basis = np.concatenate((basis.flatten(), [False]))
        phi = np.concatenate((self.phi_calib, phi_test.reshape(1, -1)), axis=0)
        S = np.concatenate(
            (self.scores_calib.reshape(-1, 1), S_test.reshape(-1, 1)), axis=0
        )

        candidate_idx = phi.shape[0] - 1
        num_iters = 0
        while True:
            # get direction vector for dual variable step
            direction = (
                -1
                * np.linalg.solve(
                    phi[basis].T, phi[candidate_idx].reshape(-1, 1)
                ).flatten()
            )

            # only consider non-zero entries of the direction vector
            active_indices = ~np.isclose(direction, 0)
            active_direction = direction[active_indices]
            active_basis = basis.copy()
            active_basis[np.where(basis)[0][~active_indices]] = False

            positive_step = True if duals[candidate_idx] <= 0 else False
            if candidate_idx == phi.shape[0] - 1:
                positive_step = True if dual_threshold >= 0 else False

            if positive_step:
                gap_to_bounds = np.maximum(
                    (quantiles[active_basis].flatten() - duals[active_basis])
                    / active_direction,
                    ((quantiles[active_basis].flatten() - 1) - duals[active_basis])
                    / active_direction,
                )
                step_size = np.min(gap_to_bounds)
                departing_idx = np.where(active_basis)[0][np.argmin(gap_to_bounds)]
            else:
                gap_to_bounds = np.minimum(
                    (quantiles[active_basis].flatten() - duals[active_basis])
                    / active_direction,
                    ((quantiles[active_basis].flatten() - 1) - duals[active_basis])
                    / active_direction,
                )
                step_size = np.max(gap_to_bounds)
                departing_idx = np.where(active_basis)[0][np.argmax(gap_to_bounds)]
            step_size_clip = np.clip(
                step_size,
                a_max=quantiles[candidate_idx, 0] - duals[candidate_idx],
                a_min=(quantiles[candidate_idx, 0] - 1) - duals[candidate_idx],
            )

            duals[basis] += step_size_clip * direction
            duals[candidate_idx] += step_size_clip
            # print("Current value of final dual", duals[-1], "target threshold", dual_threshold)

            if dual_threshold > 0 and duals[-1] > dual_threshold:
                break

            if dual_threshold < 0 and duals[-1] < dual_threshold:
                break

            if step_size_clip == step_size:
                basis[departing_idx] = False
                basis[candidate_idx] = True

            if np.isclose(duals[-1], dual_threshold):
                break

            reduced_A = np.linalg.solve(phi[basis].T, phi[~basis].T)
            reduced_costs = (S[~basis].T - S[basis].T @ reduced_A).flatten()
            bottom = reduced_A[-1]
            bottom[np.isclose(bottom, 0)] = np.inf
            req_change = reduced_costs / bottom
            if dual_threshold >= 0:
                ignore_entries = np.isclose(bottom, 0) | np.asarray(req_change <= 1e-5)
            else:
                ignore_entries = np.isclose(bottom, 0) | np.asarray(req_change >= -1e-5)
            if np.sum(~ignore_entries) == 0:  # pragma: no cover
                S[-1] = np.inf if quantiles[-1] >= 0.5 else -np.inf
                break
            if dual_threshold >= 0:
                candidate_idx = np.where(~basis)[0][
                    np.where(~ignore_entries, req_change, np.inf).argmin()
                ]
                S[-1] += np.min(req_change[~ignore_entries])
            else:
                candidate_idx = np.where(~basis)[0][
                    np.where(~ignore_entries, req_change, -np.inf).argmax()
                ]
                S[-1] += np.max(req_change[~ignore_entries])
            num_iters += 1
            if num_iters > 10000:  # pragma: no cover
                S[-1] = np.inf if dual_threshold > 0 else -1 * np.inf
        return S[-1]

    def _predict_conditional_cutoff(
        self,
        quantile: float,
        x_test: np.ndarray,
        S_min: Optional[float] = None,
        S_max: Optional[float] = None,
    ) -> float:
        """
        Computes the conditional conformity-score cutoff S^* for a single test
        point, i.e. the (conditionally valid) threshold such that the
        prediction set is ``{y : S(x, y) <= S^*}``.

        Whether the cutoff is computed exactly or by binary search, and whether
        the dual threshold is randomized, is controlled by the ``exact`` and
        ``randomize`` attributes set at initialisation.

        Arguments
        ---------
        quantile : float
            Nominal quantile level.
        x_test : np.ndarray
            Single test point, of shape ``(1, n_features)``.
        S_min : float = None
            Lower bound (if available) on the conformity scores
        S_max : float = None
            Upper bound (if available) on the conformity scores

        Returns
        -------
        float
            The conditional score cutoff S^*.
        """
        quantiles = np.ones((len(self.scores_calib) + 1, 1)) * quantile
        if self.randomize:
            threshold = self.rng.uniform(low=quantile - 1, high=quantile)
        elif quantile < 0.5:
            threshold = quantile - 1
        else:
            threshold = quantile

        if self.exact:
            if self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]):
                raise ValueError(
                    "Exact computation doesn't support RKHS quantile regression for now."
                )
            naive_duals, naive_primals = self._get_calibration_solution(quantile)
            score_cutoff = self._compute_exact_cutoff(
                quantiles,
                naive_primals,
                naive_duals,
                self.feature_map(x_test),
                threshold,
            )
        else:
            _solve = partial(
                _solve_dual,
                gcc=self,
                x_test=x_test,
                quantiles=quantiles,
                threshold=threshold,
            )

            if S_min is None:
                S_min = np.min(self.scores_calib)
            if S_max is None:
                S_max = np.max(self.scores_calib)
            lower, upper = binary_search(_solve, S_min, S_max * 2)

            if quantile < 0.5:
                score_cutoff = self._get_threshold(lower, x_test, quantiles)
            else:
                score_cutoff = self._get_threshold(upper, x_test, quantiles)

        return float(np.ravel(score_cutoff)[0])

    def _get_primal_solution(self, S: float, x: np.ndarray, quantiles: np.ndarray):
        if self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]):
            cp = _import_cvxpy()
            prob = finish_dual_setup(
                self.cvx_problem,
                S,
                x,
                quantiles[-1][0],
                self.feature_map(x),
                self.x_calib,
                self.infinite_params,
            )
            if "MOSEK" in cp.installed_solvers():
                prob.solve(solver="MOSEK")  # pragma: no cover
            else:
                prob.solve()

            weights = prob.var_dict["weights"].value
            beta = prob.constraints[-1].dual_value
        else:
            scores = np.concatenate([self.scores_calib, [S]])
            Phi = np.concatenate([self.phi_calib, self.feature_map(x)], axis=0)
            zeros = np.zeros((Phi.shape[1],))
            bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
            res = linprog(
                -1 * scores,
                A_eq=Phi.T,
                b_eq=zeros,
                bounds=bounds,
                method="highs-ds",
                options={"presolve": False},
            )
            beta = -1 * res.eqlin.marginals
            weights = None
        return beta, weights

    def _get_threshold(self, S: float, x: np.ndarray, quantiles: np.ndarray):
        beta, weights = self._get_primal_solution(S, x, quantiles)

        threshold = self.feature_map(x) @ beta
        if self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]):
            K = pairwise_kernels(
                X=np.concatenate([self.x_calib, x.reshape(1, -1)], axis=0),
                Y=np.concatenate([self.x_calib, x.reshape(1, -1)], axis=0),
                metric=self.infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"]),
                gamma=self.infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"]),
            )
            threshold = (K @ weights)[-1] + threshold
        return threshold


class ConditionalSplitConformalRegressor(
    _ConditionalConformalMixin, SplitConformalRegressor
):
    def __init__(
        self,
        feature_map: Callable,
        estimator: RegressorMixin = LinearRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseRegressionScore] = "absolute",
        prefit: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        randomize: bool = False,
        exact: bool = True,
        infinite_params: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        """
        Split conformal regressor with conditional validity guarantees.

        In addition to the parameters of
        :class:`~mapie.regression.SplitConformalRegressor`, this class accepts
        settings for the conditional conformal procedure.

        Parameters
        ----------
        feature_map : Callable
            Function mapping covariates to a finite basis used for exact
            conditional guarantees.

        estimator : RegressorMixin, default=LinearRegression()
            Base regressor used to predict points.

        confidence_level : float or iterable of float, default=0.9
            Desired coverage probability of the prediction intervals.

        conformity_score : str or BaseRegressionScore, default="absolute"
            Method used to compute conformity scores. See
            :class:`~mapie.regression.SplitConformalRegressor`.

        prefit : bool, default=True
            Whether the base regressor is already fitted.

        n_jobs : int, optional
            Number of parallel jobs when applicable.

        verbose : int, default=0
            Verbosity level.

        randomize : bool, default=False
            Randomize the dual threshold for exact (non-conservative) coverage.

        exact : bool, default=True
            Compute the conditional score cutoff exactly rather than by binary
            search.

        infinite_params : dict, optional
            Parameters for the RKHS component of the fit. Valid keys are
            ``kernel``, ``gamma``, and ``lambda``.
        """
        super().__init__(
            estimator=estimator,
            confidence_level=confidence_level,
            conformity_score=conformity_score,
            prefit=prefit,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self._init_conditional(feature_map, randomize, exact, infinite_params, seed)

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> "ConditionalSplitConformalRegressor":
        """
        Conformalize the regressor and set up the final fitting problem
        for the given conformalization set.

        Performs the standard split-conformal conformalization step from
        :meth:`SplitConformalRegressor.conformalize`, then builds the
        cvxpy problem used for the conditional procedure.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Features of the conformalization set.

        y_conformalize : ArrayLike
            Targets of the conformalization set.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` method of the base
            regressor.

        Returns
        -------
        Self
            The conformalized ConditionalSplitConformalRegressor instance.
        """
        super().conformalize(
            X_conformalize, y_conformalize, predict_params=predict_params
        )

        self.y_calib = np.asarray(y_conformalize)
        self._conformalize_conditional(
            np.asarray(X_conformalize),
            self.conformity_scores,  # computed in super().conformalize
        )

        return self

    def predict_interval(
        self,
        X: ArrayLike,
        minimize_interval_width: bool = False,
        allow_infinite_bounds: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predicts points (using the base regressor) and conditionally valid
        intervals.

        If several confidence levels were provided during initialisation,
        several intervals will be predicted for each sample. See the return
        signature.

        Parameters
        ----------
        X : ArrayLike
            Features.

        minimize_interval_width : bool, default=False
            Not supported by the conditional procedure; provided for API
            compatibility with
            :class:`~mapie.regression.SplitConformalRegressor`.

        allow_infinite_bounds : bool, default=False
            Accepted for API compatibility with
            :class:`~mapie.regression.SplitConformalRegressor`. Note that the
            conditional procedure may return infinite bounds regardless of this
            flag (e.g. when no finite cutoff can be found).

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction points, of shape `(n_samples,)`
            - Prediction intervals, of shape
              `(n_samples, 2, n_confidence_levels)`
        """
        _raise_error_if_previous_method_not_called(
            "predict_interval",
            "conformalize",
            self._is_conformalized,
        )
        if minimize_interval_width:
            raise NotImplementedError(
                "minimize_interval_width is not supported by "
                "ConditionalSplitConformalRegressor."
            )

        X = np.asarray(X)
        y_pred = self.predict(X)

        score = self._conformity_score
        alphas = list(self._alphas)
        n_samples = len(X)
        intervals = np.empty((n_samples, 2, len(alphas)))

        for j, alpha in enumerate(alphas):
            for i in range(n_samples):
                x_row = X[i].reshape(1, -1)
                if score.sym:
                    # Symmetric scores are absolute, so a single cutoff at the
                    # 1 - alpha quantile inverts to a two-sided interval.
                    cutoff = self._predict_conditional_cutoff(1 - alpha, x_row)
                    low = score.get_estimation_distribution(y_pred[i], -cutoff, X=x_row)
                    up = score.get_estimation_distribution(y_pred[i], cutoff, X=x_row)
                else:
                    # Signed scores need one cutoff per side.
                    cutoff_low = self._predict_conditional_cutoff(alpha / 2, x_row)
                    cutoff_up = self._predict_conditional_cutoff(1 - alpha / 2, x_row)
                    low = score.get_estimation_distribution(
                        y_pred[i], cutoff_low, X=x_row
                    )
                    up = score.get_estimation_distribution(
                        y_pred[i], cutoff_up, X=x_row
                    )
                intervals[i, 0, j] = float(low)
                intervals[i, 1, j] = float(up)

        return y_pred, intervals


class ConditionalSplitConformalClassifier(
    _ConditionalConformalMixin, SplitConformalClassifier
):
    def __init__(
        self,
        feature_map: Callable,
        estimator: ClassifierMixin = LogisticRegression(),
        confidence_level: Union[float, Iterable[float]] = 0.9,
        conformity_score: Union[str, BaseClassificationScore] = "lac",
        prefit: bool = True,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        randomize: bool = False,
        exact: bool = True,
        infinite_params: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        """
        Split conformal classifier with conditional validity guarantees.

        In addition to the parameters of
        :class:`~mapie.classification.SplitConformalClassifier`, this class
        accepts settings for the conditional conformal procedure.

        Parameters
        ----------
        feature_map : Callable
            Function mapping covariates to a finite basis used for exact
            conditional guarantees.

        estimator : ClassifierMixin, default=LogisticRegression()
            Base classifier used to predict labels.

        confidence_level : float or iterable of float, default=0.9
            Desired coverage probability of the prediction sets.

        conformity_score : str or BaseClassificationScore, default="lac"
            Method used to compute conformity scores. The conditional
            procedure inverts a real-valued score cutoff into a prediction
            set, so only scores whose prediction sets are obtained by
            thresholding real-valued scores are supported ("lac", "aps");
            "top_k" and "raps" are not.

        prefit : bool, default=True
            Whether the base classifier is already fitted.

        n_jobs : int, optional
            Number of parallel jobs when applicable.

        verbose : int, default=0
            Verbosity level.

        randomize : bool, default=False
            Randomize the dual threshold for exact (non-conservative) coverage.

        exact : bool, default=True
            Compute the conditional score cutoff exactly rather than by binary
            search.

        infinite_params : dict, optional
            Parameters for the RKHS component of the fit. Valid keys are
            ``kernel``, ``gamma``, and ``lambda``.
        """
        super().__init__(
            estimator=estimator,
            confidence_level=confidence_level,
            conformity_score=conformity_score,
            prefit=prefit,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        if isinstance(
            self._conformity_score, (RAPSConformityScore, TopKConformityScore)
        ):
            raise ValueError(
                "ConditionalSplitConformalClassifier requires a conformity "
                "score whose prediction sets are obtained by thresholding "
                'real-valued scores, e.g. "lac" or "aps".'
            )
        self._init_conditional(feature_map, randomize, exact, infinite_params, seed)

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        predict_params: Optional[dict] = None,
    ) -> "ConditionalSplitConformalClassifier":
        """
        Conformalize the classifier and set up the final fitting problem
        for the given conformalization set.

        Performs the standard split-conformal conformalization step from
        :meth:`SplitConformalClassifier.conformalize`, then builds the
        cvxpy problem used for the conditional procedure.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Features of the conformalization set.

        y_conformalize : ArrayLike
            Targets of the conformalization set.

        predict_params : Optional[dict], default=None
            Parameters to pass to the ``predict`` and ``predict_proba``
            methods of the base classifier.

        Returns
        -------
        Self
            The conformalized ConditionalSplitConformalClassifier instance.
        """
        super().conformalize(
            X_conformalize, y_conformalize, predict_params=predict_params
        )

        self._conformalize_conditional(
            np.asarray(X_conformalize),
            self.conformity_scores,  # computed in super().conformalize
        )

        return self

    def predict_set(
        self,
        X: ArrayLike,
        conformity_score_params: Optional[dict] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        For each sample in X, predicts a label (using the base classifier)
        and a conditionally valid set of labels.

        If several confidence levels were provided during initialisation,
        several sets will be predicted for each sample. See the return
        signature.

        Parameters
        ----------
        X : ArrayLike
            Features.

        conformity_score_params : Optional[dict], default=None
            Parameters specific to conformity scores, used at prediction time
            (e.g. ``include_last_label`` for the "aps" conformity score).

        Returns
        -------
        Tuple[NDArray, NDArray]
            Two arrays:

            - Prediction labels, of shape `(n_samples,)`
            - Prediction sets, of shape
              `(n_samples, n_class, n_confidence_levels)`
        """
        _raise_error_if_previous_method_not_called(
            "predict_set",
            "conformalize",
            self._is_conformalized,
        )
        conformity_score_params_ = _prepare_params(conformity_score_params)
        include_last_label = conformity_score_params_.get("include_last_label", True)

        X = np.asarray(X)
        mapie_classifier = self._mapie_classifier
        y_pred_proba = mapie_classifier.estimator_.single_estimator_.predict_proba(
            X, **self._predict_params
        )
        y_pred_proba = check_proba_normalized(y_pred_proba, axis=1)
        y_pred = mapie_classifier.label_encoder_.inverse_transform(
            np.argmax(y_pred_proba, axis=1)
        )

        score = mapie_classifier.conformity_score_function_
        alphas = np.asarray(self._alphas)
        n_samples, n_classes = y_pred_proba.shape
        prediction_sets = np.empty((n_samples, n_classes, len(alphas)), dtype=bool)

        y_pred_proba = score.get_predictions(
            X,
            alphas,
            y_pred_proba,
            cv="prefit",
            include_last_label=include_last_label,
        )

        for i in range(n_samples):
            x_row = X[i].reshape(1, -1)
            # Classification scores are one-sided ("higher = less conforming"),
            # so a single cutoff at the 1 - alpha quantile inverts to the
            # prediction set {y : S(x, y) <= cutoff}. The inversion itself is
            # delegated to the conformity score, with the conditional cutoffs
            # in place of the marginal quantiles.
            score.quantiles_ = np.asarray(
                [self._predict_conditional_cutoff(1 - alpha, x_row) for alpha in alphas]
            )
            prediction_sets[i] = score.get_prediction_sets(
                y_pred_proba[[i]],
                self.scores_calib,
                alphas,
                cv="prefit",
                include_last_label=include_last_label,
            )[0]

        return y_pred, prediction_sets


def binary_search(func, min, max, tol=1e-3):
    min, max = float(min), float(max)
    assert (max + tol) > max
    while (max - min) > tol:
        mid = (min + max) / 2
        if func(mid) > 0:
            max = mid
        else:
            min = mid
    return min, max


def _solve_dual(S, gcc, x_test, quantiles, threshold=None):
    if gcc.infinite_params.get("kernel", None):
        cp = _import_cvxpy()
        prob = finish_dual_setup(
            gcc.cvx_problem,
            S,
            x_test,
            quantiles[-1][0],
            gcc.feature_map(x_test),
            gcc.x_calib,
            gcc.infinite_params,
        )
        if "MOSEK" in cp.installed_solvers():
            prob.solve(solver="MOSEK")  # pragma: no cover
        else:
            prob.solve(solver="OSQP")
        weights = prob.var_dict["weights"].value
    else:
        S = np.concatenate([gcc.scores_calib, [S]], dtype=float)
        Phi = np.concatenate(
            [gcc.phi_calib, gcc.feature_map(x_test)], axis=0, dtype=float
        )
        zeros = np.zeros((Phi.shape[1],))

        bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
        res = linprog(
            -1 * S,
            A_eq=Phi.T,
            b_eq=zeros,
            bounds=bounds,
            method="highs",
            options={"presolve": False},
        )
        weights = res.x

    if threshold is None:
        if quantiles[-1] < 0.5:
            threshold = quantiles[-1] - 1
        else:
            threshold = quantiles[-1]
    # if quantile < 0.5:
    #     return weights[-1] + (1 - quantile)
    return weights[-1] - threshold


def setup_cvx_problem(x_calib, scores_calib, phi_calib, infinite_params={}):
    cp = _import_cvxpy()

    n_calib = len(scores_calib)
    if phi_calib is None:
        phi_calib = np.ones((n_calib, 1))

    eta = cp.Variable(name="weights", shape=n_calib + 1)

    quantile = cp.Parameter(name="quantile")

    scores_const = cp.Constant(scores_calib.reshape(-1, 1))
    scores_param = cp.Parameter(name="S_test", shape=(1, 1))
    scores = cp.vstack([scores_const, scores_param])

    Phi_calibration = cp.Constant(phi_calib)
    Phi_test = cp.Parameter(name="Phi_test", shape=(1, phi_calib.shape[1]))
    Phi = cp.vstack([Phi_calibration, Phi_test])

    kernel = infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"])
    gamma = infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])

    if kernel is None:  # no RKHS fitting
        constraints = [(quantile - 1) <= eta, quantile >= eta, eta.T @ Phi == 0]
        prob = cp.Problem(
            cp.Minimize(-1 * cp.sum(cp.multiply(eta, cp.vec(scores, order="F")))),
            constraints,
        )
    else:  # RKHS fitting
        radius = cp.Parameter(name="radius", nonneg=True)

        _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)

        L_11_const = cp.Constant(np.hstack([L_11, np.zeros((L_11.shape[0], 1))]))
        L_21_22_param = cp.Parameter(name="L_21_22", shape=(1, n_calib + 1))
        L = cp.vstack([L_11_const, L_21_22_param])

        C = radius / (n_calib + 1)

        # this is really C * (quantile - 1) and C * quantile
        constraints = [(quantile - 1) <= eta, quantile >= eta, eta.T @ Phi == 0]
        prob = cp.Problem(
            cp.Minimize(
                0.5 * C * cp.sum_squares(L.T @ eta)
                - cp.sum(cp.multiply(eta, cp.vec(scores, order="F")))
            ),
            constraints,
        )
    return prob


def _get_kernel_matrix(x_calib, kernel, gamma):
    K = pairwise_kernels(X=x_calib, metric=kernel, gamma=gamma) + 1e-5 * np.eye(
        len(x_calib)
    )

    K_chol = np.linalg.cholesky(K)
    return K, K_chol


def finish_dual_setup(
    prob: Any,
    S: Any,
    X: np.ndarray,
    quantile: Any,
    Phi: np.ndarray,
    x_calib: np.ndarray,
    infinite_params={},
):
    prob.param_dict["S_test"].value = np.asarray([[S]])
    prob.param_dict["Phi_test"].value = Phi.reshape(1, -1)
    prob.param_dict["quantile"].value = quantile

    kernel = infinite_params.get("kernel", FUNCTION_DEFAULTS["kernel"])
    gamma = infinite_params.get("gamma", FUNCTION_DEFAULTS["gamma"])
    radius = 1 / infinite_params.get("lambda", FUNCTION_DEFAULTS["lambda"])

    if kernel is not None:
        K_12 = pairwise_kernels(
            X=np.concatenate([x_calib, X.reshape(1, -1)], axis=0),
            Y=X.reshape(1, -1),
            metric=kernel,
            gamma=gamma,
        )

        _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
        K_22 = pairwise_kernels(X=X.reshape(1, -1), metric=kernel, gamma=gamma)
        L_21 = np.linalg.solve(L_11, K_12[:-1]).T
        L_22 = K_22 - L_21 @ L_21.T
        L_22[L_22 < 0] = 0
        L_22 = np.sqrt(L_22)
        prob.param_dict["L_21_22"].value = np.hstack([L_21, L_22])

        prob.param_dict["radius"].value = radius

        # update quantile definition for silly cvxpy reasons
        prob.param_dict["quantile"].value = quantile
        # prob.param_dict['quantile'].value *= radius / (len(x_calib) + 1)

    return prob

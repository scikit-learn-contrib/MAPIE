from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.risk_control.methods import _check_risk_monotonicity
from mapie.risk_control.risks import RiskLoss, RiskLossLike, _resolve_risk, recall_loss
from mapie.utils import _transform_confidence_level_to_alpha


def _import_torch():
    """Import PyTorch lazily, raising a helpful error if it is not installed.

    PyTorch is an optional dependency of MAPIE (the ``conditional`` extra), so
    it is imported only when this module (and thus
    :class:`ConditionalExpectedRiskController`) is actually accessed.
    ``mapie.risk_control`` loads this module lazily, so importing it stays cheap
    and free of a hard PyTorch dependency.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for ConditionalExpectedRiskController. "
            "Install it with: pip install mapie[conditional]"
        ) from e
    return torch


# This module is imported lazily (see ``risk_control/__init__.py``), so importing
# PyTorch here does not make it a dependency of ``mapie.risk_control``. If it is
# missing, ``_import_torch`` raises an actionable error pointing at the extra.
if TYPE_CHECKING:
    import torch
else:
    torch = _import_torch()
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class ConditionalExpectedRiskController:
    """
    Conformal risk control with a learned, input-dependent prediction parameter.

    Unlike threshold-based controllers that select a single global decision
    threshold, ``ConditionalExpectedRiskController`` learns a smooth function of
    the input (its embedding) that returns a per-input prediction parameter.
    The prediction-parameter model is trained on the conformalization set so
    that a user-provided bounded, monotone risk is controlled at the target
    level ``alpha = 1 - confidence_level``.

    The loss is evaluated in PyTorch inside the AA-CRC objective [1]_, so its
    gradient with respect to the prediction parameter is preserved.

    This implementation relies on PyTorch (an optional dependency, installable
    via ``pip install mapie[conditional]``). PyTorch is imported lazily, so it
    is only required when this class is actually accessed; importing
    ``mapie.risk_control`` itself does not require PyTorch.

    Parameters
    ----------
    predict_function : Callable
        Function returning the raw predictions when called with ``X``. When
        called with ``X`` and one prediction parameter per sample, it must
        return the corresponding final predictions.

    feature_map : Callable[[ArrayLike], NDArray]
        Function mapping each input to a fixed-size embedding of shape
        ``(n_samples, n_features)``. The same function must accept a single input
        and return an embedding of shape ``(1, n_features)``.

    confidence_level : float
        Confidence level with which the risk is controlled. Must be in ``(0, 1)``.
        The target risk level is ``alpha = 1 - confidence_level``.

    risk : {"recall", "miscoverage"} or RiskLoss
        Differentiable loss or performance metric to control.
        Pass a :class:`RiskLoss` for a custom loss.
        Its wrapped PyTorch function must be monotone with respect to the
        prediction parameter. The monotonic direction is inferred during
        :meth:`conformalize`.

    predict_param_range : tuple of float, default=(0.0, 1.0)
        Lower and upper bounds of the learned prediction parameter.

    base_model : torch.nn.Module, optional
        Mapping from an embedding to the requested prediction-parameter range.
        If ``None`` (default), a single linear layer sized to the embedding
        dimension is created during :meth:`conformalize`.

    learning_rate : float, default=1e-4
        Learning rate of the Adam optimiser used to train the
        prediction-parameter model.

    weight_decay : float, default=1e-5
        Weight decay (L2 penalty) of the Adam optimiser.

    Attributes
    ----------
    X_conformalize_embedded : NDArray
        Embeddings of the conformalization inputs, as returned by
        ``feature_map``. Set by :meth:`conformalize`.

    y_conformalize : ArrayLike
        Targets of the conformalization set. Set by :meth:`conformalize`.

    y_conformalize_pred : NDArray
        Raw predictions on the conformalization set. Set by
        :meth:`conformalize`.

    base_model : torch.nn.Module
        The fitted prediction-parameter model. Set by :meth:`conformalize`.

    References
    ----------
    .. [1] Blot, V., Angelopoulos, A. N., Jordan, M. I., & Brunel, N. J-B.
        (2025). "Automatically Adaptive Conformal Risk Control."
        Proceedings of The 28th International Conference on Artificial
        Intelligence and Statistics, PMLR 258:19-27.
        https://arxiv.org/abs/2406.17819v4

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.risk_control.adaptive_conformal_risk_control import (
    ...     ConditionalExpectedRiskController,
    ... )
    >>> rng = np.random.default_rng(42)
    >>> X = rng.random((8, 4, 4))
    >>> y = (X > 0.5).astype(float)
    >>> def feature_map(X):
    ...     X = np.asarray(X)
    ...     return X.reshape(1, -1) if X.ndim == 2 else X.reshape(X.shape[0], -1)
    >>> def predict_function(X, predict_param=None):
    ...     X = np.asarray(X)
    ...     if predict_param is None:
    ...         return X
    ...     shape = (len(predict_param),) + (1,) * (X.ndim - 1)
    ...     return X >= np.asarray(predict_param).reshape(shape)
    >>> crc = ConditionalExpectedRiskController(
    ...     predict_function=predict_function,
    ...     feature_map=feature_map,
    ...     confidence_level=0.9,
    ...     risk="recall",
    ... )
    >>> crc.conformalize(X, y, n_epochs=3, batch_size=4)
    >>> y_pred = crc.predict(X[:2], n_epochs=2, batch_size=4)
    >>> y_pred.shape
    (2, 4, 4)
    """

    def __init__(
        self,
        predict_function: Callable[..., NDArray],
        feature_map: Callable[[ArrayLike], NDArray],
        confidence_level: float,
        risk: RiskLossLike,
        predict_param_range: Tuple[float, float] = (0.0, 1.0),
        base_model: Optional["torch.nn.Module"] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        self.predict_function = predict_function
        self.feature_map = feature_map
        self.confidence_level = confidence_level
        self.risk = risk
        self._risk = _resolve_risk(risk)
        self.predict_param_range = predict_param_range
        self._alpha = _transform_confidence_level_to_alpha(confidence_level)

        self.base_model = base_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def conformalize(
        self,
        X_conformalize: ArrayLike,
        y_conformalize: ArrayLike,
        n_epochs: int = 100,
        batch_size: int = 256,
    ) -> None:
        """
        Conformalize the controller on a held-out conformalization set.

        Stores the embedded inputs, targets and raw predictions, then trains
        the parameter model (:attr:`base_model`) so that the risk is controlled
        at the level ``alpha = 1 - confidence_level``.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Inputs of the conformalization set.

        y_conformalize : ArrayLike
            Targets of the conformalization set (e.g. binary masks).

        n_epochs : int, default=100
            Number of training epochs over the conformalization set.

        batch_size : int, default=256
            Mini-batch size used to train the prediction-parameter model.
        """
        if not callable(self.predict_function):
            raise TypeError("`predict_function` must be callable.")
        if len(self.predict_param_range) != 2:
            raise ValueError("`predict_param_range` must contain two values.")
        lower, upper = self.predict_param_range
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(
                "`predict_param_range` values must be finite and increasing."
            )

        X_conformalize_ = np.asarray(X_conformalize)
        self.X_conformalize_embedded = self.feature_map(X_conformalize_)
        self.y_conformalize = y_conformalize
        self.y_conformalize_pred = self.predict_function(X_conformalize_)
        self._objective_sign = _infer_objective_sign(
            self._risk,
            self.y_conformalize,
            self.y_conformalize_pred,
            self.predict_param_range,
        )

        random_idx = np.random.randint(len(X_conformalize_))
        x_n_plus_1 = X_conformalize_[random_idx]

        if self.base_model is None:
            self.base_model = _LinearHead(self.X_conformalize_embedded.shape[1])
            torch.nn.init.zeros_(self.base_model.fc.weight)
            torch.nn.init.constant_(self.base_model.fc.bias, (lower + upper) / 2)
        self.base_model = _train_model(
            self.base_model,
            self.y_conformalize,
            self.y_conformalize_pred,
            self.X_conformalize_embedded,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            alpha=self._alpha,
            x_n_plus_1=self.feature_map(x_n_plus_1),
            risk=self._risk,
            integration_start=lower,
            objective_sign=self._objective_sign,
        )

    def predict(
        self, X: ArrayLike, n_epochs: int = 2, batch_size: int = 256
    ) -> NDArray:
        """
        Predict outputs for new inputs.

        For each input, the prediction-parameter model is fine-tuned (starting
        from the model fitted during :meth:`conformalize`) and the per-input
        prediction parameter is read off. ``predict_function`` then constructs
        the outputs.

        Parameters
        ----------
        X : ArrayLike
            Inputs to predict for.

        n_epochs : int, default=2
            Number of fine-tuning epochs used to estimate each input's
            prediction parameter.

        batch_size : int, default=256
            Mini-batch size used while fine-tuning.

        Returns
        -------
        NDArray
            Predictions returned by ``predict_function``.
        """
        X_ = np.asarray(X)
        X_embedded = self.feature_map(X_)
        best_predict_params = np.empty(len(X_), dtype=float)
        for i in range(len(X_)):
            # Compute the conditional prediction parameter for this test point.
            x_n_plus_1 = X_embedded[i : i + 1]
            model = _train_model(
                deepcopy(self.base_model),
                self.y_conformalize,
                self.y_conformalize_pred,
                self.X_conformalize_embedded,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                n_epochs=n_epochs,
                batch_size=batch_size,
                alpha=self._alpha,
                x_n_plus_1=x_n_plus_1,
                risk=self._risk,
                integration_start=self.predict_param_range[0],
                objective_sign=self._objective_sign,
            )
            best_predict_params[i] = np.clip(
                model(torch.tensor(x_n_plus_1.astype(np.float32)).to(DEVICE)).item(),
                *self.predict_param_range,
            )

        self.best_predict_params_ = best_predict_params
        return np.asarray(self.predict_function(X_, best_predict_params))


def _infer_objective_sign(
    risk: RiskLoss,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    predict_param_range: Tuple[float, float],
    n_grid_points: int = 100,
) -> float:
    """Infer the sign that makes the AA-CRC objective locally minimizable."""
    y_true_tensor = torch.tensor(
        np.asarray(y_true, dtype=np.float32),
        device=DEVICE,
    )
    y_pred_tensor = torch.tensor(
        np.asarray(y_pred, dtype=np.float32),
        device=DEVICE,
    )
    n_samples = len(y_true_tensor)
    parameter_shape = (n_samples,) + (1,) * (y_pred_tensor.ndim - 1)
    parameters = torch.linspace(
        *predict_param_range,
        n_grid_points,
        dtype=y_pred_tensor.dtype,
        device=DEVICE,
    )

    losses = []
    with torch.no_grad():
        for parameter in parameters:
            values = risk(
                y_true_tensor,
                y_pred_tensor,
                parameter.expand(parameter_shape),
            )
            values = torch.as_tensor(values, device=DEVICE).reshape(-1)
            if len(values) != n_samples:
                raise ValueError(
                    "The risk must return one value per prediction parameter."
                )
            losses.append(values)
    loss_values = torch.stack(losses)

    if not bool(torch.isfinite(loss_values).all()):
        raise ValueError("The risk must return only finite values.")
    tolerance = 1e-6
    if bool((loss_values < -tolerance).any() or (loss_values > 1 + tolerance).any()):
        raise ValueError("The risk values must lie in [0, 1].")

    mean_losses = loss_values.mean(dim=1).cpu().numpy()
    direction = _check_risk_monotonicity(mean_losses)
    if direction == "none":
        raise ValueError(
            "The risk must be monotone with respect to the prediction parameter."
        )
    return 1.0 if direction == "increasing" else -1.0


def _train_model(
    model: "torch.nn.Module",
    y_true: ArrayLike,
    y_pred: ArrayLike,
    embeddings: ArrayLike,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
    alpha: float,
    x_n_plus_1: ArrayLike,
    risk: RiskLoss,
    integration_start: float,
    objective_sign: float = 1.0,
) -> "torch.nn.Module":
    """
    Train the prediction-parameter model on the conformalization set.

    The model is optimised against :class:`_AACRCLoss` and the parameters that
    achieved the lowest loss are returned in evaluation mode.

    Parameters
    ----------
    model : torch.nn.Module
        Threshold model to train.

    y_true : ArrayLike
        Targets of the conformalization set.

    y_pred : ArrayLike
        Raw predictions on the conformalization set.

    embeddings : ArrayLike
        Embeddings of the conformalization set.

    lr : float
        Learning rate of the optimiser.

    weight_decay : float
        Weight decay of the optimiser.

    n_epochs : int
        Number of training epochs.

    batch_size : int
        Mini-batch size.

    alpha : float
        Target risk level.

    x_n_plus_1 : ArrayLike
        Embedding of the new input whose prediction parameter must be controlled.

    risk : RiskLoss
        Differentiable PyTorch loss or performance metric.

    integration_start : float
        Lower integration bound.

    objective_sign : float, default=1.0
        Sign applied to the objective for the inferred monotonic direction.

    Returns
    -------
    torch.nn.Module
        The trained prediction-parameter model, in evaluation mode.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("`batch_size` must be an integer.")
    if batch_size <= 0:
        raise ValueError("`batch_size` must be strictly positive.")

    y_true_tensor = torch.tensor(np.asarray(y_true, dtype=np.float32), device=DEVICE)
    y_pred_tensor = torch.tensor(np.asarray(y_pred, dtype=np.float32), device=DEVICE)
    embeddings_tensor = torch.tensor(np.asarray(embeddings, dtype=np.float32)).to(
        DEVICE
    )
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = _AACRCLoss(
        alpha,
        len(y_true_tensor),
        risk,
        integration_start,
        objective_sign,
    )
    best_model = deepcopy(model)
    best_loss: float = np.inf
    for _ in range(n_epochs):
        indices = torch.randperm(len(y_true_tensor), device=DEVICE)
        for i in range(0, len(y_true_tensor), batch_size):
            batch_indices = indices[i : i + batch_size]
            y_true_batch = y_true_tensor[batch_indices]
            y_pred_batch = y_pred_tensor[batch_indices]
            embeddings_batch = embeddings_tensor[batch_indices]
            optimizer.zero_grad()
            predict_params = model(embeddings_batch)
            predict_param_n_plus_1 = model(
                torch.tensor(np.asarray(x_n_plus_1, dtype=np.float32)).to(DEVICE)
            )
            loss = criterion(
                y_true_batch,
                y_pred_batch,
                predict_params,
                predict_param_n_plus_1,
            )
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_model = deepcopy(model)
    return best_model.eval()


class _LinearHead(torch.nn.Module):
    """
    Single-layer linear head mapping an embedding to a prediction parameter.

    Parameters
    ----------
    input_size : int
        Dimension of the input embedding.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.fc(x)


class _LogisticHead(torch.nn.Module):
    """
    Single-layer logistic regression head mapping an embedding to a prediction
    parameter.

    A linear layer projecting the embedding to a scalar, followed by a sigmoid.

    Parameters
    ----------
    input_size : int
        Dimension of the input embedding.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(input_size, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return torch.sigmoid(self.fc(x))


class _AACRCLoss(torch.nn.Module):
    """
    Differentiable surrogate of the conformal risk.

    For each conformalization point, the integral of the differentiable loss
    minus ``alpha`` is approximated by the trapezoidal rule
    (:meth:`_compute_integrals`). The loss is the average of these integrals
    plus the worst-case integral for the new input. The complete objective is
    negated when the loss decreases with the prediction parameter.

    Parameters
    ----------
    alpha : float
        Target risk level.

    n : int
        Number of conformalization points (used to normalise the loss).

    risk : RiskLoss
        Differentiable PyTorch loss or performance metric.

    integration_start : float, default=0.0
        Lower integration bound.

    objective_sign : float, default=1.0
        ``1`` for an increasing loss and ``-1`` for a decreasing loss.
    """

    def __init__(
        self,
        alpha: float,
        n: int,
        risk: RiskLoss = recall_loss,
        integration_start: float = 0.0,
        objective_sign: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.n = n
        self.risk = risk
        self.integration_start = integration_start
        self.objective_sign = objective_sign

    def forward(
        self,
        y_true: "torch.Tensor",
        y_pred: "torch.Tensor",
        predict_params: "torch.Tensor",
        predict_param_n_plus_1: "torch.Tensor",
    ) -> "torch.Tensor":
        integrals = self._compute_integrals(y_true, y_pred, predict_params)
        current_batch_size = len(y_true)
        batch_scale = self.n / current_batch_size
        worst_case_integral = (1 - self.alpha) * (
            predict_param_n_plus_1.squeeze() - self.integration_start
        )
        objective = (batch_scale * torch.sum(integrals) + worst_case_integral) / (
            self.n + 1
        )
        return self.objective_sign * objective

    def _compute_integrals(
        self,
        y_true: "torch.Tensor",
        y_pred: "torch.Tensor",
        predict_params: "torch.Tensor",
        steps_trapz: int = 100,
    ) -> "torch.Tensor":
        # Use a list to accumulate the differentiable integrals
        integrals: list = []
        for i in range(len(y_true)):
            target = torch.clone(y_true[i]).to(DEVICE)
            prediction = torch.clone(y_pred[i]).to(DEVICE)
            predict_param = predict_params[i]

            target = torch.repeat_interleave(
                target[None, ...],
                steps_trapz,
                dim=0,
            )
            prediction = torch.repeat_interleave(
                prediction[None, ...],
                steps_trapz,
                dim=0,
            )
            start = torch.tensor(
                self.integration_start,
                dtype=predict_param.dtype,
                device=DEVICE,
                requires_grad=True,
            )
            end = predict_param
            steps = steps_trapz

            # Differentiable linspace
            parameters = torch.lerp(
                start,
                end,
                torch.linspace(0, 1, steps, device=DEVICE),
            )
            parameter_shape = (steps_trapz,) + (1,) * (prediction.ndim - 1)
            loss = self.risk(
                target,
                prediction,
                parameters.reshape(parameter_shape),
            )
            integral = torch.trapz(loss - self.alpha, parameters)
            integrals.append(integral)

        # Stack to create a tensor with gradients
        return torch.stack(integrals)

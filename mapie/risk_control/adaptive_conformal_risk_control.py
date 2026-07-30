from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.risk_control.risks import RiskLoss, RiskLossLike, _resolve_risk, recall_loss


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
    risk level.

    The risk is evaluated in PyTorch at the learned prediction parameter. Its
    value provides the exact gradient of the integrated AA-CRC objective [1]_;
    the derivative of the risk itself is not needed.

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

    target_level : float
        Target risk level. Must be in ``(0, 1)``.

    risk : {"recall", "miscoverage"} or RiskLoss
        PyTorch-compatible loss or performance metric to control. Pass a
        :class:`RiskLoss` for a custom loss.
        Its wrapped PyTorch function must be monotone with respect to the
        prediction parameter, and the direction of the resulting controlled
        loss must be declared through ``RiskLoss.monotonicity``.

    predict_param_range : tuple of float, optional
        Lower and upper bounds applied to the prediction parameter. By default,
        the linear model output is used without clipping, as required by the
        vector-space assumption in the AA-CRC guarantee. Providing bounds is a
        practical option, but the clipped predictor does not strictly retain
        that theoretical guarantee.

    base_model : torch.nn.Module, optional
        Mapping from an embedding to a prediction parameter. If ``None``
        (default), a single linear layer sized to the embedding dimension is
        created during :meth:`conformalize`. A user-provided model retains its
        initialization, so custom initialization should be applied to the model
        before it is passed to the controller.

    learning_rate : float, default=1e-4
        Learning rate of the Adam optimiser used to train the
        prediction-parameter model.

    weight_decay : float, default=0
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
    ...     target_level=0.1,
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
        target_level: float,
        risk: RiskLossLike,
        predict_param_range: Optional[Tuple[float, float]] = None,
        base_model: Optional["torch.nn.Module"] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0,
    ) -> None:
        self.predict_function = predict_function
        self.feature_map = feature_map
        if (
            isinstance(target_level, bool)
            or not isinstance(target_level, (float, np.floating))
            or not 0 < target_level < 1
        ):
            raise ValueError("`target_level` must be a float strictly between 0 and 1.")
        self.target_level = target_level
        self.risk = risk
        self._risk = _resolve_risk(risk)
        self.predict_param_range = predict_param_range

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
        at :attr:`target_level`.

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
        if self.predict_param_range is None:
            lower = 0.0
            initial_predict_param = 0.0
        else:
            if len(self.predict_param_range) != 2:
                raise ValueError("`predict_param_range` must contain two values.")
            lower, upper = self.predict_param_range
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(
                    "`predict_param_range` values must be finite and increasing."
                )
            initial_predict_param = (lower + upper) / 2
        X_conformalize_ = np.asarray(X_conformalize)
        self.X_conformalize_embedded = self.feature_map(X_conformalize_)
        self.y_conformalize = y_conformalize
        self.y_conformalize_pred = self.predict_function(X_conformalize_)

        random_idx = np.random.randint(len(X_conformalize_))
        x_n_plus_1 = X_conformalize_[random_idx]

        if self.base_model is None:
            self.base_model = _LinearHead(self.X_conformalize_embedded.shape[1])
            torch.nn.init.zeros_(self.base_model.fc.weight)
            torch.nn.init.constant_(self.base_model.fc.bias, initial_predict_param)
        self.base_model = _train_model(
            self.base_model,
            self.y_conformalize,
            self.y_conformalize_pred,
            self.X_conformalize_embedded,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            alpha=self.target_level,
            x_n_plus_1=self.feature_map(x_n_plus_1),
            risk=self._risk,
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
                alpha=self.target_level,
                x_n_plus_1=x_n_plus_1,
                risk=self._risk,
            )
            predict_param = model(
                torch.tensor(x_n_plus_1.astype(np.float32)).to(DEVICE)
            ).item()
            if self.predict_param_range is not None:
                predict_param = np.clip(
                    predict_param,
                    *self.predict_param_range,
                )
            best_predict_params[i] = predict_param

        self.best_predict_params_ = best_predict_params
        return np.asarray(self.predict_function(X_, best_predict_params))


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
) -> "torch.nn.Module":
    """
    Train the prediction-parameter model on the conformalization set.

    The model is optimised against :class:`_AACRCLoss` using one optimizer step
    per mini-batch. After each epoch, the actual AA-CRC objective is evaluated
    on the complete conformalization set. The parameters achieving the lowest
    objective, including the parameters before the first epoch, are returned in
    evaluation mode.

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
        PyTorch-compatible loss or performance metric.

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
    x_n_plus_1_tensor = torch.tensor(
        np.asarray(x_n_plus_1, dtype=np.float32), device=DEVICE
    )
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = _AACRCLoss(
        alpha,
        len(y_true_tensor),
        risk,
    )
    best_model_state = deepcopy(model.state_dict())
    best_objective = _evaluate_aacrc_objective(
        model,
        y_true_tensor,
        y_pred_tensor,
        embeddings_tensor,
        x_n_plus_1_tensor,
        alpha,
        risk,
        weight_decay,
    )
    for _ in range(n_epochs):
        indices = torch.randperm(len(y_true_tensor), device=DEVICE)
        for i in range(0, len(y_true_tensor), batch_size):
            batch_indices = indices[i : i + batch_size]
            y_true_batch = y_true_tensor[batch_indices]
            y_pred_batch = y_pred_tensor[batch_indices]
            embeddings_batch = embeddings_tensor[batch_indices]
            optimizer.zero_grad()
            predict_params = model(embeddings_batch)
            predict_param_n_plus_1 = model(x_n_plus_1_tensor)
            loss = criterion(
                y_true_batch,
                y_pred_batch,
                predict_params,
                predict_param_n_plus_1,
            )
            loss.backward()
            optimizer.step()
        objective = _evaluate_aacrc_objective(
            model,
            y_true_tensor,
            y_pred_tensor,
            embeddings_tensor,
            x_n_plus_1_tensor,
            alpha,
            risk,
            weight_decay,
        )
        if objective < best_objective:
            best_objective = objective
            best_model_state = deepcopy(model.state_dict())
    model.load_state_dict(best_model_state)
    return model.eval()


def _evaluate_aacrc_objective(
    model: "torch.nn.Module",
    y_true: "torch.Tensor",
    y_pred: "torch.Tensor",
    embeddings: "torch.Tensor",
    x_n_plus_1: "torch.Tensor",
    alpha: float,
    risk: RiskLoss,
    weight_decay: float,
    integration_steps: int = 100,
) -> float:
    """Evaluate the actual AA-CRC objective without computing gradients."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            predict_params = model(embeddings).reshape(-1)
            integrals = []
            for target, prediction, predict_param in zip(
                y_true,
                y_pred,
                predict_params,
            ):
                parameters = torch.linspace(
                    0.0,
                    predict_param.item(),
                    integration_steps,
                    device=predict_param.device,
                    dtype=predict_param.dtype,
                )
                target_repeated = target.unsqueeze(0).expand(
                    integration_steps,
                    *target.shape,
                )
                prediction_repeated = prediction.unsqueeze(0).expand(
                    integration_steps,
                    *prediction.shape,
                )
                parameter_shape = (integration_steps,) + (1,) * prediction.ndim
                losses = risk(
                    target_repeated,
                    prediction_repeated,
                    parameters.reshape(parameter_shape),
                )
                integrals.append(torch.trapz(losses - alpha, parameters))

            predict_param_n_plus_1 = model(x_n_plus_1).squeeze()
            worst_case_integral = (1 - alpha) * predict_param_n_plus_1
            objective = (
                risk.objective_sign
                * (torch.stack(integrals).sum() + worst_case_integral)
                / (len(y_true) + 1)
            )
            regularization = (
                0.5
                * weight_decay
                * sum(parameter.square().sum() for parameter in model.parameters())
            )
            return float((objective + regularization).item())
    finally:
        if was_training:
            model.train()


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

    A linear layer projecting the embedding to a scalar, followed by a sigmoid
    scaled to the requested prediction-parameter range.

    Parameters
    ----------
    input_size : int
        Dimension of the input embedding.

    predict_param_range : tuple of float, default=(0.0, 1.0)
        Lower and upper bounds of the output.
    """

    def __init__(
        self,
        input_size: int,
        predict_param_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(input_size, 1)
        self.lower_bound, self.upper_bound = predict_param_range

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        output_range = self.upper_bound - self.lower_bound
        return self.lower_bound + output_range * torch.sigmoid(self.fc(x))


class _AACRCLoss(torch.nn.Module):
    """
    Gradient surrogate for the integrated AA-CRC objective.

    The numerical value of the integrated objective is not needed during
    training. This module therefore returns a zero-valued scalar whose gradient
    matches that objective exactly. For each conformalization point, the
    endpoint identity gives ``dI/du = loss(u) - alpha``. A zero-valued gradient
    proxy supplies this derivative directly, avoiding numerical integration.
    The complete gradient is negated when the loss decreases with the prediction
    parameter.

    Parameters
    ----------
    alpha : float
        Target risk level.

    n : int
        Number of conformalization points (used to normalise the loss).

    risk : RiskLoss
        PyTorch loss or performance metric.

    """

    def __init__(
        self,
        alpha: float,
        n: int,
        risk: RiskLoss = recall_loss,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.n = n
        self.risk = risk

    def forward(
        self,
        y_true: "torch.Tensor",
        y_pred: "torch.Tensor",
        predict_params: "torch.Tensor",
        predict_param_n_plus_1: "torch.Tensor",
    ) -> "torch.Tensor":
        detached_predict_params = predict_params.detach()
        parameter_shape = (len(predict_params),) + (1,) * (y_pred.ndim - 1)

        # By the endpoint identity, these values are dI_i / dp_i, where
        # p_i = model(x_i). The risk itself is not differentiated.
        with torch.no_grad():
            endpoint_losses = self.risk(
                y_true,
                y_pred,
                detached_predict_params.reshape(parameter_shape),
            )
            integrals_endpoint_derivatives = endpoint_losses - self.alpha

        # The proxy is not numerically zero in the forward pass
        # because predict_params - detached_predict_params is zero. During
        # backpropagation, the derivative of this difference with respect to
        # predict_params is one. Therefore, the proxy supplies dI_i / dp_i,
        # and PyTorch applies the chain rule through p_i = model(x_i) to obtain
        # dI_i / dtheta = (dI_i / dp_i) * (dp_i / dtheta).
        integrals_gradient_proxy = integrals_endpoint_derivatives * (
            predict_params.reshape(len(predict_params))
            - detached_predict_params.reshape(len(predict_params))
        )

        current_batch_size = len(y_true)
        batch_scale = self.n / current_batch_size
        detached_predict_param_n_plus_1 = predict_param_n_plus_1.detach()

        # Use the same zero-valued proxy for the worst-case integral, whose
        # endpoint derivative is 1 - alpha.
        worst_case_endpoint_derivative = 1 - self.alpha
        worst_case_gradient_proxy = (
            worst_case_endpoint_derivative
            * (predict_param_n_plus_1 - detached_predict_param_n_plus_1).squeeze()
        )
        objective_gradient_proxy = (
            batch_scale * torch.sum(integrals_gradient_proxy)
            + worst_case_gradient_proxy
        ) / (self.n + 1)
        return self.risk.objective_sign * objective_gradient_proxy

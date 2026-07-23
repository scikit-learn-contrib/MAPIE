from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mapie.utils import _transform_confidence_level_to_alpha


def _import_torch():
    """Import PyTorch lazily, raising a helpful error if it is not installed.

    PyTorch is an optional dependency of MAPIE (the ``conditional`` extra), so
    it is imported only when this module (and thus
    :class:`AutoAdaptiveConformalRiskControl`) is actually accessed.
    ``mapie.risk_control`` loads this module lazily, so importing it stays cheap
    and free of a hard PyTorch dependency.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for AutoAdaptiveConformalRiskControl. "
            "Install it with: pip install mapie[conditional]"
        ) from e
    return torch


# This module is imported lazily (see ``risk_control/__init__.py``), so importing
# PyTorch here does not make it a dependency of ``mapie.risk_control``. If it is
# missing, ``_import_torch`` raises an actionable error pointing at the extra.
torch = _import_torch()
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class AutoAdaptiveConformalRiskControl:
    """
    Conformal risk control with a learned, input-dependent decision threshold.

    Unlike threshold-based controllers that select a single global decision
    threshold, ``AutoAdaptiveConformalRiskControl`` learns a smooth function of
    the input (its embedding) that returns a per-input decision threshold in
    ``[0, 1]``. The threshold model is trained on the conformalization set so
    that a recall-type risk is controlled at the target level
    ``alpha = 1 - confidence_level``: the empirical risk is approximated by a
    differentiable surrogate (the integral, for each conformalization point, of
    the per-threshold risk minus ``alpha``), which is minimised together with the
    threshold of the new input so that the risk is controlled while the decision
    rule remains as decisive as possible.

    It is designed for structured-output prediction (e.g. semantic
    segmentation), where ``predict_function`` returns predicted probability maps
    and ``feature_map`` returns a vector embedding summarising each input.

    This implementation relies on PyTorch (an optional dependency, installable
    via ``pip install mapie[conditional]``). PyTorch is imported lazily, so it
    is only required when this class is actually accessed; importing
    ``mapie.risk_control`` itself does not require PyTorch.

    Parameters
    ----------
    predict_function : Callable[[ArrayLike], NDArray]
        Function returning the predicted probabilities for each input. For
        structured outputs it must return an array of shape ``(n_samples, ...)``;
        for the typical segmentation setting, ``(n_samples, height, width)``.
        Values must lie in ``[0, 1]``.

    feature_map : Callable[[ArrayLike], NDArray]
        Function mapping each input to a fixed-size embedding of shape
        ``(n_samples, n_features)``. The same function must accept a single input
        and return an embedding of shape ``(1, n_features)``.

    confidence_level : float
        Confidence level with which the risk is controlled. Must be in ``(0, 1)``.
        The target risk level is ``alpha = 1 - confidence_level``.

    risk : Any
        Currently unused. Reserved for a future integration with the
        :mod:`mapie.risk_control` risk system.

    base_model : torch.nn.Module, optional
        Mapping from an embedding to a threshold in ``[0, 1]``. If ``None``
        (default), a :class:`LogisticHead` (a single linear layer followed by a
        sigmoid) sized to the embedding dimension is created during
        :meth:`conformalize`.

    learning_rate : float, default=1e-4
        Learning rate of the Adam optimiser used to train the threshold model.

    weight_decay : float, default=1e-5
        Weight decay (L2 penalty) of the Adam optimiser.

    Attributes
    ----------
    X_conformalize_embedded : NDArray
        Embeddings of the conformalization inputs, as returned by
        ``feature_map``. Set by :meth:`conformalize`.

    y_conformalize : ArrayLike
        Targets of the conformalization set. Set by :meth:`conformalize`.

    y_conformalize_pred_proba : NDArray
        Predicted probabilities of the conformalization set. Set by
        :meth:`conformalize`.

    base_model : torch.nn.Module
        The fitted threshold model. Set by :meth:`conformalize`.

    Examples
    --------
    >>> import numpy as np
    >>> from mapie.risk_control.adaptive_conformal_risk_control import (
    ...     AutoAdaptiveConformalRiskControl,
    ... )
    >>> rng = np.random.default_rng(42)
    >>> X = rng.random((8, 4, 4))
    >>> y = (X > 0.5).astype(float)
    >>> def feature_map(X):
    ...     X = np.asarray(X)
    ...     return X.reshape(1, -1) if X.ndim == 2 else X.reshape(X.shape[0], -1)
    >>> def predict_function(X):
    ...     return np.asarray(X)
    >>> crc = AutoAdaptiveConformalRiskControl(
    ...     predict_function=predict_function,
    ...     feature_map=feature_map,
    ...     confidence_level=0.9,
    ...     risk=None,
    ... )
    >>> crc.conformalize(X, y, n_epochs=3, batch_size=4)
    >>> y_pred = crc.predict(X[:2], n_epochs=2, batch_size=4)
    >>> y_pred.shape
    (2, 4, 4)
    """

    def __init__(
        self,
        predict_function: Callable[[ArrayLike], NDArray],
        feature_map: Callable[[ArrayLike], NDArray],
        confidence_level: float,
        risk: Any,
        base_model: Optional["torch.nn.Module"] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ) -> None:
        self.predict_function = predict_function
        self.feature_map = feature_map
        self.confidence_level = confidence_level
        self.risk = risk
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

        Stores the embedded inputs, targets and predicted probabilities, then
        trains the threshold model (:attr:`base_model`) so that the risk is
        controlled at the level ``alpha = 1 - confidence_level``.

        Parameters
        ----------
        X_conformalize : ArrayLike
            Inputs of the conformalization set.

        y_conformalize : ArrayLike
            Targets of the conformalization set (e.g. binary masks).

        n_epochs : int, default=100
            Number of training epochs over the conformalization set.

        batch_size : int, default=256
            Mini-batch size used to train the threshold model.
        """
        X_conformalize_ = np.asarray(X_conformalize)
        self.X_conformalize_embedded = self.feature_map(X_conformalize_)
        self.y_conformalize = y_conformalize
        self.y_conformalize_pred_proba = self.predict_function(X_conformalize_)

        random_idx = np.random.randint(len(X_conformalize_))
        x_n_plus_1 = X_conformalize_[random_idx]

        if self.base_model is None:
            self.base_model = LogisticHead(self.X_conformalize_embedded.shape[1])
        self.base_model = _train_model(
            self.base_model,
            self.y_conformalize,
            self.y_conformalize_pred_proba,
            self.X_conformalize_embedded,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            alpha=self._alpha,
            x_n_plus_1=self.feature_map(x_n_plus_1),
        )

    def predict(
        self, X: ArrayLike, n_epochs: int = 2, batch_size: int = 256
    ) -> NDArray:
        """
        Predict the binary outputs for new inputs.

        For each input, the threshold model is fine-tuned (starting from the
        model fitted during :meth:`conformalize`) and the per-input threshold is
        read off; the prediction is then the predicted-probability map
        thresholded at that value.

        Parameters
        ----------
        X : ArrayLike
            Inputs to predict for.

        n_epochs : int, default=2
            Number of fine-tuning epochs used to estimate each input's
            threshold.

        batch_size : int, default=256
            Mini-batch size used while fine-tuning.

        Returns
        -------
        NDArray
            Binary predictions, with the same shape as the output of
            ``predict_function``.
        """
        # compute probabilities
        X_ = np.asarray(X)
        y_pred_proba = self.predict_function(X_)

        X_embedded = self.feature_map(X_)
        y_pred: NDArray = np.zeros_like(y_pred_proba)
        for i in range(len(X_)):
            # compute the conditional threshold for this test point
            x_n_plus_1 = X_embedded[i : i + 1]
            model = _train_model(
                self.base_model,
                self.y_conformalize,
                self.y_conformalize_pred_proba,
                self.X_conformalize_embedded,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                n_epochs=n_epochs,
                batch_size=batch_size,
                alpha=self._alpha,
                x_n_plus_1=x_n_plus_1,
            )
            best_predict_param = model(
                torch.tensor(x_n_plus_1.astype(np.float32)).to(DEVICE)
            ).item()
            # compute predictions
            y_pred[i] = (y_pred_proba[i] >= best_predict_param).astype(int)

        return y_pred


def _train_model(
    model: "torch.nn.Module",
    masks: ArrayLike,
    masks_pred: ArrayLike,
    embeddings: ArrayLike,
    lr: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
    alpha: float,
    x_n_plus_1: ArrayLike,
) -> "torch.nn.Module":
    """
    Train the threshold model on the conformalization set.

    The model is optimised against :class:`CustomLoss` and the parameters that
    achieved the lowest loss are returned in evaluation mode.

    Parameters
    ----------
    model : torch.nn.Module
        Threshold model to train.

    masks : ArrayLike
        Targets of the conformalization set.

    masks_pred : ArrayLike
        Predicted probabilities of the conformalization set.

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
        Embedding of the new input whose threshold must be controlled.

    Returns
    -------
    torch.nn.Module
        The trained threshold model, in evaluation mode.
    """
    masks = torch.tensor(np.asarray(masks, dtype=np.float32))
    masks_pred = torch.tensor(np.asarray(masks_pred, dtype=np.float32))
    embeddings = torch.tensor(np.asarray(embeddings, dtype=np.float32)).to(DEVICE)
    model = model.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CustomLoss(alpha, batch_size)
    losses = []
    best_model = deepcopy(model)
    best_loss: float = np.inf
    for epoch in range(n_epochs):
        for i in range(0, len(masks), batch_size):
            masks_batch = masks[i : i + batch_size]
            masks_pred_batch = masks_pred[i : i + batch_size]
            embeddings_batch = embeddings[i : i + batch_size]
            optimizer.zero_grad()
            ths_pred = model(embeddings_batch)
            th_n_plus_1 = model(
                torch.tensor(np.asarray(x_n_plus_1, dtype=np.float32)).to(DEVICE)
            )
            loss = criterion(masks_batch, masks_pred_batch, ths_pred, th_n_plus_1)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_model = deepcopy(model)
    return best_model.eval()


class LogisticHead(torch.nn.Module):
    """
    Single-layer logistic regression head mapping an embedding to a threshold.

    A linear layer projecting the embedding to a scalar, followed by a sigmoid,
    so that the output lies in ``[0, 1]`` and can be interpreted as a decision
    threshold.

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


class CustomLoss(torch.nn.Module):
    """
    Differentiable surrogate of the conformal risk.

    For each conformalization point, the integral of the (smoothed)
    per-threshold risk minus ``alpha`` is approximated by trapezoidal rule
    (:meth:`_I_gpu`). The loss is the average of these integrals plus the
    threshold of the new input, scaled so that minimising it controls the risk at
    level ``alpha``.

    Parameters
    ----------
    alpha : float
        Target risk level.

    n : int
        Number of conformalization points (used to normalise the loss).
    """

    def __init__(self, alpha: float, n: int) -> None:
        super().__init__()
        self.alpha = alpha
        self.n = n

    def forward(
        self,
        masks: "torch.Tensor",
        masks_pred: "torch.Tensor",
        preds_th: "torch.Tensor",
        th_n_plus_1: "torch.Tensor",
    ) -> "torch.Tensor":
        integrals = self._I_gpu(masks, masks_pred, preds_th)
        return torch.sum(integrals) / (self.n + 1) + (1 - self.alpha) * th_n_plus_1 / (
            self.n + 1
        )

    def _I_gpu(
        self,
        masks: "torch.Tensor",
        masks_pred: "torch.Tensor",
        preds_th: "torch.Tensor",
        steps_trapz: int = 100,
    ) -> "torch.Tensor":
        # Use a list to accumulate the differentiable integrals
        integrals: list = []
        for i in range(len(masks)):
            mask = torch.clone(masks[i]).to(DEVICE)
            mask_pred = torch.clone(masks_pred[i]).to(DEVICE)
            pred_th = preds_th[i]

            mask = torch.repeat_interleave(mask[None, :, :], steps_trapz, dim=0)
            mask_pred = torch.repeat_interleave(
                mask_pred[None, :, :], steps_trapz, dim=0
            )
            start = torch.tensor(0.0, device=mask.device, requires_grad=True)
            end = pred_th  # pred_th should already have requires_grad=True
            steps = steps_trapz

            # Differentiable linspace
            us = torch.lerp(start, end, torch.linspace(0, 1, steps, device=mask.device))
            us = us.view(-1, 1, 1)  # Add dimensions for broadcasting

            mask_pred_th = torch.sigmoid(1000 * (mask_pred - us))

            loss = 1 - ((mask_pred_th * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)))
            integral = torch.trapz(loss - self.alpha, us.squeeze())
            integrals.append(integral)

            del mask, mask_pred, start, end, steps, us, mask_pred_th, loss, integral

        # Stack to create a tensor with gradients
        return torch.stack(integrals)

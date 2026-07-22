from copy import deepcopy

import numpy as np
import torch

from mapie.utils import _transform_confidence_level_to_alpha


class AutoAdaptiveConformalRiskControl:
    def __init__(
        self,
        predict_function,
        feature_map,
        confidence_level,
        risk,
        base_model=None,  # lambda= base_model=LinearModel() by default TODO: find good name. If None, is initialized as a linear model from the features
        learning_rate=1e-4,
        weight_decay=1e-5,
    ):
        self.predict_function = predict_function
        self.feature_map = feature_map
        self.confidence_level = confidence_level
        self._alpha = _transform_confidence_level_to_alpha(confidence_level)

        self.base_model = base_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        pass

    def conformalize(
        self, X_conformalize, y_conformalize, n_epochs=100, batch_size=256
    ):
        self.X_conformalize_embedded = self.feature_map(X_conformalize)
        self.y_conformalize = y_conformalize
        self.y_conformalize_pred_proba = self.predict_function(X_conformalize)

        random_idx = np.random.randint(len(X_conformalize))
        x_n_plus_1 = X_conformalize[random_idx]

        if self.base_model is None:
            self.base_model = LinearModel(self.X_conformalize_embedded.shape[1])
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

    def predict(self, X, n_epochs=2, batch_size=256):
        # compute probabilities
        y_pred_proba = self.predict_function(X)
        y_pred = np.zeros_like(y_pred_proba)

        for i in range(len(X)):
            # compute thresholds (predict_param)
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
                x_n_plus_1=self.feature_map(X),
            )
            best_predict_param = model(
                torch.tensor(test_emb.astype(np.float32)).to("cuda")
            )
            # compute predictions
            y_pred[i] = (y_pred_proba[i] >= best_predict_param).astype(int)

        return y_pred


def _train_model(
    model,
    masks,
    masks_pred,
    embeddings,
    lr,
    weight_decay,
    n_epochs,
    batch_size,
    alpha,
    x_n_plus_1,
):
    masks = torch.tensor(masks.astype(np.float32))
    masks_pred = torch.tensor(masks_pred.astype(np.float32))
    embeddings = torch.tensor(embeddings.astype(np.float32)).to("cuda")
    model = model.to("cuda")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CustomLoss(alpha, batch_size)
    losses = []
    best_model = deepcopy(model)
    best_loss = np.inf
    for epoch in range(n_epochs):
        for i in range(0, len(masks), batch_size):
            masks_batch = masks[i : i + batch_size]
            masks_pred_batch = masks_pred[i : i + batch_size]
            embeddings_batch = embeddings[i : i + batch_size]
            optimizer.zero_grad()
            ths_pred = model(embeddings_batch)
            th_n_plus_1 = model(torch.tensor(x_n_plus_1).to("cuda"))
            loss = criterion(masks_batch, masks_pred_batch, ths_pred, th_n_plus_1)
            losses.append(loss.item())
            # print(f"Epoch {epoch} / {n_epochs} -- Loss: {loss.item()} -- min th: {ths_pred.min().item()} -- max th: {ths_pred.max().item()}", end="\r")
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = deepcopy(model)
    return best_model.eval()


class LinearModel(torch.nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, int(input_size / 2))

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x[:, 0]


class CustomLoss(torch.nn.Module):
    def __init__(self, alpha, n):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.n = n

    def forward(self, masks, masks_pred, preds_th, th_n_plus_1):
        integrals = self._I_gpu(masks, masks_pred, preds_th)

        # Ensure the returned loss is differentiable
        return torch.sum(integrals) / (self.n + 1) + (1 - self.alpha) * th_n_plus_1 / (
            self.n + 1
        )

    def _I_gpu(self, masks, masks_pred, preds_th, steps_trapz=100):
        integrals = []  # Use a list to accumulate the results
        for i in range(len(masks)):
            mask = torch.clone(masks[i]).cuda()
            mask_pred = torch.clone(masks_pred[i]).cuda()
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

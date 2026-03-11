import copy
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


class SimpleTabularMLP(nn.Module):
    """
    Robust Backbone: ResNet-MLP for Tabular Data. Used by MultivariateResidualNormalisedScore.
    """

    def __init__(
        self,
        num_cont: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Initializes a robust ResNet-style Multilayer Perceptron (MLP) for tabular data.

        Parameters
        ----------
        num_cont : int
            The number of continuous input features (input dimensionality).
        hidden_dim : int, optional
            The number of hidden units in each layer, by default 128.
        num_layers : int, optional
            The number of residual blocks to include, by default 3.
        dropout : float, optional
            The dropout probability applied within the residual blocks, by default 0.1.
        """
        super().__init__()
        self.first_layer = nn.Linear(num_cont, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the ResNet-MLP blocks.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of continuous features with shape (Batch, num_cont).

        Returns
        -------
        torch.Tensor
            The extracted feature representations of shape (Batch, hidden_dim)
            after applying residual connections and layer normalization.
        """
        x = self.first_layer(x)
        for block in self.blocks:
            x = x + block(x)
        return self.norm(x)


class RobustCovarianceHead(nn.Module):
    """
    Unified Head: Switches between Full Cholesky and Low-Rank. Used by MultivariateResidualNormalisedScore.
    """

    def __init__(
        self,
        input_dim: int,
        y_dim: int,
        init_sigma: float = 1.0,
        mode: str = "full_cholesky",
    ) -> None:
        """
        Initializes a unified head that outputs parameters for a multivariate normal distribution.

        It switches between predicting a full Cholesky decomposition or a low-rank approximation.

        Parameters
        ----------
        input_dim : int
            The dimensionality of the incoming feature representations (e.g., from an MLP).
        y_dim : int
            The dimensionality of the target outputs.
        init_sigma : float, optional
            The initial scaling factor for the diagonal covariance elements, by default 1.0.
        mode : str, optional
            The covariance modeling mode, either 'full_cholesky' or 'low_rank', by default 'full_cholesky'.

        Raises
        ------
        ValueError
            If the provided `mode` is neither 'full_cholesky' nor 'low_rank'.
        """
        super().__init__()
        self.y_dim = y_dim
        if mode == "low_rank":
            self.mode = "low_rank"
            rank = int(math.ceil(math.sqrt(y_dim)))
            self.fc_mu = nn.Linear(input_dim, y_dim)
            self.fc_log_diag = nn.Linear(input_dim, y_dim)
            self.fc_factors = nn.Linear(input_dim, y_dim * rank)
            self.rank = rank
        elif mode == "full_cholesky":
            if y_dim > 10:
                print(
                    "Large output dimension, initializing with mode = 'low_rank' is recommanded."
                )
            self.mode = "full_cholesky"
            self.fc_mu = nn.Linear(input_dim, y_dim)
            num_chol = (y_dim * (y_dim + 1)) // 2
            self.fc_chol = nn.Linear(input_dim, num_chol)
            self.register_buffer("tril_indices", torch.tril_indices(y_dim, y_dim))

            # Initialize diagonal to be positive/stable
            with torch.no_grad():
                diag_mask = self.tril_indices[0] == self.tril_indices[1]
                inv_softplus = math.log(math.exp(init_sigma) - 1)
                self.fc_chol.bias[diag_mask] = inv_softplus
        else:
            raise ValueError("The mode must either be 'full_cholesky' or 'low_rank'.")

    def forward(
        self, x: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Performs a forward pass to compute the distributional parameters of the covariance matrix.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (Batch, input_dim).

        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
            If mode is 'low_rank', returns a tuple:
                - mu: (Batch, y_dim) Mean predictions.
                - D: (Batch, y_dim) Diagonal variances.
                - V: (Batch, y_dim, rank) Low-rank factor matrix.
            If mode is 'full_cholesky', returns a tuple:
                - mu: (Batch, y_dim) Mean predictions.
                - L: (Batch, y_dim, y_dim) Lower triangular Cholesky factor.
        """
        if self.mode == "low_rank":
            B = x.shape[0]
            mu = self.fc_mu(x)
            D = torch.exp(self.fc_log_diag(x)) + 1e-6
            V = self.fc_factors(x).view(B, self.y_dim, self.rank)
            return (mu, D, V)
        else:
            B = x.shape[0]
            mu = self.fc_mu(x)
            chol_flat = self.fc_chol(x)
            L = torch.zeros(B, self.y_dim, self.y_dim, device=x.device)
            L[:, self.tril_indices[0], self.tril_indices[1]] = chol_flat
            # Softplus diagonal
            diag_idx = torch.arange(self.y_dim, device=x.device)
            L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-6
            return (mu, L)


class Trainer:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        init_sigma: float = 1.0,
        mode: str = "full_cholesky",
        center_model: Optional[Any] = None,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        Initializes the Trainer with a simple tabular MLP backbone and a robust covariance head.
        Used by MultivariateResidualNormalisedScore.

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input features.
        output_dim : int
            Dimensionality of the target outputs.
        hidden_dim : int, optional
            Number of hidden units in each MLP layer, by default 128.
        num_layers : int, optional
            Number of layers in the MLP backbone, by default 3.
        dropout : float, optional
            Dropout probability for the MLP, by default 0.1.
        init_sigma : float, optional
            Initial scaling factor for the covariance matrix, by default 1.0.
        mode : str, optional
            Mode for the covariance head, e.g., "full_cholesky" or "low_rank", by default "full_cholesky".
        center_model : Optional[Any], optional
            An optional pre-trained model used to predict the center (mean), by default None.
        dtype : torch.dtype, optional
            The data type to use for tensors, by default torch.float32.
        device : Union[str, torch.device], optional
            The device to map the model to ("cpu" or "cuda"), by default "cpu".
        """
        self.center_model = center_model
        self.y_dim = output_dim
        self.mode = mode
        self.dtype = dtype
        self.device = torch.device(device)
        self.fitted = False
        self.backbone = SimpleTabularMLP(
            num_cont=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
        self.head = RobustCovarianceHead(
            input_dim=hidden_dim, y_dim=output_dim, init_sigma=init_sigma, mode=mode
        ).to(self.device)
        if self.mode == "low_rank":
            self.eye_rank = torch.eye(self.head.rank, device=self.device).unsqueeze(0)

    def forward(self, bx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Performs a forward pass through the backbone and head, optionally incorporating the center_model.

        Parameters
        ----------
        bx : torch.Tensor
            A batch of input features of shape (Batch, input_dim).

        Returns
        -------
        Tuple[torch.Tensor, ...]
            A tuple containing the predicted parameters (e.g., Center, Covariance components).
        """
        feats = self.backbone(bx)
        preds = self.head(feats)

        if self.center_model is not None:
            with torch.no_grad():
                center_pred = self.center_model(bx)
                preds = list(preds)

                # Use as_tensor to avoid warnings if it's already a tensor
                center_tensor = torch.as_tensor(
                    center_pred, dtype=self.dtype, device=self.device
                )
                preds[0] = center_tensor
                preds = tuple(preds)

        return preds

    def fit(
        self,
        X_train: Union[NDArray, torch.Tensor],
        y_train: Union[NDArray, torch.Tensor],
        y_pred: Optional[Union[NDArray, torch.Tensor]] = None,
        X_val: Optional[Union[NDArray, torch.Tensor]] = None,
        y_val: Optional[Union[NDArray, torch.Tensor]] = None,
        val_size: float = 0.2,
        batch_size: int = 32,
        num_epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        verbose: int = -1,
    ) -> None:
        """
        Trains the neural network using negative log-likelihood (NLL) loss.

        Parameters
        ----------
        X_train : Union[NDArray, torch.Tensor]
            Training input features.
        y_train : Union[NDArray, torch.Tensor]
            Training target outputs.
        y_train : Union[NDArray, torch.Tensor]
            Training target predictions.
        X_val : Optional[Union[NDArray, torch.Tensor]], optional
            Validation input features, by default None.
        y_val : Optional[Union[NDArray, torch.Tensor]], optional
            Validation target outputs, by default None.
        val_size : float, optional
            Fraction of the training data to use for validation if X_val is None, by default 0.2.
        batch_size : int, optional
            Number of samples per training batch, by default 32.
        num_epochs : int, optional
            Maximum number of epochs to train, by default 300.
        lr : float, optional
            Learning rate for the AdamW optimizer, by default 1e-3.
        weight_decay : float, optional
            Weight decay (L2 penalty) for the optimizer, by default 1e-4.
        verbose : int, optional
            Verbosity mode (-1 for silent, 1 for sparse, 2 for detailed), by default -1.
        """
        optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        if y_pred is not None:
            raise Exception("Class not implemented with y_pred yet.")

        X_train = torch.as_tensor(X_train, dtype=self.dtype, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=self.dtype, device=self.device)

        if X_val is None and y_val is None and val_size > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=42
            )
        elif X_val is not None and y_val is not None:
            X_val = torch.as_tensor(X_val, dtype=self.dtype, device=self.device)
            y_val = torch.as_tensor(y_val, dtype=self.dtype, device=self.device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        if X_val is not None and y_val is not None:
            val_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_val, y_val),
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            val_loader = None

        best_validation_loss = float("inf")
        best_backbone_state = copy.deepcopy(self.backbone.state_dict())
        best_head_state = copy.deepcopy(self.head.state_dict())

        print_every = (
            max(1, num_epochs // 10)
            if verbose == 1
            else (1 if verbose == 2 else num_epochs + 1)
        )

        for epoch in range(num_epochs):
            self.backbone.train()
            self.head.train()
            total_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()

                preds = self.forward(bx)
                loss = self.loss(preds, by)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            self.backbone.eval()
            self.head.eval()
            total_validation_loss = 0.0

            if val_loader is not None:
                with torch.no_grad():
                    for bx, by in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        preds = self.forward(bx)
                        loss = self.loss(preds, by)
                        total_validation_loss += loss.item()
                    avg_validation_loss = total_validation_loss / len(val_loader)
            else:
                avg_validation_loss = avg_train_loss

            if verbose != -1 and epoch % print_every == 0:
                print(
                    f"Epoch {epoch}: Avg NLL Loss = {avg_train_loss:.4f} -- Validation loss: {avg_validation_loss:.4f} -- Best Validation Loss: {best_validation_loss}"
                )

            if avg_validation_loss < best_validation_loss:
                best_validation_loss = avg_validation_loss
                best_backbone_state = {
                    k: v.cpu().clone() for k, v in self.backbone.state_dict().items()
                }
                best_head_state = {
                    k: v.cpu().clone() for k, v in self.head.state_dict().items()
                }

        self.backbone.load_state_dict(best_backbone_state)
        self.head.load_state_dict(best_head_state)
        self.fitted = True

        if verbose != -1:
            print(f"Best validation loss achieved: {best_validation_loss:.4f}")

    def loss(
        self, params: Tuple[torch.Tensor, ...], y_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Negative Log-Likelihood (NLL) loss based on the selected covariance mode.

        Parameters
        ----------
        params : Tuple[torch.Tensor, ...]
            A tuple of predicted distributional parameters (e.g., mu, D, V for low_rank or mu, L for full_cholesky).
        y_target : torch.Tensor
            The ground truth target values.

        Returns
        -------
        torch.Tensor
            The computed scalar loss value.
        """
        if self.mode == "low_rank":
            # Woodbury Identity Loss
            mu, D, V = params
            r = y_target - mu
            B, Y, K = V.shape
            inv_std = 1.0 / D
            W = V * inv_std.unsqueeze(-1)
            z = r * inv_std

            M = self.eye_rank + torch.bmm(W.transpose(1, 2), W)
            L_M = torch.linalg.cholesky(M)

            log_det = 2 * torch.sum(torch.log(D), 1) + 2 * torch.sum(
                torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)), 1
            )

            z_sq = torch.sum(z**2, 1)
            p = torch.bmm(W.transpose(1, 2), z.unsqueeze(-1))
            # q = torch.linalg.cholesky_solve(L_M, p)
            q = torch.cholesky_solve(p, L_M)
            quad = torch.bmm(p.transpose(1, 2), q).squeeze()

            return 0.5 * (z_sq - quad + log_det).mean()

        else:
            # Full Cholesky Loss
            mu, L = params
            diff = (y_target - mu).unsqueeze(-1)
            z = torch.linalg.solve_triangular(L, diff, upper=False)
            mahalanobis = torch.sum(z.squeeze(-1) ** 2, dim=1)
            log_det = 2 * torch.sum(
                torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1
            )

            return 0.5 * (mahalanobis + log_det).mean()

    def get_distribution(
        self, x: Union[NDArray, torch.Tensor]
    ) -> Tuple[NDArray, NDArray]:
        """
        Computes the full conditional distribution of Y given X.

        Parameters
        ----------
        x : Union[NDArray, torch.Tensor]
            Input features of shape (B, input_dim).

        Returns
        -------
        Tuple[NDArray, NDArray]
            - mu: Point predictions (Mean) of shape (B, y_dim).
            - Sigma: Full covariance matrix of shape (B, y_dim, y_dim).
        """
        x_ = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            params = self.forward(x_)

        if self.mode == "low_rank":
            mu, D, V = params
            # Sigma = D + V @ V^T
            # Compute low rank part: V @ V^T
            Sigma = torch.bmm(V, V.transpose(1, 2))

            # Add diagonal D efficiently
            # We create an index for the diagonal to avoid creating a full diagonal matrix first
            diag_indices = torch.arange(self.y_dim, device=x_.device)
            Sigma[:, diag_indices, diag_indices] += D
        else:
            mu, L = params
            # Sigma = L @ L^T
            Sigma = torch.bmm(L, L.transpose(1, 2))

        return mu.detach().cpu().numpy(), Sigma.detach().cpu().numpy()

    def get_covariance_matrix(self, x: Union[NDArray, torch.Tensor]) -> NDArray:
        """
        Predicts only the covariance matrix for a given set of inputs.

        Parameters
        ----------
        x : Union[NDArray, torch.Tensor]
            Input features of shape (B, input_dim).

        Returns
        -------
        NDArray
            Full covariance matrix of shape (B, y_dim, y_dim).
        """
        _, Sigma = self.get_distribution(x)
        return Sigma

    def _compute_mahalanobis_low_rank(
        self, y: torch.Tensor, mu: torch.Tensor, D: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper method: Computes the Mahalanobis distance using the Woodbury Matrix Identity.

        Parameters
        ----------
        y : torch.Tensor
            Target tensor of shape (B, y_dim).
        mu : torch.Tensor
            Predicted mean tensor of shape (B, y_dim).
        D : torch.Tensor
            Diagonal variance components.
        V : torch.Tensor
            Low-rank covariance components.

        Returns
        -------
        torch.Tensor
            A 1D tensor of length B containing the computed distance scores.
        """
        r = y - mu
        inv_std = 1.0 / D
        W = V * inv_std.unsqueeze(-1)  # (B, Y, Rank)
        z = r * inv_std  # (B, Y)

        # M = I + W^T W
        B, Y, K = V.shape
        M = torch.eye(K, device=V.device).unsqueeze(0) + torch.bmm(W.transpose(1, 2), W)
        L_M = torch.linalg.cholesky(M)

        # Term 1: z^T z
        z_sq = torch.sum(z**2, dim=1)

        # Term 2: z^T V (I + V^T D^-2 V)^-1 V^T z
        # Let p = W^T z
        p = torch.bmm(W.transpose(1, 2), z.unsqueeze(-1))  # (B, Rank, 1)

        # Solve M q = p  => q = M^-1 p
        # q = torch.linalg.cholesky_solve(L_M, p)
        q = torch.cholesky_solve(p, L_M)

        # quad = p^T q
        quad = torch.bmm(p.transpose(1, 2), q).squeeze(-1).squeeze(-1)

        return torch.sqrt(z_sq - quad)

    def _compute_mahalanobis_full_chol(
        self, y: torch.Tensor, mu: torch.Tensor, L: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper method: Computes the Mahalanobis distance using a triangular solve.

        Parameters
        ----------
        y : torch.Tensor
            Target tensor of shape (B, y_dim).
        mu : torch.Tensor
            Predicted mean tensor of shape (B, y_dim).
        L : torch.Tensor
            Lower triangular Cholesky factor of the covariance matrix.

        Returns
        -------
        torch.Tensor
            A 1D tensor of length B containing the computed distance scores.
        """
        diff = (y - mu).unsqueeze(-1)
        # Solve L z = (y - mu)
        z = torch.linalg.solve_triangular(L, diff, upper=False)
        return torch.sqrt(torch.sum(z.squeeze(-1) ** 2, dim=1))

    def predict(self, x: Union[NDArray, torch.Tensor]) -> NDArray:
        """
        Returns the predictions y_pred for the associated X values.

        Parameters
        ----------
        X : ArrayLike
            The input feature values.

        Returns
        -------
        NDArray
            An array y_pred of the prediction of the model.
        """
        x_ = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        return self.forward(x_)[0].detach().cpu().numpy()

    def get_standardized_score(
        self, x: Union[NDArray, torch.Tensor], y: Union[NDArray, torch.Tensor]
    ) -> NDArray:
        """
        Calculates the standardized score (Mahalanobis distance) for a set of predictions.

        Parameters
        ----------
        x : Union[NDArray, torch.Tensor]
            Input features of shape (B, input_dim).
        y : Union[NDArray, torch.Tensor]
            Target outputs of shape (B, y_dim).

        Returns
        -------
        NDArray
            A 1D array of shape (B,) containing the standardized scores.
        """
        x_ = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        y_ = torch.as_tensor(y, dtype=self.dtype, device=self.device)
        params = self.forward(x_)

        if self.mode == "low_rank":
            mu, D, V = params
            return (
                self._compute_mahalanobis_low_rank(y_, mu, D, V).detach().cpu().numpy()
            )
        else:
            mu, L = params
            return self._compute_mahalanobis_full_chol(y_, mu, L).detach().cpu().numpy()

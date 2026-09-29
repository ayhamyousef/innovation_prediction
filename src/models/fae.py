"""
Fractal Autoencoder (FAE) for unsupervised feature selection.

Based on: "Fractal Autoencoders for Feature Selection"
by Xinxing Wu & Qiang Cheng (AAAI 2021).

This implements the linear version of FAE (Section 4.1 of the paper),
the variant used in this study.

Architecture:
    Input X (N, m) -> diagonal importance weights W_I (m,)
    Global path:  X * W_I -> Encoder W_E (m->k) -> Decoder W_D (k->m) -> X_hat
    Sub-NN path:  X * W_I^maxk -> same Encoder -> same Decoder -> X_hat_sub

    W_I^maxk = W_I with only the top-K largest weights kept, rest zeroed.

Loss (Equation 3):
    L = ||X - f(g(X * W_I))||_F^2
        + lambda1 * ||X - f(g(X * W_I^maxk))||_F^2
        + lambda2 * ||W_I||_1
    subject to W_I >= 0
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("innovation_prediction.fae")


class FractalAutoencoder(nn.Module):
    """
    Linear Fractal Autoencoder for feature selection.

    Args:
        input_dim: number of input features (m)
        k: number of features to select (bottleneck dimension)
        lambda1: weight for sub-network reconstruction loss (default 2.0)
        lambda2: weight for L1 sparsity on W_I (default 0.1)
    """

    def __init__(self, input_dim: int, k: int,
                 lambda1: float = 2.0, lambda2: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Diagonal importance weights W_I (m,)
        # Initialized near 1.0 per paper: Uniform[0.999999, 0.9999999]
        wi_init = torch.empty(input_dim).uniform_(0.999999, 0.9999999)
        self.w_i = nn.Parameter(wi_init)

        # Linear encoder: m -> k
        self.encoder = nn.Linear(input_dim, k, bias=False)
        # Linear decoder: k -> m
        self.decoder = nn.Linear(k, input_dim, bias=False)

        # Xavier normal initialization for encoder/decoder
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.xavier_normal_(self.decoder.weight)

    def _apply_importance(self, x: torch.Tensor,
                          w: torch.Tensor) -> torch.Tensor:
        """Apply diagonal importance weights: X * W_I."""
        return x * w.unsqueeze(0)  # broadcast (N, m) * (1, m)

    def _get_topk_mask(self) -> torch.Tensor:
        """
        Create mask for top-K importance weights (W_I^maxk operation).
        Returns binary mask of shape (m,) with 1s at top-K positions.
        """
        _, topk_indices = torch.topk(self.w_i.detach(), self.k)
        mask = torch.zeros_like(self.w_i)
        mask[topk_indices] = 1.0
        return mask

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing both reconstruction paths.

        Args:
            x: input features (N, m)

        Returns:
            dict with x_hat_global, x_hat_sub, w_i, topk_mask
        """
        # Global path: use all importance weights
        x_weighted = self._apply_importance(x, self.w_i)
        encoded = self.encoder(x_weighted)
        x_hat_global = self.decoder(encoded)

        # Sub-NN path: use only top-K importance weights
        topk_mask = self._get_topk_mask()
        w_i_masked = self.w_i * topk_mask
        x_weighted_sub = self._apply_importance(x, w_i_masked)
        encoded_sub = self.encoder(x_weighted_sub)
        x_hat_sub = self.decoder(encoded_sub)

        return {
            "x_hat_global": x_hat_global,
            "x_hat_sub": x_hat_sub,
            "w_i": self.w_i,
            "topk_mask": topk_mask,
        }

    def compute_loss(self, x: torch.Tensor,
                     outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute FAE loss (Equation 3 from paper).

        Returns dict with total_loss, recon_global, recon_sub, sparsity.
        """
        # Term 1: Global reconstruction loss
        recon_global = torch.norm(x - outputs["x_hat_global"], p="fro") ** 2

        # Term 2: Sub-network reconstruction loss (top-K only)
        recon_sub = torch.norm(x - outputs["x_hat_sub"], p="fro") ** 2

        # Term 3: L1 sparsity on importance weights
        sparsity = self.w_i.abs().sum()

        total_loss = recon_global + self.lambda1 * recon_sub + self.lambda2 * sparsity

        return {
            "total_loss": total_loss,
            "recon_global": recon_global,
            "recon_sub": recon_sub,
            "sparsity": sparsity,
        }

    def clamp_weights(self):
        """Enforce W_I >= 0 constraint after optimizer step."""
        with torch.no_grad():
            self.w_i.clamp_(min=0.0)

    def get_selected_features(self, feature_names: Optional[List[str]] = None
                              ) -> Dict:
        """
        Get the top-K selected features and their importance weights.

        Returns:
            dict with selected_indices, selected_names (if provided),
            all_weights, and ranking.
        """
        weights = self.w_i.detach().cpu().numpy()
        topk_vals, topk_indices = torch.topk(self.w_i.detach().cpu(), self.k)

        indices = topk_indices.numpy().tolist()
        result = {
            "selected_indices": indices,
            "selected_weights": topk_vals.numpy().tolist(),
            "all_weights": weights.tolist(),
            "ranking": np.argsort(-weights).tolist(),
        }

        if feature_names is not None:
            result["selected_names"] = [feature_names[i] for i in indices]
            result["all_feature_ranking"] = [
                (feature_names[i], float(weights[i]))
                for i in np.argsort(-weights)
            ]

        return result


def train_fae(X: np.ndarray, k: int,
              feature_names: Optional[List[str]] = None,
              lambda1: float = 2.0, lambda2: float = 0.1,
              lr: float = 0.001, max_epochs: int = 1000,
              patience: int = 50, min_delta: float = 1e-6,
              device: str = "cpu",
              verbose: bool = True) -> Tuple[FractalAutoencoder, Dict]:
    """
    Train a linear FAE model for feature selection.

    Args:
        X: input data (N, m) as numpy array
        k: number of features to select
        feature_names: optional list of feature names
        lambda1: sub-network reconstruction weight (paper default: 2.0)
        lambda2: L1 sparsity weight (paper default: 0.1)
        lr: learning rate (paper default: 0.001)
        max_epochs: maximum training epochs
        patience: early stopping patience
        min_delta: minimum improvement for early stopping
        device: "cpu" or "cuda"
        verbose: print progress

    Returns:
        (trained_model, results_dict)
    """
    m = X.shape[1]
    assert k < m, f"k ({k}) must be less than number of features ({m})"

    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Initialize model
    model = FractalAutoencoder(input_dim=m, k=k,
                                lambda1=lambda1, lambda2=lambda2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_loss = float("inf")
    patience_counter = 0
    history = {"total_loss": [], "recon_global": [], "recon_sub": [], "sparsity": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_tensor)
        losses = model.compute_loss(X_tensor, outputs)
        losses["total_loss"].backward()
        optimizer.step()

        # Enforce W_I >= 0
        model.clamp_weights()

        # Record
        total = losses["total_loss"].item()
        history["total_loss"].append(total)
        history["recon_global"].append(losses["recon_global"].item())
        history["recon_sub"].append(losses["recon_sub"].item())
        history["sparsity"].append(losses["sparsity"].item())

        # Early stopping
        if total < best_loss - min_delta:
            best_loss = total
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                logger.info(f"FAE converged at epoch {epoch} "
                            f"(patience={patience})")
            break

        if verbose and epoch % 100 == 0:
            w = model.w_i.detach().cpu().numpy()
            logger.info(
                f"Epoch {epoch:4d} | Loss: {total:.6f} | "
                f"Recon: {losses['recon_global'].item():.6f} | "
                f"Sub: {losses['recon_sub'].item():.6f} | "
                f"W_I range: [{w.min():.4f}, {w.max():.4f}]"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Get results
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_tensor)
        final_losses = model.compute_loss(X_tensor, final_outputs)

    selection = model.get_selected_features(feature_names)

    results = {
        "k": k,
        "input_dim": m,
        "n_samples": X.shape[0],
        "lambda1": lambda1,
        "lambda2": lambda2,
        "final_loss": final_losses["total_loss"].item(),
        "final_recon_global": final_losses["recon_global"].item(),
        "final_recon_sub": final_losses["recon_sub"].item(),
        "final_sparsity": final_losses["sparsity"].item(),
        "epochs_trained": epoch,
        "selection": selection,
        "history": history,
    }

    if verbose:
        logger.info(f"\nFAE Results (K={k}):")
        logger.info(f"  Final loss: {results['final_loss']:.6f}")
        logger.info(f"  Recon (global): {results['final_recon_global']:.6f}")
        logger.info(f"  Recon (sub-NN): {results['final_recon_sub']:.6f}")
        if feature_names:
            logger.info(f"  Selected features: {selection['selected_names']}")
            logger.info(f"  Full ranking:")
            for name, weight in selection["all_feature_ranking"]:
                marker = " <-- selected" if name in selection["selected_names"] else ""
                logger.info(f"    {name}: {weight:.6f}{marker}")

    return model, results

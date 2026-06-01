"""
Tabular deep learning models for technology trajectory classification.

Implements wrappers for three architecturally distinct approaches:
  1. TabNet (CNN/attention-based, instance-wise feature selection)
  2. TabM (Ensemble of MLPs with batch ensembling)
  3. FT-Transformer (Feature Tokenizer + Transformer encoder)

All models share a common interface:
  - fit(X_train, y_train, X_val, y_val)
  - predict(X)
  - predict_proba(X)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("innovation_prediction.tabular_models")


# ============================================================
# TabNet
# ============================================================

class TabNetWrapper:
    """
    TabNet: Attentive Interpretable Tabular Learning (Arik & Pfister, AAAI 2021).
    Uses pytorch-tabnet package.
    """

    def __init__(self, n_classes: int, device: str = "auto",
                 n_d: int = 16, n_a: int = 16,
                 n_steps: int = 5, gamma: float = 1.5,
                 lr: float = 0.02, max_epochs: int = 200,
                 patience: int = 20, batch_size: int = 256,
                 class_weights: Optional[Dict] = None,
                 seed: int = 42):
        self.n_classes = n_classes
        self.device = device
        self.params = {
            "n_d": n_d,
            "n_a": n_a,
            "n_steps": n_steps,
            "gamma": gamma,
            "seed": seed,
            "verbose": 1,
        }
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        from pytorch_tabnet.tab_model import TabNetClassifier

        self.model = TabNetClassifier(
            **self.params,
            optimizer_params={"lr": self.lr},
            device_name=self.device if self.device != "auto" else "cpu",
        )

        fit_kwargs = dict(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["accuracy"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
        )
        if self.class_weights:
            fit_kwargs["weights"] = self.class_weights

        self.model.fit(**fit_kwargs)
        logger.info(f"TabNet training complete. "
                     f"Best epoch: {self.model.best_epoch}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ============================================================
# FT-Transformer
# ============================================================

class FTTransformerWrapper:
    """
    FT-Transformer: Feature Tokenizer + Transformer
    (Gorishniy et al., NeurIPS 2021).

    Uses rtdl-revisiting-models or a minimal PyTorch implementation.
    """

    def __init__(self, n_features: int, n_classes: int,
                 d_token: int = 64, n_blocks: int = 3,
                 attention_n_heads: int = 4,
                 attention_dropout: float = 0.2,
                 ffn_d_hidden_multiplier: float = 4.0 / 3.0,
                 ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0,
                 lr: float = 1e-4, weight_decay: float = 1e-5,
                 max_epochs: int = 200, patience: int = 20,
                 batch_size: int = 256, seed: int = 42,
                 device: str = "cpu"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_d_hidden_multiplier = ffn_d_hidden_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.model = None
        self._scaler = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.seed)
        device = torch.device(self.device)

        # Build model
        self.model = _FTTransformerModel(
            n_features=self.n_features,
            n_classes=self.n_classes,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_n_heads=self.attention_n_heads,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden_multiplier=self.ffn_d_hidden_multiplier,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Data loaders
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            val_loss = 0
            n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = self.model(xb)
                    val_loss += criterion(logits, yb).item() * len(yb)
                    n += len(yb)
            val_loss /= n

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                logger.info(f"FT-Transformer epoch {epoch}: "
                            f"val_loss={val_loss:.4f}")

            if patience_counter >= self.patience:
                logger.info(f"FT-Transformer early stop at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info(f"FT-Transformer training complete. "
                     f"Best val_loss={best_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = torch.device(self.device)
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[i:i+self.batch_size], dtype=torch.float32).to(device)
                logits = self.model(xb)
                all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(all_proba, axis=0)


class _FTTransformerModel(object):
    """Minimal FT-Transformer implementation in PyTorch."""
    pass  # Replaced below with actual implementation


import torch
import torch.nn as nn
import math


class _FTTransformerModel(nn.Module):
    """
    Minimal FT-Transformer: each numerical feature gets a learned token
    embedding, then standard Transformer encoder, then [CLS] → classifier.
    """

    def __init__(self, n_features: int, n_classes: int,
                 d_token: int = 64, n_blocks: int = 3,
                 attention_n_heads: int = 4,
                 attention_dropout: float = 0.2,
                 ffn_d_hidden_multiplier: float = 4.0 / 3.0,
                 ffn_dropout: float = 0.1,
                 residual_dropout: float = 0.0):
        super().__init__()

        # Feature tokenizer: each feature gets its own linear → d_token
        self.feature_tokenizer = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(n_features)
        ])

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_n_heads,
            dim_feedforward=int(d_token * ffn_d_hidden_multiplier),
            dropout=ffn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_blocks
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_features) numerical features

        Returns:
            logits: (B, n_classes)
        """
        B = x.shape[0]

        # Tokenize each feature independently
        tokens = []
        for i, tokenizer in enumerate(self.feature_tokenizer):
            tokens.append(tokenizer(x[:, i:i+1]))  # (B, d_token)
        tokens = torch.stack(tokens, dim=1)  # (B, n_features, d_token)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_features+1, d_token)

        # Transformer
        out = self.transformer(tokens)  # (B, n_features+1, d_token)

        # [CLS] output → classification
        cls_out = out[:, 0, :]  # (B, d_token)
        return self.head(cls_out)


# ============================================================
# TabM (Ensemble of MLPs with batch ensembling)
# ============================================================

class TabMWrapper:
    """
    TabM: Parameter-Efficient Ensembling of MLPs
    (Gorishniy et al., ICLR 2025).

    Minimal implementation using batch ensembling: single MLP backbone
    with per-ensemble-member scaling vectors (r_i, s_i).
    """

    def __init__(self, n_features: int, n_classes: int,
                 n_ensemble: int = 8,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 max_epochs: int = 200, patience: int = 20,
                 batch_size: int = 256, seed: int = 42,
                 device: str = "cpu"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_ensemble = n_ensemble
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.seed)
        device = torch.device(self.device)

        self.model = _TabMModel(
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_ensemble=self.n_ensemble,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)  # (B, n_classes)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            val_loss = 0
            n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = self.model(xb)
                    val_loss += criterion(logits, yb).item() * len(yb)
                    n += len(yb)
            val_loss /= n

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                logger.info(f"TabM epoch {epoch}: val_loss={val_loss:.4f}")

            if patience_counter >= self.patience:
                logger.info(f"TabM early stop at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info(f"TabM training complete. Best val_loss={best_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = torch.device(self.device)
        self.model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(device)
            logits = self.model(xt)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba


class _TabMModel(nn.Module):
    """
    TabM with batch ensembling: shared MLP backbone with per-member
    scaling vectors at input and output of each layer.

    Each ensemble member m has vectors r_m (input scaling) and s_m
    (output scaling) per layer, applied as element-wise multiplication.
    Final prediction averages logits across ensemble members.
    """

    def __init__(self, n_features: int, n_classes: int,
                 n_ensemble: int = 8,
                 hidden_dims: List[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.n_ensemble = n_ensemble

        hidden_dims = hidden_dims or [128, 64]
        dims = [n_features] + hidden_dims

        self.layers = nn.ModuleList()
        self.r_vectors = nn.ParameterList()  # input scaling per member
        self.s_vectors = nn.ParameterList()  # output scaling per member

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            # r: (n_ensemble, dims[i]), s: (n_ensemble, dims[i+1])
            self.r_vectors.append(
                nn.Parameter(torch.randn(n_ensemble, dims[i]) * 0.1 + 1.0)
            )
            self.s_vectors.append(
                nn.Parameter(torch.randn(n_ensemble, dims[i + 1]) * 0.1 + 1.0)
            )

        self.head = nn.Linear(hidden_dims[-1], n_classes)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_features)
        Returns:
            logits: (B, n_classes) — averaged across ensemble members
        """
        B = x.shape[0]

        # Expand x for ensemble: (B, 1, n_features) → broadcast with (1, M, n_features)
        all_logits = []

        for m in range(self.n_ensemble):
            h = x
            for i, layer in enumerate(self.layers):
                # Input scaling: h * r_m
                h = h * self.r_vectors[i][m].unsqueeze(0)  # (B, d_in)
                h = layer(h)  # (B, d_out)
                # Output scaling: h * s_m
                h = h * self.s_vectors[i][m].unsqueeze(0)  # (B, d_out)
                h = self.act(h)
                h = self.dropout(h)

            logits_m = self.head(h)  # (B, n_classes)
            all_logits.append(logits_m)

        # Average across ensemble members
        stacked = torch.stack(all_logits, dim=0)  # (M, B, n_classes)
        return stacked.mean(dim=0)  # (B, n_classes)


# ============================================================
# TabKAN (Kolmogorov-Arnold mixer for tabular)
# ============================================================

class TabKANWrapper:
    """TabKAN: KAN-flavored mixer for tabular data.

    Uses ChebyshevKANMixer from the tabkan package (Eslamian et al., 2025,
    https://arxiv.org/abs/2504.06559). Wrapped in our standard train loop
    so it composes with the rest of the pipeline.
    """

    def __init__(self, n_features: int, n_classes: int,
                 num_layers: int = 4,
                 token_dim: int = 64, channel_dim: int = 128,
                 token_order: int = 3, channel_order: int = 3,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 max_epochs: int = 200, patience: int = 20,
                 batch_size: int = 256, seed: int = 42,
                 device: str = "cpu"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.token_dim = token_dim
        self.channel_dim = channel_dim
        self.token_order = token_order
        self.channel_order = channel_order
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        try:
            from tabkan import KANMixer
            from tabkan.chebyshev.model import ChebyKANLayer
        except ImportError as e:
            raise ImportError(
                "tabkan package not installed. Run: pip install tabkan"
            ) from e

        torch.manual_seed(self.seed)
        device = torch.device(self.device)

        # KANMixer takes a per-layer KAN class. ChebyKANLayer is the single-
        # layer Chebyshev KAN; its constructor is (input_dim, output_dim, degree),
        # so we pass degree as the extra kwarg.
        self.model = KANMixer(
            num_features=self.n_features,
            num_classes=self.n_classes,
            kan_layer_class=ChebyKANLayer,
            num_layers=self.num_layers,
            token_dim=self.token_dim,
            channel_dim=self.channel_dim,
            token_kan_kwargs={"degree": self.token_order},
            channel_kan_kwargs={"degree": self.channel_order},
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(self.model(xb), yb).item() * len(xb)
                    n_val += len(xb)
            val_loss /= max(n_val, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                logger.info(f"TabKAN epoch {epoch}: val_loss={val_loss:.4f}")

            if patience_counter >= self.patience:
                logger.info(f"TabKAN early stop at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info(f"TabKAN training complete. "
                    f"Best val_loss={best_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = torch.device(self.device)
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[i:i+self.batch_size],
                                  dtype=torch.float32).to(device)
                logits = self.model(xb)
                all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(all_proba, axis=0)


# ============================================================
# TabMixer (MLP-Mixer variant for tabular)
# ============================================================

class TabMixerWrapper:
    """TabMixer: MLP-Mixer architecture for tabular data.

    Uses TabMixer from the TabMixer package (Eslamian et al., 2024,
    https://arxiv.org/abs/2409.07564). The base TabMixer expects
    (B, dim_tokens, dim_features) input, so we wrap it with a per-feature
    embedding layer and a classification head.
    """

    def __init__(self, n_features: int, n_classes: int,
                 dim_features: int = 64,
                 dim_feedforward: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 max_epochs: int = 200, patience: int = 20,
                 batch_size: int = 256, seed: int = 42,
                 device: str = "cpu"):
        self.n_features = n_features
        self.n_classes = n_classes
        self.dim_features = dim_features
        self.dim_feedforward = dim_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.seed)
        device = torch.device(self.device)

        self.model = _TabMixerModel(
            n_features=self.n_features,
            n_classes=self.n_classes,
            dim_features=self.dim_features,
            dim_feedforward=self.dim_feedforward,
        ).to(device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                   shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(self.model(xb), yb).item() * len(xb)
                    n_val += len(xb)
            val_loss /= max(n_val, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 20 == 0:
                logger.info(f"TabMixer epoch {epoch}: val_loss={val_loss:.4f}")

            if patience_counter >= self.patience:
                logger.info(f"TabMixer early stop at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
        self.model.eval()
        logger.info(f"TabMixer training complete. "
                    f"Best val_loss={best_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        device = torch.device(self.device)
        self.model.eval()
        all_proba = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = torch.tensor(X[i:i+self.batch_size],
                                  dtype=torch.float32).to(device)
                logits = self.model(xb)
                all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(all_proba, axis=0)


class _TabMixerModel(nn.Module):
    """Wraps the upstream TabMixer module with per-feature embedding and
    a classification head. The TabMixer block itself expects
    (B, dim_tokens, dim_features); we set dim_tokens = n_features.
    """

    def __init__(self, n_features: int, n_classes: int,
                 dim_features: int = 64, dim_feedforward: int = 256):
        super().__init__()
        try:
            from tabmixer import TabMixer
        except ImportError as e:
            raise ImportError(
                "TabMixer package not installed. "
                "Run: pip install TabMixer"
            ) from e

        # Per-feature embedding: each scalar feature -> dim_features vector
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, dim_features) for _ in range(n_features)
        ])

        self.mixer = TabMixer(
            dim_tokens=n_features,
            dim_features=dim_features,
            dim_feedforward=dim_feedforward,
        )

        # Pool across tokens then classify
        self.head = nn.Sequential(
            nn.LayerNorm(dim_features),
            nn.Linear(dim_features, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_features). Returns logits (B, n_classes)."""
        B = x.shape[0]
        tokens = [emb(x[:, i:i+1]) for i, emb in enumerate(self.feature_embeddings)]
        tokens = torch.stack(tokens, dim=1)  # (B, n_features, dim_features)
        mixed = self.mixer(tokens)  # (B, n_features, dim_features)
        pooled = mixed.mean(dim=1)  # (B, dim_features)
        return self.head(pooled)


# ============================================================
# Factory
# ============================================================

class ExtraTreesWrapper:
    """Extremely Randomized Trees, sklearn implementation.

    Used by Wu & Cheng (FAE paper, AAAI 2021) as the downstream classifier
    for evaluating selected feature subsets.
    """

    def __init__(self, n_classes: int, device: str = "cpu", seed: int = 42,
                 n_estimators: int = 100, n_jobs: int = -1, **kwargs):
        from sklearn.ensemble import ExtraTreesClassifier
        self.n_classes = n_classes
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class GBDTWrapper:
    """Gradient-Boosted Decision Trees, sklearn implementation.

    Required for comparison against Chen et al. (Scientometrics 2025), who
    used GBDT as their best classification model.
    """

    def __init__(self, n_classes: int, device: str = "cpu", seed: int = 42,
                 n_estimators: int = 200, max_depth: int = 5,
                 learning_rate: float = 0.1, **kwargs):
        from sklearn.ensemble import GradientBoostingClassifier
        self.n_classes = n_classes
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=seed,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def build_tabular_model(model_name: str, n_features: int, n_classes: int,
                        device: str = "cpu", seed: int = 42,
                        **kwargs) -> object:
    """
    Factory function to create tabular model by name.

    Args:
        model_name: "tabnet", "tabm", "ft_transformer", "extra_trees", or "gbdt"
        n_features: number of input features
        n_classes: number of output classes
        device: "cpu" or "cuda"
        seed: random seed
    """
    if model_name == "tabnet":
        return TabNetWrapper(
            n_classes=n_classes, device=device, seed=seed, **kwargs
        )
    elif model_name == "tabm":
        return TabMWrapper(
            n_features=n_features, n_classes=n_classes,
            device=device, seed=seed, **kwargs
        )
    elif model_name == "ft_transformer":
        return FTTransformerWrapper(
            n_features=n_features, n_classes=n_classes,
            device=device, seed=seed, **kwargs
        )
    elif model_name == "extra_trees":
        return ExtraTreesWrapper(
            n_classes=n_classes, device=device, seed=seed, **kwargs
        )
    elif model_name == "gbdt":
        return GBDTWrapper(
            n_classes=n_classes, device=device, seed=seed, **kwargs
        )
    elif model_name == "tabkan":
        return TabKANWrapper(
            n_features=n_features, n_classes=n_classes,
            device=device, seed=seed, **kwargs
        )
    elif model_name == "tabmixer":
        return TabMixerWrapper(
            n_features=n_features, n_classes=n_classes,
            device=device, seed=seed, **kwargs
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: tabnet, tabm, ft_transformer, "
                         f"extra_trees, gbdt, tabkan, tabmixer")

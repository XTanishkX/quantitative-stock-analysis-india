"""
================================================================================
PHASE 3: ADVANCED ML & DEEP LEARNING ENGINES
================================================================================
Implementations:
    1. QuantModel        — Abstract base class for all quant models
    2. PatchTSTModel     — Time-Series Transformer with Bayesian Dropout
    3. NiftyGNNModel     — Spatiotemporal Graph Attention Network (GAT)
    4. OrthogXGBModel    — Factor-orthogonalized XGBoost alpha predictor

Dependencies: torch, torch_geometric, xgboost, numpy, sklearn
================================================================================
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("DeepQuant")


# ──────────────────────────────────────────────────────────────────────────────
# BASE MODEL ABC
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    predictions: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None   # Epistemic uncertainty (std)
    attention_weights: Optional[torch.Tensor] = None


class QuantModel(ABC):
    """
    Abstract base class for all institutional quantitative models.
    Enforces a consistent interface across ML, DL, and RL paradigms.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "QuantModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelOutput: ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> "QuantModel": ...

    def _to_tensor(self, arr: np.ndarray, device: str = "cpu") -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# 1. PatchTST — TIME-SERIES TRANSFORMER
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Segment time series into non-overlapping patches and project to d_model.

    Patch approach (Nie et al., 2023 — PatchTST):
        - Divides length-L context window into patches of size P.
        - N_patches = floor((L - P) / stride) + 1
        - Each patch is linearly projected to d_model.
        - Channel-independence: each variate is processed independently.

    This dramatically reduces the quadratic attention cost and lets the
    transformer focus on local temporal structure.
    """

    def __init__(self, patch_size: int, stride: int, d_model: int, n_variates: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_variates = n_variates
        # One linear projection per variate (channel independence)
        self.projection = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — batch, time, variates
        Returns:
            patches: (B * C, N_patches + 1, d_model)  [+ CLS token]
        """
        B, T, C = x.shape
        # Unfold → (B, C, N_patches, patch_size)
        x = x.permute(0, 2, 1)  # (B, C, T)
        patches = x.unfold(dimension=2, size=self.patch_size, step=self.stride)
        B, C, N, P = patches.shape
        patches = patches.reshape(B * C, N, P)         # (B*C, N, P)
        patches = self.projection(patches)              # (B*C, N, d_model)

        # Prepend CLS token for aggregation
        cls = self.cls_token.expand(B * C, -1, -1)    # (B*C, 1, d_model)
        patches = torch.cat([cls, patches], dim=1)     # (B*C, N+1, d_model)
        return patches, B, C, N


class BayesianDropout(nn.Module):
    """
    MC Dropout for epistemic uncertainty estimation (Gal & Ghahramani, 2016).
    Kept active during inference — multiple forward passes yield a predictive
    distribution, from which we extract mean and variance.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always active — training=True forces dropout even at eval time
        return F.dropout(x, p=self.p, training=True)


class PatchTSTBlock(nn.Module):
    """Single Transformer encoder block with pre-LayerNorm and Bayesian Dropout."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            BayesianDropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = BayesianDropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_out)
        # Pre-LN feed-forward
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, attn_weights


class PatchTSTHead(nn.Module):
    """
    Prediction head: projects from CLS token representation to
    multi-horizon return distributions (mean only; uncertainty from MC Dropout).
    """

    def __init__(self, d_model: int, n_variates: int, horizons: list[int]) -> None:
        super().__init__()
        self.horizons = horizons
        # Separate linear head per horizon for multi-horizon forecasting
        self.heads = nn.ModuleList([
            nn.Linear(d_model * n_variates, 1) for _ in horizons
        ])

    def forward(self, cls_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_tokens: (B, C, d_model) — CLS token per variate per batch
        Returns:
            preds: (B, len(horizons))
        """
        B, C, D = cls_tokens.shape
        flat = cls_tokens.reshape(B, C * D)  # (B, C * d_model)
        return torch.stack([head(flat).squeeze(-1) for head in self.heads], dim=1)


class PatchTSTModel(QuantModel, nn.Module):
    """
    PatchTST: Patch-based Time-Series Transformer for multi-horizon return forecasting.

    Architecture:
        Input (B, T, C)
          → PatchEmbedding (B*C, N+1, d_model)
          → L × TransformerEncoderBlock
          → Extract CLS token (B, C, d_model)
          → Multi-horizon prediction head (B, H)

    Epistemic uncertainty via MC Dropout:
        Run N forward passes → mean prediction + std (uncertainty).

    Args:
        n_variates:   Number of input features / asset return series (C).
        seq_len:      Input context window length (T).
        patch_size:   Number of time steps per patch (P).
        stride:       Stride for patch extraction.
        d_model:      Transformer embedding dimension.
        n_heads:      Multi-head attention heads.
        d_ff:         Feed-forward hidden dimension.
        n_layers:     Number of transformer encoder layers.
        dropout:      Dropout probability (also used for MC Dropout at inference).
        horizons:     List of forward return horizons to predict (e.g., [1, 5, 15] minutes).
        mc_samples:   Number of MC Dropout forward passes for uncertainty.
    """

    def __init__(
        self,
        n_variates: int = 10,
        seq_len: int = 128,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
        horizons: list[int] | None = None,
        mc_samples: int = 30,
        device: str = "cpu",
    ) -> None:
        nn.Module.__init__(self)
        self.n_variates = n_variates
        self.seq_len = seq_len
        self.horizons = horizons or [1, 5, 15]
        self.mc_samples = mc_samples
        self.device = device

        self.patch_embed = PatchEmbedding(patch_size, stride, d_model, n_variates)
        n_patches = (seq_len - patch_size) // stride + 1

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)

        self.encoder = nn.ModuleList([
            PatchTSTBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = PatchTSTHead(d_model, n_variates, self.horizons)
        self.to(device)

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass through PatchTST."""
        patches, B, C, N = self.patch_embed(x)          # (B*C, N+1, d_model)
        patches = patches + self.pos_embed[:, : N + 1, :]

        attn_weights_list = []
        for block in self.encoder:
            patches, attn_w = block(patches)
            attn_weights_list.append(attn_w)

        patches = self.norm(patches)
        cls_tokens = patches[:, 0, :]                   # (B*C, d_model) — CLS token
        cls_tokens = cls_tokens.reshape(B, C, -1)       # (B, C, d_model)
        return self.head(cls_tokens)                     # (B, H)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        MC Dropout inference: run mc_samples forward passes, compute mean and std.

        Args:
            x: (B, T, C) input tensor.

        Returns:
            ModelOutput with predictions (B, H) and uncertainty (B, H).
        """
        self.train()  # Activate dropout for MC sampling
        samples = torch.stack([self._forward_once(x) for _ in range(self.mc_samples)], dim=0)
        self.eval()

        mean_pred = samples.mean(dim=0)      # (B, H)
        uncertainty = samples.std(dim=0)     # (B, H) — epistemic uncertainty
        return ModelOutput(predictions=mean_pred, uncertainty=uncertainty)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "PatchTSTModel":
        """
        Training loop stub. In production, use a DataLoader and proper trainer.

        Args:
            X: (N, T, C) training inputs.
            y: (N, H) training targets (multi-horizon returns).
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, steps_per_epoch=len(X), epochs=kwargs.get("epochs", 10)
        )
        criterion = nn.HuberLoss(delta=0.01)

        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)

        self.train()
        for epoch in range(kwargs.get("epochs", 10)):
            self.train()
            preds = self._forward_once(x_t)
            loss = criterion(preds, y_t)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            logger.info(f"  [PatchTST] Epoch {epoch+1} | Loss: {loss.item():.6f}")
        return self

    def predict(self, X: np.ndarray) -> ModelOutput:
        x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.forward(x_t)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info(f"[PatchTST] Model saved → {path}")

    def load(self, path: str) -> "PatchTSTModel":
        self.load_state_dict(torch.load(path, map_location=self.device))
        return self


# ──────────────────────────────────────────────────────────────────────────────
# 2. SPATIOTEMPORAL GRAPH ATTENTION NETWORK
# ──────────────────────────────────────────────────────────────────────────────

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning("[NiftyGNN] torch_geometric not found. GNN model disabled.")


class TemporalEncoder(nn.Module):
    """
    Per-node temporal feature extractor using a stacked GRU.
    Encodes the T-step return history of each asset into a fixed-size embedding.
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_nodes, T, input_size)
        Returns:
            h: (N_nodes, hidden_size) — last hidden state
        """
        _, h_n = self.gru(x)
        return h_n[-1]  # Take last layer hidden state


class NiftyGATLayer(nn.Module):
    """
    GATv2 (Brody et al., 2022) attention layer: more expressive than GAT v1.
    Attention is dynamic — it depends on both source and target features.

        e_{ij} = LeakyReLU(a^T [W h_i || W h_j])   (GATv2)
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int, dropout: float) -> None:
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric required for NiftyGNNModel.")
        self.conv = GATv2Conv(in_channels, out_channels, heads=heads, dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_channels * heads)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        out = self.conv(x, edge_index, edge_attr=edge_attr)
        return self.act(self.norm(out))


class NiftyGNNModel(QuantModel, nn.Module):
    """
    Spatiotemporal Graph Neural Network for cross-asset spillover prediction.

    Architecture:
        1. Temporal Encoding: GRU per node encodes T-step return history → h_i ∈ R^{d_h}
        2. Graph Construction: Edges defined by rolling Pearson correlation (|ρ| > threshold).
           Edge weights = |ρ_{ij}|.
        3. GATv2 Layers: 2 stacked layers with multi-head attention.
        4. Prediction Head: Linear → softmax (classification) or linear (regression).

    The graph structure captures cross-asset momentum spillover: a shock to one
    sector propagates through correlated edges to related assets.

    Args:
        n_assets:      Number of nodes in the graph (e.g., 50 for Nifty 50).
        seq_len:       Historical window fed to temporal encoder.
        n_features:    Per-asset feature dimension (e.g., returns, volume, OBI).
        gru_hidden:    GRU hidden state size.
        gat_hidden:    GAT hidden dimension per head.
        gat_heads:     Number of GAT attention heads.
        n_gat_layers:  Depth of GAT stack.
        dropout:       Dropout rate.
        corr_threshold: Minimum |correlation| to add an edge.
    """

    def __init__(
        self,
        n_assets: int = 50,
        seq_len: int = 60,
        n_features: int = 5,
        gru_hidden: int = 64,
        gat_hidden: int = 32,
        gat_heads: int = 4,
        n_gat_layers: int = 2,
        dropout: float = 0.1,
        corr_threshold: float = 0.3,
        device: str = "cpu",
    ) -> None:
        nn.Module.__init__(self)
        if not _HAS_PYG:
            raise ImportError("Install torch_geometric: pip install torch_geometric")

        self.n_assets = n_assets
        self.corr_threshold = corr_threshold
        self.device = device

        self.temporal_encoder = TemporalEncoder(n_features, gru_hidden)

        gat_layers = []
        in_ch = gru_hidden
        for _ in range(n_gat_layers):
            gat_layers.append(NiftyGATLayer(in_ch, gat_hidden, gat_heads, dropout))
            in_ch = gat_hidden * gat_heads
        self.gat_layers = nn.ModuleList(gat_layers)

        self.head = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),  # Predict 1-step forward return per asset
        )
        self.to(device)

    def build_graph(self, returns_window: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dynamically construct the correlation graph from a returns window.

        Args:
            returns_window: (T, N) returns matrix.

        Returns:
            edge_index: (2, E) COO format edge indices.
            edge_attr:  (E,) edge weights = |correlation|.
        """
        corr = np.corrcoef(returns_window.T)  # (N, N)
        rows, cols = np.where(
            (np.abs(corr) > self.corr_threshold) & (np.eye(self.n_assets) == 0)
        )
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(np.abs(corr[rows, cols]), dtype=torch.float32, device=self.device)
        return edge_index, edge_attr

    def forward(
        self,
        node_features: torch.Tensor,     # (N, T, F)
        edge_index: torch.Tensor,         # (2, E)
        edge_attr: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Args:
            node_features: (N, T, F) — per-asset temporal feature sequences.
            edge_index:    (2, E) — COO format sparse adjacency.
            edge_attr:     (E,) — edge weights.

        Returns:
            ModelOutput with predictions (N, 1).
        """
        # Step 1: Temporal encoding per node
        h = self.temporal_encoder(node_features)   # (N, gru_hidden)

        # Step 2: Graph attention layers
        attn_weights_all = []
        for layer in self.gat_layers:
            h = layer(h, edge_index, edge_attr)    # (N, gat_hidden * heads)

        # Step 3: Per-node predictions
        preds = self.head(h)                       # (N, 1)
        return ModelOutput(predictions=preds)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "NiftyGNNModel":
        """
        X: (episodes, N, T, F) — multiple graph snapshots.
        y: (episodes, N) — per-asset forward returns.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(kwargs.get("epochs", 20)):
            total_loss = 0.0
            for i in range(len(X)):
                node_feats = torch.tensor(X[i], dtype=torch.float32, device=self.device)
                returns_window = X[i, :, :, 0]  # use first feature (returns) for graph construction
                edge_index, edge_attr = self.build_graph(returns_window.T)
                preds = self.forward(node_feats, edge_index, edge_attr).predictions
                target = torch.tensor(y[i], dtype=torch.float32, device=self.device).unsqueeze(-1)
                loss = criterion(preds, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"  [NiftyGNN] Epoch {epoch+1} | Avg Loss: {total_loss/len(X):.6f}")
        return self

    def predict(self, X: np.ndarray) -> ModelOutput:
        self.eval()
        node_feats = torch.tensor(X, dtype=torch.float32, device=self.device)
        returns_window = X[:, :, 0].T
        edge_index, edge_attr = self.build_graph(returns_window)
        with torch.no_grad():
            return self.forward(node_feats, edge_index, edge_attr)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> "NiftyGNNModel":
        self.load_state_dict(torch.load(path, map_location=self.device))
        return self


# ──────────────────────────────────────────────────────────────────────────────
# 3. ORTHOGONALIZED XGBOOST ALPHA PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("[OrthogXGB] xgboost not found.")


class OrthogXGBModel(QuantModel):
    """
    Factor-orthogonalized XGBoost cross-sectional alpha predictor.

    The critical problem in factor investing: if you train an ML model to predict
    returns, it will simply re-discover known risk premia (market beta, size, value,
    momentum). This produces *factor exposure*, not *idiosyncratic alpha*.

    Solution — Orthogonalization (Gu, Kelly, Xiu, 2020):
        1. Regress forward returns on known Fama-French/Barra factors.
        2. Use the OLS residuals as the target variable.
        3. Train XGBoost to predict these pure idiosyncratic residuals.
        4. The model learns alpha unexplained by standard risk factors.

    Risk Factors used:
        - Market: cross-sectional demeaned return (beta)
        - Size: log market cap
        - Value: book-to-market ratio
        - Momentum: 12-1 month return

    Args:
        xgb_params: XGBoost hyperparameters dict.
    """

    DEFAULT_XGB_PARAMS = {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 20,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
    }

    def __init__(self, xgb_params: Optional[dict] = None) -> None:
        if not _HAS_XGB:
            raise ImportError("Install xgboost: pip install xgboost")
        self.xgb_params = xgb_params or self.DEFAULT_XGB_PARAMS
        self.model: Optional[xgb.XGBRegressor] = None
        self._factor_betas: Optional[np.ndarray] = None

    def _orthogonalize_target(
        self,
        y: np.ndarray,
        factors: np.ndarray,
    ) -> np.ndarray:
        """
        Project out known risk factors from target returns via OLS.

        y_residual = y - F @ (F^T F)^{-1} F^T y  [projection onto factor null space]

        Args:
            y:       (N,) raw forward returns.
            factors: (N, K) risk factor exposures for each asset.

        Returns:
            y_ortho: (N,) orthogonalized (idiosyncratic) returns.
        """
        # Add intercept
        F = np.column_stack([np.ones(len(factors)), factors])
        # OLS: β = (F'F)^{-1} F'y
        betas, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
        self._factor_betas = betas
        fitted = F @ betas
        return y - fitted

    def _construct_factors(
        self,
        returns_matrix: np.ndarray,
        log_mkt_cap: np.ndarray,
        book_to_market: np.ndarray,
        momentum_12_1: np.ndarray,
    ) -> np.ndarray:
        """
        Construct cross-sectional Fama-French factor exposures.

        Cross-sectional rank-normalization is applied to each factor to
        reduce outlier sensitivity and improve signal quality.

        Returns:
            factors: (N, 4) matrix — [Market, Size, Value, Momentum]
        """
        def rank_norm(x: np.ndarray) -> np.ndarray:
            """Rank-normalize to [-0.5, +0.5] cross-sectionally."""
            ranks = np.argsort(np.argsort(x)).astype(float)
            return (ranks / (len(ranks) - 1)) - 0.5

        market = rank_norm(returns_matrix.mean(axis=0))  # equal-weighted market return
        size = rank_norm(-log_mkt_cap)                   # SMB: small minus big
        value = rank_norm(book_to_market)                # HML: high minus low
        momentum = rank_norm(momentum_12_1)              # WML: winners minus losers

        return np.column_stack([market, size, value, momentum])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        factors: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "OrthogXGBModel":
        """
        Fit the orthogonalized XGBoost model.

        Args:
            X:       (N, F) feature matrix (microstructure signals, fracdiff prices, etc.)
            y:       (N,) raw forward returns.
            factors: (N, K) risk factor exposures. If None, skip orthogonalization.

        Returns:
            self (fitted model).
        """
        if factors is not None:
            y_target = self._orthogonalize_target(y, factors)
            logger.info("[OrthogXGB] Target orthogonalized against risk factors.")
        else:
            y_target = y
            logger.warning("[OrthogXGB] No factors provided — predicting raw returns (non-orthogonalized).")

        eval_set = kwargs.get("eval_set", None)
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(
            X, y_target,
            eval_set=eval_set,
            verbose=kwargs.get("verbose", 100),
        )
        logger.info("[OrthogXGB] Model fitted successfully.")
        return self

    def predict(self, X: np.ndarray) -> ModelOutput:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        preds = self.model.predict(X)
        return ModelOutput(predictions=torch.tensor(preds, dtype=torch.float32))

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importances from the fitted XGBoost model."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_model(path)

    def load(self, path: str) -> "OrthogXGBModel":
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        return self
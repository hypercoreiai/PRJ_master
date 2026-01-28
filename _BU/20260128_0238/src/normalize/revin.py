"""
RevIN: Reversible Instance Normalization for time series.

Provides:
- RevinTransform: DataFrame-level transform that adds normalized, denormalized,
  mean, and stdev columns using a feature reference (column names or num_features).
- RevIN (nn.Module): Layer for LSTM/Transformer that normalizes at input and
  denormalizes at output; supports pretrain statistics.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd


def _resolve_features(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
    num_features: Optional[int] = None,
) -> List[str]:
    """Return the list of feature column names from df using feature_columns or num_features."""
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not in dataframe: {missing}")
        return list(feature_columns)
    if num_features is not None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(cols) < num_features:
            raise ValueError(
                f"DataFrame has {len(cols)} numeric columns, num_features={num_features}"
            )
        return cols[:num_features]
    raise ValueError("Provide either feature_columns or num_features.")


class RevinTransform:
    """
    RevIN-style transform for DataFrames: adds normalized, denormalized, mean,
    and stdev columns. Uses either explicit feature_columns or num_features
    as a reference to which columns are features.

    Statistics are computed per feature over rows (axis=0). Use this when each
    row is an observation and columns are features.
    """

    def __init__(
        self,
        *,
        feature_columns: Optional[List[str]] = None,
        num_features: Optional[int] = None,
        eps: float = 1e-5,
    ):
        """
        Parameters
        ----------
        feature_columns : list of str or None
            Column names to treat as features. Mutually exclusive with num_features.
        num_features : int or None
            Use the first `num_features` numeric columns as features.
            Mutually exclusive with feature_columns.
        eps : float
            Small constant added to stdev to avoid division by zero.
        """
        if (feature_columns is None) == (num_features is None):
            raise ValueError("Provide exactly one of feature_columns or num_features.")
        self.feature_columns = feature_columns
        self.num_features = num_features
        self.eps = eps
        self.mean_: Optional[pd.Series] = None
        self.stdev_: Optional[pd.Series] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> RevinTransform:
        """Compute mean and stdev per feature from df (over rows)."""
        cols = _resolve_features(df, feature_columns=self.feature_columns, num_features=self.num_features)
        self.mean_ = df[cols].mean()
        var = df[cols].var(ddof=0)
        self.stdev_ = np.sqrt(var + self.eps)
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        inplace: bool = False,
        suffix_normalized: str = "_normalized",
        suffix_denormalized: str = "_denormalized",
        suffix_mean: str = "_mean",
        suffix_stdev: str = "_stdev",
    ) -> pd.DataFrame:
        """
        Add normalized, denormalized (round-trip), mean, and stdev columns.

        Requires fit() or fit_transform() first. Denormalized columns are
        inverse of normalized for sanity check.
        """
        if not self._fitted or self.mean_ is None or self.stdev_ is None:
            raise RuntimeError("Call fit() or fit_transform() before transform().")
        out = df if inplace else df.copy()
        cols = list(self.mean_.index)
        for c in cols:
            m, s = self.mean_[c], self.stdev_[c]
            norm = (out[c] - m) / s
            denorm = norm * s + m
            out[c + suffix_normalized] = norm
            out[c + suffix_denormalized] = denorm
            out[c + suffix_mean] = m
            out[c + suffix_stdev] = s
        return out

    def inverse_transform(
        self,
        df: pd.DataFrame,
        normalized_columns: Optional[List[str]] = None,
        *,
        inplace: bool = False,
        suffix_denormalized: str = "_denormalized",
    ) -> pd.DataFrame:
        """
        Map normalized columns back to original scale.

        If normalized_columns is None, uses {feat}_normalized for each feature.
        """
        if not self._fitted or self.mean_ is None or self.stdev_ is None:
            raise RuntimeError("Call fit() or fit_transform() before inverse_transform().")
        out = df if inplace else df.copy()
        cols = list(self.mean_.index)
        norm_cols = normalized_columns
        if norm_cols is None:
            norm_cols = [c + "_normalized" for c in cols]
        if len(norm_cols) != len(cols):
            raise ValueError("normalized_columns length must match number of features.")
        for c, ncol in zip(cols, norm_cols):
            if ncol not in out.columns:
                continue
            m, s = self.mean_[c], self.stdev_[c]
            out[c + suffix_denormalized] = out[ncol] * s + m
        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        inplace: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Fit on df then add normalized, denormalized, mean, stdev columns."""
        self.fit(df)
        return self.transform(df, inplace=inplace, **kwargs)

    @property
    def n_features_(self) -> int:
        """Number of features (after fit)."""
        if self.mean_ is None:
            raise RuntimeError("Call fit() first.")
        return len(self.mean_)


# --- PyTorch RevIN layer (for LSTM/Transformer step-by-step norm/denorm) ---
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class RevIN(nn.Module):
        """
        RevIN layer for use at input (normalize) and output (denormalize) of
        LSTM/Transformer. Call with mode='norm' at model input and mode='denorm'
        at model output so predictions are on the original scale.

        Supports pretrain data: set mean/stdev from a pretrain batch or dataset
        via set_pretrain_stats(), then norm/denorm use those fixed statistics.

        Integration::
            # In model forward:
            x_norm = self.revin(x, mode='norm')      # normalize input
            out = self.backbone(x_norm)
            return self.revin(out, mode='denorm')     # denormalize output
        """

        def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.affine_weight = nn.Parameter(torch.ones(num_features))
                self.affine_bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.register_parameter("affine_weight", None)
                self.register_parameter("affine_bias", None)
            self._mean: Optional[torch.Tensor] = None
            self._stdev: Optional[torch.Tensor] = None
            self._pretrain_mean: Optional[torch.Tensor] = None
            self._pretrain_stdev: Optional[torch.Tensor] = None

        def set_pretrain_stats(
            self,
            mean: Union[torch.Tensor, np.ndarray],
            stdev: Union[torch.Tensor, np.ndarray],
        ) -> None:
            """
            Set statistics from pretrain data. Shapes like (num_features,) or
            (1, num_features) or (1, 1, num_features) for [B, S, F].
            """
            if not isinstance(mean, torch.Tensor):
                mean = torch.as_tensor(mean, dtype=torch.float32)
            if not isinstance(stdev, torch.Tensor):
                stdev = torch.as_tensor(stdev, dtype=torch.float32)
            # Flatten to (num_features,) then expand for (1, 1, num_features)
            m = mean.flatten()[: self.num_features]
            s = stdev.flatten()[: self.num_features]
            if m.numel() < self.num_features or s.numel() < self.num_features:
                raise ValueError(
                    f"mean/stdev must have at least num_features={self.num_features} elements."
                )
            self._pretrain_mean = m.view(1, 1, -1).expand(1, 1, self.num_features)
            self._pretrain_stdev = s.view(1, 1, -1).expand(1, 1, self.num_features)

        def clear_pretrain_stats(self) -> None:
            """Use batch statistics again instead of pretrain."""
            self._pretrain_mean = None
            self._pretrain_stdev = None

        def _get_statistics(self, x: torch.Tensor) -> None:
            # x: [batch, sequence_length, num_features]
            self._mean = torch.mean(x, dim=1, keepdim=True).detach()
            self._stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + self.eps).detach()

        def _normalize(self, x: torch.Tensor) -> torch.Tensor:
            mean = self._mean
            stdev = self._stdev
            if mean is None or stdev is None:
                raise RuntimeError("RevIN: run forward(..., mode='norm') first or set_pretrain_stats().")
            x = x - mean
            x = x / stdev
            if self.affine and self.affine_weight is not None:
                x = x * self.affine_weight
                x = x + self.affine_bias
            return x

        def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
            mean = self._mean
            stdev = self._stdev
            if mean is None or stdev is None:
                raise RuntimeError("RevIN: mean/stdev not set. Run norm pass or set_pretrain_stats().")
            if self.affine and self.affine_weight is not None:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps)  # FIX: Add + self.eps for numerical stability
            x = x * stdev
            x = x + mean
            return x

        def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
            """
            mode='norm': normalize (use at model input).
            mode='denorm': denormalize (use at model output).
            """
            if mode == "norm":
                if self._pretrain_mean is not None and self._pretrain_stdev is not None:
                    # Expand to [B, 1, F] and move to x's device
                    b, _, f = x.size()
                    self._mean = self._pretrain_mean.to(x.device).expand(b, 1, f)
                    self._stdev = self._pretrain_stdev.to(x.device).expand(b, 1, f)
                else:
                    self._get_statistics(x)
                return self._normalize(x)
            if mode == "denorm":
                return self._denormalize(x)
            raise NotImplementedError(f"mode must be 'norm' or 'denorm', got {mode!r}.")

else:
    RevIN = None  # type: ignore[misc, assignment]

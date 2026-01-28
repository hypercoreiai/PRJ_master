# Normalize — RevIN Usage & Specifications

Reference for **RevinTransform** (DataFrame) and **RevIN** (PyTorch layer) in `revin.py` and `revin_layer.py`. RevIN (Reversible Instance Normalization) normalizes time-series inputs and denormalizes outputs so predictions stay on the original scale.

---

## Overview

| Component        | Purpose |
|-----------------|---------|
| **RevinTransform** | DataFrame transform: adds normalized, denormalized, mean, and stdev columns. Uses `feature_columns` or `num_features` as the feature reference. |
| **RevIN**        | PyTorch `nn.Module`: normalize at model input, denormalize at model output. For LSTM/Transformer. Supports pretrain statistics. |

**Dependencies:** `pandas`, `numpy`. `RevIN` additionally requires `torch`.

---

## RevinTransform — Specifications

### Constructor

```
RevinTransform(*, feature_columns=None, num_features=None, eps=1e-5)
```

| Argument           | Type              | Description |
|--------------------|-------------------|-------------|
| `feature_columns` | `list[str]` or None | Column names to treat as features. Must exist in the DataFrame. Mutually exclusive with `num_features`. |
| `num_features`    | `int` or None      | Use the first `num_features` numeric columns (left-to-right) as features. Mutually exclusive with `feature_columns`. |
| `eps`             | `float`            | Small constant added to stdev to avoid division by zero. Default `1e-5`. |

Exactly one of `feature_columns` or `num_features` must be provided.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(df)` | `self` | Compute mean and stdev per feature over rows (axis=0). |
| `transform(df, inplace=False, suffix_normalized='_normalized', suffix_denormalized='_denormalized', suffix_mean='_mean', suffix_stdev='_stdev')` | `pd.DataFrame` | Add, for each feature column `c`: `c_normalized`, `c_denormalized`, `c_mean`, `c_stdev`. Requires prior `fit()` or `fit_transform()`. Denormalized = round-trip of normalized. |
| `inverse_transform(df, normalized_columns=None, inplace=False, suffix_denormalized='_denormalized')` | `pd.DataFrame` | Map normalized columns back to original scale using stored mean/stdev. If `normalized_columns` is None, uses `{feat}_normalized` for each feature. |
| `fit_transform(df, inplace=False, **kwargs)` | `pd.DataFrame` | Equivalent to `fit(df)` then `transform(df, ...)`. |
| `n_features_`     | `int` (property)   | Number of features (defined after `fit`). |

### Output columns (per feature `c`)

- `c_normalized`   = `(c - mean) / stdev`
- `c_denormalized` = `c_normalized * stdev + mean` (round-trip check)
- `c_mean`         = fitted mean
- `c_stdev`        = fitted stdev

Statistics are computed over **rows** (axis=0).

---

## RevinTransform — Usage

### By column names

```python
from normalize import RevinTransform
import pandas as pd

df = pd.DataFrame({
    "date": pd.date_range("2020-01-01", periods=100, freq="D"),
    "open": np.random.randn(100).cumsum() + 100,
    "high": np.random.randn(100).cumsum() + 102,
    "close": np.random.randn(100).cumsum() + 101,
})

r = RevinTransform(feature_columns=["open", "high", "close"])
out = r.fit_transform(df)
# out has: open_normalized, open_denormalized, open_mean, open_stdev, same for high/close
```

### By number of features

```python
r = RevinTransform(num_features=2)  # first 2 numeric columns
out = r.fit_transform(df)
```

### Transform only (after fit)

```python
r = RevinTransform(feature_columns=["open", "close"])
r.fit(df)
test_out = r.transform(df_test)
```

### Inverse (normalized → original scale)

```python
r.fit(df)
df_norm = r.transform(df)
# ... use df_norm for modeling ...
# map predictions in normalized space back:
df_norm["pred_normalized"] = model_predictions
back = r.inverse_transform(df_norm, normalized_columns=["pred_normalized"])
# back has pred_normalized_denormalized on original scale
```

---

## RevIN (PyTorch) — Specifications

### Constructor

```
RevIN(num_features, eps=1e-5, affine=True)
```

| Argument        | Type   | Description |
|-----------------|--------|-------------|
| `num_features`  | `int`  | Number of features (last dimension of input). Input shape: `[B, S, F]` with `F = num_features`. |
| `eps`           | `float`| Constant added to variance before sqrt. Default `1e-5`. |
| `affine`        | `bool` | If True, use learnable affine weight and bias after normalization. Default True. |

### Forward

```
forward(x, mode='norm' | 'denorm') -> Tensor
```

- **`mode='norm'`** — Normalize `x`. Statistics are computed over the **sequence dimension** (dim=1), i.e. per batch element: `mean`, `stdev` shape `[B, 1, F]`. Use at **model input**.
- **`mode='denorm'`** — Denormalize using the same mean/stdev stored in the last `norm` call (or from pretrain). Use at **model output** so predictions are on the original scale.

**Input shape:** `x` is `[batch, sequence_length, num_features]`.

### Pretrain statistics

| Method | Description |
|--------|-------------|
| `set_pretrain_stats(mean, stdev)` | Set fixed mean and stdev from pretrain data. `mean`/`stdev` can be `torch.Tensor` or `numpy.ndarray`; shapes like `(num_features,)`, `(1, num_features)`, or `(1, 1, num_features)` are accepted. |
| `clear_pretrain_stats()` | Clear fixed stats; forward will use batch statistics again when `mode='norm'`. |

When pretrain stats are set, `forward(..., mode='norm')` uses them (expanded to `[B, 1, F]`) instead of computing from the current batch.

---

## RevIN — Usage & model integration

### Step-by-step (norm → backbone → denorm)

Use RevIN at the **input** of the forward pass (normalize) and at the **output** (denormalize):

```python
import torch
from normalize.revin_layer import RevIN  # or: from normalize import RevIN

class RevINModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size=64):
        super().__init__()
        self.revin = RevIN(num_features)
        self.backbone = torch.nn.LSTM(num_features, hidden_size, batch_first=True)
        self.head = torch.nn.Linear(hidden_size, num_features)

    def forward(self, x, mode="norm"):
        # 1. Normalize input
        x_norm = self.revin(x, mode="norm")
        # 2. Backbone (LSTM / Transformer / etc.)
        out, _ = self.backbone(x_norm)
        out = self.head(out)
        # 3. Denormalize output to original scale
        return self.revin(out, mode="denorm")
```

### Using pretrain statistics

```python
revin = RevIN(num_features=3)

# From a pretrain batch [B, S, F]: compute mean/stdev over dim=1 (sequence)
pretrain_batch = torch.randn(32, 100, 3) * 10 + 50
mean = pretrain_batch.mean(dim=1, keepdim=True).mean(dim=0)   # (3,)
stdev = pretrain_batch.std(dim=1, keepdim=True).mean(dim=0)   # (3,)
revin.set_pretrain_stats(mean, stdev)

# Later: norm/denorm use these fixed stats
x = some_batch  # [B, S, 3]
x_norm = revin(x, mode="norm")
x_back = revin(x_norm, mode="denorm")
```

### Round-trip check

```python
x = torch.randn(4, 50, 2) * 100 + 500
revin = RevIN(num_features=2)
xn = revin(x, mode="norm")
xd = revin(xn, mode="denorm")
assert torch.allclose(x, xd)  # denorm(norm(x)) == x
```

---

## File layout

| File            | Contents |
|-----------------|----------|
| **`revin.py`**  | `RevinTransform`, `RevIN` (when `torch` is available), and feature-resolution helpers. |
| **`revin_layer.py`** | Re-exports `RevIN` for model code; raises if `torch` is missing. |
| **`__init__.py`**   | Exposes `RevinTransform` and `RevIN` (when `torch` is available). |

---

## Import summary

```python
# DataFrame transform (no torch)
from normalize import RevinTransform

# PyTorch layer (requires torch)
from normalize import RevIN
# or, for model scripts:
from normalize.revin_layer import RevIN
```

---

## Quick reference

| Need | Use |
|------|-----|
| Add normalized/denormalized/mean/stdev columns to a DataFrame | `RevinTransform(feature_columns=...)` or `RevinTransform(num_features=...)` then `fit_transform(df)`. |
| Normalize/denormalize inside an LSTM/Transformer | `RevIN(num_features)`; call `revin(x, 'norm')` at input and `revin(out, 'denorm')` at output. |
| Use fixed statistics from pretrain data | `RevIN.set_pretrain_stats(mean, stdev)` before training/inference. |

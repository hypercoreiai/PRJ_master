"""
RevIN nn.Module for LSTM/Transformer: step-by-step norm at input, denorm at output.

Import this module when integrating RevIN into a model:
  - Call revin(x, mode='norm') at the start of forward.
  - Call revin(out, mode='denorm') at the end so predictions are on the original scale.

Supports pretrain data via set_pretrain_stats(mean, stdev). Requires torch.
"""

from __future__ import annotations

from .revin import RevIN

if RevIN is None:
    raise ImportError("RevIN requires torch. Install with: pip install torch")

__all__ = ["RevIN"]

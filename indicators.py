#!/usr/bin/env python3
"""indicators.py — signal indicators.

The SuperTrend / ATR / ADX primitives are reused directly from the public
`futures_foundation` library so the bot's live signals match exactly how the
models were trained. EMA is a trivial recursive (causal) helper.
"""
import numpy as np
import pandas as pd

from futures_foundation.chronos._primitives import (
    compute_adx, compute_atr, compute_supertrend,
)


def _hlc(bars: pd.DataFrame):
    return (bars["high"].to_numpy(float),
            bars["low"].to_numpy(float),
            bars["close"].to_numpy(float))


def atr(bars: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder ATR (float64[n], NaN before `period`)."""
    h, l, c = _hlc(bars)
    return compute_atr(h, l, c, period)


def supertrend(bars: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Returns (line, direction): direction +1 bull / -1 bear (a flip is a
    change between adjacent bars); line = the SuperTrend trailing level."""
    h, l, c = _hlc(bars)
    direction, line, _atr = compute_supertrend(h, l, c, period, mult)
    return np.asarray(line, dtype=float), np.asarray(direction, dtype=float)


def adx(bars: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ADX (float64[n], NaN early)."""
    h, l, c = _hlc(bars)
    return compute_adx(h, l, c, period)


def ema(close, span: int) -> np.ndarray:
    """Causal recursive EMA (adjust=False → strictly trailing)."""
    return pd.Series(np.asarray(close, dtype=float)).ewm(
        span=span, adjust=False).mean().to_numpy()

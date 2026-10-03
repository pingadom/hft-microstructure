"""Features computed from the order book: mid, spread, imbalance, microprice, order flow imbalance.

Notation (matching the papers):
  P^b, P^a   best bid / best ask price
  q^b, q^a   size resting at the best bid / best ask
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mid(quotes: pd.DataFrame) -> np.ndarray:
    return (quotes["bid"].to_numpy() + quotes["ask"].to_numpy()) / 2


def spread_ticks(quotes: pd.DataFrame, tick: float) -> np.ndarray:
    """Spread measured in ticks (rounded, because floats like 0.1 aren't exact)."""
    return np.rint((quotes["ask"].to_numpy() - quotes["bid"].to_numpy()) / tick).astype(int)


def imbalance(bid_qty, ask_qty) -> np.ndarray:
    """Queue imbalance I = q^b / (q^b + q^a), between 0 and 1.

    I near 1: lots of buyers waiting, few sellers, so the ask is likely to be eaten
    and the price to tick UP. I near 0: the reverse.
    """
    b = np.asarray(bid_qty, dtype=float)
    a = np.asarray(ask_qty, dtype=float)
    return b / (b + a)


def weighted_mid(quotes: pd.DataFrame) -> np.ndarray:
    """Imbalance-weighted mid: I * ask + (1 - I) * bid.

    Note the cross-over: a big BID queue pulls the estimate toward the ASK, because a big
    bid queue means the next move is probably up.
    """
    i = imbalance(quotes["bid_qty"], quotes["ask_qty"])
    return i * quotes["ask"].to_numpy() + (1 - i) * quotes["bid"].to_numpy()


def ofi_events(bid, bid_qty, ask, ask_qty) -> np.ndarray:
    """Order flow imbalance contribution of each book update (Cont, Kukanov & Stoikov 2014).

    For update n (compared with update n-1):

      e_n =   q^b_n     * 1[P^b_n >= P^b_{n-1}]     (bid queue now, if bid didn't fall)
            - q^b_{n-1} * 1[P^b_n <= P^b_{n-1}]     (bid queue before, if bid didn't rise)
            - q^a_n     * 1[P^a_n <= P^a_{n-1}]     (ask queue now, if ask didn't rise)
            + q^a_{n-1} * 1[P^a_n >= P^a_{n-1}]     (ask queue before, if ask didn't fall)

    Translate the cases for the bid side:
      bid price unchanged -> e gets (q^b_n - q^b_{n-1}): +size added, -size cancelled or eaten
      bid price went up   -> e gets +q^b_n: a whole new, better bid appeared (buying pressure)
      bid price went down -> e gets -q^b_{n-1}: the old best bid was wiped out (selling pressure)
    The ask side is the mirror image with signs flipped. Positive e = buying pressure.

    Works on 1-D arrays (level 1), or 2-D (n, levels) arrays to get the OFI at each level
    separately (Xu, Gould & Howison 2018, "multi-level OFI").
    Returns an array the same shape as the input with e_0 = 0.
    """
    Pb, qb, Pa, qa = (np.asarray(v, dtype=float) for v in (bid, bid_qty, ask, ask_qty))
    e = np.zeros_like(qb)
    e[1:] = (
        qb[1:] * (Pb[1:] >= Pb[:-1])
        - qb[:-1] * (Pb[1:] <= Pb[:-1])
        - qa[1:] * (Pa[1:] <= Pa[:-1])
        + qa[:-1] * (Pa[1:] >= Pa[:-1])
    )
    return e


def resample_last(ts: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Value prevailing at each grid time: the last observation at or before it.

    This is "previous-tick" sampling, the standard way to turn irregular tick data into
    a regular time series without peeking into the future.
    """
    idx = np.searchsorted(ts, grid, side="right") - 1
    out = np.asarray(values, dtype=float)[np.clip(idx, 0, None)]
    return np.where(idx >= 0, out, np.nan)


def sum_in_buckets(ts: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sum of values with grid[k] < ts <= grid[k+1], for each consecutive pair of grid points.

    Returns len(grid) - 1 sums. Used to add up OFI or signed volume over each interval.
    """
    c = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=float))])
    idx = np.searchsorted(ts, grid, side="right")
    return np.diff(c[idx])


def time_grid(start, end, freq: str) -> np.ndarray:
    """Regular datetime64[ms] grid from start to end (inclusive), e.g. freq='1s'."""
    return pd.date_range(start, end, freq=freq).to_numpy().astype("datetime64[ms]")


def to_ms(ts) -> np.ndarray:
    """Timestamps (Series or array, tz-aware or not) as numpy datetime64[ms]."""
    if isinstance(ts, pd.Series):
        ts = ts.dt.tz_localize(None) if ts.dt.tz is not None else ts
        return ts.to_numpy().astype("datetime64[ms]")
    return np.asarray(ts).astype("datetime64[ms]")

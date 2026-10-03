"""Trade-sign classification: who was the aggressor?

On most stock exchanges the public tape does NOT say whether a trade was buyer- or
seller-initiated, so researchers infer it. Binance tells us the truth, which lets us
measure how good the classic inference rules are.
"""

from __future__ import annotations

import numpy as np


def tick_rule(prices) -> np.ndarray:
    """+1 if price rose since the previous trade, -1 if it fell.
    If unchanged, copy the previous trade's sign (the "zero tick" rule).
    The first trade (and any leading zero-ticks) get 0 = unknown.
    """
    p = np.asarray(prices, dtype=float)
    s = np.zeros(len(p), dtype=np.int8)
    s[1:] = np.sign(np.diff(p))
    # Forward-fill zeros with the last non-zero sign.
    idx = np.where(s != 0, np.arange(len(s)), 0)
    np.maximum.accumulate(idx, out=idx)
    return s[idx]


def quote_rule(prices, mids) -> np.ndarray:
    """+1 if the trade printed above the prevailing mid (someone paid up = buyer),
    -1 if below, 0 if exactly at the mid (ambiguous)."""
    return np.sign(np.asarray(prices, dtype=float) - np.asarray(mids, dtype=float)).astype(np.int8)


def lee_ready(prices, mids) -> np.ndarray:
    """Lee & Ready (1991): use the quote rule; for trades exactly at the mid, fall back to
    the tick rule."""
    q = quote_rule(prices, mids)
    t = tick_rule(prices)
    return np.where(q != 0, q, t).astype(np.int8)

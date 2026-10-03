"""Price impact and transaction costs.

Every trade moves the price against the person who made it. Measuring by how much,
and how that depends on size and time, is chapter 7. The spread decomposition here
(chapter 4) says who wins each trade: the market maker or the aggressor.
"""

from __future__ import annotations

import numpy as np

from .book import resample_last


def mid_at(quote_ts, mids, at_ts, strictly_before: bool = False) -> np.ndarray:
    """Mid prevailing at each time in at_ts.

    strictly_before=True uses only quotes timestamped BEFORE at_ts. Use this for the quote
    "just before a trade", since a quote update in the same millisecond may be the trade's
    own footprint.
    """
    quote_ts = np.asarray(quote_ts).astype("datetime64[ms]")
    at_ts = np.asarray(at_ts).astype("datetime64[ms]")
    if strictly_before:
        at_ts = at_ts - np.timedelta64(1, "ms")
    return resample_last(quote_ts, mids, at_ts)


def spread_decomposition(price, sign, mid_before, mid_after) -> dict[str, np.ndarray]:
    """Split the cost of each trade into two parts (all in price units, per unit traded).

    effective spread  = 2 * sign * (price - mid_before)
        What the aggressor paid relative to the mid, times 2 (so it's comparable to the
        quoted spread). A buyer at the ask with a 1-tick spread pays exactly 1 tick.

    price impact      = 2 * sign * (mid_after - mid_before)
        How far the mid moved in the trade's direction afterwards. The market maker lost
        this to the aggressor: it's the adverse-selection cost.

    realized spread   = 2 * sign * (price - mid_after)
        What the maker actually kept once the price settled. effective = realized + impact.
    """
    s = np.asarray(sign, dtype=float)
    p = np.asarray(price, dtype=float)
    return {
        "effective": 2 * s * (p - mid_before),
        "impact": 2 * s * (mid_after - mid_before),
        "realized": 2 * s * (p - mid_after),
    }


def response_function(signs, mids, lags) -> np.ndarray:
    """Bouchaud's response function R(l) = E[ s_n * (m_{n+l} - m_n) ], in TRADE time.

    m_n is the mid just before trade n, s_n its sign. R(l) is the average move in the
    trade's direction l trades later. It typically rises and then levels off, because
    order flow is persistent (chapter 2) but prices stay unpredictable.
    """
    s = np.asarray(signs, dtype=float)
    m = np.asarray(mids, dtype=float)
    out = []
    for lag in lags:
        out.append(np.nanmean(s[: len(s) - lag] * (m[lag:] - m[: len(m) - lag])))
    return np.array(out)

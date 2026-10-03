"""Volatility at high frequency, and why naive estimates go wrong.

The core model (used in chapters 3 and 4):

    observed price  p_t = efficient price  m_t  +  noise u_t

m_t is a random walk (the "true" value). u_t is microstructure noise: bid-ask bounce,
tick rounding, etc. Noise doesn't grow with time; the random walk does. So at long
horizons noise is negligible, and at very short horizons it dominates.
"""

from __future__ import annotations

import numpy as np

from .book import resample_last, time_grid


def log_returns(prices) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))


def realized_variance(prices) -> float:
    """RV = sum of squared log returns. Without noise, RV -> true integrated variance
    as sampling gets finer. With noise, RV explodes instead (chapter 3)."""
    r = log_returns(prices)
    r = r[~np.isnan(r)]
    return float(r @ r)


def signature_plot(ts, prices, freqs: list[str], start=None, end=None) -> dict[str, float]:
    """Realized variance of the day computed at each sampling frequency.

    Plotting RV against sampling interval is the "volatility signature plot"
    (Andersen, Bollerslev, Diebold & Labys 2000). Flat = no noise. Rising as the
    interval shrinks = noise.
    """
    ts = np.asarray(ts).astype("datetime64[ms]")
    start = ts[0] if start is None else start
    end = ts[-1] if end is None else end
    out = {}
    for f in freqs:
        grid = time_grid(start, end, f)
        out[f] = realized_variance(resample_last(ts, prices, grid))
    return out


def noise_variance(prices) -> float:
    """Estimate Var(u) from the highest-frequency returns.

    With n tick-by-tick returns, r_i = (m_i - m_{i-1}) + (u_i - u_{i-1}), and for small
    steps the noise term dominates: E[RV] ~ 2 n Var(u). So Var(u) ~ RV / (2n)
    (Bandi & Russell 2008; Zhang, Mykland & Ait-Sahalia 2005).
    """
    r = log_returns(prices)
    return float(r @ r / (2 * len(r)))


def roll_spread(prices) -> float:
    """Roll (1984) spread estimator from trade prices ONLY.

    Model: trade price = m_t + (s/2) * q_t, where q_t = +1 for buys, -1 for sells, and
    the q_t are independent coin flips. Then consecutive price changes have
        Cov(dp_t, dp_{t-1}) = -s^2 / 4
    so s = 2 * sqrt(-Cov). Returns nan if the covariance is positive (the model fails).
    """
    dp = np.diff(np.asarray(prices, dtype=float))
    c = np.cov(dp[1:], dp[:-1])[0, 1]
    return float(2 * np.sqrt(-c)) if c < 0 else float("nan")

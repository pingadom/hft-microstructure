"""Hawkes processes: arrivals that make more arrivals.

A Poisson process has a constant rate: events don't care about the past. Trades are
nothing like that. One trade triggers others (other algos react, iceberg orders refill,
stop losses chain). A Hawkes process models this with an intensity that jumps after
each event and then decays:

    lambda(t) = mu + sum over past events t_i < t of  alpha * exp(-beta * (t - t_i))

  mu     baseline rate (events/second that arrive "from outside")
  alpha  jump in intensity caused by each event
  beta   how fast that excitement decays (1/beta = memory in seconds)

Branching ratio n = alpha / beta = the expected number of "child" events each event
triggers. n < 1 is needed for stationarity; the long-run rate is mu / (1 - n).
Empirically, n is often 0.7-0.9 on financial markets: most trades are reactions to trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def simulate(mu: float, alpha: float, beta: float, T: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate event times on [0, T] by Ogata's thinning algorithm.

    Idea: between events the intensity only decays, so its current value is an upper
    bound until the next event. Propose a candidate time from a Poisson process at that
    bound, then accept it with probability lambda(candidate) / bound.
    """
    times = []
    t = 0.0
    excite = 0.0  # sum of alpha * exp(-beta * (t - t_i)) at the current time t
    while True:
        bound = mu + excite
        w = rng.exponential(1 / bound)
        t += w
        if t > T:
            break
        excite *= np.exp(-beta * w)
        if rng.uniform() * bound <= mu + excite:  # accept
            times.append(t)
            excite += alpha
    return np.array(times)


def loglik(params, times: np.ndarray, T: float) -> float:
    """Exact log-likelihood of an exponential Hawkes process on [0, T].

    log L = sum_i log lambda(t_i)  -  integral_0^T lambda(t) dt
    The integral = mu*T + (alpha/beta) * sum_i (1 - exp(-beta (T - t_i))).

    The sum inside lambda(t_i) uses the recursion
        A_i = exp(-beta (t_i - t_{i-1})) * (1 + A_{i-1}),  A_1 = 0,
    so lambda(t_i) = mu + alpha * A_i. That is O(n) instead of O(n^2).
    """
    mu, alpha, beta = params
    dt = np.diff(times)
    decay = np.exp(-beta * dt)
    A = np.zeros(len(times))
    for i in range(1, len(times)):  # a simple loop is clearer than any vectorised trick
        A[i] = decay[i - 1] * (1 + A[i - 1])
    lam = mu + alpha * A
    integral = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - times)))
    return float(np.sum(np.log(lam)) - integral)


@dataclass
class HawkesFit:
    mu: float
    alpha: float
    beta: float
    loglik: float
    poisson_loglik: float
    n_events: int
    T: float

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    @property
    def lr_stat(self) -> float:
        """Likelihood-ratio statistic vs a plain Poisson process (2 extra parameters).
        Above ~6 (chi-squared, 2 df, 5%) rejects Poisson."""
        return 2 * (self.loglik - self.poisson_loglik)

    def __repr__(self) -> str:
        return (f"HawkesFit(mu={self.mu:.4g}/s, alpha={self.alpha:.4g}, beta={self.beta:.4g}/s, "
                f"branching ratio={self.branching_ratio:.3f}, memory 1/beta={1 / self.beta:.3g}s, "
                f"LR vs Poisson={self.lr_stat:,.0f})")


def fit(times: np.ndarray, T: float, x0=None) -> HawkesFit:
    """Maximum-likelihood fit. Optimises over log-parameters so they stay positive."""
    times = np.asarray(times, dtype=float)
    n = len(times)
    rate = n / T
    if x0 is None:
        x0 = (0.3 * rate, 0.7 * 10.0, 10.0)  # guess: 70% branching, 0.1 s memory

    def neg(logp):
        mu, alpha, beta = np.exp(logp)
        if alpha >= beta:  # non-stationary region: forbid
            return 1e18
        return -loglik((mu, alpha, beta), times, T)

    res = minimize(neg, np.log(x0), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
    mu, alpha, beta = np.exp(res.x)
    poisson = n * np.log(rate) - n  # MLE of a Poisson process: rate = n / T
    return HawkesFit(mu, alpha, beta, -res.fun, poisson, n, T)


def compensator(times: np.ndarray, mu: float, alpha: float, beta: float) -> np.ndarray:
    """Lambda(t_i) = integral_0^{t_i} lambda(s) ds at each event.

    Time-rescaling theorem: if the model is right, the gaps Lambda(t_i) - Lambda(t_{i-1})
    are i.i.d. Exponential(1). That gives a goodness-of-fit test (QQ plot vs Exp(1)).
    """
    times = np.asarray(times, dtype=float)
    out = np.empty(len(times))
    A = 0.0     # sum of exp(-beta (t - t_j)) over past events, at the current event
    total = 0.0
    prev = 0.0
    for i, t in enumerate(times):
        w = t - prev
        # integral over (prev, t] of mu + alpha * A * exp(-beta (s - prev)) ds
        total += mu * w + (alpha / beta) * A * (1 - np.exp(-beta * w))
        out[i] = total
        A = A * np.exp(-beta * w) + 1
        prev = t
    return out

"""Statistical tools, written out by hand so you can see exactly what they compute.

Each one is checked against a library implementation (statsmodels) in tests/test_stats.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation at lags 0..nlags.

    rho(k) = sum_t (x_t - xbar)(x_{t+k} - xbar) / sum_t (x_t - xbar)^2

    Computed with the FFT, which makes lags up to 10,000 on millions of points instant.
    (The direct double loop would take hours.)
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    size = 1 << int(np.ceil(np.log2(2 * n)))  # zero-pad to avoid circular wrap-around
    f = np.fft.rfft(x, size)
    full = np.fft.irfft(f * np.conj(f), size)[: nlags + 1]
    return full / full[0]


@dataclass
class OLSResult:
    beta: np.ndarray      # coefficients (intercept first if add_const)
    se: np.ndarray        # standard errors
    t: np.ndarray         # t-statistics = beta / se
    r2: float             # R-squared
    n: int                # observations
    names: list[str]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.beta, "std err": self.se, "t": self.t}, index=self.names)

    def __repr__(self) -> str:
        return f"OLS(n={self.n:,}, R2={self.r2:.4f})\n{self.summary().to_string(float_format='%.4g')}"


def ols(y, X, add_const: bool = True, hac_lags: int | None = None, names: list[str] | None = None) -> OLSResult:
    """Ordinary least squares: beta = (X'X)^{-1} X'y.

    Standard errors:
      hac_lags=None  classic OLS errors, which assume residuals are independent and have
                     constant variance. Usually WRONG for market data.
      hac_lags=L     Newey-West (HAC) errors, robust to heteroskedasticity and to
                     autocorrelation up to lag L. Use these whenever samples overlap in time
                     (e.g. 10-second returns sampled every second) or volatility clusters.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if names is None:
        names = [f"x{i}" for i in range(X.shape[1])]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        names = ["const"] + list(names)
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))

    if hac_lags is None:
        sigma2 = resid @ resid / (n - k)
        cov = sigma2 * XtX_inv
    else:
        # "Meat" of the sandwich: S = sum over lags of weighted autocovariances of X_t * e_t.
        u = X * resid[:, None]
        S = u.T @ u
        for lag in range(1, hac_lags + 1):
            w = 1 - lag / (hac_lags + 1)          # Bartlett kernel keeps S positive definite
            G = u[lag:].T @ u[:-lag]
            S += w * (G + G.T)
        cov = XtX_inv @ S @ XtX_inv           # (X'X)^-1 S (X'X)^-1  -- the "sandwich"
    se = np.sqrt(np.diag(cov))
    return OLSResult(beta=beta, se=se, t=beta / se, r2=float(r2), n=n, names=names)


def binned_mean(x, y, bins) -> pd.DataFrame:
    """E[y | x in bin], with a standard error for each bin. The workhorse for
    "how does y depend on x?" plots when you don't want to assume a straight line.

    bins: an int (that many equal-count bins) or an array of bin edges.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isscalar(bins):
        edges = np.unique(np.quantile(x, np.linspace(0, 1, int(bins) + 1)))
    else:
        edges = np.asarray(bins, dtype=float)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    df = pd.DataFrame({"bin": idx, "x": x, "y": y})
    g = df.groupby("bin")
    out = pd.DataFrame({
        "x_mean": g["x"].mean(),
        "y_mean": g["y"].mean(),
        "y_se": g["y"].std() / np.sqrt(g["y"].count()),
        "count": g["y"].count(),
    })
    return out.reset_index(drop=True)


def loglog_slope(x, y) -> tuple[float, float]:
    """Fit y = C * x^slope by OLS on log y = log C + slope * log x.

    Returns (slope, C). Only positive points are used.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = (x > 0) & (y > 0)
    res = ols(np.log(y[ok]), np.log(x[ok]))
    return float(res.beta[1]), float(np.exp(res.beta[0]))

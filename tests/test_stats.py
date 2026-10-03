import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf as sm_acf

from micro.stats import acf, binned_mean, loglog_slope, ols


def test_acf_matches_statsmodels():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=2000)) * 0.1 + rng.normal(size=2000)
    assert np.allclose(acf(x, 30), sm_acf(x, nlags=30, fft=False))


def test_acf_of_ar1_is_geometric():
    rng = np.random.default_rng(1)
    phi, n = 0.6, 200_000
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    r = acf(x, 3)
    assert np.allclose(r, phi ** np.arange(4), atol=0.01)


def test_ols_matches_statsmodels_plain_and_hac():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.normal(size=(n, 2))
    # autocorrelated, heteroskedastic errors: the case HAC errors are for
    e = np.convolve(rng.normal(size=n + 4), np.ones(5) / 5, mode="valid") * (1 + np.abs(X[:, 0]))
    y = 0.5 + X @ np.array([1.0, -2.0]) + e

    mine = ols(y, X)
    ref = sm.OLS(y, sm.add_constant(X)).fit()
    assert np.allclose(mine.beta, ref.params)
    assert np.allclose(mine.se, ref.bse)
    assert np.isclose(mine.r2, ref.rsquared)

    mine_hac = ols(y, X, hac_lags=5)
    ref_hac = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": False})
    assert np.allclose(mine_hac.se, ref_hac.bse)


def test_binned_mean_recovers_conditional_mean():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 100_000)
    y = 3 * x + rng.normal(size=x.size)
    b = binned_mean(x, y, 10)
    assert len(b) == 10
    assert np.allclose(b["y_mean"], 3 * b["x_mean"], atol=0.05)


def test_loglog_slope():
    x = np.logspace(0, 3, 50)
    slope, c = loglog_slope(x, 2.5 * x ** -0.4)
    assert np.isclose(slope, -0.4) and np.isclose(c, 2.5)

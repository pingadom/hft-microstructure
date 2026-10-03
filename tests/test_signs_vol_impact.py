import numpy as np
import pandas as pd

from micro.data import aggregate_trades
from micro.impact import response_function, spread_decomposition
from micro.signs import lee_ready, quote_rule, tick_rule
from micro.volatility import noise_variance, realized_variance, roll_spread


def test_tick_rule_carries_sign_through_zero_ticks():
    p = [10.0, 10.0, 10.1, 10.1, 10.0, 10.0, 10.2]
    assert list(tick_rule(p)) == [0, 0, 1, 1, -1, -1, 1]


def test_lee_ready_falls_back_to_tick_rule_at_mid():
    p = np.array([10.0, 10.1, 10.05, 10.0])
    m = np.array([10.05, 10.05, 10.05, 10.05])
    assert list(quote_rule(p, m)) == [-1, 1, 0, -1]
    assert list(lee_ready(p, m)) == [-1, 1, -1, -1]  # 10.05 < 10.1 by tick rule -> sell


def simulate_roll(n, s, sigma, rng):
    m = 100 + np.cumsum(rng.normal(0, sigma, n))
    q = rng.choice([-1, 1], n)
    return m + s / 2 * q, m, q


def test_roll_estimator_recovers_spread():
    rng = np.random.default_rng(0)
    p, _, _ = simulate_roll(500_000, s=0.2, sigma=0.05, rng=rng)
    assert abs(roll_spread(p) - 0.2) < 0.005


def test_noise_variance_and_rv():
    rng = np.random.default_rng(1)
    n, sigma, noise_sd = 200_000, 1e-5, 1e-4
    m = np.cumsum(rng.normal(0, sigma, n))
    p = np.exp(m + rng.normal(0, noise_sd, n))
    # noise dominates tick-by-tick, so Var(u) is recovered
    assert abs(noise_variance(p) / noise_sd**2 - 1) < 0.05
    # without noise RV ~ n * sigma^2
    assert abs(realized_variance(np.exp(m)) / (n * sigma**2) - 1) < 0.02


def test_spread_decomposition_identity():
    rng = np.random.default_rng(2)
    price, sign = 100 + rng.normal(size=100), rng.choice([-1, 1], 100)
    before, after = 100 + rng.normal(size=100), 100 + rng.normal(size=100)
    d = spread_decomposition(price, sign, before, after)
    assert np.allclose(d["effective"], d["realized"] + d["impact"])


def test_response_function_on_known_impact():
    # each trade moves the mid permanently by 0.1 in its direction
    rng = np.random.default_rng(3)
    s = rng.choice([-1, 1], 100_000)
    mids = np.concatenate([[0.0], np.cumsum(0.1 * s)])[:-1]  # mid just BEFORE each trade
    r = response_function(s, mids, [1, 5])
    assert np.allclose(r, [0.1, 0.1], atol=0.01)


def test_aggregate_trades_merges_same_ms_same_side():
    ts = pd.to_datetime([0, 0, 0, 1, 1, 2], unit="ms", utc=True)
    t = pd.DataFrame({"ts": ts, "price": [10, 10.1, 10.2, 10.0, 10.0, 10.0],
                      "qty": [1, 1, 2, 1, 1, 5], "sign": [1, 1, 1, 1, -1, -1]})
    a = aggregate_trades(t)
    assert list(a["n_fills"]) == [3, 1, 1, 1]
    assert list(a["qty"]) == [4, 1, 1, 5]
    assert a["last_price"].iloc[0] == 10.2
    assert np.isclose(a["vwap"].iloc[0], (10 + 10.1 + 2 * 10.2) / 4)

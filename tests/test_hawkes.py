import numpy as np

from micro import hawkes


def brute_force_loglik(mu, alpha, beta, t, T):
    lam = [mu + alpha * np.sum(np.exp(-beta * (ti - t[:i]))) for i, ti in enumerate(t)]
    integral = mu * T + alpha / beta * np.sum(1 - np.exp(-beta * (T - t)))
    return np.sum(np.log(lam)) - integral


def test_loglik_recursion_matches_brute_force():
    rng = np.random.default_rng(0)
    t = hawkes.simulate(1.0, 0.8, 2.0, 100, rng)
    assert np.isclose(hawkes.loglik((1.0, 0.8, 2.0), t, 100), brute_force_loglik(1.0, 0.8, 2.0, t, 100))


def test_simulation_has_right_mean_rate():
    rng = np.random.default_rng(1)
    mu, alpha, beta, T = 2.0, 1.5, 3.0, 20_000
    t = hawkes.simulate(mu, alpha, beta, T, rng)
    assert abs(len(t) / T / (mu / (1 - alpha / beta)) - 1) < 0.05


def test_fit_recovers_parameters():
    rng = np.random.default_rng(2)
    mu, alpha, beta, T = 1.0, 6.0, 10.0, 5_000
    t = hawkes.simulate(mu, alpha, beta, T, rng)
    f = hawkes.fit(t, T)
    assert abs(f.mu / mu - 1) < 0.15
    assert abs(f.branching_ratio - 0.6) < 0.05
    assert abs(f.beta / beta - 1) < 0.2
    assert f.lr_stat > 100


def test_compensator_gaps_are_exponential_under_true_model():
    rng = np.random.default_rng(3)
    t = hawkes.simulate(1.0, 0.8, 2.0, 5_000, rng)
    gaps = np.diff(hawkes.compensator(t, 1.0, 0.8, 2.0))
    assert abs(gaps.mean() - 1) < 0.05
    assert abs(gaps.std() - 1) < 0.05

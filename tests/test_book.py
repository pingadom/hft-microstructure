import numpy as np
import pandas as pd

from micro.book import imbalance, ofi_events, resample_last, spread_ticks, sum_in_buckets, weighted_mid


def test_ofi_hand_worked_cases():
    # Each row is the book after an update. Starting book: 100.0 x 5  /  100.1 x 3
    bid =     [100.0, 100.0, 100.0, 100.1, 100.0, 100.0]
    bid_qty = [5,     7,     4,     2,     6,     6]
    ask =     [100.1, 100.1, 100.1, 100.2, 100.2, 100.1]
    ask_qty = [3,     3,     3,     4,     4,     1]
    e = ofi_events(bid, bid_qty, ask, ask_qty)
    assert e[0] == 0
    assert e[1] == 2          # 2 added to the bid queue
    assert e[2] == -3         # 3 removed from the bid queue
    # bid improves to 100.1 x 2 (+2) and ask moves up to 100.2 x 4 (old ask queue of 3 wiped: +3)
    assert e[3] == 2 + 3
    # bid falls back to 100.0: old best bid of 2 gone (-2); ask unchanged at 100.2, size same
    assert e[4] == -2
    # ask improves down to 100.1 x 1: new ask liquidity = selling pressure (-1)
    assert e[5] == -1


def test_ofi_multilevel_shape_and_level0_matches_1d():
    rng = np.random.default_rng(0)
    n, L = 50, 5
    bid = 100 - 0.1 * np.arange(L) + rng.integers(-1, 2, size=(n, 1)) * 0.1
    ask = bid + 0.1
    bq, aq = rng.uniform(1, 5, (n, L)), rng.uniform(1, 5, (n, L))
    e = ofi_events(bid, bq, ask, aq)
    assert e.shape == (n, L)
    assert np.allclose(e[:, 0], ofi_events(bid[:, 0], bq[:, 0], ask[:, 0], aq[:, 0]))


def test_imbalance_and_weighted_mid():
    q = pd.DataFrame({"bid": [100.0], "bid_qty": [3.0], "ask": [100.1], "ask_qty": [1.0]})
    assert imbalance(q.bid_qty, q.ask_qty)[0] == 0.75
    # big bid queue -> estimate leans toward the ask
    assert np.isclose(weighted_mid(q)[0], 0.75 * 100.1 + 0.25 * 100.0)
    assert spread_ticks(q, 0.1)[0] == 1


def test_resample_last_uses_only_past():
    ts = np.array([10, 20, 30], dtype="datetime64[ms]")
    v = np.array([1.0, 2.0, 3.0])
    grid = np.array([5, 10, 15, 29, 30, 100], dtype="datetime64[ms]")
    out = resample_last(ts, v, grid)
    assert np.isnan(out[0])
    assert list(out[1:]) == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_sum_in_buckets_left_open_right_closed():
    ts = np.array([1, 2, 3, 4, 5], dtype="datetime64[ms]")
    v = np.array([1.0, 10, 100, 1000, 10000])
    grid = np.array([0, 2, 4, 6], dtype="datetime64[ms]")
    assert list(sum_in_buckets(ts, v, grid)) == [11, 1100, 10000]

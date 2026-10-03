"""Loading tick data into tidy pandas DataFrames.

Two sources:
  1. Historical Binance files (one full day of BTCUSDT futures, 2024-03-27):
     quotes = every change to the best bid/ask, trades = every fill.
  2. Our own live recording (scripts/record_binance.py): 20-level depth snapshots,
     best bid/ask updates and aggressive trades.

Raw CSVs are slow to parse, so the first load converts them to parquet in
data/processed/ and every later load reads the parquet (a few seconds).

Conventions used across the whole project:
  ts    timestamp, pandas datetime64[ms, UTC]
  sign  +1 if the aggressor (the person crossing the spread) was a BUYER, -1 if a seller
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_DATE = "2024-03-27"
TICK = 0.1  # BTCUSDT futures minimum price increment, in USDT


def _read_zipped_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: .venv\\Scripts\\python scripts\\download_binance.py")
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            # The pyarrow engine is multi-threaded: ~10x faster than the default parser.
            return pd.read_csv(f, engine="pyarrow")


def _is_fresh(cache: Path, sources: list[Path]) -> bool:
    """True if the cache exists and is newer than every source file."""
    return cache.exists() and all(cache.stat().st_mtime >= s.stat().st_mtime for s in sources)


def _cached(name: str, build, sources: list[Path] = ()) -> pd.DataFrame:
    """Return data/processed/<name>.parquet, building it with build() the first time
    (or again if any of the source files changed since)."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    if _is_fresh(path, list(sources)):
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df


# ---------------------------------------------------------------------------
# Historical Binance files
# ---------------------------------------------------------------------------

def load_quotes(symbol: str = DEFAULT_SYMBOL, date: str = DEFAULT_DATE) -> pd.DataFrame:
    """Best bid/ask ("level 1") updates. One row per change to price OR size on either side.

    Columns: ts, bid, bid_qty, ask, ask_qty
    """

    def build():
        raw = _read_zipped_csv(RAW / f"{symbol}-bookTicker-{date}.zip")
        raw = raw.sort_values("update_id", kind="stable")
        df = pd.DataFrame({
            # transaction_time = when the matching engine changed the book. We use it
            # (not event_time = when Binance sent the message) so quotes line up with trades.
            "ts": pd.to_datetime(raw["transaction_time"].to_numpy(), unit="ms", utc=True),
            "bid": raw["best_bid_price"].to_numpy(),
            "bid_qty": raw["best_bid_qty"].to_numpy(),
            "ask": raw["best_ask_price"].to_numpy(),
            "ask_qty": raw["best_ask_qty"].to_numpy(),
        })
        # Drop rows where nothing at level 1 actually changed (rare duplicates).
        cols = ["bid", "bid_qty", "ask", "ask_qty"]
        changed = (df[cols].diff().abs().sum(axis=1) > 0) | (np.arange(len(df)) == 0)
        return df[changed].reset_index(drop=True)

    return _cached(f"{symbol}-{date}-quotes", build)


def load_trades(symbol: str = DEFAULT_SYMBOL, date: str = DEFAULT_DATE) -> pd.DataFrame:
    """Every fill. One aggressive order that eats through several resting orders shows up
    as several rows with the same timestamp and sign.

    Columns: ts, price, qty, sign, id
    """

    def build():
        raw = _read_zipped_csv(RAW / f"{symbol}-trades-{date}.zip").sort_values("id")
        return pd.DataFrame({
            "ts": pd.to_datetime(raw["time"].to_numpy(), unit="ms", utc=True),
            "price": raw["price"].to_numpy(),
            "qty": raw["qty"].to_numpy(),
            # is_buyer_maker = True means the BUYER was resting (the maker), so the
            # SELLER crossed the spread: a sell-initiated trade, sign -1.
            "sign": np.where(raw["is_buyer_maker"].to_numpy(), -1, 1).astype(np.int8),
            "id": raw["id"].to_numpy(),
        })

    return _cached(f"{symbol}-{date}-trades", build)


def aggregate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive fills with the same timestamp and side into one "market order".

    If someone sends a 5 BTC market buy and it fills against 12 resting orders, the
    trades file has 12 rows. For questions about *decisions* (how often do people
    trade? do buys follow buys?) we want 1 row. This is an approximation: two
    different people buying in the same millisecond get merged, which is rare.

    Columns: ts, sign, qty, first_price, last_price, vwap, n_fills
    """
    ts = trades["ts"].to_numpy()
    sign = trades["sign"].to_numpy()
    # A new order starts whenever the timestamp or the side changes.
    new = np.ones(len(trades), dtype=bool)
    new[1:] = (ts[1:] != ts[:-1]) | (sign[1:] != sign[:-1])
    group = np.cumsum(new) - 1

    notional = trades["price"].to_numpy() * trades["qty"].to_numpy()
    g = pd.DataFrame({"group": group, "qty": trades["qty"].to_numpy(), "notional": notional,
                      "price": trades["price"].to_numpy()})
    agg = g.groupby("group").agg(qty=("qty", "sum"), notional=("notional", "sum"),
                                 first_price=("price", "first"), last_price=("price", "last"),
                                 n_fills=("qty", "size"))
    out = pd.DataFrame({
        "ts": trades["ts"].to_numpy()[new],
        "sign": sign[new],
        "qty": agg["qty"].to_numpy(),
        "first_price": agg["first_price"].to_numpy(),
        "last_price": agg["last_price"].to_numpy(),
        "vwap": (agg["notional"] / agg["qty"]).to_numpy(),
        "n_fills": agg["n_fills"].to_numpy(),
    })
    return out


# ---------------------------------------------------------------------------
# Live recordings (scripts/record_binance.py)
# ---------------------------------------------------------------------------

LIVE = RAW / "live"


def _live_files(kind: str) -> list[Path]:
    files = sorted(LIVE.glob(f"*_{kind}.jsonl"))
    if not files:
        raise FileNotFoundError(f"No live {kind} files in {LIVE}. Run scripts/record_binance.py first.")
    return files


def _read_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # a half-written last line if the recorder was killed
                    pass
    return rows


def load_live_quotes() -> pd.DataFrame:
    """Live best bid/ask stream. Same columns as load_quotes()."""

    files = _live_files("bookTicker")

    def build():
        rows = _read_jsonl(files)
        df = pd.DataFrame({
            "u": [r["u"] for r in rows],
            "ts": pd.to_datetime([r["T"] for r in rows], unit="ms", utc=True),
            "bid": [float(r["b"]) for r in rows],
            "bid_qty": [float(r["B"]) for r in rows],
            "ask": [float(r["a"]) for r in rows],
            "ask_qty": [float(r["A"]) for r in rows],
        })
        df = df.drop_duplicates("u").sort_values("u")
        return df.drop(columns="u").reset_index(drop=True)

    return _cached("live-quotes", build, files)


def load_live_trades() -> pd.DataFrame:
    """Live aggTrade stream: one row per (taker order, price level).

    Columns: ts, price, qty, sign
    """

    files = _live_files("aggTrade")

    def build():
        rows = _read_jsonl(files)
        df = pd.DataFrame({
            "a": [r["a"] for r in rows],
            "ts": pd.to_datetime([r["T"] for r in rows], unit="ms", utc=True),
            "price": [float(r["p"]) for r in rows],
            "qty": [float(r["q"]) for r in rows],
            "sign": np.where([r["m"] for r in rows], -1, 1).astype(np.int8),
        })
        return df.drop_duplicates("a").sort_values("a").drop(columns="a").reset_index(drop=True)

    return _cached("live-trades", build, files)


def load_live_depth(levels: int = 20) -> dict:
    """Live 20-level depth snapshots (one every 100 ms).

    Returns a dict of numpy arrays:
      ts                  (n,)          datetime64[ms], matching-engine time
      event_ts, recv_ts   (n,)          when Binance sent it / when we received it
      bid_px, bid_qty     (n, levels)   level 0 = best bid, level 1 = next one down, ...
      ask_px, ask_qty     (n, levels)   level 0 = best ask, level 1 = next one up, ...

    Arrays rather than a DataFrame because "price at level k" is naturally a matrix.
    """
    cache = PROCESSED / "live-depth.npz"
    files = _live_files("depth20")
    if not _is_fresh(cache, files):
        rows = _read_jsonl(files)
        rows = [r for r in rows if len(r["b"]) >= levels and len(r["a"]) >= levels]
        rows.sort(key=lambda r: r["u"])
        b = np.array([r["b"][:levels] for r in rows], dtype=float)  # (n, levels, 2)
        a = np.array([r["a"][:levels] for r in rows], dtype=float)
        ts = np.array([r["T"] for r in rows], dtype="int64")
        recv = np.array([r["_recv"] for r in rows], dtype="int64")
        event = np.array([r["E"] for r in rows], dtype="int64")
        keep = np.ones(len(ts), dtype=bool)
        keep[1:] = np.diff([r["u"] for r in rows]) != 0  # drop duplicate snapshots
        PROCESSED.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ts=ts[keep], b=b[keep], a=a[keep], recv=recv[keep], event=event[keep])
    z = np.load(cache)
    return {
        "ts": z["ts"].astype("datetime64[ms]"),
        # Binance's send time and our receive time (our clock), for latency checks
        "event_ts": z["event"].astype("datetime64[ms]"), "recv_ts": z["recv"].astype("datetime64[ms]"),
        "bid_px": z["b"][:, :, 0], "bid_qty": z["b"][:, :, 1],
        "ask_px": z["a"][:, :, 0], "ask_qty": z["a"][:, :, 1],
    }

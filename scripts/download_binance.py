"""Download one day of Binance USD-M futures tick data from data.binance.vision.

Two files per day:
    bookTicker  every change to the best bid/ask (price + size on each side)
    trades      every trade, with a flag telling us which side was the aggressor

Binance stopped publishing bookTicker files after March 2024, so the default day is
2024-03-27 (a normal Wednesday). Trades files exist for every day up to yesterday.

Usage:
    .venv\\Scripts\\python scripts\\download_binance.py
    .venv\\Scripts\\python scripts\\download_binance.py --symbol ETHUSDT --date 2024-03-26

Each zip is checked against the SHA-256 checksum Binance publishes next to it.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://data.binance.vision/data/futures/um/daily"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(kind: str, symbol: str, date: str) -> Path:
    name = f"{symbol}-{kind}-{date}.zip"
    url = f"{BASE}/{kind}/{symbol}/{name}"
    dest = RAW / name
    expected = requests.get(url + ".CHECKSUM", timeout=30)
    expected.raise_for_status()
    expected_hash = expected.text.split()[0]

    if dest.exists() and sha256(dest) == expected_hash:
        print(f"already have {name}")
        return dest

    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if sha256(dest) != expected_hash:
        dest.unlink()
        sys.exit(f"checksum mismatch for {name}; deleted it, please re-run")
    print(f"  checksum OK")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--date", default="2024-03-27")
    args = p.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for kind in ("bookTicker", "trades"):
        download(kind, args.symbol.upper(), args.date)


if __name__ == "__main__":
    main()

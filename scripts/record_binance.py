"""Record live Binance USD-M futures market data to disk.

Binance stopped publishing historical order-book files in 2024, so if we want
multi-level depth we have to record it ourselves. This script subscribes to three
public websocket streams for one symbol and writes every message to JSON-lines files:

    depth20@100ms  top 20 price levels on each side, snapshot every 100 ms
    bookTicker     every change to the best bid / best ask (level 1), real time
    aggTrade       every aggressive order, with its side

No API key is needed: this is public market data.

Usage (from the project folder):
    .venv\\Scripts\\python scripts\\record_binance.py --minutes 120
    .venv\\Scripts\\python scripts\\record_binance.py --symbol ethusdt --minutes 30

Then convert to parquet with:
    .venv\\Scripts\\python scripts\\convert_recording.py
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "live"
# Since 2026 Binance serves order-book streams and trade streams from different paths,
# so we hold one connection for each.
CONNECTIONS = {
    "wss://fstream.binance.com/public/stream?streams=": ["depth20@100ms", "bookTicker"],
    "wss://fstream.binance.com/market/stream?streams=": ["aggTrade"],
}
STREAMS = [s for group in CONNECTIONS.values() for s in group]


async def listen(base: str, streams: list[str], symbol: str, files: dict, counts: dict, deadline: float) -> None:
    """Hold one websocket open until the deadline, reconnecting after any error."""
    url = base + "/".join(f"{symbol}@{s}" for s in streams)
    backoff = 1
    while time.time() < deadline:
        try:
            async with websockets.connect(url, open_timeout=10, ping_interval=20) as ws:
                backoff = 1
                print(f"connected: {url}", flush=True)
                while time.time() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    recv_ms = int(time.time() * 1000)
                    msg = json.loads(raw)
                    stream = msg["stream"].split("@", 1)[1]
                    data = msg["data"]
                    data["_recv"] = recv_ms  # our local receive time, useful for latency checks
                    files[stream].write(json.dumps(data, separators=(",", ":")) + "\n")
                    counts[stream] += 1
        except Exception as exc:  # network blips happen; log the gap and reconnect
            print(f"{time.strftime('%H:%M:%S')} {streams} disconnected ({exc!r}); retrying in {backoff}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def report(files: dict, counts: dict, deadline: float) -> None:
    while time.time() < deadline:
        await asyncio.sleep(60)
        for f in files.values():
            f.flush()
        left = (deadline - time.time()) / 60
        print(f"{time.strftime('%H:%M:%S')} counts={counts} ({left:.0f} min left)", flush=True)


async def record(symbol: str, minutes: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    # One file per stream. Plain text (not gzip) so a crash never corrupts the file.
    files = {
        s: open(OUT_DIR / f"{symbol}_{stamp}_{s.split('@')[0]}.jsonl", "a", encoding="utf-8")
        for s in STREAMS
    }
    counts = {s: 0 for s in STREAMS}
    deadline = time.time() + minutes * 60
    await asyncio.gather(
        *(listen(base, streams, symbol, files, counts, deadline) for base, streams in CONNECTIONS.items()),
        report(files, counts, deadline),
    )
    for f in files.values():
        f.close()
    print(f"done. counts={counts}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="btcusdt")
    p.add_argument("--minutes", type=float, default=60)
    args = p.parse_args()
    asyncio.run(record(args.symbol.lower(), args.minutes))


if __name__ == "__main__":
    main()

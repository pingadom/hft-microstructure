"""Optional: loader for LOBSTER Nasdaq data (https://lobsterdata.com).

LOBSTER reconstructs the full Nasdaq order book from every order message. Their sample
files now need a request (proof of purchase of their book, institutional email) and
their terms forbid publishing anything derived from them, so nothing in this public
repo uses LOBSTER data. If you get access, put the two CSVs in data/raw/lobster/ and use:

    msgs, book = load_lobster("data/raw/lobster/AAPL_..._message_10.csv",
                              "data/raw/lobster/AAPL_..._orderbook_10.csv")

Then all the tools in micro/ work on it: ofi_events() on book["bid_px"] etc.
Keep your results private, per their terms.

File format (from the LOBSTER documentation):
  message file, one row per event, no header:
      time (seconds after midnight, with nanosecond decimals), type, order id,
      size (shares), price (dollars * 10000), direction (+1 buy order, -1 sell order)
  orderbook file, one row per event, no header, the book AFTER that event:
      ask price 1, ask size 1, bid price 1, bid size 1, ask price 2, ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_TYPES = {
    1: "submit",          # new limit order
    2: "cancel_partial",  # part of an order cancelled
    3: "cancel",          # whole order deleted
    4: "exec_visible",    # a visible limit order was hit by a market order
    5: "exec_hidden",     # a hidden order was hit
    6: "cross",           # auction cross trade
    7: "halt",            # trading halt indicator
}


def load_lobster(message_path, orderbook_path) -> tuple[pd.DataFrame, dict]:
    msgs = pd.read_csv(message_path, header=None,
                       names=["time", "type", "order_id", "size", "price", "direction"])
    msgs["price"] = msgs["price"] / 10_000
    msgs["event"] = msgs["type"].map(EVENT_TYPES)
    # For executions, "direction" is the side of the RESTING order that got hit,
    # so the aggressor's sign is the opposite.
    is_exec = msgs["type"].isin([4, 5])
    msgs["aggressor_sign"] = np.where(is_exec, -msgs["direction"], 0)

    ob = pd.read_csv(orderbook_path, header=None).to_numpy(dtype=float)
    levels = ob.shape[1] // 4
    ob = ob.reshape(len(ob), levels, 4)  # [ask px, ask size, bid px, bid size] per level
    book = {
        "time": msgs["time"].to_numpy(),
        "ask_px": ob[:, :, 0] / 10_000, "ask_qty": ob[:, :, 1],
        "bid_px": ob[:, :, 2] / 10_000, "bid_qty": ob[:, :, 3],
    }
    return msgs, book

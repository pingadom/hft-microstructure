import numpy as np

from micro.lobster import load_lobster


def test_load_lobster_synthetic(tmp_path):
    # Two events, 2 levels. Prices are in dollars * 10000, as in real LOBSTER files.
    (tmp_path / "msg.csv").write_text(
        "34200.000000001,1,111,100,1000000,1\n"
        "34200.5,4,222,50,1000100,-1\n"
    )
    (tmp_path / "ob.csv").write_text(
        "1000100,200,1000000,100,1000200,300,999900,400\n"
        "1000200,300,1000000,100,1000300,10,999900,400\n"
    )
    msgs, book = load_lobster(tmp_path / "msg.csv", tmp_path / "ob.csv")
    assert list(msgs["event"]) == ["submit", "exec_visible"]
    assert msgs["price"].iloc[0] == 100.0
    # executed resting SELL order (-1) means the aggressor was a BUYER
    assert list(msgs["aggressor_sign"]) == [0, 1]
    assert book["ask_px"].shape == (2, 2)
    assert np.allclose(book["ask_px"][0], [100.01, 100.02])
    assert np.allclose(book["bid_qty"][1], [100, 400])

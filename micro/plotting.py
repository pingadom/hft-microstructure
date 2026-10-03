"""Shared plot style so every chapter looks the same."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"

BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#6b7280"


def style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4.2),
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "font.size": 10,
        "legend.frameon": False,
    })


def save(fig, name: str) -> None:
    """Save to figures/<name>.png (these are the images the README shows)."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", dpi=110)

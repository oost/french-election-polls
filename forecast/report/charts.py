"""Forecast charts (matplotlib, static). Candidate identity is carried by
direct labels; colour is a single neutral series plus the reference palette's
first slots where two measures sit side by side."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from forecast.backtest.charts import GRID, INK, MUTED, _style

RUNOFF, WIN = "#2a78d6", "#eb6834"      # reference categorical slots 1–2


def presidential_probabilities(candidates: pl.DataFrame, path: Path | None = None, top: int = 10):
    """Grouped horizontal bars: P(reach runoff) and P(win) per candidate."""
    df = candidates.sort("p_runoff", descending=True).head(top).reverse()
    y = np.arange(df.height)
    fig, ax = plt.subplots(figsize=(8, 0.45 * df.height + 1.2))
    ax.barh(y + 0.2, df["p_runoff"], height=0.38, color=RUNOFF, label="reaches the runoff",
            edgecolor="white", linewidth=2)
    ax.barh(y - 0.2, df["p_win"], height=0.38, color=WIN, label="wins", edgecolor="white", linewidth=2)
    for yi, (pr, pw) in zip(y, zip(df["p_runoff"], df["p_win"])):
        ax.text(pr + 0.01, yi + 0.2, f"{pr:.0%}", va="center", fontsize=8, color=INK)
        ax.text(pw + 0.01, yi - 0.2, f"{pw:.0%}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, df["name"].to_list())
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    _style(ax, "Chance of reaching the runoff and of winning", "probability", "")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def presidential_shares(candidates: pl.DataFrame, path: Path | None = None, top: int = 12):
    """First-round share: median with 50% and 90% simulated ranges (given the candidate runs)."""
    df = candidates.sort("share_q50", descending=True).head(top).reverse()
    y = np.arange(df.height)
    fig, ax = plt.subplots(figsize=(8, 0.42 * df.height + 1.2))
    ax.hlines(y, df["share_q05"], df["share_q95"], color=GRID, linewidth=6)
    ax.hlines(y, df["share_q25"], df["share_q75"], color=MUTED, linewidth=6)
    ax.scatter(df["share_q50"], y, s=60, color=INK, zorder=3, edgecolors="white", linewidths=2)
    for yi, m in zip(y, df["share_q50"]):
        ax.text(m, yi + 0.3, f"{m:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_yticks(y, df["name"].to_list())
    _style(ax, "First-round vote share (if running): median, 50% and 90% ranges", "% of votes cast", "")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def legislative_seats(totals: pl.DataFrame, path: Path | None = None, majority: int = 289):
    df = totals.sort("mean_seats")
    y = np.arange(df.height)
    fig, ax = plt.subplots(figsize=(8, 0.5 * df.height + 1.2))
    ax.hlines(y, df["q05"], df["q95"], color=GRID, linewidth=6)
    ax.scatter(df["mean_seats"], y, s=60, color=INK, zorder=3, edgecolors="white", linewidths=2)
    for yi, m, p in zip(y, df["mean_seats"], df["p_majority"]):
        ax.text(m, yi + 0.3, f"{m:.0f}" + (f"  (P(majority) {p:.0%})" if p > 0.005 else ""), ha="center",
                fontsize=8, color=INK)
    ax.axvline(majority, color=MUTED, linestyle=":", linewidth=1)
    ax.text(majority, df.height - 0.4, " majority 289", color=MUTED, fontsize=8)
    ax.set_yticks(y, df["unit"].to_list())
    _style(ax, "Seats won: mean and 90% range", "seats", "")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig

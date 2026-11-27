"""Forecast charts (matplotlib, static). Candidate identity is carried by
direct labels; colour is a single neutral series plus the reference palette's
first slots where two measures sit side by side."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from forecast.backtest.charts import GRID, INK, MUTED, _style

RUNOFF, WIN = "#2a78d6", "#eb6834"      # reference categorical slots 1–2
# reference sequential ramp (blue, steps 100–700): light = near zero
BLUES = LinearSegmentedColormap.from_list("blues", [
    "#cce3fd", "#b6d5f8", "#9dc8f7", "#84b9f3", "#6aabf1", "#519ced", "#338ced",
    "#257de0", "#2370cd", "#1b64be", "#1757ac", "#0f4a96", "#0e3d7e"])
MEASURES = {"p_win": "Chance of winning", "p_runoff": "Chance of reaching the runoff"}


def conditional_matrix(summary, slot: str = "centre_right", measure: str = "p_win",
                       path: Path | None = None):
    """Each candidate's chance (rows) given each ballot option of a scenario
    slot (columns), from `PresSummary.conditional_matrix`. Blank cells: not on
    that ballot. The last row is the slot's candidates taken together."""
    m = summary.conditional_matrix(slot, measure)
    heads = summary.conditional.filter(pl.col("slot") == slot).unique("ballot", maintain_order=True)
    ballots = heads["ballot"].to_list()
    vals = m.select(ballots).to_numpy().astype(float)
    nr, nc = vals.shape
    fig, ax = plt.subplots(figsize=(max(0.95 * nc + 2.4, 7.0), 0.42 * nr + 2.2))
    norm = PowerNorm(0.5, vmin=0, vmax=1)   # spreads the small probabilities most cells hold
    cmap = BLUES.with_extremes(bad="white")
    ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")
    for i in range(nr):
        for j in range(nc):
            v = vals[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            txt = "<1%" if 0 < v < 0.005 else f"{v:.0%}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if norm(v) > 0.6 else INK)
    # 2px surface gaps between cells, a wider one above the combined row
    ax.set_xticks(np.arange(nc + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nr + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.axhline(nr - 1.5, color="white", linewidth=6)
    ax.tick_params(which="both", length=0)
    labels = [b.replace(" + ", " +\n") + f"\n{p:.0%} likely" for b, p in zip(ballots, heads["p_ballot"])]
    ax.set_xticks(np.arange(nc), labels, fontsize=8, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(nr), m["name"].to_list(), fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    label = slot.upper() if len(slot) <= 3 else slot.replace("_", "/")
    ax.set_title(f"{MEASURES.get(measure, measure)}, by {label} ballot",
                 loc="left", fontsize=11, fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


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
    # headroom for the top row's label; the majority label sits below the last row
    ax.set_ylim(-0.9, df.height - 0.25)
    ax.text(majority, -0.7, f" majority {majority}", color=MUTED, fontsize=8, va="center")
    ax.set_yticks(y, df["unit"].to_list())
    _style(ax, "Seats won: mean and 90% range", "seats", "")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def turnout_history(summary, path: Path | None = None):
    """Turnout in both rounds of past presidential elections, and the forecast:
    median with its 90% interval, from `PresSummary.turnout`."""
    from forecast.simulate.turnout import turnout_history as history
    h = history().filter(pl.col("year") < summary.year)
    fc = {r["measure"]: r for r in summary.turnout.iter_rows(named=True)}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for rnd, label, key, color, dx in ((1, "first round", "First round turnout", RUNOFF, -0.7),
                                       (2, "runoff", "Runoff turnout", WIN, 0.7)):
        d = h.filter(pl.col("round") == rnd)
        ax.plot(d["year"], 100 * d["turnout"], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.5, label=label)
        r = fc[key]
        x = summary.year + dx
        ax.vlines(x, r["q05"], r["q95"], color=color, linewidth=2, alpha=0.45)
        ax.vlines(x, r["q25"], r["q75"], color=color, linewidth=5, alpha=0.6)
        ax.scatter([x], [r["q50"]], s=55, color=color, edgecolors="white", linewidths=1.5, zorder=3)
        ax.plot([d["year"][-1], x], [100 * d["turnout"][-1], r["q50"]], color=color, linewidth=1, linestyle=":")
    top = max(fc["First round turnout"]["q95"], fc["Runoff turnout"]["q95"])
    ax.annotate(f"{summary.year} forecast\nmedian, 50% and 90% ranges", (summary.year, top), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(h["year"].unique().to_list() + [summary.year])
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlim(h["year"].min() - 2, summary.year + 4)
    ax.set_ylim(60, 95)
    _style(ax, "Turnout in presidential elections", "", "% of registered voters")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig

"""Backtest charts: error by horizon, weights over time, calibration."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from forecast.backtest.metrics import calibration

# Reference categorical palette (dataviz skill), slots 1–3, fixed order.
SERIES = {"polls": "#2a78d6", "fundamentals": "#eb6834", "ensemble": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def error_by_horizon(archive: pl.DataFrame, path: Path) -> Path:
    df = (archive.with_columns((100 * pl.col("mu_polls").exp()).alias("sp"),
                               (100 * pl.col("mu_fundamentals").exp()).alias("sf"))
          .group_by("days_out").agg(
              (pl.col("sp") - pl.col("result_share")).abs().mean().alias("polls"),
              (pl.col("sf") - pl.col("result_share")).abs().mean().alias("fundamentals"),
              (pl.col("share_ens") - pl.col("result_share")).abs().mean().alias("ensemble"))
          .sort("days_out"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # legend only: the polls and ensemble lines nearly coincide, so direct labels would collide
    for k, color in SERIES.items():
        ax.plot(df["days_out"], df[k], color=color, linewidth=2, label=k)
    ax.invert_xaxis()
    _style(ax, "Out-of-sample error by days to the first round", "days before the first round",
           "mean absolute error (points)")
    ax.legend(frameon=False, loc="center right")   # the empty band between the two groups of lines
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def weights_over_time(archive: pl.DataFrame, path: Path) -> Path:
    df = archive.group_by("days_out").agg(pl.col("w_polls").mean(), pl.col("w_fundamentals").mean()).sort("days_out")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = df["days_out"].to_numpy()
    wp, wf = df["w_polls"].to_numpy(), df["w_fundamentals"].to_numpy()
    ax.stackplot(x, wp, wf, colors=[SERIES["polls"], SERIES["fundamentals"]], labels=["polls", "fundamentals"],
                 edgecolor="white", linewidth=2)
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    _style(ax, "Ensemble weight by component", "days before the first round", "mean weight")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def calibration_chart(archive: pl.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (prob, outcome, title) in zip(axes, (("p_runoff", "made_runoff", "Reaching the runoff"),
                                                 ("p_win", "won", "Winning"))):
        cal = calibration(archive, prob, outcome).filter(pl.col("n") > 0)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
        ax.errorbar(cal["mean_p"], cal["freq"],
                    yerr=[(cal["freq"] - cal["lo"]).clip(lower_bound=0), (cal["hi"] - cal["freq"]).clip(lower_bound=0)],
                    fmt="o", color=SERIES["ensemble"], ecolor=SERIES["ensemble"], elinewidth=1.5,
                    markersize=8, markeredgecolor="white", markeredgewidth=2, capsize=0)
        for r in cal.iter_rows(named=True):
            ax.annotate(f"n={r['n']}", (r["mean_p"], r["freq"]), xytext=(6, -10), textcoords="offset points",
                        fontsize=8, color=MUTED)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        _style(ax, title, "forecast probability", "observed frequency")
    fig.suptitle("Calibration, all snapshots 2002–2022 (95% Wilson intervals)", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_all(archive: pl.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [error_by_horizon(archive, out_dir / "error_by_horizon.png"),
            weights_over_time(archive, out_dir / "weights_over_time.png"),
            calibration_chart(archive, out_dir / "calibration.png")]

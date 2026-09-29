"""Backtest metrics mirroring the post's § 2 and § 5."""
from __future__ import annotations

import numpy as np
import polars as pl

CALIBRATION_BUCKETS = (0.0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0000001)
HORIZON_BINS = ((0, 14, "0–14"), (15, 45, "15–45"), (46, 90, "46–90"), (91, 200, "91–156"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def _horizon_bin(expr: pl.Expr) -> pl.Expr:
    out = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, label in reversed(HORIZON_BINS):
        out = pl.when((expr >= lo) & (expr <= hi)).then(pl.lit(label)).otherwise(out)
    return out


def share_errors(archive: pl.DataFrame) -> pl.DataFrame:
    """Mean absolute error (points) of each component and the ensemble, by horizon."""
    df = archive.with_columns(
        (100 * pl.col("mu_polls").exp()).alias("share_polls"),
        (100 * pl.col("mu_fundamentals").exp()).alias("share_fundamentals"),
        _horizon_bin(pl.col("days_out")).alias("horizon"),
    )
    return (
        df.group_by("horizon")
        .agg(
            (pl.col("share_polls") - pl.col("result_share")).abs().mean().alias("mae_polls"),
            (pl.col("share_fundamentals") - pl.col("result_share")).abs().mean().alias("mae_fundamentals"),
            (pl.col("share_ens") - pl.col("result_share")).abs().mean().alias("mae_ensemble"),
            pl.col("w_polls").mean().alias("mean_w_polls"),
            pl.len().alias("n"),
        )
        .sort("horizon", descending=True)
    )


def coverage(archive: pl.DataFrame) -> pl.DataFrame:
    """Share of results inside the simulated 50% and 90% intervals."""
    return archive.select(
        ((pl.col("result_share") >= pl.col("q25")) & (pl.col("result_share") <= pl.col("q75"))).mean().alias("cov50"),
        ((pl.col("result_share") >= pl.col("q05")) & (pl.col("result_share") <= pl.col("q95"))).mean().alias("cov90"),
        pl.len().alias("n"),
    )


def calibration(archive: pl.DataFrame, prob: str, outcome: str) -> pl.DataFrame:
    """Predicted-probability buckets vs realised frequency (with Wilson 95% intervals)."""
    rows = []
    p = archive[prob].to_numpy()
    y = archive[outcome].to_numpy().astype(bool)
    for lo, hi in zip(CALIBRATION_BUCKETS[:-1], CALIBRATION_BUCKETS[1:]):
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        k = int(y[m].sum())
        lo_ci, hi_ci = wilson(k, n)
        rows.append({"bucket": f"{lo:.0%}–{min(hi, 1):.0%}", "mean_p": float(p[m].mean()) if n else np.nan,
                     "freq": k / n if n else np.nan, "lo": lo_ci, "hi": hi_ci, "n": n})
    return pl.DataFrame(rows)


def scores(archive: pl.DataFrame) -> dict[str, float]:
    """Brier score and log loss for reaching the runoff and winning; share of
    calls right (favourite to win), overall and excluding toss-ups (< 65%)."""
    out = {}
    for prob, outcome in (("p_runoff", "made_runoff"), ("p_win", "won")):
        p = np.clip(archive[prob].to_numpy(), 1e-4, 1 - 1e-4)
        y = archive[outcome].to_numpy().astype(float)
        out[f"brier_{prob}"] = float(np.mean((p - y) ** 2))
        out[f"logloss_{prob}"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    fav = archive.sort("p_win", descending=True).group_by("cycle", "days_out", maintain_order=True).first()
    out["winner_called"] = float(fav["won"].mean())
    firm = fav.filter(pl.col("p_win") >= 0.65)
    out["winner_called_excl_tossups"] = float(firm["won"].mean()) if firm.height else np.nan
    out["n_snapshots"] = float(fav.height)
    return out

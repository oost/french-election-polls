"""Expected change to election day — the French analog of the post's
midterm swing (post § 3.1.3; PLAN.md › Presidential step 9).

French first rounds consolidate late: voters desert small and mid-sized
candidates for the leading candidate of their camp (*vote utile*). In the
2002–2022 backtest, candidates polling above 10% beat the ensemble by 1.6
points on average and those at 3–10% fell 1.2 points short. The drift is

    E[log result − log poll] = β(h) · (log s − mean log s),   β(h) = b0 + b1·√h

fitted on earlier cycles only (ridge-shrunk toward zero), applied to the polls
component's mean and renormalised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class DriftModel:
    b0: float = 0.0
    b1: float = 0.0
    n_rows: int = 0

    def beta(self, h: float) -> float:
        return self.b0 + self.b1 * np.sqrt(max(h, 0.0))

    def apply(self, mu: np.ndarray, h: float) -> np.ndarray:
        rel = mu - mu.mean()
        out = mu + self.beta(h) * rel
        return out - np.log(np.exp(out).sum())


def fit_drift(archive: pl.DataFrame, ridge: float = 50.0) -> DriftModel:
    """Least squares of the polls component's log error on β(h)·relative size,
    with a ridge penalty (in units of rows) pulling (b0, b1) toward zero."""
    if archive.is_empty() or "mu_polls_raw" not in archive.columns:
        return DriftModel()
    df = archive.with_columns(
        (pl.col("mu_polls_raw") - pl.col("mu_polls_raw").mean().over("cycle", "days_out")).alias("rel"),
        (pl.col("result_log") - pl.col("mu_polls_raw")).alias("e"),
    )
    # the log-share error is only identified up to a constant per snapshot (renormalisation)
    df = df.with_columns((pl.col("e") - pl.col("e").mean().over("cycle", "days_out")).alias("e"))
    rel = df["rel"].to_numpy()
    h = df["days_out"].to_numpy().astype(float)
    X = np.column_stack([rel, rel * np.sqrt(h)])
    y = df["e"].to_numpy()
    A = X.T @ X + ridge * np.eye(2)
    b = np.linalg.solve(A, X.T @ y)
    return DriftModel(float(b[0]), float(b[1]), df.height)

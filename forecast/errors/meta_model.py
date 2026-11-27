"""Error meta-model (post § 4.1.1) — fit on the backtest archive, used by
the simulator.

Training rows are ensemble errors on the log-share scale,
err = log(result share) − ensemble mean, from strictly out-of-sample
snapshots. Shared shocks are grouped by (cycle, snapshot, level group), e.g.
all candidates of one bloc in one weekly snapshot of one cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from forecast.stan.build import cached_sample


@dataclass(frozen=True)
class ErrorModel:
    """Posterior means of the meta-model's parameters."""
    levels: tuple[str, ...]
    a: dict[str, float]
    b: dict[str, float]
    m0: float
    m1: float
    m2: float
    nu: float
    n_cycles: int
    source: str = "stan"
    extra: dict = field(default_factory=dict)

    def shock_sd(self, level: str, h: float) -> float:
        """Shared-shock sd at horizon h. Beyond the longest horizon in the
        training archive the √h growth is not extrapolated (the components'
        own stated uncertainty already grows with h)."""
        h_max = self.extra.get("max_days", np.inf)
        return self.a[level] + self.b[level] * np.sqrt(min(max(h, 0.0), h_max))

    def individual_sd(self, sd_ens: np.ndarray, n_polls: np.ndarray | float, disagree: np.ndarray | float) -> np.ndarray:
        scarcity = 1.0 / (1.0 + np.asarray(n_polls, dtype=float))
        return np.asarray(sd_ens) * self.m0 * (1 + self.m1 * scarcity + self.m2 * np.asarray(disagree))


def prior_error_model(levels: tuple[str, ...], nu: float = 3.0) -> ErrorModel:
    """Fixed weakly informative table used before any out-of-sample archive
    exists (the post's disclosed exception): no shared shocks beyond a small
    bloc term, individual errors as stated by the ensemble."""
    return ErrorModel(levels, {l: 0.02 for l in levels}, {l: 0.0 for l in levels}, 1.0, 0.0, 0.0, nu, 0, "prior")


def fit_error_model(
    archive: pl.DataFrame,
    levels: tuple[str, ...],
    nu_floor: float = 3.0,
    thin_every: int = 1,
    **sample_kwargs,
) -> ErrorModel:
    """Fit on archive rows with columns: cycle, days_out, err, sd_ens,
    n_polls, disagree, and one column per level (group label, null = none)."""
    df = archive.filter(pl.col("err").is_finite() & (pl.col("sd_ens") > 0))
    if thin_every > 1:
        snaps = sorted(df["days_out"].unique().to_list())
        keep = set(snaps[::thin_every])
        df = df.filter(pl.col("days_out").is_in(list(keep)))
    n_cycles = df["cycle"].n_unique()
    nu = max(nu_floor, float(n_cycles))
    groups: dict[tuple, int] = {}
    level_of, sqrt_h = [], []
    gid = np.zeros((df.height, len(levels)), dtype=int)
    rows = df.select("cycle", "days_out", *levels).iter_rows()
    for n, row in enumerate(rows):
        cycle, days = row[0], row[1]
        for li, label in enumerate(row[2:]):
            if label is None:
                continue
            key = (li, cycle, days, label)
            if key not in groups:
                groups[key] = len(groups) + 1
                level_of.append(li + 1)
                sqrt_h.append(np.sqrt(max(days, 0)))
            gid[n, li] = groups[key]
    data = {
        "N": df.height, "err": df["err"].to_numpy(), "sd_ens": df["sd_ens"].to_numpy(),
        "scarcity": 1.0 / (1.0 + df["n_polls"].to_numpy().astype(float)),
        "disagree": df["disagree"].to_numpy(), "L": len(levels), "M": len(groups),
        "level_of": level_of or [1], "sqrt_h": sqrt_h or [0.0], "gid": gid, "nu": nu,
    }
    if not groups:   # Stan needs M ≥ 1 for array sizes; add an unused group
        data.update({"M": 1, "level_of": [1], "sqrt_h": [0.0]})
    draws = cached_sample("error_meta", data, ["a", "b", "m0", "m1", "m2"],
                          **({"adapt_delta": 0.9, "iter_warmup": 800, "iter_sampling": 800} | sample_kwargs))
    a = draws["a"].mean(axis=0)
    b = draws["b"].mean(axis=0)
    return ErrorModel(
        levels, {l: float(a[i]) for i, l in enumerate(levels)}, {l: float(b[i]) for i, l in enumerate(levels)},
        float(draws["m0"].mean()), float(draws["m1"].mean()), float(draws["m2"].mean()), nu, n_cycles,
        extra={"n_rows": df.height, "n_groups": len(groups), "max_days": float(df["days_out"].max())},
    )

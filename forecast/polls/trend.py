"""Cross-lineup candidate trend — the French stand-in for the generic ballot.

Model on the log-share scale:  log y[row, c] = θ_c(t) + λ[race, c] + ε.
θ_c is a local-linear kernel trend over *all* races that include candidate c; λ absorbs each race's composition (a candidate polls lower in a
crowded lineup). Old polls are moved by θ_c(d) − θ_c(t) before averaging,
exactly like the post's generic-ballot trendline adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import polls_helpers as ph
from forecast.config import PollParams


@dataclass(frozen=True)
class Trend:
    days: np.ndarray        # grid of days relative to election (ascending, ints)
    theta: np.ndarray       # (len(days), C) log-share trend
    lam: np.ndarray         # (S, C) race effects (NaN where the race lacks c)
    flat: np.ndarray        # (C,) True where the candidate had too few polls for a trend

    def at(self, day: np.ndarray | int) -> np.ndarray:
        """θ at the given day(s); days outside the grid are clamped."""
        idx = np.clip(np.asarray(day, dtype=int) - self.days[0], 0, len(self.days) - 1)
        return self.theta[idx]


def _fill(col: np.ndarray) -> np.ndarray:
    """Forward-fill gaps, then back-fill the leading ones."""
    col = col.copy()
    valid = ~np.isnan(col)
    if not valid.any():
        return col
    idx = np.where(valid, np.arange(len(col)), 0)
    np.maximum.accumulate(idx, out=idx)
    col = col[idx]
    first = np.argmax(valid)
    col[:first] = col[first]
    return col


def _smooth_columns(days: np.ndarray, t: np.ndarray, w: np.ndarray, z: np.ndarray, bandwidth: float) -> np.ndarray:
    """Weighted local-linear kernel trend per column (`polls_helpers.smooth_polls`).

    A causal EWMA lags turning points, which leaves late movers (Bayrou 2012,
    Mélenchon 2022) under-adjusted. The local-linear smoother uses every poll
    already published — all of them precede the averaging date — so it has no
    look-ahead, and at the last poll it is one-sided with a slope term.
    Gaps where polls are too sparse are filled with the nearest estimate."""
    out = np.full((len(days), z.shape[1]), np.nan)
    for c in range(z.shape[1]):
        ok = ~np.isnan(z[:, c])
        if ok.sum() < 2:
            continue
        col = ph.smooth_polls(t[ok], z[ok, c], w[ok], days, bandwidth_days=bandwidth, min_effective_polls=0.0)
        # Keep estimates only where the kernel mass is a real fraction of the
        # candidate's peak (row weights can be ≪ 1 for many-lineup polls, so an
        # absolute threshold would blank rarely-tested candidates), then fill.
        mass = (np.exp(-0.5 * ((t[ok][None, :] - days[:, None]) / bandwidth) ** 2) * w[ok][None, :]).sum(axis=1)
        col[mass < 0.1 * mass.max()] = np.nan
        # Never extrapolate a slope past a candidate's first or last poll:
        # the trend is held flat there (a candidate last polled in July keeps
        # July's level rather than following July's slope into September).
        col[(days < t[ok].min()) | (days > t[ok].max())] = np.nan
        out[:, c] = _fill(col)
    return out


def fit_trend(
    logy: np.ndarray,
    t: np.ndarray,
    race: np.ndarray,
    n_races: int,
    row_weight: np.ndarray,
    days: np.ndarray,
    p: PollParams,
) -> Trend:
    """Alternate θ (kernel trend of residuals from race effects) and λ (shrunk,
    centred race effects) `trend_iterations` times."""
    R, C = logy.shape
    lam = np.zeros((n_races, C))
    present = ~np.isnan(logy)
    n_rows = present.sum(axis=0)
    flat = n_rows < p.trend_min_polls
    theta = np.zeros((len(days), C))
    for _ in range(max(1, p.trend_iterations)):
        z = logy - lam[race]
        theta = _smooth_columns(days, t, row_weight, z, p.trend_bandwidth_days)
        for c in np.where(flat)[0]:
            ok = present[:, c]
            theta[:, c] = np.average(z[ok, c], weights=row_weight[ok]) if ok.any() else np.nan
        idx = np.clip(t - days[0], 0, len(days) - 1)
        resid = logy - theta[idx]
        new_lam = np.full((n_races, C), np.nan)
        for s in range(n_races):
            rows = race == s
            for c in range(C):
                ok = rows & present[:, c]
                if not ok.any():
                    continue
                n = row_weight[ok].sum()
                new_lam[s, c] = np.average(resid[ok, c], weights=row_weight[ok]) * n / (n + p.trend_lineup_k)
        # centre each candidate's race effects (weighted by race size) so θ keeps the level
        for c in range(C):
            ok = ~np.isnan(new_lam[:, c])
            if ok.sum() > 1:
                sizes = np.array([row_weight[(race == s) & present[:, c]].sum() for s in np.where(ok)[0]])
                new_lam[ok, c] -= np.average(new_lam[ok, c], weights=sizes)
        lam = np.nan_to_num(new_lam, nan=0.0)
    return Trend(days=days, theta=theta, lam=np.where(np.isnan(new_lam), np.nan, lam), flat=flat)

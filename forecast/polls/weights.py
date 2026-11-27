"""Per-poll weights for the polling average (PLAN.md › Presidential model, step 2)."""
from __future__ import annotations

import numpy as np

from forecast.config import PollParams


def decay_rate(days_to_election: np.ndarray | float, p: PollParams) -> np.ndarray:
    """Daily recency decay δ(h): `decay_early` from `decay_ramp_start_days` out,
    `decay_late` inside `decay_ramp_end_days`, linear in between."""
    h = np.asarray(days_to_election, dtype=float)
    frac = (p.decay_ramp_start_days - h) / (p.decay_ramp_start_days - p.decay_ramp_end_days)
    frac = np.clip(frac, 0.0, 1.0)
    return p.decay_early + frac * (p.decay_late - p.decay_early)


def recency_weight(age_days: np.ndarray, days_to_election: np.ndarray | float, p: PollParams) -> np.ndarray:
    """(1 − δ(h))^age, where h is measured at the averaging day."""
    age = np.maximum(np.asarray(age_days, dtype=float), 0.0)
    return (1.0 - decay_rate(days_to_election, p)) ** age


def sample_size_weight(n_eff: np.ndarray, p: PollParams) -> np.ndarray:
    """√(min(n, cap) / ref); polls with unknown size get 1."""
    n = np.asarray(n_eff, dtype=float)
    w = np.sqrt(np.minimum(n, p.sample_size_cap) / p.sample_size_ref)
    return np.where(np.isfinite(w) & (w > 0), w, 1.0)


def base_weight(overlap: np.ndarray, n_eff: np.ndarray, partisan: np.ndarray, p: PollParams) -> np.ndarray:
    """Time-independent part of a poll's weight: overlap × size × sponsor."""
    ov = np.nan_to_num(np.asarray(overlap, dtype=float), nan=1.0)
    sponsor = np.where(np.asarray(partisan, dtype=bool), p.partisan_weight, 1.0)
    return ov * sample_size_weight(n_eff, p) * sponsor


def partisan_shift(shares: np.ndarray, sponsor_mask: np.ndarray, shift_pts: float) -> np.ndarray:
    """Move `shift_pts` away from the sponsor's candidates (pro rata among them)
    to everyone else (pro rata). Row totals are preserved. NaN = not listed."""
    y = np.array(shares, dtype=float)
    present = ~np.isnan(y)
    sp = sponsor_mask & present
    other = ~sponsor_mask & present
    s_tot = np.nansum(np.where(sp, y, 0.0))
    o_tot = np.nansum(np.where(other, y, 0.0))
    if s_tot <= 0 or o_tot <= 0:
        return y
    shift = min(shift_pts, 0.9 * s_tot)
    y[sp] -= shift * y[sp] / s_tot
    y[other] += shift * y[other] / o_tot
    return y


def anchor_slack(n_polls: np.ndarray | float, one: float = 2.0, floor: float = 1 / 3, n_floor: float = 5.0) -> np.ndarray:
    """Allowed gap between a race's polls and the now-cast: 2 pts at one poll,
    shrinking to the ⅓-pt floor by five polls, s(n) = max(floor, one·n^(−ln(one/floor)/ln n_floor))."""
    n = np.maximum(np.asarray(n_polls, dtype=float), 1e-9)
    exponent = np.log(one / floor) / np.log(n_floor)
    return np.maximum(floor, one * n ** (-exponent))

"""Presidential fundamentals features (PLAN.md › Presidential model, step 10).

The post's five predictors, French versions:

    log_baseline   log of the candidate's bloc share in the last two
                   presidential first rounds (80/20), split equally among the
                   bloc's candidates in the lineup ("seat partisan lean")
    incumbency     1 sitting president; 0.5 sitting PM or nominee of the
                   president's / governing party ("first-termers get half")
    experience     office grade 0–3, scaled to 0–1
    personal_vote  recency-weighted past over-performance of the baseline
                   (0 for first-timers)
    (fundraising)  dropped: no pre-election campaign-finance data in France
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast.config import default_config
from forecast.ingest.registry import CandidateRegistry

PREDICTORS = ("log_baseline", "incumbency", "experience", "personal_vote")
BASELINE_FLOOR_PCT = 1.0


@dataclass
class PresFeatures:
    registry: CandidateRegistry
    recent_weight: float = 0.8

    def __post_init__(self):
        self.years = sorted({y for y in self.registry._by_year if self.registry.ballot(y)})

    def previous_years(self, year: int) -> list[int]:
        return [y for y in self.years if y < year][::-1]

    def bloc_results(self, year: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in self.registry.ballot(year):
            out[c.bloc] = out.get(c.bloc, 0.0) + (c.result_t1 or 0.0)
        return out

    def baseline(self, year: int) -> dict[str, float] | None:
        """Bloc baseline (%) for `year`: 80/20 blend of the two previous
        presidentials (or the one previous, if that is all there is)."""
        prev = self.previous_years(year)
        if not prev:
            return None
        last = self.bloc_results(prev[0])
        if len(prev) == 1:
            blend = dict(last)
        else:
            older = self.bloc_results(prev[1])
            blend = {b: self.recent_weight * last.get(b, 0.0) + (1 - self.recent_weight) * older.get(b, 0.0)
                     for b in set(last) | set(older)}
        return blend

    def log_baseline(self, year: int, lineup: list[str]) -> np.ndarray:
        base = self.baseline(year) or {}
        blocs = [self.registry.get(year, c).bloc for c in lineup]
        k = {b: blocs.count(b) for b in blocs}
        return np.array([np.log(max(base.get(b, 0.0), BASELINE_FLOOR_PCT) / k[b] / 100) for b in blocs])

    def personal_vote(self, year: int, candidate_id: str) -> float:
        """Recency-weighted mean of log(own share) − log(expected share) over
        the candidate's earlier presidential runs (weights 1, ½, ¼, …)."""
        vals, wts = [], []
        for rank, y in enumerate(self.previous_years(year)):
            ballot = self.registry.ballot(y)
            ids = [c.candidate_id for c in ballot]
            if candidate_id not in ids or self.baseline(y) is None:
                continue
            c = ballot[ids.index(candidate_id)]
            expected = self.log_baseline(y, ids)[ids.index(candidate_id)]
            vals.append(np.log(c.result_t1 / 100) - expected)
            wts.append(0.5 ** len(vals[:-1]))
        return float(np.average(vals, weights=wts)) if vals else 0.0

    def matrix(self, year: int, lineup: list[str]) -> np.ndarray:
        """(len(lineup), 4) design matrix in PREDICTORS order."""
        rows = []
        lb = self.log_baseline(year, lineup)
        for i, cid in enumerate(lineup):
            c = self.registry.get(year, cid)
            rows.append([lb[i], c.incumbency or 0.0, (c.experience or 0.0) / 3.0, self.personal_vote(year, cid)])
        return np.array(rows)

    def training_races(self, years: list[int]) -> list[tuple[int, list[str], np.ndarray, np.ndarray]]:
        """(year, lineup, X, shares) for each year's actual ballot."""
        out = []
        for y in years:
            if self.baseline(y) is None:
                continue
            ballot = self.registry.ballot(y)
            ids = [c.candidate_id for c in ballot]
            shares = np.array([c.result_t1 for c in ballot], dtype=float)
            out.append((y, ids, self.matrix(y, ids), shares / shares.sum()))
        return out


def default_features() -> PresFeatures:
    return PresFeatures(CandidateRegistry.load(), default_config().lean.recent_weight)

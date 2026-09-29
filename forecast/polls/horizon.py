"""How wrong a polling average is, as a function of days to the election.

v0 of the error model (PLAN.md › M3): the log-share error of the pass-2
average against the result, fitted as

    sd(h, share) = (a + b·√h) · (share / 10%)^(−γ)

— the post's σ(h) = level + growth·√days, with a size term because small
candidates miss by less in points but more in relative terms (measured on
2002–2022: log-scale sd 0.40 under 3% vs 0.19 above 10%, two days out).
`bloc_corr` is the correlation of standardised errors between candidates of
the same bloc, the v0 split between bloc-level and candidate-level shocks.
The Stan error meta-model (M5) replaces this.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize

from forecast.config import PollParams
from forecast.ingest.registry import CandidateRegistry
from forecast.ingest.wiki_long import election_dates
from forecast.polls.ewma import main_race, polling_average

DEFAULT_HORIZONS = (2, 7, 14, 30, 60, 90, 120, 150, 200, 250)


@dataclass(frozen=True)
class HorizonModel:
    a: float
    b: float
    gamma: float
    bloc_corr: float
    cycles: tuple[int, ...] = ()

    def sd(self, h: np.ndarray | float, share_pct: np.ndarray | float) -> np.ndarray:
        h = np.maximum(np.asarray(h, dtype=float), 0.0)
        s = np.clip(np.asarray(share_pct, dtype=float), 0.2, 60.0)
        return (self.a + self.b * np.sqrt(h)) * (s / 10.0) ** (-self.gamma)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> "HorizonModel":
        d = json.loads(path.read_text())
        d["cycles"] = tuple(d.get("cycles", ()))
        return cls(**d)


def historical_errors(
    long: pl.DataFrame,
    registry: CandidateRegistry,
    cycles: list[int],
    p: PollParams,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pl.DataFrame:
    """Log-share error of each cycle's final-lineup average, as known h days
    out, against the first-round result (both renormalised over the ballot)."""
    recs = []
    for y in cycles:
        t1, _ = election_dates(y)
        sub = long.filter((pl.col("year") == y) & (pl.col("kind") == "lineup"))
        ballot = {c.candidate_id: (c.result_t1, c.bloc) for c in registry.ballot(y)}
        for h in horizons:
            as_of = t1 - timedelta(days=h)
            if sub.filter(pl.col("publish_est") <= as_of).is_empty():
                continue
            res = polling_average(sub, t1, p, as_of=as_of, eval_days=[-h])
            rid = main_race(res, -h, set(ballot))
            if rid is None:
                continue
            avg = res.race_average(rid, -h).filter(pl.col("option_id").is_in(list(ballot)))
            if avg.height < 3:
                continue
            s = avg["share"].to_numpy()
            s = s / s.sum()
            r = np.array([ballot[o][0] for o in avg["option_id"]])
            r = r / r.sum()
            for o, si, ri in zip(avg["option_id"], s, r):
                recs.append({"cycle": y, "h": h, "option_id": o, "bloc": ballot[o][1], "poll": 100 * si,
                             "result": 100 * ri, "err_log": float(np.log(ri / si))})
    return pl.DataFrame(recs)


def fit_horizon_model(errors: pl.DataFrame) -> HorizonModel:
    """Maximum likelihood (normal) for a, b, γ; bloc_corr by moments."""
    h = errors["h"].to_numpy().astype(float)
    s = errors["poll"].to_numpy()
    e = errors["err_log"].to_numpy()

    def nll(x):
        a, b, g = np.exp(x[0]), np.exp(x[1]), x[2]
        sd = (a + b * np.sqrt(h)) * (np.clip(s, 0.2, 60) / 10.0) ** (-g)
        return np.sum(np.log(sd) + 0.5 * (e / sd) ** 2)

    sol = minimize(nll, x0=np.array([np.log(0.1), np.log(0.03), 0.3]), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    a, b, g = float(np.exp(sol.x[0])), float(np.exp(sol.x[1])), float(sol.x[2])
    model = HorizonModel(a, b, g, 0.0, tuple(sorted(errors["cycle"].unique().to_list())))

    # correlation of standardised errors within (cycle, h, bloc)
    z = errors.with_columns(pl.Series("z", e / model.sd(h, s)))
    prods, n = 0.0, 0
    for _, g_ in z.group_by("cycle", "h", "bloc"):
        v = g_["z"].to_numpy()
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                prods += v[i] * v[j]
                n += 1
    corr = float(np.clip(prods / n, 0.0, 0.8)) if n else 0.0
    return HorizonModel(a, b, g, corr, model.cycles)

"""Now-cast for lineups nobody has polled (post § 3.1.2, "unpolled seats").

A nested-logit substitution model with political blocs as nests:

    P(j) = P(bloc b) · P(j | b),   P(j | b) = e^{u_j/μ_b} / Σ_{k∈b} e^{u_k/μ_b},
    P(b) ∝ exp(μ_b · log Σ_{k∈b} e^{u_k/μ_b})

u_j is a candidate's strength and μ_b ∈ (0, 1] how much a bloc's voters stay
inside the bloc when one of its candidates drops out (μ = 1 is plain logit;
smaller μ = stronger within-bloc substitution). Both are fitted to the pass-2
averages of every race polled so far, projected to the forecast day along the
cross-lineup trend. Races tested side by side in one poll (with / without X)
are what pin μ down.

Polled lineups are then re-anchored to their own average with the post's
slack rule: final = average + clip(now-cast − average, ±s(n)).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

from forecast.config import NowcastParams
from forecast.polls import weights as W
from forecast.polls.ewma import AverageResult


@dataclass(frozen=True)
class RaceObs:
    race_id: str
    options: tuple[str, ...]
    log_share: np.ndarray       # renormalised over `options`, projected to the forecast day
    weight: float
    n_polls: int
    staleness: int              # days between the race's last poll and the forecast day


@dataclass
class NestedLogit:
    candidates: list[str]
    blocs: dict[str, str]
    u: np.ndarray
    mu: dict[str, float]
    sigma_imp: float = 0.15
    staleness: dict[str, int] | None = None

    def shares(self, lineup: list[str] | tuple[str, ...]) -> np.ndarray:
        """Predicted shares (sum 1) for `lineup`, in the given order."""
        idx = [self.candidates.index(c) for c in lineup]
        u = self.u[idx]
        blocs = [self.blocs[c] for c in lineup]
        out = np.zeros(len(lineup))
        inclusive = {}
        for b in set(blocs):
            members = [i for i, bb in enumerate(blocs) if bb == b]
            mu = self.mu.get(b, 1.0)
            z = u[members] / mu
            zmax = z.max()
            inclusive[b] = (members, mu, zmax + np.log(np.exp(z - zmax).sum()))
        top = {b: mu * iv for b, (_, mu, iv) in inclusive.items()}
        tmax = max(top.values())
        denom = sum(np.exp(v - tmax) for v in top.values())
        for b, (members, mu, iv) in inclusive.items():
            pb = np.exp(top[b] - tmax) / denom
            within = np.exp(u[members] / mu - iv)
            out[members] = pb * within
        return out

    def knows(self, lineup) -> bool:
        return all(c in self.candidates for c in lineup)


def race_observations(result: AverageResult, day: int, staleness_decay_days: float = 60.0) -> list[RaceObs]:
    """Each race's latest pass-2 average, moved to `day` along the trend.

    A race last polled before `day` is averaged at its last poll day and then
    shifted by θ(day) − θ(last), so stale races (e.g. Bardella lineups after
    July 2026) still inform the candidates only they contain, with a weight
    that decays with staleness."""
    frame = result.frame
    obs: list[RaceObs] = []
    last_by_race = {s: int(frame.t[frame.race == s].max()) for s in range(len(frame.race_ids))}
    at_days = sorted({min(day, v) for v in last_by_race.values()})
    by_key: dict[tuple[int, str], list[dict]] = {}
    for r in result.averages_at(at_days):
        by_key.setdefault((r["day"], r["race_id"]), []).append(r)
    for s, rid in enumerate(frame.race_ids):
        at = min(day, last_by_race[s])
        recs = by_key.get((at, rid), [])
        if not recs:
            continue
        opts = tuple(r["option_id"] for r in recs)
        cidx = [frame.options.index(o) for o in opts]
        logs = np.array([r["log_share"] for r in recs]) + result.trend.at(day)[cidx] - result.trend.at(at)[cidx]
        logs = logs - np.log(np.exp(logs).sum())
        stale = day - at
        weight = recs[0]["n_polls"] * np.exp(-stale / staleness_decay_days)
        obs.append(RaceObs(rid, opts, logs, float(weight), recs[0]["n_polls"], stale))
    return obs


def nested_log_shares(u: np.ndarray, mu: np.ndarray, bidx: np.ndarray, mask: np.ndarray, n_blocs: int) -> np.ndarray:
    """Vectorised nested-logit log shares for padded (race × slot) arrays."""
    z = np.where(mask, u / mu, -np.inf)
    top = np.full((u.shape[0], n_blocs), -np.inf)
    log_within = np.full(u.shape, -np.inf)
    for b in range(n_blocs):
        in_b = mask & (bidx == b)
        if not in_b.any():
            continue
        zb = np.where(in_b, z, -np.inf)
        m = zb.max(axis=1, keepdims=True)
        m_safe = np.where(np.isfinite(m), m, 0.0)
        with np.errstate(divide="ignore"):
            iv = m_safe + np.log(np.exp(zb - m_safe).sum(axis=1, keepdims=True))   # log Σ e^{u/μ}
        mu_b = np.where(in_b, mu, np.nan)
        mu_b = np.nanmax(np.where(in_b, mu_b, -np.inf), axis=1, keepdims=True)
        top[:, b] = np.where(in_b.any(axis=1), (mu_b * iv)[:, 0], -np.inf)
        log_within = np.where(in_b, zb - iv, log_within)
    tm = top.max(axis=1, keepdims=True)
    log_top = top - (tm + np.log(np.exp(top - tm).sum(axis=1, keepdims=True)))
    return np.where(mask, np.take_along_axis(log_top, bidx, axis=1) + log_within, 0.0)


def fit_nested_logit(
    obs: list[RaceObs],
    blocs: dict[str, str],
    p: NowcastParams,
    exclude: str | None = None,
) -> NestedLogit:
    """Weighted least squares on log shares with a normal prior on each μ_b."""
    use = [o for o in obs if o.race_id != exclude and o.weight > 1e-6]
    candidates = sorted({c for o in use for c in o.options})
    bloc_list = sorted({blocs[c] for c in candidates})
    ci = {c: i for i, c in enumerate(candidates)}
    nC, nB = len(candidates), len(bloc_list)

    # Padded (race × slot) arrays so the loss is vectorised over races.
    K = max(len(o.options) for o in use)
    cand = np.zeros((len(use), K), dtype=int)
    bidx = np.zeros((len(use), K), dtype=int)
    mask = np.zeros((len(use), K), dtype=bool)
    target = np.zeros((len(use), K))
    for r, o in enumerate(use):
        k = len(o.options)
        cand[r, :k] = [ci[c] for c in o.options]
        bidx[r, :k] = [bloc_list.index(blocs[c]) for c in o.options]
        mask[r, :k] = True
        target[r, :k] = o.log_share
    wts = np.array([o.weight for o in use])

    def loss(x):
        u = x[:nC]
        mu = 0.15 + 0.85 / (1 + np.exp(-x[nC:]))
        pred = nested_log_shares(u[cand], mu[bidx], bidx, mask, nB)
        resid = np.where(mask, target - pred, 0.0)
        prior = np.sum(((mu - p.nest_mu_prior) / p.nest_mu_prior_sd) ** 2)
        return float(np.sum(wts[:, None] * resid ** 2) + prior + 1e-4 * np.sum(u ** 2))

    # initialise u from average log shares
    u0 = np.zeros(nC)
    counts = np.zeros(nC)
    for o in use:
        for c, v in zip(o.options, o.log_share):
            u0[ci[c]] += v
            counts[ci[c]] += 1
    u0 = u0 / np.maximum(counts, 1)
    mu_logit0 = np.log((p.nest_mu_prior - 0.15) / (1 - p.nest_mu_prior))
    x0 = np.concatenate([u0, np.full(nB, mu_logit0)])
    sol = minimize(loss, x0, method="L-BFGS-B")
    u = sol.x[:nC]
    mu = {b: float(0.15 + 0.85 / (1 + np.exp(-sol.x[nC + k]))) for k, b in enumerate(bloc_list)}
    stale = {c: min(o.staleness for o in use if c in o.options) for c in candidates}
    return NestedLogit(candidates, blocs, u, mu, staleness=stale)


def imputation_sigma(obs: list[RaceObs], blocs: dict[str, str], p: NowcastParams, min_polls: int = 3) -> float:
    """Leave-one-race-out RMSE (log shares) of the now-cast for races with
    at least `min_polls` current polls — the post's imputation error."""
    errs = []
    for o in obs:
        if o.n_polls < min_polls or o.staleness > 14:
            continue
        model = fit_nested_logit(obs, blocs, p, exclude=o.race_id)
        if not model.knows(o.options):
            continue
        pred = np.log(np.maximum(model.shares(o.options), 1e-9))
        errs.extend((o.log_share - pred).tolist())
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.15


@dataclass(frozen=True)
class LineupForecast:
    lineup: tuple[str, ...]
    log_share: np.ndarray        # anchored now-cast, renormalised
    sd_avg: np.ndarray           # uncertainty of the current average (log scale)
    polled: bool
    n_polls: int
    staleness: np.ndarray        # per candidate, days since last polled

    def shares(self) -> dict[str, float]:
        return {c: float(100 * np.exp(v)) for c, v in zip(self.lineup, self.log_share)}


def nowcast_lineup(
    lineup: tuple[str, ...],
    model: NestedLogit,
    result: AverageResult,
    day: int,
    p: NowcastParams,
    minor_pct: float = 2.0,
) -> LineupForecast:
    """Now-cast for `lineup`, anchored to its own polls when it has been polled."""
    lineup = tuple(lineup)
    pred = model.shares(lineup)
    stale = np.array([model.staleness.get(c, 0) if model.staleness else 0 for c in lineup])
    sd = np.full(len(lineup), np.nan)
    # find the polled race with the same major candidates
    frame = result.frame
    major = {o for o in lineup if pred[lineup.index(o)] * 100 >= minor_pct}
    avg = result.averages.filter(pl.col("day") == day)
    match = None
    for rid, opts in zip(frame.race_ids, frame.race_options):
        if set(opts) == major and rid in avg["race_id"].to_list():
            match = rid
            break
    if match is None:
        return LineupForecast(lineup, np.log(pred), np.zeros(len(lineup)), False, 0, stale)

    race = avg.filter(pl.col("race_id") == match)
    polled_share = dict(zip(race["option_id"], race["share"]))
    polled_sd = dict(zip(race["option_id"], race["sd_log"]))
    n = int(race["n_polls"][0])
    slack = float(W.anchor_slack(n, p.anchor_slack_one, p.anchor_floor, p.anchor_floor_n))
    out = 100 * pred.copy()
    for i, c in enumerate(lineup):
        if c in polled_share:
            gap = out[i] - polled_share[c]
            out[i] = polled_share[c] + np.clip(gap, -slack, slack)
            sd[i] = polled_sd[c]
    out = out / out.sum()
    sd = np.where(np.isnan(sd), 0.0, sd)
    return LineupForecast(lineup, np.log(out), sd, True, n, stale)

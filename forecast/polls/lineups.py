"""Now-cast for lineups nobody has polled (post § 3.1.2, "unpolled seats").

A nested-logit substitution model with political blocs as nests, and blocs
optionally grouped (`NowcastParams.nest_groups`, e.g. centre + right):

    P(j) = P(group g) · P(bloc b | g) · P(j | b)
    P(j | b) = e^{u_j/μ_b} / e^{I_b},        I_b = log Σ_{k∈b} e^{u_k/μ_b}
    P(b | g) = e^{μ_b I_b/λ_g} / e^{J_g},    J_g = log Σ_{b∈g} e^{μ_b I_b/λ_g}
    P(g) ∝ e^{λ_g J_g}

u_j is a candidate's strength and μ_b ∈ (0, 1] how much a bloc's voters stay
inside the bloc when one of its candidates drops out (μ = 1 is plain logit;
smaller μ = stronger within-bloc substitution). λ_g does the same one level
up: when a bloc empties, a small λ_g sends its voters to the other blocs of
its group rather than across the field in proportion to size. A bloc outside
any group is a group of its own, where λ cancels out and the model is the
plain nested logit. All are fitted to the pass-2 averages of every race polled
so far, projected to the forecast day along the cross-lineup trend. Races
tested side by side in one poll (with / without X) are what pin μ and λ down.

Polled lineups are then re-anchored to their own average with the post's
slack rule: final = average + clip(now-cast − average, ±s(n)).
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    groups: dict[str, str] = field(default_factory=dict)   # bloc → group (absent: its own group)
    lam: dict[str, float] = field(default_factory=dict)    # group → λ (absent: 1)

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
        top = {b: mu * iv for b, (_, mu, iv) in inclusive.items()}       # μ_b I_b
        group_of = {b: self.groups.get(b, b) for b in top}
        lam = {g: self.lam.get(g, 1.0) for g in group_of.values()}
        j = {}
        for g in lam:
            v = np.array([top[b] / lam[g] for b in top if group_of[b] == g])
            j[g] = v.max() + np.log(np.exp(v - v.max()).sum())          # J_g
        gtop = {g: lam[g] * j[g] for g in lam}
        tmax = max(gtop.values())
        denom = sum(np.exp(v - tmax) for v in gtop.values())
        for b, (members, mu, iv) in inclusive.items():
            g = group_of[b]
            pb = np.exp(gtop[g] - tmax) / denom * np.exp(top[b] / lam[g] - j[g])
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


def nested_log_shares(
    u: np.ndarray, mu: np.ndarray, bidx: np.ndarray, mask: np.ndarray, n_blocs: int,
    group: np.ndarray | None = None, lam: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised nested-logit log shares for padded (race × slot) arrays.

    `group[b]` is bloc b's group index and `lam[g]` that group's λ; without
    them every bloc is its own group (the plain nested logit)."""
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
        with np.errstate(invalid="ignore"):                              # bloc absent from a race: masked
            log_within = np.where(in_b, zb - iv, log_within)
    if group is None:
        group, lam = np.arange(n_blocs), np.ones(n_blocs)
    n_groups = int(group.max()) + 1
    scaled = top / lam[group][None, :]                                   # μ_b I_b / λ_g
    gtop = np.full((u.shape[0], n_groups), -np.inf)
    giv = np.full((u.shape[0], n_groups), -np.inf)
    for g in range(n_groups):
        s = scaled[:, group == g]
        m = s.max(axis=1, keepdims=True)
        m_safe = np.where(np.isfinite(m), m, 0.0)
        with np.errstate(divide="ignore"):
            jg = (m_safe + np.log(np.exp(s - m_safe).sum(axis=1, keepdims=True)))[:, 0]   # J_g
        giv[:, g] = jg
        gtop[:, g] = np.where(np.isfinite(m[:, 0]), lam[g] * jg, -np.inf)
    tm = gtop.max(axis=1, keepdims=True)
    log_g = gtop - (tm + np.log(np.exp(gtop - tm).sum(axis=1, keepdims=True)))
    with np.errstate(invalid="ignore"):                                  # absent blocs: −∞ − (−∞)
        log_bloc = log_g[:, group] + scaled - giv[:, group]
    return np.where(mask, np.take_along_axis(log_bloc, bidx, axis=1) + log_within, 0.0)


def fit_nested_logit(
    obs: list[RaceObs],
    blocs: dict[str, str],
    p: NowcastParams,
    exclude: str | None = None,
) -> NestedLogit:
    """Weighted least squares on log shares with a normal prior on each μ_b
    and λ_g (the same prior: a priori a group is as tight as a bloc)."""
    use = [o for o in obs if o.race_id != exclude and o.weight > 1e-6]
    candidates = sorted({c for o in use for c in o.options})
    bloc_list = sorted({blocs[c] for c in candidates})
    ci = {c: i for i, c in enumerate(candidates)}
    nC, nB = len(candidates), len(bloc_list)
    # Groups of blocs ("CEN+DRO"); only a group with two or more blocs in this
    # election's polls has a λ to fit. Every other bloc is its own group.
    group_of = {b: b for b in bloc_list}
    for spec in p.nest_groups:
        members = [b for b in spec.split("+") if b in group_of]
        if len(members) >= 2:
            group_of.update({b: spec for b in members})
    group_list = sorted(set(group_of.values()))
    fitted_groups = [g for g in group_list if g not in bloc_list]
    gidx = np.array([group_list.index(group_of[b]) for b in bloc_list])
    nF = len(fitted_groups)

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

    def unpack(x):
        u = x[:nC]
        mu = 0.15 + 0.85 / (1 + np.exp(-x[nC:nC + nB]))
        lam_f = 0.15 + 0.85 / (1 + np.exp(-x[nC + nB:]))
        lam = np.ones(len(group_list))
        lam[[group_list.index(g) for g in fitted_groups]] = lam_f
        return u, mu, lam_f, lam

    def loss(x):
        u, mu, lam_f, lam = unpack(x)
        pred = nested_log_shares(u[cand], mu[bidx], bidx, mask, nB, gidx, lam)
        resid = np.where(mask, target - pred, 0.0)
        nests = np.concatenate([mu, lam_f])
        prior = np.sum(((nests - p.nest_mu_prior) / p.nest_mu_prior_sd) ** 2)
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
    x0 = np.concatenate([u0, np.full(nB + nF, mu_logit0)])
    sol = minimize(loss, x0, method="L-BFGS-B")
    u, mu_v, lam_f, _ = unpack(sol.x)
    mu = {b: float(v) for b, v in zip(bloc_list, mu_v)}
    lam = {g: float(v) for g, v in zip(fitted_groups, lam_f)}
    groups = {b: g for b, g in group_of.items() if g in lam}
    stale = {c: min(o.staleness for o in use if c in o.options) for c in candidates}
    return NestedLogit(candidates, blocs, u, mu, staleness=stale, groups=groups, lam=lam)


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
) -> LineupForecast:
    """Now-cast for `lineup`, anchored to its own polls when it has been polled."""
    lineup = tuple(lineup)
    pred = model.shares(lineup)
    stale = np.array([model.staleness.get(c, 0) if model.staleness else 0 for c in lineup])
    sd = np.full(len(lineup), np.nan)
    # Find the polled race with the same major candidates. "Major" is the poll
    # frame's own set (`build_frame` keys races on it), not a fresh cut of the
    # now-cast at `minor_pct`: with two definitions, a candidate hovering
    # around the threshold makes a heavily polled lineup look unpolled.
    frame = result.frame
    major = set(lineup) & set().union(*map(set, frame.race_options))
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
    # The race average sums to 100 over the options its polls listed; the
    # lineup can hold minor candidates those polls left out (and the reverse).
    # Put the average on the lineup's basis before measuring the gap.
    shared = [i for i, c in enumerate(lineup) if c in polled_share]
    scale = out[shared].sum() / sum(polled_share[lineup[i]] for i in shared)
    for i in shared:
        anchor = polled_share[lineup[i]] * scale
        out[i] = anchor + np.clip(out[i] - anchor, -slack, slack)
        sd[i] = polled_sd[lineup[i]]
    out = out / out.sum()
    sd = np.where(np.isnan(sd), 0.0, sd)
    return LineupForecast(lineup, np.log(out), sd, True, n, stale)

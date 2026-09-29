"""Monte Carlo simulation of a presidential election (post § 4.1.2).

Each simulation draws a lineup from the scenario mixture, first-round shares
(anchored now-cast + Student-t errors on the log-share scale, drawn in
antithetic pairs so they cancel exactly and never move the central estimate),
applies the two-round rule, and draws the runoff. Runoff errors are
correlated with the finalists' first-round surprises (the counterpart of the
post's national shock shared across chambers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecast.polls.horizon import HorizonModel
from forecast.polls.lineups import LineupForecast
from forecast.simulate.runoff import RunoffModel
from forecast.simulate.scenarios import Lineup, sample_lineups


def antithetic_t(rng: np.random.Generator, shape: tuple[int, ...], nu: float, unit_variance: bool = True) -> np.ndarray:
    """Student-t draws in ± pairs along axis 0 (exact zero mean).

    unit_variance=True rescales to variance 1 — right when the sd it multiplies
    was estimated as a standard deviation (normal likelihood). The Stan error
    meta-model estimates a Student-t *scale*, so its draws must stay unscaled
    (rescaling with ν = 3 would shrink the middle of the distribution by 42%)."""
    n = shape[0]
    half = (n + 1) // 2
    z = rng.standard_t(nu, size=(half, *shape[1:]))
    if unit_variance:
        z = z * np.sqrt((nu - 2) / nu)
    return np.concatenate([z, -z])[:n]


@dataclass(frozen=True)
class SimInput:
    """Everything the simulator needs for one lineup."""
    candidates: tuple[str, ...]
    mu: np.ndarray                 # central log shares
    sd: np.ndarray                 # individual error sd (log scale)
    blocs: tuple[str, ...]         # shock group per candidate
    bloc_sd: dict[str, float]      # shared shock sd per group (0 = none)
    v_mean: np.ndarray | None = None

    @classmethod
    def from_forecast(cls, fc: LineupForecast, sd: np.ndarray, blocs: dict[str, str],
                      bloc_sd: dict[str, float] | None = None) -> "SimInput":
        return cls(tuple(fc.lineup), fc.log_share, sd, tuple(blocs[c] for c in fc.lineup), bloc_sd or {})


def candidate_sd(fc: LineupForecast, hm: HorizonModel, h: int, sigma_imp: float) -> np.ndarray:
    """Total log-share sd per candidate: current average + movement to election
    day + imputation (unpolled lineups) + staleness (candidates not polled lately)."""
    share = 100 * np.exp(fc.log_share)
    move = hm.sd(h, share)
    stale = hm.b * np.sqrt(np.maximum(fc.staleness, 0)) * (np.clip(share, 0.2, 60) / 10) ** (-hm.gamma)
    imp = 0.0 if fc.polled else sigma_imp
    return np.sqrt(fc.sd_avg ** 2 + move ** 2 + imp ** 2 + stale ** 2)


@dataclass
class PresSims:
    candidates: list[str]
    lineups: list[Lineup]
    lineup_idx: np.ndarray        # (N,)
    shares: np.ndarray            # (N, C) first-round %, NaN if not running
    finalists: np.ndarray         # (N, 2) candidate indices (second = -1 if outright win)
    winner: np.ndarray            # (N,) candidate index
    runoff_share: np.ndarray      # (N,) winner's runoff share (NaN if outright)

    @property
    def n(self) -> int:
        return len(self.winner)


@dataclass(frozen=True)
class Decision:
    """A mid-campaign choice made from the polls at `date`, e.g. which centrist
    runs. On the decision date the leading member of `pool` runs alone if its
    polling average is above `threshold` (%); otherwise the top `fallback_top`
    plus anyone above `fallback_above` (%) run.

    `realised` is the fraction of the forecast's error variance that will
    have played out by the decision date (the rest happens afterwards)."""
    name: str
    pool: tuple[str, ...]
    threshold: float
    fallback_top: int
    fallback_above: float
    realised: float

    def choose(self, shares: dict[str, np.ndarray]) -> list[frozenset[str]]:
        """Per simulation, the pool members who run (shares in %)."""
        names = list(shares)
        mat = np.column_stack([shares[c] for c in names])
        order = np.argsort(-mat, axis=1)
        out = []
        for i in range(mat.shape[0]):
            lead = order[i, 0]
            if mat[i, lead] > self.threshold:
                out.append(frozenset([names[lead]]))
                continue
            keep = {names[j] for j in order[i, : self.fallback_top]}
            keep |= {names[j] for j in range(len(names)) if mat[i, j] > self.fallback_above}
            out.append(frozenset(keep))
        return out


def _lineup_key(cands) -> str:
    return "+".join(sorted(cands))


def simulate_presidential(
    lineups: list[Lineup],
    inputs: dict[str, SimInput],
    runoff: RunoffModel,
    *,
    days_out_t2: int,
    nu: float,
    runoff_a: float,
    runoff_b: float,
    runoff_rho: float,
    n_sims: int = 40_000,
    seed: int = 0,
    sd_is_t_scale: bool = False,
    decision: Decision | None = None,
) -> PresSims:
    """`inputs` must hold every lineup a simulation can end up with: the
    scenario lineups and, with a `decision`, each of them with any subset of
    the decision pool removed."""
    rng = np.random.default_rng(seed)
    candidates = sorted({c for key in inputs for c in inputs[key].candidates}
                        | {c for l in lineups for c in l.candidates})
    ci = {c: i for i, c in enumerate(candidates)}
    N, C = n_sims, len(candidates)
    idx = sample_lineups(lineups, N, rng)
    shares = np.full((N, C), np.nan)
    finalists = np.full((N, 2), -1)
    winner = np.full(N, -1)
    runoff_share = np.full(N, np.nan)
    sigma_r = runoff_a + runoff_b * np.sqrt(max(days_out_t2, 0))
    unit_var = not sd_is_t_scale
    tvar = 1.0 if unit_var else nu / (nu - 2)     # variance of one unscaled t_ν draw is ν/(ν−2)
    f = decision.realised if decision is not None else 0.0

    for li, lineup in enumerate(lineups):
        rows = np.where(idx == li)[0]
        if len(rows) == 0:
            continue
        base = inputs[lineup.key]
        n = len(rows)
        # standardised shocks per candidate and per bloc, in two independent
        # parts: before the decision date (share f of the variance) and after
        zc = {c: (antithetic_t(rng, (n,), nu, unit_var), antithetic_t(rng, (n,), nu, unit_var))
              for c in base.candidates}
        zb = {g: (antithetic_t(rng, (n,), nu, unit_var), antithetic_t(rng, (n,), nu, unit_var))
              for g in set(base.blocs)}

        def errors(inp: SimInput, part: str) -> tuple[np.ndarray, np.ndarray]:
            w = {"before": (np.sqrt(f), 0.0), "total": (np.sqrt(f), np.sqrt(1 - f))}[part]
            err = np.zeros((n, len(inp.candidates)))
            var = np.zeros(len(inp.candidates))
            for j, (c, g) in enumerate(zip(inp.candidates, inp.blocs)):
                zc1, zc2 = zc[c]
                zg1, zg2 = zb.get(g, (np.zeros(n), np.zeros(n)))
                gsd = inp.bloc_sd.get(g, 0.0)
                err[:, j] = inp.sd[j] * (w[0] * zc1 + w[1] * zc2) + gsd * (w[0] * zg1 + w[1] * zg2)
                var[j] = (inp.sd[j] ** 2 + gsd ** 2) * tvar * (w[0] ** 2 + w[1] ** 2)
            return err, var

        # which lineup each simulation ends up with
        final_key = np.array([lineup.key] * n, dtype=object)
        if decision is not None and set(decision.pool) <= set(base.candidates):
            err0, var0 = errors(base, "before")
            logv = (base.mu - var0 / 2)[None, :] + err0
            v0 = np.exp(logv - logv.max(axis=1, keepdims=True))
            v0 /= v0.sum(axis=1, keepdims=True)
            pool_shares = {c: 100 * v0[:, base.candidates.index(c)] for c in decision.pool}
            for i, keep in enumerate(decision.choose(pool_shares)):
                dropped = set(decision.pool) - keep
                final_key[i] = _lineup_key(c for c in base.candidates if c not in dropped)

        for key in dict.fromkeys(final_key):
            sel = np.where(final_key == key)[0]
            inp = inputs[key]
            cands = list(inp.candidates)
            cols = np.array([ci[c] for c in cands])
            err_all, total_var = errors(inp, "total")
            err = err_all[sel]
            logv = (inp.mu - total_var / 2)[None, :] + err
            v = np.exp(logv - logv.max(axis=1, keepdims=True))
            v /= v.sum(axis=1, keepdims=True)
            r = rows[sel]
            m_ = len(sel)
            shares[np.ix_(r, cols)] = 100 * v
            v_mean = inp.v_mean if inp.v_mean is not None else np.exp(inp.mu)
            zstd = err / np.sqrt(total_var)[None, :]

            order = np.argsort(-v, axis=1)
            first, second = order[:, 0], order[:, 1]
            outright = v[np.arange(m_), first] > 0.5
            winner[r[outright]] = cols[first[outright]]
            finalists[r, 0] = cols[first]
            finalists[r[~outright], 1] = cols[second[~outright]]

            eps = antithetic_t(rng, (m_,), nu)   # runoff error sd is a normal-likelihood sd
            runoff_rows = np.where(~outright)[0]
            pairs = {(int(a), int(b)) for a, b in zip(first[runoff_rows], second[runoff_rows])}
            for a, b in pairs:
                m = runoff_rows[(first[runoff_rows] == a) & (second[runoff_rows] == b)]
                lg = runoff.simulate_logit(cands[a], cands[b], cands, v_mean, v[m])
                shock = (zstd[m, a] - zstd[m, b]) / np.sqrt(2)
                lg = lg + sigma_r * (np.sqrt(runoff_rho) * shock + np.sqrt(1 - runoff_rho) * eps[m])
                share_a = 1 / (1 + np.exp(-lg))
                a_wins = share_a > 0.5
                winner[r[m]] = np.where(a_wins, cols[a], cols[b])
                runoff_share[r[m]] = 100 * np.where(a_wins, share_a, 1 - share_a)
    return PresSims(candidates, lineups, idx, shares, finalists, winner, runoff_share)

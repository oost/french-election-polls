"""Turnout in the presidential election, and what it does to the runoff.

**First round.** Turnout (voters / registered) gets a prior from past first
rounds (recency-weighted mean on the logit scale; its spread is that
predictor's own forward error on earlier elections). Pollsters' abstention
estimates update it, after a correction for how far off and how noisy such
estimates were at the same distance from the vote in past campaigns.

**Runoff.** Each first-round electorate k chooses between the finalists A, B
and not casting a valid vote:

    η_kX = α − β·|pos_k − pos_X| ± gap/2 + shift_k,   P(X | k) = e^{η_kX} / (1 + e^{η_kA} + e^{η_kB})

(the finalists' own voters stay). A share `m` of first-round non-voters casts
a valid vote and splits like everyone else. α, β and m are fitted on past
runoffs — the ratio of valid votes between the rounds, with `gap` solved so
that each runoff's actual result is reproduced — and on vote-transfer polls
(how each electorate said it would split, including not voting).

In a simulation `shift_k = τ + ε_k`: τ is one mobilisation shock shared by
everyone (and by the non-voters' m), ε_k is each electorate's own. τ's spread
is what past runoffs needed to match their actual participation; ε's is the
gap between modelled and polled non-voting by electorate. The shocks move who
shows up and therefore the result: the change in logit(A's share) is added to
the runoff, and the generic runoff error is reduced by the average variance
such shocks produced in past runoffs, so the total stays calibrated while a
runoff that depends more on turnout than usual gets more uncertainty.
Abstention polls for a matchup shift τ.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import polars as pl
from scipy.optimize import brentq, minimize

from forecast import paths
from forecast.config import TurnoutParams
from forecast.ingest.turnout import load_abstention_polls, load_transfer_polls
from forecast.ingest.wiki_long import election_dates
from forecast.simulate.runoff import BLOC_POSITION, OWN_CANDIDATE_UTILITY

BUCKETS = (7, 21, 60)        # days before the vote: ≤7, 8–21, 22–60, 61+
RADICAL_LEFT = ("FG", "FI", "LFI", "PCF")   # as the legislative model's RAD unit


def position(bloc: str, party: str | None, p: TurnoutParams) -> float:
    """Left–right position used for abstention: the bloc's, except that the
    radical left sits between the left and the far left."""
    return p.radical_left_position if party in RADICAL_LEFT else BLOC_POSITION[bloc]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------------- history

@lru_cache(maxsize=1)
def turnout_history() -> pl.DataFrame:
    """Official national figures per round, 1965–2022, with `turnout` (voters /
    registered), `valid` (valid votes / registered) and `blank` (blank and
    spoiled ballots / voters)."""
    return pl.read_csv(paths.REFERENCE / "turnout_pres.csv").with_columns(
        (pl.col("voters") / pl.col("registered")).alias("turnout"),
        (pl.col("expressed") / pl.col("registered")).alias("valid"),
        (1 - pl.col("expressed") / pl.col("voters")).alias("blank"),
    ).sort("year", "round")


@dataclass(frozen=True)
class PastRunoff:
    year: int
    names: tuple[str, ...]
    pos: np.ndarray          # left–right position of each first-round candidate's bloc
    v: np.ndarray            # first-round shares (sum 1)
    ia: int
    ib: int
    share_a: float           # A's runoff share
    ratio: float             # valid votes, round 2 / round 1
    nonvoters: float         # first round: (registered − valid votes) / valid votes


@lru_cache(maxsize=4)
def _all_runoffs(p: TurnoutParams) -> tuple[PastRunoff, ...]:
    hist = turnout_history()
    cand = (pl.read_csv(paths.REFERENCE / "candidates_pres.csv")
            .filter(pl.col("result_t1").is_not_null())
            .select("year", pl.col("candidate_id").alias("candidate"), "party", "bloc", "result_t1", "result_t2"))
    pre = pl.read_csv(paths.REFERENCE / "runoffs_pre1981.csv", schema_overrides={"result_t2": pl.Float64})
    cand = pl.concat([pre, cand.cast(pre.schema)]).sort("year")
    out = []
    for (year,), g in cand.group_by("year", maintain_order=True):
        fin = g.filter(pl.col("result_t2").is_not_null())
        rounds = {r["round"]: r for r in hist.filter(pl.col("year") == year).iter_rows(named=True)}
        if fin.height != 2 or len(rounds) != 2:
            continue
        names = tuple(g["candidate"])
        a, b = fin["candidate"]
        v = g["result_t1"].to_numpy()
        out.append(PastRunoff(
            year, names, np.array([position(b_, pt, p) for b_, pt in zip(g["bloc"], g["party"])]), v / v.sum(),
            names.index(a), names.index(b),
            fin["result_t2"][0] / fin["result_t2"].sum(), rounds[2]["expressed"] / rounds[1]["expressed"],
            (rounds[1]["registered"] - rounds[1]["expressed"]) / rounds[1]["expressed"]))
    return tuple(out)


def past_runoffs(p: TurnoutParams, before: int | None = None) -> list[PastRunoff]:
    """Every runoff with known first-round results and turnout (1965–2022), before year `before`."""
    return [r for r in _all_runoffs(p) if before is None or r.year < before]


# ------------------------------------------------------------ abstention model

def choice(pos: np.ndarray, ia: int, ib: int, alpha: float, beta: float, gap: float = 0.0, shift=0.0):
    """(P(A | k), P(B | k)) per electorate; `shift` is a scalar or (…, K)."""
    k = np.arange(len(pos))
    ua = np.where(k == ia, OWN_CANDIDATE_UTILITY, alpha - beta * np.abs(pos - pos[ia]) + gap / 2 + shift)
    ub = np.where(k == ib, OWN_CANDIDATE_UTILITY, alpha - beta * np.abs(pos - pos[ib]) - gap / 2 + shift)
    m = np.maximum(np.maximum(ua, ub), 0.0)
    ea, eb, e0 = np.exp(ua - m), np.exp(ub - m), np.exp(-m)
    return ea / (e0 + ea + eb), eb / (e0 + ea + eb)


@dataclass(frozen=True)
class AbstentionModel:
    alpha: float
    beta: float
    mobilisation: float      # share of first-round non-voters casting a valid runoff vote
    sd_common: float         # sd of the shared mobilisation shock τ
    sd_electorate: float     # sd of each electorate's own shock ε_k
    share_var: float         # mean variance of logit(share) the shocks produced in the training runoffs
    years: tuple[int, ...] = ()
    shocks_by_year: dict[int, float] = field(default_factory=dict, hash=False, compare=False)

    def share(self, pos, ia, ib, v, gap=0.0, shift=0.0):
        pa, pb = choice(pos, ia, ib, self.alpha, self.beta, gap, shift)
        return (v * pa).sum(axis=-1) / (v * (pa + pb)).sum(axis=-1)

    def ratio(self, pos, ia, ib, v, nonvoters, gap=0.0, shift=0.0, common=0.0):
        """Valid runoff votes / valid first-round votes. `shift` moves the
        electorates, `common` (the shared part of it) the non-voters."""
        pa, pb = choice(pos, ia, ib, self.alpha, self.beta, gap, shift)
        return (v * (pa + pb)).sum(axis=-1) + nonvoters * _sigmoid(_logit(self.mobilisation) + common)

    def solve_gap(self, pos, ia, ib, v, share_a: float, shift: float = 0.0) -> float:
        """The appeal gap between the finalists that gives A the share `share_a`."""
        f = lambda g: float(self.share(pos, ia, ib, v, g, shift)) - share_a
        lo, hi = -30.0, 30.0
        if f(lo) >= 0:
            return lo
        if f(hi) <= 0:
            return hi
        return float(brentq(f, lo, hi, xtol=1e-6))

    def effect(self, pos, ia, ib, v, nonvoters, gap, tau, eps, base: float = 0.0):
        """(change in logit(A's share), valid-vote ratio, the ratio without
        shocks) per simulation for shocks τ (n,) and ε (n, K), against the
        no-shock reference `base`."""
        shift = tau[:, None] + eps
        d = _logit(self.share(pos, ia, ib, v, gap, shift)) - _logit(self.share(pos, ia, ib, v, gap, base))
        return (d, self.ratio(pos, ia, ib, v, nonvoters, gap, shift, tau),
                self.ratio(pos, ia, ib, v, nonvoters, gap, base, base))

    def shock_for(self, r: PastRunoff, ratio: float | None = None) -> float:
        """The shared shock τ that reproduces a runoff's valid-vote ratio (and result)."""
        target = r.ratio if ratio is None else ratio

        def f(tau: float) -> float:
            gap = self.solve_gap(r.pos, r.ia, r.ib, r.v, r.share_a, tau)
            return float(self.ratio(r.pos, r.ia, r.ib, r.v, r.nonvoters, gap, tau, tau)) - target
        lo, hi = -6.0, 6.0
        if f(lo) >= 0:
            return lo
        if f(hi) <= 0:
            return hi
        return float(brentq(f, lo, hi, xtol=1e-5))


def crosstabs(before: int | None, days: int) -> dict[tuple[int, str], dict[str, float]]:
    """Vote-transfer polls of the last `days` before each runoff, averaged:
    (year, first-round candidate) → {runoff choice or "NONE": share}."""
    df = load_transfer_polls().filter(pl.col("days_out") <= days)
    if before is not None:
        df = df.filter(pl.col("year") < before)
    out: dict[tuple[int, str], dict[str, float]] = {}
    for r in df.group_by("year", "source", "target").agg(pl.col("share").mean()).iter_rows(named=True):
        out.setdefault((r["year"], r["source"]), {})[r["target"]] = r["share"] / 100
    return out


@lru_cache(maxsize=32)
def fit_abstention(before: int | None, p: TurnoutParams) -> AbstentionModel:
    """α, β and m from the runoffs (and transfer polls) before year `before`,
    then the size of the shocks those runoffs imply."""
    runoffs = past_runoffs(p, before)
    xt = crosstabs(before, p.crosstab_days)
    m0 = _logit(p.mobilisation_prior)

    def parts(x):
        return x[0], max(x[1], 1e-3), _sigmoid(x[2])

    def terms(x):
        """(squared-error total, per-electorate residuals of logit(not voting))."""
        alpha, beta, mob = parts(x)
        model = AbstentionModel(alpha, beta, mob, 0, 0, 0)
        total, none_res = 0.0, []
        for r in runoffs:
            gap = model.solve_gap(r.pos, r.ia, r.ib, r.v, r.share_a)
            total += ((np.log(model.ratio(r.pos, r.ia, r.ib, r.v, r.nonvoters, gap)) - np.log(r.ratio)) / p.ratio_sd) ** 2
            pa, pb = choice(r.pos, r.ia, r.ib, alpha, beta, gap)
            a, b = r.names[r.ia], r.names[r.ib]
            for (year, source), obs in xt.items():
                if year != r.year or source not in r.names or not {a, b, "NONE"} <= set(obs):
                    continue
                k = r.names.index(source)
                none_res.append(float(_logit(1 - pa[k] - pb[k]) - _logit(obs["NONE"])))
                split = _logit(pa[k] / (pa[k] + pb[k])) - _logit(obs[a] / (obs[a] + obs[b]))
                total += (none_res[-1] / p.crosstab_none_sd) ** 2 + (split / p.crosstab_split_sd) ** 2
        return total, none_res

    def loss(x):
        prior = (((x[0] - p.alpha_prior) / p.alpha_prior_sd) ** 2 + ((x[1] - p.beta_prior) / p.beta_prior_sd) ** 2
                 + ((x[2] - m0) / p.mobilisation_prior_sd) ** 2)
        return terms(x)[0] + prior

    x = np.array([p.alpha_prior, p.beta_prior, m0])
    if runoffs:
        x = minimize(loss, x, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 4000}).x
    alpha, beta, mob = parts(x)
    base = AbstentionModel(alpha, beta, mob, 0, 0, 0)
    taus = {r.year: base.shock_for(r) for r in runoffs}
    sd_common = max(p.common_sd_floor, float(np.sqrt(np.mean(np.square(list(taus.values()))))) if taus else 0.0)
    none_res = terms(x)[1] if runoffs else []
    sd_el = float(np.sqrt(np.mean(np.square(none_res)))) if len(none_res) >= 3 else p.electorate_sd_default
    shocked = AbstentionModel(alpha, beta, mob, sd_common, sd_el, 0.0)
    rng = np.random.default_rng(0)
    variances = []
    for r in runoffs:
        gap = base.solve_gap(r.pos, r.ia, r.ib, r.v, r.share_a)
        tau = sd_common * rng.standard_normal(4000)
        eps = sd_el * rng.standard_normal((4000, len(r.pos)))
        variances.append(float(shocked.effect(r.pos, r.ia, r.ib, r.v, r.nonvoters, gap, tau, eps)[0].var()))
    return AbstentionModel(alpha, beta, mob, sd_common, sd_el, float(np.mean(variances)) if variances else 0.0,
                           tuple(r.year for r in runoffs), taus)


# ----------------------------------------------------------------------- polls

def _bucket(days: float) -> int:
    return int(np.searchsorted(BUCKETS, days, side="left"))


def finalists(year: int) -> str | None:
    r = next((r for r in _all_runoffs(TurnoutParams()) if r.year == year), None)
    return "|".join(sorted((r.names[r.ia], r.names[r.ib]))) if r else None


def poll_calibration(polls: pl.DataFrame, before: int | None) -> dict[tuple[str, int], tuple[float, float, int]]:
    """How pollsters' abstention estimates compared with the result in past
    campaigns: (round, horizon bucket) → (mean error, spread between
    campaigns, number of campaigns), on the logit scale. Second-round
    estimates count only for the runoff that took place."""
    hist = {(r["year"], "first" if r["round"] == 1 else "second"): 1 - r["turnout"]
            for r in turnout_history().iter_rows(named=True)}
    per: dict[tuple[str, int], dict[int, list[float]]] = {}
    for r in polls.iter_rows(named=True):
        key = (r["year"], r["round"])
        if key not in hist or (before is not None and r["year"] >= before):
            continue
        if r["round"] == "second" and r["matchup"] != finalists(r["year"]):
            continue
        err = float(_logit(r["abstention"] / 100) - _logit(hist[key]))
        per.setdefault((r["round"], _bucket(r["days_out"])), {}).setdefault(r["year"], []).append(err)
    out = {}
    for key, by_year in per.items():
        means = np.array([np.mean(v) for v in by_year.values()])
        n = len(means)
        bias = float(means.mean() * n / (n + 1))               # shrunk: few campaigns
        between = float(np.sqrt(np.mean((means - bias) ** 2)))
        out[key] = (bias, between, n)
    return out


def poll_estimate(polls: pl.DataFrame, as_of: date, vote_day: date, round_: str, matchup: str | None,
                  calibration: dict, p: TurnoutParams) -> tuple[float, float, int] | None:
    """(logit abstention, sd, number of polls) from the freshest abstention
    polls published by `as_of`, corrected with `calibration`."""
    df = polls.filter((pl.col("round") == round_) & (pl.col("publish_est") <= as_of) & (pl.col("fieldwork_end") < vote_day))
    df = df.filter(pl.col("matchup") == matchup) if matchup is not None else df
    if df.is_empty():
        return None
    latest = df["fieldwork_end"].max()
    df = df.filter((latest - pl.col("fieldwork_end")).dt.total_days() <= p.poll_window_days)
    x = _logit(df["abstention"].to_numpy() / 100)
    days = float(np.median([(vote_day - d).days for d in df["fieldwork_end"]]))
    bias, between, n_camp = calibration.get((round_, _bucket(days)), (0.0, p.poll_sd_default, 0))
    # one past campaign cannot show how much the error varies between campaigns
    between = max(between, p.poll_sd_floor) if n_camp >= 2 else max(between, abs(bias), p.poll_sd_default)
    sd = float(np.sqrt(between ** 2 + p.poll_within_sd ** 2 / len(x)))
    return float(x.mean() - bias), sd, len(x)


def combine(prior: tuple[float, float], estimate: tuple[float, float] | None) -> tuple[float, float]:
    """Precision-weighted mean of a prior and an estimate (mean, sd)."""
    if estimate is None:
        return prior
    w0, w1 = 1 / prior[1] ** 2, 1 / estimate[1] ** 2
    return (w0 * prior[0] + w1 * estimate[0]) / (w0 + w1), float(np.sqrt(1 / (w0 + w1)))


def first_round_prior(before: int | None, p: TurnoutParams) -> tuple[float, float]:
    """logit(first-round turnout): recency-weighted mean of past first rounds;
    sd = the same predictor's rms forward error on those elections."""
    h = turnout_history().filter(pl.col("round") == 1)
    h = h.filter(pl.col("year") < before) if before is not None else h
    x = _logit(h["turnout"].to_numpy())

    def predict(past):
        return float(np.average(past, weights=p.recency ** np.arange(len(past))[::-1]))
    errors = [predict(x[:i]) - x[i] for i in range(4, len(x))]
    sd = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) >= 3 else p.first_round_sd_default
    return predict(x), sd


# ---------------------------------------------------------- model for one forecast

@dataclass(frozen=True)
class TurnoutModel:
    """Everything the simulator needs about turnout for one forecast date."""
    abstention: AbstentionModel
    t1: tuple[float, float]            # logit first-round turnout: mean, sd
    t1_prior: tuple[float, float]      # the same before polls
    n_polls_t1: int
    blank1: float                      # blank and spoiled ballots / first-round voters
    blank_frac2: float                 # blank and spoiled ballots / registered voters without a valid runoff vote
    position: dict[str, float]         # candidate → left–right position
    runoff_polls: dict[str, tuple[float, float, int]]   # matchup → (logit abstention, sd, polls)
    residual_floor: float = 0.5

    @property
    def valid1(self) -> float:
        """Expected valid votes / registered voters in the first round."""
        return float(_sigmoid(self.t1[0]) * (1 - self.blank1))

    def positions(self, candidates: list[str]) -> np.ndarray:
        return np.array([self.position[c] for c in candidates])

    def runoff_turnout(self, valid2):
        """Voters / registered from valid votes / registered."""
        return valid2 + self.blank_frac2 * (1 - valid2)

    def shock_prior(self, key: str, pos, ia, ib, v_mean, gap: float) -> tuple[float, float]:
        """Mean and sd of the shared shock τ for a matchup: zero and the
        historical spread, moved by that matchup's abstention polls."""
        prior = (0.0, self.abstention.sd_common)
        est = self.runoff_polls.get(key)
        if est is None:
            return prior
        n = (1 - self.valid1) / self.valid1

        def abstention(tau: float) -> float:
            valid2 = self.valid1 * float(self.abstention.ratio(pos, ia, ib, v_mean, n, gap, tau, tau))
            return float(_logit(1 - self.runoff_turnout(min(valid2, 0.99))))
        slope = (abstention(0.25) - abstention(-0.25)) / 0.5
        if abs(slope) < 1e-6:
            return prior
        return combine(prior, ((est[0] - abstention(0.0)) / slope, est[1] / abs(slope)))

    def residual_sd(self, sigma_r: float) -> float:
        """Generic runoff error left once the turnout shocks carry their part."""
        return float(np.sqrt(max(sigma_r ** 2 - self.abstention.share_var, (self.residual_floor * sigma_r) ** 2)))


def build_turnout_model(year: int, as_of: date, blocs: dict[str, str], p: TurnoutParams,
                        polls: pl.DataFrame | None = None, parties: dict[str, str] | None = None) -> TurnoutModel:
    """The turnout model for election `year` as known on `as_of`. Only earlier
    elections are used; `polls` (ingest.turnout.load_abstention_polls) defaults
    to the processed table. `blocs` and `parties` map the candidates."""
    polls = load_abstention_polls() if polls is None else polls
    t1_day, t2_day = election_dates(year)
    calibration = poll_calibration(polls, year)
    mine = polls.filter(pl.col("year") == year)
    prior = first_round_prior(year, p)
    est = poll_estimate(mine, as_of, t1_day, "first", None, calibration, p)
    # polls estimate abstention; the model works with turnout
    t1 = combine(prior, None if est is None else (-est[0], est[1]))
    runoff_polls = {}
    for (matchup,) in mine.filter(pl.col("round") == "second").select("matchup").unique().drop_nulls().iter_rows():
        e = poll_estimate(mine, as_of, t2_day, "second", matchup, calibration, p)
        if e is not None:
            runoff_polls[matchup] = e
    hist = turnout_history().filter(pl.col("year") < year)
    blank1 = float(hist.filter(pl.col("round") == 1)["blank"].tail(3).mean())
    h2 = hist.filter(pl.col("round") == 2).tail(5)
    blank_frac2 = float(((h2["voters"] - h2["expressed"]) / (h2["registered"] - h2["expressed"])).mean())
    parties = parties or {}
    return TurnoutModel(fit_abstention(year, p), t1, prior, 0 if est is None else est[2], blank1, blank_frac2,
                        {c: position(b, parties.get(c), p) for c, b in blocs.items()}, runoff_polls, p.residual_floor)


# -------------------------------------------------------------------- backtest

def backtest_turnout(p: TurnoutParams, cycles=(2002, 2007, 2012, 2017, 2022),
                     days=(150, 60, 21, 7, 2), polls: pl.DataFrame | None = None) -> pl.DataFrame:
    """Turnout predicted for past elections from earlier elections and the
    polls published by each date, against the result. First round: `days`
    before it. Runoff: given the actual first round and result, 10, 5 and 2
    days before it."""
    polls = load_abstention_polls() if polls is None else polls
    hist = turnout_history()
    rows = []
    for year in cycles:
        t1_day, t2_day = election_dates(year)
        actual = {r["round"]: r for r in hist.filter(pl.col("year") == year).iter_rows(named=True)}
        runoff = next(r for r in _all_runoffs(p) if r.year == year)
        key = finalists(year)
        for d in days:
            tm = build_turnout_model(year, t1_day - timedelta(days=d), {}, p, polls)
            for label, (mu, sd), n in (("history only", tm.t1_prior, 0), ("with polls", tm.t1, tm.n_polls_t1)):
                if label == "with polls" and n == 0:
                    continue
                rows.append({"year": year, "round": 1, "days": d, "basis": label, "n_polls": n,
                             "predicted": 100 * float(_sigmoid(mu)), "lo": 100 * float(_sigmoid(mu - 1.645 * sd)),
                             "hi": 100 * float(_sigmoid(mu + 1.645 * sd)), "actual": 100 * actual[1]["turnout"]})
        am = fit_abstention(year, p)
        gap = am.solve_gap(runoff.pos, runoff.ia, runoff.ib, runoff.v, runoff.share_a)
        rng = np.random.default_rng(year)
        for d in (10, 5, 2):
            as_of = t2_day - timedelta(days=d)
            tm = build_turnout_model(year, as_of, {}, p, polls)
            known = as_of > t1_day
            valid1 = actual[1]["valid"] if known else tm.valid1
            n = (1 - valid1) / valid1
            for label in ("history only", "with polls"):
                if label == "with polls" and key not in tm.runoff_polls:
                    continue
                mu_t, sd_t = (0.0, am.sd_common)
                if label == "with polls":
                    # the first round has taken place: its turnout is known
                    known_t1 = replace(tm, t1=(float(_logit(valid1 / (1 - tm.blank1))), 0.01))
                    mu_t, sd_t = known_t1.shock_prior(key, runoff.pos, runoff.ia, runoff.ib, runoff.v, gap)
                tau = mu_t + sd_t * rng.standard_normal(4000)
                eps = am.sd_electorate * rng.standard_normal((4000, len(runoff.pos)))
                ratio = am.effect(runoff.pos, runoff.ia, runoff.ib, runoff.v, n, gap, tau, eps)[1]
                turnout2 = 100 * tm.runoff_turnout(np.minimum(valid1 * ratio, 0.99))
                rows.append({"year": year, "round": 2, "days": d, "basis": label,
                             "n_polls": tm.runoff_polls[key][2] if label == "with polls" else 0,
                             "predicted": float(np.median(turnout2)), "lo": float(np.quantile(turnout2, 0.05)),
                             "hi": float(np.quantile(turnout2, 0.95)), "actual": 100 * actual[2]["turnout"]})
    return pl.DataFrame(rows).with_columns((pl.col("predicted") - pl.col("actual")).alias("error"),
                                           ((pl.col("actual") >= pl.col("lo")) & (pl.col("actual") <= pl.col("hi"))).alias("inside"))

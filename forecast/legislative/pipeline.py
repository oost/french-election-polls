"""Legislative forecast: components → ensemble → five-level errors → chunked
simulation of both rounds in 577 seats (PLAN.md › Legislative model).

Components per seat × unit (log-share scale):
    polls         seat now-cast from the national polling average (generic ballot)
    fundamentals  Stan softmax DLM with the previous national result as offset
Errors (post § 4.1.1): national unit shock, region × unit, département × unit,
vote-pattern cluster × unit, and an individual seat error.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import polars as pl
import yaml

from forecast import paths
from forecast.config import Config, default_config
from forecast.ensemble.stacking import PrecisionModel, combine
from forecast.errors.meta_model import ErrorModel, fit_error_model
from forecast.fundamentals.dlm import SHARE_FLOOR
from forecast.ingest.legi_polls import LEGI_ELECTION_DATES, build_legislative_polls
from forecast.legislative import data as D
from forecast.legislative import rounds as R
from forecast.legislative.clusters import seat_clusters
from forecast.legislative.features import PREDICTORS, national_result, seat_design, with_outcome
from forecast.legislative.nowcast import SeatNowcast, fit_seat_nowcast
from forecast.polls.ewma import main_race, polling_average
from forecast.simulate.presidential import antithetic_t
from forecast.stan.build import cached_sample

LEVELS = ("national", "region", "dept", "cluster")
UNITS = D.UNITS
log = logging.getLogger(__name__)


def _priors() -> dict:
    return yaml.safe_load((paths.REFERENCE / "priors.yaml").read_text())


# --- National average (the generic ballot) ---------------------------------

@dataclass(frozen=True)
class NationalAverage:
    shares: dict[str, float]      # 0–1
    sd_log: dict[str, float]
    n_polls: int


@lru_cache(maxsize=None)
def legislative_polls() -> pl.DataFrame:
    return build_legislative_polls()


def national_average(year: int, as_of: date, eday: date, cfg: Config) -> NationalAverage | None:
    polls = legislative_polls().filter((pl.col("year") == year) & (pl.col("publish_est") <= as_of))
    if polls.select(pl.col("poll_id").n_unique()).item() < 2:
        return None
    d = (as_of - eday).days
    res = polling_average(polls, eday, cfg.polls, as_of=as_of, eval_days=[d], prior_cap=cfg.polls.prior_cap_legi,
                          minor_pct=0.0)
    rid = main_race(res, d)
    if rid is None:
        return None
    avg = res.race_average(rid, d)
    tot = avg["share"].sum()
    return NationalAverage({u: s / tot for u, s in zip(avg["option_id"], avg["share"])},
                           dict(zip(avg["option_id"], avg["sd_log"])), int(avg["n_polls"][0]))


# --- Fundamentals (Stan softmax DLM over seats) ----------------------------

@dataclass
class LegiFundamentals:
    beta_next: np.ndarray    # (draws, K)
    phi: np.ndarray
    train_years: tuple[int, ...]

    def predict(self, design: pl.DataFrame, prev_national: dict[str, float], n_draws: int = 400,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.beta_next), size=min(n_draws, len(self.beta_next)), replace=False)
        X = np.column_stack([design[k].to_numpy() for k in PREDICTORS])
        base = np.log(np.maximum(np.array([prev_national.get(u, 0.005) for u in design["unit"]]), 1e-3)) \
            + design["presence"].to_numpy()
        seats = design["seat"].to_list()
        uniq = {s: i for i, s in enumerate(dict.fromkeys(seats))}
        si = np.array([uniq[s] for s in seats])
        eta = base[None, :] + self.beta_next[idx] @ X.T                    # (draws, rows)
        e = np.exp(eta - eta.max(axis=1, keepdims=True))
        tot = np.zeros((len(idx), len(uniq)))
        for j in range(len(idx)):
            np.add.at(tot[j], si, e[j])
        p = e / tot[:, si]
        g = rng.gamma(np.maximum(p * self.phi[idx, None], 1e-3))           # Dirichlet via gammas
        gt = np.zeros_like(tot)
        for j in range(len(idx)):
            np.add.at(gt[j], si, g[j])
        draws = np.log(np.maximum(g / gt[:, si], 1e-6))
        return draws.mean(axis=0), draws.std(axis=0)


def fit_legi_fundamentals(years: list[int]) -> LegiFundamentals | None:
    years = [y for y in sorted(years) if y in D.PREVIOUS_LEGI]
    if not years:
        return None
    pri = _priors()["fundamentals_legi"]
    keys = {"lean": "log_baseline", "lagged": "lagged", "has_lagged": None, "inc": "incumbency"}
    X, base, y, race_t, start, length = [], [], [], [], [], []
    for t, year in enumerate(years, start=1):
        df = with_outcome(seat_design(year), year).filter(pl.col("share").is_not_null())
        prev_nat = national_result(D.PREVIOUS_LEGI[year])
        if year not in D.SPLIT_LEFT and D.PREVIOUS_LEGI[year] in D.SPLIT_LEFT:
            prev_nat = {**prev_nat, "GAU": prev_nat.get("GAU", 0) + prev_nat.pop("RAD", 0)}
        for seat, g in df.group_by("seat", maintain_order=True):
            if g.height < 2:
                continue
            start.append(len(y) + 1)
            length.append(g.height)
            race_t.append(t)
            X.extend(np.column_stack([g[k].to_numpy() for k in PREDICTORS]).tolist())
            base.extend((np.log(np.maximum(np.array([prev_nat.get(u, 0.005) for u in g["unit"]]), 1e-3))
                         + g["presence"].to_numpy()).tolist())
            s = np.maximum(g["share"].to_numpy(), SHARE_FLOOR)
            y.extend((s / s.sum()).tolist())
    prior = {"lean": pri["log_baseline"], "lagged": pri["lagged"], "has_lagged": [0.0, 0.5, 0.2],
             "inc": pri["incumbency"]}
    data = {
        "T": len(years), "K": len(PREDICTORS), "R": len(start), "N": len(y), "race_t": race_t, "start": start,
        "len": length, "X": np.array(X), "base": np.array(base), "y": np.array(y),
        "b1_mu": [prior[k][0] for k in PREDICTORS], "b1_sd": [prior[k][1] for k in PREDICTORS],
        "tau_scale": [prior[k][2] for k in PREDICTORS], "phi_log_mu": pri["phi_log_mu"],
    }
    draws = cached_sample("fundamentals_dlm", data, ["beta_next", "phi"], iter_warmup=500, iter_sampling=500,
                          adapt_delta=0.9)
    return LegiFundamentals(draws["beta_next"], draws["phi"], tuple(years))


# --- Models trained on earlier cycles --------------------------------------

@dataclass
class LegiModels:
    nowcast: SeatNowcast
    fundamentals: LegiFundamentals | None
    precision: PrecisionModel
    errors: ErrorModel
    withdrawals: R.WithdrawalModel
    transfers: R.TransferMatrix
    national_sd: tuple[float, float]         # (a, b): sd of national log-share error at horizon h


def prior_legi_errors(nu: float) -> ErrorModel:
    p = _priors()["errors_legi"]
    return ErrorModel(LEVELS, {l: p[l][0] for l in LEVELS}, {l: p[l][1] for l in LEVELS}, 1.0, 0.0, 0.0, nu, 0,
                      "prior")


def train_models(target: int, archive: pl.DataFrame, cfg: Config) -> LegiModels:
    years = [y for y in D.LEGI_YEARS if y < target]
    training = [(with_outcome(seat_design(y), y), national_result(y)) for y in years]
    nowcast = fit_seat_nowcast(training)
    fund = fit_legi_fundamentals(years)
    past = archive.filter(pl.col("cycle") < target) if not archive.is_empty() else archive
    precision = PrecisionModel.fit(past, ("polls", "fundamentals")) if not past.is_empty() else PrecisionModel()
    if past.is_empty() or past["cycle"].n_unique() < 1:
        errors = prior_legi_errors(cfg.errors.nu_floor)
    else:
        sub = past.filter(pl.col("seat_sample"))
        errors = fit_error_model(sub, LEVELS, nu_floor=cfg.errors.nu_floor, iter_warmup=400, iter_sampling=400)
    wd = R.fit_withdrawals(years)
    tm = R.fit_transfers([y for y in years if y >= 2012][-3:])
    p = _priors()["national_legi"]
    return LegiModels(nowcast, fund, precision, errors, wd, tm, (p["a"], p["b"]))


# --- Components and ensemble for one date ----------------------------------

@dataclass
class SeatForecast:
    design: pl.DataFrame          # seat, unit, mu_polls, mu_fund, mu_ens, sd_ens, disagree, region, dept, cluster
    national: NationalAverage
    days_out: int


def seat_forecast(split: bool, design: pl.DataFrame, prev_year: int, nat: NationalAverage, models: LegiModels,
                  h: int, seed: int = 0) -> SeatForecast:
    """Components and ensemble per seat × unit. `split`: the left runs as two
    units (RAD = LFI, GAU = the rest) rather than one union."""
    nat_units = dict(nat.shares)
    if not split and "RAD" in nat_units:
        nat_units["GAU"] = nat_units.get("GAU", 0) + nat_units.pop("RAD")
    pred = models.nowcast.predict(design, nat_units)
    nat_sd = models.national_sd[0] + models.national_sd[1] * np.sqrt(h)
    sd_nat = np.array([np.hypot(nat.sd_log.get(u, 0.1), nat_sd) for u in pred["unit"]])
    sd_p = np.sqrt(models.nowcast.resid_sd ** 2 + sd_nat ** 2)
    mu_p = pred["mu"].to_numpy()
    comps_f = None
    if models.fundamentals is not None:
        prev_nat = national_result(prev_year)
        if not split and "RAD" in prev_nat:
            prev_nat = {**prev_nat, "GAU": prev_nat.get("GAU", 0) + prev_nat.pop("RAD", 0)}
        elif split and "RAD" not in prev_nat and "RAD" in nat_units:
            # previous election had a left union: divide it as the polls do now
            lfi = nat_units["RAD"] / (nat_units["RAD"] + nat_units["GAU"])
            prev_nat = {**prev_nat, "RAD": prev_nat["GAU"] * lfi, "GAU": prev_nat["GAU"] * (1 - lfi)}
        comps_f = models.fundamentals.predict(design, prev_nat, seed=seed)
    seats = design["seat"].to_numpy()
    mu_ens = np.zeros(len(seats))
    sd_ens = np.zeros(len(seats))
    dis = np.zeros(len(seats))
    w_p = np.ones(len(seats))
    for s in np.unique(seats):
        m = seats == s
        comps = {"polls": (mu_p[m], sd_p[m])}
        if comps_f is not None:
            mf = comps_f[0][m]
            comps["fundamentals"] = (mf - np.log(np.exp(mf).sum()), comps_f[1][m])
        ens = combine(comps, models.precision, h, nat.n_polls, open_race=False)
        mu_ens[m], sd_ens[m], dis[m], w_p[m] = ens.mu, ens.sd_bar, ens.disagree, ens.weights["polls"]
    meta = D.seat_meta().join(seat_clusters(), on="seat", how="left")
    out = pred.select("seat", "unit").with_columns(
        pl.Series("mu_polls", mu_p), pl.Series("sd_polls", sd_p),
        pl.Series("mu_fundamentals", comps_f[0] if comps_f is not None else np.full(len(seats), np.nan)),
        pl.Series("sd_fundamentals", comps_f[1] if comps_f is not None else np.full(len(seats), np.nan)),
        pl.Series("mu_ens", mu_ens), pl.Series("sd_ens", sd_ens), pl.Series("disagree", dis),
        pl.Series("w_polls", w_p),
    ).join(meta, on="seat", how="left")
    return SeatForecast(out, nat, h)


# --- Simulation --------------------------------------------------------------

@dataclass
class LegiSims:
    seats: list[str]
    seat_win_prob: np.ndarray       # (S, K)
    totals: np.ndarray              # (N, K) seats won per unit
    n_triangulaires: np.ndarray     # (N,)
    n_first_round: np.ndarray       # (N,)

    def summary(self, majority: int = 289) -> pl.DataFrame:
        rows = []
        leader = self.totals.argmax(axis=1)
        for k, u in enumerate(UNITS):
            t = self.totals[:, k]
            if t.max() == 0:
                continue
            rows.append({"unit": u, "mean_seats": float(t.mean()), "q05": float(np.quantile(t, 0.05)),
                         "q50": float(np.quantile(t, 0.5)), "q95": float(np.quantile(t, 0.95)),
                         "p_majority": float((t >= majority).mean()), "p_largest": float((leader == k).mean())})
        return pl.DataFrame(rows).sort("mean_seats", descending=True)


def simulate_legislative(fc: SeatForecast, models: LegiModels, turnout_ratio: dict[str, float],
                         turnout_sd: float, cfg: Config, n_sims: int, seed: int = 0,
                         front_republicain_sd: float | None = None, transfer_sd: float = 0.3) -> LegiSims:
    rng = np.random.default_rng(seed)
    d = fc.design
    seats = list(dict.fromkeys(d["seat"].to_list()))
    si = {s: i for i, s in enumerate(seats)}
    S, K = len(seats), len(UNITS)
    mu = np.full((S, K), -np.inf)
    sd = np.zeros((S, K))
    ind = models.errors.individual_sd(d["sd_ens"].to_numpy(), 0.0, d["disagree"].to_numpy())
    for (seat, unit, m), s_ in zip(d.select("seat", "unit", "mu_ens").iter_rows(), ind):
        mu[si[seat], R.U[unit]] = m
        sd[si[seat], R.U[unit]] = s_
    present = np.isfinite(mu)
    meta = d.select("seat", "region", "dept", "cluster").unique(subset="seat").sort("seat")
    meta = meta.with_columns(pl.col("seat").replace_strict(si).alias("i")).sort("i")
    lv = {}
    for level in ("region", "dept", "cluster"):
        labels = meta[level].fill_null("NA").to_list()
        uniq = sorted(set(labels))
        lv[level] = (np.array([uniq.index(x) for x in labels]), len(uniq))
    h = fc.days_out
    em = models.errors
    t_scale = em.source == "stan"
    nu = em.nu
    tvar = nu / (nu - 2) if t_scale else 1.0
    level_sd = {l: em.shock_sd(l, h) for l in LEVELS}
    total_var = sd ** 2 * tvar + tvar * sum(level_sd[l] ** 2 for l in LEVELS)
    ratio = np.array([turnout_ratio.get(s, np.mean(list(turnout_ratio.values()))) for s in seats])
    fr_sd = cfg.legislative.front_republicain_sd if front_republicain_sd is None else front_republicain_sd
    chunk = min(200, n_sims)   # runoff utilities are (chunk × 577 × 8 × 8) floats
    win_counts = np.zeros((S, K))
    totals, tri, first = [], [], []
    done = 0
    while done < n_sims:
        n = min(chunk, n_sims - done)
        z = antithetic_t(rng, (n, S, K), nu, not t_scale) * sd[None]
        z += antithetic_t(rng, (n, 1, K), nu, not t_scale) * level_sd["national"]
        for level in ("region", "dept", "cluster"):
            idx, m = lv[level]
            z += antithetic_t(rng, (n, m, K), nu, not t_scale)[:, idx, :] * level_sd[level]
        logit = np.where(present[None], (mu - total_var / 2)[None] + z, -np.inf)
        e = np.exp(logit - logit.max(axis=2, keepdims=True))
        v1 = (e / e.sum(axis=2, keepdims=True)).astype(np.float32)
        r = np.clip(ratio[None, :] * (1 + turnout_sd * rng.standard_normal((n, 1))), 0.15, 0.95)
        win1, qualified = R.outcome_first_round(v1, r)
        shock = rng.standard_normal(n) * fr_sd
        standing = R.apply_withdrawals(v1, qualified, models.withdrawals, shock[:, None], rng)
        order = np.argsort(-v1, axis=2)
        top2 = np.zeros_like(standing)
        np.put_along_axis(top2, order[..., :2], True, axis=2)
        standing = np.where((standing.sum(axis=2) < 2)[..., None], top2 & qualified | top2, standing)
        A = models.transfers.A[None] + transfer_sd * rng.standard_normal((n, K, K)) * ~np.eye(K, dtype=bool)
        w2 = models.transfers.second_round(v1, standing, A[:, None, :, :].astype(np.float32))
        w2 = np.where(standing, w2 * np.exp(0.05 * rng.standard_normal(w2.shape)), -1)
        win2 = w2.argmax(axis=2)
        winner = np.where(win1 >= 0, win1, win2)
        onehot = np.zeros((n, S, K))
        np.put_along_axis(onehot, winner[..., None], 1, axis=2)
        win_counts += onehot.sum(axis=0)
        totals.append(onehot.sum(axis=1).astype(np.int16))
        tri.append(((standing.sum(axis=2) >= 3) & (win1 < 0)).sum(axis=1))
        first.append((win1 >= 0).sum(axis=1))
        done += n
    return LegiSims(seats, win_counts / n_sims, np.vstack(totals), np.concatenate(tri), np.concatenate(first))


# --- Backtest ------------------------------------------------------------------

LEGI_SNAPSHOT_DAYS = (28, 14, 7, 2)


def run_legislative_backtest(cycles=(2017, 2022, 2024), snapshot_days=LEGI_SNAPSHOT_DAYS,
                             n_sims: int = 4000, cfg: Config | None = None, write: bool = True):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    cfg = cfg or default_config()
    archive = pl.DataFrame()
    totals_rows = []
    rng = np.random.default_rng(0)
    for cycle in cycles:
        models = train_models(cycle, archive, cfg)
        eday = LEGI_ELECTION_DATES[cycle]
        prev = D.PREVIOUS_LEGI[cycle]
        design = seat_design(cycle)
        actual = with_outcome(design, cycle)
        turnout, _ = D.legi_results(prev, "t1")
        ratio = dict(zip(turnout["seat"], (turnout["exprimes"] / turnout["inscrits"]).to_list()))
        winners = D.seat_winners(cycle).group_by("unit").len()
        rows = []
        for i, h in enumerate(snapshot_days):
            as_of = eday - timedelta(days=h)
            nat = national_average(cycle, as_of, eday, cfg)
            if nat is None:
                continue
            fc = seat_forecast(cycle in D.SPLIT_LEFT, design, prev, nat, models, h, seed=i)
            sims = simulate_legislative(fc, models, ratio, 0.12, cfg, n_sims, seed=i)
            summ = sims.summary(cfg.legislative.majority_seats)
            for r in summ.iter_rows(named=True):
                act = winners.filter(pl.col("unit") == r["unit"])["len"]
                totals_rows.append({"cycle": cycle, "days_out": h, **r, "actual": int(act[0]) if len(act) else 0})
            df = fc.design.join(actual.select("seat", "unit", "share"), on=["seat", "unit"], how="left")
            df = df.filter(pl.col("share").is_not_null() & (pl.col("share") > 0))
            rows.append(df.with_columns(
                pl.lit(cycle).alias("cycle"), pl.lit(h).alias("days_out"),
                pl.col("share").log().alias("result_log"),
                (pl.col("share").log() - pl.col("mu_ens")).alias("err"),
                pl.lit(0).alias("n_polls"),
                pl.col("unit").alias("national"),
                pl.concat_str(["region", "unit"], separator="|").alias("region"),
                pl.concat_str(["dept", "unit"], separator="|").alias("dept"),
                pl.concat_str(["cluster", "unit"], separator="|").alias("cluster"),
                pl.Series("seat_sample", rng.random(df.height) < 0.35),
            ))
            log.info("legislative %s h=%s done", cycle, h)
        if rows:
            archive = pl.concat([archive, *rows], how="diagonal_relaxed")
    totals = pl.DataFrame(totals_rows)
    if write:
        paths.ensure_dirs()
        archive.write_parquet(paths.ARCHIVE / "legi_backtest.parquet")
        totals.write_parquet(paths.ARCHIVE / "legi_backtest_totals.parquet")
    return archive, totals


# --- Live forecast for the next election -----------------------------------

def forecast_design(split: bool, units: list[str], prev_year: int = 2024, pres_years: tuple[int, ...] = (2022, 2017),
                    rerun_p: float = 0.8, recent_weight: float = 0.8) -> pl.DataFrame:
    """Design table for an election that hasn't happened: every listed unit
    stands in every seat; the last winner's unit gets expected incumbency
    rerun_p × (½ if first term); lagged shares from `prev_year`."""
    seats = D.seat_meta()["seat"].to_list()
    df = pl.DataFrame({"seat": [s for s in seats for _ in units], "unit": [u for _ in seats for u in units]})
    lean = D.lean(prev_year, split, recent_weight, pres_years)
    prev = D.seat_units_t1(prev_year).select("seat", "unit", "share", "exprimes")
    if split and prev_year not in D.SPLIT_LEFT:
        prev = prev   # the previous left union maps onto GAU; RAD gets no lagged share
    nat = D.national_shares(prev)
    lag = prev.select("seat", "unit", (pl.col("share").clip(1e-3).log()
                                       - pl.col("unit").replace_strict(nat).log()).alias("lagged"))
    win = D.seat_winners(prev_year)
    older = D.seat_winners(D.PREVIOUS_LEGI[prev_year]) if prev_year in D.PREVIOUS_LEGI else None
    key = lambda df: pl.concat_str([pl.col("nom").map_elements(D._norm, return_dtype=pl.Utf8),
                                    pl.col("prenom").map_elements(D._norm, return_dtype=pl.Utf8)], separator="|")
    win = win.with_columns(key(win).alias("k"))
    re = set(zip(older["seat"], older.with_columns(key(older).alias("k"))["k"])) if older is not None else set()
    inc = win.select("seat", "unit", pl.Series("inc", [rerun_p * (1.0 if (s, k) in re else 0.5)
                                                       for s, k in zip(win["seat"], win["k"])]))
    df = (df.join(lean, on=["seat", "unit"], how="left").join(lag, on=["seat", "unit"], how="left")
          .join(inc, on=["seat", "unit"], how="left"))
    return df.with_columns(
        pl.col("lean").fill_null(0.0), pl.col("lagged").is_not_null().cast(pl.Float64).alias("has_lagged"),
        pl.col("lagged").fill_null(0.0), pl.col("inc").fill_null(0.0), pl.lit(0.0).alias("presence"),
        pl.lit(0).alias("year"),
    ).sort("seat", "unit")


@dataclass
class LegiSummary:
    as_of: date
    election_t1: date
    scenarios: dict[str, float]
    totals: pl.DataFrame
    by_scenario: dict[str, pl.DataFrame]
    seat_probs: pl.DataFrame
    national: dict[str, float]
    last_poll: date | None = None

    def report(self) -> str:
        lines = [f"French legislative election (first round assumed {self.election_t1}) — forecast as of {self.as_of}",
                 f"National polling average: " + ", ".join(f"{u} {100 * s:.1f}%" for u, s in
                                                          sorted(self.national.items(), key=lambda x: -x[1])),
                 f"Latest national legislative poll: {self.last_poll}",
                 f"Alliance scenarios: {self.scenarios}", "",
                 f"{'unit':<6}{'mean seats':>11}{'90% range':>14}{'P(289+)':>9}{'P(largest)':>12}"]
        for r in self.totals.iter_rows(named=True):
            rng = f"{r['q05']:.0f}–{r['q95']:.0f}"
            lines.append(f"{r['unit']:<6}{r['mean_seats']:>11.0f}{rng:>14}{r['p_majority']:>9.1%}{r['p_largest']:>12.1%}")
        return "\n".join(lines)


def run_legislative_forecast(as_of: date | None = None, n_sims: int | None = None, cfg: Config | None = None,
                             write: bool = False) -> LegiSummary:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    cfg = cfg or default_config()
    spec = yaml.safe_load((paths.REFERENCE / "scenarios_legi.yaml").read_text())
    t1 = spec["election"]["t1"]
    polls = legislative_polls().filter(pl.col("year") == 2029)
    as_of = as_of or min(polls["publish_est"].max(), date.today())
    h = max((t1 - as_of).days, 0)
    arch_path = paths.ARCHIVE / "legi_backtest.parquet"
    archive = pl.read_parquet(arch_path) if arch_path.exists() else pl.DataFrame()
    models = train_models(2030, archive, cfg)
    nat_polls = polls.filter(pl.col("publish_est") <= as_of)
    # No national legislative election is scheduled, so polls are sparse: if
    # none falls in the 90-day window, average at the last poll and widen the
    # uncertainty for the months since (staleness, like stale lineups).
    last_day = (nat_polls["midpoint"].max() - t1).days
    at = min(-h, last_day)
    stale = -h - at
    res = polling_average(nat_polls, t1, cfg.polls, as_of=as_of, eval_days=[at], prior_cap=cfg.polls.prior_cap_legi,
                          minor_pct=0.0)
    rid = main_race(res, at)
    avg = res.race_average(rid, at)
    tot = avg["share"].sum()
    grow = models.national_sd[1] * np.sqrt(stale)
    base = NationalAverage({u: s / tot for u, s in zip(avg["option_id"], avg["share"])},
                           {u: float(np.hypot(sd, grow)) for u, sd in zip(avg["option_id"], avg["sd_log"])},
                           int(avg["n_polls"][0]))
    turnout, _ = D.legi_results(2024, "t1")
    ratio = dict(zip(turnout["seat"], (turnout["exprimes"] / turnout["inscrits"]).to_list()))
    n = n_sims or cfg.simulation.n_sims
    all_totals, by_scen, probs = [], {}, []
    scen_p = {k: v["probability"] for k, v in spec["alliances"].items()}
    for k, (name, p) in enumerate(scen_p.items()):
        split = name == "split_left"
        shares = dict(base.shares)
        sds = dict(base.sd_log)
        if split:
            lfi = spec["alliances"][name]["lfi_share_of_left"]
            shares["RAD"], shares["GAU"] = shares["GAU"] * lfi, shares["GAU"] * (1 - lfi)
            sds["RAD"] = sds.get("GAU", 0.1)
        units = [u for u in UNITS if shares.get(u, 0) >= 0.005]
        design = forecast_design(split, units, rerun_p=spec.get("incumbent_rerun_probability", 0.8),
                                 recent_weight=cfg.lean.recent_weight)
        n_polls = base.n_polls
        nat = NationalAverage(shares, sds, n_polls)
        fc = seat_forecast(split, design, 2024, nat, models, h, seed=k)
        sims = simulate_legislative(fc, models, ratio, spec["turnout"]["relative_sd"], cfg, max(int(n * p), 200),
                                    seed=cfg.simulation.seed + k,
                                    front_republicain_sd=spec["withdrawals"]["front_republicain_sd"])
        by_scen[name] = sims.summary(cfg.legislative.majority_seats)
        all_totals.append(sims.totals)
        probs.append((p, sims.seat_win_prob, sims.seats))
    totals = np.vstack(all_totals)
    pooled = LegiSims(probs[0][2], sum(p * w for p, w, _ in probs), totals, np.zeros(1), np.zeros(1))
    seat_probs = pl.DataFrame({"seat": pooled.seats, **{u: pooled.seat_win_prob[:, i] for i, u in enumerate(UNITS)}})
    summary = LegiSummary(as_of, t1, scen_p, pooled.summary(cfg.legislative.majority_seats), by_scen, seat_probs,
                          base.shares, last_poll=nat_polls["midpoint"].max())
    if write:
        out = paths.OUTPUTS / f"legi_{as_of.isoformat()}"
        out.mkdir(parents=True, exist_ok=True)
        summary.totals.write_csv(out / "seat_totals.csv")
        summary.seat_probs.write_csv(out / "seat_probabilities.csv")
        (out / "report.txt").write_text(summary.report(), encoding="utf-8")
    return summary

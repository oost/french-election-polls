"""Strict out-of-sample backtest — "no time travel" (post § 5).

One chronological pass over cycles. For cycle c, every snapshot d uses:

* polls published by d (`publish_est ≤ d`);
* a horizon model, fundamentals fit, ensemble precision curves and error
  meta-model trained only on cycles before c (their archive rows were
  themselves produced out of sample, so no inner loop is needed);
* the fixed prior table (`priors.yaml`, config) where no earlier cycle exists.

Disclosed leak: lineups are each cycle's actual ballot (restricted to
candidates the substitution model knows), so lineup uncertainty is not
backtested — declaration dates are not curated yet.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
import yaml

from forecast import paths
from forecast.config import Config, default_config
from forecast.ensemble.stacking import PrecisionModel, combine
from forecast.errors.meta_model import ErrorModel, fit_error_model, prior_error_model
from forecast.fundamentals.dlm import FundamentalsFit, fit_fundamentals
from forecast.fundamentals.features_pres import PresFeatures
from forecast.ingest.build import load_polls_long
from forecast.ingest.registry import CandidateRegistry
from forecast.ingest.wiki_long import election_dates
from forecast.polls.ewma import polling_average
from forecast.polls.horizon import HorizonModel, fit_horizon_model, historical_errors
from forecast.polls.drift import DriftModel, fit_drift
from forecast.polls.lineups import fit_nested_logit, imputation_sigma, nowcast_lineup, race_observations
from forecast.simulate.presidential import SimInput, candidate_sd, simulate_presidential
from forecast.simulate.runoff import (
    RunoffModel, far_right_runoff_bias, fit_transfer_model, matchup_observations, _logit,
)
from forecast.simulate.turnout import build_turnout_model
from forecast.simulate.scenarios import Lineup

PRES_CYCLES = (2002, 2007, 2012, 2017, 2022)
SNAPSHOT_DAYS = tuple(range(156, 1, -7))           # 156, 149, …, 9, 2 days before the first round
FUND_YEARS = (1995, 2002, 2007, 2012, 2017, 2022)  # elections with a two-election baseline

log = logging.getLogger(__name__)


def _prior(section: str) -> dict:
    return yaml.safe_load((paths.REFERENCE / "priors.yaml").read_text())[section]


def prior_horizon_model() -> HorizonModel:
    p = _prior("horizon_pres")
    return HorizonModel(p["a"], p["b"], p["gamma"], 0.0, ())


@dataclass
class CycleModels:
    cycle: int
    horizon: HorizonModel
    fundamentals: FundamentalsFit | None
    precision: PrecisionModel
    errors: ErrorModel
    runoff_a: float
    runoff_b: float
    drift: DriftModel
    runoff_unpolled: float | None = None    # set with errors.runoff_by_polled
    far_right: tuple[float, float, float] | None = None   # (probability, mean, sd), from earlier runoffs only


def fit_runoff_scale(archive: pl.DataFrame, default: tuple[float, float]) -> tuple[float, float]:
    """σ_r(h) = a + b·√h for the logit error of the finalists' runoff share."""
    from scipy.optimize import minimize
    if archive.is_empty() or archive.height < 10:
        return default
    h = archive["days_to_t2"].to_numpy().astype(float)
    e = archive["err_logit"].to_numpy()

    def nll(x):
        sd = np.exp(x[0]) + np.exp(x[1]) * np.sqrt(h)
        return np.sum(np.log(sd) + 0.5 * (e / sd) ** 2)

    sol = minimize(nll, np.log(np.array(default)), method="Nelder-Mead")
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))


def fit_runoff_scales(archive: pl.DataFrame, errors) -> tuple[float, float, float]:
    """(a, b, unpolled): σ_r(h) = a + b·√h for matchups pollsters have tested,
    and a flat σ for those they have not (`errors.runoff_by_polled`).

    A matchup the polls cover and one only the transfer model predicts miss
    by very different amounts, so each kind is scaled on its own past
    runoffs. Polled: the prior curve `runoff_a + runoff_b·√h` times one
    factor. Snapshots of one election share one error, so each election counts
    once, and the priors count as `runoff_prior_weight` elections."""
    a0, b0, u0, k0 = errors.runoff_a, errors.runoff_b, errors.runoff_unpolled, errors.runoff_prior_weight
    if archive.is_empty():
        return a0, b0, u0
    polled, unpolled = archive.filter(pl.col("polled_matchup")), archive.filter(~pl.col("polled_matchup"))
    ratios = [float(np.mean((g["err_logit"].to_numpy() / (a0 + b0 * np.sqrt(g["days_to_t2"].to_numpy()))) ** 2))
              for _, g in polled.group_by("cycle")]
    factor = float(np.sqrt((k0 + sum(ratios)) / (k0 + len(ratios))))
    squares = [float(np.mean(g["err_logit"].to_numpy() ** 2)) for _, g in unpolled.group_by("cycle")]
    return a0 * factor, b0 * factor, float(np.sqrt((k0 * u0 ** 2 + sum(squares)) / (k0 + len(squares))))


def cycle_models(cycle: int, archive: pl.DataFrame, runoff_archive: pl.DataFrame, long: pl.DataFrame,
                 registry: CandidateRegistry, features: PresFeatures, cfg: Config) -> CycleModels:
    earlier = tuple(c for c in PRES_CYCLES if c < cycle)
    hm = fit_horizon_model(historical_errors(long, registry, list(earlier), cfg.polls)) if earlier \
        else prior_horizon_model()
    fund_years = [y for y in FUND_YEARS if y < cycle]
    fund = fit_fundamentals(features, fund_years) if fund_years else None
    past = archive.filter(pl.col("cycle") < cycle) if not archive.is_empty() else archive
    precision = PrecisionModel.fit(past, ("polls", "fundamentals")) if not past.is_empty() else PrecisionModel()
    if past.is_empty():
        errors = prior_error_model(("bloc",), nu=cfg.errors.nu_floor)
    else:
        errors = fit_error_model(past, ("bloc",), nu_floor=cfg.errors.nu_floor, thin_every=2)
    r_past = runoff_archive.filter(pl.col("cycle") < cycle) if not runoff_archive.is_empty() else runoff_archive
    if cfg.errors.runoff_by_polled:
        ra, rb, ru = fit_runoff_scales(r_past, cfg.errors)
    else:
        (ra, rb), ru = fit_runoff_scale(r_past, (cfg.errors.runoff_a, cfg.errors.runoff_b)), None
    drift = fit_drift(past) if cfg.polls.use_drift and not past.is_empty() else DriftModel()
    far_right = None
    if cfg.errors.far_right_runoff_probability > 0:
        bias = far_right_runoff_bias(long, registry, cycle, cfg.errors.far_right_runoff_noise,
                                     cfg.errors.far_right_runoff_prior_sd, cfg.errors.far_right_runoff_days)
        if bias is not None:
            far_right = (cfg.errors.far_right_runoff_probability, bias[0], bias[1])
    return CycleModels(cycle, hm, fund, precision, errors, ra, rb, drift, ru, far_right)


def snapshot(
    cycle: int, h: int, polls: pl.DataFrame, registry: CandidateRegistry, models: CycleModels,
    cfg: Config, n_sims: int, seed: int,
) -> tuple[list[dict], list[dict]]:
    """Candidate rows and runoff rows for one snapshot h days before the first round."""
    t1, t2 = election_dates(cycle)
    as_of = t1 - timedelta(days=h)
    d1, d2 = -h, (as_of - t2).days
    lineup_polls = polls.filter((pl.col("kind") == "lineup") & (pl.col("publish_est") <= as_of))
    if lineup_polls.select(pl.col("poll_id").n_unique()).item() < 5:
        return [], []
    blocs = registry.bloc_of(cycle)
    res = polling_average(lineup_polls, t1, cfg.polls, as_of=as_of, eval_days=[d1])
    obs = race_observations(res, d1)
    nl = fit_nested_logit(obs, blocs, cfg.nowcast)
    nl.sigma_imp = imputation_sigma(sorted(obs, key=lambda o: -o.weight)[:10], blocs, cfg.nowcast)
    ballot = [c.candidate_id for c in registry.ballot(cycle) if c.candidate_id in nl.candidates]
    fc = nowcast_lineup(tuple(ballot), nl, res, d1, cfg.nowcast)
    sd_p = candidate_sd(fc, models.horizon, h, nl.sigma_imp)
    mu_p = models.drift.apply(fc.log_share, h)
    comps = {"polls": (mu_p, sd_p)}
    if models.fundamentals is not None:
        mu_f, sd_f, _ = models.fundamentals.predict(cycle, ballot, seed=seed)
        comps["fundamentals"] = (mu_f - np.log(np.exp(mu_f).sum()), sd_f)
    open_race = not any((registry.get(cycle, c).incumbency or 0) >= 1 for c in ballot)
    ens = combine(comps, models.precision, h, fc.n_polls, open_race)

    # simulate this snapshot with the error model trained on earlier cycles
    ind_sd = models.errors.individual_sd(ens.sd_bar, fc.n_polls, ens.disagree)
    lineup = Lineup(tuple(ballot), 1.0, {})
    inp = SimInput(tuple(ballot), ens.mu, ind_sd, tuple(blocs[c] for c in ballot),
                   {b: models.errors.shock_sd("bloc", h) for b in set(blocs[c] for c in ballot)})
    runoff, runoff_rows = _runoff_model(cycle, polls, as_of, d2, ballot, ens.mu, nl, blocs, cfg, registry)
    # turnout as it could be known then: earlier elections, abstention polls published by `as_of`
    turnout = (build_turnout_model(cycle, as_of, blocs, cfg.turnout,
                                   parties={c.candidate_id: c.party for c in registry.candidates(cycle)})
               if cfg.turnout.enabled else None)
    sims = simulate_presidential([lineup], {lineup.key: inp}, runoff, days_out_t2=-d2,
                                 nu=models.errors.nu, runoff_a=models.runoff_a, runoff_b=models.runoff_b,
                                 runoff_rho=cfg.errors.runoff_rho, n_sims=n_sims, seed=seed, turnout=turnout,
                                 runoff_unpolled=models.runoff_unpolled,
                                 net_of_first_round=cfg.errors.runoff_net_of_first_round,
                                 far_right=models.far_right,
                                 sd_is_t_scale=models.errors.source == "stan")
    ci = {c: i for i, c in enumerate(sims.candidates)}

    results = np.array([registry.get(cycle, c).result_t1 for c in ballot], dtype=float)
    result_log = np.log(results / results.sum())
    finalists = sorted([c.candidate_id for c in registry.ballot(cycle) if c.result_t2 is not None])
    winner = max(registry.ballot(cycle), key=lambda c: c.result_t2 or 0).candidate_id
    rows = []
    for i, c in enumerate(ballot):
        k = ci[c]
        s = sims.shares[:, k]
        rows.append({
            "cycle": cycle, "days_out": h, "candidate_id": c, "bloc": blocs[c], "n_polls": fc.n_polls,
            "polled": fc.polled, "mu_polls": float(mu_p[i]), "mu_polls_raw": float(fc.log_share[i]),
            "sd_polls": float(sd_p[i]),
            "mu_fundamentals": float(comps["fundamentals"][0][i]) if "fundamentals" in comps else None,
            "sd_fundamentals": float(comps["fundamentals"][1][i]) if "fundamentals" in comps else None,
            "w_polls": float(ens.weights["polls"][i]),
            "w_fundamentals": float(ens.weights["fundamentals"][i]) if "fundamentals" in comps else 0.0,
            "mu_ens": float(ens.mu[i]), "sd_ens": float(ens.sd_bar[i]), "disagree": float(ens.disagree[i]),
            "result_log": float(result_log[i]), "err": float(result_log[i] - ens.mu[i]),
            "share_ens": float(100 * np.exp(ens.mu[i])), "result_share": float(100 * np.exp(result_log[i])),
            "q05": float(np.quantile(s, 0.05)), "q25": float(np.quantile(s, 0.25)),
            "q75": float(np.quantile(s, 0.75)), "q95": float(np.quantile(s, 0.95)),
            "p_runoff": float((sims.finalists == k).any(axis=1).mean()), "p_win": float((sims.winner == k).mean()),
            "made_runoff": c in finalists, "won": c == winner,
        })
    for r in runoff_rows:
        r.update({"cycle": cycle, "days_out": h})
    return rows, runoff_rows


def _runoff_model(cycle, polls, as_of, d2, ballot, mu, nl, blocs, cfg, registry):
    """Runoff model as known on `as_of`, plus an archive row for the actual finalists."""
    t1, t2 = election_dates(cycle)
    mpolls = polls.filter((pl.col("kind") == "matchup") & (pl.col("publish_est") <= as_of))
    v_mean = np.exp(mu)
    mobs = []
    if mpolls.select(pl.col("poll_id").n_unique()).item() >= 3:
        mres = polling_average(mpolls, t2, cfg.polls, as_of=as_of, eval_days=[d2])
        mobs = [o for o in matchup_observations(mres, d2) if o.a in ballot and o.b in ballot]
    first_round = {(o.a, o.b): (list(ballot), v_mean) for o in mobs}
    transfer = fit_transfer_model(mobs, first_round, blocs)
    model = RunoffModel(transfer, {(o.a, o.b): o for o in mobs}, cfg.nowcast.anchor_slack_one,
                        cfg.nowcast.anchor_floor, cfg.nowcast.anchor_floor_n)
    finalists = sorted(c.candidate_id for c in registry.ballot(cycle) if c.result_t2 is not None)
    rows = []
    if all(f in ballot for f in finalists):
        a, b = finalists
        exp_a = model.expected(a, b, list(ballot), v_mean)
        res_a = registry.get(cycle, a).result_t2 / 100
        rows.append({"a": a, "b": b, "expected_a": exp_a, "result_a": res_a,
                     "days_to_t2": -d2, "err_logit": float(_logit(res_a) - _logit(exp_a)),
                     "polled_matchup": (a, b) in model.polled})
    return model, rows


def run_presidential_backtest(
    cycles: list[int] | tuple[int, ...] = PRES_CYCLES,
    n_sims: int | None = None,
    snapshot_days: tuple[int, ...] = SNAPSHOT_DAYS,
    cfg: Config | None = None,
    write: bool = True,
) -> pl.DataFrame:
    """Run the backtest; returns (and writes) the candidate archive."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    cfg = cfg or default_config()
    n_sims = n_sims or cfg.simulation.n_sims_backtest
    registry = CandidateRegistry.load()
    features = PresFeatures(registry, cfg.lean.recent_weight)
    long = load_polls_long()
    archive = pl.DataFrame()
    runoff_archive = pl.DataFrame()
    for cycle in sorted(PRES_CYCLES):
        if cycle > max(cycles):
            break
        models = cycle_models(cycle, archive, runoff_archive, long, registry, features, cfg)
        polls = long.filter(pl.col("year") == cycle)
        rows, rrows = [], []
        for i, h in enumerate(snapshot_days):
            r, rr = snapshot(cycle, h, polls, registry, models, cfg, n_sims, seed=cfg.simulation.seed + i)
            rows.extend(r)
            rrows.extend(rr)
        log.info("cycle %s: %d rows", cycle, len(rows))
        archive = pl.concat([archive, pl.DataFrame(rows)], how="diagonal_relaxed") if rows else archive
        runoff_archive = (pl.concat([runoff_archive, pl.DataFrame(rrows)], how="diagonal_relaxed")
                          if rrows else runoff_archive)
    archive = archive.filter(pl.col("cycle").is_in(list(cycles)))
    if write:
        paths.ensure_dirs()
        archive.write_parquet(paths.ARCHIVE / "pres_backtest.parquet")
        runoff_archive.write_parquet(paths.ARCHIVE / "pres_backtest_runoff.parquet")
    return archive

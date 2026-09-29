"""Presidential forecast, polls-only v0 (PLAN.md › M3).

    polls_long → pass-2 averages (lineups, matchups) → nested-logit now-cast
    per scenario lineup → horizon error model → runoff transfer model →
    40,000 simulations → summary tables in data/outputs/.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from forecast import paths
from forecast.config import Config, default_config
from forecast.ingest.build import load_polls_long
from forecast.ingest.registry import CandidateRegistry
from forecast.ingest.wiki_long import election_dates
from forecast.polls.ewma import polling_average
from forecast.polls.horizon import HorizonModel, fit_horizon_model, historical_errors
from forecast.polls.lineups import fit_nested_logit, imputation_sigma, nowcast_lineup, race_observations
from forecast.simulate.presidential import Decision, PresSims, SimInput, candidate_sd, simulate_presidential
from forecast.simulate.runoff import RunoffModel, fit_transfer_model, matchup_observations
from forecast.simulate.scenarios import Lineup, ScenarioSpec, decision_variants

HISTORY = (2002, 2007, 2012, 2017, 2022)


def horizon_model(long: pl.DataFrame, registry: CandidateRegistry, cycles: tuple[int, ...], cfg: Config) -> HorizonModel:
    """v0 error model fitted on earlier cycles (cached per set of cycles)."""
    path = paths.PROCESSED / f"horizon_{'_'.join(map(str, cycles))}.json"
    if path.exists():
        return HorizonModel.load(path)
    model = fit_horizon_model(historical_errors(long, registry, list(cycles), cfg.polls))
    paths.ensure_dirs()
    model.save(path)
    return model


def _reference_lineup(pair: tuple[str, str], lineups: list[Lineup], spec: ScenarioSpec) -> tuple[str, ...]:
    """A plausible first round for fitting a matchup: the likeliest scenario
    lineup containing both finalists, else the likeliest lineup with each
    missing finalist swapped in for its slot rivals."""
    with_both = [l for l in lineups if set(pair) <= set(l.candidates)]
    if with_both:
        return max(with_both, key=lambda l: l.probability).candidates
    base = set(max(lineups, key=lambda l: l.probability).candidates)
    for x in pair:
        if x in base:
            continue
        for opts in spec.slots.values():
            if x in opts:
                base -= {o for o in opts if o}
        base.add(x)
    return tuple(sorted(base))


@dataclass
class PresSummary:
    year: int
    as_of: date
    days_out: int
    candidates: pl.DataFrame
    matchups: pl.DataFrame
    blocs: pl.DataFrame
    lineups: pl.DataFrame
    meta: dict = field(default_factory=dict)

    def report(self, top: int = 12) -> str:
        lines = [f"French presidential election {self.year} — {self.meta.get('mode', 'polls-only')} forecast as of {self.as_of} "
                 f"({self.days_out} days to the first round, {self.meta.get('n_sims')} simulations)", ""]
        lines.append(f"{'candidate':<24}{'P(run)':>8}{'P(runoff)':>11}{'P(win)':>8}{'median %':>10}{'90% interval':>16}")
        for r in self.candidates.head(top).iter_rows(named=True):
            interval = f"{r['share_q05']:.1f}–{r['share_q95']:.1f}"
            lines.append(f"{r['name']:<24}{r['p_run']:>8.0%}{r['p_runoff']:>11.1%}{r['p_win']:>8.1%}"
                         f"{r['share_q50']:>10.1f}{interval:>16}")
        if self.meta.get("decision_outcomes"):
            d = self.meta["decision"]
            lines += ["", f"Centrist decision ({d.name}, {d.realised:.0%} of the forecast error realised by then): "
                      + ", ".join(f"{k} {v:.0%}" for k, v in self.meta["decision_outcomes"].items())]
        lines += ["", "Most likely runoffs:"]
        for r in self.matchups.head(6).iter_rows(named=True):
            lines.append(f"  {r['matchup']:<40}{r['p']:>7.1%}   first named wins {r['p_first_wins']:.0%}")
        lines += ["", "By bloc:"]
        for r in self.blocs.iter_rows(named=True):
            lines.append(f"  {r['bloc']:<6} P(runoff) {r['p_runoff']:.1%}   P(win) {r['p_win']:.1%}")
        return "\n".join(lines)

    def write(self, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or paths.OUTPUTS / f"pres{self.year}_{self.as_of.isoformat()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates.write_csv(out_dir / "candidates.csv")
        self.matchups.write_csv(out_dir / "matchups.csv")
        self.blocs.write_csv(out_dir / "blocs.csv")
        self.lineups.write_csv(out_dir / "lineups.csv")
        (out_dir / "report.txt").write_text(self.report(top=40), encoding="utf-8")
        return out_dir


def summarise(sims: PresSims, registry: CandidateRegistry, year: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    names = {c.candidate_id: c.surname for c in registry.candidates(year)}
    blocs = registry.bloc_of(year)
    runs = ~np.isnan(sims.shares)
    rows = []
    for i, c in enumerate(sims.candidates):
        ran = runs[:, i]
        in_runoff = (sims.finalists == i).any(axis=1)
        s = sims.shares[ran, i]
        rows.append({
            "candidate_id": c, "name": names.get(c, c), "bloc": blocs.get(c),
            "p_run": ran.mean(), "p_runoff": in_runoff.mean(), "p_win": (sims.winner == i).mean(),
            "p_runoff_if_running": in_runoff[ran].mean() if ran.any() else np.nan,
            "p_win_if_running": (sims.winner[ran] == i).mean() if ran.any() else np.nan,
            **{f"share_q{int(q*100):02d}": float(np.quantile(s, q)) if len(s) else np.nan
               for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        })
    cand = pl.DataFrame(rows).sort("p_win", "p_runoff", descending=True)

    pairs: dict[tuple[int, int], list[int]] = {}
    has_runoff = sims.finalists[:, 1] >= 0
    for k in np.where(has_runoff)[0]:
        a, b = sorted(sims.finalists[k])
        pairs.setdefault((a, b), []).append(k)
    mrows = []
    for (a, b), ks in pairs.items():
        ks = np.array(ks)
        mrows.append({"matchup": f"{names.get(sims.candidates[a])} vs {names.get(sims.candidates[b])}",
                      "a": sims.candidates[a], "b": sims.candidates[b], "p": len(ks) / sims.n,
                      "p_first_wins": float((sims.winner[ks] == a).mean())})
    match = pl.DataFrame(mrows).sort("p", descending=True)

    brows = []
    for bl in sorted(set(blocs.get(c) for c in sims.candidates)):
        cols = [i for i, c in enumerate(sims.candidates) if blocs.get(c) == bl]
        brows.append({"bloc": bl, "p_runoff": float(np.isin(sims.finalists, cols).any(axis=1).mean()),
                      "p_win": float(np.isin(sims.winner, cols).mean())})
    return cand, match, pl.DataFrame(brows).sort("p_win", descending=True)


def run_forecast(
    year: int = 2027,
    as_of: date | None = None,
    n_sims: int | None = None,
    cfg: Config | None = None,
    long: pl.DataFrame | None = None,
    spec: ScenarioSpec | None = None,
    write: bool = False,
    use_ensemble: bool = True,
) -> PresSummary:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = cfg or default_config()
    registry = CandidateRegistry.load()
    long = long if long is not None else load_polls_long()
    spec = spec or ScenarioSpec.load()
    t1, t2 = election_dates(year)
    polls = long.filter(pl.col("year") == year)
    as_of = as_of or min(polls["publish_est"].max(), date.today())
    d1, d2 = (as_of - t1).days, (as_of - t2).days
    blocs = registry.bloc_of(year)

    # first round: averages → nested-logit now-cast per scenario lineup
    res = polling_average(polls.filter(pl.col("kind") == "lineup"), t1, cfg.polls, as_of=as_of, eval_days=[d1])
    obs = race_observations(res, d1)
    nl = fit_nested_logit(obs, blocs, cfg.nowcast)
    nl.sigma_imp = imputation_sigma(obs, blocs, cfg.nowcast)
    lineups = [l for l in spec.enumerate() if nl.knows(l.candidates)]
    dropped = 1 - sum(l.probability for l in lineups)
    if dropped > 1e-9:
        tot = sum(l.probability for l in lineups)
        lineups = [Lineup(l.candidates, l.probability / tot, l.slots) for l in lineups]
    # every lineup a simulation can end up in: scenario lineups, and each with
    # the losers of a mid-campaign decision (e.g. the January centrist rule) removed
    variants = list({v: Lineup(v, l.probability, l.slots) for l in lineups
                     for v in decision_variants(l.candidates, spec.decisions)}.values())
    variants = [v for v in variants if nl.knows(v.candidates)]
    forecasts = {v.key: nowcast_lineup(v.candidates, nl, res, d1, cfg.nowcast) for v in variants}

    # runoff: matchup averages + transfer model
    mres = polling_average(polls.filter(pl.col("kind") == "matchup"), t2, cfg.polls, as_of=as_of, eval_days=[d2])
    mobs = [o for o in matchup_observations(mres, d2) if nl.knows((o.a, o.b))]
    first_round = {}
    for o in mobs:
        ref = _reference_lineup((o.a, o.b), lineups, spec)
        first_round[(o.a, o.b)] = (list(ref), nl.shares(ref))
    transfer = fit_transfer_model(mobs, first_round, blocs)
    runoff = RunoffModel(transfer, {(o.a, o.b): o for o in mobs}, cfg.nowcast.anchor_slack_one,
                         cfg.nowcast.anchor_floor, cfg.nowcast.anchor_floor_n)

    hm = horizon_model(long, registry, tuple(c for c in HISTORY if c < year), cfg)
    nu = max(cfg.errors.nu_floor, len(hm.cycles))
    n = n_sims or cfg.simulation.n_sims
    mode, ens_meta = "polls-only", {}
    runoff_a, runoff_b = cfg.errors.runoff_a, cfg.errors.runoff_b
    archive_path = paths.ARCHIVE / "pres_backtest.parquet"
    if use_ensemble and archive_path.exists():
        inputs, ens_meta, (runoff_a, runoff_b), nu = _ensemble_inputs(
            year, variants, forecasts, hm, nl, blocs, registry, cfg, -d1, pl.read_parquet(archive_path))
        mode = "ensemble"
    else:
        inputs = {v.key: SimInput.from_forecast(fc, candidate_sd(fc, hm, -d1, nl.sigma_imp), blocs)
                  for v in variants for fc in [forecasts[v.key]]}
    decision = None
    for dspec in spec.decisions:   # one decision supported per run (the centrist rule)
        h_dec = max((t1 - dspec.date).days, 0)
        s_now, s_dec = hm.a + hm.b * np.sqrt(-d1), hm.a + hm.b * np.sqrt(h_dec)
        realised = 0.0 if as_of >= dspec.date else float(1 - (s_dec / s_now) ** 2)
        decision = Decision(dspec.name, dspec.pool, dspec.threshold, dspec.fallback_top, dspec.fallback_above,
                            realised)
    sims = simulate_presidential(
        lineups, inputs, runoff, days_out_t2=-d2, nu=nu,
        runoff_a=runoff_a, runoff_b=runoff_b, runoff_rho=cfg.errors.runoff_rho,
        n_sims=n, seed=cfg.simulation.seed, sd_is_t_scale=(mode == "ensemble"), decision=decision,
    )
    decision_outcomes = {}
    if decision is not None:
        runs = {c: ~np.isnan(sims.shares[:, sims.candidates.index(c)]) for c in decision.pool}
        names = {c: registry.get(year, c).surname for c in decision.pool}
        for c in decision.pool:
            others = [o for o in decision.pool if o != c]
            decision_outcomes[f"{names[c]} alone"] = float((runs[c] & ~np.any([runs[o] for o in others], axis=0)).mean())
        decision_outcomes["several"] = float((np.sum([runs[c] for c in decision.pool], axis=0) >= 2).mean())
    cand, match, bl = summarise(sims, registry, year)
    lu = pl.DataFrame([{"lineup": " ".join(registry.get(year, c).surname for c in l.candidates),
                        "probability": l.probability, "polled": forecasts[l.key].polled,
                        "n_polls": forecasts[l.key].n_polls} for l in lineups]).sort("probability", descending=True)
    summary = PresSummary(year, as_of, -d1, cand, match, bl, lu, meta={
        "n_sims": n, "nu": nu, "sigma_imp": nl.sigma_imp, "nest_mu": nl.mu, "horizon_model": hm,
        "transfer": {"alpha": transfer.alpha, "beta": transfer.beta, "appeal": transfer.appeal},
        "scenario_mass_dropped": dropped, "mode": mode, "decision": decision,
        "decision_outcomes": decision_outcomes, **ens_meta,
    })
    if write:
        summary.write()
    return summary


def _ensemble_inputs(year, lineups, forecasts, hm, nl, blocs, registry, cfg, h, archive):
    """Polls + fundamentals ensemble per lineup, errors from the meta-model,
    all trained on the full out-of-sample backtest archive."""
    from forecast.backtest.harness import FUND_YEARS, fit_runoff_scale
    from forecast.ensemble.stacking import PrecisionModel, combine
    from forecast.errors.meta_model import fit_error_model
    from forecast.fundamentals.dlm import fit_fundamentals
    from forecast.fundamentals.features_pres import PresFeatures

    features = PresFeatures(registry, cfg.lean.recent_weight)
    fund = fit_fundamentals(features, [y for y in FUND_YEARS if y < year])
    precision = PrecisionModel.fit(archive, ("polls", "fundamentals"))
    errors = fit_error_model(archive, ("bloc",), nu_floor=cfg.errors.nu_floor, thin_every=2)
    runoff_path = paths.ARCHIVE / "pres_backtest_runoff.parquet"
    runoff_scale = fit_runoff_scale(pl.read_parquet(runoff_path) if runoff_path.exists() else pl.DataFrame(),
                                    (cfg.errors.runoff_a, cfg.errors.runoff_b))
    inputs = {}
    weights = []
    for l in lineups:
        fc = forecasts[l.key]
        comps = {"polls": (fc.log_share, candidate_sd(fc, hm, h, nl.sigma_imp))}
        mu_f, sd_f, _ = fund.predict(year, list(fc.lineup))
        comps["fundamentals"] = (mu_f - np.log(np.exp(mu_f).sum()), sd_f)
        open_race = not any((registry.get(year, c).incumbency or 0) >= 1 for c in fc.lineup)
        ens = combine(comps, precision, h, fc.n_polls, open_race)
        # (the drift term is off: it did not improve the out-of-sample backtest)
        sd = errors.individual_sd(ens.sd_bar, fc.n_polls, ens.disagree)
        inputs[l.key] = SimInput(tuple(fc.lineup), ens.mu, sd, tuple(blocs[c] for c in fc.lineup),
                                 {b: errors.shock_sd("bloc", h) for b in set(blocs[c] for c in fc.lineup)},
                                 v_mean=np.exp(ens.mu))
        weights.append(float(np.mean(ens.weights["polls"])))
    meta = {"mean_polls_weight": float(np.average(weights, weights=[l.probability for l in lineups])),
            "error_model": errors, "precision": precision.scales, "fundamentals_years": fund.train_years}
    return inputs, meta, runoff_scale, errors.nu


def write_averages(year: int, as_of: date | None = None, cfg: Config | None = None) -> Path:
    """Daily pass-2 averages for every race (for charts)."""
    cfg = cfg or default_config()
    long = load_polls_long().filter(pl.col("year") == year)
    t1, t2 = election_dates(year)
    out_dir = paths.OUTPUTS / f"pres{year}_averages"
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, day0 in (("lineup", t1), ("matchup", t2)):
        res = polling_average(long.filter(pl.col("kind") == kind), day0, cfg.polls, as_of=as_of)
        res.averages.write_parquet(out_dir / f"{kind}_averages.parquet")
        res.house_table.write_csv(out_dir / f"{kind}_house_effects.csv")
    return out_dir

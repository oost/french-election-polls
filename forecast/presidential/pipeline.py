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
from forecast.simulate.runoff import (
    FAR_RIGHT, RunoffModel, far_right_runoff_bias, fit_transfer_model, matchup_observations,
)
from forecast.simulate.scenarios import Lineup, ScenarioSpec, decision_variants, option_candidates
from forecast.simulate.turnout import build_turnout_model, turnout_history

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
            members = set().union(*map(option_candidates, opts))
            if x in members:
                base -= members
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
    conditional: pl.DataFrame = field(default_factory=pl.DataFrame)   # see summarise_by_slot
    turnout: pl.DataFrame = field(default_factory=pl.DataFrame)       # see summarise_turnout
    far_right: pl.DataFrame = field(default_factory=pl.DataFrame)     # see summarise_far_right
    turnout_matchups: pl.DataFrame = field(default_factory=pl.DataFrame)
    meta: dict = field(default_factory=dict)

    def conditional_matrix(self, slot: str = "centre_right", measure: str = "p_win",
                           min_runoff: float = 0.01) -> pl.DataFrame:
        """One row per candidate, one column per option of scenario slot `slot`
        (in the YAML's order): `measure` among the simulations that drew that
        option, null where the candidate is not on that ballot.

        Rows are the slot's candidates and anyone with at least `min_runoff`
        overall chance of the runoff, by overall P(win), then the slot's
        candidates taken together ("any …")."""
        df = self.conditional.filter(pl.col("slot") == slot) if not self.conditional.is_empty() else self.conditional
        if df.is_empty():
            raise ValueError(f"no scenario slot {slot!r} with several options in this forecast")
        members = set().union(*map(option_candidates, df["option"].unique()))
        overall = self.candidates.filter(pl.col("candidate_id").is_in(members) | (pl.col("p_runoff") >= min_runoff))
        order = pl.DataFrame({"candidate_id": [*overall["candidate_id"], ANY]}).with_row_index("row")
        wide = (df.with_columns(pl.when(pl.col("p_run") > 0).then(pl.col(measure)).alias("value"))
                .pivot(on="ballot", index=["candidate_id", "name"], values="value"))
        return order.join(wide, on="candidate_id", how="left").sort("row").drop("row", "candidate_id")

    def _conditional_lines(self, slot: str) -> list[str]:
        df = self.conditional.filter(pl.col("slot") == slot)
        heads = df.unique("ballot", maintain_order=True)
        short = _short_labels(heads["ballot"].to_list())
        width = max(7, max(map(len, short.values())) + 2)
        rows = {m: self.conditional_matrix(slot, m) for m in ("p_runoff", "p_win")}
        lines = ["", f"Given the {slot} ballot (each column uses only the simulations that drew it), %:",
                 f"  {'':<22}" + "".join(f"{short[b]:>{width}}" for b in heads["ballot"]),
                 f"  {'ballot probability':<22}" + "".join(f"{100 * p:>{width}.0f}" for p in heads["p_ballot"])]
        for m, label in (("p_runoff", "reaches the runoff"), ("p_win", "wins")):
            lines.append(f"  {label}")
            for r in rows[m].iter_rows(named=True):
                cells = ("–" if r[b] is None else f"{100 * r[b]:.1f}" for b in heads["ballot"])
                lines.append(f"    {r['name']:<20}" + "".join(f"{c:>{width}}" for c in cells))
        n_min = int(heads["n_sims"].min())
        key = sorted({(s, n) for b in heads["ballot"] for s, n in zip(short[b].split("+"), b.split(" + "))
                      if s != n})
        notes = ["– not on that ballot", f"simulation noise up to ±{1.96 * 50 / np.sqrt(n_min):.1f} points "
                                         f"in the smallest column ({n_min:,} simulations)"]
        if key:
            notes.insert(0, ", ".join(f"{s} = {n}" for s, n in key))
        lines.append("  " + "; ".join(notes))
        return lines

    def report(self, top: int = 12) -> str:
        lines = [f"French presidential election {self.year} — {self.meta.get('mode', 'polls-only')} forecast as of {self.as_of} "
                 f"({self.days_out} days to the first round, {self.meta.get('n_sims')} simulations)", ""]
        dropped = self.meta.get("scenario_mass_dropped", 0.0)
        if dropped > 1e-3:
            lines += [f"WARNING: {dropped:.0%} of the scenario probability is in lineups the now-cast cannot place "
                      "(a candidate with no polls, or a scenario option it does not recognise). Those lineups "
                      "were dropped and the rest renormalised.", ""]
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
        lines += self._far_right_lines()
        lines += self._turnout_lines()
        if not self.conditional.is_empty():
            for slot in self.conditional["slot"].unique(maintain_order=True):
                lines += self._conditional_lines(slot)
        return "\n".join(lines)

    def _far_right_lines(self) -> list[str]:
        if self.far_right.is_empty():
            return []
        p, mean, sd, by_year = self.meta["far_right"]
        past = ", ".join(f"{y} {v:+.2f}" for y, v in by_year.items())
        lines = ["", f"Far-right runoff scenario: in {p:.0%} of simulations a far-right finalist does worse in the runoff than",
                 f"expected, by {-mean:.2f} ± {sd:.2f} on the logit of their share (about {100 * (0.55 - 1 / (1 + np.exp(-(np.log(0.55 / 0.45) + mean)))):.1f} "
                 f"points at 55%). Past cases, result against final polls: {past}.",
                 f"  {'chance of winning':<24}{'overall':>9}{'if it happens':>15}{'if it does not':>16}"]
        for r in self.far_right.iter_rows(named=True):
            lines.append(f"  {r['name']:<24}{r['p_win']:>9.1%}{r['p_win_if_shortfall']:>15.1%}{r['p_win_if_not']:>16.1%}")
        return lines

    def _turnout_lines(self) -> list[str]:
        if self.turnout.is_empty():
            return []
        tm = self.meta.get("turnout_model")
        basis = "past elections only (no abstention polls yet)"
        if tm is not None and (tm.n_polls_t1 or tm.runoff_polls):
            basis = (f"past elections and abstention polls ({tm.n_polls_t1} first-round, "
                     f"{sum(e[2] for e in tm.runoff_polls.values())} runoff)")
        lines = ["", f"Turnout, % of registered voters — from {basis}:",
                 f"  {'':<28}{'median':>8}{'90% interval':>16}{'last time':>11}"]
        for r in self.turnout.iter_rows(named=True):
            interval = f"{r['q05']:.1f}–{r['q95']:.1f}"
            lines.append(f"  {r['measure']:<28}{r['q50']:>8.1f}{interval:>16}{r['last']:>11.1f}")
        if not self.turnout_matchups.is_empty():
            m = self.turnout_matchups
            lines += ["", "Runoff by matchup: turnout, and the first-named candidate's chance of winning when turnout comes",
                      f"in about {-m['surprise_low'].mean():.0f} points below or {m['surprise_high'].mean():.0f} points above "
                      "what the first round would normally give (lowest and",
                      "highest third of that matchup's simulations; ± is the 95% simulation margin):",
                      f"  {'':<22}{'turnout':>8}{'wins':>7}{'if turnout low':>17}{'if turnout high':>17}{'share per +3 pts':>18}"]
            for r in m.iter_rows(named=True):
                low = f"{r['p_first_wins_low']:.0%} ±{196 * r['p_first_wins_low_se']:.0f}"
                high = f"{r['p_first_wins_high']:.0%} ±{196 * r['p_first_wins_high_se']:.0f}"
                slope = f"{r['share_per_3pts']:+.1f} ±{1.96 * r['share_per_3pts_se']:.1f}"
                lines.append(f"  {r['matchup']:<22}{r['turnout_q50']:>8.1f}{r['p_first_wins']:>7.0%}{low:>17}{high:>17}{slope:>18}"
                             + ("" if r["clear_effect"] else "   no clear effect"))
        return lines

    def write(self, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or paths.OUTPUTS / f"pres{self.year}_{self.as_of.isoformat()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.candidates.write_csv(out_dir / "candidates.csv")
        self.matchups.write_csv(out_dir / "matchups.csv")
        self.blocs.write_csv(out_dir / "blocs.csv")
        self.lineups.write_csv(out_dir / "lineups.csv")
        if not self.conditional.is_empty():
            self.conditional.write_csv(out_dir / "conditional.csv")
        if not self.far_right.is_empty():
            self.far_right.write_csv(out_dir / "far_right_runoff.csv")
        if not self.turnout.is_empty():
            self.turnout.write_csv(out_dir / "turnout.csv")
            self.turnout_matchups.write_csv(out_dir / "turnout_matchups.csv")
        (out_dir / "report.txt").write_text(self.report(top=40), encoding="utf-8")
        return out_dir


ANY = "ANY"   # candidate_id of the rows summing up a scenario slot's candidates


def _short_labels(ballots: list[str]) -> dict[str, str]:
    """Column labels for the text report: the ballots themselves when short,
    else initials ("Philippe + Lisnard" → "P+L"; first three letters if two
    surnames share an initial)."""
    if all(len(b) <= 8 for b in ballots):
        return {b: b for b in ballots}
    names = sorted({n for b in ballots if b != "nobody" for n in b.split(" + ")})
    k = 1 if len({n[0] for n in names}) == len(names) else 3
    return {b: b if b == "nobody" else "+".join(n.replace(" ", "")[:k] for n in b.split(" + ")) for b in ballots}


def _quantiles(s: np.ndarray) -> dict[str, float]:
    return {f"share_q{int(q*100):02d}": float(np.quantile(s, q)) if len(s) else np.nan
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)}


def _candidate_rows(sims: PresSims, names: dict[str, str], blocs: dict[str, str],
                    rows: np.ndarray | None = None) -> list[dict]:
    """Each candidate's chances over all simulations, or over `rows` only."""
    shares, finalists, winner = sims.shares, sims.finalists, sims.winner
    if rows is not None:
        shares, finalists, winner = shares[rows], finalists[rows], winner[rows]
    runs = ~np.isnan(shares)
    out = []
    for i, c in enumerate(sims.candidates):
        ran = runs[:, i]
        in_runoff = (finalists == i).any(axis=1)
        out.append({
            "candidate_id": c, "name": names.get(c, c), "bloc": blocs.get(c),
            "p_run": ran.mean(), "p_runoff": in_runoff.mean(), "p_win": (winner == i).mean(),
            "p_runoff_if_running": in_runoff[ran].mean() if ran.any() else np.nan,
            "p_win_if_running": (winner[ran] == i).mean() if ran.any() else np.nan,
            **_quantiles(shares[ran, i]),
        })
    return out


def summarise_by_slot(sims: PresSims, slots: dict[str, dict[str | None, float]],
                      names: dict[str, str], blocs: dict[str, str]) -> pl.DataFrame:
    """Each candidate's chances given each option of every scenario slot that
    has several, from the simulations that drew it: one row per slot × option ×
    candidate. Averaging an option's rows weighted by `p_ballot` gives back the
    headline numbers.

    Each slot and option also gets a row for the slot's candidates taken
    together (candidate_id "ANY"): whether any of them reaches the runoff or
    wins, and their combined first-round share."""
    out = []
    for slot, opts in slots.items():
        drawn = np.array([l.slots.get(slot) or "" for l in sims.lineups], dtype=object)[sims.lineup_idx]
        present = [o for o in opts if (drawn == (o or "")).any()]
        members = sorted(sims.candidates.index(c) for c in set().union(*map(option_candidates, present)))
        if len(present) < 2 or not members:
            continue
        label = slot.upper() if len(slot) <= 3 else slot.replace("_", "/")
        for opt in present:
            rows = np.where(drawn == (opt or ""))[0]
            head = {"slot": slot, "option": opt or "",
                    "ballot": " + ".join(names.get(c, c) for c in opt.split("+")) if opt else "nobody",
                    "p_ballot": len(rows) / sims.n, "n_sims": len(rows)}
            out += [{**head, **r} for r in _candidate_rows(sims, names, blocs, rows)]
            fin, win = sims.finalists[rows], sims.winner[rows]
            sh = sims.shares[np.ix_(rows, members)]
            ran = ~np.isnan(sh).all(axis=1)
            runoff, won = np.isin(fin, members).any(axis=1), np.isin(win, members)
            out.append({**head, "candidate_id": ANY, "name": f"any {label}", "bloc": None,
                        "p_run": ran.mean(), "p_runoff": runoff.mean(), "p_win": won.mean(),
                        "p_runoff_if_running": runoff[ran].mean() if ran.any() else np.nan,
                        "p_win_if_running": won[ran].mean() if ran.any() else np.nan,
                        **_quantiles(np.nansum(sh[ran], axis=1))})
    return pl.DataFrame(out, infer_schema_length=None)


def summarise_far_right(sims: PresSims, names: dict[str, str], min_p: float = 0.01) -> pl.DataFrame:
    """Each candidate's chance of winning overall, in the simulations that drew
    "the far right does worse than its runoff polls again" and in those that
    did not (candidates with at least `min_p` overall)."""
    if sims.far_right_shortfall is None:
        return pl.DataFrame()
    yes = sims.far_right_shortfall
    rows = [{"candidate_id": c, "name": names.get(c, c), "p_win": float((sims.winner == i).mean()),
             "p_win_if_shortfall": float((sims.winner[yes] == i).mean()),
             "p_win_if_not": float((sims.winner[~yes] == i).mean())} for i, c in enumerate(sims.candidates)]
    return pl.DataFrame(rows).filter(pl.col("p_win") >= min_p).sort("p_win", descending=True)


CLEAR_SHARE = 0.3    # points of vote share per three points of turnout
CLEAR_Z = 3.0        # … and this many simulation standard errors


def summarise_turnout(sims: PresSims, names: dict[str, str], year: int,
                      min_p: float = 0.03) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Turnout over the simulations (needs a turnout model in the simulation).

    First table: quantiles of first-round turnout, runoff turnout and runoff
    valid votes, in % of registered voters, with the last election's figure.

    Second: per runoff matchup in at least `min_p` of simulations, its
    turnout, and what a turnout surprise does to the first-named candidate.
    The surprise is runoff turnout minus what the same first round gives
    without mobilisation shocks (points; `surprise_q05`/`q95` its range). The
    table has the chance of winning overall and in the thirds of the matchup's
    simulations with the lowest and highest surprise, and the change in vote
    share that goes with three more points of turnout (`share_per_3pts`, a
    regression slope). The `_se` columns are simulation standard errors.
    `clear_effect` is true when that change is at least `CLEAR_SHARE` points
    and at least `CLEAR_Z` times its standard error; otherwise the low/high
    difference should not be read as an effect of turnout."""
    if sims.turnout_t1 is None:
        return pl.DataFrame(), pl.DataFrame()
    last = {r["round"]: r for r in turnout_history().filter(pl.col("year") < year).tail(2).iter_rows(named=True)}
    has = ~np.isnan(sims.turnout_t2)
    qs = (0.05, 0.25, 0.5, 0.75, 0.95)
    rows = [{"measure": label, **{f"q{int(q * 100):02d}": float(100 * np.quantile(x, q)) for q in qs},
             "mean": float(100 * x.mean()), "last": 100 * ref}
            for label, x, ref in (("First round turnout", sims.turnout_t1, last[1]["turnout"]),
                                  ("Runoff turnout", sims.turnout_t2[has], last[2]["turnout"]),
                                  ("Runoff valid votes", sims.valid_t2[has], last[2]["valid"]))]
    mrows = []
    fin = np.sort(sims.finalists, axis=1)
    for a, b in sorted({(int(x), int(y)) for x, y in fin[has]}):
        k = np.where(has & (fin[:, 0] == a) & (fin[:, 1] == b))[0]
        if len(k) / sims.n < min_p:
            continue
        t, won = 100 * sims.turnout_t2[k], sims.winner[k] == a
        change = t - 100 * sims.turnout_t1[k]
        surprise = sims.turnout_surprise[k]
        share_a = np.where(won, sims.runoff_share[k], 100 - sims.runoff_share[k])
        lo, hi = np.quantile(surprise, [1 / 3, 2 / 3])
        low, high = won[surprise <= lo], won[surprise >= hi]
        slope, intercept = np.polyfit(surprise, share_a, 1)
        slope_se = float(np.sqrt((share_a - slope * surprise - intercept).var() / (len(k) * surprise.var())))
        mrows.append({"matchup": f"{names.get(sims.candidates[a])} vs {names.get(sims.candidates[b])}",
                      "a": sims.candidates[a], "b": sims.candidates[b], "p": len(k) / sims.n,
                      **{f"turnout_q{int(q * 100):02d}": float(np.quantile(t, q)) for q in (0.05, 0.5, 0.95)},
                      "valid_q50": float(100 * np.median(sims.valid_t2[k])),
                      "change_q50": float(np.median(change)),
                      "surprise_q05": float(np.quantile(surprise, 0.05)), "surprise_q95": float(np.quantile(surprise, 0.95)),
                      "surprise_low": float(surprise[surprise <= lo].mean()), "surprise_high": float(surprise[surprise >= hi].mean()),
                      "p_first_wins": float(won.mean()),
                      "p_first_wins_low": float(low.mean()), "p_first_wins_high": float(high.mean()),
                      "p_first_wins_low_se": float(np.sqrt(low.mean() * (1 - low.mean()) / len(low))),
                      "p_first_wins_high_se": float(np.sqrt(high.mean() * (1 - high.mean()) / len(high))),
                      "share_per_3pts": float(3 * slope), "share_per_3pts_se": 3 * slope_se,
                      "clear_effect": bool(abs(3 * slope) >= CLEAR_SHARE and abs(slope) >= CLEAR_Z * slope_se),
                      "n_sims": len(k)})
    return pl.DataFrame(rows), pl.DataFrame(mrows).sort("p", descending=True) if mrows else pl.DataFrame()


def summarise(sims: PresSims, registry: CandidateRegistry, year: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    names = {c.candidate_id: c.surname for c in registry.candidates(year)}
    blocs = registry.bloc_of(year)
    cand = pl.DataFrame(_candidate_rows(sims, names, blocs)).sort("p_win", "p_runoff", descending=True)

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
    runoff_a, runoff_b, runoff_unpolled = cfg.errors.runoff_a, cfg.errors.runoff_b, None
    archive_path = paths.ARCHIVE / "pres_backtest.parquet"
    if use_ensemble and archive_path.exists():
        inputs, ens_meta, (runoff_a, runoff_b, runoff_unpolled), nu = _ensemble_inputs(
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
    turnout = (build_turnout_model(year, as_of, blocs, cfg.turnout,
                                   parties={c.candidate_id: c.party for c in registry.candidates(year)})
               if cfg.turnout.enabled else None)
    far_right, bias = None, None
    if cfg.errors.far_right_runoff_probability > 0:
        bias = far_right_runoff_bias(long, registry, year, cfg.errors.far_right_runoff_noise,
                                     cfg.errors.far_right_runoff_prior_sd, cfg.errors.far_right_runoff_days)
        if bias is not None:
            far_right = (cfg.errors.far_right_runoff_probability, bias[0], bias[1])
    sims = simulate_presidential(
        lineups, inputs, runoff, days_out_t2=-d2, nu=nu,
        runoff_a=runoff_a, runoff_b=runoff_b, runoff_rho=cfg.errors.runoff_rho,
        n_sims=n, seed=cfg.simulation.seed, sd_is_t_scale=(mode == "ensemble"), decision=decision,
        turnout=turnout, runoff_unpolled=runoff_unpolled, net_of_first_round=cfg.errors.runoff_net_of_first_round,
        far_right=far_right,
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
    names = {c.candidate_id: c.surname for c in registry.candidates(year)}
    cond = summarise_by_slot(sims, spec.slots, names, blocs)
    turn, turn_m = summarise_turnout(sims, names, year)
    far = summarise_far_right(sims, names)
    lu = pl.DataFrame([{"lineup": " ".join(registry.get(year, c).surname for c in l.candidates),
                        "probability": l.probability, "polled": forecasts[l.key].polled,
                        "n_polls": forecasts[l.key].n_polls} for l in lineups]).sort("probability", descending=True)
    summary = PresSummary(year, as_of, -d1, cand, match, bl, lu, conditional=cond, turnout=turn,
                          turnout_matchups=turn_m, far_right=far, meta={
        "far_right": None if far_right is None else (*far_right, bias[2]),
        "turnout_model": turnout,
        "n_sims": n, "nu": nu, "sigma_imp": nl.sigma_imp, "nest_mu": nl.mu, "nest_lambda": nl.lam, "horizon_model": hm,
        "transfer": {"alpha": transfer.alpha, "beta": transfer.beta, "appeal": transfer.appeal},
        "runoff_scale": (runoff_a, runoff_b, runoff_unpolled),
        "scenario_mass_dropped": dropped, "mode": mode, "decision": decision, "spec": spec,
        "decision_outcomes": decision_outcomes, **ens_meta,
    })
    if write:
        summary.write()
    return summary


def _ensemble_inputs(year, lineups, forecasts, hm, nl, blocs, registry, cfg, h, archive):
    """Polls + fundamentals ensemble per lineup, errors from the meta-model,
    all trained on the full out-of-sample backtest archive."""
    from forecast.backtest.harness import FUND_YEARS, fit_runoff_scale, fit_runoff_scales
    from forecast.ensemble.stacking import PrecisionModel, combine
    from forecast.errors.meta_model import fit_error_model
    from forecast.fundamentals.dlm import fit_fundamentals
    from forecast.fundamentals.features_pres import PresFeatures

    features = PresFeatures(registry, cfg.lean.recent_weight)
    fund = fit_fundamentals(features, [y for y in FUND_YEARS if y < year])
    precision = PrecisionModel.fit(archive, ("polls", "fundamentals"))
    errors = fit_error_model(archive, ("bloc",), nu_floor=cfg.errors.nu_floor, thin_every=2)
    runoff_path = paths.ARCHIVE / "pres_backtest_runoff.parquet"
    runoff_archive = pl.read_parquet(runoff_path) if runoff_path.exists() else pl.DataFrame()
    if cfg.errors.runoff_by_polled:
        runoff_scale = fit_runoff_scales(runoff_archive, cfg.errors)
    else:
        runoff_scale = (*fit_runoff_scale(runoff_archive, (cfg.errors.runoff_a, cfg.errors.runoff_b)), None)
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

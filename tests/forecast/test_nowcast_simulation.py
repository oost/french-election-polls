"""Lineup scenarios, the nested-logit now-cast, runoff transfers and the
presidential simulation (invariants the post's method relies on)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
import pytest
import yaml

from forecast.config import NowcastParams, PollParams
from forecast.polls.ewma import polling_average
from forecast.polls.horizon import HorizonModel
from forecast.polls.lineups import (
    LineupForecast, NestedLogit, RaceObs, fit_nested_logit, nested_log_shares, nowcast_lineup, race_observations,
)
from forecast.presidential.pipeline import ANY, PresSummary, _candidate_rows, _short_labels, summarise_by_slot
from forecast.simulate.presidential import SimInput, antithetic_t, candidate_sd, simulate_presidential
from forecast.simulate.runoff import MatchupObs, RunoffModel, TransferModel, fit_transfer_model
from forecast.simulate.scenarios import Lineup, ScenarioSpec, sample_lineups
from tests.forecast.synthetic import BLOCS, ELECTION, make_polls


def _spec(tmp_path, **over) -> ScenarioSpec:
    doc = {"election": {"id": "x", "t1": "2030-04-14", "t2": "2030-04-28"},
           "slots": {"s": {"A": 0.7, "B": 0.3}}, "fixed": ["C", "D"], "optional": {"E": 0.5}, "constraints": []}
    doc.update(over)
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump(doc))
    return ScenarioSpec.load(path)


class TestScenarios:
    def test_enumerate_probabilities(self, tmp_path):
        ls = _spec(tmp_path).enumerate()
        assert len(ls) == 4
        assert sum(l.probability for l in ls) == pytest.approx(1.0)
        p = {l.key: l.probability for l in ls}
        assert p["A+C+D+E"] == pytest.approx(0.35)

    def test_constraints_renormalise(self, tmp_path):
        ls = _spec(tmp_path, constraints=[{"if": {"s": "B"}, "exclude": ["E"]}]).enumerate()
        assert all(not ({"B", "E"} <= set(l.candidates)) for l in ls)
        assert sum(l.probability for l in ls) == pytest.approx(1.0)

    def test_require_any_drops_lineups_without_one(self, tmp_path):
        spec = _spec(tmp_path, fixed=["C"], optional={"D": 0.8, "E": 0.5},
                     constraints=[{"require_any": ["D", "E"]}])
        ls = spec.enumerate()
        assert len(ls) == 6                       # 2 slot options × (4 optional combinations − "neither")
        assert all({"D", "E"} & set(l.candidates) for l in ls)
        assert sum(l.probability for l in ls) == pytest.approx(1.0)
        p_both = sum(l.probability for l in ls if {"D", "E"} <= set(l.candidates))
        assert p_both == pytest.approx(0.4 / 0.9)  # 0.8·0.5 renormalised over 1 − 0.2·0.5

    def test_bad_slot_probabilities(self, tmp_path):
        with pytest.raises(ValueError):
            _spec(tmp_path, slots={"s": {"A": 0.7, "B": 0.2}})

    def test_sampler_frequencies(self, tmp_path):
        ls = _spec(tmp_path).enumerate()
        idx = sample_lineups(ls, 40_000, np.random.default_rng(0))
        for i, l in enumerate(ls):
            f = (idx == i).mean()
            assert abs(f - l.probability) < 3 * np.sqrt(l.probability * (1 - l.probability) / 40_000)

    def test_option_can_name_several_candidates(self, tmp_path):
        spec = _spec(tmp_path, slots={"s": {"A+B": 0.6, "A": 0.3, None: 0.1}}, optional={})
        ls = {l.key: l for l in spec.enumerate()}
        assert set(ls) == {"A+B+C+D", "A+C+D", "C+D"}
        assert ls["A+B+C+D"].probability == pytest.approx(0.6)
        assert ls["A+B+C+D"].slots == {"s": "A+B"}
        assert {"A", "B"} <= spec.candidates()

    def test_2027_file_loads(self):
        spec = ScenarioSpec.load()
        ls = spec.enumerate()
        assert sum(l.probability for l in ls) == pytest.approx(1.0)
        assert spec.slots["rn"] == {"LE_PEN_MARINE": 0.95, "BARDELLA_JORDAN": 0.05}
        # every lineup has one to four of the centre/right candidates, from the slot alone
        four = {"PHILIPPE_EDOUARD", "ATTAL_GABRIEL", "RETAILLEAU_BRUNO", "LISNARD_DAVID"}
        assert all(1 <= len(four & set(l.candidates)) <= 4 for l in ls)
        assert not four & (set(spec.fixed) | set(spec.optional))


class TestNestedLogit:
    def test_vectorised_matches_scalar(self):
        rng = np.random.default_rng(1)
        cands = list("ABCDEF")
        blocs = dict(zip(cands, "XXYYYZ"))
        m = NestedLogit(cands, blocs, rng.normal(size=6), {"X": 0.4, "Y": 0.8, "Z": 1.0})
        lu = ["A", "B", "D", "F"]
        bl = ["X", "Y", "Z"]
        got = nested_log_shares(m.u[[cands.index(c) for c in lu]][None], np.array([[m.mu[blocs[c]] for c in lu]]),
                                np.array([[bl.index(blocs[c]) for c in lu]]), np.ones((1, 4), bool), 3)[0]
        assert got == pytest.approx(np.log(m.shares(lu)))

    def test_low_mu_keeps_votes_in_bloc(self):
        cands = ["A1", "A2", "B"]
        blocs = {"A1": "A", "A2": "A", "B": "B"}
        u = np.array([0.0, 0.0, 0.0])
        strong = NestedLogit(cands, blocs, u, {"A": 0.2, "B": 1.0})
        weak = NestedLogit(cands, blocs, u, {"A": 1.0, "B": 1.0})
        # dropping A2: with strong nesting A1 inherits more of A2's voters
        assert strong.shares(["A1", "B"])[0] > weak.shares(["A1", "B"])[0] - 1e-12
        assert strong.shares(["A1", "A2", "B"])[2] > weak.shares(["A1", "A2", "B"])[2]

    def test_vectorised_matches_scalar_with_groups(self):
        rng = np.random.default_rng(2)
        cands = list("ABCDEFG")
        blocs = dict(zip(cands, "XXYYZZW"))
        m = NestedLogit(cands, blocs, rng.normal(size=7), {"X": 0.4, "Y": 0.7, "Z": 0.5, "W": 1.0},
                        groups={"X": "X+Y", "Y": "X+Y"}, lam={"X+Y": 0.8})
        bl = ["W", "X", "Y", "Z"]
        group, lam = np.array([0, 1, 1, 2]), np.array([1.0, 0.8, 1.0])
        for lu in (["A", "B", "C", "E", "G"], ["A", "E", "F"], ["C", "D", "G"]):
            got = nested_log_shares(m.u[[cands.index(c) for c in lu]][None],
                                    np.array([[m.mu[blocs[c]] for c in lu]]),
                                    np.array([[bl.index(blocs[c]) for c in lu]]), np.ones((1, len(lu)), bool), 4,
                                    group, lam)[0]
            assert got == pytest.approx(np.log(m.shares(lu)))

    def test_lambda_one_is_the_plain_nested_logit(self):
        rng = np.random.default_rng(3)
        cands = list("ABCDE")
        blocs = dict(zip(cands, "XXYYZ"))
        u, mu = rng.normal(size=5), {"X": 0.4, "Y": 0.6, "Z": 1.0}
        plain = NestedLogit(cands, blocs, u, mu)
        grouped = NestedLogit(cands, blocs, u, mu, groups={"X": "X+Y", "Y": "X+Y"}, lam={"X+Y": 1.0})
        for lu in (list("ABCDE"), list("ACE"), list("AB")):
            assert grouped.shares(lu) == pytest.approx(plain.shares(lu))

    def test_group_sends_an_emptied_bloc_to_its_sibling(self):
        # Y (one candidate, C) drops out entirely. Without grouping C's voters
        # split between A and E in proportion to size (half each here); with X
        # and Y grouped at a low λ, A, in the sibling bloc, takes most of them.
        cands = ["A", "C", "E"]
        blocs = {"A": "X", "C": "Y", "E": "Z"}
        u, mu = np.zeros(3), {"X": 0.5, "Y": 0.5, "Z": 1.0}
        plain = NestedLogit(cands, blocs, u, mu)
        grouped = NestedLogit(cands, blocs, u, mu, groups={"X": "X+Y", "Y": "X+Y"}, lam={"X+Y": 0.5})

        def to_sibling(m):
            both, lone = m.shares(["A", "C", "E"]), m.shares(["A", "E"])
            return (lone[0] - both[0]) / both[1]     # fraction of C's voters A inherits

        assert to_sibling(plain) == pytest.approx(0.5)
        assert to_sibling(grouped) > 0.65

    def test_fit_recovers_a_known_lambda(self):
        cands = ["A1", "A2", "B1", "B2", "C1", "C2"]
        blocs = {"A1": "X", "A2": "X", "B1": "Y", "B2": "Y", "C1": "Z", "C2": "Z"}
        truth = NestedLogit(cands, blocs, np.array([0.3, 0.0, 0.2, -0.1, 0.4, 0.1]),
                            {"X": 0.4, "Y": 0.45, "Z": 0.6}, groups={"X": "X+Y", "Y": "X+Y"}, lam={"X+Y": 0.55})
        obs = []
        import itertools
        for k in (3, 4, 5, 6):
            for lu in itertools.combinations(cands, k):
                obs.append(RaceObs("+".join(lu), lu, np.log(truth.shares(lu)), 1.0, 5, 0))
        p = NowcastParams(nest_groups=("X+Y",), nest_mu_prior_sd=10.0)
        fit = fit_nested_logit(obs, blocs, p)
        assert fit.groups == {"X": "X+Y", "Y": "X+Y"}
        assert fit.lam["X+Y"] == pytest.approx(0.55, abs=0.03)
        assert fit.shares(["A1", "C1", "C2"]) == pytest.approx(truth.shares(["A1", "C1", "C2"]), abs=0.005)

    def test_group_needs_two_blocs_in_the_polls(self):
        # A group whose second bloc nobody polled has no λ to fit.
        res = polling_average(make_polls(), ELECTION, PollParams(), eval_days=[-3])
        obs = race_observations(res, -3)
        model = fit_nested_logit(obs, BLOCS, NowcastParams(nest_groups=("CEN+NOBODY",)))
        assert model.lam == {} and model.groups == {}

    def test_recovers_synthetic_lineups(self):
        res = polling_average(make_polls(), ELECTION, PollParams(), eval_days=[-3])
        obs = race_observations(res, -3)
        model = fit_nested_logit(obs, BLOCS, NowcastParams())
        for o in obs:
            pred = np.log(model.shares(o.options))
            assert np.max(np.abs(pred - o.log_share)) < 0.08

    def test_anchoring_respects_slack(self):
        res = polling_average(make_polls(), ELECTION, PollParams(), eval_days=[-3])
        obs = race_observations(res, -3)
        model = fit_nested_logit(obs, BLOCS, NowcastParams())
        full = tuple("ABCDEF")
        fc = nowcast_lineup(full, model, res, -3, NowcastParams())
        assert fc.polled and fc.n_polls >= 5
        avg = res.race_average("+".join(full), -3)
        polled = dict(zip(avg["option_id"], avg["share"]))
        for c, s in fc.shares().items():
            assert abs(s - polled[c]) <= 1 / 3 + 0.5    # ⅓-pt slack, plus renormalisation


    def test_anchor_match_ignores_the_two_percent_cut_of_the_nowcast(self):
        # E polls near 8%, so the poll frame keys races on it. A now-cast that
        # puts E just under 2% must still find the lineup's own polled race.
        res = polling_average(make_polls(), ELECTION, PollParams(), eval_days=[-3])
        full = tuple("ABCDEF")
        u = np.log(np.array([30.0, 25.0, 20.0, 15.0, 1.9, 6.0]))
        model = NestedLogit(list(full), BLOCS, u, {b: 1.0 for b in set(BLOCS.values())})
        assert 100 * model.shares(full)[4] < 2.0
        fc = nowcast_lineup(full, model, res, -3, NowcastParams())
        assert fc.polled and fc.n_polls >= 5
        avg = res.race_average("+".join(full), -3)
        polled_e = avg.filter(pl.col("option_id") == "E")["share"][0]
        assert fc.shares()["E"] == pytest.approx(polled_e, abs=1 / 3 + 0.5)   # pulled up to its polls

    def test_anchor_rescales_for_candidates_the_polls_left_out(self):
        # G is in the lineup but in no poll: the race average (100% over A–F)
        # is put on the lineup's basis instead of squeezing G out.
        res = polling_average(make_polls(), ELECTION, PollParams(), eval_days=[-3])
        lineup = tuple("ABCDEFG")
        avg = res.race_average("+".join("ABCDEF"), -3)
        polled = dict(zip(avg["option_id"], avg["share"]))
        u = np.log(np.array([polled[c] for c in "ABCDEF"] + [4.0]))
        model = NestedLogit(list(lineup), BLOCS | {"G": "DIV"}, u, {b: 1.0 for b in [*set(BLOCS.values()), "DIV"]})
        g_pred = 100 * model.shares(lineup)[6]
        fc = nowcast_lineup(lineup, model, res, -3, NowcastParams())
        assert fc.polled
        shares = fc.shares()
        assert shares["G"] == pytest.approx(g_pred, abs=0.05)
        for c in "ABCDEF":
            assert shares[c] == pytest.approx(polled[c] * (100 - g_pred) / 100, abs=0.05)


class TestRunoff:
    def test_transfer_share_between_zero_and_one(self):
        m = TransferModel(1.0, 1.0, {}, {"A": "GAU", "B": "EXD", "C": "CEN"})
        s = m.share_a("A", "B", ["A", "B", "C"], np.array([0.3, 0.4, 0.3]))
        assert 0 < s < 1

    def test_centrist_voters_break_away_from_extremes(self):
        m = TransferModel(1.0, 1.0, {}, {"A": "GAU", "B": "EXD", "C": "CEN"})
        # equal first rounds, the centre sits closer to the left → A wins
        assert m.share_a("A", "B", ["A", "B", "C"], np.array([0.35, 0.35, 0.3])) > 0.5

    def test_fit_reproduces_polled_matchup(self):
        blocs = {"A": "GAU", "B": "EXD", "C": "CEN"}
        obs = [MatchupObs("A", "B", 0.45, 20, 0, 20.0)]
        fr = {("A", "B"): (["A", "B", "C"], np.array([0.3, 0.4, 0.3]))}
        m = fit_transfer_model(obs, fr, blocs, appeal_sd=2.0)
        assert m.share_a("A", "B", *fr[("A", "B")]) == pytest.approx(0.45, abs=0.02)

    def test_anchor_limits_departure_from_polls(self):
        m = TransferModel(3.0, 0.0, {}, {"A": "GAU", "B": "EXD"})
        polled = MatchupObs("A", "B", 0.40, 10, 0, 10.0)
        rm = RunoffModel(m, {("A", "B"): polled})
        exp = rm.expected("A", "B", ["A", "B"], np.array([0.5, 0.5]))
        assert exp == pytest.approx(0.40 + 1 / 300)       # 10 polls → ⅓-pt slack


TOY_BLOCS = {"A": "CEN", "B": "EXD", "C": "GAU", "D": "DRO"}


def _toy_sims(lineups: list[Lineup], turnout=None, **sim):
    """Four candidates (B leads, D trails) simulated over the given lineups."""
    fcs = {}
    for l in lineups:
        s = np.array([{"A": 30, "B": 35, "C": 20, "D": 15}[c] for c in l.candidates], float)
        s = s / s.sum()
        fcs[l.key] = LineupForecast(l.candidates, np.log(s), np.zeros(len(s)), True, 10, np.zeros(len(s)))
    tm = TransferModel(1.0, 1.0, {}, TOY_BLOCS)
    hm = HorizonModel(0.1, 0.02, 0.25, 0.0)
    inputs = {k: SimInput.from_forecast(fc, candidate_sd(fc, hm, 60, 0.1), TOY_BLOCS, {"GAU": 0.05})
              for k, fc in fcs.items()}
    return simulate_presidential(
        lineups, inputs, RunoffModel(tm), days_out_t2=74, nu=5, runoff_a=0.1, runoff_b=0.015,
        runoff_rho=0.5, n_sims=20_000, seed=3, turnout=turnout, **sim)


class TestSimulation:
    def test_antithetic_exact_zero_mean(self):
        z = antithetic_t(np.random.default_rng(0), (10_000, 4), nu=5)
        assert np.abs(z.mean(axis=0)).max() < 1e-12
        assert z.std() == pytest.approx(1.0, abs=0.05)

    @pytest.fixture(scope="class")
    def sims(self):
        return _toy_sims([Lineup(("A", "B", "C", "D"), 0.6, {}), Lineup(("A", "B", "C"), 0.4, {})])

    def test_everyone_either_wins_outright_or_meets_one_opponent(self, sims):
        assert (sims.winner >= 0).all()
        has_runoff = sims.finalists[:, 1] >= 0
        assert np.isin(sims.winner[has_runoff], sims.finalists[has_runoff]).all()

    def test_two_finalists_per_simulation(self, sims):
        p_runoff = np.array([(sims.finalists == i).any(axis=1).mean() for i in range(len(sims.candidates))])
        runoff_rate = (sims.finalists[:, 1] >= 0).mean()
        assert p_runoff.sum() == pytest.approx(1 + runoff_rate)
        assert np.array([(sims.winner == i).mean() for i in range(len(sims.candidates))]).sum() == pytest.approx(1)

    def test_lineup_frequencies(self, sims):
        assert (sims.lineup_idx == 0).mean() == pytest.approx(0.6, abs=0.015)
        d = sims.candidates.index("D")
        assert np.isnan(sims.shares[sims.lineup_idx == 1, d]).all()

    def test_shares_sum_to_100(self, sims):
        assert np.nansum(sims.shares, axis=1) == pytest.approx(100.0)

    def test_leader_usually_reaches_runoff(self, sims):
        b = sims.candidates.index("B")
        assert (sims.finalists == b).any(axis=1).mean() > 0.8


SLOTS = {"s": {"D": 0.6, None: 0.4}}     # slot "s" puts D on the ballot, or nobody


@pytest.fixture(scope="module")
def slot_summary():
    sims = _toy_sims([Lineup(("A", "B", "C", "D"), 0.6, {"s": "D"}), Lineup(("A", "B", "C"), 0.4, {"s": None})])
    names = {c: c for c in TOY_BLOCS}
    cand = pl.DataFrame(_candidate_rows(sims, names, TOY_BLOCS)).sort("p_win", "p_runoff", descending=True)
    cond = summarise_by_slot(sims, SLOTS, names, TOY_BLOCS)
    empty = pl.DataFrame(schema={"matchup": pl.String, "p": pl.Float64, "p_first_wins": pl.Float64})
    return PresSummary(2030, date(2030, 1, 1), 100, cand, empty, pl.DataFrame(schema={"bloc": pl.String}),
                       pl.DataFrame(), conditional=cond, meta={"n_sims": sims.n})


class TestConditional:
    def test_options_weighted_by_probability_give_the_headline(self, slot_summary):
        cond = slot_summary.conditional.filter(pl.col("candidate_id") != ANY)
        back = cond.group_by("candidate_id").agg(
            *[(pl.col(m) * pl.col("p_ballot")).sum() for m in ("p_run", "p_runoff", "p_win")])
        both = back.join(slot_summary.candidates, on="candidate_id", suffix="_all")
        for m in ("p_run", "p_runoff", "p_win"):
            assert both[m].to_numpy() == pytest.approx(both[f"{m}_all"].to_numpy(), abs=1e-12)
        heads = slot_summary.conditional.unique("ballot")
        assert heads["p_ballot"].sum() == pytest.approx(1)
        assert dict(zip(heads["ballot"], heads["p_ballot"]))["D"] == pytest.approx(0.6, abs=0.015)

    def test_combined_row(self, slot_summary):
        c = slot_summary.conditional
        def row(ballot, cid):
            return c.filter(pl.col("ballot") == ballot, pl.col("candidate_id") == cid).row(0, named=True)
        assert row("D", ANY)["p_win"] == row("D", "D")["p_win"]     # one member: the same as D
        assert row("D", ANY)["share_q50"] == pytest.approx(row("D", "D")["share_q50"])
        assert row("nobody", ANY)["p_run"] == 0 and row("nobody", ANY)["p_runoff"] == 0

    def test_matrix(self, slot_summary):
        m = slot_summary.conditional_matrix("s", "p_win")
        assert m.columns == ["name", "D", "nobody"]          # options in the YAML's order
        assert m["name"][-1] == "any S"
        assert m.filter(pl.col("name") == "D")["nobody"][0] is None   # not on that ballot
        assert set(m["name"][:-1]) == set(TOY_BLOCS)
        with pytest.raises(ValueError):
            slot_summary.conditional_matrix("no_such_slot")

    def test_report(self, slot_summary):
        text = slot_summary.report()
        assert "Given the s ballot" in text and "WARNING" not in text
        dropped = replace(slot_summary, meta={**slot_summary.meta, "scenario_mass_dropped": 0.7})
        assert "WARNING: 70% of the scenario probability" in dropped.report()

    def test_pdf_report(self, slot_summary, tmp_path):
        pytest.importorskip("reportlab")
        from forecast.report.pdf import write_pdf
        data = write_pdf(slot_summary, path=tmp_path / "report.pdf").read_bytes()
        assert data.startswith(b"%PDF") and len(data) > 20_000   # charts embedded

    def test_short_labels(self):
        assert _short_labels(["Le Pen", "Bardella"]) == {"Le Pen": "Le Pen", "Bardella": "Bardella"}
        assert _short_labels(["Philippe + Lisnard", "Attal", "nobody"]) == {
            "Philippe + Lisnard": "P+L", "Attal": "A", "nobody": "nobody"}
        assert _short_labels(["Philippe + Lisnard", "Le Pen + Attal"])["Le Pen + Attal"] == "LeP+Att"


class TestCentristDecision:
    from forecast.simulate.presidential import Decision
    rule = Decision("centrist", ("P", "A"), threshold=15, fallback_top=2, fallback_above=10, realised=0.4)

    def test_leader_above_threshold_runs_alone(self):
        out = self.rule.choose({"P": np.array([17.0, 9.0]), "A": np.array([12.0, 16.0])})
        assert out == [frozenset({"P"}), frozenset({"A"})]

    def test_nobody_above_threshold_top_two_run(self):
        out = self.rule.choose({"P": np.array([14.0]), "A": np.array([6.0])})
        assert out == [frozenset({"P", "A"})]

    def test_fallback_adds_anyone_above_ten(self):
        rule = type(self.rule)("c", ("P", "A", "D"), 15, 2, 10, 0.4)
        out = rule.choose({"P": np.array([14.0]), "A": np.array([12.0]), "D": np.array([11.0])})
        assert out == [frozenset({"P", "A", "D"})]

    def test_variants_from_a_spec(self, tmp_path):
        from forecast.simulate.scenarios import decision_variants
        spec = _spec(tmp_path, fixed=["C", "P", "Q"], optional={},
                     decisions={"c": {"date": "2030-01-15", "pool": ["P", "Q"], "threshold": 15,
                                      "fallback_top": 2, "fallback_above": 10}})
        lineup = spec.enumerate()[0]
        assert {"P", "Q"} <= set(lineup.candidates)              # pool members start in every lineup
        variants = decision_variants(lineup.candidates, spec.decisions)
        assert len(variants) == 3                                # P alone, Q alone, both

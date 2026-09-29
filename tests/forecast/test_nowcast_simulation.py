"""Lineup scenarios, the nested-logit now-cast, runoff transfers and the
presidential simulation (invariants the post's method relies on)."""
from __future__ import annotations

import numpy as np
import pytest
import yaml

from forecast.config import NowcastParams, PollParams
from forecast.polls.ewma import polling_average
from forecast.polls.horizon import HorizonModel
from forecast.polls.lineups import (
    LineupForecast, NestedLogit, RaceObs, fit_nested_logit, nested_log_shares, nowcast_lineup, race_observations,
)
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

    def test_bad_slot_probabilities(self, tmp_path):
        with pytest.raises(ValueError):
            _spec(tmp_path, slots={"s": {"A": 0.7, "B": 0.2}})

    def test_sampler_frequencies(self, tmp_path):
        ls = _spec(tmp_path).enumerate()
        idx = sample_lineups(ls, 40_000, np.random.default_rng(0))
        for i, l in enumerate(ls):
            f = (idx == i).mean()
            assert abs(f - l.probability) < 3 * np.sqrt(l.probability * (1 - l.probability) / 40_000)

    def test_2027_file_loads(self):
        ls = ScenarioSpec.load().enumerate()
        assert sum(l.probability for l in ls) == pytest.approx(1.0)


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


class TestSimulation:
    def test_antithetic_exact_zero_mean(self):
        z = antithetic_t(np.random.default_rng(0), (10_000, 4), nu=5)
        assert np.abs(z.mean(axis=0)).max() < 1e-12
        assert z.std() == pytest.approx(1.0, abs=0.05)

    @pytest.fixture(scope="class")
    def sims(self):
        blocs = {"A": "CEN", "B": "EXD", "C": "GAU", "D": "DRO"}
        lineups = [Lineup(("A", "B", "C", "D"), 0.6, {}), Lineup(("A", "B", "C"), 0.4, {})]
        fcs = {}
        for l in lineups:
            s = np.array([{"A": 30, "B": 35, "C": 20, "D": 15}[c] for c in l.candidates], float)
            s = s / s.sum()
            fcs[l.key] = LineupForecast(l.candidates, np.log(s), np.zeros(len(s)), True, 10, np.zeros(len(s)))
        tm = TransferModel(1.0, 1.0, {}, blocs)
        hm = HorizonModel(0.1, 0.02, 0.25, 0.0)
        inputs = {k: SimInput.from_forecast(fc, candidate_sd(fc, hm, 60, 0.1), blocs, {"GAU": 0.05})
                  for k, fc in fcs.items()}
        return simulate_presidential(
            lineups, inputs, RunoffModel(tm), days_out_t2=74, nu=5, runoff_a=0.1, runoff_b=0.015,
            runoff_rho=0.5, n_sims=20_000, seed=3)

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

    def test_variants_and_2027_spec(self):
        from forecast.simulate.scenarios import ScenarioSpec, decision_variants
        spec = ScenarioSpec.load()
        lineup = spec.enumerate()[0]
        variants = decision_variants(lineup.candidates, spec.decisions)
        assert len(variants) == 3
        assert {"PHILIPPE_EDOUARD", "ATTAL_GABRIEL"} <= set(lineup.candidates)
        assert spec.slots["rn"] == {"LE_PEN_MARINE": 0.95, "BARDELLA_JORDAN": 0.05}

"""Turnout: reference data, the abstention model, poll calibration, and its
place in the presidential simulation; the legislative turnout level."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from forecast.backtest.harness import fit_runoff_scales
from forecast.config import ErrorParams, TurnoutParams
from forecast.ingest.turnout import build_abstention_polls, build_transfer_polls
from forecast.legislative.pipeline import scale_turnout
from forecast.presidential.pipeline import CLEAR_SHARE, summarise_far_right, summarise_turnout
from forecast.simulate.runoff import far_right_runoff_bias
from forecast.simulate import turnout as T
from forecast.simulate.scenarios import Lineup
from tests.forecast.test_nowcast_simulation import TOY_BLOCS, _toy_sims

P = TurnoutParams()


@pytest.fixture(scope="module")
def model():
    return T.fit_abstention(2027, P)


@pytest.fixture(scope="module")
def abstention_polls():
    return build_abstention_polls()


class TestHistory:
    def test_official_rates(self):
        h = {(r["year"], r["round"]): r for r in T.turnout_history().iter_rows(named=True)}
        assert len(h) == 22
        assert 100 * h[(2022, 1)]["turnout"] == pytest.approx(73.69, abs=0.01)
        assert 100 * h[(2017, 2)]["turnout"] == pytest.approx(74.56, abs=0.01)
        assert 100 * h[(1969, 2)]["turnout"] == pytest.approx(68.85, abs=0.01)
        assert 100 * h[(2017, 2)]["blank"] == pytest.approx(11.52, abs=0.01)
        assert all(r["expressed"] < r["voters"] < r["registered"] for r in h.values())

    def test_past_runoffs(self):
        runoffs = T.past_runoffs(P)
        assert [r.year for r in runoffs] == [1965, 1969, 1974, 1981, 1988, 1995, 2002, 2007, 2012, 2017, 2022]
        assert all(r.v.sum() == pytest.approx(1) and 0 < r.share_a < 1 for r in runoffs)
        by_year = {r.year: r for r in runoffs}
        assert by_year[2002].ratio > 1 > by_year[2017].ratio       # mobilisation against Le Pen; record blank votes
        r = by_year[2017]
        assert r.pos[r.names.index("MELENCHON_JEAN_LUC")] == P.radical_left_position
        assert r.pos[r.names.index("HAMON_BENOIT")] == -1.0
        assert [x.year for x in T.past_runoffs(P, before=2002)][-1] == 1995


class TestAbstentionModel:
    pos = np.array([2.0, 0.0, -1.0, -1.5, 1.0])     # far right, centre, left, radical left, right
    v = np.array([0.30, 0.25, 0.15, 0.20, 0.10])

    def test_choice(self):
        pa, pb = T.choice(self.pos, 0, 1, 1.5, 1.0)
        assert pa[0] > 0.99 and pb[1] > 0.99                     # the finalists' own voters stay
        assert (pa + pb < 1).all()
        assert (pa + pb)[3] < (pa + pb)[2]                        # the further from both, the more stay home
        up_a, up_b = T.choice(self.pos, 0, 1, 1.5, 1.0, shift=0.5)
        assert ((up_a + up_b)[2:] > (pa + pb)[2:]).all()
        ga, _ = T.choice(self.pos, 0, 1, 1.5, 1.0, gap=1.0)
        assert (ga[2:] > pa[2:]).all()

    def test_gap_reproduces_a_share(self, model):
        for share in (0.35, 0.5, 0.8):
            gap = model.solve_gap(self.pos, 0, 1, self.v, share)
            assert float(model.share(self.pos, 0, 1, self.v, gap)) == pytest.approx(share, abs=1e-4)

    def test_no_shock_no_effect(self, model):
        z = np.zeros(5)
        d, ratio, ratio0 = model.effect(self.pos, 0, 1, self.v, 0.3, 0.2, z, np.zeros((5, 5)))
        assert d == pytest.approx(0) and ratio == pytest.approx(ratio0)

    def test_more_mobilisation_helps_the_finalist_the_abstainers_prefer(self, model):
        # far right v centre: the left electorates are the ones staying home, and they lean to the centre
        gap = model.solve_gap(self.pos, 0, 1, self.v, 0.5)
        tau = np.array([-1.0, 1.0])
        d, ratio, _ = model.effect(self.pos, 0, 1, self.v, 0.3, gap, tau, np.zeros((2, 5)))
        assert ratio[1] > ratio[0]
        assert d[1] < 0 < d[0]

    def test_fit_uses_only_earlier_runoffs(self, model):
        early = T.fit_abstention(2017, P)
        assert max(early.years) == 2012 and max(model.years) == 2022
        assert 0.5 < model.alpha < 4 and 0.3 < model.beta < 3 and 0.05 < model.mobilisation < 0.6
        assert model.sd_common >= P.common_sd_floor and model.share_var > 0
        shocks = model.shocks_by_year
        assert shocks[1969] < -0.5 and shocks[2017] < 0 < shocks[2002]   # boycott, blank votes, mobilisation

    def test_forward_prediction_beats_the_historical_mean(self):
        runoffs = T.past_runoffs(P)
        model_err, naive_err = [], []
        for r in runoffs:
            past = [q for q in runoffs if q.year < r.year]
            if r.year < 2002:
                continue
            m = T.fit_abstention(r.year, P)
            gap = m.solve_gap(r.pos, r.ia, r.ib, r.v, r.share_a)
            model_err.append(np.log(float(m.ratio(r.pos, r.ia, r.ib, r.v, r.nonvoters, gap)) / r.ratio))
            naive_err.append(np.log(np.mean([q.ratio for q in past]) / r.ratio))
        rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
        assert rms(model_err) < 0.8 * rms(naive_err)


class TestPolls:
    def test_ingested_tables(self, abstention_polls):
        a = abstention_polls
        first_2017 = a.filter((pl.col("year") == 2017) & (pl.col("round") == "first"))
        assert first_2017.height > 80 and first_2017["matchup"].null_count() == first_2017.height
        second_2022 = a.filter((pl.col("year") == 2022) & (pl.col("round") == "second"))
        assert "LE_PEN_MARINE|MACRON_EMMANUEL" in second_2022["matchup"].to_list()
        assert a["days_out"].min() >= 1 and a["abstention"].is_between(3, 70).all()
        t = build_transfer_polls()
        assert set(t["year"]) == {2012, 2017} and "NONE" in t["target"].to_list()
        mel = t.filter((pl.col("year") == 2017) & (pl.col("source") == "MELENCHON_JEAN_LUC"))
        one = mel.filter((pl.col("pollster") == mel["pollster"][0]) & (pl.col("fieldwork_end") == mel["fieldwork_end"][0]))
        assert one.height == 3 and one["share"].sum() == pytest.approx(100, abs=3)

    def test_calibration_measures_past_campaigns_only(self, abstention_polls):
        cal = T.poll_calibration(abstention_polls, 2027)
        bias, between, n = cal[("first", 2)]                 # 22–60 days out: pollsters overstate abstention
        assert n == 2 and bias > 0.15 and between > 0.15
        bias, between, n = cal[("second", 0)]                # last week before the runoff: close
        assert n == 3 and abs(bias) < 0.05
        assert ("first", 0) not in T.poll_calibration(abstention_polls, 2012)

    def test_combine(self):
        mu, sd = T.combine((0.0, 0.3), (1.0, 0.3))
        assert mu == pytest.approx(0.5) and sd == pytest.approx(0.3 / np.sqrt(2))
        assert T.combine((0.2, 0.3), None) == (0.2, 0.3)

    def test_estimate_uses_the_freshest_polls(self):
        day = date(2030, 4, 14)
        rows = [{"year": 2030, "round": "first", "pollster": f"P{i}", "fieldwork_start": day - timedelta(days=d + 1),
                 "fieldwork_end": day - timedelta(days=d), "publish_est": day - timedelta(days=d - 1), "days_out": d,
                 "matchup": None, "abstention": a, "source_page": "x", "source_table": 0}
                for i, (d, a) in enumerate([(100, 40.0), (12, 26.0), (10, 24.0), (3, 20.0)])]
        polls = pl.DataFrame(rows)
        assert T.poll_estimate(polls, day - timedelta(days=120), day, "first", None, {}, P) is None
        mu, sd, n = T.poll_estimate(polls, day - timedelta(days=8), day, "first", None, {}, P)
        assert n == 2 and 100 * T._sigmoid(mu) == pytest.approx(25, abs=0.2)     # the 100-day-old poll is left out
        mu2, sd2, n2 = T.poll_estimate(polls, day, day, "first", None, {("first", 0): (0.2, 0.05, 3)}, P)
        assert n2 == 2 and sd2 < sd                              # one week of polls; a calibrated horizon is trusted more
        assert mu2 < T._logit(np.mean([0.24, 0.20]))             # and corrected for its bias

    def test_first_round_prior(self):
        mu, sd = T.first_round_prior(2027, P)
        assert 0.72 < T._sigmoid(mu) < 0.80 and 0.2 < sd < 0.4
        assert T.first_round_prior(2002, P)[0] > mu              # turnout was higher before 2002


class TestForecastModel:
    def test_2027_rests_on_history_until_polls_exist(self):
        tm = T.build_turnout_model(2027, date(2026, 11, 24), {"A": "GAU", "B": "CEN"}, P, parties={"A": "LFI"})
        assert tm.n_polls_t1 == 0 and tm.runoff_polls == {} and tm.t1 == tm.t1_prior
        assert tm.position == {"A": P.radical_left_position, "B": 0.0}
        assert 0.02 < tm.blank1 < 0.03 and 0.15 < tm.blank_frac2 < 0.25
        assert tm.residual_sd(0.3) < 0.3 and tm.residual_sd(0.01) == pytest.approx(P.residual_floor * 0.01)

    def test_polls_move_a_past_forecast(self, abstention_polls):
        late = T.build_turnout_model(2017, date(2017, 4, 21), {}, P, abstention_polls)
        assert late.n_polls_t1 > 3 and late.t1[1] < late.t1_prior[1]
        runoff = T.build_turnout_model(2017, date(2017, 5, 5), {}, P, abstention_polls)
        assert "LE_PEN_MARINE|MACRON_EMMANUEL" in runoff.runoff_polls

    def test_backtest(self, abstention_polls):
        bt = T.backtest_turnout(P, polls=abstention_polls)
        history = bt.filter(pl.col("basis") == "history only")
        assert history["inside"].mean() >= 0.8
        late = bt.filter((pl.col("round") == 2) & (pl.col("days") == 2))
        mae = {b: late.filter(pl.col("basis") == b)["error"].abs().mean() for b in ("history only", "with polls")}
        assert mae["with polls"] < mae["history only"] < 5


@pytest.fixture(scope="module")
def both(model):
    """The toy election simulated without and with a turnout model."""
    lineups = [Lineup(("A", "B", "C", "D"), 0.6, {}), Lineup(("A", "B", "C"), 0.4, {})]
    tm = T.TurnoutModel(model, (1.2, 0.28), (1.2, 0.28), 0, 0.022, 0.19,
                        {"A": 0.0, "B": 2.0, "C": -1.5, "D": 1.0}, {})
    return _toy_sims(lineups), _toy_sims(lineups, turnout=tm)


class TestSimulation:
    def test_first_round_is_untouched(self, both):
        plain, with_turnout = both
        assert plain.turnout_t1 is None
        assert np.array_equal(plain.shares, with_turnout.shares, equal_nan=True)
        assert np.array_equal(plain.finalists, with_turnout.finalists)
        assert (plain.winner == with_turnout.winner).mean() > 0.9

    def test_turnout_draws(self, both):
        s = both[1]
        assert ((s.turnout_t1 > 0.4) & (s.turnout_t1 < 0.97)).all()
        runoff = s.finalists[:, 1] >= 0
        assert np.isnan(s.turnout_t2[~runoff]).all() and not np.isnan(s.turnout_t2[runoff]).any()
        assert (s.valid_t2[runoff] < s.turnout_t2[runoff]).all()
        assert abs(np.nanmean(s.turnout_surprise)) < 1.0 and 2 < np.nanstd(s.turnout_surprise) < 8

    def test_higher_turnout_helps_the_centre_against_the_far_right(self, both):
        s = both[1]
        a, b = s.candidates.index("A"), s.candidates.index("B")
        k = np.where((np.sort(s.finalists, axis=1) == [a, b]).all(axis=1))[0]
        won = s.winner[k] == a
        lo, hi = np.quantile(s.turnout_surprise[k], [1 / 3, 2 / 3])
        assert won[s.turnout_surprise[k] >= hi].mean() > won[s.turnout_surprise[k] <= lo].mean() + 0.03

    def test_summary_tables(self, both):
        names = {c: c for c in TOY_BLOCS}
        assert all(t.is_empty() for t in summarise_turnout(both[0], names, 2027))
        overall, matchups = summarise_turnout(both[1], names, 2027)
        assert overall["measure"].to_list() == ["First round turnout", "Runoff turnout", "Runoff valid votes"]
        assert (overall["q05"] < overall["q50"]).all() and overall["last"][0] == pytest.approx(73.69, abs=0.01)
        top = matchups.row(0, named=True)
        assert top["matchup"] == "A vs B" and top["surprise_low"] < 0 < top["surprise_high"]
        assert top["share_per_3pts"] > 0 and top["clear_effect"]
        assert 0 < top["p_first_wins_low_se"] < 0.03 and 0 < top["share_per_3pts_se"] < top["share_per_3pts"] / 3

    def test_unclear_rows_are_flagged(self, both):
        s = both[1]
        _, matchups = summarise_turnout(s, {c: c for c in TOY_BLOCS}, 2027, min_p=0.0)
        assert matchups["clear_effect"].dtype == pl.Boolean
        weak = matchups.filter(pl.col("share_per_3pts").abs() < CLEAR_SHARE)
        assert not weak["clear_effect"].any()


class TestLegislativeTurnout:
    def test_rescaling_hits_the_national_level(self):
        ratio = {"a": 0.70, "b": 0.60, "c": 0.45}
        registered = {"a": 100.0, "b": 300.0, "c": 200.0}
        out = scale_turnout(ratio, registered, 0.50)
        national = sum(out[s] * registered[s] for s in out) / sum(registered.values())
        assert national == pytest.approx(0.50, abs=1e-6)
        assert out["a"] > out["b"] > out["c"] and all(0 < v < 1 for v in out.values())
        same = scale_turnout(ratio, registered, sum(ratio[s] * registered[s] for s in ratio) / 600)
        assert same == pytest.approx(ratio)


class TestRunoffErrorScale:
    """errors.runoff_by_polled and errors.runoff_net_of_first_round."""

    def _archive(self, rows):
        return pl.DataFrame([{"cycle": c, "days_to_t2": h, "err_logit": e, "polled_matchup": polled}
                             for c, polled, e in rows for h in (20, 80, 160)])

    def test_priors_without_history(self):
        e = ErrorParams()
        assert fit_runoff_scales(pl.DataFrame(), e) == (e.runoff_a, e.runoff_b, e.runoff_unpolled)

    def test_polled_and_unpolled_runoffs_are_scaled_separately(self):
        e = ErrorParams()
        # one unpolled runoff with a large miss, three polled ones with small misses
        a, b, u = fit_runoff_scales(self._archive([(2002, False, 0.6), (2007, True, 0.05), (2012, True, -0.05),
                                                   (2017, True, 0.05)]), e)
        assert a < e.runoff_a and b < e.runoff_b            # polled runoffs did better than the prior curve
        assert a / e.runoff_a == pytest.approx(b / e.runoff_b)   # the curve keeps its shape
        assert e.runoff_unpolled < u < 0.6                  # pulled toward the one unpolled miss
        big = fit_runoff_scales(self._archive([(2007, True, 0.6)]), e)
        assert big[0] > e.runoff_a and big[2] == e.runoff_unpolled
        # 23 snapshots of one election weigh as one election, not 23
        many = pl.DataFrame([{"cycle": 2007, "days_to_t2": 20 + k, "err_logit": 0.6, "polled_matchup": True} for k in range(23)])
        assert fit_runoff_scales(many, e)[0] == pytest.approx(big[0], rel=0.15)

    def test_narrower_runoff_error_makes_the_favourite_safer(self):
        lineups = [Lineup(("A", "B", "C", "D"), 1.0, {})]
        plain = _toy_sims(lineups)
        net = _toy_sims(lineups, net_of_first_round=True)
        wide = _toy_sims(lineups, runoff_unpolled=1.0)          # no matchup is polled in the toy election
        assert np.array_equal(plain.shares, net.shares, equal_nan=True)
        a, b = plain.candidates.index("A"), plain.candidates.index("B")

        def favourite(s):
            k = (np.sort(s.finalists, axis=1) == [a, b]).all(axis=1)
            share_b = np.where(s.winner[k] == b, s.runoff_share[k], 100 - s.runoff_share[k])
            return (s.winner[k] == b).mean(), share_b.std()
        (p0, sd0), (p1, sd1), (p2, sd2) = favourite(plain), favourite(net), favourite(wide)
        assert sd1 < sd0 < sd2                               # error net of the first round is smaller
        assert abs(p2 - 0.5) < abs(p0 - 0.5) <= abs(p1 - 0.5) + 0.01



class TestFarRightRunoffScenario:
    def test_estimate_uses_only_earlier_runoffs(self):
        from forecast.ingest.build import load_polls_long
        from forecast.ingest.registry import CandidateRegistry
        long, reg, e = load_polls_long(), CandidateRegistry.load(), ErrorParams()
        args = (e.far_right_runoff_noise, e.far_right_runoff_prior_sd, e.far_right_runoff_days)
        assert far_right_runoff_bias(long, reg, 2002, *args) is None
        mean17, sd17, cases17 = far_right_runoff_bias(long, reg, 2017, *args)
        mean27, sd27, cases27 = far_right_runoff_bias(long, reg, 2027, *args)
        assert list(cases17) == [2002] and list(cases27) == [2002, 2017, 2022]
        assert all(v < -0.1 for v in cases27.values())            # the far right fell short each time
        assert min(cases27.values()) < mean27 < mean17 < 0        # pooled toward zero, less so with more cases
        assert sd27 < sd17 < e.far_right_runoff_prior_sd

    def test_scenario_in_the_simulation(self):
        lineups = [Lineup(("A", "B", "C", "D"), 1.0, {})]          # B is the far-right candidate and the favourite
        plain = _toy_sims(lineups)
        sims = _toy_sims(lineups, far_right=(0.5, -0.3, 0.05))
        assert plain.far_right_shortfall is None
        assert sims.far_right_shortfall.mean() == pytest.approx(0.5, abs=0.02)
        assert np.array_equal(plain.finalists, sims.finalists)      # the first round is untouched
        b = sims.candidates.index("B")
        yes = sims.far_right_shortfall
        assert (sims.winner[yes] == b).mean() < (sims.winner[~yes] == b).mean() - 0.05
        assert (sims.winner[~yes] == b).mean() == pytest.approx((plain.winner[~yes] == b).mean(), abs=1e-12)
        table = summarise_far_right(sims, {c: c for c in TOY_BLOCS})
        row = table.filter(pl.col("candidate_id") == "B").row(0, named=True)
        assert row["p_win_if_shortfall"] < row["p_win"] < row["p_win_if_not"]
        assert summarise_far_right(plain, {}).is_empty()

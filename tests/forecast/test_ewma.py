"""The three-pass polling average on a synthetic election with known truth."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from forecast.config import PollParams
from forecast.polls.ewma import build_frame, polling_average
from forecast.polls.trend import _fill
from tests.forecast.synthetic import ELECTION, make_polls, true_shares

FULL = ("A", "B", "C", "D", "E", "F")


@pytest.fixture(scope="module")
def result():
    return polling_average(make_polls(), ELECTION, PollParams(), eval_days=[-60, -3])


def test_races_keyed_on_major_candidates():
    frame = build_frame(make_polls(), ELECTION)
    assert len(frame.race_ids) == 2
    assert set(frame.race_options[0]) | set(frame.race_options[1]) == set(FULL)


def test_final_average_recovers_truth(result):
    race = "+".join(FULL)
    avg = result.race_average(race, -3)
    truth = true_shares(-3, FULL)
    for oid, share in avg.select("option_id", "share").iter_rows():
        assert share == pytest.approx(truth[oid], abs=0.8), oid


def test_trend_adjusts_old_polls(result):
    # A rises steadily; a lagging average would sit well below the truth at day −3.
    avg = result.race_average("+".join(FULL), -3)
    a = avg.filter(pl.col("option_id") == "A")["share"][0]
    assert a > true_shares(-30, FULL)["A"]


def test_house_effects_recovered_and_centred(result):
    ht = result.house_table.filter(pl.col("option_id") == "B")
    lean = dict(zip(ht["firm"], ht["lean"]))
    assert lean["P1"] > 0.03 and lean["P2"] < -0.03
    n = dict(zip(ht["firm"], ht["n"]))
    centred = sum(np.sqrt(n[f]) * lean[f] for f in lean) / sum(np.sqrt(n[f]) for f in lean)
    assert centred == pytest.approx(0.0, abs=1e-9)


def test_as_of_filters_unpublished_polls():
    polls = make_polls()
    as_of = ELECTION.replace(month=ELECTION.month - 2)
    res = polling_average(polls, ELECTION, PollParams(), as_of=as_of, eval_days=[(as_of - ELECTION).days])
    assert res.frame.rows["midpoint"].max() <= as_of


def test_uncertainty_is_positive(result):
    assert (result.averages["sd_log"] > 0).all()


def test_fill():
    out = _fill(np.array([np.nan, 1.0, np.nan, 3.0, np.nan]))
    assert out.tolist() == [1.0, 1.0, 1.0, 3.0, 3.0]

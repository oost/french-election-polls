"""Tests for polls_helpers — parsing and data manipulation against mock data,
plus regression checks against the pinned Wikipedia snapshots in wiki_cache/."""
from __future__ import annotations

import itertools
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest

import matplotlib

matplotlib.use("Agg")  # headless rendering for chart tests

import polls_helpers
from polls_helpers import (
    CACHE_DIR,
    CANDIDATE_COLORS,
    ELECTION_DATES,
    FINAL_POLL_DAY,
    FIRST_ROUND_RESULTS,
    SECOND_ROUND_DATES,
    SECOND_ROUND_RESULTS,
    WIKIPEDIA_REVISIONS,
    _extract_numeric,
    _find_candidate_columns,
    _parse_sample_size,
    classify_section,
    clean_pollster,
    collapse_scenarios,
    compute_poll_weights,
    compute_polling_changes,
    convert_dates_to_relative_days,
    days_until_next_election,
    create_election_chart,
    create_relative_day_charts,
    export_to_csv,
    extract_poll_records,
    filter_by_time_cutoff,
    get_candidate_color,
    get_first_round_candidates,
    get_second_round_matchups,
    get_smoothed_value_at_relative_day,
    is_valid_first_round_row,
    is_valid_second_round_row,
    parse_date,
    parse_fieldwork,
    parse_polling_tables,
    parse_time_cutoff,
    process_year_data,
    select_chart_candidates,
    smooth_polls,
)


# --- _extract_numeric ------------------------------------------------------

class TestExtractNumeric:
    def test_none_returns_none(self):
        assert _extract_numeric(None) is None

    def test_nan_returns_none(self):
        assert _extract_numeric(float("nan")) is None
        assert _extract_numeric(np.nan) is None

    def test_zero_returns_none(self):
        assert _extract_numeric(0) is None
        assert _extract_numeric(0.0) is None

    def test_negative_returns_none(self):
        assert _extract_numeric(-5) is None

    def test_positive_int(self):
        assert _extract_numeric(15) == 15.0

    def test_positive_float(self):
        assert _extract_numeric(23.5) == 23.5

    def test_string_with_percent(self):
        assert _extract_numeric("23.5%") == 23.5

    def test_string_with_leading_text(self):
        assert _extract_numeric("about 12.3 percent") == 12.3

    def test_string_no_digits(self):
        assert _extract_numeric("—") is None
        assert _extract_numeric("N/A") is None

    def test_empty_string(self):
        assert _extract_numeric("") is None

    def test_numpy_float(self):
        assert _extract_numeric(np.float64(18.7)) == 18.7


class TestParseSampleSize:
    def test_thousands_separator(self):
        assert _parse_sample_size("1,351") == 1351

    def test_dot_thousands_separator(self):
        # continental notation, as in the Ifop 25–29 Sep 2026 rows ("1.393")
        assert _parse_sample_size("1.393") == 1393
        assert _parse_sample_size("12.005") == 12005
        # a dot not followed by exactly three digits is not a separator
        assert _parse_sample_size("1.5") == 1

    def test_plain_and_footnote(self):
        assert _parse_sample_size("957") == 957
        assert _parse_sample_size("12,542[a]") == 12542

    def test_missing(self):
        assert _parse_sample_size("–") is None
        assert _parse_sample_size(float("nan")) is None


# --- Dates -----------------------------------------------------------------

class TestParseDate:
    def test_none_returns_none(self):
        assert parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert parse_date("") is None
        assert parse_date("   ") is None

    def test_iso_format(self):
        assert parse_date("2022-04-08") == date(2022, 4, 8)

    def test_full_month_name(self):
        assert parse_date("8 April 2022") == date(2022, 4, 8)

    def test_abbrev_month_name(self):
        assert parse_date("8 Apr 2022") == date(2022, 4, 8)

    def test_date_range_takes_last(self):
        assert parse_date("15–18 April 2022") == date(2022, 4, 18)

    def test_date_range_with_hyphen(self):
        assert parse_date("15-18 April 2022") == date(2022, 4, 18)

    def test_datetime_passthrough(self):
        assert parse_date(datetime(2022, 4, 8, 12, 0)) == date(2022, 4, 8)

    def test_date_passthrough(self):
        d = date(2022, 4, 8)
        assert parse_date(d) == d

    def test_nan_returns_none(self):
        assert parse_date(float("nan")) is None

    def test_garbage_returns_none(self):
        assert parse_date("not a date") is None


class TestParseFieldwork:
    def test_single_day(self):
        assert parse_fieldwork("8 April 2022") == (date(2022, 4, 8), date(2022, 4, 8))

    def test_same_month_range(self):
        assert parse_fieldwork("27–30 Mar 2012") == (date(2012, 3, 27), date(2012, 3, 30))

    def test_cross_month_range(self):
        assert parse_fieldwork("28 Feb–3 Mar 2022") == (date(2022, 2, 28), date(2022, 3, 3))

    def test_cross_year_range(self):
        assert parse_fieldwork("30 Dec 2021–3 Jan 2022") == (date(2021, 12, 30), date(2022, 1, 3))

    def test_cross_year_without_start_year(self):
        assert parse_fieldwork("30 Dec–3 Jan 2022") == (date(2021, 12, 30), date(2022, 1, 3))

    def test_footnote_and_iso(self):
        assert parse_fieldwork("5–7 Apr 2022[a]") == (date(2022, 4, 5), date(2022, 4, 7))
        assert parse_fieldwork("2022-04-08") == (date(2022, 4, 8), date(2022, 4, 8))

    def test_unparseable(self):
        assert parse_fieldwork("nan") is None
        assert parse_fieldwork(float("nan")) is None


# --- parse_time_cutoff / filter_by_time_cutoff -----------------------------

class TestParseTimeCutoff:
    def test_none(self):
        assert parse_time_cutoff(None) is None

    def test_years(self):
        assert parse_time_cutoff("1y").days == 365
        assert parse_time_cutoff("2y").days == 730

    def test_months(self):
        assert parse_time_cutoff("6m").days == 180
        assert parse_time_cutoff("3m").days == 90

    def test_days(self):
        assert parse_time_cutoff("30d").days == 30

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_time_cutoff("nonsense")
        with pytest.raises(ValueError):
            parse_time_cutoff("5w")


class TestFilterByTimeCutoff:
    def _df(self):
        return pl.DataFrame({
            "date": [date(2021, 1, 1), date(2021, 6, 1), date(2022, 1, 1), date(2022, 4, 1)],
            "Macron": [10.0, 12.0, 15.0, 20.0],
        })

    def test_one_year(self):
        result = filter_by_time_cutoff(self._df(), "1y", date(2022, 4, 10))
        assert result.height == 3
        assert result["date"].min() == date(2021, 6, 1)

    def test_six_months(self):
        assert filter_by_time_cutoff(self._df(), "6m", date(2022, 4, 10)).height == 2

    def test_none_cutoff_keeps_all(self):
        df = self._df()
        assert filter_by_time_cutoff(df, None, date(2022, 4, 10)).height == df.height


# --- Row validation --------------------------------------------------------

class TestIsValidFirstRoundRow:
    def test_normal_first_round(self):
        assert is_valid_first_round_row([28.0, 23.0, 17.0, 9.0]) is True

    def test_tiny_minor_candidate_values_allowed(self):
        # A 0.3% minor candidate must not reject the whole poll.
        assert is_valid_first_round_row([28.0, 23.0, 17.0, 0.3]) is True

    def test_below_three_candidates_rejected(self):
        assert is_valid_first_round_row([28.0, 23.0]) is False

    def test_value_above_50_rejected(self):
        assert is_valid_first_round_row([55.0, 23.0, 12.0]) is False

    def test_sum_above_100_rejected(self):
        # Head-to-head shares mistaken for a first-round row: 44 + 46 + 44 + 23.
        assert is_valid_first_round_row([44.0, 23.0, 46.0, 44.0]) is False

    def test_empty_rejected(self):
        assert is_valid_first_round_row([]) is False


class TestIsValidSecondRoundRow:
    def test_normal_matchup(self):
        assert is_valid_second_round_row([55.0, 45.0]) is True

    def test_landslide_allowed(self):
        # 2002: Chirac 82 / Le Pen 18.
        assert is_valid_second_round_row([82.0, 18.0]) is True

    def test_first_round_pair_rejected(self):
        # Macron 27 / Le Pen 23 from a first-round poll sums to 50, not ~100.
        assert is_valid_second_round_row([27.0, 23.0]) is False

    def test_three_values_rejected(self):
        assert is_valid_second_round_row([55.0, 45.0, 30.0]) is False


# --- Candidate lookups -----------------------------------------------------

class TestCandidateLookups:
    def test_first_round_is_full_ballot(self):
        for year, results in FIRST_ROUND_RESULTS.items():
            assert get_first_round_candidates(year) == list(results)

    def test_2002_includes_previously_missing_candidates(self):
        cands = get_first_round_candidates(2002)
        for name in ("Saint-Josse", "Mégret", "Taubira", "Lepage", "Boutin"):
            assert name in cands

    def test_first_round_unknown_year(self):
        assert get_first_round_candidates(1999) == []

    def test_second_round_known_year(self):
        assert get_second_round_matchups(2017) == [("Macron", "Le Pen")]

    def test_second_round_unknown_year(self):
        assert get_second_round_matchups(1999) == []


class TestDaysUntilNextElection:
    def test_fixed_today(self):
        assert days_until_next_election(date(2026, 9, 28)) == 202
        assert days_until_next_election(date(2027, 4, 18)) == 0

    def test_defaults_to_current_date(self):
        assert days_until_next_election() == days_until_next_election(date.today())


class TestResults:
    def test_all_election_years_present(self):
        assert set(FIRST_ROUND_RESULTS) == {2002, 2007, 2012, 2017, 2022}
        assert set(SECOND_ROUND_RESULTS) == set(FIRST_ROUND_RESULTS)

    def test_winners_match_known_history(self):
        assert FIRST_ROUND_RESULTS[2022]["Macron"] == 27.85
        assert FIRST_ROUND_RESULTS[2017]["Macron"] == 24.01
        assert FIRST_ROUND_RESULTS[2012]["Hollande"] == 28.63
        assert FIRST_ROUND_RESULTS[2007]["Sarkozy"] == 31.18
        assert FIRST_ROUND_RESULTS[2002]["Chirac"] == 19.88

    def test_results_are_realistic_percentages(self):
        for year, results in FIRST_ROUND_RESULTS.items():
            for candidate, pct in results.items():
                assert 0 < pct <= 50, f"{year} {candidate}={pct} out of range"

    def test_results_sum_near_100(self):
        for year, results in FIRST_ROUND_RESULTS.items():
            assert 99 <= sum(results.values()) <= 101, year
        for year, results in SECOND_ROUND_RESULTS.items():
            assert abs(sum(results.values()) - 100) < 0.05, year

    def test_second_round_finalists_ran_in_first_round(self):
        for year, pairs in polls_helpers.SECOND_ROUND_MATCHUPS.items():
            for c in pairs[0]:
                assert c in FIRST_ROUND_RESULTS[year]
                assert c in SECOND_ROUND_RESULTS[year]
            assert SECOND_ROUND_DATES[year] > ELECTION_DATES[year]


class TestCandidateColor:
    def test_unknown_returns_grey(self):
        assert get_candidate_color("Mystery Candidate") == "#999999"

    def test_every_candidate_has_a_color(self):
        for results in FIRST_ROUND_RESULTS.values():
            for c in results:
                assert c in CANDIDATE_COLORS, c

    def test_no_shared_colors_within_an_election(self):
        for year, results in FIRST_ROUND_RESULTS.items():
            charted = [c for c, v in results.items() if v >= 0.7]
            colors = [get_candidate_color(c) for c in charted]
            assert len(set(colors)) == len(colors), year


# --- Column matching / cleaning --------------------------------------------

class TestFindCandidateColumns:
    def test_wrapped_hyphenated_name(self):
        # Wikipedia renders "Dupont-Aignan" across two lines → "Dupont- Aignan".
        cols = [("Unnamed: 12_level_0", "Dupont- Aignan DLF", "Unnamed: 12_level_2")]
        assert _find_candidate_columns(cols, ["Dupont-Aignan"]) == {"Dupont-Aignan": cols[0]}

    def test_full_name_header(self):
        assert _find_candidate_columns(["Marine Le Pen"], ["Le Pen"]) == {"Le Pen": "Marine Le Pen"}

    def test_whole_word_only(self):
        assert _find_candidate_columns(["Lepage Cap21"], ["Le Pen"]) == {}
        assert _find_candidate_columns(["Maréchal-Le Pen"], ["Le Pen"]) == {}
        assert _find_candidate_columns(["Hueber"], ["Hue"]) == {}


class TestCleanPollster:
    def test_strips_archive_note_and_footnotes(self):
        assert clean_pollster("Ipsos Archived 1 March 2018 at the Wayback Machine") == "Ipsos"
        assert clean_pollster("Ifop-Fiducial[12]") == "Ifop-Fiducial"


class TestClassifySection:
    def test_national_rounds(self):
        assert classify_section(["First round", "2022", "Official campaign"]) == "first"
        assert classify_section(["Second round", "Macron vs. Le Pen"]) == "second"

    def test_subnational_and_crosstabs_excluded(self):
        assert classify_section(["First round", "By region"]) is None
        assert classify_section(["First round", "By department"]) is None
        assert classify_section(["First round", "By commune"]) is None
        assert classify_section(["Second round", "Hollande–Sarkozy", "By first round vote"]) is None

    def test_unrelated_section(self):
        assert classify_section(["Graphical summary"]) is None


# --- extract_poll_records --------------------------------------------------

def _first_round_table() -> pd.DataFrame:
    return pd.DataFrame({
        "Polling firm": ["Ifop", "Ipsos", "BVA", "2017 election"],
        "Fieldwork date": ["1–3 April 2022", "5 April 2022", "7 April 2022", "23 Apr 2017"],
        "Sample size": ["1,351", "1,000", "900", "–"],
        "Macron": ["28%", "27%", "26%", "24.01%"],
        "Le Pen": ["23%", "24%", "23.5%", "21.30%"],
        "Mélenchon": ["17%", "18%", "16%", "19.58%"],
        "Pécresse": ["9%", "8%", "8.5%", "–"],
    })


_EVENT_TEXT = ("Faure, Glucksmann, Guedj, Maurel and Royal are the 5 candidates "
               "in the left-wing primary")


def _table_with_event_row() -> pd.DataFrame:
    """A 2027-style table: a real poll, then a timeline row (NaN pollster,
    one sentence repeated across every other cell) like the 16 Sep 2026 row
    of revision 1377088060."""
    candidates = ["Arthaud", "Mélenchon", "Roussel", "Glucksmann", "Philippe", "Retailleau", "Le Pen"]
    real = ["1%", "17%", "2%", "10%", "13%", "7%", "35%"]
    return pd.DataFrame({
        "Polling firm": ["Harris Interactive", np.nan],
        "Fieldwork date": ["22–24 Sep 2026", "16 Sep 2026"],
        "Sample size": ["1974", _EVENT_TEXT],
        **{c: [v, _EVENT_TEXT] for c, v in zip(candidates, real)},
    })


class TestIsEventRow:
    def test_real_2027_event_row(self):
        assert polls_helpers.is_event_row(np.nan, [_EVENT_TEXT] * 20)

    def test_short_event_text_with_digits(self):
        assert polls_helpers.is_event_row(None, ["Zemmour announces his candidacy on 17 Sep"] * 12)

    def test_real_poll_is_not_event(self):
        assert not polls_helpers.is_event_row("Ifop", ["1974", "1%", "17%", "–", "–", "35%"])

    def test_row_of_dashes_is_not_event(self):
        # A sparse lineup row: pollster missing but the repeated cells are placeholders.
        assert not polls_helpers.is_event_row(np.nan, ["1000", "20%", "–", "–", "–", "–", "–", "–", "–"])

    def test_event_row_no_longer_becomes_phantom_poll(self):
        # Before the fix, the prose row parsed as a poll with every candidate
        # at 5.0% (the "5" in "the 5 candidates"), summing to 35 < 105.
        table = _table_with_event_row()
        names = ["Arthaud", "Mélenchon", "Roussel", "Glucksmann", "Philippe", "Retailleau", "Le Pen"]
        records = extract_poll_records(table, names, "first")
        assert len(records) == 1
        assert records[0]["pollster"] == "Harris Interactive"


class TestExtractPollRecords:
    def test_first_round_table(self):
        records = extract_poll_records(_first_round_table(), ["Macron", "Le Pen", "Mélenchon", "Pécresse"], "first")
        assert len(records) == 3  # the "2017 election" result row is skipped
        r = records[0]
        assert (r["Macron"], r["Le Pen"]) == (28.0, 23.0)
        assert (r["fieldwork_start"], r["fieldwork_end"]) == (date(2022, 4, 1), date(2022, 4, 3))
        assert r["date"] == date(2022, 4, 2)  # fieldwork midpoint
        assert r["pollster"] == "Ifop"
        assert r["sample_size"] == 1351

    def test_first_round_needs_three_candidate_columns(self):
        df = _first_round_table()[["Polling firm", "Fieldwork date", "Macron", "Le Pen"]]
        assert extract_poll_records(df, ["Macron", "Le Pen"], "first") == []

    def test_second_round_table(self):
        df = pd.DataFrame({
            "Polling firm": ["Ifop", "Ipsos"],
            "Fieldwork date": ["28 April 2017", "1 May 2017"],
            "Macron": ["58%", "60%"],
            "Le Pen": ["42%", "40%"],
        })
        records = extract_poll_records(df, ["Macron", "Le Pen"], "second")
        assert [(r["Macron"], r["Le Pen"]) for r in records] == [(58.0, 42.0), (60.0, 40.0)]

    def test_second_round_ignores_first_round_rows(self):
        # First-round table in second-round mode: the finalists' pair sums to ~50.
        df = _first_round_table()[["Polling firm", "Fieldwork date", "Macron", "Le Pen"]]
        assert extract_poll_records(df, ["Macron", "Le Pen"], "second") == []

    def test_table_without_polling_firm_skipped(self):
        bad = pd.DataFrame({"Date": ["1 April 2022"], "Macron": ["28"], "Le Pen": ["23"], "Jadot": ["5"]})
        assert extract_poll_records(bad, ["Macron", "Le Pen", "Jadot"], "first") == []

    def test_empty_table(self):
        assert extract_poll_records(pd.DataFrame(), ["Macron"], "first") == []


# --- Scenario collapsing and weights ----------------------------------------

def _rec(pollster, start, end, n, **vals):
    return {"date": start + (end - start) / 2, "fieldwork_start": start, "fieldwork_end": end,
            "pollster": pollster, "sample_size": n, "_row": 0, **vals}


class TestCollapseScenarios:
    CANDS = ["Macron", "Le Pen", "Pécresse", "Zemmour"]

    def test_keeps_lineup_with_most_actual_candidates(self):
        d0, d1 = date(2021, 11, 23), date(2021, 11, 25)
        rows = [
            _rec("Ifop", d0, d1, 1351, Macron=25.0, **{"Le Pen": 19.0}, Zemmour=14.0),  # Bertrand lineup
            _rec("Ifop", d0, d1, 1351, Macron=25.0, **{"Le Pen": 19.5}, Zemmour=15.0, Pécresse=10.0),
        ]
        out = collapse_scenarios(rows, self.CANDS)
        assert len(out) == 1
        assert out[0]["Pécresse"] == 10.0
        assert out[0]["n_scenarios"] == 2

    def test_tie_prefers_larger_share_for_actual_candidates(self):
        # Same real candidates; one lineup adds a non-candidate who takes votes.
        d = date(2022, 1, 24)
        rows = [
            _rec("Harris", d, d, 2496, Macron=24.0, **{"Le Pen": 17.0}, Pécresse=15.0),
            _rec("Harris", d, d, 2496, Macron=25.0, **{"Le Pen": 17.0}, Pécresse=15.0),
        ]
        assert collapse_scenarios(rows, self.CANDS)[0]["Macron"] == 25.0

    def test_distinct_polls_kept(self):
        rows = [
            _rec("Ifop", date(2022, 1, 1), date(2022, 1, 3), 1000, Macron=25.0, **{"Le Pen": 18.0}, Pécresse=15.0),
            _rec("Ifop", date(2022, 1, 2), date(2022, 1, 4), 1000, Macron=26.0, **{"Le Pen": 18.0}, Pécresse=15.0),
            _rec("Elabe", date(2022, 1, 1), date(2022, 1, 3), 1000, Macron=24.0, **{"Le Pen": 18.0}, Pécresse=15.0),
        ]
        assert len(collapse_scenarios(rows, self.CANDS)) == 3


class TestComputePollWeights:
    def test_independent_polls_weight_one(self):
        starts = [date(2022, 1, 1), date(2022, 1, 1), date(2022, 2, 1)]
        ends = [date(2022, 1, 3), date(2022, 1, 3), date(2022, 2, 3)]
        w = compute_poll_weights(starts, ends, ["Ifop", "Elabe", "Ifop"])
        assert np.allclose(w, 1.0)

    def test_rolling_tracker_about_one_over_window(self):
        # Daily releases of a 4-day rolling sample.
        ends = [date(2022, 3, 1) + timedelta(days=i) for i in range(30)]
        starts = [e - timedelta(days=3) for e in ends]
        w = compute_poll_weights(starts, ends, ["Tracker"] * 30)
        assert np.allclose(w[5:-5], 0.25)


# --- Smoothing -------------------------------------------------------------

class TestSmoothPolls:
    def test_recovers_linear_trend(self):
        days = np.arange(-200, -1, 3, dtype=float)
        values = 20 + 0.02 * (days + 200)
        est = smooth_polls(days, values, np.ones_like(days), np.array([-150.0, -50.0]))
        assert np.allclose(est, 20 + 0.02 * np.array([50, 150]), atol=1e-6)

    def test_no_extrapolation_far_from_polls(self):
        days = np.arange(-100, -1, 2, dtype=float)
        est = smooth_polls(days, np.full_like(days, 10.0), np.ones_like(days), np.array([-365.0, -50.0]))
        assert np.isnan(est[0])
        assert est[1] == pytest.approx(10.0)

    def test_weights_matter(self):
        days = np.zeros(4)
        values = np.array([10.0, 10.0, 10.0, 30.0])
        heavy_last = smooth_polls(days, values, np.array([0.5, 0.5, 0.5, 10.0]), np.array([0.0]))[0]
        assert heavy_last > 25

    def test_time_based_not_index_based(self):
        # 50 polls crammed into the final week shouldn't drag the estimate
        # 150 days out, which only sees the early polls.
        early = np.arange(-200, -100, 5, dtype=float)
        late = np.linspace(-8, -2, 50)
        days = np.concatenate([early, late])
        values = np.concatenate([np.full(early.size, 10.0), np.full(late.size, 30.0)])
        est = smooth_polls(days, values, np.ones_like(days), np.array([-150.0]))[0]
        assert est == pytest.approx(10.0, abs=1e-6)


class TestConvertDatesToRelativeDays:
    def test_basic(self):
        dates = np.array([date(2022, 1, 10), date(2022, 4, 5), date(2022, 4, 10)], dtype=object)
        assert list(convert_dates_to_relative_days(dates, date(2022, 4, 10))) == [-90, -5, 0]

    def test_handles_none(self):
        out = convert_dates_to_relative_days(np.array([date(2022, 1, 10), None], dtype=object), date(2022, 4, 10))
        assert out[1] is None and out[0] == -90


# --- Synthetic data for pipeline tests ---------------------------------------

def _synthetic_polls(year: int = 2022, first_day: int = -170) -> pl.DataFrame:
    """Macron climbs 20 → 28, Le Pen flat ~22, Zemmour only polled late."""
    election = ELECTION_DATES[year]
    rows = []
    for offset in range(first_day, 0, 3):
        t = (offset - first_day) / -first_day
        rows.append({
            "date": election + timedelta(days=offset), "weight": 1.0,
            "Macron": 20.0 + 8.0 * t, "Le Pen": 22.0 + 1.0 * t,
            "Zemmour": 10.0 if offset > -60 else None,
        })
    return pl.DataFrame(rows, schema={"date": pl.Date, "weight": pl.Float64,
                                      "Macron": pl.Float64, "Le Pen": pl.Float64, "Zemmour": pl.Float64})


class TestSelectChartCandidates:
    def test_threshold_on_result_or_final_month(self):
        election = ELECTION_DATES[2022]
        polls = pl.DataFrame({
            "date": [election - timedelta(days=d) for d in (400, 20, 10)],
            "weight": [1.0, 1.0, 1.0],
            "Macron": [25.0, 27.0, 27.0],     # result 27.85 → keep
            "Poutou": [0.5, 1.5, 1.5],        # result 0.77 but final month 1.5 → keep
            "Arthaud": [3.0, 0.5, 0.5],       # early 3% only; result 0.56 → drop
        })
        assert select_chart_candidates(2022, polls) == ["Macron", "Poutou"]


class TestComputePollingChanges:
    def test_movement_plus_error_equals_result_minus_start(self):
        polls = _synthetic_polls()
        changes = compute_polling_changes(2022, polls, -150, None, ["Macron", "Le Pen"])
        for row in changes.iter_rows(named=True):
            result = FIRST_ROUND_RESULTS[2022][row["candidate"]]
            assert row["movement"] + row["error"] == pytest.approx(result - row["start"])
            assert row["total"] == pytest.approx(result - row["start"])
        macron = changes.filter(pl.col("candidate") == "Macron").row(0, named=True)
        assert macron["end"] == pytest.approx(
            get_smoothed_value_at_relative_day(polls, "Macron", ELECTION_DATES[2022], FINAL_POLL_DAY))

    def test_start_is_none_when_not_polled(self):
        changes = compute_polling_changes(2022, _synthetic_polls(), -150, None, ["Zemmour"])
        row = changes.row(0, named=True)
        assert row["start"] is None and row["movement"] is None and row["total"] is None
        assert row["error"] == pytest.approx(FIRST_ROUND_RESULTS[2022]["Zemmour"] - row["end"])

    def test_explicit_end_day(self):
        changes = compute_polling_changes(2022, _synthetic_polls(), -150, -30, ["Macron"])
        row = changes.row(0, named=True)
        assert row["result"] is None and row["error"] is None
        assert row["movement"] == pytest.approx(8.0 * 120 / 170, abs=0.05)


class TestCharts:
    def test_relative_day_chart_explicit_end(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        create_relative_day_charts({2022: _synthetic_polls()}, start_days=-150, end_days=-30)
        assert any("Day -30" in ax.get_title(loc="left") for ax in plt.gcf().axes)

    def test_relative_day_chart_actual_result(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        create_relative_day_charts({2022: _synthetic_polls()}, start_days=-150, end_days=None)
        ax = plt.gcf().axes[0]
        assert "Actual Result" in ax.get_title(loc="left")
        assert "percentage points" in ax.get_ylabel()
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("movement" in l for l in labels) and any("error" in l for l in labels)

    def test_election_chart_runs(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        assert create_election_chart(2022, _synthetic_polls(), ["Macron", "Le Pen", "Zemmour"], "first", "1y")


# --- process_year_data with a synthetic page --------------------------------

def _html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False)


def _synthetic_page() -> str:
    main = pd.DataFrame({
        "Polling firm": ["2017 election", "Ifop", "Ifop", "Elabe", "Ipsos"],
        "Fieldwork date": ["23 Apr 2017", "1–3 Mar 2022", "1–3 Mar 2022", "5–6 Mar 2022", "8–10 Apr 2022"],
        "Sample size": ["–", "1,000", "1,000", "900", "1,200"],
        "Macron": ["24.01%", "27%", "26%", "28%", "27%"],
        "Le Pen": ["21.30%", "18%", "18%", "17%", "23%"],
        "Mélenchon": ["19.58%", "12%", "12%", "13%", "18%"],
        "Pécresse": ["20.01%", "–", "11%", "12%", "8%"],
        "Bertrand": ["–", "12%", "–", "–", "–"],
    })
    region = pd.DataFrame({
        "Polling firm": ["Ifop"], "Fieldwork date": ["2 Mar 2022"], "Sample size": ["500"],
        "Macron": ["40%"], "Le Pen": ["10%"], "Mélenchon": ["30%"], "Pécresse": ["5%"],
    })
    runoff = pd.DataFrame({
        "Polling firm": ["Ifop"], "Fieldwork date": ["12–13 Apr 2022"], "Sample size": ["1,000"],
        "Macron": ["54%"], "Le Pen": ["46%"],
    })
    return (
        "<html><body><h2>First round</h2><h3>Official campaign</h3>" + _html_table(main)
        + "<h3>By region</h3><h4>Paris</h4>" + _html_table(region)
        + "<h2>Second round</h2><h3>Macron vs. Le Pen</h3>" + _html_table(runoff)
        + "</body></html>"
    )


class TestProcessYearData:
    def test_first_round_pipeline(self):
        df = process_year_data(2022, fetch_html=lambda _y: _synthetic_page(), write_csv=False)
        # Result row, the post-election-date poll, the regional poll and the
        # Bertrand lineup are all gone; one row per poll remains.
        assert df.height == 2
        assert df["pollster"].to_list() == ["Ifop", "Elabe"]
        ifop = df.row(0, named=True)
        assert ifop["Pécresse"] == 11.0 and ifop["n_scenarios"] == 2
        assert ifop["fieldwork_start"] == date(2022, 3, 1) and ifop["date"] == date(2022, 3, 2)
        assert df["fieldwork_end"].max() < ELECTION_DATES[2022]
        assert set(df.columns) >= {"weight", "sample_size", "year"}

    def test_second_round_pipeline(self):
        df = process_year_data(2022, "second", fetch_html=lambda _y: _synthetic_page(), write_csv=False)
        assert df.height == 1
        assert (df["Macron"][0], df["Le Pen"][0]) == (54.0, 46.0)

    def test_returns_empty_on_no_tables(self):
        df = process_year_data(2022, fetch_html=lambda _y: "<html></html>", write_csv=False)
        assert df is not None and df.height == 0

    def test_unknown_year_returns_none(self):
        assert process_year_data(1999, fetch_html=lambda _y: "", write_csv=False) is None

    def test_parse_polling_tables_records_section_path(self):
        paths = [p for p, _ in parse_polling_tables(_synthetic_page())]
        assert paths == [["First round", "Official campaign"],
                         ["First round", "By region", "Paris"],
                         ["Second round", "Macron vs. Le Pen"]]


class TestExportToCsv:
    def test_writes_file(self, tmp_path):
        df = pl.DataFrame({"date": [date(2022, 1, 1)], "Macron": [25.0], "year": [2022]})
        filename = export_to_csv(2022, df, "first", out_dir=str(tmp_path))
        assert filename.endswith("2022_first_round_polls.csv")
        loaded = pl.read_csv(filename, try_parse_dates=True)
        assert loaded.height == 1 and loaded["Macron"][0] == 25.0


# --- Regression tests on the pinned Wikipedia snapshots ----------------------

def _snapshot_available(year: int) -> bool:
    return (CACHE_DIR / f"{year}_{WIKIPEDIA_REVISIONS[year]}.html.gz").exists()


@pytest.fixture(scope="module")
def snapshot_polls() -> dict[tuple[int, str], pl.DataFrame]:
    out = {}
    for year in FIRST_ROUND_RESULTS:
        if _snapshot_available(year):
            for rnd in ("first", "second"):
                out[(year, rnd)] = process_year_data(year, rnd, write_csv=False)
    if not out:
        pytest.skip("no cached Wikipedia snapshots")
    return out


YEARS = sorted(FIRST_ROUND_RESULTS)


@pytest.mark.parametrize("year", YEARS)
class TestSnapshots:
    def test_one_row_per_poll(self, snapshot_polls, year):
        df = snapshot_polls[(year, "first")]
        keys = df.select("pollster", "fieldwork_start", "fieldwork_end", "sample_size")
        assert keys.height == keys.unique().height

    def test_rows_are_plausible_national_polls(self, snapshot_polls, year):
        df = snapshot_polls[(year, "first")]
        cands = [c for c in get_first_round_candidates(year) if c in df.columns]
        sums = df.select(pl.sum_horizontal(cands).alias("s"))["s"]
        assert sums.max() <= 105
        assert df["fieldwork_end"].max() < ELECTION_DATES[year]
        assert not df["pollster"].str.contains("(?i)election").any()

    def test_every_candidate_above_one_percent_is_polled(self, snapshot_polls, year):
        df = snapshot_polls[(year, "first")]
        recent = df.filter(pl.col("date") >= ELECTION_DATES[year] - timedelta(days=60))
        for c, result in FIRST_ROUND_RESULTS[year].items():
            if result >= 1.0:
                assert c in df.columns, f"{year}: {c} missing"
                assert recent[c].drop_nulls().len() >= 10, f"{year}: {c} barely polled"

    def test_final_polls_close_to_result(self, snapshot_polls, year):
        df = snapshot_polls[(year, "first")]
        errors = compute_polling_changes(year, df, FINAL_POLL_DAY, None,
                                         [c for c, v in FIRST_ROUND_RESULTS[year].items() if v >= 1.0 and c in df.columns])
        mae = errors["error"].abs().mean()
        assert mae < 1.5, f"{year}: mean abs final error {mae:.2f}"

    def test_second_round_rows_are_runoff_polls(self, snapshot_polls, year):
        df = snapshot_polls[(year, "second")]
        a, b = get_second_round_matchups(year)[0]
        assert df.height > 0
        total = df[a] + df[b]
        assert total.min() >= 95 and total.max() <= 105
        assert df["fieldwork_end"].max() < SECOND_ROUND_DATES[year]

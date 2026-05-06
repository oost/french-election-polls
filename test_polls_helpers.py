"""Tests for polls_helpers — parsing and data manipulation against mock data."""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest

import matplotlib

matplotlib.use("Agg")  # headless rendering for create_relative_day_charts test

from polls_helpers import (
    ELECTION_DATES,
    FIRST_ROUND_RESULTS,
    MIN_MEAN_POLL_PCT,
    _extract_numeric,
    convert_dates_to_relative_days,
    create_relative_day_charts,
    drop_low_polling_candidates,
    export_to_csv,
    extract_poll_records,
    filter_by_time_cutoff,
    get_candidate_color,
    get_first_round_candidates,
    get_loess_value_at_relative_day,
    get_second_round_matchups,
    is_valid_first_round_row,
    is_valid_second_round_row,
    parse_date,
    parse_time_cutoff,
    process_year_data,
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


# --- parse_date ------------------------------------------------------------

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
        # "15–18 April 2022" → 2022-04-18
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


# --- parse_time_cutoff -----------------------------------------------------

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


# --- filter_by_time_cutoff -------------------------------------------------

class TestFilterByTimeCutoff:
    def _df(self):
        return pl.DataFrame({
            "date": [date(2021, 1, 1), date(2021, 6, 1), date(2022, 1, 1), date(2022, 4, 1)],
            "Macron": [10.0, 12.0, 15.0, 20.0],
        })

    def test_one_year(self):
        df = self._df()
        result = filter_by_time_cutoff(df, "1y", date(2022, 4, 10))
        # Cutoff = 2021-04-10; rows from 2021-06-01 onward survive
        assert result.height == 3
        assert result["date"].min() == date(2021, 6, 1)

    def test_six_months(self):
        df = self._df()
        result = filter_by_time_cutoff(df, "6m", date(2022, 4, 10))
        # Cutoff = ~2021-10-12; only the two 2022 rows survive
        assert result.height == 2

    def test_none_cutoff_keeps_all(self):
        df = self._df()
        result = filter_by_time_cutoff(df, None, date(2022, 4, 10))
        assert result.height == df.height


# --- Row validation --------------------------------------------------------

class TestIsValidFirstRoundRow:
    def test_normal_first_round(self):
        # Macron 28, Le Pen 23, Mélenchon 17, Pécresse 9
        assert is_valid_first_round_row([28.0, 23.0, 17.0, 9.0]) is True

    def test_includes_minor_candidates(self):
        # Major + minor mix; min < 5 must still pass (≥ 0.5)
        assert is_valid_first_round_row([28.0, 23.0, 17.0, 9.0, 1.0]) is True

    def test_below_three_candidates_rejected(self):
        assert is_valid_first_round_row([28.0, 23.0]) is False

    def test_value_above_40_rejected(self):
        # Second-round head-to-head: e.g. 44, 23, 46, 44
        assert is_valid_first_round_row([44.0, 23.0, 46.0, 44.0]) is False

    def test_min_below_threshold_rejected(self):
        # 0.3% is below the 0.5 floor
        assert is_valid_first_round_row([28.0, 23.0, 0.3]) is False

    def test_empty_rejected(self):
        assert is_valid_first_round_row([]) is False


class TestIsValidSecondRoundRow:
    def test_normal_matchup(self):
        assert is_valid_second_round_row([55.0, 45.0]) is True

    def test_three_values_rejected(self):
        assert is_valid_second_round_row([55.0, 45.0, 30.0]) is False

    def test_out_of_range_rejected(self):
        assert is_valid_second_round_row([15.0, 85.0]) is False
        assert is_valid_second_round_row([10.0, 50.0]) is False


# --- Candidate lookups -----------------------------------------------------

class TestCandidateLookups:
    def test_first_round_known_year(self):
        cands = get_first_round_candidates(2022)
        assert "Macron" in cands
        assert "Le Pen" in cands
        assert "Mélenchon" in cands
        assert len(cands) >= 5

    def test_first_round_unknown_year(self):
        assert get_first_round_candidates(1999) == []

    def test_second_round_known_year(self):
        assert get_second_round_matchups(2017) == [("Macron", "Le Pen")]

    def test_second_round_unknown_year(self):
        assert get_second_round_matchups(1999) == []


class TestFirstRoundResults:
    def test_all_election_years_present(self):
        assert set(FIRST_ROUND_RESULTS.keys()) == {2002, 2007, 2012, 2017, 2022}

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
        # Per-year totals should sum to ~100% with allowance for rounding
        # of single-decimal-place official figures across many candidates.
        for year, results in FIRST_ROUND_RESULTS.items():
            total = sum(results.values())
            assert 99 <= total <= 101, f"{year} total = {total}"

    def test_polled_candidates_have_results(self):
        # Every candidate we keep in the candidate list should have a known result.
        for year, candidates in [
            (2022, get_first_round_candidates(2022)),
            (2017, get_first_round_candidates(2017)),
            (2012, get_first_round_candidates(2012)),
            (2007, get_first_round_candidates(2007)),
            (2002, get_first_round_candidates(2002)),
        ]:
            results = FIRST_ROUND_RESULTS[year]
            for candidate in candidates:
                assert candidate in results, f"{year}: missing result for {candidate}"


class TestCandidateColor:
    def test_known_categories(self):
        assert get_candidate_color("Macron") == "#FFD700"
        assert get_candidate_color("Le Pen") == "#003366"
        assert get_candidate_color("Mélenchon") == "#C41E3A"
        assert get_candidate_color("Hollande") == "#E91B68"
        assert get_candidate_color("Jadot") == "#4A7B3C"
        assert get_candidate_color("Sarkozy") == "#4169E1"

    def test_unknown_returns_grey(self):
        assert get_candidate_color("Mystery Candidate") == "#999999"


# --- extract_poll_records --------------------------------------------------

def _mock_first_round_table() -> pd.DataFrame:
    """A small Wikipedia-style first-round polling table."""
    return pd.DataFrame({
        "Polling firm": ["Ifop", "Ipsos", "BVA"],
        "Fieldwork date": ["1–3 April 2022", "5 April 2022", "7 April 2022"],
        "Macron": ["28%", "27%", "26%"],
        "Le Pen": ["23%", "24%", "23.5%"],
        "Mélenchon": ["17%", "18%", "16%"],
        "Pécresse": ["9%", "8%", "8.5%"],
    })


def _mock_second_round_table() -> pd.DataFrame:
    """A second-round head-to-head table — should be REJECTED in first-round mode
    because every value exceeds 40%."""
    return pd.DataFrame({
        "Polling firm": ["Ifop", "Ipsos"],
        "Fieldwork date": ["28 April 2017", "1 May 2017"],
        "Macron": ["58%", "60%"],
        "Le Pen": ["42%", "40%"],
    })


def _mock_second_round_leak_in_first_round_format() -> pd.DataFrame:
    """A row that has 4 first-round-style columns but values clearly belonging
    to second-round projections (each candidate's % vs Macron)."""
    return pd.DataFrame({
        "Polling firm": ["Ifop"],
        "Fieldwork date": ["2 May 2017"],
        "Macron": ["44"],
        "Le Pen": ["23"],
        "Mélenchon": ["46"],
        "Fillon": ["44"],
    })


class TestExtractPollRecords:
    def test_first_round_table(self):
        records = extract_poll_records(
            _mock_first_round_table(),
            ["Macron", "Le Pen", "Mélenchon", "Pécresse"],
            "first",
        )
        assert len(records) == 3
        assert records[0]["Macron"] == 28.0
        assert records[0]["Le Pen"] == 23.0
        assert records[0]["date"] == date(2022, 4, 3)  # last date in range

    def test_second_round_leak_rejected_in_first_round_mode(self):
        # Values 44/23/46/44 all pass the >= 0.5 floor but max > 40 → reject
        records = extract_poll_records(
            _mock_second_round_leak_in_first_round_format(),
            ["Macron", "Le Pen", "Mélenchon", "Fillon"],
            "first",
        )
        assert records == []

    def test_second_round_mode_extracts_matchup(self):
        records = extract_poll_records(
            _mock_second_round_table(),
            ["Macron", "Le Pen"],
            "second",
        )
        assert len(records) == 2
        assert records[0]["Macron"] == 58.0
        assert records[0]["Le Pen"] == 42.0

    def test_table_without_polling_firm_skipped(self):
        bad = pd.DataFrame({"Date": ["1 April 2022"], "Macron": ["28"], "Le Pen": ["23"]})
        assert extract_poll_records(bad, ["Macron", "Le Pen"], "first") == []

    def test_empty_table(self):
        assert extract_poll_records(pd.DataFrame(), ["Macron"], "first") == []

    def test_unmatched_candidates(self):
        # Only Macron column present; need >= 2 candidate columns to proceed
        df = pd.DataFrame({
            "Polling firm": ["Ifop"],
            "Fieldwork date": ["1 April 2022"],
            "Macron": ["28"],
        })
        assert extract_poll_records(df, ["Macron", "Le Pen"], "first") == []


# --- drop_low_polling_candidates -------------------------------------------

class TestDropLowPolling:
    def test_drops_low_mean_columns(self):
        df = pl.DataFrame({
            "date": [date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1)],
            "Macron": [25.0, 26.0, 27.0],
            "Arthaud": [0.5, 0.7, 0.8],     # mean ≈ 0.67 ≤ 1.0 → drop
            "Le Pen": [22.0, 21.0, 23.0],
            "year": [2022, 2022, 2022],
        })
        result, dropped = drop_low_polling_candidates(df)
        assert dropped == ["Arthaud"]
        assert "Arthaud" not in result.columns
        assert {"date", "Macron", "Le Pen", "year"}.issubset(set(result.columns))

    def test_keeps_above_threshold(self):
        df = pl.DataFrame({
            "date": [date(2022, 1, 1)],
            "Macron": [25.0],
            "Poutou": [1.5],   # > 1.0 → keep
            "year": [2022],
        })
        result, dropped = drop_low_polling_candidates(df)
        assert dropped == []
        assert "Poutou" in result.columns

    def test_drops_at_threshold(self):
        # Threshold check is `<= threshold`; exactly 1.0 should drop
        df = pl.DataFrame({
            "date": [date(2022, 1, 1), date(2022, 2, 1)],
            "Macron": [25.0, 25.0],
            "Borderline": [1.0, 1.0],  # mean = 1.0, ≤ 1.0 → drop
            "year": [2022, 2022],
        })
        result, dropped = drop_low_polling_candidates(df)
        assert "Borderline" in dropped

    def test_protects_date_and_year(self):
        df = pl.DataFrame({"date": [date(2022, 1, 1)], "Macron": [25.0], "year": [2022]})
        result, dropped = drop_low_polling_candidates(df)
        assert "date" in result.columns
        assert "year" in result.columns

    def test_custom_threshold(self):
        df = pl.DataFrame({
            "date": [date(2022, 1, 1)],
            "Macron": [25.0],
            "Hamon": [3.0],   # 3.0 ≤ 5.0 → drop with threshold=5
            "year": [2022],
        })
        _, dropped = drop_low_polling_candidates(df, threshold=5.0)
        assert dropped == ["Hamon"]


# --- process_year_data (with mocked fetcher) -------------------------------

class TestProcessYearData:
    def test_filters_post_election_polls(self):
        """A poll dated on the election day must be excluded for the first round."""
        def fake_fetch(_url):
            return [pd.DataFrame({
                "Polling firm": ["A", "B", "C"],
                "Fieldwork date": ["1 April 2022", "5 April 2022", "10 April 2022"],
                "Macron": ["28", "27", "26"],
                "Le Pen": ["23", "24", "23"],
                "Mélenchon": ["17", "18", "16"],
                "Pécresse": ["9", "8", "8"],
            })]

        df = process_year_data(2022, fetch_tables=fake_fetch, write_csv=False)
        assert df is not None
        # 2022-04-10 == election date → excluded; 2022-04-01 and 2022-04-05 kept
        assert df.height == 2
        assert df["date"].max() < ELECTION_DATES[2022]

    def test_drops_low_polling_candidates_after_concat(self):
        def fake_fetch(_url):
            return [pd.DataFrame({
                "Polling firm": ["A", "B"],
                "Fieldwork date": ["1 March 2022", "1 February 2022"],
                "Macron": ["28", "27"],
                "Le Pen": ["23", "24"],
                "Mélenchon": ["17", "18"],
                "Pécresse": ["9", "8"],
                "Arthaud": ["0.5", "0.7"],   # mean < 1% → drop
            })]

        df = process_year_data(2022, fetch_tables=fake_fetch, write_csv=False)
        assert df is not None
        assert "Arthaud" not in df.columns
        assert "Macron" in df.columns

    def test_returns_empty_on_no_tables(self):
        df = process_year_data(2022, fetch_tables=lambda _u: [], write_csv=False)
        assert df is not None
        assert df.height == 0

    def test_unknown_year_returns_none(self):
        assert process_year_data(1999, fetch_tables=lambda _u: [], write_csv=False) is None


# --- export_to_csv ---------------------------------------------------------

class TestExportToCsv:
    def test_writes_file(self, tmp_path):
        df = pl.DataFrame({
            "date": [date(2022, 1, 1)],
            "Macron": [25.0],
            "year": [2022],
        })
        filename = export_to_csv(2022, df, "first", out_dir=str(tmp_path))
        assert filename.endswith("2022_first_round_polls.csv")
        # Round-trip
        loaded = pl.read_csv(filename, try_parse_dates=True)
        assert loaded.height == 1
        assert loaded["Macron"][0] == 25.0


# --- Relative-day helpers --------------------------------------------------

class TestConvertDatesToRelativeDays:
    def test_basic(self):
        dates = np.array([date(2022, 1, 10), date(2022, 4, 5), date(2022, 4, 10)], dtype=object)
        out = convert_dates_to_relative_days(dates, date(2022, 4, 10))
        assert list(out) == [-90, -5, 0]

    def test_handles_none(self):
        dates = np.array([date(2022, 1, 10), None], dtype=object)
        out = convert_dates_to_relative_days(dates, date(2022, 4, 10))
        assert out[1] is None
        assert out[0] == -90


class TestCreateRelativeDayCharts:
    """End_days=None should pull the end value from FIRST_ROUND_RESULTS rather
    than LOESS-smoothing poll values. We can't easily inspect bar heights via
    matplotlib's API, so verify the path runs without error and the rendered
    title reflects the chosen mode."""

    def _build_polls(self, year: int) -> dict[int, "pl.DataFrame"]:
        import polars as pl
        # Synthetic polls for ~150 days before the election, climbing 20→27%.
        dates = []
        macron = []
        lepen = []
        election = ELECTION_DATES[year]
        for offset in range(170, 0, -5):
            dates.append(date(election.year, election.month, election.day) - __import__("datetime").timedelta(days=offset))
            # Macron rises 20 → 28; Le Pen flat-ish around 22.
            t = (170 - offset) / 170
            macron.append(20.0 + t * 8.0)
            lepen.append(22.0 + t * 1.0)
        return {year: pl.DataFrame({"date": dates, "Macron": macron, "Le Pen": lepen, "year": [year] * len(dates)})}

    def test_with_explicit_end_days_runs(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        data = self._build_polls(2022)
        create_relative_day_charts(data, start_days=-150, end_days=-30)
        # Title should mention the explicit end day
        fig = plt.gcf()
        assert any("Day -30" in ax.get_title() for ax in fig.axes)

    def test_with_end_days_none_uses_actual_result(self):
        import matplotlib.pyplot as plt
        plt.close("all")
        data = self._build_polls(2022)
        create_relative_day_charts(data, start_days=-150, end_days=None)
        fig = plt.gcf()
        # Title should mention "Actual Result"
        assert any("Actual Result" in ax.get_title() for ax in fig.axes)

    def test_end_days_none_change_matches_actual_minus_loess_start(self):
        """For end_days=None, computed change = (actual result) - (LOESS @ start)."""
        import matplotlib.pyplot as plt
        plt.close("all")
        data = self._build_polls(2022)
        # Compute Macron's expected change from the same data the function will use.
        polls = data[2022]
        cd = polls.select(["date", "Macron"]).sort("date")
        dates = cd["date"].to_numpy()
        values = cd["Macron"].to_numpy()
        loess_start = get_loess_value_at_relative_day(dates, values, ELECTION_DATES[2022], -150)
        expected = FIRST_ROUND_RESULTS[2022]["Macron"] - loess_start

        create_relative_day_charts(data, start_days=-150, end_days=None)
        fig = plt.gcf()
        # Find the Macron bar by label and read its height
        ax = fig.axes[0]
        macron_bars = [bar for bar, lbl in zip(ax.patches, ax.get_xticklabels()) if "Macron" in lbl.get_text()]
        assert macron_bars, "no Macron bar found"
        assert abs(macron_bars[0].get_height() - expected) < 0.01


class TestGetLoessValueAtRelativeDay:
    def test_smooth_trend(self):
        """A monotonic upward series should LOESS-smooth to roughly its trend."""
        dates = np.array([date(2022, 1, 1), date(2022, 2, 1), date(2022, 3, 1),
                          date(2022, 3, 15), date(2022, 4, 1), date(2022, 4, 5)], dtype=object)
        # values rising 20→30
        values = np.array([20.0, 22.0, 25.0, 27.0, 29.0, 30.0])
        result = get_loess_value_at_relative_day(dates, values, date(2022, 4, 10), -10)
        assert result is not None
        assert 25 <= result <= 32  # somewhere on the upward trend near the end

    def test_too_few_points(self):
        dates = np.array([date(2022, 1, 1), date(2022, 2, 1)], dtype=object)
        values = np.array([20.0, 25.0])
        assert get_loess_value_at_relative_day(dates, values, date(2022, 4, 10), -10) is None

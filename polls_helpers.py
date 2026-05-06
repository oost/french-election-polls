"""Shared helpers for the French election polling notebooks.

Used by `french_elections_polls.ipynb` (polars). Anything the notebook
needs that is not configuration (round, time cutoff, START/END_DAYS)
or top-level orchestration lives here.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl


# --- Constants -------------------------------------------------------------

ELECTION_DATES: dict[int, date] = {
    2022: datetime(2022, 4, 10).date(),
    2017: datetime(2017, 4, 23).date(),
    2012: datetime(2012, 4, 22).date(),
    2007: datetime(2007, 4, 22).date(),
    2002: datetime(2002, 4, 21).date(),
}

# First-round candidate lists are deliberately generous; columns whose mean
# polling is at or below MIN_MEAN_POLL_PCT are dropped after extraction.
FIRST_ROUND_CANDIDATES: dict[int, list[str]] = {
    2022: ["Macron", "Le Pen", "Mélenchon", "Zemmour", "Pécresse", "Hidalgo", "Jadot", "Roussel", "Dupont-Aignan", "Arthaud", "Poutou", "Lassalle"],
    # Jadot withdrew on 23 Feb 2017 to support Hamon — no first-round result for him.
    2017: ["Macron", "Le Pen", "Fillon", "Mélenchon", "Hamon", "Dupont-Aignan", "Asselineau", "Lassalle", "Arthaud", "Poutou"],
    2012: ["Hollande", "Sarkozy", "Le Pen", "Mélenchon", "Bayrou", "Joly", "Dupont-Aignan", "Poutou", "Arthaud"],
    2007: ["Sarkozy", "Royal", "Bayrou", "Le Pen", "Besancenot", "Villiers", "Buffet", "Voynet", "Laguiller", "Bové"],
    2002: ["Chirac", "Jospin", "Le Pen", "Bayrou", "Madelin", "Chevènement", "Mamère", "Laguiller", "Hue", "Besancenot"],
}

SECOND_ROUND_MATCHUPS: dict[int, list[tuple[str, str]]] = {
    2022: [("Macron", "Le Pen")],
    2017: [("Macron", "Le Pen")],
    2012: [("Hollande", "Sarkozy")],
    2007: [("Sarkozy", "Royal")],
    2002: [("Chirac", "Le Pen")],
}

WIKIPEDIA_URLS: dict[int, str] = {
    2022: "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2022_French_presidential_election",
    2017: "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2017_French_presidential_election",
    2012: "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2012_French_presidential_election",
    2007: "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2007_French_presidential_election",
    2002: "https://en.wikipedia.org/wiki/Opinion_polling_for_the_2002_French_presidential_election",
}

# Actual first-round vote results (% of votes cast). Source: official Ministry
# of the Interior figures published on Wikipedia. Used as a reference marker
# on the chart on each election day.
FIRST_ROUND_RESULTS: dict[int, dict[str, float]] = {
    2022: {
        "Macron": 27.85, "Le Pen": 23.15, "Mélenchon": 21.95, "Zemmour": 7.07,
        "Pécresse": 4.78, "Jadot": 4.63, "Lassalle": 3.13, "Roussel": 2.28,
        "Dupont-Aignan": 2.06, "Hidalgo": 1.75, "Poutou": 0.77, "Arthaud": 0.56,
    },
    2017: {
        "Macron": 24.01, "Le Pen": 21.30, "Fillon": 20.01, "Mélenchon": 19.58,
        "Hamon": 6.36, "Dupont-Aignan": 4.70, "Lassalle": 1.21, "Poutou": 1.09,
        "Asselineau": 0.92, "Arthaud": 0.64, "Cheminade": 0.18,
    },
    2012: {
        "Hollande": 28.63, "Sarkozy": 27.18, "Le Pen": 17.90, "Mélenchon": 11.10,
        "Bayrou": 9.13, "Joly": 2.31, "Dupont-Aignan": 1.79, "Poutou": 1.15,
        "Arthaud": 0.56, "Cheminade": 0.25,
    },
    2007: {
        "Sarkozy": 31.18, "Royal": 25.87, "Bayrou": 18.57, "Le Pen": 10.44,
        "Besancenot": 4.08, "Villiers": 2.23, "Buffet": 1.93, "Voynet": 1.57,
        "Laguiller": 1.33, "Bové": 1.32, "Nihous": 1.15, "Schivardi": 0.34,
    },
    2002: {
        "Chirac": 19.88, "Le Pen": 16.86, "Jospin": 16.18, "Bayrou": 6.84,
        "Laguiller": 5.72, "Chevènement": 5.33, "Mamère": 5.25, "Besancenot": 4.25,
        "Saint-Josse": 4.23, "Madelin": 3.91, "Hue": 3.37, "Mégret": 2.34,
        "Taubira": 2.32, "Lepage": 1.88, "Boutin": 1.19, "Gluckstein": 0.47,
    },
}

MIN_MEAN_POLL_PCT: float = 1.0


# --- Lookups ---------------------------------------------------------------

def get_first_round_candidates(year: int) -> list[str]:
    return FIRST_ROUND_CANDIDATES.get(year, [])


def get_second_round_matchups(year: int) -> list[tuple[str, str]]:
    return SECOND_ROUND_MATCHUPS.get(year, [])


def get_candidate_color(candidate: str) -> str:
    """Return a hex color for a candidate based on French political affiliation."""
    far_left = {"Arthaud", "Poutou", "Mélenchon", "Roussel", "Besancenot", "Buffet", "Laguiller", "Hue"}
    socialist = {"Hollande", "Jospin", "Hamon", "Royal", "Hidalgo"}
    ecologist = {"Jadot", "Joly", "Mamère", "Duflot", "Voynet", "Bové"}
    centrist = {"Macron", "Bayrou"}
    right = {"Fillon", "Sarkozy", "Chirac", "Pécresse", "Juppé", "Villiers", "Dupont-Aignan", "Madelin"}
    far_right = {"Le Pen", "Zemmour"}

    if candidate in far_left:
        return "#C41E3A"
    if candidate in socialist:
        return "#E91B68"
    if candidate in ecologist:
        return "#4A7B3C"
    if candidate in centrist:
        return "#FFD700"
    if candidate in right:
        return "#4169E1"
    if candidate in far_right:
        return "#003366"
    return "#999999"


# --- Parsing ---------------------------------------------------------------

def _extract_numeric(value: Any) -> float | None:
    """Extract a strictly-positive numeric value from a cell. Returns None for
    missing/NaN/zero/negative or strings that contain no digits."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v if v > 0 else None
    if isinstance(value, str):
        match = re.search(r"[\d.]+", value)
        if match:
            try:
                num = float(match.group())
                return num if num > 0 else None
            except ValueError:
                return None
    return None


def parse_date(date_str: Any) -> date | None:
    """Parse Wikipedia date cells. Handles ranges like "15–18 April 2022" by
    taking the last component. Returns None if it can't parse."""
    if date_str is None:
        return None
    if isinstance(date_str, datetime):
        return date_str.date()
    if isinstance(date_str, date):
        return date_str
    if not isinstance(date_str, str):
        return None

    s = date_str.strip()
    if not s:
        return None

    formats = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%Y-%m-%d %H:%M:%S")

    # Try the full string first so well-formed ISO dates ("2022-04-08") aren't
    # mistaken for ranges and split on the hyphen.
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    # Fall back to range handling: "15–18 April 2022" or "15-18 April 2022"
    if "–" in s or "-" in s:
        parts = re.split(r"–|-", s)
        tail = parts[-1].strip()
        for fmt in formats:
            try:
                return datetime.strptime(tail, fmt).date()
            except ValueError:
                continue
    return None


# --- Time filtering --------------------------------------------------------

def parse_time_cutoff(cutoff: str | None) -> timedelta | None:
    """Convert "1y" / "6m" / "30d" → timedelta. Returns None if cutoff is None."""
    if cutoff is None:
        return None
    match = re.fullmatch(r"(\d+)([ymd])", cutoff.lower())
    if not match:
        raise ValueError(f"Invalid time cutoff: {cutoff!r}. Use forms like '1y', '6m', '30d'.")
    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "y":
        return timedelta(days=amount * 365)
    if unit == "m":
        return timedelta(days=amount * 30)
    return timedelta(days=amount)


def filter_by_time_cutoff(df: pl.DataFrame, time_cutoff: str | None, election_date: date) -> pl.DataFrame:
    """Keep rows within `time_cutoff` of the election date."""
    delta = parse_time_cutoff(time_cutoff)
    if delta is None:
        return df
    cutoff_date = election_date - delta
    return df.filter(pl.col("date") >= cutoff_date)


# --- Row validation --------------------------------------------------------

def is_valid_first_round_row(row_values: list[float]) -> bool:
    """First-round rows: 3+ candidates, no value above 40% (rules out
    second-round head-to-head columns), allow tiny minor-candidate values."""
    if len(row_values) < 3:
        return False
    val_min = min(row_values)
    val_max = max(row_values)
    return val_max <= 40 and val_min >= 0.5


def is_valid_second_round_row(row_values: list[float]) -> bool:
    """Second-round rows: exactly 2 candidates, both in 20-60% range."""
    if len(row_values) != 2:
        return False
    return all(20 <= v <= 60 for v in row_values)


# --- Wikipedia fetching ----------------------------------------------------

def fetch_wikipedia_tables(url: str) -> list[pd.DataFrame]:
    """Fetch HTML tables from Wikipedia using pandas (handles rowspan/colspan)."""
    return pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})


# --- Core processing -------------------------------------------------------

def _find_date_column(columns: Iterable[Any]) -> Any | None:
    for col in columns:
        s = str(col).lower()
        if "date" in s or "fieldwork" in s:
            return col
    return None


def _find_candidate_columns(columns: Iterable[Any], candidates: list[str]) -> dict[str, Any]:
    columns = list(columns)
    found: dict[str, Any] = {}
    for candidate in candidates:
        cand_lower = candidate.lower()
        for col in columns:
            if cand_lower in str(col).lower():
                found[candidate] = col
                break
    return found


def _table_has_polling_firm_column(columns: Iterable[Any]) -> bool:
    return any("Polling firm" in str(col) or "Polling organisation" in str(col) for col in columns)


def extract_poll_records(table: pd.DataFrame, candidates: list[str], round_type: str) -> list[dict[str, Any]]:
    """Pull validated poll records out of one Wikipedia table.

    Returns an empty list if the table doesn't look like a polling table or
    no rows pass validation. Caller is responsible for date-boundary and
    column-mean filtering across the combined dataset.
    """
    if table.empty or not _table_has_polling_firm_column(table.columns):
        return []

    date_col = _find_date_column(table.columns)
    if date_col is None:
        return []

    parsed_dates = [parse_date(v) for v in table[date_col]]
    if not any(d is not None for d in parsed_dates):
        return []

    candidate_headers = _find_candidate_columns(table.columns, candidates)
    if len(candidate_headers) < 2:
        return []

    # List comp preserves None — Series.apply would convert it back to NaN.
    candidate_values: dict[str, list[float | None]] = {
        candidate: [_extract_numeric(v) for v in table[col]]
        for candidate, col in candidate_headers.items()
    }

    records: list[dict[str, Any]] = []
    for idx, date_val in enumerate(parsed_dates):
        if date_val is None:
            continue

        record: dict[str, Any] = {"date": date_val}
        row_values: list[float] = []
        for candidate, vals in candidate_values.items():
            if idx < len(vals) and vals[idx] is not None:
                record[candidate] = vals[idx]
                row_values.append(vals[idx])

        if round_type == "first":
            valid = is_valid_first_round_row(row_values)
        else:
            valid = is_valid_second_round_row(row_values)

        if valid:
            records.append(record)

    return records


def drop_low_polling_candidates(
    df: pl.DataFrame,
    threshold: float = MIN_MEAN_POLL_PCT,
    protected: tuple[str, ...] = ("date", "year"),
) -> tuple[pl.DataFrame, list[str]]:
    """Drop columns whose non-null mean is at or below `threshold`.

    Returns (filtered_df, dropped_column_names).
    """
    candidate_cols = [c for c in df.columns if c not in protected]
    to_drop: list[str] = []
    for col in candidate_cols:
        mean_val = df[col].drop_nulls().mean()
        if mean_val is None or mean_val <= threshold:
            to_drop.append(col)
    if to_drop:
        df = df.drop(to_drop)
    return df, to_drop


def export_to_csv(year: int, polls_data: pl.DataFrame, round_type: str = "first", out_dir: str = "polling_data") -> str:
    """Write `polls_data` to `{out_dir}/{year}_{round_type}_round_polls.csv`."""
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/{year}_{round_type}_round_polls.csv"
    polls_data.write_csv(filename)
    columns = polls_data.columns
    col_display = ", ".join(columns[:5]) + ("..." if len(columns) > 5 else "")
    print(f"  📊 CSV exported: {filename} ({polls_data.height} rows, {len(columns)} columns)")
    print(f"     Columns: {col_display}")
    return filename


def process_year_data(
    year: int,
    round_type: str = "first",
    *,
    fetch_tables: Any = fetch_wikipedia_tables,
    write_csv: bool = True,
) -> pl.DataFrame | None:
    """Download, parse, filter, and (optionally) export polling data for one year.

    `fetch_tables` is injectable so tests can supply mock tables.
    """
    url = WIKIPEDIA_URLS.get(year)
    if not url:
        return None

    print(f"Fetching {round_type.upper()} ROUND data for {year}...")

    if round_type == "first":
        candidates = get_first_round_candidates(year)
    else:
        seen: list[str] = []
        for c1, c2 in get_second_round_matchups(year):
            for c in (c1, c2):
                if c not in seen:
                    seen.append(c)
        candidates = seen

    tables = fetch_tables(url)
    all_records: list[dict[str, Any]] = []
    for table in tables:
        all_records.extend(extract_poll_records(table, candidates, round_type))

    if not all_records:
        print(f"Warning: No {round_type} round polling data found for {year}")
        return pl.DataFrame()

    combined = pl.DataFrame(all_records).with_columns(pl.lit(year).alias("year"))

    if round_type == "first":
        election_date = ELECTION_DATES.get(year)
        if election_date is not None:
            before = combined.height
            combined = combined.filter(pl.col("date") < election_date)
            removed = before - combined.height
            if removed:
                print(f"  Removed {removed} polls dated on/after {election_date}")

        combined, dropped = drop_low_polling_candidates(combined)
        if dropped:
            print(f"  Dropped low-polling candidates (≤{MIN_MEAN_POLL_PCT}%): {', '.join(dropped)}")

    print(f"✓ Found {combined.height} {round_type} round polls for {year}")

    if write_csv:
        export_to_csv(year, combined, round_type)

    return combined


# --- Charting --------------------------------------------------------------

def create_election_chart(
    year: int,
    polls_data: pl.DataFrame,
    candidates: list[str],
    round_type: str = "first",
    time_cutoff: str | None = None,
) -> tuple[Any, Any] | None:
    """LOESS-smoothed scatter chart for one election."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if polls_data.height == 0 or not candidates:
        print(f"Skipping {year} - insufficient data")
        return None

    if time_cutoff:
        election_date = ELECTION_DATES.get(year)
        if election_date:
            polls_data = filter_by_time_cutoff(polls_data, time_cutoff, election_date)
            if polls_data.height == 0:
                print(f"Skipping {year} - no polls within {time_cutoff} of election")
                return None

    fig, ax = plt.subplots(figsize=(14, 8))

    election_date = ELECTION_DATES.get(year)
    actual_results = FIRST_ROUND_RESULTS.get(year, {}) if round_type == "first" else {}

    for candidate in candidates:
        if candidate not in polls_data.columns:
            continue
        color = get_candidate_color(candidate)
        candidate_data = polls_data.select(["date", candidate]).filter(
            (pl.col(candidate).is_not_null()) & (pl.col(candidate) > 0)
        ).sort("date")

        if candidate_data.height <= 3:
            continue

        dates = candidate_data["date"].to_numpy()
        values = candidate_data[candidate].to_numpy()
        ax.scatter(dates, values, alpha=0.2, color=color, s=50)

        try:
            numeric_dates = np.arange(len(dates))
            smoothed = lowess(values, numeric_dates, frac=min(0.3, max(0.1, 5 / len(dates))))
            smoothed_indices = smoothed[:, 0].astype(int).clip(0, len(dates) - 1)
            ax.plot(dates[smoothed_indices], smoothed[:, 1], color=color, linewidth=2.5, label=candidate, alpha=0.8)
        except Exception:
            ax.plot(dates, values, color=color, linewidth=2, label=candidate, alpha=0.6)

        # Actual first-round result on election day: solid filled circle in the
        # candidate's color, drawn on top of everything else. clip_on=False so
        # the marker isn't sliced in half when it sits at the axis edge.
        if election_date is not None and candidate in actual_results:
            result_value = actual_results[candidate]
            ax.scatter(
                [election_date], [result_value],
                color=color, s=120, zorder=10, alpha=1.0,
                edgecolors="black", linewidths=1.2, clip_on=False,
            )
            # Label the result with the candidate name and exact percentage,
            # placed to the right of the marker.
            ax.annotate(
                f"{candidate} {result_value:.1f}%",
                xy=(election_date, result_value),
                xytext=(12, 0), textcoords="offset points",
                ha="left", va="center",
                fontsize=9, fontweight="bold", color=color,
                zorder=11, clip_on=False,
            )

    # Extend x-axis past the election date so the result markers and their
    # text labels have room to breathe (labels extend ~75 px past the marker).
    if election_date is not None and actual_results:
        ax.set_xlim(right=mdates.date2num(election_date) + 90)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Polling Support (%)", fontsize=12)
    cutoff_text = f" (Last {time_cutoff})" if time_cutoff else ""
    round_text = "FIRST ROUND" if round_type == "first" else "SECOND ROUND"
    ax.set_title(f"French Election {year} - {round_text} Polling{cutoff_text}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"Chart created for {year} {round_type.upper()} ROUND ({polls_data.height} polls)")
    return fig, ax


# --- Relative-day analysis -------------------------------------------------

def convert_dates_to_relative_days(dates: np.ndarray, election_date: date) -> np.ndarray:
    """Days relative to `election_date` (negative = before)."""
    out: list[int | None] = []
    for d in dates:
        if d is None:
            out.append(None)
        else:
            if isinstance(d, np.datetime64):
                d = d.astype("datetime64[D]").astype(date)
            out.append((d - election_date).days)
    return np.array(out, dtype=object)


def get_loess_value_at_relative_day(
    dates: np.ndarray,
    values: np.ndarray,
    election_date: date,
    target_day: int,
) -> float | None:
    """LOESS-smooth the series and return the value closest to `target_day`."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if len(dates) < 4:
        return None

    relative_days = convert_dates_to_relative_days(dates, election_date)
    mask = ~np.isnan(values.astype(float))
    if mask.sum() < 4:
        return None

    rel_days_clean = relative_days[mask].astype(float)
    values_clean = values[mask].astype(float)

    sort_idx = np.argsort(rel_days_clean)
    rel_sorted = rel_days_clean[sort_idx]
    val_sorted = values_clean[sort_idx]

    try:
        smoothed = lowess(val_sorted, rel_sorted, frac=min(0.3, max(0.1, 5 / len(val_sorted))))
        closest_idx = int(np.argmin(np.abs(smoothed[:, 0] - target_day)))
        return float(smoothed[closest_idx, 1])
    except Exception:
        return None


def create_relative_day_charts(
    all_election_data: dict[int, pl.DataFrame],
    start_days: int,
    end_days: int | None,
) -> None:
    """Bar charts of polling change between two relative-to-election days.

    If `end_days` is None, the end value is the actual first-round result
    (from FIRST_ROUND_RESULTS) instead of a LOESS-smoothed poll value —
    useful for measuring how each candidate's late-stage polling compared
    to the actual outcome.
    """
    if not all_election_data:
        print("⚠️ Error: No election data available.")
        return

    end_label = "Actual Result" if end_days is None else f"Day {end_days}"
    end_descr = (
        "actual first-round result"
        if end_days is None
        else f"{end_days} days before election"
    )

    num_years = len(all_election_data)
    fig, axes = plt.subplots(num_years, 1, figsize=(14, 4 * num_years))
    if num_years == 1:
        axes = [axes]

    for ax_idx, year in enumerate(sorted(all_election_data.keys())):
        polls_data = all_election_data[year]
        candidates = get_first_round_candidates(year)
        election_date = ELECTION_DATES.get(year)
        ax = axes[ax_idx]

        if polls_data.height == 0 or election_date is None:
            continue

        results = FIRST_ROUND_RESULTS.get(year, {})
        changes: dict[str, float] = {}
        for candidate in candidates:
            if candidate not in polls_data.columns:
                continue
            cd = polls_data.select(["date", candidate]).filter(
                (pl.col(candidate).is_not_null()) & (pl.col(candidate) > 0)
            ).sort("date")
            if cd.height < 4:
                continue
            dates = cd["date"].to_numpy()
            values = cd[candidate].to_numpy()
            start_val = get_loess_value_at_relative_day(dates, values, election_date, start_days)
            if start_val is None:
                continue

            if end_days is None:
                end_val: float | None = results.get(candidate)
            else:
                end_val = get_loess_value_at_relative_day(dates, values, election_date, end_days)

            if end_val is not None:
                changes[candidate] = end_val - start_val

        if not changes:
            print(f"⚠️ No data for {year}")
            continue

        ordered = sorted(changes.keys(), key=lambda c: changes[c], reverse=True)
        values_list = [changes[c] for c in ordered]
        colors = [get_candidate_color(c) for c in ordered]

        bars = ax.bar(ordered, values_list, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        for bar, val in zip(bars, values_list):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h, f"{val:+.1f}%",
                    ha="center", va="bottom" if h > 0 else "top", fontweight="bold", fontsize=10)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_ylabel("Change in Support (%)", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Election {year} - Polling Change from Day {start_days} to {end_label}",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)
        ax.text(
            0.98, 0.97,
            f"Day {start_days}: {start_days} days before election\n{end_label}: {end_descr}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.show()

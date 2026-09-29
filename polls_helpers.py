"""Shared helpers for the French election polling notebook.

Used by `french_elections_polls.ipynb` (polars). Anything the notebook
needs that is not configuration (round, time cutoff, START/END_DAYS)
or top-level orchestration lives here.

Pipeline: pinned Wikipedia revision (cached HTML) → polling tables classified
by section heading → one row per poll (hypothetical lineups collapsed) →
per-poll weights (rolling trackers down-weighted) → time-based kernel smoothing.
"""
from __future__ import annotations

import gzip
import io
import os
import re
import unicodedata
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from bs4 import BeautifulSoup


# --- Constants -------------------------------------------------------------

ELECTION_DATES: dict[int, date] = {
    2022: date(2022, 4, 10),
    2017: date(2017, 4, 23),
    2012: date(2012, 4, 22),
    2007: date(2007, 4, 22),
    2002: date(2002, 4, 21),
}

SECOND_ROUND_DATES: dict[int, date] = {
    2022: date(2022, 4, 24),
    2017: date(2017, 5, 7),
    2012: date(2012, 5, 6),
    2007: date(2007, 5, 6),
    2002: date(2002, 5, 5),
}

# 2027 first round, set by the Council of Ministers on 1 July 2026.
NEXT_ELECTION_DATE: date = date(2027, 4, 18)

SECOND_ROUND_MATCHUPS: dict[int, list[tuple[str, str]]] = {
    2022: [("Macron", "Le Pen")],
    2017: [("Macron", "Le Pen")],
    2012: [("Hollande", "Sarkozy")],
    2007: [("Sarkozy", "Royal")],
    2002: [("Chirac", "Le Pen")],
}

WIKIPEDIA_TITLES: dict[int, str] = {
    year: f"Opinion_polling_for_the_{year}_French_presidential_election"
    for year in ELECTION_DATES
}
WIKIPEDIA_URLS: dict[int, str] = {
    year: f"https://en.wikipedia.org/wiki/{title}" for year, title in WIKIPEDIA_TITLES.items()
}
# Pinned page revisions so results are reproducible. Bump a revision (and
# delete its cache file) to pick up later edits to the Wikipedia page.
WIKIPEDIA_REVISIONS: dict[int, int] = {
    2022: 1363121499,  # 2026-07-08
    2017: 1213598918,  # 2024-03-14
    2012: 1279812574,  # 2025-03-10
    2007: 1341388824,  # 2026-03-02
    2002: 1329460026,  # 2025-12-26
}
CACHE_DIR = Path(__file__).resolve().parent / "wiki_cache"
USER_AGENT = "french-elections-polling-analysis/1.0 (research notebook)"

# Actual first-round vote results (% of votes cast), every candidate on the
# ballot. Source: official Ministry of the Interior figures published on
# Wikipedia. Doubles as the candidate list — polls are only tracked for
# candidates who actually ran.
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

SECOND_ROUND_RESULTS: dict[int, dict[str, float]] = {
    2022: {"Macron": 58.55, "Le Pen": 41.45},
    2017: {"Macron": 66.10, "Le Pen": 33.90},
    2012: {"Hollande": 51.64, "Sarkozy": 48.36},
    2007: {"Sarkozy": 53.06, "Royal": 46.94},
    2002: {"Chirac": 82.21, "Le Pen": 17.79},
}

# A candidate is charted if their actual result, or their weighted polling
# average over the final FINAL_WINDOW_DAYS, exceeds this share.
MIN_SUPPORT_PCT: float = 1.0
FINAL_WINDOW_DAYS: int = 30

# French law bans publishing polls on the eve and day of the vote, so the
# last fieldwork ends on the Friday — two days out.
FINAL_POLL_DAY: int = -2

# Gaussian kernel SD (days) for the local-linear smoother; ±2 SD ≈ 6 weeks.
SMOOTHING_BANDWIDTH_DAYS: float = 10.0
# Minimum kernel-weighted poll count for a smoothed value to be reported.
# Below this the smoother would be extrapolating from distant polls.
MIN_EFFECTIVE_POLLS: float = 2.0

# Section headings whose tables are not national voting-intention polls.
_EXCLUDED_SECTION = re.compile(
    r"\bby (region|department|commune|first[- ]round vote)\b|overseas|abroad", re.I
)
# Rows that report an election result rather than a poll ("2017 election").
_RESULT_ROW = re.compile(r"\belection\b|\bresults?\b", re.I)


# --- Lookups ---------------------------------------------------------------

def get_first_round_candidates(year: int) -> list[str]:
    """Everyone on the first-round ballot, in finishing order."""
    return list(FIRST_ROUND_RESULTS.get(year, {}))


def get_second_round_matchups(year: int) -> list[tuple[str, str]]:
    return SECOND_ROUND_MATCHUPS.get(year, [])


def days_until_next_election(today: date | None = None) -> int:
    """Days from `today` (default: the current date) to the 2027 first round.

    Used as the default look-back for the polling change analysis, so past
    campaigns are compared at the point the 2027 race has reached now.
    """
    return (NEXT_ELECTION_DATE - (today or date.today())).days


def get_round_candidates(year: int, round_type: str = "first") -> list[str]:
    if round_type == "first":
        return get_first_round_candidates(year)
    seen: list[str] = []
    for pair in get_second_round_matchups(year):
        for c in pair:
            if c not in seen:
                seen.append(c)
    return seen


def get_round_date(year: int, round_type: str = "first") -> date | None:
    return (ELECTION_DATES if round_type == "first" else SECOND_ROUND_DATES).get(year)


def get_round_results(year: int, round_type: str = "first") -> dict[str, float]:
    return (FIRST_ROUND_RESULTS if round_type == "first" else SECOND_ROUND_RESULTS).get(year, {})


# Party-family hue (the convention French readers expect) with a distinct
# shade per candidate. Shades were tuned so that within each election every
# pair of charted candidates differs by OKLab ΔE ≥ 15 and every color keeps
# ≥ 3:1 contrast on white. Candidates are also direct-labeled, so identity
# never rests on color alone.
CANDIDATE_COLORS: dict[str, str] = {
    # Far left / communist
    "Mélenchon": "#aa0c3e", "Roussel": "#531c14", "Hue": "#571909", "Buffet": "#561f2a",
    "Poutou": "#f1331e", "Besancenot": "#f0321d",
    "Arthaud": "#8e1b4a", "Laguiller": "#a40540", "Gluckstein": "#b56576", "Schivardi": "#b56576",
    # Socialist / left
    "Hollande": "#be62a9", "Jospin": "#fb43ac", "Royal": "#c84dad", "Hamon": "#d557b4", "Hidalgo": "#c762ab",
    "Taubira": "#9911ad", "Chevènement": "#a56c8e",
    # Ecologist
    "Jadot": "#2da155", "Joly": "#2f7218", "Mamère": "#4ea71e", "Voynet": "#079600",
    "Bové": "#5e9e94", "Lepage": "#268986",
    # Centre
    "Macron": "#bf8929", "Bayrou": "#c4881a",
    # Right
    "Pécresse": "#5760f6", "Fillon": "#4f5af2", "Sarkozy": "#5550f1", "Chirac": "#484dd7",
    "Madelin": "#4596f6", "Boutin": "#ae62fe", "Villiers": "#571f92",
    "Dupont-Aignan": "#459cc4", "Asselineau": "#3c923b", "Cheminade": "#7f8ea3",
    # Rural / hunting
    "Lassalle": "#716053", "Saint-Josse": "#655a00", "Nihous": "#586333",
    # Far right
    "Le Pen": "#05132f", "Zemmour": "#73308e", "Mégret": "#323d72",
}


def get_candidate_color(candidate: str) -> str:
    """Return a hex color for a candidate (party-family hue, per-candidate shade)."""
    return CANDIDATE_COLORS.get(candidate, "#999999")


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


def _parse_sample_size(value: Any) -> int | None:
    """Sample sizes use thousands separators ("1,351"), which `_extract_numeric`
    would truncate at the comma."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    digits = re.sub(r"\[.*?\]", "", str(value))
    match = re.search(r"\d[\d,\s ]*", digits)
    if not match:
        return None
    n = int(re.sub(r"\D", "", match.group()))
    return n or None


_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%Y-%m-%d %H:%M:%S")


def _clean_date_text(s: str) -> str:
    s = re.sub(r"\[.*?\]", "", s).replace(" ", " ")
    s = re.sub(r"\bSept\b", "Sep", s)
    return re.sub(r"\s+", " ", s).strip()


def _strptime_any(s: str, formats: Iterable[str]) -> date | None:
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
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

    s = _clean_date_text(date_str)
    if not s:
        return None

    # Try the full string first so well-formed ISO dates ("2022-04-08") aren't
    # mistaken for ranges and split on the hyphen.
    parsed = _strptime_any(s, _DATE_FORMATS)
    if parsed is not None:
        return parsed

    # Fall back to range handling: "15–18 April 2022" or "15-18 April 2022"
    if "–" in s or "-" in s or "—" in s:
        tail = re.split(r"[–—-]", s)[-1].strip()
        return _strptime_any(tail, _DATE_FORMATS)
    return None


def parse_fieldwork(value: Any) -> tuple[date, date] | None:
    """Parse a fieldwork cell into (start, end).

    Handles "8 April 2022", "15–18 April 2022", "28 Feb–3 Mar 2022" and
    "30 Dec 2021–3 Jan 2022". Returns None if the end date can't be parsed.
    """
    if isinstance(value, (date, datetime)):
        d = parse_date(value)
        return (d, d)
    if not isinstance(value, str):
        return None
    s = _clean_date_text(value)
    whole = _strptime_any(s, _DATE_FORMATS)
    if whole is not None:
        return (whole, whole)

    parts = [p.strip() for p in re.split(r"[–—-]", s) if p.strip()]
    if len(parts) < 2:
        return None
    end = _strptime_any(parts[-1], _DATE_FORMATS)
    if end is None:
        return None
    head = parts[0]
    start = _strptime_any(head, _DATE_FORMATS)
    if start is None:
        day_month = _strptime_any(f"{head} {end.year}", ("%d %B %Y", "%d %b %Y"))
        if day_month is not None:
            start = day_month if day_month <= end else day_month.replace(year=end.year - 1)
        elif head.isdigit() and 1 <= int(head) <= end.day:
            start = end.replace(day=int(head))
    if start is None or start > end:
        start = end
    return (start, end)


def clean_pollster(value: Any) -> str:
    """Strip Wayback-archive notes and footnote markers from a pollster cell."""
    s = re.sub(r"\s*Archived .*$", "", str(value))
    s = re.sub(r"\[.*?\]", "", s)
    return re.sub(r"\s+", " ", s).strip()


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
    """First-round rows: 3+ candidates, each at most 50%, shares summing to at
    most ~100% (rounding and "<0.5" cells allow a little slack)."""
    if len(row_values) < 3:
        return False
    return max(row_values) <= 50 and sum(row_values) <= 105


def is_valid_second_round_row(row_values: list[float]) -> bool:
    """Second-round rows: exactly 2 candidates whose shares sum to ~100%."""
    if len(row_values) != 2:
        return False
    return 95 <= sum(row_values) <= 105


# --- Wikipedia fetching ----------------------------------------------------

def load_wikipedia_html(year: int, cache_dir: Path | str = CACHE_DIR, refresh: bool = False) -> str:
    """Return the HTML of the pinned revision for `year`, downloading it into
    `cache_dir` (gzipped) on first use."""
    revision = WIKIPEDIA_REVISIONS[year]
    path = Path(cache_dir) / f"{year}_{revision}.html.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    url = f"https://en.wikipedia.org/w/index.php?oldid={revision}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(html)
    return html


def _section_path(table: Any) -> list[str]:
    """Heading chain (h2 → h3 → …) that a table sits under."""
    path: list[str] = []
    level = 7
    for heading in table.find_all_previous(["h2", "h3", "h4", "h5", "h6"]):
        lvl = int(heading.name[1])
        if lvl < level:
            path.append(heading.get_text(" ", strip=True))
            level = lvl
        if lvl == 2:
            break
    return path[::-1]


def classify_section(path: list[str]) -> str | None:
    """'first' / 'second' for national polling sections, None otherwise
    (subnational breakdowns, vote-transfer crosstabs, unrelated sections)."""
    if any(_EXCLUDED_SECTION.search(p) for p in path):
        return None
    for p in path:
        if re.search(r"first round", p, re.I):
            return "first"
        if re.search(r"second round", p, re.I):
            return "second"
    return None


def parse_polling_tables(html: str) -> list[tuple[list[str], pd.DataFrame]]:
    """All HTML tables on the page with their section heading path."""
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[list[str], pd.DataFrame]] = []
    for table in soup.find_all("table"):
        try:
            frames = pd.read_html(io.StringIO(str(table)))
        except ValueError:
            continue
        if frames:
            out.append((_section_path(table), frames[0]))
    return out


# --- Core processing -------------------------------------------------------

def _column_label(col: Any) -> str:
    """Flatten a (possibly multi-level) header into readable text."""
    parts = col if isinstance(col, tuple) else (col,)
    seen: list[str] = []
    for p in parts:
        s = str(p)
        if s.startswith("Unnamed:") or s in seen:
            continue
        seen.append(s)
    return " ".join(seen)


def _normalize_name(s: str) -> str:
    """Lowercase, NFC, and repair hyphenated names that Wikipedia wraps
    across lines ("Dupont- Aignan" → "dupont-aignan")."""
    s = unicodedata.normalize("NFC", str(s)).replace("­", "").replace(" ", " ")
    s = re.sub(r"\s*[-‐‑]\s*", "-", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _find_date_column(columns: Iterable[Any]) -> Any | None:
    for col in columns:
        s = _column_label(col).lower()
        if "date" in s or "fieldwork" in s:
            return col
    return None


def _find_column(columns: Iterable[Any], *needles: str) -> Any | None:
    for col in columns:
        s = _column_label(col).lower()
        if any(n in s for n in needles):
            return col
    return None


def _find_candidate_columns(columns: Iterable[Any], candidates: list[str]) -> dict[str, Any]:
    """Map candidate → column, matching the surname as a whole word."""
    columns = list(columns)
    labels = [_normalize_name(_column_label(c)) for c in columns]
    found: dict[str, Any] = {}
    for candidate in candidates:
        pattern = re.compile(rf"(?<![\w-]){re.escape(_normalize_name(candidate))}(?![\w-])")
        for col, label in zip(columns, labels):
            if pattern.search(label):
                found[candidate] = col
                break
    return found


def _table_has_polling_firm_column(columns: Iterable[Any]) -> bool:
    return _find_column(columns, "polling firm", "polling organisation", "pollster") is not None


_PLACEHOLDER_CELLS = {"", "nan", "none", "–", "-", "—", "n/a", "—n/a", "–n/a", "tbd"}


def _is_blank_cell(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return True
    return str(value).strip().lower() in _PLACEHOLDER_CELLS


def is_event_row(pollster_cell: Any, cells: Iterable[Any], min_same: float = 0.8) -> bool:
    """True for timeline rows ("16 Sep 2026 | Faure, Glucksmann, … are the 5
    candidates…") that Wikipedia spreads across every column of a polling table.

    Such a row has no pollster and repeats one piece of prose in (almost) every
    other cell. `_extract_numeric` would otherwise read digits out of the
    prose and turn the row into a phantom poll. `cells` should exclude the
    fieldwork-date cell.
    """
    if not _is_blank_cell(pollster_cell):
        return False
    cells = list(cells)
    texts = [str(v).strip() for v in cells if not _is_blank_cell(v)]
    if not texts:
        return False
    most_common = max(set(texts), key=texts.count)
    is_prose = len(most_common) >= 4 and re.search(r"[^\W\d_]{3,}", most_common) is not None
    return is_prose and texts.count(most_common) >= min_same * len(cells)


def extract_poll_records(table: pd.DataFrame, candidates: list[str], round_type: str) -> list[dict[str, Any]]:
    """Pull validated poll rows out of one Wikipedia table.

    Every row carries poll metadata (pollster, fieldwork dates, sample size)
    so hypothetical-lineup rows of the same poll can be collapsed later.
    Returns an empty list if the table doesn't look like a polling table of
    the requested round.
    """
    if table.empty or not _table_has_polling_firm_column(table.columns):
        return []

    date_col = _find_date_column(table.columns)
    if date_col is None:
        return []
    firm_col = _find_column(table.columns, "polling firm", "polling organisation", "pollster")
    sample_col = _find_column(table.columns, "sample")

    candidate_headers = _find_candidate_columns(table.columns, candidates)
    if round_type == "first" and len(candidate_headers) < 3:
        return []
    if round_type == "second" and len(candidate_headers) != 2:
        return []

    candidate_values = {
        candidate: [_extract_numeric(v) for v in table[col]]
        for candidate, col in candidate_headers.items()
    }
    raw_firms = table[firm_col].tolist()
    firms = [clean_pollster(v) for v in raw_firms]
    samples = [_parse_sample_size(v) for v in table[sample_col]] if sample_col is not None else [None] * len(table)
    other_cols = [c for c in table.columns if c not in (date_col, firm_col)]

    records: list[dict[str, Any]] = []
    for idx, fieldwork_raw in enumerate(table[date_col]):
        fieldwork = parse_fieldwork(fieldwork_raw)
        if fieldwork is None or _RESULT_ROW.search(firms[idx]):
            continue
        if is_event_row(raw_firms[idx], table.iloc[idx][other_cols].tolist()):
            continue
        start, end = fieldwork

        values = {c: vals[idx] for c, vals in candidate_values.items() if vals[idx] is not None}
        row_values = list(values.values())
        valid = is_valid_first_round_row(row_values) if round_type == "first" else is_valid_second_round_row(row_values)
        if not valid:
            continue

        records.append({
            "date": start + (end - start) / 2,
            "fieldwork_start": start,
            "fieldwork_end": end,
            "pollster": firms[idx],
            "sample_size": samples[idx],
            "_row": idx,
            **values,
        })
    return records


def collapse_scenarios(records: list[dict[str, Any]], candidates: list[str]) -> list[dict[str, Any]]:
    """Keep one row per poll.

    Wikipedia lists a poll once per hypothetical lineup (e.g. with Bertrand
    instead of Pécresse). Rows sharing pollster, fieldwork and sample size are
    one poll; keep the lineup closest to the real ballot — most actual
    candidates, then the largest combined share for them — and ties go to the
    row listed first. The kept row records how many lineups the poll tested.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for order, rec in enumerate(records):
        key = (rec["pollster"], rec["fieldwork_start"], rec["fieldwork_end"], rec["sample_size"])
        groups.setdefault(key, []).append({**rec, "_order": order})

    out: list[dict[str, Any]] = []
    for rows in groups.values():
        def score(r: dict[str, Any]) -> tuple:
            present = [r[c] for c in candidates if r.get(c) is not None]
            return (len(present), sum(present), -r["_order"])
        best = dict(max(rows, key=score))
        best["n_scenarios"] = len(rows)
        best.pop("_order", None)
        best.pop("_row", None)
        out.append(best)
    return sorted(out, key=lambda r: (r["date"], r["pollster"]))


def compute_poll_weights(starts: list[date], ends: list[date], pollsters: list[str]) -> np.ndarray:
    """Down-weight overlapping releases from the same pollster.

    A rolling tracker publishes daily over a multi-day window, so consecutive
    releases share most respondents. A poll's weight is its fieldwork length
    divided by the total days of same-pollster fieldwork overlapping it
    (itself included): independent polls get 1, a daily 4-day rolling tracker
    ≈ 1/4 per release — one fresh day of sample each.
    """
    n = len(starts)
    weights = np.ones(n)
    s = np.array([d.toordinal() for d in starts])
    e = np.array([d.toordinal() for d in ends])
    firms = np.array(pollsters, dtype=object)
    for i in range(n):
        same = firms == firms[i]
        overlap = np.minimum(e[same], e[i]) - np.maximum(s[same], s[i]) + 1
        total = overlap[overlap > 0].sum()
        weights[i] = (e[i] - s[i] + 1) / total
    return weights


def select_chart_candidates(
    year: int,
    polls: pl.DataFrame,
    round_type: str = "first",
    threshold: float = MIN_SUPPORT_PCT,
    window_days: int = FINAL_WINDOW_DAYS,
) -> list[str]:
    """Candidates worth charting, in finishing order.

    First round: keep a candidate whose actual result, or weighted polling
    average over the final `window_days`, exceeds `threshold`. Averaging over
    the whole multi-year table would drop late risers and candidates polled
    in hypothetical lineups years earlier.
    """
    candidates = [c for c in get_round_candidates(year, round_type) if c in polls.columns]
    if round_type != "first":
        return candidates
    election_date = get_round_date(year, round_type)
    results = get_round_results(year, round_type)
    recent = polls.filter(pl.col("date") >= election_date - timedelta(days=window_days))
    keep: list[str] = []
    for c in candidates:
        sub = recent.filter(pl.col(c).is_not_null())
        recent_mean = (
            float((sub[c] * sub["weight"]).sum() / sub["weight"].sum()) if sub.height else 0.0
        )
        if results.get(c, 0.0) >= threshold or recent_mean > threshold:
            keep.append(c)
    return keep


def export_to_csv(year: int, polls_data: pl.DataFrame, round_type: str = "first", out_dir: str = "polling_data") -> str:
    """Write `polls_data` to `{out_dir}/{year}_{round_type}_round_polls.csv`."""
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/{year}_{round_type}_round_polls.csv"
    polls_data.write_csv(filename)
    print(f"  📊 CSV exported: {filename} ({polls_data.height} rows, {len(polls_data.columns)} columns)")
    return filename


_META_COLUMNS = ["date", "fieldwork_start", "fieldwork_end", "pollster", "sample_size", "n_scenarios", "weight"]


def process_year_data(
    year: int,
    round_type: str = "first",
    *,
    fetch_html: Callable[[int], str] = load_wikipedia_html,
    write_csv: bool = True,
) -> pl.DataFrame | None:
    """Parse, clean and (optionally) export one election's polls: one row per
    poll, with metadata columns followed by one column per candidate.

    `fetch_html` is injectable so tests can supply synthetic pages.
    """
    if year not in WIKIPEDIA_REVISIONS:
        return None

    print(f"Parsing {round_type.upper()} ROUND polls for {year}...")
    candidates = get_round_candidates(year, round_type)
    round_date = get_round_date(year, round_type)

    raw: list[dict[str, Any]] = []
    for path, table in parse_polling_tables(fetch_html(year)):
        if classify_section(path) == round_type:
            raw.extend(extract_poll_records(table, candidates, round_type))

    before = len(raw)
    raw = [r for r in raw if r["fieldwork_end"] < round_date]
    if before - len(raw):
        print(f"  Removed {before - len(raw)} rows with fieldwork ending on/after {round_date}")

    polls = collapse_scenarios(raw, candidates)
    if len(polls) < len(raw):
        print(f"  Collapsed {len(raw)} lineup rows into {len(polls)} polls")

    if not polls:
        print(f"Warning: No {round_type} round polling data found for {year}")
        return pl.DataFrame()

    weights = compute_poll_weights(
        [p["fieldwork_start"] for p in polls],
        [p["fieldwork_end"] for p in polls],
        [p["pollster"] for p in polls],
    )
    present = [c for c in candidates if any(p.get(c) is not None for p in polls)]
    schema: dict[str, Any] = {
        "date": pl.Date, "fieldwork_start": pl.Date, "fieldwork_end": pl.Date,
        "pollster": pl.Utf8, "sample_size": pl.Int64, "n_scenarios": pl.Int64,
        "weight": pl.Float64, **{c: pl.Float64 for c in present},
    }
    rows = [{**{k: p.get(k) for k in schema if k != "weight"}, "weight": float(w)} for p, w in zip(polls, weights)]
    df = pl.DataFrame(rows, schema=schema).with_columns(pl.lit(year).alias("year"))

    missing = [c for c in candidates if c not in present]
    if missing:
        print(f"  No polls found for: {', '.join(missing)}")
    print(f"✓ Found {df.height} {round_type} round polls for {year}")

    if write_csv:
        export_to_csv(year, df, round_type)
    return df


# --- Smoothing -------------------------------------------------------------

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


def smooth_polls(
    days: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    eval_days: np.ndarray,
    bandwidth_days: float = SMOOTHING_BANDWIDTH_DAYS,
    min_effective_polls: float = MIN_EFFECTIVE_POLLS,
) -> np.ndarray:
    """Weighted local-linear regression with a Gaussian kernel in *time*.

    Unlike LOESS with a fraction-of-points span, the window is the same
    width everywhere, however densely polls cluster. Returns NaN wherever the
    kernel-weighted number of polls is below `min_effective_polls`, i.e. no
    silent extrapolation into periods with no polling.
    """
    x = np.asarray(days, dtype=float)
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    g = np.atleast_1d(np.asarray(eval_days, dtype=float))
    ok = ~np.isnan(y)
    x, y, w = x[ok], y[ok], w[ok]
    if x.size == 0:
        return np.full(g.shape, np.nan)

    u = x[None, :] - g[:, None]
    k = np.exp(-0.5 * (u / bandwidth_days) ** 2) * w[None, :]
    s0 = k.sum(axis=1)
    s1 = (k * u).sum(axis=1)
    s2 = (k * u * u).sum(axis=1)
    t0 = (k * y).sum(axis=1)
    t1 = (k * u * y).sum(axis=1)
    denom = s0 * s2 - s1 * s1
    with np.errstate(invalid="ignore", divide="ignore"):
        local_linear = (s2 * t0 - s1 * t1) / denom
        local_mean = t0 / s0
    # Near-degenerate designs (all nearby polls on one day) → local mean.
    degenerate = denom <= 1e-9 * np.maximum(s0 * s2, 1e-12)
    est = np.where(degenerate, local_mean, local_linear)
    est[s0 < min_effective_polls] = np.nan
    return est


def candidate_series(polls: pl.DataFrame, candidate: str, election_date: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(relative_days, values, weights) for one candidate's non-null polls."""
    sub = polls.filter(pl.col(candidate).is_not_null())
    days = np.array([(d - election_date).days for d in sub["date"].to_list()], dtype=float)
    weights = sub["weight"].to_numpy() if "weight" in sub.columns else np.ones(sub.height)
    return days, sub[candidate].to_numpy().astype(float), weights


def get_smoothed_value_at_relative_day(
    polls: pl.DataFrame, candidate: str, election_date: date, target_day: int
) -> float | None:
    """Smoothed support for `candidate` at `target_day` (relative to the
    election), or None if there were too few polls around that day."""
    if candidate not in polls.columns:
        return None
    days, values, weights = candidate_series(polls, candidate, election_date)
    est = smooth_polls(days, values, weights, np.array([target_day]))[0]
    return None if np.isnan(est) else float(est)


# --- Tables ----------------------------------------------------------------

def summarise_polls(
    year: int,
    polls: pl.DataFrame,
    candidates: list[str],
    round_type: str = "first",
    time_cutoff: str | None = None,
) -> pl.DataFrame:
    """Per-candidate poll count, weighted mean, range, final smoothed value
    and actual result over the `time_cutoff` window."""
    round_date = get_round_date(year, round_type)
    results = get_round_results(year, round_type)
    window = filter_by_time_cutoff(polls, time_cutoff, round_date)
    rows = []
    for c in candidates:
        if c not in window.columns:
            continue
        sub = window.filter(pl.col(c).is_not_null())
        if sub.height == 0:
            continue
        final = get_smoothed_value_at_relative_day(polls, c, round_date, FINAL_POLL_DAY)
        result = results.get(c)
        rows.append({
            "candidate": c,
            "polls": sub.height,
            "weighted_mean": float((sub[c] * sub["weight"]).sum() / sub["weight"].sum()),
            "min": float(sub[c].min()),
            "max": float(sub[c].max()),
            "final_poll": final,
            "result": result,
            "final_error": None if final is None or result is None else result - final,
        })
    return pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(1)) if rows else pl.DataFrame()


def compute_polling_changes(
    year: int,
    polls: pl.DataFrame,
    start_days: int,
    end_days: int | None,
    candidates: list[str] | None = None,
) -> pl.DataFrame:
    """Per-candidate change in smoothed support between two relative days.

    With `end_days=None` the change from `start_days` to the actual result is
    split into `movement` (start → final polls at FINAL_POLL_DAY) and
    `error` (final polls → result), so opinion shifts during the campaign
    aren't conflated with how far off the final polls were. `start` is None
    when the candidate wasn't polled around `start_days`.
    """
    election_date = ELECTION_DATES[year]
    results = FIRST_ROUND_RESULTS.get(year, {})
    if candidates is None:
        candidates = select_chart_candidates(year, polls)
    end_day = FINAL_POLL_DAY if end_days is None else end_days
    rows = []
    for c in candidates:
        start = get_smoothed_value_at_relative_day(polls, c, election_date, start_days)
        end = get_smoothed_value_at_relative_day(polls, c, election_date, end_day)
        result = results.get(c) if end_days is None else None
        rows.append({
            "candidate": c,
            "start": start,
            "end": end,
            "result": result,
            "movement": None if start is None or end is None else end - start,
            "error": None if end is None or result is None else result - end,
            "total": None if start is None or result is None else result - start,
        })
    schema = {"candidate": pl.Utf8, **{k: pl.Float64 for k in ("start", "end", "result", "movement", "error", "total")}}
    return pl.DataFrame(rows, schema=schema)


def final_polling_errors(all_election_data: dict[int, pl.DataFrame]) -> pl.DataFrame:
    """Final smoothed polls vs actual first-round result, all elections."""
    frames = []
    for year in sorted(all_election_data):
        polls = all_election_data[year]
        changes = compute_polling_changes(year, polls, FINAL_POLL_DAY, None)
        frames.append(changes.select(
            pl.lit(year).alias("year"), "candidate",
            pl.col("end").alias("final_poll"), "result", "error",
        ))
    return pl.concat(frames) if frames else pl.DataFrame()


# --- Charting --------------------------------------------------------------

_INK = "#1f1f1f"
_INK_MUTED = "#5f5e5a"
_MOVEMENT_COLOR = "#2a78d6"
_ERROR_COLOR = "#eb6834"


def _spread_labels(values: list[float], min_gap: float) -> list[float]:
    """Nudge label positions apart by at least `min_gap`, staying as close
    to their anchors as possible. Returns positions in input order."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    # Recentre so labels are displaced both up and down, not only upward.
    shift = (pos - np.array(values, dtype=float)[order]).mean()
    pos -= shift
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + min_gap)
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()


def create_election_chart(
    year: int,
    polls_data: pl.DataFrame,
    candidates: list[str],
    round_type: str = "first",
    time_cutoff: str | None = None,
) -> tuple[Any, Any] | None:
    """Scatter of individual polls plus a time-smoothed trend per candidate,
    with the actual result marked and labeled on election day."""
    if polls_data.height == 0 or not candidates:
        print(f"Skipping {year} - insufficient data")
        return None

    election_date = get_round_date(year, round_type)
    actual_results = get_round_results(year, round_type)
    window = filter_by_time_cutoff(polls_data, time_cutoff, election_date)
    if window.height == 0:
        print(f"Skipping {year} - no polls within {time_cutoff} of election")
        return None

    first_day = (window["date"].min() - election_date).days
    grid = np.arange(first_day, FINAL_POLL_DAY + 1)
    grid_dates = np.array([election_date + timedelta(days=int(d)) for d in grid])

    fig, ax = plt.subplots(figsize=(14, 8))
    labels: list[tuple[str, float, str]] = []
    for candidate in candidates:
        if candidate not in window.columns:
            continue
        color = get_candidate_color(candidate)
        sub = window.filter(pl.col(candidate).is_not_null())
        if sub.height == 0:
            continue
        # Marker area tracks poll weight so overlapping tracker releases
        # don't visually outweigh independent polls.
        ax.scatter(sub["date"].to_numpy(), sub[candidate].to_numpy(), color=color,
                   alpha=0.25, s=12 + 38 * sub["weight"].to_numpy(), linewidths=0)

        days, values, weights = candidate_series(polls_data, candidate, election_date)
        trend = smooth_polls(days, values, weights, grid)
        ax.plot(grid_dates, trend, color=color, linewidth=2.2, label=candidate)

        if candidate in actual_results:
            result_value = actual_results[candidate]
            ax.scatter([election_date], [result_value], color=color, s=90, zorder=10,
                       edgecolors="white", linewidths=1.5, clip_on=False)
            labels.append((candidate, result_value, color))

    if labels:
        ymin, ymax = ax.get_ylim()
        spread = _spread_labels([v for _, v, _ in labels], min_gap=0.028 * (ymax - ymin))
        label_x = mdates.date2num(election_date) + 9
        for (candidate, value, color), y in zip(labels, spread):
            ax.annotate(
                f"{candidate} {value:.1f}%", xy=(election_date, value), xytext=(label_x, y),
                textcoords="data", ha="left", va="center", fontsize=9, color=_INK,
                arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8, shrinkA=0, shrinkB=4),
                annotation_clip=False,
            )
        ax.set_xlim(right=mdates.date2num(election_date) + 75)

    ax.axvline(election_date, color=_INK_MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date (fieldwork midpoint)", fontsize=11, color=_INK_MUTED)
    ax.set_ylabel("Voting intention (%)", fontsize=11, color=_INK_MUTED)
    cutoff_text = f" (last {time_cutoff})" if time_cutoff else ""
    round_text = "first round" if round_type == "first" else "second round"
    ax.set_title(f"French presidential election {year} — {round_text} polling{cutoff_text}",
                 fontsize=14, fontweight="bold", color=_INK, loc="left")
    # Below the plot: every line is also direct-labeled at election day, and
    # a legend inside the axes would cover early-campaign polls.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9,
              ncols=min(8, len(labels) or 1), frameon=False)
    ax.grid(True, alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.show()

    print(f"Chart created for {year} {round_type.upper()} ROUND ({window.height} polls in window)")
    return fig, ax


def create_relative_day_charts(
    all_election_data: dict[int, pl.DataFrame],
    start_days: int,
    end_days: int | None,
) -> None:
    """Bar charts of change in smoothed support between two relative days.

    With `end_days=None` each candidate gets two bars — campaign movement
    (start → final polls) and final polling error (final polls → actual
    result) — plus a marker for the net change.
    """
    if not all_election_data:
        print("⚠️ Error: No election data available.")
        return

    end_label = "Actual Result" if end_days is None else f"Day {end_days}"
    years = sorted(all_election_data.keys())
    fig, axes = plt.subplots(len(years), 1, figsize=(14, 4.2 * len(years)), squeeze=False)

    for ax, year in zip(axes[:, 0], years):
        polls = all_election_data[year]
        if polls.height == 0:
            continue
        changes = compute_polling_changes(year, polls, start_days, end_days)
        if changes.height == 0:
            print(f"⚠️ No data for {year}")
            continue

        names = changes["candidate"].to_list()
        x = np.arange(len(names))
        not_polled = changes.filter(pl.col("start").is_null())["candidate"].to_list()

        def bars(offset: float, width: float, col: str, color: str, label: str) -> None:
            vals = changes[col].to_list()
            heights = [v if v is not None else 0.0 for v in vals]
            rects = ax.bar(x + offset, heights, width, color=color, label=label)
            for rect, v in zip(rects, vals):
                if v is None:
                    continue
                ax.annotate(f"{round(v, 1) + 0.0:+.1f}", (rect.get_x() + rect.get_width() / 2, v),
                            xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=_INK)

        if end_days is None:
            bars(-0.2, 0.38, "movement", _MOVEMENT_COLOR, f"Poll movement (day {start_days} → day {FINAL_POLL_DAY})")
            bars(0.2, 0.38, "error", _ERROR_COLOR, f"Final polling error (day {FINAL_POLL_DAY} → result)")
            totals = changes["total"].to_list()
            tx = [xi for xi, t in zip(x, totals) if t is not None]
            ty = [t for t in totals if t is not None]
            ax.scatter(tx, ty, marker="D", s=28, color=_INK, zorder=5, label="Net change (start → result)")
        else:
            bars(0.0, 0.6, "movement", _MOVEMENT_COLOR, f"Poll movement (day {start_days} → day {end_days})")

        ax.axhline(0, color=_INK, linewidth=0.8)
        ax.set_xticks(x, names, rotation=30, ha="right")
        ax.set_ylabel("Change (percentage points)", fontsize=10, color=_INK_MUTED)
        ax.set_title(f"Election {year} - Polling Change from Day {start_days} to {end_label}",
                     fontsize=12, fontweight="bold", color=_INK, loc="left")
        ax.grid(True, alpha=0.25, axis="y")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        if not_polled:
            ax.text(0.01, 0.03, f"No polls near day {start_days}: {', '.join(not_polled)}",
                    transform=ax.transAxes, fontsize=8, color=_INK_MUTED, va="bottom")

    plt.tight_layout()
    plt.show()

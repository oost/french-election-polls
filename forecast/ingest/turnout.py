"""Turnout information on the pinned presidential pages.

Two things the long poll table leaves out:

* pollsters' **abstention estimates** (the "Abs." / "Abstention" column of the
  first- and second-round tables), one row per poll;
* **vote-transfer cross-tabs** (tables under "By first round vote"): how each
  first-round electorate says it will split in the runoff between the two
  finalists and not voting.

Both feed the turnout model (forecast/simulate/turnout.py). They are written
to data/processed/ by `forecast ingest`.
"""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import polars as pl

import polls_helpers as ph
from forecast import paths
from forecast.ingest.pages import load_page_html, load_page_specs, select_tables
from forecast.ingest.registry import CandidateRegistry, PollsterRegistry
from forecast.ingest.wiki_long import election_dates, parse_share

ABSTENTION_POLLS = paths.PROCESSED / "abstention_polls.parquet"
TRANSFER_POLLS = paths.PROCESSED / "transfer_polls.parquet"

_ABSTENTION = re.compile(r"^(abs\.?|abstention)\b", re.I)
_TRANSFER_SECTION = re.compile(r"by first[- ]round vote", re.I)
_TRANSFER_HEADER = re.compile(r"^(?P<source>.+?)\s*\((?P<share>[\d.]+)\s*% in the first round\)\s*(?P<target>.+)$")
_NO_VOTE = re.compile(r"^(no vote|abs|abstention|none|blank|undecided)", re.I)

ABSTENTION_SCHEMA = {
    "year": pl.Int64, "round": pl.Utf8, "pollster": pl.Utf8, "fieldwork_start": pl.Date, "fieldwork_end": pl.Date,
    "publish_est": pl.Date, "days_out": pl.Int64, "matchup": pl.Utf8, "abstention": pl.Float64,
    "source_page": pl.Utf8, "source_table": pl.Int64,
}
TRANSFER_SCHEMA = {
    "year": pl.Int64, "pollster": pl.Utf8, "fieldwork_end": pl.Date, "days_out": pl.Int64,
    "source": pl.Utf8, "target": pl.Utf8, "share": pl.Float64, "source_page": pl.Utf8,
}


def _poll_rows(table, firm_col, date_col, cutoff):
    """(row index, pollster text, fieldwork) of the genuine poll rows before `cutoff`."""
    other = [c for c in table.columns if c not in (date_col, firm_col)]
    for idx in range(len(table)):
        raw_firm = table[firm_col].iloc[idx]
        fieldwork = ph.parse_fieldwork(table[date_col].iloc[idx])
        if fieldwork is None or ph.is_event_row(raw_firm, table.iloc[idx][other].tolist()):
            continue
        firm = ph.clean_pollster(raw_firm)
        if ph._RESULT_ROW.search(firm) or re.search(r"ministry", firm, re.I) or fieldwork[1] >= cutoff:
            continue
        yield idx, firm, fieldwork


def build_abstention_polls(keys: list[str] | None = None, candidates: CandidateRegistry | None = None,
                           pollsters: PollsterRegistry | None = None) -> pl.DataFrame:
    """One row per poll and round with the pollster's abstention estimate (%).

    `matchup` is the pair the table polls, for second-round tables (a poll
    lists one abstention figure per matchup). `days_out` counts to the round
    the estimate is for."""
    specs = load_page_specs()
    candidates = candidates or CandidateRegistry.load()
    pollsters = pollsters or PollsterRegistry.load()
    rows: list[dict[str, Any]] = []
    for key in keys or [k for k, s in specs.items() if s.election_type == "pres"]:
        spec = specs[key]
        t1, t2 = election_dates(spec.year)
        for sel in select_tables(spec, load_page_html(spec)):
            if sel.rule.kind not in ("lineup", "matchup"):
                continue
            table = sel.table
            date_col = ph._find_date_column(table.columns)
            firm_col = ph._find_column(table.columns, "polling firm", "polling organisation", "pollster")
            abs_col = next((c for c in table.columns if _ABSTENTION.search(ph._column_label(c).strip())), None)
            if date_col is None or firm_col is None or abs_col is None:
                continue
            day0 = t1 if sel.rule.round == "first" else t2
            matchup = None
            if sel.rule.kind == "matchup":
                ids = {candidates.match_header(spec.year, ph._column_label(c)) for c in table.columns
                       if c not in (date_col, firm_col, abs_col)}
                ids.discard(None)
                matchup = "|".join(sorted(ids)) if len(ids) == 2 else None
            for idx, firm, (start, end) in _poll_rows(table, firm_col, date_col, day0):
                value, _ = parse_share(table[abs_col].iloc[idx])
                if value is None or not 3 <= value <= 70:
                    continue
                info = pollsters.canonical(firm)
                lag = pollsters.meta(info.firm, spec.year).publication_lag_days
                rows.append({"year": spec.year, "round": sel.rule.round, "pollster": info.firm,
                             "fieldwork_start": start, "fieldwork_end": end,
                             "publish_est": end + timedelta(days=lag), "days_out": (day0 - end).days,
                             "matchup": matchup, "abstention": value, "source_page": spec.key,
                             "source_table": sel.index})
    df = pl.DataFrame(rows, schema=ABSTENTION_SCHEMA)
    # a poll repeats its figure on every hypothetical-lineup row
    return df.unique(subset=["year", "round", "pollster", "fieldwork_start", "fieldwork_end", "matchup", "abstention"],
                     keep="first", maintain_order=True)


def build_transfer_polls(keys: list[str] | None = None, candidates: CandidateRegistry | None = None,
                         pollsters: PollsterRegistry | None = None) -> pl.DataFrame:
    """One row per poll × first-round electorate × runoff choice (share in %;
    `target` is a candidate id or "NONE" for not voting)."""
    specs = load_page_specs()
    candidates = candidates or CandidateRegistry.load()
    pollsters = pollsters or PollsterRegistry.load()
    rows: list[dict[str, Any]] = []
    for key in keys or [k for k, s in specs.items() if s.election_type == "pres"]:
        spec = specs[key]
        _, t2 = election_dates(spec.year)
        for path, table in ph.parse_polling_tables(load_page_html(spec)):
            if not _TRANSFER_SECTION.search(" › ".join(path)) or not path[0].lower().startswith("second"):
                continue
            date_col = ph._find_date_column(table.columns)
            firm_col = ph._find_column(table.columns, "polling firm", "polling organisation", "pollster")
            if date_col is None or firm_col is None:
                continue
            cols: dict[Any, tuple[str, str]] = {}
            for c in table.columns:
                m = _TRANSFER_HEADER.match(ph._column_label(c).strip())
                if not m:
                    continue
                source = candidates.match_header(spec.year, m["source"])
                target = "NONE" if _NO_VOTE.search(m["target"].strip()) else candidates.match_header(spec.year, m["target"])
                if source and target:
                    cols[c] = (source, target)
            for idx, firm, (_, end) in _poll_rows(table, firm_col, date_col, t2):
                info = pollsters.canonical(firm)
                for c, (source, target) in cols.items():
                    share, _ = parse_share(table[c].iloc[idx])
                    if share is not None:
                        rows.append({"year": spec.year, "pollster": info.firm, "fieldwork_end": end,
                                     "days_out": (t2 - end).days, "source": source, "target": target,
                                     "share": share, "source_page": spec.key})
    return pl.DataFrame(rows, schema=TRANSFER_SCHEMA)


def write_turnout_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    paths.ensure_dirs()
    abstention, transfers = build_abstention_polls(), build_transfer_polls()
    abstention.write_parquet(ABSTENTION_POLLS)
    transfers.write_parquet(TRANSFER_POLLS)
    return abstention, transfers


def load_abstention_polls(rebuild: bool = False) -> pl.DataFrame:
    if rebuild or not ABSTENTION_POLLS.exists():
        write_turnout_tables()
    return pl.read_parquet(ABSTENTION_POLLS)


def load_transfer_polls(rebuild: bool = False) -> pl.DataFrame:
    if rebuild or not TRANSFER_POLLS.exists():
        write_turnout_tables()
    return pl.read_parquet(TRANSFER_POLLS)

"""Wikipedia polling tables → the long poll table (one row per poll row × option).

Unlike `polls_helpers.process_year_data`, which collapses each poll to the
lineup closest to the real ballot, this keeps every lineup a poll tested and
every runoff matchup, because the forecast needs them all (scenario mixture,
substitution model, house effects).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


import polars as pl

import polls_helpers as ph
from forecast.ingest.pages import PageSpec, SelectedTable, load_page_html, load_page_specs, select_tables
from forecast.ingest.registry import CandidateRegistry, PollsterRegistry

# Columns that are neither metadata nor an option.
_IGNORED_LABEL = re.compile(
    r"^(abs\.?|abstention|lead|others?|no vote|undecided|none( of these)?|don'?t know|blank|turnout)\b", re.I
)
_LT_CELL = re.compile(r"^\s*<\s*([\d.]+)")

LONG_SCHEMA: dict[str, Any] = {
    "election_id": pl.Utf8, "year": pl.Int64, "round": pl.Utf8, "kind": pl.Utf8,
    "poll_id": pl.Utf8, "row_id": pl.Utf8, "lineup_id": pl.Utf8, "geography": pl.Utf8,
    "pollster_raw": pl.Utf8, "pollster": pl.Utf8, "sponsor": pl.Utf8, "sponsor_bloc": pl.Utf8, "partisan": pl.Boolean,
    "fieldwork_start": pl.Date, "fieldwork_end": pl.Date, "midpoint": pl.Date, "publish_est": pl.Date,
    "sample_size": pl.Int64, "n_eff": pl.Float64, "population": pl.Utf8, "mode": pl.Utf8,
    "overlap_weight": pl.Float64, "n_lineups": pl.Int64,
    "option_id": pl.Utf8, "option_kind": pl.Utf8, "bloc": pl.Utf8, "label_raw": pl.Utf8,
    "share": pl.Float64, "share_lt": pl.Boolean,
    "source_page": pl.Utf8, "source_revision": pl.Int64, "source_table": pl.Int64, "source_row": pl.Int64,
    "section_path": pl.Utf8,
}


def election_dates(year: int) -> tuple[date, date]:
    """(first round, runoff) dates."""
    if year in ph.ELECTION_DATES:
        return ph.ELECTION_DATES[year], ph.SECOND_ROUND_DATES[year]
    if year == ph.NEXT_ELECTION_DATE.year:
        return ph.NEXT_ELECTION_DATE, ph.NEXT_ELECTION_DATE + timedelta(days=14)
    raise KeyError(f"no election dates for {year}")


@dataclass
class IngestReport:
    unmatched_headers: dict[str, int] = field(default_factory=dict)
    rows_by_reason: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.rows_by_reason[reason] = self.rows_by_reason.get(reason, 0) + 1


def parse_share(value: Any) -> tuple[float | None, bool]:
    """Cell → (share, was "<x"). "<1%" becomes 0.5 (half the bound)."""
    if isinstance(value, str):
        m = _LT_CELL.match(value)
        if m:
            return float(m.group(1)) / 2, True
    return ph._extract_numeric(value), False


def _hash(*parts: Any, n: int = 12) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:n]


def lineup_key(option_ids: list[str]) -> str:
    return _hash(*sorted(option_ids), n=10)


def _generic_option(label: str) -> str | None:
    m = re.match(r"^\s*generic\s+(.+?)\s*$", label, re.I)
    return f"GEN_{re.sub(r'[^A-Z0-9]+', '_', m.group(1).upper()).strip('_')}" if m else None


def _valid(kind: str, values: list[float]) -> bool:
    if kind == "matchup":
        return len(values) == 2 and 95 <= sum(values) <= 105
    return len(values) >= 3 and 50 <= sum(values) <= 105 and max(values) <= 60


def extract_table(
    sel: SelectedTable,
    candidates: CandidateRegistry,
    pollsters: PollsterRegistry,
    report: IngestReport,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(long rows, event rows) for one selected table."""
    spec, table, kind = sel.page, sel.table, sel.rule.kind
    year = spec.year
    t1, t2 = election_dates(year)
    cutoff = t1 if kind in ("lineup", "generic") else t2
    bloc_of = candidates.bloc_of(year)

    date_col = ph._find_date_column(table.columns)
    firm_col = ph._find_column(table.columns, "polling firm", "polling organisation", "pollster")
    sample_col = ph._find_column(table.columns, "sample")
    if date_col is None or firm_col is None:
        return [], []

    options: dict[Any, tuple[str, str]] = {}   # column → (option_id, label)
    for col in table.columns:
        if col in (date_col, firm_col, sample_col):
            continue
        label = ph._column_label(col)
        if _IGNORED_LABEL.search(label.strip()):
            continue
        oid = _generic_option(label) if kind == "generic" else candidates.match_header(year, label)
        if oid is None:
            report.unmatched_headers[f"{spec.key}: {label}"] = report.unmatched_headers.get(f"{spec.key}: {label}", 0) + 1
            continue
        if oid not in {o for o, _ in options.values()}:
            options[col] = (oid, label)

    other_cols = [c for c in table.columns if c not in (date_col, firm_col)]
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for idx in range(len(table)):
        raw_firm = table[firm_col].iloc[idx]
        fieldwork = ph.parse_fieldwork(table[date_col].iloc[idx])
        if ph.is_event_row(raw_firm, table.iloc[idx][other_cols].tolist()):
            texts = [str(v) for v in table.iloc[idx][other_cols].tolist() if not ph._is_blank_cell(v)]
            events.append({"election_id": spec.election, "date": fieldwork[1] if fieldwork else None,
                           "text": max(set(texts), key=texts.count), "source_page": spec.key,
                           "source_revision": spec.revision, "source_table": sel.index})
            report.skip("event row")
            continue
        if fieldwork is None:
            report.skip("unparseable fieldwork")
            continue
        firm_text = ph.clean_pollster(raw_firm)
        if ph._RESULT_ROW.search(firm_text) or re.search(r"ministry", firm_text, re.I):
            report.skip("result row")
            continue
        start, end = fieldwork
        if end >= cutoff:
            report.skip("fieldwork on/after election day")
            continue

        values: dict[str, tuple[float, bool, str]] = {}
        for col, (oid, label) in options.items():
            share, lt = parse_share(table[col].iloc[idx])
            if share is not None:
                values[oid] = (share, lt, label)
        if not _valid(kind, [v[0] for v in values.values()]):
            report.skip("implausible or incomplete row")
            continue

        info = pollsters.canonical(firm_text)
        meta = pollsters.meta(info.firm, year)
        if meta.exclude:
            report.skip("excluded firm")
            continue
        n = ph._parse_sample_size(table[sample_col].iloc[idx]) if sample_col is not None else None
        poll_id = _hash(info.firm, firm_text, start, end, n, spec.key)
        option_ids = sorted(values)
        lineup_id = "|".join(option_ids) if kind == "matchup" else lineup_key(option_ids)
        base = {
            "election_id": f"{spec.election}_t1" if sel.rule.round == "first" else f"{spec.election}_t2",
            "year": year, "round": sel.rule.round, "kind": kind,
            "poll_id": poll_id, "row_id": f"{poll_id}:{sel.index}:{idx}", "lineup_id": lineup_id,
            "geography": "national",
            "pollster_raw": firm_text, "pollster": info.firm, "sponsor": info.sponsor,
            "sponsor_bloc": info.sponsor_bloc, "partisan": info.sponsor is not None,
            "fieldwork_start": start, "fieldwork_end": end, "midpoint": start + (end - start) / 2,
            "publish_est": end + timedelta(days=meta.publication_lag_days),
            "sample_size": n, "population": meta.population, "mode": meta.mode,
            "source_page": spec.key, "source_revision": spec.revision,
            "source_table": sel.index, "source_row": idx, "section_path": " › ".join(sel.section_path),
        }
        for oid, (share, lt, label) in values.items():
            rows.append({
                **base, "option_id": oid, "option_kind": "bloc" if kind == "generic" else "candidate",
                "bloc": oid.removeprefix("GEN_") if kind == "generic" else bloc_of.get(oid),
                "label_raw": label, "share": share, "share_lt": lt,
            })
    return rows, events


def drop_restricted_lineups(df: pl.DataFrame, minor_pct: float = 2.0) -> pl.DataFrame:
    """The post's "use the multi-way ballot": within one poll, drop a lineup
    whose options are a strict subset of another lineup's when every extra
    option in the larger lineup is minor (< `minor_pct`)."""
    lineups = df.filter(pl.col("kind") == "lineup")
    if lineups.is_empty():
        return df
    drop: set[str] = set()
    for _, grp in lineups.group_by("poll_id"):
        rows = {rid: dict(zip(g["option_id"].to_list(), g["share"].to_list()))
                for (rid,), g in grp.group_by("row_id")}
        for a, va in rows.items():
            for b, vb in rows.items():
                if a == b or not set(va) < set(vb):
                    continue
                extras = set(vb) - set(va)
                if all(vb[o] < minor_pct for o in extras):
                    drop.add(a)
    return df.filter(~pl.col("row_id").is_in(list(drop)))


def add_poll_weights(df: pl.DataFrame) -> pl.DataFrame:
    """overlap_weight (rolling trackers, via `compute_poll_weights` on the raw
    pollster label so a tracker isn't merged with the firm's one-off polls),
    n_lineups and n_eff."""
    if df.is_empty():
        return df.with_columns(pl.lit(None, pl.Float64).alias("overlap_weight"),
                               pl.lit(None, pl.Int64).alias("n_lineups"), pl.lit(None, pl.Float64).alias("n_eff"))
    polls = df.select("election_id", "kind", "poll_id", "pollster_raw", "fieldwork_start", "fieldwork_end",
                      "sample_size").unique(subset=["election_id", "kind", "poll_id"])
    parts = []
    for _, grp in polls.group_by("election_id", "kind"):
        w = ph.compute_poll_weights(grp["fieldwork_start"].to_list(), grp["fieldwork_end"].to_list(),
                                    grp["pollster_raw"].to_list())
        parts.append(grp.with_columns(pl.Series("overlap_weight", w)))
    weights = pl.concat(parts).select("election_id", "kind", "poll_id", "overlap_weight")
    n_lineups = df.group_by("election_id", "kind", "poll_id").agg(pl.col("row_id").n_unique().alias("n_lineups"))
    return (
        df.drop("overlap_weight", "n_lineups", "n_eff", strict=False)
        .join(weights, on=["election_id", "kind", "poll_id"], how="left")
        .join(n_lineups, on=["election_id", "kind", "poll_id"], how="left")
        .with_columns((pl.col("sample_size").cast(pl.Float64) * pl.col("overlap_weight")).alias("n_eff"))
    )


def build_page(
    spec: PageSpec,
    candidates: CandidateRegistry,
    pollsters: PollsterRegistry,
    html: str | None = None,
    report: IngestReport | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(long table, events) for one page."""
    report = report or IngestReport()
    html = html if html is not None else load_page_html(spec)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for sel in select_tables(spec, html):
        r, e = extract_table(sel, candidates, pollsters, report)
        rows.extend(r)
        events.extend(e)
    long = pl.DataFrame(rows, schema={k: v for k, v in LONG_SCHEMA.items()
                                      if k not in ("overlap_weight", "n_lineups", "n_eff")})
    # A poll can repeat a lineup verbatim across tables (e.g. period tables
    # that overlap); keep one copy.
    long = long.unique(subset=["poll_id", "kind", "lineup_id", "option_id", "share"], keep="first", maintain_order=True)
    long = add_poll_weights(drop_restricted_lineups(long))
    ev = pl.DataFrame(events, schema={"election_id": pl.Utf8, "date": pl.Date, "text": pl.Utf8,
                                      "source_page": pl.Utf8, "source_revision": pl.Int64, "source_table": pl.Int64})
    return long.select(list(LONG_SCHEMA)), ev


def build_presidential(
    keys: list[str] | None = None,
    candidates: CandidateRegistry | None = None,
    pollsters: PollsterRegistry | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, IngestReport]:
    specs = load_page_specs()
    candidates = candidates or CandidateRegistry.load()
    pollsters = pollsters or PollsterRegistry.load()
    report = IngestReport()
    keys = keys or [k for k, s in specs.items() if s.election_type == "pres"]
    longs, events = [], []
    for key in keys:
        long, ev = build_page(specs[key], candidates, pollsters, report=report)
        longs.append(long)
        events.append(ev)
    return pl.concat(longs), pl.concat(events), report


def lineups_table(long: pl.DataFrame) -> pl.DataFrame:
    """One row per (election, kind, lineup): its options and poll count."""
    return (
        long.group_by("election_id", "kind", "lineup_id")
        .agg(
            pl.col("option_id").unique().sort().alias("options"),
            pl.col("poll_id").n_unique().alias("n_polls"),
            pl.col("midpoint").min().alias("first_seen"),
            pl.col("midpoint").max().alias("last_seen"),
        )
        .sort("election_id", "kind", "n_polls", descending=[False, False, True])
    )


def wide_rows(long: pl.DataFrame) -> pl.DataFrame:
    """One row per poll row with option shares as columns (handy for checks)."""
    meta = ["election_id", "kind", "poll_id", "row_id", "lineup_id", "pollster", "pollster_raw",
            "fieldwork_start", "fieldwork_end", "midpoint", "sample_size", "overlap_weight", "n_lineups"]
    return long.pivot(on="option_id", index=meta, values="share", aggregate_function="first")


def collapse_to_ballot(long: pl.DataFrame, candidates: CandidateRegistry, year: int) -> pl.DataFrame:
    """Reproduce `polls_helpers.process_year_data`'s one-row-per-poll table from
    the long table: actual-ballot candidates only, lineup closest to the ballot."""
    ballot = candidates.ballot(year)
    ids = {c.candidate_id: c.surname for c in ballot}
    sub = long.filter((pl.col("year") == year) & (pl.col("kind") == "lineup") & pl.col("option_id").is_in(list(ids)))
    records = []
    for (rid,), g in sub.group_by("row_id", maintain_order=True):
        first = g.row(0, named=True)
        vals = {ids[o]: s for o, s in zip(g["option_id"].to_list(), g["share"].to_list())}
        if not ph.is_valid_first_round_row(list(vals.values())):
            continue
        records.append({
            "date": first["midpoint"], "fieldwork_start": first["fieldwork_start"],
            "fieldwork_end": first["fieldwork_end"], "pollster": first["pollster_raw"],
            "sample_size": first["sample_size"], **vals,
        })
    # infer_schema_length=None: minor candidates first appear late in the cycle
    return pl.DataFrame(ph.collapse_scenarios(records, [c.surname for c in ballot]), infer_schema_length=None)

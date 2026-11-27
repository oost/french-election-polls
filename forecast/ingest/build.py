"""Write the processed poll tables (data/processed/)."""
from __future__ import annotations

import polars as pl

from forecast import paths
from forecast.ingest.wiki_long import IngestReport, build_presidential, lineups_table

POLLS_LONG = paths.PROCESSED / "polls_long.parquet"
LINEUPS = paths.PROCESSED / "lineups.parquet"
EVENTS = paths.PROCESSED / "events.parquet"


def write_presidential() -> tuple[pl.DataFrame, pl.DataFrame, IngestReport]:
    paths.ensure_dirs()
    long, events, report = build_presidential()
    long.write_parquet(POLLS_LONG)
    lineups_table(long).write_parquet(LINEUPS)
    events.write_parquet(EVENTS)
    from forecast.ingest.turnout import write_turnout_tables
    write_turnout_tables()      # abstention estimates and vote-transfer cross-tabs, for the turnout model
    return long, events, report


def load_polls_long(rebuild: bool = False) -> pl.DataFrame:
    """The long poll table, rebuilt from the cached pages if missing."""
    if rebuild or not POLLS_LONG.exists():
        write_presidential()
    return pl.read_parquet(POLLS_LONG)

"""Command line: python -m forecast <command> [options]

    ingest     parse the pinned Wikipedia pages → data/processed/polls_long.parquet
    average    polling averages for one election (writes data/outputs/)
    run        simulate an election (writes data/outputs/)
    backtest   out-of-sample presidential backtest (writes data/archive/, data/outputs/)
    legi       legislative forecast for the next election (scenarios_legi.yaml)
    legi-backtest  out-of-sample legislative backtest, 2017–2024
"""
from __future__ import annotations

import argparse
import sys
from datetime import date


def _cmd_ingest(args: argparse.Namespace) -> int:
    from forecast.ingest.build import write_presidential
    long, events, report = write_presidential()
    print(f"polls_long: {long.height} rows, {long['poll_id'].n_unique()} polls")
    print(f"events: {events.height}; skipped rows: {report.rows_by_reason}")
    if report.unmatched_headers:
        print("unmatched headers:", ", ".join(sorted(report.unmatched_headers)))
    return 0


def _cmd_average(args: argparse.Namespace) -> int:
    from forecast.presidential.pipeline import write_averages
    path = write_averages(args.year, as_of=args.as_of)
    print(f"wrote {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from forecast.presidential.pipeline import run_forecast
    summary = run_forecast(args.year, as_of=args.as_of, n_sims=args.sims, write=True)
    print(summary.report())
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from forecast.backtest.report import write_report
    from forecast.backtest.harness import run_presidential_backtest
    archive = run_presidential_backtest(args.cycles, n_sims=args.sims)
    print(write_report(archive))
    return 0


def _cmd_legi(args: argparse.Namespace) -> int:
    from forecast.legislative.pipeline import run_legislative_forecast
    summary = run_legislative_forecast(as_of=args.as_of, n_sims=args.sims, write=True)
    print(summary.report())
    return 0


def _cmd_legi_backtest(args: argparse.Namespace) -> int:
    import polars as pl
    from forecast.legislative.pipeline import run_legislative_backtest
    _, totals = run_legislative_backtest(n_sims=args.sims or 4000)
    with pl.Config(tbl_rows=100, tbl_cols=12):
        print(totals)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m forecast", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="build the long poll table").set_defaults(func=_cmd_ingest)

    p = sub.add_parser("average", help="polling averages")
    p.add_argument("--year", type=int, default=2027)
    p.add_argument("--as-of", type=date.fromisoformat, default=None)
    p.set_defaults(func=_cmd_average)

    p = sub.add_parser("run", help="simulate an election")
    p.add_argument("--year", type=int, default=2027)
    p.add_argument("--as-of", type=date.fromisoformat, default=None)
    p.add_argument("--sims", type=int, default=None)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("backtest", help="out-of-sample backtest")
    p.add_argument("--cycles", type=int, nargs="+", default=[2007, 2012, 2017, 2022])
    p.add_argument("--sims", type=int, default=None)
    p.set_defaults(func=_cmd_backtest)

    p = sub.add_parser("legi", help="legislative forecast")
    p.add_argument("--as-of", type=date.fromisoformat, default=None)
    p.add_argument("--sims", type=int, default=None)
    p.set_defaults(func=_cmd_legi)

    p = sub.add_parser("legi-backtest", help="legislative backtest")
    p.add_argument("--sims", type=int, default=None)
    p.set_defaults(func=_cmd_legi_backtest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

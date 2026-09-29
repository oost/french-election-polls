"""Text + chart report of a backtest archive (data/outputs/backtest_pres/)."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from forecast import paths
from forecast.backtest import charts
from forecast.backtest.metrics import calibration, coverage, scores, share_errors


def write_report(archive: pl.DataFrame, out_dir: Path | None = None) -> str:
    out_dir = out_dir or paths.OUTPUTS / "backtest_pres"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts.write_all(archive, out_dir)
    with pl.Config(tbl_rows=30, tbl_width_chars=120, float_precision=3):
        parts = [
            f"Presidential backtest: cycles {sorted(archive['cycle'].unique().to_list())}, "
            f"{archive.select(pl.struct('cycle', 'days_out').n_unique()).item()} snapshots, {archive.height} rows",
            "", "Mean absolute error (points) by horizon:", str(share_errors(archive)),
            "", "Interval coverage (simulated first-round shares):", str(coverage(archive)),
            "", "Calibration — P(reach runoff):", str(calibration(archive, "p_runoff", "made_runoff")),
            "", "Calibration — P(win):", str(calibration(archive, "p_win", "won")),
            "", "Scores:", "\n".join(f"  {k}: {v:.3f}" for k, v in scores(archive).items()),
        ]
    text = "\n".join(parts)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    return text

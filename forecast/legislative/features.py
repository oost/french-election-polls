"""Seat × unit design tables for the legislative now-cast and fundamentals.

Predictors (post § 3.2, French versions):
    lean      log(seat / national) presidential share, 80/20 over two elections
    lagged    log(seat / national) share in the previous legislative first round
    inc       1 if the seat's incumbent stands for this unit, ½ in a first term
    presence  log(577 / seats the unit contests): a unit standing in fewer
              seats concentrates its national vote in them
"""
from __future__ import annotations

import numpy as np
import polars as pl

from forecast.legislative import data as D

PREDICTORS = ("lean", "lagged", "has_lagged", "inc")


def _units_config(year: int) -> bool:
    """True when the left runs split (RAD separate)."""
    return year in D.SPLIT_LEFT


def seat_design(year: int, units_present: pl.DataFrame | None = None, pres_years: tuple[int, ...] | None = None,
                recent_weight: float = 0.8) -> pl.DataFrame:
    """seat × unit rows with predictors. `units_present` (seat, unit) defaults
    to the units that actually stood in `year` (historical years)."""
    split = _units_config(year)
    if units_present is None:
        units_present = D.seat_units_t1(year).select("seat", "unit")
    lean = D.lean(year, split, recent_weight, pres_years) if pres_years or year in D.PRES_FOR_LEGI else None
    df = units_present
    df = df.join(lean, on=["seat", "unit"], how="left") if lean is not None else df.with_columns(pl.lit(0.0).alias("lean"))
    lag = D.lagged(year, split) if year in D.PREVIOUS_LEGI else None
    if lag is not None and not lag.is_empty():
        df = df.join(lag, on=["seat", "unit"], how="left")
    else:
        df = df.with_columns(pl.lit(None, pl.Float64).alias("lagged"))
    inc = D.incumbents(year) if year in D.PREVIOUS_LEGI else None
    if inc is not None and not inc.is_empty():
        inc = inc.filter(pl.col("inc_runs")).select("seat", "unit",
                                                   pl.when(pl.col("first_term")).then(0.5).otherwise(1.0).alias("inc"))
        df = df.join(inc, on=["seat", "unit"], how="left")
    else:
        df = df.with_columns(pl.lit(None, pl.Float64).alias("inc"))
    n_present = df.group_by("unit").agg(pl.col("seat").n_unique().alias("n_present"))
    return (df.join(n_present, on="unit")
            .with_columns(
                pl.col("lean").fill_null(0.0),
                pl.col("lagged").is_not_null().cast(pl.Float64).alias("has_lagged"),
                pl.col("lagged").fill_null(0.0),
                pl.col("inc").fill_null(0.0),
                (np.log(577) - pl.col("n_present").cast(pl.Float64).log()).alias("presence"),
            )
            .with_columns(pl.lit(year).alias("year"))
            .sort("seat", "unit"))


def with_outcome(design: pl.DataFrame, year: int) -> pl.DataFrame:
    res = D.seat_units_t1(year).select("seat", "unit", "share", "votes", "exprimes", "inscrits")
    return design.join(res, on=["seat", "unit"], how="left")


def national_result(year: int) -> dict[str, float]:
    df = D.seat_units_t1(year)
    tot = df.group_by("unit").agg(pl.col("votes").sum())
    total = df.select("seat", "exprimes").unique()["exprimes"].sum()
    return {u: v / total for u, v in tot.iter_rows()}

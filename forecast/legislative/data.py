"""Seat-level legislative data (PLAN.md › Legislative model, steps 1–2).

Units: EXG, RAD (radical left, only when the left runs split: 2012, 2017),
GAU, CEN, DRO, RN, EXD, DIV. First-round votes of candidates sharing a unit in
a seat are summed (the unit is the forecasting unit; within-unit dissidents
are not modelled separately).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import polars as pl

from forecast import paths
from forecast.ingest.harmonised import seat_results
from forecast.ingest.ministry import load_legi_2024
from forecast.ingest.registry import CandidateRegistry

UNITS = ("EXG", "RAD", "GAU", "CEN", "DRO", "RN", "EXD", "DIV")
LEGI_YEARS = (2012, 2017, 2022, 2024)
SPLIT_LEFT = {2012, 2017}
PREVIOUS_LEGI = {2017: 2012, 2022: 2017, 2024: 2022}
# Presidential elections whose seat results make up the lean (80/20).
PRES_FOR_LEGI = {2012: (2012,), 2017: (2017, 2012), 2022: (2022, 2017), 2024: (2022, 2017)}


def _norm(s: str | None) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).upper().replace("-", " ").strip()


@lru_cache(maxsize=None)
def unit_map(year: int) -> dict[str, str]:
    df = pl.read_csv(paths.REFERENCE / "party_bloc_map.csv")
    return dict(df.filter(pl.col("election") == f"{year}_legi").select("nuance", "unit").iter_rows())


@lru_cache(maxsize=None)
def legi_results(year: int, round_: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(turnout, candidates) with a `unit` column; round_ in {'t1', 't2'}."""
    if year == 2024:
        turnout, votes = load_legi_2024(round_)
    else:
        turnout, votes = seat_results(f"{year}_legi_{round_}")
    umap = unit_map(year)
    missing = sorted(set(votes["nuance"].drop_nulls().unique().to_list()) - set(umap))
    if missing:
        raise KeyError(f"{year}: nuances without a unit in party_bloc_map.csv: {missing}")
    votes = votes.with_columns(pl.col("nuance").replace_strict(umap, default="DIV").alias("unit"))
    return turnout, votes


def seat_units_t1(year: int) -> pl.DataFrame:
    """seat × unit first-round votes and shares (of votes cast)."""
    turnout, votes = legi_results(year, "t1")
    agg = votes.group_by("seat", "unit").agg(pl.col("voix").sum().alias("votes"), pl.len().alias("n_cands"))
    return (agg.join(turnout, on="seat")
            .with_columns((pl.col("votes") / pl.col("exprimes")).alias("share"),
                          (pl.col("votes") / pl.col("inscrits")).alias("share_registered"))
            .with_columns(pl.lit(year).alias("year")))


def seat_winners(year: int) -> pl.DataFrame:
    """seat → winning unit and the winner's name (first round outright or runoff)."""
    t1_turn, t1 = legi_results(year, "t1")
    t2_turn, t2 = legi_results(year, "t2")
    t2_win = t2.sort("voix", descending=True).group_by("seat").first().select("seat", "unit", "nom", "prenom")
    decided_t1 = set(t1["seat"].unique().to_list()) - set(t2["seat"].unique().to_list())
    t1_win = (t1.filter(pl.col("seat").is_in(list(decided_t1))).sort("voix", descending=True)
              .group_by("seat").first().select("seat", "unit", "nom", "prenom"))
    return (pl.concat([t2_win.with_columns(pl.lit("t2").alias("decided")),
                       t1_win.with_columns(pl.lit("t1").alias("decided"))])
            .with_columns(pl.lit(year).alias("year")))


def pres_unit(registry: CandidateRegistry, year: int, candidate_id: str) -> str:
    """Legislative unit of a presidential candidate (split-left definitions)."""
    c = registry.get(year, candidate_id)
    if c.party in ("FN", "RN"):
        return "RN"
    if c.party in ("FG", "FI", "LFI", "PCF"):
        return "RAD"
    return c.bloc


@lru_cache(maxsize=None)
def pres_seat_units(year: int) -> pl.DataFrame:
    """seat × unit presidential first-round shares (split-left units)."""
    registry = CandidateRegistry.load()
    turnout, votes = seat_results(f"{year}_pres_t1")
    by_surname = {_norm(c.surname): c.candidate_id for c in registry.ballot(year)}
    def unit_of(nom: str) -> str:
        n = _norm(nom)
        cid = by_surname.get(n) or next((v for k, v in by_surname.items() if k.endswith(n) or n.endswith(k)), None)
        if cid is None:
            raise KeyError(f"{year}: presidential candidate {nom!r} not in the registry")
        return pres_unit(registry, year, cid)
    names = {n: unit_of(n) for n in votes["nom"].unique().to_list()}
    df = (votes.with_columns(pl.col("nom").replace_strict(names).alias("unit"))
          .group_by("seat", "unit").agg(pl.col("voix").sum())
          .join(turnout.select("seat", "exprimes"), on="seat")
          .with_columns((pl.col("voix") / pl.col("exprimes")).alias("share")))
    return df.select("seat", "unit", "share")


def _merge_left(df: pl.DataFrame, split: bool, share_col: str = "share") -> pl.DataFrame:
    if split:
        return df
    return (df.with_columns(pl.when(pl.col("unit") == "RAD").then(pl.lit("GAU")).otherwise(pl.col("unit")).alias("unit"))
            .group_by([c for c in df.columns if c not in (share_col, "unit")] + ["unit"])
            .agg(pl.col(share_col).sum()))


def national_shares(df: pl.DataFrame, weight: str = "exprimes") -> dict[str, float]:
    tot = df.group_by("unit").agg((pl.col("share") * pl.col(weight)).sum().alias("v"))
    total = df.select("seat", weight).unique()[weight].sum()
    return {u: v / total for u, v in tot.iter_rows()}


def lean(legi_year: int, split: bool, recent_weight: float = 0.8, pres_years: tuple[int, ...] | None = None) -> pl.DataFrame:
    """seat × unit lean: log(seat share / national share) in the last two
    presidential first rounds, blended recent_weight / (1 − recent_weight)."""
    pres_years = pres_years or PRES_FOR_LEGI[legi_year]
    parts = []
    weights = [recent_weight, 1 - recent_weight] if len(pres_years) == 2 else [1.0]
    for w, py in zip(weights, pres_years):
        ps = _merge_left(pres_seat_units(py), split)
        turnout, _ = seat_results(f"{py}_pres_t1")
        nat = national_shares(ps.join(turnout.select("seat", "exprimes"), on="seat"))
        parts.append(ps.with_columns(
            (w * (pl.col("share").clip(1e-3).log() - pl.col("unit").replace_strict(nat).log())).alias("lean_part")))
    out = pl.concat(parts).group_by("seat", "unit").agg(pl.col("lean_part").sum().alias("lean"))
    return out


@dataclass(frozen=True)
class SeatMeta:
    table: pl.DataFrame     # seat, dept, region


@lru_cache(maxsize=None)
def seat_meta() -> pl.DataFrame:
    turnout, _ = legi_results(2024, "t1")
    reg = pl.read_csv(paths.REFERENCE / "departement_region.csv", schema_overrides={"dept": pl.Utf8})
    return (turnout.select("seat").with_columns(pl.col("seat").str.split("-").list.first().alias("dept"))
            .join(reg, on="dept", how="left").sort("seat"))


def incumbents(year: int) -> pl.DataFrame:
    """Previous winner per seat, whether they stand again in `year` (name
    match in the first-round candidates) and under which unit, and whether
    they were in their first term."""
    prev = PREVIOUS_LEGI.get(year)
    if prev is None:
        return pl.DataFrame(schema={"seat": pl.Utf8, "unit": pl.Utf8, "inc_runs": pl.Boolean, "first_term": pl.Boolean})
    win = seat_winners(prev).with_columns(pl.concat_str([pl.col("nom").map_elements(_norm, return_dtype=pl.Utf8),
                                                         pl.col("prenom").map_elements(_norm, return_dtype=pl.Utf8)],
                                                        separator="|").alias("key"))
    _, t1 = legi_results(year, "t1")
    t1 = t1.with_columns(pl.concat_str([pl.col("nom").map_elements(_norm, return_dtype=pl.Utf8),
                                        pl.col("prenom").map_elements(_norm, return_dtype=pl.Utf8)],
                                       separator="|").alias("key"))
    runs = win.join(t1.select("seat", "key", pl.col("unit").alias("unit_now")), on=["seat", "key"], how="left")
    first_term = pl.Series([True] * runs.height)
    if prev in PREVIOUS_LEGI:
        before = seat_winners(PREVIOUS_LEGI[prev]).with_columns(
            pl.concat_str([pl.col("nom").map_elements(_norm, return_dtype=pl.Utf8),
                           pl.col("prenom").map_elements(_norm, return_dtype=pl.Utf8)], separator="|").alias("key"))
        re_elected = set(zip(before["seat"], before["key"]))
        first_term = pl.Series([(s, k) not in re_elected for s, k in zip(runs["seat"], runs["key"])])
    return runs.select(
        "seat", pl.coalesce("unit_now", "unit").alias("unit"),
        pl.col("unit_now").is_not_null().alias("inc_runs"),
    ).with_columns(first_term.alias("first_term"))


def lagged(year: int, split: bool) -> pl.DataFrame:
    """seat × unit: log(previous legislative seat share / previous national share)."""
    prev = PREVIOUS_LEGI.get(year)
    if prev is None:
        return pl.DataFrame(schema={"seat": pl.Utf8, "unit": pl.Utf8, "lagged": pl.Float64})
    df = _merge_left(seat_units_t1(prev).select("seat", "unit", "share", "exprimes"), split)
    nat = national_shares(df)
    return df.select("seat", "unit",
                     (pl.col("share").clip(1e-3).log() - pl.col("unit").replace_strict(nat).log()).alias("lagged"))

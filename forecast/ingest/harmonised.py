"""data.gouv.fr "Données des élections agrégées" → per-constituency results.

Polling-station results for every election since 1999. `general_results`
carries the constituency (département + seat number) for the 2012, 2017 and
2022 presidential and legislative elections only; `candidats_results` has no
seat column and is joined on (id_election, id_brut_miom). 2024 legislative
results come from the Ministry's per-constituency files instead
(`forecast.ingest.ministry`).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from forecast import paths

DATASET_API = "https://www.data.gouv.fr/api/1/datasets/donnees-des-elections-agregees/"
FILES = {
    "general_results.parquet": "Résultats généraux",
    "candidats_results.parquet": "Résultats par candidat",
}
USER_AGENT = "french-elections-forecast/0.1 (research)"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def download(raw_dir: Path = paths.RAW, refresh: bool = False) -> dict[str, Path]:
    """Fetch both parquet files (once) and record them in raw/manifest.json."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    out = {name: raw_dir / name for name in FILES}
    if not refresh and all(p.exists() for p in out.values()):
        return out
    meta = json.loads(_get(DATASET_API))
    for name, title in FILES.items():
        url = next(r["url"] for r in meta["resources"] if r["title"] == title and r["format"] == "parquet")
        data = _get(url)
        out[name].write_bytes(data)
        manifest[name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                          "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return out


def seat_id(dept: pl.Expr, circo: pl.Expr) -> pl.Expr:
    """Canonical seat id "DD-NN" (e.g. "01-05", "2A-01", "ZZ-11")."""
    return pl.concat_str([dept.str.zfill(2), pl.lit("-"), circo.cast(pl.Utf8).str.zfill(2)])


def seat_results(election_id: str, raw_dir: Path = paths.RAW) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(turnout per seat, votes per seat × candidate) for an election with
    constituency codes (2012/2017/2022 pres and legi)."""
    files = download(raw_dir)
    gen = (pl.scan_parquet(files["general_results.parquet"])
           .filter(pl.col("id_election") == election_id)
           .select("id_brut_miom", "code_departement", "code_circonscription", "inscrits", "votants", "exprimes")
           .collect())
    if gen["code_circonscription"].null_count() == gen.height:
        raise ValueError(f"{election_id} has no constituency codes in the harmonised data")
    gen = gen.with_columns(seat_id(pl.col("code_departement"), pl.col("code_circonscription")).alias("seat"))
    turnout = gen.group_by("seat").agg(pl.col("inscrits").sum(), pl.col("votants").sum(), pl.col("exprimes").sum())
    cand = (pl.scan_parquet(files["candidats_results.parquet"])
            .filter(pl.col("id_election") == election_id)
            .select("id_brut_miom", "no_panneau", "voix", "nuance", "nom", "prenom")
            .collect())
    votes = (cand.join(gen.select("id_brut_miom", "seat"), on="id_brut_miom", how="inner")
             .group_by("seat", "no_panneau")
             .agg(pl.col("voix").sum(), pl.col("nuance").first(),
                  pl.col("nom").first(), pl.col("prenom").first()))
    return turnout, votes

"""Ministry of the Interior per-constituency files (2024 legislative).

The harmonised polling-station data has no constituency code for 2024, so the
Ministry's per-constituency result files are used instead. They are wide (one
block of columns per candidate) and use numeric overseas codes; seats are
mapped to the harmonised ids ("DD-NN", overseas "ZA-01" …, abroad "ZZ-NN").
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from forecast import paths

LEGI_2024 = {
    "t1": ("elections-legislatives-des-30-juin-et-7-juillet-2024-resultats-definitifs-du-1er-tour",
           "resultats-definitifs-par-circonscriptions-legislatives.csv"),
    "t2": ("elections-legislatives-des-30-juin-et-7-juillet-2024-resultats-definitifs-du-2nd-tour",
           "resultats-definitifs-par-circonscription.csv"),
}

# Ministry numeric overseas codes → the harmonised Z-codes.
OVERSEAS = {"971": "ZA", "972": "ZB", "973": "ZC", "974": "ZD", "976": "ZM", "975": "ZS",
            "977": "ZX", "986": "ZW", "987": "ZP", "988": "ZN", "99": "ZZ", "ZZ": "ZZ"}


def _download(slug: str, fname: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"legi2024_{fname}"
    if out.exists():
        return out
    ua = {"User-Agent": "french-elections-forecast/0.1 (research)"}
    meta = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://www.data.gouv.fr/api/1/datasets/{slug}/", headers=ua), timeout=120).read())
    url = next(r["url"] for r in meta["resources"] if r["title"] == fname)
    data = urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=300).read()
    out.write_bytes(data)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[out.name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                          "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return out


def seat_from_codes(dept: str, circo: str) -> str:
    dept = str(dept).strip().strip('"')
    circo = str(circo).strip().strip('"')
    num = int(circo[-2:])
    if dept in OVERSEAS:
        d = OVERSEAS[dept]
    elif dept.isdigit():
        d = f"{int(dept):02d}"
    else:
        d = dept.upper()
    return f"{d}-{num:02d}"


def _int(s) -> int:
    return int(re.sub(r"\D", "", str(s)) or 0)


def load_legi_2024(round_: str, raw_dir: Path = paths.RAW) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(turnout per seat, votes per seat × candidate incl. `elected`)."""
    path = _download(*LEGI_2024[round_], raw_dir=raw_dir)
    df = pl.read_csv(path, separator=";", infer_schema_length=0, encoding="utf8-lossy", truncate_ragged_lines=True)
    turnout, votes = [], []
    n_blocks = max(int(m.group(1)) for c in df.columns if (m := re.match(r"Voix (\d+)$", c)))
    for r in df.iter_rows(named=True):
        seat = seat_from_codes(r["Code département"], r["Code circonscription législative"])
        turnout.append({"seat": seat, "inscrits": _int(r["Inscrits"]), "votants": _int(r["Votants"]),
                        "exprimes": _int(r["Exprimés"])})
        for k in range(1, n_blocks + 1):
            v = r.get(f"Voix {k}")
            if v in (None, ""):
                continue
            votes.append({"seat": seat, "no_panneau": _int(r[f"Numéro de panneau {k}"]), "voix": _int(v),
                          "nuance": r[f"Nuance candidat {k}"], "nom": r[f"Nom candidat {k}"],
                          "prenom": r[f"Prénom candidat {k}"],
                          "elected": (r.get(f"Elu {k}") or "").strip().lower().startswith("élu")})
    return pl.DataFrame(turnout), pl.DataFrame(votes)

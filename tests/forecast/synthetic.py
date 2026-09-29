"""Synthetic elections for fast, offline pipeline tests."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from forecast.ingest.wiki_long import LONG_SCHEMA, lineup_key

ELECTION = date(2030, 4, 14)
BLOCS = {"A": "CEN", "B": "EXD", "C": "GAU", "D": "DRO", "E": "EXG", "F": "GAU"}


def true_shares(day: int, lineup: tuple[str, ...]) -> dict[str, float]:
    """Truth on `day` (relative to the election): A rises 26 → 32 over the
    last 180 days, everything else flat; F (when present) takes from C."""
    base = {"A": 26 + 6 * (day + 180) / 180, "B": 25.0, "C": 20.0, "D": 15.0, "E": 8.0, "F": 6.0}
    if "F" not in lineup:
        base["C"] += 4.0
        base["A"] += 2.0
    s = {c: base[c] for c in lineup}
    tot = sum(s.values())
    return {c: 100 * v / tot for c, v in s.items()}


HOUSE = {"P1": {"B": 0.10}, "P2": {"B": -0.10}, "P3": {}}


def make_polls(n_polls: int = 60, seed: int = 1, noise: float = 0.03, lineups=None) -> pl.DataFrame:
    """Long-format polls from three pollsters with known log-scale house effects."""
    rng = np.random.default_rng(seed)
    lineups = lineups or [("A", "B", "C", "D", "E", "F"), ("A", "B", "C", "D", "E")]
    rows = []
    days = np.sort(rng.integers(-180, -2, n_polls))
    for k, d in enumerate(days):
        firm = ("P1", "P2", "P3")[k % 3]
        mid = ELECTION + timedelta(days=int(d))
        tested = lineups if k % 4 == 0 else lineups[:1]
        poll_id = f"poll{k}"
        for j, lu in enumerate(tested):
            truth = true_shares(int(d), lu)
            logs = {c: np.log(v) + HOUSE[firm].get(c, 0.0) + rng.normal(0, noise) for c, v in truth.items()}
            tot = sum(np.exp(v) for v in logs.values())
            for c, v in logs.items():
                rows.append({
                    "election_id": "2030_pres_t1", "year": 2030, "round": "first", "kind": "lineup",
                    "poll_id": poll_id, "row_id": f"{poll_id}:{j}", "lineup_id": lineup_key(list(lu)),
                    "geography": "national", "pollster_raw": firm, "pollster": firm, "sponsor": None,
                    "sponsor_bloc": None, "partisan": False, "fieldwork_start": mid - timedelta(days=1),
                    "fieldwork_end": mid + timedelta(days=1), "midpoint": mid, "publish_est": mid + timedelta(days=2),
                    "sample_size": 1000, "n_eff": 1000.0, "population": "certain_to_vote", "mode": "online",
                    "overlap_weight": 1.0, "n_lineups": len(tested), "option_id": c, "option_kind": "candidate",
                    "bloc": BLOCS[c], "label_raw": c, "share": round(100 * np.exp(v) / tot, 1), "share_lt": False,
                    "source_page": "synthetic", "source_revision": 0, "source_table": 0, "source_row": k,
                    "section_path": "First round",
                })
    return pl.DataFrame(rows, schema=LONG_SCHEMA)

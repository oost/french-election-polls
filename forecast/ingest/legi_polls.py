"""National legislative vote-intention polls → long table of model units.

Columns are parties (PS, LFI, ENS, LR, RN, …); they are summed into the
legislative units (EXG, RAD, GAU, CEN, DRO, RN, EXD, DIV). RAD (the radical
left) is a separate unit only in elections where the left ran split (2012,
2017); from 2022 the left union is one unit, GAU.

Wikipedia merges cells when a poll reports only an alliance total (e.g. one
NFP figure spanning the LFI/PCF/LE/PS sub-columns); pandas repeats that value
in every sub-column, so identical values inside one alliance group are
counted once.
"""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import polars as pl

import polls_helpers as ph
from forecast.ingest.pages import PageSpec, load_page_html, load_page_specs, select_tables
from forecast.ingest.registry import PollsterRegistry
from forecast.ingest.wiki_long import LONG_SCHEMA, _hash, add_poll_weights, parse_share

SPLIT_LEFT_YEARS = {2012, 2017}
LEGI_ELECTION_DATES = {
    2012: ph.date(2012, 6, 10), 2017: ph.date(2017, 6, 11), 2022: ph.date(2022, 6, 12), 2024: ph.date(2024, 6, 30),
}

_RULES: list[tuple[str, str]] = [
    (r"^(abs\.?|turnout|lead)$", "IGNORE"),
    (r"\b(lo|npa|exg)\b", "EXG"),
    (r"\b(fg|pcf|fi|lfi)\b", "RAD"),
    (r"\b(ps|prg|mrc|dvg|eelv|pé|le|fgr)\b", "GAU"),
    (r"\b(modem|ac|pr|nc|lrem|ens|dvc)\b", "CEN"),
    (r"\b(fn|rn|uxd|udr)\b", "RN"),
    (r"\b(ump|lr|udi|udc|dvd|dsv|dlr|dlf|cni|lp)\b", "DRO"),   # UDI ran with LR in 2017 and 2022
    (r"\b(exd|rec)\b", "EXD"),
    (r"\b(aei|eco|pa|reg|div|upr|others?)\b", "DIV"),
]


def unit_of(label: str, year: int) -> str | None:
    s = re.sub(r"\[.*?\]", " ", label).lower()
    s = s.replace("rn and allies", "rn")
    # the group prefix (NUPES / NFP / RN and allies) decides before sub-party names
    if re.match(r"^(nupes|nfp)\b", s):
        return "GAU" if year not in SPLIT_LEFT_YEARS else None
    if re.match(r"^rn\b", s):
        return "RN"
    for pattern, unit in _RULES:
        if re.search(pattern, s):
            if unit == "IGNORE":
                return None
            if unit == "RAD" and year not in SPLIT_LEFT_YEARS:
                return "GAU"
            return unit
    return None


def merge_cells(cells: list[tuple[str, float | None]]) -> dict[str, float]:
    """Unit totals for one row, undoing Wikipedia's merged cells.

    A merged cell (one alliance figure spanning several party columns) shows
    up as a run of identical adjacent values. Runs are collapsed — the value
    goes to the run's most common unit — when that brings the row total
    closer to 100% than leaving them as separate figures."""
    def totals(merge: bool) -> dict[str, float]:
        out: dict[str, float] = {}
        i = 0
        while i < len(cells):
            unit, v = cells[i]
            if v is None:
                i += 1
                continue
            j = i
            while merge and j + 1 < len(cells) and cells[j + 1][1] == v:
                j += 1
            run_units = [u for u, _ in cells[i:j + 1]]
            u = max(dict.fromkeys(run_units), key=run_units.count)
            out[u] = out.get(u, 0.0) + v
            i = j + 1
        return out

    separate, merged = totals(False), totals(True)
    return merged if abs(sum(merged.values()) - 100) < abs(sum(separate.values()) - 100) else separate


def _group(label: str) -> str:
    """Alliance group of a column ("nupes", "nfp", "rn") or the label itself."""
    s = label.lower()
    for g in ("nupes", "nfp", "rn and allies"):
        if s.startswith(g):
            return g
    return s


def extract_legislative(spec: PageSpec, pollsters: PollsterRegistry, html: str | None = None) -> pl.DataFrame:
    html = html if html is not None else load_page_html(spec)
    year = spec.year
    eday = LEGI_ELECTION_DATES.get(year)
    rows: list[dict[str, Any]] = []
    for sel in select_tables(spec, html):
        t = sel.table
        date_col = ph._find_date_column(t.columns)
        firm_col = ph._find_column(t.columns, "polling firm", "polling organisation", "pollster")
        sample_col = ph._find_column(t.columns, "sample")
        cols = []
        for c in t.columns:
            if c in (date_col, firm_col, sample_col):
                continue
            label = ph._column_label(c)
            unit = unit_of(label, year)
            if unit:
                cols.append((c, label, unit, _group(label)))
        other = [c for c in t.columns if c not in (date_col, firm_col)]
        for idx in range(len(t)):
            raw_firm = t[firm_col].iloc[idx]
            if ph.is_event_row(raw_firm, t.iloc[idx][other].tolist()):
                continue
            fw = ph.parse_fieldwork(t[date_col].iloc[idx])
            firm_text = ph.clean_pollster(raw_firm)
            if fw is None or ph._RESULT_ROW.search(firm_text) or re.search(r"ministry|n/a", firm_text, re.I):
                continue
            start, end = fw
            if eday is not None and end >= eday:
                continue
            cells = []
            for c, label, unit, group in cols:
                v, _ = parse_share(t[c].iloc[idx])
                cells.append((unit, v))
            units = merge_cells(cells)
            if len(units) < 3 or not (80 <= sum(units.values()) <= 105):
                continue
            info = pollsters.canonical(firm_text)
            meta = pollsters.meta(info.firm, year)
            n = ph._parse_sample_size(t[sample_col].iloc[idx]) if sample_col is not None else None
            poll_id = _hash(info.firm, firm_text, start, end, n, spec.key)
            opts = sorted(units)
            base = {
                "election_id": f"{spec.election}_t1", "year": year, "round": "first", "kind": "generic",
                "poll_id": poll_id, "row_id": f"{poll_id}:{sel.index}:{idx}", "lineup_id": "+".join(opts),
                "geography": "national", "pollster_raw": firm_text, "pollster": info.firm, "sponsor": info.sponsor,
                "sponsor_bloc": None, "partisan": info.sponsor is not None,
                "fieldwork_start": start, "fieldwork_end": end, "midpoint": start + (end - start) / 2,
                "publish_est": end + timedelta(days=meta.publication_lag_days), "sample_size": n,
                "population": meta.population, "mode": meta.mode, "source_page": spec.key,
                "source_revision": spec.revision, "source_table": sel.index, "source_row": idx,
                "section_path": " › ".join(sel.section_path),
            }
            for u in opts:
                rows.append({**base, "option_id": u, "option_kind": "bloc", "bloc": u, "label_raw": u,
                             "share": units[u], "share_lt": False})
    schema = {k: v for k, v in LONG_SCHEMA.items() if k not in ("overlap_weight", "n_lineups", "n_eff")}
    df = pl.DataFrame(rows, schema=schema).unique(subset=["poll_id", "option_id"], keep="first", maintain_order=True)
    return add_poll_weights(df).select(list(LONG_SCHEMA))


def build_legislative_polls() -> pl.DataFrame:
    specs = load_page_specs()
    pollsters = PollsterRegistry.load()
    return pl.concat([extract_legislative(s, pollsters) for s in specs.values() if s.election_type == "legi"])

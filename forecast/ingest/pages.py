"""Pinned Wikipedia pages and the per-page rules that decide which tables to read.

Generalises `polls_helpers.load_wikipedia_html` (keyed by year) to any page in
`data/reference/pages.yaml`. Cache files are `wiki_cache/{key}_{revision}.html.gz`,
so the historical presidential pages reuse the existing cache files.
"""
from __future__ import annotations

import gzip
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

import polls_helpers as ph
from forecast import paths


@dataclass(frozen=True)
class TableRule:
    match: str
    exclude: bool = False
    round: str | None = None
    kind: str | None = None

    def matches(self, section: str) -> bool:
        return re.search(self.match, section, re.I) is not None


@dataclass(frozen=True)
class PageSpec:
    key: str
    election: str          # e.g. "2027_pres"
    title: str
    revision: int
    rules: tuple[TableRule, ...] = field(default_factory=tuple)

    @property
    def year(self) -> int:
        return int(self.election.split("_")[0])

    @property
    def election_type(self) -> str:
        return self.election.split("_")[1]

    def classify(self, section_path: list[str]) -> TableRule | None:
        """First rule matching the table's section path, or None if the table
        is excluded or matches nothing."""
        section = " › ".join(section_path)
        for rule in self.rules:
            if rule.matches(section):
                return None if rule.exclude else rule
        return None


def load_page_specs(path: Path | str = paths.REFERENCE / "pages.yaml") -> dict[str, PageSpec]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rule_sets = {
        name: tuple(TableRule(**r) for r in (_strip_notes(x) for x in rules))
        for name, rules in (doc.get("rule_sets") or {}).items()
    }
    specs: dict[str, PageSpec] = {}
    for page in doc["pages"]:
        rules = page["rules"]
        rules = rule_sets[rules] if isinstance(rules, str) else tuple(TableRule(**_strip_notes(r)) for r in rules)
        spec = PageSpec(key=str(page["key"]), election=page["election"], title=page["title"],
                        revision=int(page["revision"]), rules=rules)
        specs[spec.key] = spec
    return specs


def _strip_notes(rule: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rule.items() if k != "note"}


def _http_fetch(revision: int) -> str:
    url = f"https://en.wikipedia.org/w/index.php?oldid={revision}"
    request = urllib.request.Request(url, headers={"User-Agent": ph.USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def load_page_html(
    spec: PageSpec,
    cache_dir: Path | str = paths.WIKI_CACHE,
    fetch: Callable[[int], str] = _http_fetch,
    refresh: bool = False,
) -> str:
    """HTML of the pinned revision, downloaded into the cache on first use."""
    path = Path(cache_dir) / f"{spec.key}_{spec.revision}.html.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    html = fetch(spec.revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(html)
    return html


@dataclass(frozen=True)
class SelectedTable:
    page: PageSpec
    index: int
    section_path: tuple[str, ...]
    rule: TableRule
    table: pd.DataFrame


def select_tables(spec: PageSpec, html: str) -> list[SelectedTable]:
    """Polling tables on the page that the page's rules keep."""
    out: list[SelectedTable] = []
    for i, (path, table) in enumerate(ph.parse_polling_tables(html)):
        if not ph._table_has_polling_firm_column(table.columns):
            continue
        rule = spec.classify(path)
        if rule is not None:
            out.append(SelectedTable(spec, i, tuple(path), rule, table))
    return out

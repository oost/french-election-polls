"""Canonical names: candidates (per election year) and polling firms.

Wikipedia headers name candidates many ways — "Marine Le Pen", "Le Pen RN",
"Arthaud (LO)", "Philippe HOR", "Philippe Poutou". A plain whole-word match
on "Philippe" would also hit "Philippe Poutou", so headers are matched with
a score: full name > "First Surname" ending > "Surname PARTY" start.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import polars as pl

import polls_helpers as ph
from forecast import paths

BLOCS = ("EXG", "GAU", "CEN", "DRO", "EXD", "DIV")


@dataclass(frozen=True)
class Candidate:
    year: int
    candidate_id: str
    first_name: str
    surname: str
    aliases: tuple[str, ...]
    party: str
    bloc: str
    ran: bool | None
    incumbency: float | None
    experience: float | None
    result_t1: float | None
    result_t2: float | None

    @property
    def display(self) -> str:
        return self.surname


def _clean_label(label: str) -> str:
    """Drop parentheticals and footnotes, then normalise spacing/case."""
    s = re.sub(r"\(.*?\)", " ", str(label))
    s = re.sub(r"\[.*?\]", " ", s)
    return ph._normalize_name(s)


def _opt_float(v) -> float | None:
    return None if v is None else float(v)


class CandidateRegistry:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self._by_year: dict[int, list[Candidate]] = {}
        for r in df.iter_rows(named=True):
            aliases = tuple(a.strip() for a in (r["aliases"] or "").split("|") if a.strip())
            ran = None if r["ran"] is None else bool(r["ran"])
            c = Candidate(
                year=int(r["year"]), candidate_id=r["candidate_id"], first_name=r["first_name"] or "",
                surname=r["surname"], aliases=aliases, party=r["party"] or "", bloc=r["bloc"],
                ran=ran, incumbency=_opt_float(r["incumbency"]), experience=_opt_float(r["experience"]),
                result_t1=_opt_float(r["result_t1"]), result_t2=_opt_float(r["result_t2"]),
            )
            self._by_year.setdefault(c.year, []).append(c)

    @classmethod
    def load(cls, path: Path | str = paths.REFERENCE / "candidates_pres.csv") -> "CandidateRegistry":
        df = pl.read_csv(path, schema_overrides={
            "ran": pl.Int64, "incumbency": pl.Float64, "experience": pl.Float64,
            "result_t1": pl.Float64, "result_t2": pl.Float64, "aliases": pl.Utf8, "first_name": pl.Utf8,
        })
        return cls(df)

    def candidates(self, year: int) -> list[Candidate]:
        return list(self._by_year.get(year, []))

    def get(self, year: int, candidate_id: str) -> Candidate:
        for c in self._by_year.get(year, []):
            if c.candidate_id == candidate_id:
                return c
        raise KeyError(f"{candidate_id} not registered for {year}")

    def bloc_of(self, year: int) -> dict[str, str]:
        return {c.candidate_id: c.bloc for c in self._by_year.get(year, [])}

    def ballot(self, year: int) -> list[Candidate]:
        """Candidates who actually ran (historical years)."""
        return [c for c in self._by_year.get(year, []) if c.ran]

    def match_header(self, year: int, label: str) -> str | None:
        """candidate_id for a table header, or None if no candidate matches."""
        text = _clean_label(label)
        if not text:
            return None
        best: tuple[int, int, str] | None = None
        ambiguous = False
        for c in self._by_year.get(year, []):
            score = _match_score(text, c)
            if score == 0:
                continue
            key = (score, len(c.surname), c.candidate_id)
            if best is None or key[:2] > best[:2]:
                best, ambiguous = key, False
            elif key[:2] == best[:2]:
                ambiguous = True
        if best is None or ambiguous:
            return None
        return best[2]


def _match_score(text: str, c: Candidate) -> int:
    surname = ph._normalize_name(c.surname)
    full = ph._normalize_name(f"{c.first_name} {c.surname}") if c.first_name else surname
    names = [surname, *(ph._normalize_name(a) for a in c.aliases)]
    if text == full:
        return 4
    if text in names:
        return 3
    if any(text.endswith(" " + n) for n in names):
        return 2
    if any(text.startswith(n + " ") and len(text[len(n):].split()) <= 2 for n in names):
        return 1
    return 0


# --- Polling firms ---------------------------------------------------------

@dataclass(frozen=True)
class FirmInfo:
    firm: str
    sponsor: str | None
    sponsor_bloc: str | None


@dataclass(frozen=True)
class FirmMeta:
    mode: str
    population: str
    publication_lag_days: int
    exclude: bool


class PollsterRegistry:
    def __init__(self, aliases: pl.DataFrame, meta: pl.DataFrame):
        self._aliases = [
            (re.compile(r["pattern"], re.I), FirmInfo(r["firm"], r["sponsor"] or None, r["sponsor_bloc"] or None))
            for r in aliases.iter_rows(named=True)
        ]
        self._meta = list(meta.iter_rows(named=True))

    @classmethod
    def load(cls, ref_dir: Path | str = paths.REFERENCE) -> "PollsterRegistry":
        ref_dir = Path(ref_dir)
        return cls(pl.read_csv(ref_dir / "pollster_aliases.csv"), pl.read_csv(ref_dir / "pollster_meta.csv"))

    def canonical(self, raw: str) -> FirmInfo:
        s = re.sub(r"\*+$", "", ph.clean_pollster(raw)).strip()
        for pattern, info in self._aliases:
            if pattern.search(s):
                return info
        return FirmInfo(s, None, None)

    def meta(self, firm: str, year: int) -> FirmMeta:
        for r in self._meta:
            if r["firm"] == firm and r["from_year"] <= year <= r["to_year"]:
                return FirmMeta(r["mode"], r["population"], int(r["publication_lag_days"]), bool(r["exclude"]))
        return FirmMeta("online", "certain_to_vote", 2, False)

    @cached_property
    def known_firms(self) -> set[str]:
        return {info.firm for _, info in self._aliases}

"""Lineup scenario mixtures (PLAN.md › scenarios_pres_2027.yaml)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from forecast import paths


def option_candidates(option: str | None) -> set[str]:
    """Candidates a slot option puts on the ballot: none for null, several for
    an option joined with '+' (e.g. "PHILIPPE_EDOUARD+LISNARD_DAVID")."""
    return set(option.split("+")) if option else set()


@dataclass(frozen=True)
class Lineup:
    candidates: tuple[str, ...]      # sorted candidate ids
    probability: float
    slots: dict[str, str | None]     # slot → chosen option (for pooling "the primary winner")

    @property
    def key(self) -> str:
        return "+".join(self.candidates)


@dataclass(frozen=True)
class DecisionSpec:
    name: str
    date: date
    pool: tuple[str, ...]
    threshold: float
    fallback_top: int
    fallback_above: float


@dataclass(frozen=True)
class ScenarioSpec:
    election_id: str
    t1: date
    t2: date
    slots: dict[str, dict[str | None, float]]
    fixed: tuple[str, ...]
    optional: dict[str, float]
    constraints: tuple[dict[str, Any], ...]
    decisions: tuple[DecisionSpec, ...] = ()

    @classmethod
    def load(cls, path: Path | str = paths.REFERENCE / "scenarios_pres_2027.yaml") -> "ScenarioSpec":
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        el = doc["election"]
        slots = {name: {(None if k in (None, "null", "None") else k): float(v) for k, v in opts.items()}
                 for name, opts in (doc.get("slots") or {}).items()}
        for name, opts in slots.items():
            if abs(sum(opts.values()) - 1.0) > 1e-6:
                raise ValueError(f"slot {name!r} probabilities sum to {sum(opts.values())}, not 1")
        return cls(
            election_id=el["id"], t1=el["t1"], t2=el["t2"], slots=slots,
            fixed=tuple(doc.get("fixed") or ()), optional={k: float(v) for k, v in (doc.get("optional") or {}).items()},
            constraints=tuple(doc.get("constraints") or ()),
            decisions=tuple(DecisionSpec(name, d["date"], tuple(d["pool"]), float(d["threshold"]),
                                         int(d["fallback_top"]), float(d["fallback_above"]))
                            for name, d in (doc.get("decisions") or {}).items()),
        )

    def candidates(self) -> set[str]:
        out = set(self.fixed) | set(self.optional) | {c for d in self.decisions for c in d.pool}
        for opts in self.slots.values():
            for k in opts:
                out |= option_candidates(k)
        return out

    def enumerate(self) -> list[Lineup]:
        """Every lineup with non-zero probability (constraints applied and the
        remaining mass renormalised)."""
        slot_names = list(self.slots)
        slot_choices = [list(self.slots[s].items()) for s in slot_names]
        opt_names = list(self.optional)
        out: list[Lineup] = []
        for picks in itertools.product(*slot_choices) if slot_choices else [()]:
            p_slots = float(np.prod([p for _, p in picks])) if picks else 1.0
            chosen = {s: c for s, (c, _) in zip(slot_names, picks)}
            for flags in itertools.product([True, False], repeat=len(opt_names)):
                p = p_slots
                for name, on in zip(opt_names, flags):
                    p *= self.optional[name] if on else 1 - self.optional[name]
                if p <= 0:
                    continue
                cands = (set(self.fixed) | set().union(*map(option_candidates, chosen.values()))
                         | {n for n, on in zip(opt_names, flags) if on}
                         | {c for d in self.decisions for c in d.pool})   # decided later, per simulation
                if self._violates(chosen, cands):
                    continue
                out.append(Lineup(tuple(sorted(cands)), p, chosen))
        total = sum(l.probability for l in out)
        return [Lineup(l.candidates, l.probability / total, l.slots) for l in out]

    def _violates(self, chosen: dict[str, str | None], cands: set[str]) -> bool:
        """A rule applies when its `if` slot choices all hold (always, without
        an `if`). It is violated by a lineup containing any `exclude`
        candidate, or none of the `require_any` candidates."""
        for rule in self.constraints:
            cond = rule.get("if", {})
            if all(chosen.get(slot) == opt for slot, opt in cond.items()):
                if cands & set(rule.get("exclude", [])):
                    return True
                if rule.get("require_any") and not cands & set(rule["require_any"]):
                    return True
        return False


def decision_variants(candidates: tuple[str, ...], decisions: tuple[DecisionSpec, ...]) -> list[tuple[str, ...]]:
    """Every lineup a pre-decision lineup can turn into (non-empty subsets of each pool kept)."""
    variants = [tuple(sorted(candidates))]
    for d in decisions:
        pool = [c for c in d.pool if c in candidates]
        new = []
        for v in variants:
            for k in range(1, len(pool) + 1):
                for keep in itertools.combinations(pool, k):
                    new.append(tuple(sorted(c for c in v if c not in set(pool) - set(keep))))
        variants = list(dict.fromkeys(new)) or variants
    return variants


def sample_lineups(lineups: list[Lineup], n: int, rng: np.random.Generator) -> np.ndarray:
    """Index of the lineup drawn by each of `n` simulations."""
    p = np.array([l.probability for l in lineups])
    return rng.choice(len(lineups), size=n, p=p / p.sum())

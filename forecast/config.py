"""Model configuration: frozen dataclasses built from config/model.yaml.

Every parameter in the YAML carries a provenance tag (post | law | assumption |
estimated); `Config.provenance` keeps them so reports can say where each
number came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from forecast import paths

SOURCES = {"post", "law", "assumption", "estimated"}


@dataclass(frozen=True)
class PollParams:
    window_days: int = 90
    decay_early: float = 0.03
    decay_late: float = 0.14
    decay_ramp_start_days: int = 45
    decay_ramp_end_days: int = 14
    sample_size_ref: float = 1000
    sample_size_cap: float = 4000
    partisan_weight: float = 0.5
    partisan_shift_pts: float = 2.0
    house_k_cycle: float = 5
    house_k_race: float = 20
    prior_cycle_decay: float = 0.5
    prior_cap_pres: float = 0.10
    prior_cap_legi: float = 0.20
    prior_k: float = 5
    mode_k: float = 5
    trend_bandwidth_days: float = 7.0
    trend_iterations: int = 3
    trend_lineup_k: float = 5
    trend_min_polls: int = 5
    share_floor: float = 0.1
    use_drift: bool = False


@dataclass(frozen=True)
class NowcastParams:
    anchor_slack_one: float = 2.0
    anchor_floor: float = 1 / 3
    anchor_floor_n: float = 5
    nest_mu_prior: float = 0.6
    nest_mu_prior_sd: float = 0.2
    imputation_df: float = 4


@dataclass(frozen=True)
class HorizonParams:
    move_a: float = 0.06
    move_b: float = 0.018
    bloc_share_of_var: float = 0.5


@dataclass(frozen=True)
class ErrorParams:
    nu_floor: float = 3
    runoff_rho: float = 0.5
    runoff_a: float = 0.10
    runoff_b: float = 0.015


@dataclass(frozen=True)
class SimParams:
    n_sims: int = 40_000
    n_sims_backtest: int = 10_000
    seed: int = 20270418


@dataclass(frozen=True)
class LeanParams:
    recent_weight: float = 0.8


@dataclass(frozen=True)
class LegislativeParams:
    majority_seats: int = 289
    n_seats: int = 577
    qualify_registered: float = 0.125
    outright_expressed: float = 0.5
    outright_registered: float = 0.25
    front_republicain_sd: float = 0.8
    chunk_size: int = 2000


_SECTIONS = {
    "polls": PollParams, "nowcast": NowcastParams, "horizon": HorizonParams, "errors": ErrorParams,
    "simulation": SimParams, "lean": LeanParams, "legislative": LegislativeParams,
}


@dataclass(frozen=True)
class Config:
    polls: PollParams = field(default_factory=PollParams)
    nowcast: NowcastParams = field(default_factory=NowcastParams)
    horizon: HorizonParams = field(default_factory=HorizonParams)
    errors: ErrorParams = field(default_factory=ErrorParams)
    simulation: SimParams = field(default_factory=SimParams)
    lean: LeanParams = field(default_factory=LeanParams)
    legislative: LegislativeParams = field(default_factory=LegislativeParams)
    provenance: dict[str, str] = field(default_factory=dict)


def load_config(path: Path | str = paths.MODEL_CONFIG) -> Config:
    doc: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    kwargs: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for section, cls in _SECTIONS.items():
        entries = doc.get(section, {}) or {}
        known = {f.name: f.type for f in fields(cls)}
        values = {}
        for key, entry in entries.items():
            if key not in known:
                raise KeyError(f"unknown parameter {section}.{key} in {path}")
            if not isinstance(entry, dict) or "value" not in entry or entry.get("source") not in SOURCES:
                raise ValueError(f"{section}.{key} needs {{value, source}} with source in {sorted(SOURCES)}")
            values[key] = entry["value"]
            provenance[f"{section}.{key}"] = entry["source"]
        kwargs[section] = cls(**values)
    return Config(**kwargs, provenance=provenance)


_DEFAULT: Config | None = None


def default_config() -> Config:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_config()
    return _DEFAULT

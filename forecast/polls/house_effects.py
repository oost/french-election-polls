"""Pollster house effects and mode/population effects (PLAN.md › steps 5–7).

All effects are on the log-share scale and are estimated from residuals
against the leave-one-out pass-1 race averages. Row weights count a poll that
tested k lineups as 1/k per row, so a poll isn't counted k times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from forecast.config import PollParams


def shrink(n: np.ndarray | float, k: float) -> np.ndarray:
    """n/(n+k): the share of a measured lean that is kept."""
    n = np.asarray(n, dtype=float)
    return n / (n + k)


def group_effects(resid: np.ndarray, groups: np.ndarray, weight: np.ndarray, k: float) -> tuple[dict, np.ndarray]:
    """Shrunk mean residual per group (mode or population), centred so the
    size-weighted mean across groups is zero. Returns ({group: effect[C]}, per-row effect[R, C])."""
    R, C = resid.shape
    labels = [g for g in dict.fromkeys(groups.tolist())]
    eff = {}
    sizes = {}
    for g in labels:
        rows = groups == g
        e = np.zeros(C)
        n_c = np.zeros(C)
        for c in range(C):
            ok = rows & ~np.isnan(resid[:, c])
            if ok.any():
                n = weight[ok].sum()
                e[c] = np.average(resid[ok, c], weights=weight[ok]) * n / (n + k)
                n_c[c] = n
        eff[g], sizes[g] = e, n_c
    if len(labels) > 1:
        tot = sum(sizes.values())
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(tot > 0, sum(eff[g] * sizes[g] for g in labels) / tot, 0.0)
        eff = {g: eff[g] - mean for g in labels}
    else:
        eff = {g: np.zeros(C) for g in labels}
    per_row = np.stack([eff[g] for g in groups]) if R else np.zeros((0, C))
    return eff, per_row


@dataclass
class HouseHistory:
    """Past cycles' cycle-wide leans per (firm, bloc), for the historical prior.
    `leans[(firm, bloc)]` is a list of (cycle, lean, n)."""
    leans: dict[tuple[str, str], list[tuple[int, float, float]]] = field(default_factory=dict)

    def add(self, cycle: int, table: pl.DataFrame) -> None:
        for r in table.iter_rows(named=True):
            self.leans.setdefault((r["firm"], r["bloc"]), []).append((cycle, r["lean"], r["n"]))

    def prior(self, firm: str, bloc: str, before: int, decay: float) -> tuple[float, float]:
        """(prior lean, W) from cycles before `before`: each older cycle
        weighted `decay` times the next; W = Σ decay^(k−1)·n_k."""
        entries = sorted([e for e in self.leans.get((firm, bloc), []) if e[0] < before], reverse=True)
        if not entries:
            return 0.0, 0.0
        w = np.array([decay ** k * n for k, (_, _, n) in enumerate(entries)])
        x = np.array([lean for _, lean, _ in entries])
        W = float(w.sum())
        return (float(np.average(x, weights=w)) if W > 0 else 0.0), W


@dataclass(frozen=True)
class HouseEffects:
    firms: list[str]
    cycle_wide: np.ndarray          # (P, C) after prior blend and cross-firm centring
    race_specific: dict[tuple[int, int], np.ndarray]   # (firm idx, race idx) → (C,) add-back
    n: np.ndarray                   # (P, C) weighted poll counts
    alpha: np.ndarray               # (P, C) weight on the historical prior

    def per_row(self, firm_idx: np.ndarray, race: np.ndarray) -> np.ndarray:
        out = self.cycle_wide[firm_idx].copy()
        for i, (p, s) in enumerate(zip(firm_idx, race)):
            add = self.race_specific.get((int(p), int(s)))
            if add is not None:
                out[i] += add
        return out

    def table(self, options: list[str], blocs: list[str | None]) -> pl.DataFrame:
        rows = []
        for p, firm in enumerate(self.firms):
            for c, opt in enumerate(options):
                if self.n[p, c] > 0:
                    rows.append({"firm": firm, "option_id": opt, "bloc": blocs[c], "lean": float(self.cycle_wide[p, c]),
                                 "n": float(self.n[p, c]), "alpha": float(self.alpha[p, c])})
        return pl.DataFrame(rows, schema={"firm": pl.Utf8, "option_id": pl.Utf8, "bloc": pl.Utf8,
                                          "lean": pl.Float64, "n": pl.Float64, "alpha": pl.Float64})


def fit_house_effects(
    resid: np.ndarray,
    firm_idx: np.ndarray,
    firms: list[str],
    race: np.ndarray,
    weight: np.ndarray,
    blocs: list[str | None],
    p: PollParams,
    prior_cap: float,
    history: HouseHistory | None = None,
    cycle: int | None = None,
) -> HouseEffects:
    """The post's three-step house effect, per firm × option:

    1. cycle-wide lean L = mean residual, kept at n/(n+5);
    2. blended with the firm's historical lean for the option's bloc at weight
       α = cap·W/(W+k) (W = decayed past poll count), then centred across firms
       (√n-weighted) so the average firm has no lean;
    3. race-specific deviation from L added back at n_s/(n_s+20).
    """
    R, C = resid.shape
    P = len(firms)
    L = np.zeros((P, C))
    n = np.zeros((P, C))
    for f in range(P):
        rows = firm_idx == f
        for c in range(C):
            ok = rows & ~np.isnan(resid[:, c])
            if ok.any():
                n[f, c] = weight[ok].sum()
                L[f, c] = np.average(resid[ok, c], weights=weight[ok])
    L_hat = L * shrink(n, p.house_k_cycle)

    alpha = np.zeros((P, C))
    prior = np.zeros((P, C))
    if history is not None and cycle is not None:
        for f, firm in enumerate(firms):
            for c in range(C):
                if blocs[c] is None:
                    continue
                pr, W = history.prior(firm, blocs[c], before=cycle, decay=p.prior_cycle_decay)
                prior[f, c] = pr
                alpha[f, c] = prior_cap * W / (W + p.prior_k) if W > 0 else 0.0
    cw = (1 - alpha) * L_hat + alpha * prior
    cw = np.where(n > 0, cw, 0.0)

    # the average firm has no lean (√n weights)
    sw = np.sqrt(n)
    for c in range(C):
        if sw[:, c].sum() > 0:
            cw[:, c] -= np.average(cw[:, c], weights=sw[:, c])
    cw = np.where(n > 0, cw, 0.0)

    race_specific: dict[tuple[int, int], np.ndarray] = {}
    for f in range(P):
        for s in np.unique(race[firm_idx == f]):
            rows = (firm_idx == f) & (race == s)
            add = np.zeros(C)
            for c in range(C):
                ok = rows & ~np.isnan(resid[:, c])
                if ok.any():
                    n_s = weight[ok].sum()
                    add[c] = (np.average(resid[ok, c], weights=weight[ok]) - L[f, c]) * shrink(n_s, p.house_k_race)
            race_specific[(f, int(s))] = add
    return HouseEffects(firms=firms, cycle_wide=cw, race_specific=race_specific, n=n, alpha=alpha)


def bloc_leans(effects: pl.DataFrame) -> pl.DataFrame:
    """Collapse a cycle's firm × option leans to firm × bloc (n-weighted), the
    unit the historical prior is stored in."""
    return (
        effects.filter(pl.col("bloc").is_not_null())
        .group_by("firm", "bloc")
        .agg(((pl.col("lean") * pl.col("n")).sum() / pl.col("n").sum()).alias("lean"), pl.col("n").sum())
    )

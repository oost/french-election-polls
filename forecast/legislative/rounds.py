"""Two-round rules, withdrawals and vote transfers (PLAN.md › Legislative step 8).

Law (Code électoral L.126/L.162): elected in the first round with > 50% of
votes cast and ≥ 25% of registered voters; otherwise the top two plus anyone
with ≥ 12.5% of registered voters go through (top two if fewer qualify).

Withdrawals (désistements) and transfers are estimated from 2012–2024 seat
results. Transfers are a conditional logit: a first-round unit k's voters
choose among the units still standing r (and abstention, utility 0) with
utility A[k, r]; a unit's own voters stay with it. Units are aggregated, so a
unit's dissident candidates are not modelled separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

from forecast.legislative import data as D

UNITS = D.UNITS
U = {u: i for i, u in enumerate(UNITS)}
OWN_UTILITY = 5.0


def runoff_table(year: int) -> pl.DataFrame:
    """Per seat going to a runoff: first-round unit shares (of votes cast and
    of registered), the qualified units, and the units standing in round 2
    with their shares of the round-2 votes cast."""
    t1 = D.seat_units_t1(year)
    _, t2 = D.legi_results(year, "t2")
    t2u = t2.group_by("seat", "unit").agg(pl.col("voix").sum())
    t2u = t2u.with_columns((pl.col("voix") / pl.col("voix").sum().over("seat")).alias("share2"))
    return t1.join(t2u.select("seat", "unit", "share2"), on=["seat", "unit"], how="left").filter(
        pl.col("seat").is_in(t2u["seat"].unique().to_list()))


def _matrices(year: int):
    df = runoff_table(year)
    seats = df["seat"].unique(maintain_order=True).to_list()
    si = {s: i for i, s in enumerate(seats)}
    v1 = np.zeros((len(seats), len(UNITS)))
    reg = np.zeros_like(v1)
    w2 = np.full_like(v1, np.nan)
    for r in df.iter_rows(named=True):
        i, k = si[r["seat"]], U[r["unit"]]
        v1[i, k] = r["share"]
        reg[i, k] = r["share_registered"]
        if r["share2"] is not None:
            w2[i, k] = r["share2"]
    return seats, v1, reg, w2


# --- Withdrawals -------------------------------------------------------------

@dataclass(frozen=True)
class WithdrawalModel:
    """P(a qualified unit placed third or lower withdraws), by unit and by
    whether RN is in the top two ("front républicain" situations)."""
    p: dict[tuple[str, bool], float]
    default: float

    def prob(self, unit: str, rn_top2: bool) -> float:
        return self.p.get((unit, rn_top2), self.default)


def fit_withdrawals(years: list[int], prior_n: float = 5.0) -> WithdrawalModel:
    counts: dict[tuple[str, bool], list[int]] = {}
    for y in years:
        seats, v1, reg, w2 = _matrices(y)
        for i in range(len(seats)):
            order = np.argsort(-v1[i])
            qualified = [k for k in order if reg[i, k] >= 0.125]
            if len(qualified) < 3:
                continue
            rn_top2 = U["RN"] in order[:2]
            for k in qualified:
                if k in order[:2]:
                    continue
                key = (UNITS[k], bool(rn_top2))
                counts.setdefault(key, [0, 0])
                counts[key][0] += int(np.isnan(w2[i, k]))
                counts[key][1] += 1
    tot_w = sum(c[0] for c in counts.values())
    tot_n = sum(c[1] for c in counts.values())
    base = tot_w / tot_n if tot_n else 0.5
    p = {k: (w + prior_n * base) / (n + prior_n) for k, (w, n) in counts.items()}
    return WithdrawalModel(p, base)


# --- Transfers ---------------------------------------------------------------

@dataclass(frozen=True)
class TransferMatrix:
    A: np.ndarray            # (K, K) utility of source k → destination r (diagonal = own)
    resid_sd: float

    def second_round(self, v1: np.ndarray, standing: np.ndarray, A: np.ndarray | None = None) -> np.ndarray:
        """Round-2 shares of the standing units.

        v1: (..., K) first-round shares; standing: (..., K) bool. Returns (..., K)
        shares summing to 1 over standing units (0 elsewhere)."""
        A = self.A if A is None else A
        util = np.where(standing[..., None, :], A, -np.inf)          # (..., K_src, K_dst)
        m = np.maximum(util.max(axis=-1, keepdims=True), 0.0)
        e = np.exp(util - m)
        p = e / (e.sum(axis=-1, keepdims=True) + np.exp(-m))          # abstention utility 0
        votes = (v1[..., :, None] * p).sum(axis=-2)
        return votes / np.maximum(votes.sum(axis=-1, keepdims=True), 1e-12)


def fit_transfers(years: list[int], recency: float = 0.5, prior_sd: float = 2.0) -> TransferMatrix:
    """Least squares on round-2 shares; recent elections weigh more."""
    data = []
    for age, y in enumerate(sorted(years, reverse=True)):
        seats, v1, reg, w2 = _matrices(y)
        standing = ~np.isnan(w2)
        keep = standing.sum(axis=1) >= 2
        data.append((v1[keep], standing[keep], np.nan_to_num(w2[keep]), recency ** age))
    K = len(UNITS)
    off = ~np.eye(K, dtype=bool)

    def build(x):
        A = np.full((K, K), OWN_UTILITY)
        A[off] = x
        return A

    tm = TransferMatrix(np.zeros((K, K)), 0.0)

    def loss(x):
        A = build(x)
        total = 0.0
        for v1, st, w2, wt in data:
            pred = tm.second_round(v1, st, A)
            total += wt * np.sum(np.where(st, (pred - w2) ** 2, 0.0))
        return total + np.sum((x / prior_sd) ** 2) * 1e-3

    sol = minimize(loss, np.zeros(off.sum()), method="L-BFGS-B")
    A = build(sol.x)
    v1, st, w2, _ = data[0]
    resid = np.where(st, tm.second_round(v1, st, A) - w2, np.nan)
    return TransferMatrix(A, float(np.nanstd(resid)))


# --- Turnout & qualification -----------------------------------------------

def outcome_first_round(share: np.ndarray, ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """share: (..., K) of votes cast; ratio: (...,) votes cast / registered.

    Returns (winner index or −1, qualified mask (..., K))."""
    reg_share = share * ratio[..., None]
    top = share.argmax(axis=-1)
    top_share = np.take_along_axis(share, top[..., None], axis=-1)[..., 0]
    top_reg = np.take_along_axis(reg_share, top[..., None], axis=-1)[..., 0]
    outright = (top_share > 0.5) & (top_reg >= 0.25)
    qualified = reg_share >= 0.125
    order = np.argsort(-share, axis=-1)
    top2 = np.zeros_like(qualified)
    np.put_along_axis(top2, order[..., :2], True, axis=-1)
    few = qualified.sum(axis=-1) < 2
    qualified = np.where(few[..., None], top2, qualified)
    qualified &= share > 0
    return np.where(outright, top, -1), qualified


def apply_withdrawals(share: np.ndarray, qualified: np.ndarray, model: WithdrawalModel,
                      shock: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Standing units after withdrawals. `shock` (logit scale, broadcastable to
    share.shape[:-1], e.g. (n_sims, 1)) moves every withdrawal probability
    together — the shared 'front républicain'."""
    order = np.argsort(-share, axis=-1)
    rank = np.argsort(order, axis=-1)
    rn_top2 = rank[..., U["RN"]] < 2
    standing = qualified.copy()
    for k, u in enumerate(UNITS):
        for flag in (False, True):
            p = model.prob(u, flag)
            logit = np.log(p / (1 - p)) if 0 < p < 1 else (10.0 if p >= 1 else -10.0)
            prob = np.broadcast_to(1 / (1 + np.exp(-(logit + shock))), rn_top2.shape)
            mask = qualified[..., k] & (rank[..., k] >= 2) & (rn_top2 == flag)
            draw = rng.random(mask.shape) < prob
            standing[..., k] &= ~(mask & draw)
    return standing

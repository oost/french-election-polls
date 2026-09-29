"""Seat now-cast for all 577 constituencies (post § 3.1.2).

    η[s, u] = log N_u + presence_u + β·x[s, u],   share[s] = softmax over the units standing

N_u is the national share: the actual national result when calibrating on
past cycles — so the model learns what a poll *would* say in the seat, with
the national polling error removed — and the national polling average when
forecasting. β is fitted by (ridge) least squares on log shares.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize

from forecast.legislative.features import PREDICTORS

PRIOR = {"lean": (1.0, 0.5), "lagged": (0.3, 0.5), "has_lagged": (0.0, 0.5), "inc": (0.2, 0.5)}


def _seat_softmax(eta: np.ndarray, seat_idx: np.ndarray, n_seats: int) -> np.ndarray:
    m = np.full(n_seats, -np.inf)
    np.maximum.at(m, seat_idx, eta)
    e = np.exp(eta - m[seat_idx])
    tot = np.zeros(n_seats)
    np.add.at(tot, seat_idx, e)
    return e / tot[seat_idx]


@dataclass(frozen=True)
class SeatNowcast:
    beta: dict[str, float]
    resid_sd: float
    train_years: tuple[int, ...]

    def predict(self, design: pl.DataFrame, national: dict[str, float]) -> pl.DataFrame:
        """Adds `mu` (log share, renormalised within seat) to the design table."""
        seats = design["seat"].to_list()
        uniq = {s: i for i, s in enumerate(dict.fromkeys(seats))}
        si = np.array([uniq[s] for s in seats])
        eta = self._eta(design, national)
        p = _seat_softmax(eta, si, len(uniq))
        return design.with_columns(pl.Series("mu", np.log(np.maximum(p, 1e-9))))

    def _eta(self, design: pl.DataFrame, national: dict[str, float]) -> np.ndarray:
        n = np.log(np.maximum(np.array([national.get(u, 0.005) for u in design["unit"]]), 1e-3))
        eta = n + design["presence"].to_numpy()
        for k in PREDICTORS:
            eta = eta + self.beta[k] * design[k].to_numpy()
        return eta


def fit_seat_nowcast(training: list[tuple[pl.DataFrame, dict[str, float]]]) -> SeatNowcast:
    """training: (design with `share`, national result) per past cycle."""
    X, base, y, si, offsets = [], [], [], [], 0
    years = []
    for df, nat in training:
        df = df.filter(pl.col("share").is_not_null() & (pl.col("share") > 0))
        seats = {s: i for i, s in enumerate(dict.fromkeys(df["seat"].to_list()))}
        si.append(np.array([seats[s] for s in df["seat"]]) + offsets)
        offsets += len(seats)
        X.append(np.column_stack([df[k].to_numpy() for k in PREDICTORS]))
        base.append(np.log(np.maximum(np.array([nat.get(u, 0.005) for u in df["unit"]]), 1e-3)) + df["presence"].to_numpy())
        y.append(np.log(df["share"].to_numpy()))
        years.append(int(df["year"][0]))
    X, base, y, si = np.vstack(X), np.concatenate(base), np.concatenate(y), np.concatenate(si)
    mu0 = np.array([PRIOR[k][0] for k in PREDICTORS])
    sd0 = np.array([PRIOR[k][1] for k in PREDICTORS])

    def loss(b):
        p = _seat_softmax(base + X @ b, si, offsets)
        r = y - np.log(np.maximum(p, 1e-9))
        return float(np.sum(r ** 2) + np.sum(((b - mu0) / sd0) ** 2))

    sol = minimize(loss, mu0, method="L-BFGS-B")
    p = _seat_softmax(base + X @ sol.x, si, offsets)
    resid = y - np.log(np.maximum(p, 1e-9))
    return SeatNowcast(dict(zip(PREDICTORS, map(float, sol.x))), float(np.std(resid)), tuple(years))

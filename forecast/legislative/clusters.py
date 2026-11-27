"""Vote-pattern clusters of constituencies (the post's demographic clusters).

Constituency-level demographics are not published, so seats are clustered on
how they vote: presidential first-round unit shares (last two elections,
log scale) plus presidential turnout, standardised; k-means with a fixed seed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import polars as pl

from forecast.ingest.harmonised import seat_results
from forecast.legislative import data as D


def _kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), size=1)]
    for _ in range(1, k):   # k-means++ initialisation
        d = ((X[:, None, :] - centres[None]) ** 2).sum(-1).min(1)
        centres = np.vstack([centres, X[rng.choice(len(X), p=d / d.sum())]])
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = ((X[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
        if (new == labels).all() and _ > 0:
            break
        labels = new
        centres = np.vstack([X[labels == j].mean(0) if (labels == j).any() else centres[j] for j in range(k)])
    return labels


@lru_cache(maxsize=None)
def seat_clusters(pres_years: tuple[int, ...] = (2022, 2017), k: int = 10) -> pl.DataFrame:
    feats = []
    for y in pres_years:
        s = D.pres_seat_units(y).pivot(on="unit", index="seat", values="share").fill_null(1e-3)
        s = s.select("seat", *[pl.col(c).clip(1e-3).log().alias(f"{c}_{y}") for c in s.columns if c != "seat"])
        feats.append(s)
        turnout, _ = seat_results(f"{y}_pres_t1")
        feats.append(turnout.select("seat", (pl.col("votants") / pl.col("inscrits")).alias(f"turnout_{y}")))
    df = feats[0]
    for f in feats[1:]:
        df = df.join(f, on="seat", how="inner")
    df = df.sort("seat")
    X = df.drop("seat").to_numpy()
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    return df.select("seat").with_columns(pl.Series("cluster", [f"C{c}" for c in _kmeans(X, k)]))

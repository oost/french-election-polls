"""Fit the fundamentals DLM in Stan and predict any lineup (post § 3.2)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from forecast import paths
from forecast.fundamentals.features_pres import PREDICTORS, PresFeatures
from forecast.stan.build import cached_sample

SHARE_FLOOR = 0.002


def load_priors(section: str, path: Path = paths.REFERENCE / "priors.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))[section]


@dataclass
class FundamentalsFit:
    features: PresFeatures
    train_years: tuple[int, ...]
    beta_next: np.ndarray       # (draws, K)
    phi: np.ndarray             # (draws,)
    beta: np.ndarray            # (draws, K, T)

    def predict(self, year: int, lineup: list[str], seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(mean log share, sd log share, share draws) for `lineup` in `year`.

        Full posterior predictive: coefficient drift to the new election
        (beta_next) plus Dirichlet noise — the post's "honest measure of how
        unsure it is", not the hairline sd of a coefficient."""
        X = self.features.matrix(year, lineup)
        eta = self.beta_next @ X.T                                     # (draws, n)
        p = np.exp(eta - eta.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        rng = np.random.default_rng(seed)
        draws = np.array([rng.dirichlet(np.maximum(pi * f, 1e-3)) for pi, f in zip(p, self.phi)])
        logd = np.log(np.maximum(draws, 1e-6))
        return logd.mean(axis=0), logd.std(axis=0), draws


def fit_fundamentals(features: PresFeatures, train_years: list[int], **sample_kwargs) -> FundamentalsFit:
    races = features.training_races(sorted(train_years))
    years = [r[0] for r in races]
    t_index = {y: i + 1 for i, y in enumerate(years)}
    X, y, race_t, start, length = [], [], [], [], []
    for year, ids, Xr, shares in races:
        start.append(len(y) + 1)
        length.append(len(ids))
        race_t.append(t_index[year])
        X.extend(Xr.tolist())
        s = np.maximum(shares, SHARE_FLOOR)
        y.extend((s / s.sum()).tolist())
    pri = load_priors("fundamentals_pres")
    data = {
        "T": len(years), "K": len(PREDICTORS), "R": len(races), "N": len(y),
        "race_t": race_t, "start": start, "len": length, "X": np.array(X), "base": np.zeros(len(y)),
        "y": np.array(y), "b1_mu": [pri[k][0] for k in PREDICTORS], "b1_sd": [pri[k][1] for k in PREDICTORS],
        "tau_scale": [pri[k][2] for k in PREDICTORS], "phi_log_mu": pri["phi_log_mu"],
    }
    draws = cached_sample("fundamentals_dlm", data, ["beta_next", "phi", "beta"],
                          **({"adapt_delta": 0.95} | sample_kwargs))
    return FundamentalsFit(features, tuple(years), draws["beta_next"], draws["phi"], draws["beta"])

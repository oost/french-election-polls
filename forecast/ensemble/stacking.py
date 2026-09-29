"""N-component ensemble weighted by out-of-sample precision (post § 4).

Each component k states a mean and sd per unit (log-share scale). Its
out-of-sample error scale is fitted on the backtest archive as

    σ_k(h, share) = (a_k + b_k·√h) · (share / 10%)^(−γ_k)

and the ensemble weights are w_k ∝ 1/σ_k². Before an archive exists the
components' own stated sds are used. The post's adjustments:

* open races (no incumbent running): fundamentals precision × ρ_open;
* a heavily polled race whose polls sharply disagree with the structural
  estimate: polls precision × 1.25.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.optimize import minimize


@dataclass(frozen=True)
class ComponentScale:
    a: float
    b: float
    gamma: float

    def sd(self, h, share_pct):
        s = np.clip(np.asarray(share_pct, dtype=float), 0.2, 60.0)
        return (self.a + self.b * np.sqrt(np.maximum(np.asarray(h, dtype=float), 0))) * (s / 10) ** (-self.gamma)


def fit_component_scale(h: np.ndarray, share: np.ndarray, err: np.ndarray) -> ComponentScale:
    ok = np.isfinite(err)
    h, share, err = h[ok], np.clip(share[ok], 0.2, 60), err[ok]

    def nll(x):
        a, b, g = np.exp(x[0]), np.exp(x[1]), x[2]
        sd = (a + b * np.sqrt(h)) * (share / 10) ** (-g)
        return np.sum(np.log(sd) + 0.5 * (err / sd) ** 2)

    sol = minimize(nll, np.array([np.log(0.1), np.log(0.02), 0.3]), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
    return ComponentScale(float(np.exp(sol.x[0])), float(np.exp(sol.x[1])), float(sol.x[2]))


STACK_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


@dataclass
class PrecisionModel:
    scales: dict[str, ComponentScale] = field(default_factory=dict)
    stack: dict[str, float] = field(default_factory=dict)   # precision multipliers chosen out of sample
    rho_open: float = 0.8
    disagree_boost: float = 1.25
    heavy_polls: int = 10
    disagree_z: float = 2.0

    @classmethod
    def fit(cls, archive: pl.DataFrame, components: tuple[str, ...]) -> "PrecisionModel":
        """Fit each component's out-of-sample scale on archive rows
        (columns: days_out, mu_<k>, result_log, share_<k>)."""
        scales = {}
        for k in components:
            sub = archive.filter(pl.col(f"mu_{k}").is_not_null())
            if sub.height < 20:
                continue
            err = (sub["result_log"] - sub[f"mu_{k}"]).to_numpy()
            scales[k] = fit_component_scale(sub["days_out"].to_numpy().astype(float),
                                            100 * np.exp(sub[f"mu_{k}"].to_numpy()), err)
        model = cls(scales)
        if {"polls", "fundamentals"} <= set(scales):
            model.stack = {"fundamentals": model._choose_stack(archive)}
        return model

    def _choose_stack(self, archive: pl.DataFrame) -> float:
        """Stacking step: the fundamentals precision multiplier κ that minimises
        the ensemble's out-of-sample squared log error on the archive (errors
        centred per snapshot, since log shares are only defined up to a constant)."""
        df = archive.filter(pl.col("mu_fundamentals").is_not_null())
        h = df["days_out"].to_numpy().astype(float)
        mp, mf, y = df["mu_polls"].to_numpy(), df["mu_fundamentals"].to_numpy(), df["result_log"].to_numpy()
        sp = self.scales["polls"].sd(h, 100 * np.exp(mp))
        sf = self.scales["fundamentals"].sd(h, 100 * np.exp(mf))
        snap = df.select(pl.concat_str(["cycle", "days_out"], separator=":")).to_series().to_numpy()
        best, best_loss = 1.0, np.inf
        for kappa in STACK_GRID:
            wp, wf = 1 / sp ** 2, kappa / sf ** 2
            mu = (wp * mp + wf * mf) / (wp + wf)
            e = pl.DataFrame({"s": snap, "e": y - mu}).with_columns(
                (pl.col("e") - pl.col("e").mean().over("s")).alias("c"))["c"].to_numpy()
            loss = float(np.mean(e ** 2))
            if loss < best_loss:
                best, best_loss = kappa, loss
        return best

    def sd(self, component: str, h: float, mu: np.ndarray, stated_sd: np.ndarray) -> np.ndarray:
        if component in self.scales:
            return self.scales[component].sd(h, 100 * np.exp(mu))
        return stated_sd


@dataclass(frozen=True)
class EnsembleForecast:
    mu: np.ndarray            # log shares, renormalised
    sd_bar: np.ndarray        # Σ w_k σ_k (the post's weighted-average uncertainty)
    weights: dict[str, np.ndarray]
    disagree: np.ndarray


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def combine(
    components: dict[str, tuple[np.ndarray, np.ndarray]],
    precision: PrecisionModel,
    h: float,
    n_polls: int,
    open_race: bool,
) -> EnsembleForecast:
    """components: name → (mean log share, stated sd), aligned by candidate."""
    sds = {k: precision.sd(k, h, mu, sd) for k, (mu, sd) in components.items()}
    prec = {k: precision.stack.get(k, 1.0) / np.maximum(sd, 1e-6) ** 2 for k, sd in sds.items()}
    disagree = np.zeros_like(next(iter(components.values()))[0])
    if "polls" in components and "fundamentals" in components:
        mp, mf = components["polls"][0], components["fundamentals"][0]
        disagree = np.abs(mp - mf) / np.sqrt(sds["polls"] ** 2 + sds["fundamentals"] ** 2)
        if open_race:
            prec["fundamentals"] = prec["fundamentals"] * precision.rho_open
        if n_polls >= precision.heavy_polls:
            prec["polls"] = np.where(disagree > precision.disagree_z, prec["polls"] * precision.disagree_boost,
                                     prec["polls"])
    total = sum(prec.values()) + 1e-12
    weights = {k: p / total for k, p in prec.items()}
    mu = sum(weights[k] * components[k][0] for k in components)
    sd_bar = sum(weights[k] * sds[k] for k in components)
    return EnsembleForecast(log_softmax(mu), sd_bar, weights, disagree)

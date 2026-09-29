"""Stan models: compile and recover known parameters from simulated data.

Run with `pytest -m slow` (needs cmdstanpy + CmdStan)."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from forecast.stan.build import cached_sample, stan_available

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not stan_available(), reason="CmdStan not installed")]


def test_fundamentals_dlm_recovers_coefficients(tmp_path, monkeypatch):
    monkeypatch.setattr("forecast.paths.FIT_CACHE", tmp_path)
    rng = np.random.default_rng(0)
    T, K, per_t, n = 4, 2, 30, 5
    beta = np.array([1.0, -0.5])
    X, y, race_t, start, length = [], [], [], [], []
    for t in range(1, T + 1):
        for _ in range(per_t):
            x = rng.normal(0, 1, (n, K))
            p = np.exp(x @ beta)
            p /= p.sum()
            s = rng.dirichlet(p * 200)
            start.append(len(y) + 1)
            length.append(n)
            race_t.append(t)
            X.extend(x.tolist())
            s = np.maximum(s, 1e-4)
            y.extend((s / s.sum()).tolist())
    data = {"T": T, "K": K, "R": len(start), "N": len(y), "race_t": race_t, "start": start, "len": length,
            "X": np.array(X), "base": np.zeros(len(y)), "y": np.array(y), "b1_mu": [0.0, 0.0],
            "b1_sd": [1.0, 1.0], "tau_scale": [0.1, 0.1], "phi_log_mu": 5.0}
    draws = cached_sample("fundamentals_dlm", data, ["beta", "phi"], iter_warmup=400, iter_sampling=400)
    est = draws["beta"][:, :, -1].mean(axis=0)
    assert est == pytest.approx(beta, abs=0.15)
    assert 100 < draws["phi"].mean() < 400


def test_error_meta_recovers_scales(tmp_path, monkeypatch):
    monkeypatch.setattr("forecast.paths.FIT_CACHE", tmp_path)
    from forecast.errors.meta_model import fit_error_model
    rng = np.random.default_rng(1)
    rows = []
    for cycle in range(6):
        for days in (2, 30, 90):
            shocks = {b: rng.normal(0, 0.05 + 0.01 * np.sqrt(days)) for b in "ABC"}
            for j in range(40):
                b = "ABC"[j % 3]
                sd = 0.1
                rows.append({"cycle": cycle, "days_out": days, "bloc": b, "sd_ens": sd, "n_polls": 20,
                             "disagree": 0.0, "err": shocks[b] + rng.normal(0, 1.3 * sd)})
    model = fit_error_model(pl.DataFrame(rows), ("bloc",), nu_floor=30.0, iter_warmup=400, iter_sampling=400)
    assert model.m0 * (1 + model.m1 / 21) == pytest.approx(1.3, abs=0.2)
    assert model.shock_sd("bloc", 30) == pytest.approx(0.05 + 0.01 * np.sqrt(30), abs=0.05)

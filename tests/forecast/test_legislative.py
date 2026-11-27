"""Legislative model: two-round rules, withdrawals, transfers, poll parsing,
now-cast recovery and simulation invariants (synthetic, offline)."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from forecast.config import default_config
from forecast.errors.meta_model import ErrorModel
from forecast.ensemble.stacking import PrecisionModel
from forecast.ingest.legi_polls import merge_cells, unit_of
from forecast.legislative import rounds as R
from forecast.legislative.nowcast import SeatNowcast, fit_seat_nowcast
from forecast.legislative.pipeline import LEVELS, LegiModels, NationalAverage, SeatForecast, simulate_legislative

U = R.U


def _v(**shares) -> np.ndarray:
    v = np.zeros(len(R.UNITS))
    for k, s in shares.items():
        v[U[k]] = s
    return v


class TestFirstRoundRules:
    def test_outright_needs_quarter_of_registered(self):
        v = _v(RN=0.55, GAU=0.30, CEN=0.15)[None]
        win, _ = R.outcome_first_round(v, np.array([0.50]))          # 0.55 × 0.50 = 27.5% ≥ 25%
        assert win[0] == U["RN"]
        win, _ = R.outcome_first_round(v, np.array([0.40]))          # 22% < 25% → runoff
        assert win[0] == -1

    def test_qualification_at_12_5_percent_of_registered(self):
        v = _v(RN=0.36, GAU=0.30, CEN=0.22, DRO=0.12)[None]
        _, q = R.outcome_first_round(v, np.array([0.65]))           # CEN 14.3% ≥ 12.5 ; DRO 7.8%
        assert q[0, U["CEN"]] and not q[0, U["DRO"]]
        assert q[0].sum() == 3

    def test_top_two_fallback(self):
        v = _v(RN=0.40, GAU=0.35, CEN=0.25)[None]
        _, q = R.outcome_first_round(v, np.array([0.25]))           # only RN clears 12.5% of registered
        assert set(np.where(q[0])[0]) == {U["RN"], U["GAU"]}


class TestWithdrawals:
    def _model(self, p):
        return R.WithdrawalModel({(u, f): p for u in R.UNITS for f in (False, True)}, p)

    def test_top_two_never_withdraw(self):
        v = _v(RN=0.36, GAU=0.30, CEN=0.22, DRO=0.12)[None]
        q = np.array([[False] * 8])
        for k in ("RN", "GAU", "CEN"):
            q[0, U[k]] = True
        st = R.apply_withdrawals(v, q, self._model(0.999), np.zeros(1), np.random.default_rng(0))
        assert st[0, U["RN"]] and st[0, U["GAU"]] and not st[0, U["CEN"]]

    def test_zero_probability_keeps_everyone(self):
        v = _v(RN=0.36, GAU=0.30, CEN=0.22, DRO=0.12)[None]
        q = v > 0.2
        st = R.apply_withdrawals(v, q, self._model(1e-6), np.zeros(1), np.random.default_rng(0))
        assert (st == q).all()


class TestTransfers:
    def test_shares_sum_to_one_over_standing(self):
        tm = R.TransferMatrix(np.full((8, 8), 0.5) + np.eye(8) * 4.5, 0.0)
        v = np.random.default_rng(1).dirichlet(np.ones(8), size=5)
        st = np.zeros_like(v, dtype=bool)
        st[:, [U["RN"], U["GAU"]]] = True
        w = tm.second_round(v, st)
        assert np.allclose(w.sum(axis=1), 1.0)
        assert np.all(w[:, U["CEN"]] == 0)

    def test_own_voters_stay(self):
        A = np.full((8, 8), -5.0)
        np.fill_diagonal(A, R.OWN_UTILITY)
        tm = R.TransferMatrix(A, 0.0)
        v = _v(RN=0.4, GAU=0.35, CEN=0.25)[None]
        st = np.zeros_like(v, dtype=bool)
        st[0, [U["RN"], U["GAU"]]] = True
        w = tm.second_round(v, st)
        assert w[0, U["RN"]] == pytest.approx(0.4 / 0.75, abs=0.01)


class TestPollParsing:
    @pytest.mark.parametrize("label, year, unit", [
        ("NFP LFI", 2024, "GAU"), ("FI", 2017, "RAD"), ("FG", 2012, "RAD"), ("PS", 2017, "GAU"),
        ("LREM", 2017, "CEN"), ("UDI", 2017, "DRO"), ("RN and allies LR (UXD)[c]", 2024, "RN"),
        ("REC EXD", 2022, "EXD"), ("LO NPA EXG", 2022, "EXG"), ("Abs.", 2017, None), ("Lead", 2029, None),
    ])
    def test_unit_of(self, label, year, unit):
        assert unit_of(label, year) == unit

    def test_merged_cells_collapsed_when_total_improves(self):
        # one 28% NFP figure spread over four sub-columns
        cells = [("EXG", 1.0), ("GAU", 28.0), ("GAU", 28.0), ("GAU", 28.0), ("GAU", 28.0), ("CEN", 21.0),
                 ("DRO", 10.0), ("RN", 36.0), ("DIV", 4.0)]
        assert merge_cells(cells) == {"EXG": 1.0, "GAU": 28.0, "CEN": 21.0, "DRO": 10.0, "RN": 36.0, "DIV": 4.0}

    def test_separate_equal_values_kept(self):
        cells = [("RAD", 3.0), ("GAU", 3.0), ("CEN", 30.0), ("DRO", 30.0), ("RN", 34.0)]
        assert sum(merge_cells(cells).values()) == pytest.approx(100.0)


def test_nowcast_recovers_coefficients():
    rng = np.random.default_rng(0)
    units = ["GAU", "CEN", "DRO", "RN"]
    rows = []
    true = {"lean": 1.0, "lagged": 0.4, "has_lagged": 0.0, "inc": 0.5}
    for s in range(300):
        x = {u: (rng.normal(0, 0.4), rng.normal(0, 0.4), 1.0, float(rng.random() < 0.2)) for u in units}
        eta = np.array([np.log(0.25) + true["lean"] * a + true["lagged"] * b + true["inc"] * d
                        for (a, b, c, d) in x.values()]) + rng.normal(0, 0.05, 4)
        p = np.exp(eta) / np.exp(eta).sum()
        for u, (a, b, c, d), pi in zip(units, x.values(), p):
            rows.append({"seat": f"S{s}", "unit": u, "lean": a, "lagged": b, "has_lagged": c, "inc": d,
                         "presence": 0.0, "share": pi, "year": 2020})
    df = pl.DataFrame(rows)
    model = fit_seat_nowcast([(df, {u: 0.25 for u in units})])
    assert model.beta["lean"] == pytest.approx(1.0, abs=0.1)
    assert model.beta["lagged"] == pytest.approx(0.4, abs=0.1)
    assert model.beta["inc"] == pytest.approx(0.5, abs=0.1)


def test_simulation_seat_totals():
    units = ["GAU", "CEN", "DRO", "RN"]
    seats = [f"S{i:02d}" for i in range(20)]
    rng = np.random.default_rng(0)
    rows = []
    for s in seats:
        p = rng.dirichlet(np.ones(4) * 5)
        for u, pi in zip(units, p):
            rows.append({"seat": s, "unit": u, "mu_ens": float(np.log(pi)), "sd_ens": 0.2, "disagree": 0.0,
                         "region": "R1" if s < "S10" else "R2", "dept": s[:2], "cluster": "C0"})
    fc = SeatForecast(pl.DataFrame(rows), NationalAverage({u: 0.25 for u in units}, {}, 10), 7)
    A = np.full((8, 8), 0.3)
    np.fill_diagonal(A, R.OWN_UTILITY)
    models = LegiModels(
        nowcast=SeatNowcast({}, 0.1, ()), fundamentals=None, precision=PrecisionModel(),
        errors=ErrorModel(LEVELS, {l: 0.05 for l in LEVELS}, {l: 0.0 for l in LEVELS}, 1.0, 0.0, 0.0, 5.0, 0, "prior"),
        withdrawals=R.WithdrawalModel({}, 0.5), transfers=R.TransferMatrix(A, 0.0), national_sd=(0.08, 0.0),
    )
    sims = simulate_legislative(fc, models, {s: 0.6 for s in seats}, 0.1, default_config(), n_sims=450, seed=1)
    assert (sims.totals.sum(axis=1) == len(seats)).all()
    assert np.allclose(sims.seat_win_prob.sum(axis=1), 1.0)
    assert sims.totals.shape == (450, 8)

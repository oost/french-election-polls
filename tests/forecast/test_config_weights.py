"""Config loading and the weighting formulas taken from the post."""
from __future__ import annotations

import numpy as np
import pytest

from forecast.config import PollParams, load_config
from forecast.polls import weights as W
from forecast.polls.house_effects import HouseHistory, shrink


class TestConfig:
    def test_loads_with_provenance_for_every_parameter(self):
        cfg = load_config()
        assert cfg.polls.window_days == 90
        assert cfg.simulation.n_sims == 40_000
        assert cfg.provenance["polls.house_k_cycle"] == "post"
        assert cfg.provenance["legislative.qualify_registered"] == "law"
        assert set(cfg.provenance.values()) <= {"post", "law", "assumption", "estimated"}

    def test_nest_groups_load_as_a_hashable_tuple(self):
        cfg = load_config()
        assert cfg.nowcast.nest_groups == ("CEN+DRO",)
        hash(cfg.nowcast)

    def test_unknown_parameter_rejected(self, tmp_path):
        bad = tmp_path / "m.yaml"
        bad.write_text("polls:\n  nonsense: {value: 1, source: post}\n")
        with pytest.raises(KeyError):
            load_config(bad)

    def test_missing_provenance_rejected(self, tmp_path):
        bad = tmp_path / "m.yaml"
        bad.write_text("polls:\n  window_days: {value: 90}\n")
        with pytest.raises(ValueError):
            load_config(bad)


class TestDecay:
    p = PollParams()

    def test_endpoints(self):
        assert W.decay_rate(200, self.p) == pytest.approx(0.03)
        assert W.decay_rate(45, self.p) == pytest.approx(0.03)
        assert W.decay_rate(14, self.p) == pytest.approx(0.14)
        assert W.decay_rate(0, self.p) == pytest.approx(0.14)

    def test_ramp_is_monotonic(self):
        rates = W.decay_rate(np.arange(0, 60), self.p)
        assert np.all(np.diff(rates) <= 1e-12)

    def test_recency_weight(self):
        assert W.recency_weight(0, 100, self.p) == pytest.approx(1.0)
        assert W.recency_weight(10, 100, self.p) == pytest.approx(0.97 ** 10)
        assert W.recency_weight(10, 5, self.p) == pytest.approx(0.86 ** 10)


class TestSampleAndSponsor:
    p = PollParams()

    def test_sample_size_weight_capped(self):
        w = W.sample_size_weight(np.array([1000, 4000, 10_000, np.nan]), self.p)
        assert w == pytest.approx([1.0, 2.0, 2.0, 1.0])

    def test_partisan_shift_preserves_total(self):
        y = np.array([30.0, 20.0, np.nan, 50.0])
        mask = np.array([True, False, False, False])
        out = W.partisan_shift(y, mask, 2.0)
        assert np.nansum(out) == pytest.approx(np.nansum(y))
        assert out[0] == pytest.approx(28.0)
        assert np.isnan(out[2])

    def test_partisan_polls_half_weight(self):
        w = W.base_weight(np.array([1.0, 1.0]), np.array([1000, 1000]), np.array([False, True]), self.p)
        assert w[1] == pytest.approx(0.5 * w[0])


class TestAnchorSlack:
    def test_post_values(self):
        s = W.anchor_slack(np.arange(1, 7))
        assert s[0] == pytest.approx(2.0)
        assert s[4] == pytest.approx(1 / 3)
        assert s[5] == pytest.approx(1 / 3)      # floor
        assert np.all(np.diff(s[:5]) < 0)


class TestShrinkage:
    def test_cycle_wide_half_at_five(self):
        assert shrink(5, 5) == pytest.approx(0.5)
        assert shrink(50, 5) > 0.9

    def test_race_specific_skeptical(self):
        assert shrink(2, 20) < 0.1

    def test_prior_halves_per_older_cycle(self):
        h = HouseHistory()
        h.leans[("Ifop", "EXD")] = [(2017, 0.10, 10.0), (2022, 0.00, 10.0)]
        lean, W_ = h.prior("Ifop", "EXD", before=2027, decay=0.5)
        # 2022 weight 10, 2017 weight 5 → (0*10 + 0.1*5) / 15
        assert lean == pytest.approx(0.5 / 15)
        assert W_ == pytest.approx(15.0)

    def test_prior_ignores_current_and_future_cycles(self):
        h = HouseHistory()
        h.leans[("Ifop", "EXD")] = [(2022, 0.2, 10.0), (2027, 0.5, 10.0)]
        lean, W_ = h.prior("Ifop", "EXD", before=2022, decay=0.5)
        assert (lean, W_) == (0.0, 0.0)

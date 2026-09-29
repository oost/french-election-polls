"""Runoff model (PLAN.md › Presidential model, step 12), scipy v0.

Polled matchups get the three-pass average. Every matchup — polled or not —
also gets a transfer-based now-cast: each first-round electorate k splits
between A, B and abstention with

    η_kX = α − β·|pos_k − pos_X| + r_X,   P(X | k) = e^{η_kX} / (1 + e^{η_kA} + e^{η_kB})

(pos = a bloc's left–right position, r_X = candidate X's runoff appeal), so

    share_A = Σ_k v_k P(A|k) / Σ_k v_k (P(A|k) + P(B|k)).

Fitted to the polled matchups (weights = polls × staleness decay), then
polled matchups are re-anchored to their own average with the slack rule. In
a simulation the runoff mean moves with the simulated first round:
logit(A) = anchored + [T(v_sim) − T(v_expected)] + error. The Stan transfer
model (M4) replaces the positional parameterisation with a full bloc × bloc
affinity matrix fitted to polling-station flows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from forecast.polls import weights as W
from forecast.polls.ewma import AverageResult
from forecast.polls.lineups import race_observations

BLOC_POSITION = {"EXG": -2.0, "GAU": -1.0, "CEN": 0.0, "DIV": 0.0, "DRO": 1.0, "EXD": 2.0}
OWN_CANDIDATE_UTILITY = 6.0


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class TransferModel:
    alpha: float
    beta: float
    appeal: dict[str, float]
    blocs: dict[str, str]

    def utilities(self, voters: list[str], target: str) -> np.ndarray:
        pos_t = BLOC_POSITION[self.blocs[target]]
        pos_v = np.array([BLOC_POSITION[self.blocs[k]] for k in voters])
        eta = self.alpha - self.beta * np.abs(pos_v - pos_t) + self.appeal.get(target, 0.0)
        return np.where(np.array(voters) == target, OWN_CANDIDATE_UTILITY, eta)

    def share_a(self, a: str, b: str, voters: list[str], v: np.ndarray) -> np.ndarray:
        """A's runoff share; `v` is (…, K) first-round shares of `voters`."""
        ea = np.exp(self.utilities(voters, a))
        eb = np.exp(self.utilities(voters, b))
        pa = ea / (1 + ea + eb)
        pb = eb / (1 + ea + eb)
        num = (v * pa).sum(axis=-1)
        den = (v * (pa + pb)).sum(axis=-1)
        return num / den


@dataclass(frozen=True)
class MatchupObs:
    a: str
    b: str
    share_a: float          # averaged share of A (0–1)
    n_polls: int
    staleness: int
    weight: float


def matchup_observations(result: AverageResult, day: int, staleness_decay_days: float = 60.0) -> list[MatchupObs]:
    out = []
    for o in race_observations(result, day, staleness_decay_days):
        if len(o.options) != 2:
            continue
        a, b = sorted(o.options)
        share_a = float(np.exp(o.log_share[list(o.options).index(a)]))
        out.append(MatchupObs(a, b, share_a, o.n_polls, o.staleness, o.weight))
    return out


def fit_transfer_model(
    obs: list[MatchupObs],
    first_round: dict[tuple[str, str], tuple[list[str], np.ndarray]],
    blocs: dict[str, str],
    appeal_sd: float = 0.5,
) -> TransferModel:
    """Weighted least squares on logit(share_A) with normal priors
    (α ~ N(1, 1), β ~ N(1, 0.5), appeal ~ N(0, appeal_sd))."""
    names = sorted({c for o in obs for c in (o.a, o.b)})

    def unpack(x):
        return x[0], x[1], dict(zip(names, x[2:]))

    def loss(x):
        alpha, beta, appeal = unpack(x)
        m = TransferModel(alpha, beta, appeal, blocs)
        total = 0.0
        for o in obs:
            voters, v = first_round[(o.a, o.b)]
            pred = m.share_a(o.a, o.b, voters, v)
            total += o.weight * (_logit(o.share_a) - _logit(pred)) ** 2
        prior = (alpha - 1) ** 2 + ((beta - 1) / 0.5) ** 2 + sum((r / appeal_sd) ** 2 for r in appeal.values())
        return total + prior

    x0 = np.concatenate([[1.0, 1.0], np.zeros(len(names))])
    sol = minimize(loss, x0, method="L-BFGS-B")
    alpha, beta, appeal = unpack(sol.x)
    return TransferModel(float(alpha), float(beta), {k: float(v) for k, v in appeal.items()}, blocs)


@dataclass
class RunoffModel:
    transfer: TransferModel
    polled: dict[tuple[str, str], MatchupObs] = field(default_factory=dict)
    anchor_one: float = 2.0
    anchor_floor: float = 1 / 3
    anchor_floor_n: float = 5.0

    def expected(self, a: str, b: str, voters: list[str], v_mean: np.ndarray) -> float:
        """Anchored expected share of A (0–1) given the expected first round."""
        key = tuple(sorted((a, b)))
        pred = float(self.transfer.share_a(a, b, voters, v_mean))
        obs = self.polled.get(key)
        if obs is None:
            return pred
        polled_a = obs.share_a if obs.a == a else 1 - obs.share_a
        slack = float(W.anchor_slack(obs.n_polls, self.anchor_one, self.anchor_floor, self.anchor_floor_n)) / 100
        return polled_a + float(np.clip(pred - polled_a, -slack, slack))

    def simulate_logit(self, a: str, b: str, voters: list[str], v_mean: np.ndarray, v_sim: np.ndarray) -> np.ndarray:
        """logit(A share) per simulation before the runoff error term."""
        base = _logit(self.expected(a, b, voters, v_mean))
        shift = _logit(self.transfer.share_a(a, b, voters, v_sim)) - _logit(self.transfer.share_a(a, b, voters, v_mean))
        return base + shift

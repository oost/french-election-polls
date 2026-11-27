"""The three-pass exponentially weighted polling average (post § 3.1.1).

    pass 1  trendline-adjusted, recency-weighted average per race and day
    pass 2  subtract mode effects and pollster house effects, re-average
    pass 3  population adjustment (a no-op when every poll uses one frame)

A "race" is a first-round lineup (keyed on its major candidates, so a poll
that omits a 0.3% candidate isn't split into its own race), a runoff matchup,
or a generic/legislative configuration. Everything is on the log-share scale
with rows renormalised (softmax), which is the multi-option analog of the
post's two-party margins.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from forecast.config import PollParams
from forecast.polls import weights as W
from forecast.polls.house_effects import HouseEffects, HouseHistory, bloc_leans, fit_house_effects, group_effects
from forecast.polls.trend import Trend, fit_trend


@dataclass
class PollFrame:
    election_day: date
    kind: str
    options: list[str]
    blocs: list[str | None]
    rows: pl.DataFrame          # one row per poll row, aligned with Y
    Y: np.ndarray               # (R, C) shares in points, NaN = not listed
    race: np.ndarray            # (R,) race index
    race_ids: list[str]
    race_options: list[tuple[str, ...]]
    t: np.ndarray               # (R,) midpoint, days relative to election (negative before)

    @property
    def n_rows(self) -> int:
        return len(self.t)


def _race_key(options: tuple[str, ...]) -> str:
    return "+".join(options)


def build_frame(
    long: pl.DataFrame,
    election_day: date,
    as_of: date | None = None,
    minor_pct: float = 2.0,
    partisan_shift_pts: float = 2.0,
) -> PollFrame:
    """Rows of one election and kind, as known on `as_of` (publish_est ≤ as_of)."""
    kinds = long["kind"].unique().to_list()
    if len(kinds) != 1:
        raise ValueError(f"build_frame expects one kind, got {kinds}")
    kind = kinds[0]
    df = long
    if as_of is not None:
        df = df.filter(pl.col("publish_est") <= as_of)
    if df.is_empty():
        raise ValueError("no polls available")

    options = sorted(df["option_id"].unique().to_list())
    bloc_map = dict(df.select("option_id", "bloc").unique(subset="option_id").iter_rows())
    blocs = [bloc_map.get(o) for o in options]
    col = {o: i for i, o in enumerate(options)}

    meta_cols = ["row_id", "poll_id", "pollster", "pollster_raw", "midpoint", "mode", "population", "partisan",
                 "sponsor_bloc", "overlap_weight", "n_eff", "sample_size", "n_lineups", "lineup_id"]
    rows = df.select(meta_cols).unique(subset="row_id", keep="first", maintain_order=True)
    ridx = {r: i for i, r in enumerate(rows["row_id"].to_list())}
    Y = np.full((rows.height, len(options)), np.nan)
    for rid, oid, share in df.select("row_id", "option_id", "share").iter_rows():
        Y[ridx[rid], col[oid]] = share

    # partisan-sponsored polls: shift away from the sponsor (post: ~2 pts)
    sponsor = rows["sponsor_bloc"].to_list()
    for i, sb in enumerate(sponsor):
        if sb:
            mask = np.array([b == sb for b in blocs])
            Y[i] = W.partisan_shift(Y[i], mask, partisan_shift_pts)

    if kind == "matchup":
        keys = [tuple(o for o in options if not np.isnan(Y[i, col[o]])) for i in range(len(Y))]
    else:
        median = np.nanmedian(Y, axis=0)
        major = {o for o, m in zip(options, median) if m >= minor_pct}
        keys = [tuple(o for o in options if not np.isnan(Y[i, col[o]]) and o in major) for i in range(len(Y))]
    race_ids: list[str] = []
    race_options: list[tuple[str, ...]] = []
    race = np.zeros(len(Y), dtype=int)
    seen: dict[tuple[str, ...], int] = {}
    for i, k in enumerate(keys):
        if k not in seen:
            seen[k] = len(race_ids)
            race_ids.append(_race_key(k))
            race_options.append(k)
        race[i] = seen[k]

    t = np.array([(d - election_day).days for d in rows["midpoint"].to_list()], dtype=int)
    return PollFrame(election_day, kind, options, blocs, rows, Y, race, race_ids, race_options, t)


def row_log_shares(Y: np.ndarray, floor_pts: float) -> np.ndarray:
    """log of each row's shares renormalised over the options it lists."""
    y = np.where(np.isnan(Y), np.nan, np.maximum(Y, floor_pts))
    tot = np.nansum(y, axis=1, keepdims=True)
    return np.log(y / tot)


def _softmax_rows(logx: np.ndarray) -> np.ndarray:
    m = np.nanmax(logx, axis=-1, keepdims=True)
    e = np.exp(logx - m)
    return e / np.nansum(e, axis=-1, keepdims=True)


@dataclass
class AverageResult:
    frame: PollFrame
    averages: pl.DataFrame      # day, date, race_id, option_id, share, log_share, sd_log, n_polls, n_firms, weight_sum
    house: HouseEffects
    house_table: pl.DataFrame
    mode_effects: dict
    population_effects: dict
    trend: Trend
    adjusted_log: np.ndarray    # (R, C) log shares after mode/house/population adjustments
    residuals: np.ndarray       # (R, C) pass-1 leave-one-out residuals
    params: PollParams
    base_w: np.ndarray          # (R,) time-independent poll weights

    def averages_at(self, days: list[int]) -> list[dict]:
        """Pass-2 race averages at extra days (same adjusted polls and trend)."""
        return _race_averages_at(self.frame, self.adjusted_log, self.trend, self.base_w, days, self.params)

    def race_average(self, race_id: str, day: int | None = None) -> pl.DataFrame:
        df = self.averages.filter(pl.col("race_id") == race_id)
        if day is not None:
            df = df.filter(pl.col("day") == day)
        return df


def _loo_reference(frame: PollFrame, logy: np.ndarray, trend: Trend, base_w: np.ndarray, p: PollParams) -> np.ndarray:
    """Leave-one-poll-out pass-1 average of each row's race at the row's own
    date (two-sided window), on the row's option set, as log shares. Falls
    back to the trend model θ(t) + λ[race] when the race has no other polls."""
    R, C = logy.shape
    ref = np.full((R, C), np.nan)
    poll = frame.rows["poll_id"].to_numpy()
    theta_rows = trend.at(frame.t)
    for s in range(len(frame.race_ids)):
        idx = np.where(frame.race == s)[0]
        for i in idx:
            others = idx[(idx != i) & (poll[idx] != poll[i]) & (np.abs(frame.t[idx] - frame.t[i]) <= p.window_days)]
            present = ~np.isnan(logy[i])
            if len(others):
                h = max(-frame.t[i], 0)
                w = base_w[others] * W.recency_weight(np.abs(frame.t[others] - frame.t[i]), h, p)
                x = logy[others] + theta_rows[i][None, :] - theta_rows[others]
                shares = _softmax_rows(np.where(np.isnan(x), -np.inf, x))
                shares = np.where(np.isnan(logy[others]), np.nan, shares)
                num = np.nansum(w[:, None] * shares, axis=0)
                den = np.sum(w[:, None] * ~np.isnan(shares), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    avg = np.where(den > 0, num / den, np.nan)
                ok = present & ~np.isnan(avg)
                if ok.sum() == present.sum():
                    v = avg[ok] / avg[ok].sum()
                    ref[i, ok] = np.log(v)
                    continue
            lam = np.nan_to_num(trend.lam[s], nan=0.0)
            x = np.where(present, theta_rows[i] + lam, -np.inf)
            v = _softmax_rows(x[None, :])[0]
            ref[i, present] = np.log(v[present])
    return ref


def _race_averages_at(
    frame: PollFrame, logy: np.ndarray, trend: Trend, base_w: np.ndarray, days: list[int], p: PollParams,
) -> list[dict]:
    out: list[dict] = []
    firms = frame.rows["pollster"].to_numpy()
    polls = frame.rows["poll_id"].to_numpy()
    n_eff = np.nan_to_num(frame.rows["n_eff"].to_numpy().astype(float), nan=800.0)
    for d in days:
        theta_d = trend.at(d)
        for s, rid in enumerate(frame.race_ids):
            idx = np.where((frame.race == s) & (frame.t <= d) & (frame.t >= d - p.window_days))[0]
            if len(idx) == 0:
                continue
            h = max(-d, 0)
            w = base_w[idx] * W.recency_weight(d - frame.t[idx], h, p)
            x = logy[idx] + theta_d[None, :] - trend.at(frame.t[idx])
            shares = _softmax_rows(np.where(np.isnan(x), -np.inf, x))
            shares = np.where(np.isnan(logy[idx]), np.nan, shares)
            has = ~np.isnan(shares)
            den = np.sum(w[:, None] * has, axis=0)
            num = np.nansum(w[:, None] * shares, axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                avg = np.where(den > 0, num / den, np.nan)
            opts = ~np.isnan(avg)
            avg[opts] = avg[opts] / avg[opts].sum()
            wn = w / w.sum()
            n_star = 1.0 / np.sum(wn ** 2 / np.maximum(n_eff[idx], 50.0))
            for c in np.where(opts)[0]:
                ok = has[:, c]
                lv = np.log(np.maximum(shares[ok, c], 1e-6))
                wc = w[ok] / w[ok].sum()
                disp = float(np.sum(wc ** 2 * (lv - np.log(avg[c])) ** 2))
                samp = float((1 - avg[c]) / (n_star * avg[c]))
                out.append({
                    "day": int(d), "race_id": rid, "option_id": frame.options[c],
                    "share": float(100 * avg[c]), "log_share": float(np.log(avg[c])),
                    "sd_log": float(np.sqrt(disp + samp)), "n_polls": int(len(set(polls[idx]))),
                    "n_firms": int(len(set(firms[idx]))), "weight_sum": float(w.sum()),
                    "last_poll_day": int(frame.t[idx].max()),
                })
    return out


def polling_average(
    long: pl.DataFrame,
    election_day: date,
    p: PollParams,
    as_of: date | None = None,
    eval_days: list[int] | None = None,
    history: HouseHistory | None = None,
    prior_cap: float | None = None,
    cycle: int | None = None,
    minor_pct: float = 2.0,
) -> AverageResult:
    """Run the three passes for one election and kind (lineup | matchup | generic).

    `eval_days` are days relative to the election (e.g. [-30, -2]); default is
    every day from the first poll to `as_of` (or the last poll).
    """
    frame = build_frame(long, election_day, as_of=as_of, minor_pct=minor_pct,
                        partisan_shift_pts=p.partisan_shift_pts)
    rows = frame.rows
    base_w = W.base_weight(rows["overlap_weight"].to_numpy(), rows["n_eff"].to_numpy(),
                           rows["partisan"].to_numpy().astype(bool), p)
    row_w = np.nan_to_num(rows["overlap_weight"].to_numpy().astype(float), nan=1.0) / rows["n_lineups"].to_numpy()
    last = (as_of - election_day).days if as_of is not None else int(frame.t.max())
    grid = np.arange(int(frame.t.min()), max(last, int(frame.t.max())) + 1)
    logy = row_log_shares(frame.Y, p.share_floor)

    # pass 1
    trend1 = fit_trend(logy, frame.t, frame.race, len(frame.race_ids), row_w, grid, p)
    resid = logy - _loo_reference(frame, logy, trend1, base_w, p)

    # pass 2: mode, then house effects on what mode doesn't explain
    mode_eff, mode_rows = group_effects(resid, rows["mode"].to_numpy(), row_w, p.mode_k)
    firms = sorted(set(rows["pollster"].to_list()))
    firm_idx = np.array([firms.index(f) for f in rows["pollster"].to_list()])
    cap = p.prior_cap_pres if prior_cap is None else prior_cap
    house = fit_house_effects(resid - mode_rows, firm_idx, firms, frame.race, row_w, frame.blocs, p, cap,
                              history=history, cycle=cycle)
    house_rows = house.per_row(firm_idx, frame.race)

    # pass 3: population frames (no-op when there is only one)
    pop_eff, pop_rows = group_effects(resid - mode_rows - house_rows, rows["population"].to_numpy(), row_w, p.house_k_race)

    adj = logy - np.nan_to_num(mode_rows) - np.nan_to_num(house_rows) - np.nan_to_num(pop_rows)
    adj = np.where(np.isnan(logy), np.nan, adj)
    with np.errstate(divide="ignore"):   # absent options → log(0), masked on the next line
        adj = np.log(_softmax_rows(np.where(np.isnan(adj), -np.inf, adj)))
    adj = np.where(np.isnan(logy), np.nan, adj)
    trend2 = fit_trend(adj, frame.t, frame.race, len(frame.race_ids), row_w, grid, p)

    days = eval_days if eval_days is not None else list(range(int(frame.t.min()), last + 1))
    avgs = _race_averages_at(frame, adj, trend2, base_w, days, p)
    averages = pl.DataFrame(avgs, schema={
        "day": pl.Int64, "race_id": pl.Utf8, "option_id": pl.Utf8, "share": pl.Float64, "log_share": pl.Float64,
        "sd_log": pl.Float64, "n_polls": pl.Int64, "n_firms": pl.Int64, "weight_sum": pl.Float64,
        "last_poll_day": pl.Int64,
    }).with_columns((pl.lit(election_day) + pl.duration(days=pl.col("day"))).cast(pl.Date).alias("date"))
    return AverageResult(
        frame=frame, averages=averages, house=house, house_table=house.table(frame.options, frame.blocs),
        mode_effects=mode_eff, population_effects=pop_eff, trend=trend2, adjusted_log=adj, residuals=resid,
        params=p, base_w=base_w,
    )


def history_from_results(results: dict[int, AverageResult]) -> HouseHistory:
    """Historical-prior store built from earlier cycles' house effects."""
    hist = HouseHistory()
    for cycle, res in results.items():
        hist.add(cycle, bloc_leans(res.house_table))
    return hist


def main_race(result: AverageResult, day: int, options: set[str] | None = None) -> str | None:
    """The race to report at `day`: the one matching `options` exactly (by its
    major candidates) if given, else the one with the most polls."""
    df = result.averages.filter(pl.col("day") == day)
    if df.is_empty():
        return None
    if options is not None:
        for rid, opts in zip(result.frame.race_ids, result.frame.race_options):
            if set(opts) == options & set(result.frame.options) and rid in df["race_id"].to_list():
                return rid
    counts = df.group_by("race_id").agg(pl.col("n_polls").first()).sort("n_polls", descending=True)
    return counts["race_id"][0]


def date_of(election_day: date, day: int) -> date:
    return election_day + timedelta(days=day)

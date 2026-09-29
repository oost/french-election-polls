# File guide

Where everything lives, what each file does, and which files you are expected to edit. Paths are relative to the repository root.

## Top level

| Path | What it is |
| --- | --- |
| `PLAN.md` | Design document: how the post maps to France, the parameters, deliberate departures, and implementation status. |
| `README.md` | Project introduction and run commands. |
| `pyproject.toml` | Dependencies (`[stan]` adds cmdstanpy), pytest settings and markers (`slow`, `network`). |
| `config/model.yaml` | **Every model parameter, each with its source** (`post`, `law`, `assumption` or `estimated`). Edit here, not in code. |
| `polls_helpers.py`, `french_elections_polls.ipynb` | The original polling-analysis tools (kernel smoother, charts). The forecast reuses their parsers. |
| `forecast_2027.ipynb` | Configuration-only notebook: runs both forecasts and shows the backtest charts. |
| `wiki_cache/` | Gzipped HTML of the pinned Wikipedia revisions, committed so the pipeline runs offline and reproducibly. |
| `docs/` | This documentation. |

## Inputs you edit — `data/reference/`

| File | Contents | When to edit |
| --- | --- | --- |
| `scenarios_pres_2027.yaml` | Lineup mixture: slot probabilities (e.g. Le Pen 0.95 / Bardella 0.05), fixed and optional candidates, and the January centrist decision rule. | Whenever the field changes; date and disclose these judgment calls. |
| `scenarios_legi.yaml` | Assumed legislative date, united vs split left, LFI's share of a split left, incumbents' re-run rate, turnout spread. | Before running `forecast legi`. |
| `candidates_pres.csv` | One row per candidate per election (1981–2027): party, bloc, incumbency, experience, results. | When a new name appears in the polls (the ingest report lists unmatched headers). |
| `pages.yaml` | Wikipedia pages, pinned revisions, and the rules that decide which tables to read. | To pick up new polls: bump the revision, then `forecast ingest`. |
| `pollster_aliases.csv`, `pollster_meta.csv` | Raw pollster name → firm; survey mode by year; publication lag. | When a new pollster appears. |
| `party_bloc_map.csv` | Ministry party labels ("nuances") → model units, per legislative election. | For a new legislative election. |
| `departement_region.csv` | Département → region, for the regional error level. | Rarely. |
| `priors.yaml` | Weakly informative priors used when no history exists: fundamentals coefficients, prior error scales. | Only with a stated reason. |

## Generated data (gitignored)

| Path | Contents |
| --- | --- |
| `data/raw/` | Downloaded parquet/CSV results, plus `manifest.json` (URL, sha256, date). |
| `data/processed/` | `polls_long.parquet` (one row per poll row × candidate), `lineups.parquet`, `events.parquet`, cached horizon models. |
| `data/cache/fits/` | Cached Stan posterior draws, keyed by a hash of model code + data + settings. |
| `data/archive/` | Backtest archives: `pres_backtest*.parquet`, `legi_backtest*.parquet`. The ensemble and meta-model are trained on these. |
| `data/outputs/` | Forecast tables, reports and charts. |

## The `forecast` package

### Entry points

| File | Purpose |
| --- | --- |
| `forecast/__main__.py` | Command line: `ingest`, `average`, `run`, `backtest`, `legi`, `legi-backtest`. |
| `forecast/config.py` | Loads `config/model.yaml` into frozen dataclasses and keeps each parameter's provenance. |
| `forecast/paths.py` | Where every directory lives. |

### `ingest/` — getting data in

| File | Purpose |
| --- | --- |
| `pages.py` | Loads `pages.yaml`, downloads and caches pinned revisions, and selects tables by rule. |
| `registry.py` | Matches candidate column headers, scored so "Philippe Poutou" doesn't match Édouard Philippe; canonicalises pollster names. |
| `wiki_long.py` | Parses a table into the long poll format; drops results and event rows; applies the multi-way-ballot rule; computes rolling-tracker overlap weights. |
| `legi_polls.py` | National legislative polls: maps party columns to units and undoes Wikipedia's merged cells. |
| `harmonised.py` | data.gouv.fr polling-station results → votes per constituency (2012–2022). |
| `ministry.py` | Ministry per-constituency files for 2024. |
| `build.py` | Writes and loads `data/processed/polls_long.parquet`. |

### `polls/` — the polls model

| File | Purpose |
| --- | --- |
| `weights.py` | Recency decay, sample-size and partisan weights, and the anchoring slack $s(n)$. |
| `trend.py` | Cross-lineup candidate trend (kernel smoother with lineup effects). |
| `house_effects.py` | Shrunk house, mode and population effects; historical prior store. |
| `ewma.py` | The three-pass average. `polling_average()` is the function to call. |
| `lineups.py` | Nested-logit substitution model, anchoring, and imputation error for unpolled lineups. |
| `horizon.py` | $\sigma(h, s)$: how far an average can still move, fitted on past cycles. |
| `drift.py` | Late-consolidation drift. Implemented, tested, and **disabled** (it hurt out of sample). |

### `fundamentals/` — history-only second opinion

| File | Purpose |
| --- | --- |
| `features_pres.py` | Builds the four predictors for any lineup. |
| `dlm.py` | Fits `fundamentals_dlm.stan` and turns draws into predictive means and sds. |

### `stan/` — Bayesian models

| File | Purpose |
| --- | --- |
| `fundamentals_dlm.stan` | Softmax regression with random-walk coefficients and a Dirichlet likelihood. Also used for the legislative fundamentals, with seats as races. |
| `error_meta.stan` | Error meta-model: shared shocks plus an individual error, Student-t. |
| `build.py` | Compiles models once and caches fits by hash. Reads draws with `stan_variable()`. |

### `ensemble/`, `errors/`, `simulate/`

| File | Purpose |
| --- | --- |
| `ensemble/stacking.py` | Out-of-sample precision weights and the stacking multiplier; `combine()`. |
| `errors/meta_model.py` | Prepares archive rows for `error_meta.stan`; `ErrorModel` gives shock sds by horizon. |
| `simulate/scenarios.py` | Parses the scenario YAML; enumerates lineups and decision variants. |
| `simulate/runoff.py` | Presidential runoff: matchup averages, transfer model, anchoring. |
| `simulate/presidential.py` | The 40,000-draw presidential simulator: two-stage shocks, the January decision, first round, runoff. |

### `presidential/` and `legislative/`

| File | Purpose |
| --- | --- |
| `presidential/pipeline.py` | `run_forecast()`: everything from the poll table to the summary tables; `write_averages()`. |
| `legislative/data.py` | Seat results, winners, presidential results by seat, lean, lagged results, incumbents. |
| `legislative/features.py` | Seat × unit design table. |
| `legislative/nowcast.py` | Seat now-cast from the national polling average. |
| `legislative/rounds.py` | Two-round law, withdrawal model, transfer matrix. |
| `legislative/clusters.py` | k-means clusters of seats by voting pattern. |
| `legislative/pipeline.py` | Components, ensemble, chunked simulation, backtest, `run_legislative_forecast()`. |

### `backtest/` and `report/`

| File | Purpose |
| --- | --- |
| `backtest/harness.py` | Chronological, strictly out-of-sample presidential backtest; writes the archive. |
| `backtest/metrics.py` | Error by horizon, interval coverage, calibration buckets, Brier score, log loss. |
| `backtest/charts.py`, `backtest/report.py` | Backtest charts and text report (`data/outputs/backtest_pres/`). |
| `report/charts.py` | Forecast charts used by the notebook. |

## Tests — `tests/forecast/`

| File | Covers |
| --- | --- |
| `test_config_weights.py` | Config provenance; decay, shrinkage and slack formulas; the historical prior. |
| `test_ingest.py` | Header matching, pollster aliases, a synthetic 2027 page, and a regression against the committed CSVs. |
| `test_ewma.py` | The three-pass average recovers known truth and house effects on a synthetic election. |
| `test_nowcast_simulation.py` | Scenarios, nested logit, runoff, simulation invariants, the centrist decision rule. |
| `test_legislative.py` | Two-round law, withdrawals, transfers, poll parsing, now-cast recovery, seat totals. |
| `test_stan_models.py` | `slow`: both Stan models recover known parameters from simulated data. |
| `synthetic.py` | Generates the synthetic elections used above. |

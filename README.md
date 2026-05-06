# French Presidential Elections — First-Round Polling Analysis

Polars-powered analysis of opinion polls for the last five French presidential
elections (2002, 2007, 2012, 2017, 2022). Polls are scraped from Wikipedia,
LOESS-smoothed, and plotted alongside the actual first-round result for each
candidate.

## Layout

| File | What it does |
| --- | --- |
| [`french_elections_polls.ipynb`](french_elections_polls.ipynb) | Notebook: download → process → chart. Configuration only; logic lives in the helpers module. |
| [`polls_helpers.py`](polls_helpers.py) | All parsing, processing, and plotting helpers. |
| [`test_polls_helpers.py`](test_polls_helpers.py) | Pytest suite covering parsing edge cases, row validation, the data-pipeline path with mocked Wikipedia tables, and the relative-day analysis. |
| [`polling_data/`](polling_data/) | CSV exports written by the notebook (one per election year). |

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install polars pandas numpy matplotlib scipy statsmodels beautifulsoup4 lxml pytest

# tests
pytest -q

# notebook
jupyter notebook french_elections_polls.ipynb
```

The notebook auto-installs missing packages on first run.

## Configuration

In the notebook's first code cell after the imports:

```python
ELECTION_ROUND = 'first'   # 'first' or 'second'
TIME_CUTOFF    = '1y'      # '1y', '6m', '3m', '30d', or None for all polls
```

For the *Polling Change Analysis* section:

```python
START_DAYS = -150            # 150 days before the election
END_DAYS   = None            # None → compare against the actual first-round result
                             # or set to e.g. -30 to compare two LOESS-smoothed days
```

## What the charts show

For each year (1-year window before the election by default):

- **Faded scatter** — every individual poll for each candidate.
- **Solid line** — LOESS-smoothed trend.
- **Solid filled circle on the election date** — the actual first-round
  result, with the candidate's name and percentage labeled to the right.

Candidates are color-coded by political family (far-left red, socialist
pink, ecologist green, centrist gold, right blue, far-right navy).

## How polls are filtered

1. **Table detection** — Wikipedia tables are kept only if they contain a
   "Polling firm" / "Polling organisation" column and a date/fieldwork column.
2. **Row validation** — first-round rows need 3+ candidate values, all in
   the 0.5–40 % band. The 40 % ceiling rejects head-to-head columns that
   sometimes appear in the same tables (e.g., "Macron vs Mélenchon" projections).
3. **Date boundary** — polls dated on or after the election date are
   discarded (exit polls and post-election projections leak into the
   Wikipedia tables for some years).
4. **Mean threshold** — after extraction, candidates whose mean polling is
   ≤ `MIN_MEAN_POLL_PCT` (default 1.0 %) are dropped from the chart.

## Tests

The pytest suite uses synthetic / mocked Wikipedia tables, so it runs offline
in well under a second. Coverage:

- `_extract_numeric` — None, NaN, zero, negatives, percent strings, dashes.
- `parse_date` — ISO, "8 April 2022", abbreviated month, date ranges, NaN.
- `parse_time_cutoff`, `filter_by_time_cutoff` — `1y` / `6m` / `30d` / `None`.
- `is_valid_first_round_row`, `is_valid_second_round_row` — every accept/reject branch.
- `extract_poll_records` — first-round table, second-round leak rejection,
  empty / malformed tables.
- `drop_low_polling_candidates` — threshold cutoff, protected columns.
- `process_year_data` — date-boundary filter, low-poll drop, with an injected
  fake fetcher.
- `FIRST_ROUND_RESULTS` — every year present, winners match history, all
  values in (0, 50 %], yearly totals ~100 %, every candidate kept in the lists
  has a known result.
- `create_relative_day_charts` — both `end_days=int` and `end_days=None`
  modes, plus a numerical check that the bar height for `end_days=None`
  equals `actual_result − LOESS_at_start`.

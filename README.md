# French Presidential Elections — First-Round Polling Analysis

Polars-powered analysis of opinion polls for the last five French presidential
elections (2002, 2007, 2012, 2017, 2022). Polls are parsed from pinned
Wikipedia revisions, cleaned to one row per national poll, smoothed over time,
and plotted alongside the actual first-round result for each candidate.

## Layout

| File | What it does |
| --- | --- |
| [`french_elections_polls.ipynb`](french_elections_polls.ipynb) | Notebook: load → clean → chart. Configuration only; logic lives in the helpers module. |
| [`polls_helpers.py`](polls_helpers.py) | All parsing, cleaning, smoothing, and plotting helpers. |
| [`test_polls_helpers.py`](test_polls_helpers.py) | Pytest suite: unit tests on synthetic tables/pages plus regression tests on the cached Wikipedia snapshots. |
| [`wiki_cache/`](wiki_cache/) | Gzipped HTML of the pinned Wikipedia revisions (`{year}_{revision}.html.gz`). |
| [`polling_data/`](polling_data/) | Cleaned CSV exports written by the notebook (one per election year). |

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install polars pandas numpy matplotlib beautifulsoup4 lxml pytest

# tests
pytest -q

# notebook
jupyter notebook french_elections_polls.ipynb
```

The notebook auto-installs missing packages on first run. Everything runs
offline from `wiki_cache/`; a page is only downloaded if its cache file is
missing.

## Data source and reproducibility

Each election's page is pinned to a revision in `WIKIPEDIA_REVISIONS`
(`polls_helpers.py`). To pick up later edits to a page, bump its revision ID
and delete the old cache file. The next run downloads the new revision.

## Configuration

In the notebook's configuration cell:

```python
ELECTION_ROUND = 'first'   # 'first' or 'second'
TIME_CUTOFF    = '1y'      # '1y', '6m', '3m', '30d', or None for all polls
```

For the *Polling Change Analysis* section:

```python
START_DAYS = -days_until_next_election()   # as far out as 2027 is today
END_DAYS   = None   # None → split into poll movement + final polling error
                    # or e.g. -30 to compare two smoothed days
```

The default start day is recomputed on every run from `NEXT_ELECTION_DATE`
(the 2027 first round, 18 April 2027), so past campaigns are compared from the
point the 2027 race has reached. Set `START_DAYS` to a fixed number (e.g.
`-365`) to override it.

Module-level knobs in `polls_helpers.py`: `MIN_SUPPORT_PCT` (charting
threshold, 1 %), `SMOOTHING_BANDWIDTH_DAYS` (10), `MIN_EFFECTIVE_POLLS` (2),
and `FINAL_POLL_DAY` (−2).

## How polls are cleaned

1. **National tables only.** Tables are classified by their Wikipedia section
   heading ("First round" / "Second round"). Tables under *By region*, *By
   department*, *By commune* or *By first round vote* are excluded. Those are
   local polls or vote-transfer crosstabs, not national voting intention.
2. **Candidates who ran.** Columns are matched for every candidate on the
   first-round ballot (the keys of `FIRST_ROUND_RESULTS`). Surnames are matched
   as whole words after repairing headers that Wikipedia wraps across lines
   (e.g. "Dupont- Aignan").
3. **Row checks.** First-round rows need 3+ candidates with each value ≤ 50 %
   and the total ≤ 105 %. Second-round rows must come from a table containing
   exactly the two finalists, with shares summing to about 100 %. Rows that
   report a previous election's result ("2017 election") are dropped, as are
   polls whose fieldwork ended on or after election day.
4. **One row per poll.** Wikipedia lists a poll once per hypothetical lineup
   (e.g. with Bertrand instead of Pécresse, or with Villepin instead of
   Sarkozy). Rows sharing pollster, fieldwork dates and sample size are
   collapsed to the lineup closest to the real ballot. That is the one with the
   most actual candidates, then the largest combined share for them.
   `n_scenarios` records how many lineups the poll tested.
5. **Dates.** A poll is placed at its fieldwork midpoint. `fieldwork_start`
   and `fieldwork_end` are kept.
6. **Weights.** Rolling trackers publish daily from overlapping samples. A
   poll's `weight` is its fieldwork length divided by the total days of
   same-pollster fieldwork overlapping it. Independent polls get 1, and a daily
   4-day rolling tracker gets ≈ ¼ per release.

The CSV columns are `date, fieldwork_start, fieldwork_end, pollster,
sample_size, n_scenarios, weight`, then one column per candidate, then `year`.

## Smoothing

Trends use a weighted local-linear regression with a Gaussian kernel in
**time** (SD 10 days). The window is equally wide throughout the campaign,
whether polls are sparse a year out or daily in the final weeks. Where the
kernel-weighted poll count falls below 2, no value is reported. Lines show
gaps there, and change analyses list the candidate as *not polled* rather
than extrapolating from distant polls.

## What the charts show

For each year (1-year window before the election by default):

- **Faded dots**: individual polls. Dot size scales with poll weight, so
  overlapping tracker releases look lighter.
- **Solid line**: the time-smoothed trend.
- **Ringed dot on election day**: the actual result, direct-labeled with the
  candidate's name and percentage.

Candidates are charted when their actual result, or their weighted polling
average over the final 30 days, exceeds 1 %.

Colors follow political families (far left reds, socialist pinks, ecologist
greens, centre golds, right blues, far right navy/purple). Each candidate gets
a distinct shade. Within every election, all pairs of charted candidates
differ by OKLab ΔE ≥ 15, and every color has ≥ 3:1 contrast on white.
With up to 15 candidates on one chart, color-blind separation can't be
guaranteed. The election-day direct labels and the legend carry identity.

The **polling change** chart splits each candidate's change from `START_DAYS`
to the result into:

- **poll movement**: smoothed support at `START_DAYS` → final polls
  (two days out; French law bars publishing polls on the eve and day of the
  vote);
- **final polling error**: final polls → actual result;
- a marker for the **net change**.

The **final polling error** table compares final smoothed polls with the
result for every charted candidate.

## Tests

`pytest -q` runs in a few seconds, offline:

- Parsing: numeric cells, sample sizes, fieldwork ranges (cross-month,
  cross-year), pollster cleanup, header matching.
- Section classification and row validation for both rounds.
- Lineup collapsing, rolling-tracker weights, time-based smoothing (no
  extrapolation, weights respected).
- Chart-candidate selection, the movement + error decomposition, chart smoke
  tests.
- The full pipeline on a synthetic Wikipedia page (result rows, regional
  tables, hypothetical lineups, second-round tables).
- **Snapshot regressions** on `wiki_cache/`, per election:
  - one row per poll;
  - no row totals above 105 %;
  - every candidate with ≥ 1 % of the vote is polled;
  - final-poll mean absolute error < 1.5 pts;
  - second-round rows sum to about 100 %.

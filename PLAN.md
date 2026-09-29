# PLAN: FiftyPlusOne-style forecasts for French elections

## Context

**Goal.** Build probabilistic forecasts for two elections:
- the **2027 French presidential election** (first round 18 April 2027, runoff 2 May 2027);
- **French legislative elections** (577 constituencies). The election date is a config value, because the next one is due by June 2029 but a snap election is possible at any time.

**Method.** Replicate G. Elliott Morris's FiftyPlusOne 2026 House/Senate model, described in the saved page `fifty_plus_one/How the FiftyPlusOne 2026 election model works.html`. It has five parts:
- component models (polls, fundamentals, expert ratings) each give a forecast with uncertainty for every race;
- a stacked ensemble weights the components by out-of-sample precision;
- an error meta-model, trained on the ensemble's own past out-of-sample errors, sets how much error to add at each level: national, region, state, demographic cluster and seat. Each level's sigma grows with √days to the election;
- 40,000 Monte Carlo simulations turn this into probabilities;
- everything is backtested with no time travel.

**Decisions already made.**
- Build both elections; the presidential first, because it is the next scheduled election and the repo already has its polls.
- The Bayesian pieces are fitted in **Stan via CmdStanPy**.
- The expert race-ratings component is **dropped**, since France has no Cook/Sabato equivalent. The ensemble code accepts N components, so it runs with polls + fundamentals.
- 2027 lineup uncertainty is a **scenario mixture**. Each simulation draws a lineup from probabilities set in a config file, the counterpart of the post's "primaries that haven't happened yet".

**Baseline.** Commit `0a2ca40`. `polls_helpers.py` parses pinned, cached Wikipedia revisions (`wiki_cache/`) into one row per national poll, with pollster, fieldwork dates, sample size and a rolling-tracker overlap weight. The notebook charts kernel-smoothed trends against results. Nothing forecasts yet.

**Outputs.**
- *Presidential*, by candidate and pooled by bloc: P(runs), P(reaches the runoff), P(wins), first-round share intervals, matchup probabilities.
- *Legislative*: per-seat win probabilities, seat distributions, P(absolute majority of 289), P(largest bloc).

## How the post maps to France

| Post (US House/Senate) | 2027 presidential | Legislative |
|---|---|---|
| Race | One per **lineup** (first round) and one per **runoff matchup** | 577 seats |
| Generic ballot (trend signal, now-cast starting point) | Each candidate's trend across all lineups | National bloc vote-intention polls |
| House effects: n/(n+5) cycle-wide, n/(n+20) per race, historical prior halving per cycle | Per firm × candidate; historical prior per firm × bloc | Per firm × bloc |
| Mode, population and sponsor effects | Mode: phone vs online, mostly 2002–2012. Sponsor: rare, e.g. "Ifop for DLF". Population: effectively a no-op, since French polls report voters certain to vote | Same |
| Now-cast for unpolled seats, anchored with 2 pts of slack at 1 poll down to ⅓ pt by 5 polls | Unpolled **lineups** (substitution model) and unpolled **matchups** (transfer model) | Unpolled **seats** (national average + lean + incumbency) |
| Expected national change to Election Day (the midterm swing) | Optional within-bloc consolidation (*vote utile*) | "Confirmation" swing toward the new president's bloc after a presidential; none for snap elections |
| Fundamentals: Stan dynamic linear model, logistic link, random-walk coefficients | Softmax over the candidates in a lineup | Softmax over the blocs standing in each seat |
| Ratings model | Dropped | Dropped |
| Ensemble weighted by out-of-sample precision | Same | Same |
| Error levels: national, chamber, region, state, cluster, seat | Bloc shock + candidate error. Runoff shock correlated with the first round | Bloc, region, département, vote-pattern cluster, seat |
| Special cases: primaries not yet held, strong independents, runoffs | Lineup scenario mixture | Alliance scenarios, 12.5%-of-registered qualification, withdrawals, vote transfers |

## Verified data sources

**Wikipedia.** Pin and cache these pages the same way as the existing ones.
- **Presidential 2002–2022.**
  - Every parsed poll has pollster, sample size and fieldwork dates.
  - Firm names need aliases:
    - Sofres / TNS Sofres / Kantar;
    - Ifop vs the Ifop-Fiducial rolling tracker;
    - OpinionWay vs OpinionWay-Kéa;
    - Ipsos vs Ipsos-Sopra Steria;
    - "Harris Interactive" vs "Harris-Interactive";
    - Louis Harris / LH2;
    - footnote stars.
  - Polls testing several lineups: 9 / 41 / 53 / 65 / 79 (2002 → 2022).
  - Second-round sections hold many hypothetical matchups (2022 Macron–Pécresse alone has 166 rows).
  - 2012 and 2017 have "By first round vote" transfer cross-tabs.
- **2027 page** (revision 1377088060 inspected):
  - first-round lineup tables;
  - tables where Bardella is the RN candidate, filed under a heading ("Scenario polling › Cluster 17 Poll") that `classify_section` drops;
  - "Re-run of the 2022 election" tables, which the current code wrongly treats as real rounds;
  - one table of generic bloc polls;
  - matchup tables, including Bardella matchups under "Declined to be candidates";
  - event rows marking the timeline (31 Mar 2025 Le Pen barred, 7 Jul 2026 appeal ruling, Sep 2026 left primary).
- **Legislative.**
  - National first-round tables at party level: 2012 (26 rows), 2017 (28), 2022 (46, with NUPES sub-parties), 2024 (54, with turnout), and "Next French legislative election" (post-2024).
  - Constituency polls: 2012 about 42 first-round + 36 second-round; 2017 12 + 10; 2022 none; 2024 a handful.
  - Transfer tables: 2017 (3), 2024 (8).
  - Pages for 2002 and 2007 also exist.

**data.gouv.fr "Données des élections agrégées".** Polling-station results for 1999–2026, read lazily from the remote parquet with `pl.scan_parquet`.
- `code_circonscription` (the seat number within the département) is filled only for **2012, 2017 and 2022** (presidential and legislative).
  - It is empty for 2024 legislative, all European elections and everything before 2012.
  - Overseas and abroad seats use Z-codes; ZZ covers French residents abroad.
- `candidats_results` has no seat column. Join it to `general_results` on (`id_election`, `id_brut_miom`).
- The `nuance` (party label) column varies:
  - presidential 2002–2012: a candidate code;
  - presidential 2017 and 2022: NULL, so identify candidates by name;
  - legislative: the codes change every cycle, which needs a hand-made mapping per election.
- For 2024 legislative, use the Ministry's per-constituency CSVs. Alternatively, derive a polling-station-to-seat map by joining `candidats_results` to the 2024 candidate list on (département, panel number, surname).

**Ministry candidate lists 2012–2024.**
- All have an incumbent ("sortant") flag, plus seat, party code, name and birth date.
- The formats differ: 2012 CSV, 2017 xlsx (needs openpyxl), 2022 tab-separated latin-1, 2024 `;`-separated CSV.

**Répertoire national des élus.** It lists current office holders only, so it works for live forecasts but not for backtests. Candidate experience is therefore derived from past winners in the harmonised results, in both cases.

**Not available.**
- Pre-election fundraising: French campaign accounts are published after the vote.
- Demographics by constituency: clusters use voting patterns instead; commune-level INSEE data is a later enhancement.

**Toolchain.**
- cmdstanpy 1.3.0 is pure Python and needs Python ≥ 3.9. Xcode, clang 21 and make are installed.
- Not yet installed: cmdstanpy, CmdStan, openpyxl.
- `cmdstanpy.install_cmdstan()` may first need `sudo xcodebuild -license accept`.
- Under pandas 3.0, avoid cmdstanpy's pandas helpers and read draws with `stan_variable()`.

**Empirical anchors from the cached 2002–2022 polls.**
- **How wrong the smoothed average was, by horizon.** For candidates ≥ 1%, the SD of (smoothed average − result) is 1.40 pts at 2 days out, 2.85 at 30, 3.75 at 60 and 4.42 at 150. That is roughly 1.0 + 0.28·√days.
- **Small candidates miss far less** (mean absolute error 0.37 vs 1.21 pts at 2 days out). Errors are therefore modelled on the logit scale.
- **Late within-bloc consolidation.** From 150 days out:
  - Mélenchon gained +4.4 / +7.0 / +12.9 pts (2012 / 2017 / 2022);
  - the main right-wing candidate lost ground: Chirac −5.9 (2002), Fillon −6.9 (2017), Pécresse −5.5 (2022).

## Known bug to fix first

On the 2027 page, event rows can become phantom polls. These rows have an empty pollster cell and the same text in almost every other cell.

For example, "Faure, Glucksmann, Guedj, Maurel and Royal are the 5 candidates…" parses as a poll dated 2026-09-16 with 19 candidates at 5.0% each. The total (95) passes `is_valid_first_round_row`.

Fix:
- add `is_event_row(pollster_cell, cells)`: the pollster is empty and at least 80% of the non-date cells are identical;
- call it from `extract_poll_records` and send those rows to an events table;
- add a regression test with the real strings.

## Architecture

```
forecast/
  __main__.py        CLI: ingest | average | fit | run | backtest
  config.py          frozen dataclasses from config/model.yaml; each parameter tagged post | law | assumption | estimated
  ingest/            pages.py (pinned pages via pages.yaml), wiki_long.py, registry.py (pollster/candidate aliases),
                     harmonised.py (data.gouv parquet), ministry.py (candidate lists, 2024 per-seat files), experience.py
  polls/             weights.py, trend.py, ewma.py, house_effects.py, population.py, lineups.py, horizon.py
  fundamentals/      features_pres.py, features_legi.py, dlm.py
  ensemble/          stacking.py
  errors/            meta_model.py, shocks.py (antithetic, centred Student-t draws)
  simulate/          scenarios.py, rules.py (outright wins, qualification, withdrawals), presidential.py, runoff.py, legislative.py
  backtest/          harness.py, archive.py, metrics.py, charts.py
  report/            tables.py, charts.py
  stan/              fundamentals_dlm.stan, error_meta.stan, transfers.stan, build.py
config/model.yaml
data/reference/      hand-curated, committed (below)
data/raw/            downloads + manifest.json (url, sha256, fetched_at)            [gitignored]
data/processed/  data/cache/fits/  data/archive/  data/outputs/                    [gitignored]
tests/forecast/
pyproject.toml       deps; extras [stan]=cmdstanpy, [xlsx]=openpyxl; pytest markers slow, network; default -m "not slow and not network"
forecast_2027.ipynb  config-only notebook in the style of french_elections_polls.ipynb
```

`polls_helpers.py` stays the notebook's module and keeps the parsing primitives. Apart from the event-row fix, changes to it are additive, so `french_elections_polls.ipynb`, `polling_data/*.csv` and the snapshot tests keep working. `forecast` reuses:

| Helper(s) | Use in `forecast` |
|---|---|
| `load_wikipedia_html` | Generalised to cache files named `{page_key}_{revision}.html.gz`. |
| `parse_polling_tables`, `_section_path` | As they are. |
| `classify_section` | New pages use explicit rules in `pages.yaml` instead, because the heading walker misfiles 2027 tables. |
| `extract_poll_records` | Its per-lineup rows, taken before `collapse_scenarios`. |
| `collapse_scenarios` | The backtest's "final lineup" choice. |
| `compute_poll_weights` | The rolling-tracker overlap weight. |
| `parse_fieldwork`, `_parse_sample_size`, `clean_pollster`, `_extract_numeric`, `_normalize_name`, `_find_candidate_columns`, `_column_label` | Cell parsing. |
| `ELECTION_DATES`, `SECOND_ROUND_DATES`, `NEXT_ELECTION_DATE`, `FIRST_ROUND_RESULTS`, `SECOND_ROUND_RESULTS`, `FINAL_POLL_DAY`, `CANDIDATE_COLORS` | These seed the reference CSVs; a test asserts the two stay in agreement. |
| `smooth_polls`, `final_polling_errors` | Kept for charts, and as the baseline the new average must beat. |

### Long poll table: `data/processed/polls_long.parquet`

One row per poll row × option, where an option is a candidate, party or bloc.

| Column(s) | Contents |
|---|---|
| `election_id` | e.g. `2027_pres_t1` |
| `poll_id` | Hash of firm, fieldwork, sample size and page |
| `row_id` | Poll plus the row's position within it |
| `lineup_id` | Hash of the sorted options; `A\|B` for runoff matchups |
| `geography` | `national`, or a seat ID such as `01-05` or `ZZ-11` |
| `pollster_raw`, `pollster`, `sponsor`, `partisan` | Raw and canonical firm name; sponsor details |
| `fieldwork_start`, `fieldwork_end`, `midpoint`, `publish_est` | `publish_est` = fieldwork end + the firm's usual publication lag, used for "as of date d" filtering |
| `sample_size`, `n_eff` | `n_eff` = sample size × overlap weight |
| `population`, `mode`, `rolling`, `overlap_weight`, `n_lineups` | Poll metadata |
| `option_id`, `option_kind`, `share`, `share_lt` | `share_lt` flags "<0.5" cells |
| `round`, `kind` | `kind` is lineup / matchup / generic / transfer / constituency |
| `source_page`, `source_revision`, `source_table`, `source_row`, `flags` | Provenance |

Companion tables: `lineups.parquet`, `events.parquet`, `transfers.parquet`.

### Hand-curated reference data (`data/reference/`, each file carries sources)

| File | Contents |
|---|---|
| `pages.yaml` | Page titles, pinned revisions, per-section include/exclude rules (round, kind) |
| `pollster_aliases.csv`, `pollster_meta.csv` | Raw name → firm; mode by year, rolling products, sponsor rules, exclusions, publication lag |
| `candidates_pres.csv` | Per candidate: stable ID, column-match pattern, party and bloc, incumbency credit, experience 0–3, prior runs, declared/withdrawn dates, results, colour. 2002–2027 first; 1965–1995 as an extension |
| `national_results.csv` | Bloc and candidate national shares: presidential, legislative, European |
| `party_bloc_map.csv` | (election, party label / Wikipedia column / list) → family and model bloc |
| `constituencies.csv` | 577 seats: département, Z-code alias, region, overseas/abroad flags, cluster |
| `scenarios_pres_2027.yaml`, `scenarios_legi.yaml` | Lineup mixture; alliance scenarios, withdrawal policy, dates |
| `priors.yaml` | Fixed weakly informative priors shared across backtest years (the post's disclosed exception) |

`scenarios_pres_2027.yaml` sketch. All probabilities are placeholders: they are judgment calls to set and publish, like the post's Montana scenario.

```yaml
election: {id: 2027_pres, t1: 2027-04-18, t2: 2027-05-02}
slots:                          # one draw per simulation; null = nobody from this slot runs
  rn:     {LE_PEN: 0.5, BARDELLA: 0.5}
  centre: {PHILIPPE: 0.5, ATTAL: 0.5}
  left:   {GLUCKSMANN: 0.5, FAURE: 0.5}
fixed: [MELENCHON, RETAILLEAU, ROUSSEL, ZEMMOUR, DUPONT_AIGNAN, ARTHAUD]
optional: {DE_VILLEPIN: 0.5}    # independent yes/no draws
constraints: []                 # e.g. if centre = X, exclude Y
```

## Presidential model

1. **Ingest** (`ingest/wiki_long.py`).
   - **Selection.** Keep every lineup row and every matchup table. `pages.yaml` rules choose the tables, rather than headings alone.
   - **Exclude:**
     - results rows (kept as validation data);
     - event rows (sent to `events.parquet`);
     - empty rows;
     - hypothetical sections ("Re-run of the 2022 election");
     - implausible rows: a first-round sum outside 50–105, any value above 60, or a runoff sum outside 95–105;
     - fieldwork ending on or after election day;
     - excluded firms.
   - **Partisan-sponsored polls** keep half weight and are shifted 2 pts away from the sponsor (from the post).
   - **Untested candidates** ("–") are absent, not zero.
   - **Multi-way over restricted ballots.** Within one poll, drop a lineup if it is a subset of another that differs only by minor candidates (< 2%). This is the French version of the post's rule to use the multi-way ballot.
   - **Declined or withdrawn candidates.** Rows that include someone who has declined or withdrawn by date d still feed house effects and the substitution model. They only form a race if the scenario file gives that candidate a probability.

2. **Weights.** 90-day window. A poll's weight is the product of:
   - **recency:** (1−δ(h))^(d − midpoint), where δ = 0.03 from 45+ days out, ramping linearly to 0.14 at ≤ 14 days. The endpoints are from the post; the ramp is assumed and tuned;
   - **overlap:** `compute_poll_weights`;
   - **sample size:** √(min(n_eff, 4000)/1000), an assumption;
   - **sponsor:** × 0.5 for partisan-sponsored polls.

   Poll counts in the shrinkage formulas use Σ overlap weights, and a row counts 1/n_lineups in cycle-wide statistics.

3. **Trend signal** (the generic-ballot equivalent).
   - Model y_{row,c} = θ_c(t) + λ_{lineup,c} + ε on the logit scale. θ is the candidate's trend over time; λ is a lineup effect.
   - Fit it with three alternating passes. θ is an exponentially weighted average (decay 0.07/day, assumed). λ is shrunk by n/(n+5) and centred.
   - Older polls are shifted by θ_c(d) − θ_c(t) and renormalised.
   - Candidates with fewer than 5 polls use their bloc total's trend.
   - Runoff matchups get the same treatment, with matchup fixed effects.

4. **Pass 1.** Weighted average per lineup, day and candidate over [d−90, d], renormalised to 100.

5. **Mode effects** (applied before house effects). The mean leave-one-out residual by mode, shrunk by n/(n+5). This matters for 2002–2012 and is ≈ 0 for 2027.

6. **Pass 2 (house effects).** Residuals are taken against pass-1 averages that leave the poll itself out; leaving it out is our addition, not the post's.
   - **Cycle-wide lean:** L[p,c] × n/(n+5).
   - **Historical prior:** Prior[p, bloc(c)] × (candidate share ÷ bloc share), blended in with weight α = cap × W/(W+5), where W = Σ 0.5^(k−1) × n_{p, cycle−k}. The cap is 0.10 (presidential, like the post's Senate). The α formula is assumed.
   - **Lineup-specific deviation:** added back × n_s/(n_s+20).
   - **Centring:** each poll's effects sum to 0, and the √n-weighted mean across firms is 0.
   - Subtract the effects and recompute the average.

7. **Pass 3 (population).** Implemented, but effectively a no-op in France.

8. **Unpolled lineups (now-cast).**
   - A nested-logit substitution model with blocs as nests models where a missing candidate's voters go. Its nest parameters are identified from pairs of lineups tested in the same poll (same firm, fieldwork and sample), shrunk toward the 2017/2022 values.
   - Where a candidate pair has ≥ 3 such within-poll pairs, blend in the observed diversion ratio with weight n/(n+3).
   - **Re-anchor polled lineups:** final = average + clip(now-cast − average, ±s(n)), where s(n) = max(⅓, 2·n^(−ln 6/ln 5)). That gives 2.00, 0.92, 0.59, 0.43 and 0.33 pts for 1–5 polls.
   - **Imputation error:** σ_imp from leave-one-lineup-out cross-validation, drawn from a Student-t with 4 df.

9. **Horizon model.**
   - **Drift** is zero by default. An optional within-bloc consolidation term (bloc leader × gap × horizon) is fitted with heavy shrinkage and kept only if the backtest improves.
   - **Movement:** sd = a + b·√h on the logit scale, with a prior from the empirical anchor.
   - **Polls component:** mean = the anchored now-cast; sd = √(σ_avg² + σ_move(h)² [+ σ_imp² if the lineup is unpolled]).

10. **Fundamentals** (`fundamentals_dlm.stan`: softmax over the lineup, Dirichlet likelihood, random-walk coefficients).

    | Post predictor | Presidential version |
    |---|---|
    | Seat partisan lean | Offset: the log of the bloc's share in the last two presidential first rounds (80/20), split among same-bloc candidates in the lineup. Optional extra predictor: the bloc's share in the latest European or legislative election, kept only if it helps out of sample |
    | Incumbency (first-termers get half) | 1 = sitting president running; 0.5 = designated heir of the presidential camp; optional interaction with presidential approval |
    | Experience | 0–3: 1 = local office, MEP, deputy or senator; 2 = minister, party leader, region president or big-city mayor; 3 = current or former President or PM |
    | Fundraising | Dropped (no pre-election data). Optional late-cycle substitute: the candidate's share of validated sponsorship signatures (*parrainages*) |
    | Personal vote | Recency-weighted (own past first-round share − own bloc's share), on the logit scale; 0 plus a first-timer flag for new candidates |

    - **Training:** 2002–2022 (5 elections), extended to 1965–2022 (11) once the earlier results and candidates are curated.
    - **Prediction:** the posterior predictive for each lineup (the random walk stepped forward once, plus Dirichlet noise).

11. **Ensemble** (`ensemble/stacking.py`, N components).
    - **Weights:** ∝ 1/σ_k(h, n)², where σ_k = √(a_k + b_k·h^γ + c_k/(1+n)) is fitted to each component's out-of-sample errors in the backtest archive.
    - **Optional stacking:** per-horizon precision multipliers chosen to maximise out-of-sample log score.
    - **Adjustments from the post:**
      - no incumbent running (2007, 2017, 2027): fundamentals precision × ρ_open, prior 0.8;
      - a heavily polled race (n ≥ 10) where polls and fundamentals differ by more than 2 pooled sd: polls precision × 1.25.
    - **Combination:** on the logit scale, then renormalised.
    - **Handed to the error model:** σ̄ = Σ w_k·σ_k and disagreement |μ_p − μ_f| / √(σ_p² + σ_f²).
    - **Archived:** the weights, for a "weights over time" chart.

12. **Runoff.**
    - **Polled matchups:** three-pass averages on logit(A's share), with house effects per firm × bloc and the n/(n+20) add-back.
    - **Unpolled matchups** come from the transfer model (`transfers.stan`):
      - A's share = Σ_k p_k·T_{k→A} / Σ_k p_k·(T_{k→A} + T_{k→B}), where p_k are first-round shares and T are transfer rates;
      - fitted on 2002–2022 polling-station first-round → runoff flows, plus the 2012/2017 cross-tabs;
      - with a per-candidate "runoff appeal" residual taken from polled matchups.
    - **Anchoring:** the same slack rule as step 8.
    - **Per simulation:** runoff logit = ensemble matchup mean + [transfer(simulated first round) − transfer(expected first round)] + shocks. This keeps the two rounds coherent.

13. **Errors** (logit scale, `error_meta.stan`).
    - **First round:** a centred bloc shock, plus a candidate error that multiplies σ̄ by m(n, disagreement).
    - **Runoff:** a matchup shock correlated (ρ, prior 0.5) with the RN bloc's first-round shock. This mirrors the post's national shock shared across chambers.
    - Each level's σ_ℓ(h) = a_ℓ + b_ℓ·√h.
    - Student-t with ν = the number of training cycles, floor 3.

14. **Simulation** (40,000).
    - Draw a lineup per simulation, and vectorise by lineup.
    - Draw shocks in antithetic pairs, so every level has exactly zero mean and never moves the central estimate.
    - Softmax to shares.
    - A candidate above 50% wins outright; otherwise the top two go to the runoff, simulated as in step 12.

15. **Outputs.**
    - **Per candidate:** P(runs), P(reaches runoff) and P(wins), unconditionally and given that they run; share quantiles (5/25/50/75/95).
    - **Matchups:** their probabilities and runoff-share distributions.
    - **Pooled** by slot and by bloc, e.g. P(RN wins), P(no left candidate in the runoff).
    - Charts apply the post's display gate (≥ 3 polls from ≥ 2 firms). The model input has no gate.

## Legislative model

1. **Units.**
   - Model blocs:
     - EXG;
     - GAU;
     - CEN;
     - DRO;
     - RN (with the UXD/UDR allies);
     - EXD (REC and others);
     - DIV.
   - Alliance scenarios (`scenarios_legi.yaml`) define the competing units per seat. Examples: GAU split into LFI and PS-LE-PCF-PP, or CEN + DRO fielding one joint candidate in listed seats.
   - Each simulation draws one alliance scenario.

2. **Seat lean.** 0.8 × [logit(seat share) − logit(national share)] in the latest presidential first round, plus 0.2 × the same for the one before, per unit.
   - Presidential candidates are mapped to blocs.
   - 2012, 2017 and 2022 come from the harmonised data. After April 2027, the 2027 presidential becomes the recent one.
   - The 2012 legislative uses the 2012 presidential only, since 2007 results are on the old boundaries.
   - The blend is a config value: the post says 80/20 in its method text but 85/15 in its correction note.

3. **National average (the generic ballot).**
   - The three-pass average over national party/bloc polls, from the 2012–2024 pages and "Next French legislative election".
   - Each alliance configuration a pollster tests is a lineup.
   - Prior cap 0.20, like the post's House.

4. **Expected drift.**
   - A "confirmation" swing toward the new president's bloc when the legislative follows a presidential (2002–2022).
   - None for snap elections (1997, 2024).
   - Estimated from national polls vs results with heavy shrinkage, and stored as a config value.

5. **Now-cast (all 577 seats).**
   - **Starting point:** softmax(logit(national average) + lean).
   - **Calibration target.** Constituency polls are too sparse to fit the post's regression (2012 about 42, 2017 12, 2022 none), so calibrate to results instead. The target is logit(result) − [logit(national result) − logit(final national average)].
   - **Fit:** ridge regression of that target on the starting point, incumbency and personal vote, using prior cycles only.
   - **Polled seats:** constituency polls are trendline-adjusted by the national movement since fieldwork, then re-anchored with s(n).
   - **Imputation error:** σ_imp from the calibration residuals.

6. **Fundamentals** (the same Stan model: softmax over the units in each seat, with the most recent national result as the national offset, not polls).
   - **Lean.**
   - **Lagged seat result:** the unit's share in the seat's previous legislative first round, relative to its national share.
   - **Incumbency** from the sortant flag, with first-termers at 0.5. Before candidate lists are published, assume the last winner's bloc re-runs with probability 0.8.
   - **Experience (0/1):** has the candidate won an election before (deputy; head of a winning municipal list in a commune ≥ 1,000 inhabitants; departmental or cantonal councillor; head of a regional list)? Derived from harmonised results matched on (surname, first name, département), plus birth date where available.
   - **Personal vote:** the incumbent's recency-weighted over-performance of lean.
   - **Fundraising:** dropped.
   - **Training cycles:** 2012, 2017, 2022, 2024.

7. **Ensemble.** As for the presidential, plus the post's rules for open seats and for not over-shrinking competitive seats toward safe ones.

8. **Rounds** (`simulate/rules.py`).
   - **Turnout:** logit(turnout_j) = logit(national turnout) + the seat's turnout lean (last two legislative first rounds, 80/20) + a national shock + seat noise. National turnout comes from pollsters' turnout columns or a prior by election type. Expressed votes = turnout × (1 − blank/null) × registered voters.
   - **Outright first-round win:** > 50% of expressed votes and ≥ 25% of registered voters.
   - **Runoff qualification:** ≥ 12.5% of registered voters; if fewer than two qualify, the top two go through.
   - **Withdrawals** (*désistements*):
     - a policy table gives P(withdraw | own bloc, blocs of the top two) for units placed third or lower (e.g. left and centre about 0.9 against an RN leader, RN 0);
     - each simulation scales it with a shared "front républicain" shock (logit sd about 0.8);
     - defaults are estimated from how three-way runoffs turned into duels in 2012–2024, then adjusted by judgment and disclosed.
   - **Transfers:**
     - conditional-logit affinities from source bloc to destination, with abstention as the reference;
     - a random walk across elections;
     - fitted on seat-level first-round → second-round flows 2012–2024, plus the 2017/2024 survey tables;
     - one shared perturbation per simulation.

9. **Errors** (the post's levels).
   - National: a bloc vector, centred.
   - Region: 13 metropolitan regions + overseas + abroad.
   - Département: the counterpart of the post's state level.
   - Cluster: k-means (k ≈ 10) on the last two presidential first-round bloc shares, turnout and an urban proxy.
   - Seat: m(n, d) × σ̄.

   The national-level σ can also draw on national-only errors from 1997–2024.

10. **Simulation.**
    - 40,000 sims × 577 seats × ~8 units in float32 is about 740 MB per array, so run 2,000-simulation chunks (about 37 MB each).
    - Group by alliance scenario, and mask absent units with −∞ logits.
    - Accumulate seat winners (int8), seat totals per bloc (int16) and fixed-bin share histograms.
    - Outputs:
      - P(bloc wins seat);
      - seat distributions;
      - P(≥ 289) and P(largest bloc) per bloc;
      - expected numbers of three-way runoffs and first-round wins.

## Stan models

| File | Model | Fit |
|---|---|---|
| `fundamentals_dlm.stan` | Shares ~ Dirichlet(φ·softmax(offset + X·β_t)) per race. β_t = β_{t−1} + τ⊙z. Missing predictors = 0, plus indicator columns. Generates `beta_next` for prediction | NUTS; Laplace behaves badly near τ → 0. Presidential: seconds. Legislative (~18k rows, 2.3k races): minutes, once per test cycle |
| `error_meta.stan` | err ~ t_ν(Σ level shocks, σ̄·m0·(1 + m1·scarcity + m2·disagreement)). shock ~ t_ν(0, a_ℓ + b_ℓ·√h). ν fixed | NUTS on data thinned to limit week-to-week autocorrelation (legislative: every 3rd week, top 3 units per seat). `pathfinder` during development. Never `optimize`, whose mode is degenerate for hierarchical variances. Check against a method-of-moments baseline |
| `transfers.stan` | Second-round shares of registered voters = Σ_k x_k·softmax(A_t[k, units present], abstain = 0). A_t follows a random walk across elections. Survey cross-tabs add a multinomial term | NUTS |

The design pass drafted Stan code for all three models. That code goes into M4 (fundamentals, transfers), M5 (the presidential meta-model) and M8 (the legislative fits). Models compile once into `forecast/stan/build/`. Fits are cached by a hash of (Stan code, data, config). The small pieces use numpy/scipy: house effects, the substitution model, stacking multipliers and the now-cast ridge.

## Backtest (no time travel)

- **Snapshots:** weekly from 155 days out to the eve of the first round, plus one between the rounds.
- **Test cycles:**
  - presidential 2007, 2012, 2017, 2022; 2002 runs on fixed priors only;
  - legislative 2017, 2022, 2024; 2012 is training only.
- **What each snapshot can see.** At cycle c and date d:
  - polls with `publish_est` ≤ d;
  - candidate lists and incumbents known by d;
  - component models trained on cycles before c, at the same horizon where parameters depend on it;
  - ensemble precision curves and the meta-model fitted only on archive rows from cycles before c.

  One chronological pass is enough, because those archive rows were themselves produced out of sample.
- **Lineups.** Use the actual ballot, restricted to candidates who had declared by d (`declared_date`). A sensitivity run uses the most-polled lineup of the previous 30 days instead. This is the one disclosed leak.
- **Archive** (`data/archive/`, partitioned by election type and cycle). Columns:
  - run ID and git sha;
  - election, cycle, round, snapshot date, days out;
  - unit, lineup/scenario, component;
  - mean and sd (logit), share quantiles, P(runoff), P(win);
  - poll counts, disagreement, component weight;
  - actual result, error, won;
  - region, département, cluster, bloc.
- **Metrics** (mirroring the post):
  - MAE and RMSE by horizon and component;
  - calibration buckets (0–5, 5–15, 15–30, 30–50, 50–70, 70–85, 85–95, 95–100%) with Wilson intervals;
  - Brier score and log loss for winning, reaching the runoff and seat winners;
  - coverage of 50/80/95% intervals;
  - component weights over time;
  - for legislative: seat-total error and CRPS (a score for the whole seat-count distribution).
- **Runtime:** 10k simulations per snapshot, run in parallel across snapshots. Presidential takes minutes; legislative a few hours, mostly the meta-model.

## Parameters

| Parameter | Value | Source |
|---|---|---|
| Poll window | 90 days | post |
| Daily recency decay | 0.03 → 0.14 in the final 14 days (linear ramp from 45 days) | post endpoints; ramp assumed |
| Sample-size weight | √(min(n_eff, 4000)/1000) | assumption |
| Partisan-sponsored polls | weight × 0.5, 2-pt shift | post |
| House-effect shrinkage | n/(n+5) cycle-wide; n/(n+20) per race | post |
| Historical prior | each older cycle × 0.5; weight capped at 0.10 (presidential) / 0.20 (legislative) | post (α formula assumed) |
| Anchor slack | 2 pts at 1 poll → ⅓ pt by 5 polls | post |
| Lean blend | 80/20, configurable | post |
| Error σ(h) | a + b·√h per level | post |
| Error likelihood | Student-t, ν = number of training cycles (≥ 3) | post |
| Simulations | 40,000 (10,000 in backtests); antithetic and centred | post |
| Backtest snapshots | weekly, 155 days → eve | post |
| "Front républicain" withdrawal shock | logit sd ≈ 0.8 | assumption |

## Deliberate departures from the post

1. No ratings component; France has no equivalent.
2. Hypothetical lineups are modelled as a scenario mixture instead of excluded. Backtests condition on the declared ballot.
3. Rolling trackers are down-weighted for overlap, and house-effect residuals leave the poll itself out.
4. The population adjustment is effectively a no-op. Mode effects are confounded with the 2010–2012 phone-to-online shift.
5. Fundraising is dropped, with an optional *parrainages* substitute for the presidential race.
6. "Race" means a lineup or matchup, and the "generic ballot" is a cross-lineup trend. Everything is multi-option on the logit scale.
7. The legislative now-cast is calibrated to results, not constituency polls. The two-round rules, turnout, withdrawals and transfers need judgment parameters.
8. Error levels are remapped. The post's chamber level has no counterpart, except in an optional joint presidential–legislative run.
9. There are few training cycles (5 presidential, 4 legislative). That means fat tails and more reliance on `priors.yaml`.
10. Parameters the post leaves unspecified (the decay ramp, the prior-weight formula, the sample-size weight) are tuned, which adds researcher degrees of freedom. Each carries a provenance tag in `config/model.yaml`.

## Milestones (each ends runnable and tested)

- **M0 — Setup.**
  - `pyproject.toml`: dependencies, extras, pytest markers, and a default of `-m "not slow and not network"`.
  - `.gitignore` entries for `data/raw`, `data/processed`, `data/cache`, `data/archive`, `data/outputs` and `fifty_plus_one/` (a third-party page).
  - Package skeleton and `config/model.yaml`.
  - **Fix the event-row bug**, with a regression test.
- **M1 — Presidential ingest.**
  - `pages.yaml` with pinned revisions; the 2027 page is cached like the others.
  - The long table for 2002–2027: lineups, all matchups, transfer tables, events.
  - Pollster aliases and metadata; `candidates_pres.csv` for 2002–2027.
  - Collapsing the long table with `collapse_scenarios` must reproduce `polling_data/*.csv` row for row.
- **M2 — Polling averages.** The three passes for lineups and matchups, plus charts. Final averages must beat the current smoother's 1.5-pt final-poll error.
- **M3 — First 2027 forecast, polls only.** Lineup now-cast, horizon model with the empirical σ(h), scenario sampler, simulation v0. This gets useful output early.
- **M4 — Stan.** Install cmdstanpy and CmdStan; build the presidential fundamentals model and the runoff transfer model.
- **M5 — Presidential backtest.** Archive, learned ensemble, error meta-model and calibration report, ending with the full 2027 production forecast.
- **M6 — Legislative data.**
  - Seat aggregation: harmonised data for 2012–2022; for 2024, the Ministry per-seat files or the polling-station-to-seat join, checked to give 577 seats with matching totals.
  - `constituencies.csv`, the party → bloc map, candidate lists 2012–2024, experience, lean, national and constituency polls.
- **M7 — Legislative polls model.** Generic ballot, seat now-cast, turnout, first-round rules, withdrawals, transfers, chunked simulation, P(289).
- **M8 — Legislative fits and backtest.** Fundamentals, transfers and the meta-model; backtest 2017, 2022 and 2024.
- **M9 — Production.**
  - CLI, daily archive, `forecast_2027.ipynb`, README update.
  - Optional: a Streamlit dashboard (streamlit is already in the venv).
  - Optional: chain the presidential simulations into the legislative.

**Hand-curation work** (not code):
- `pages.yaml` and pinned revisions;
- pollster aliases and metadata;
- `candidates_pres.csv`, plus the 1965–1995 extension;
- national results before 1999;
- party → bloc maps per election;
- a review of `constituencies.csv`;
- both scenario files: probabilities, pacts, withdrawal policy, dates;
- `priors.yaml`;
- candidate declaration dates;
- mapping the survey transfer tables to blocs;
- optional approval and *parrainages* series.

## Verification

- **`pytest -q`** (offline, seconds): one test per formula.
  - Weights and shrinkage:
    - decay endpoints and ramp;
    - n/(n+5) = ½ at n = 5;
    - n/(n+20) < 0.1 at n = 2;
    - prior weight ≤ cap, and 0 without history;
    - both centring constraints;
    - leave-one-out residuals;
    - the sponsor shift preserves the row total.
  - Anchoring: the s(n) values, clip anchoring, renormalisation.
  - Ingest:
    - `is_event_row` on the real 2027 strings;
    - "Re-run" excluded and Bardella tables included;
    - stable IDs;
    - the subset-lineup rule.
  - Simulation:
    - scenario sampler frequencies within 3σ;
    - antithetic shocks with exactly zero mean;
    - bloc shocks summing to zero;
    - presidential outright win and top two.
  - Legislative rules:
    - the > 50% and ≥ 25% outright win;
    - 12.5% qualification with the top-two fallback;
    - ties;
    - withdrawals;
    - transfer rows summing to 1;
    - chunked and unchunked simulations match for the same seed.
- **Integration test** (fast). A synthetic election (6 candidates, 3 pollsters, 60 polls, 2 lineups) runs through the whole pipeline. Stan components are stubbed behind the same interfaces, injectable like `process_year_data`'s fetcher.
- **`pytest -m slow`:** Stan compile and parameter recovery for all three models, skipped if CmdStan is missing.
- **`pytest -m network`:** live Wikipedia and data.gouv smoke tests.
- The existing `test_polls_helpers.py` stays green, including the snapshot regressions.
- **`python -m forecast backtest --kind pres`:**
  - produces calibration, error-by-horizon and weights-over-time charts;
  - election-eve error for major candidates is about 1–1.5 pts;
  - weights shift toward polls as election day nears.
- **`python -m forecast run --election 2027_pres --as-of <date>`:**
  - within each lineup, P(runoff) sums to 2 and P(win) to 1;
  - outputs are pooled by candidate and bloc.
- **Legislative backtests:**
  - seat totals vs actual (2024: NFP 180, Ensemble 159, RN + allies 142);
  - simulated seats always sum to 577.
- `french_elections_polls.ipynb` and `forecast_2027.ipynb` run top to bottom.

## Implementation status (29 Sep 2026)

M0–M9 are built. The presidential and legislative forecasts run end to end, and both backtests have been run (`data/archive/`). The fast test suite passes: 236 tests in about 13 s, plus 2 slow Stan tests. Changes against the plan above:

- **Trend signal.** A local-linear kernel smoother (reusing `smooth_polls`) with no extrapolation past a candidate's last poll replaces the causal EWMA. The EWMA lagged late movers and cost about 0.5 pts of final-poll error.
- **Consolidation drift.** Tested and **disabled** (`use_drift: false`): out of sample it raised MAE from 2.10 to 2.70.
- **Stacking.** Stacking multipliers are now chosen out of sample. The fundamentals get κ = 0.25–0.5 because they miss by about 5.5 pts; the polls weight is 0.87–0.96.
- **Error meta-model.** Simulated draws now use Student-t *scale* draws. Rescaling them to unit variance had made intervals too narrow; 50/90% coverage went from 36/74% to 44/85%. Shared shocks are not extrapolated beyond the longest horizon in the training archive.
- **Transfer models.** The runoff transfer models are scipy fits, not `transfers.stan`. The presidential one uses bloc positions plus candidate appeal; the legislative one is a full conditional-logit bloc × bloc matrix.
- **Legislative personal vote.** Folded into the lagged seat result.
- **Scenario decisions.** Scenarios support poll-dependent decisions (`decisions:` in `scenarios_pres_2027.yaml`). The January 2027 centrist rule is simulated from each simulation's projected January polls.
- **Not yet done.** Candidate declaration dates are not curated, so backtests use the actual ballot. The 1965–1988 fundamentals extension is not done. Transfer cross-tab tables are not parsed.
- **Legislative limits.** Only three training cycles, with ν = 3 tails. The last national legislative poll on the pinned page is from October 2025. As a result, legislative seat ranges are very wide, and the 2017 and 2022 backtests missed RN badly (the transfer patterns changed between cycles).

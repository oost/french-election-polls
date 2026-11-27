"""Candidate/pollster registries and the long-table ingest (synthetic pages
plus a regression check against the committed polling_data CSVs)."""
from __future__ import annotations

import polars as pl
import pytest

from forecast.ingest.pages import PageSpec, TableRule, load_page_specs, select_tables
from forecast.ingest.registry import CandidateRegistry, PollsterRegistry
from forecast.ingest.wiki_long import (
    IngestReport, build_page, collapse_to_ballot, drop_restricted_lineups, parse_share,
)


@pytest.fixture(scope="module")
def reg() -> CandidateRegistry:
    return CandidateRegistry.load()


@pytest.fixture(scope="module")
def firms() -> PollsterRegistry:
    return PollsterRegistry.load()


class TestCandidateMatching:
    @pytest.mark.parametrize("label, year, expected", [
        ("Philippe Poutou", 2027, "POUTOU_PHILIPPE"),
        ("Philippe HOR", 2027, "PHILIPPE_EDOUARD"),
        ("Édouard Philippe (HOR)", 2027, "PHILIPPE_EDOUARD"),
        ("Le Pen RN", 2027, "LE_PEN_MARINE"),
        ("Le Pen FN", 2002, "LE_PEN_JEAN_MARIE"),
        ("de Villiers (R!)", 2027, "VILLIERS_PHILIPPE_DE"),
        ("de Villepin (LFH)", 2027, "VILLEPIN_DOMINIQUE_DE"),
        ("Jean Lassalle(RES)", 2027, "LASSALLE_JEAN"),
        ("Dupont- Aignan DLF", 2022, "DUPONT_AIGNAN_NICOLAS"),
        ("NKM LR", 2017, "KOSCIUSKO_MORIZET_NATHALIE"),
        ("Ruffin PD\xa0[fr]", 2027, "RUFFIN_FRANCOIS"),
        ("Saint- Josse CPNT", 2002, "SAINT_JOSSE_JEAN"),
    ])
    def test_headers(self, reg, label, year, expected):
        assert reg.match_header(year, label) == expected

    def test_unknown_header(self, reg):
        assert reg.match_header(2012, "(nominee) NPA") is None
        assert reg.match_header(2027, "Lead") is None

    def test_every_bloc_known(self, reg):
        assert set(reg.df["bloc"].unique().to_list()) <= {"EXG", "GAU", "CEN", "DRO", "EXD", "DIV"}

    def test_results_agree_with_polls_helpers(self, reg):
        import polls_helpers as ph
        for year, results in ph.FIRST_ROUND_RESULTS.items():
            ballot = {c.surname: c.result_t1 for c in reg.ballot(year)}
            assert ballot == pytest.approx(results)
        for year, results in ph.SECOND_ROUND_RESULTS.items():
            finalists = {c.surname: c.result_t2 for c in reg.ballot(year) if c.result_t2 is not None}
            assert finalists == pytest.approx(results)


class TestPollsters:
    @pytest.mark.parametrize("raw, firm", [
        ("Ifop-Fiducial*", "Ifop"), ("Ifop/Hexagone", "Ifop"), ("TNS Sofres", "Kantar"), ("Verian", "Kantar"),
        ("Harris-Interactive", "Harris Interactive"), ("OpinionWay-Kéa", "OpinionWay"),
        ("Louis Harris", "LH2"), ("Ipsos-Sopra Steria", "Ipsos"), ("Le Terrain***", "Le Terrain"),
        ("BVA Archived 23 February 2018 at the Wayback Machine", "BVA"),
    ])
    def test_canonical(self, firms, raw, firm):
        assert firms.canonical(raw).firm == firm

    def test_partisan_sponsor(self, firms):
        info = firms.canonical("Ifop for DLF")
        assert (info.firm, info.sponsor, info.sponsor_bloc) == ("Ifop", "DLF", "DRO")

    def test_mode_by_year(self, firms):
        assert firms.meta("Ifop", 2002).mode == "phone"
        assert firms.meta("Ifop", 2022).mode == "online"


def test_parse_share():
    assert parse_share("<1%") == (0.5, True)
    assert parse_share("<0.5%") == (0.25, True)
    assert parse_share("17%") == (17.0, False)
    assert parse_share("–") == (None, False)


_PAGE = """
<html><body>
<h2>First round</h2>
<h3>Since September 2026</h3>
<table>
<tr><th>Polling firm</th><th>Fieldwork date</th><th>Sample size</th>
<th>Nathalie Arthaud</th><th>Jean-Luc Mélenchon</th><th>Raphaël Glucksmann</th><th>Édouard Philippe</th>
<th>Gabriel Attal</th><th>Bruno Retailleau</th><th>Marine Le Pen</th></tr>
<tr><td>Harris Interactive</td><td>22–24 Sep 2026</td><td>1,974</td><td>1%</td><td>17%</td><td>10%</td><td>13%</td><td>–</td><td>7%</td><td>35%</td></tr>
<tr><td>Harris Interactive</td><td>22–24 Sep 2026</td><td>1,974</td><td>&lt;1%</td><td>18%</td><td>10%</td><td>–</td><td>12%</td><td>8%</td><td>36%</td></tr>
<tr><td></td><td>16 Sep 2026</td><td colspan="8">Faure, Glucksmann, Guedj, Maurel and Royal are the 5 candidates</td></tr>
<tr><td>Ifop for DLF</td><td>1–2 Sep 2026</td><td>1,000</td><td>1%</td><td>16%</td><td>11%</td><td>15%</td><td>–</td><td>8%</td><td>34%</td></tr>
<tr><td>2022 election</td><td>10 Apr 2022</td><td>–</td><td>0.56%</td><td>21.95%</td><td>–</td><td>–</td><td>–</td><td>–</td><td>23.15%</td></tr>
</table>
<h2>Second round</h2>
<h3>Philippe vs. Le Pen</h3>
<table>
<tr><th>Polling firm</th><th>Fieldwork date</th><th>Sample size</th><th>Philippe HOR</th><th>Le Pen RN</th></tr>
<tr><td>Ifop</td><td>7–8 Jul 2026</td><td>984</td><td>46%</td><td>54%</td></tr>
</table>
<h2>Scenario polling</h2>
<h3>Cluster 17 Poll</h3>
<h4>January – June 2026</h4>
<table>
<tr><th>Polling firm</th><th>Fieldwork date</th><th>Sample size</th><th>Mélenchon (LFI)</th><th>Philippe (HOR)</th><th>Retailleau (LR)</th><th>Bardella (RN)</th></tr>
<tr><td>Ifop</td><td>22–24 Jun 2026</td><td>1415</td><td>15%</td><td>21%</td><td>11%</td><td>37%</td></tr>
</table>
<h3>Re-run of the 2022 election</h3>
<h4>First round</h4>
<table>
<tr><th>Polling firm</th><th>Fieldwork date</th><th>Sample size</th><th>Emmanuel Macron</th><th>Marine Le Pen</th><th>Jean-Luc Mélenchon</th></tr>
<tr><td>Odoxa</td><td>5–6 Apr 2026</td><td>1005</td><td>23%</td><td>32%</td><td>19%</td></tr>
</table>
</body></html>
"""


class TestBuildPage:
    @pytest.fixture(scope="class")
    def built(self, reg, firms):
        spec = load_page_specs()["pres2027"]
        report = IngestReport()
        long, events = build_page(spec, reg, firms, html=_PAGE, report=report)
        return long, events, report

    def test_rule_selection(self):
        spec = load_page_specs()["pres2027"]
        kept = {" › ".join(s.section_path): s.rule.kind for s in select_tables(spec, _PAGE)}
        assert "Scenario polling › Cluster 17 Poll › January – June 2026" in kept   # Bardella lineups kept
        assert not any("Re-run" in k for k in kept)                                 # hypothetical re-run excluded
        assert kept["Second round › Philippe vs. Le Pen"] == "matchup"

    def test_event_and_result_rows_skipped(self, built):
        long, events, report = built
        assert events.height == 1 and "5 candidates" in events["text"][0]
        assert report.rows_by_reason["result row"] == 1
        assert long.filter(pl.col("pollster_raw") == "2022 election").is_empty()

    def test_lineups_kept_separately(self, built):
        long, _, _ = built
        harris = long.filter((pl.col("pollster") == "Harris Interactive") & (pl.col("kind") == "lineup"))
        assert harris["row_id"].n_unique() == 2
        assert harris["lineup_id"].n_unique() == 2
        assert harris["n_lineups"].unique().to_list() == [2]

    def test_less_than_cells(self, built):
        long, _, _ = built
        row = long.filter((pl.col("option_id") == "ARTHAUD_NATHALIE") & pl.col("share_lt"))
        assert row["share"].to_list() == [0.5]

    def test_partisan_sponsor_flagged(self, built):
        long, _, _ = built
        dlf = long.filter(pl.col("pollster_raw") == "Ifop for DLF")
        assert dlf["partisan"].all() and dlf["sponsor_bloc"].unique().to_list() == ["DRO"]

    def test_bardella_and_matchup(self, built):
        long, _, _ = built
        assert "BARDELLA_JORDAN" in long["option_id"].to_list()
        m = long.filter(pl.col("kind") == "matchup")
        assert m["lineup_id"].unique().to_list() == ["LE_PEN_MARINE|PHILIPPE_EDOUARD"]
        assert m["election_id"].unique().to_list() == ["2027_pres_t2"]


def test_drop_restricted_lineups():
    base = {"election_id": "x", "kind": "lineup", "poll_id": "p1"}
    rows = [
        {**base, "row_id": "a", "option_id": o, "share": s} for o, s in (("A", 40.0), ("B", 35.0), ("C", 25.0))
    ] + [
        {**base, "row_id": "b", "option_id": o, "share": s} for o, s in (("A", 39.0), ("B", 34.0), ("C", 25.5), ("D", 1.5))
    ] + [
        {**base, "row_id": "c", "option_id": o, "share": s} for o, s in (("A", 35.0), ("B", 30.0), ("C", 25.0), ("E", 10.0))
    ]
    out = drop_restricted_lineups(pl.DataFrame(rows))
    # a ⊂ b and b adds only a minor candidate → drop a; a ⊂ c but c adds a 10% candidate → keep
    assert sorted(out["row_id"].unique().to_list()) == ["b", "c"]


@pytest.fixture(scope="module")
def real_long(reg, firms):
    specs = load_page_specs()
    return pl.concat([build_page(specs[k], reg, firms)[0] for k in ("2002", "2012", "2022")])


@pytest.mark.parametrize("year", [2002, 2012, 2022])
def test_collapse_reproduces_committed_csv(real_long, reg, year):
    """Regression: the long table, collapsed like polls_helpers does, gives the
    same polls and values as polling_data/{year}_first_round_polls.csv. "<x"
    cells are compared at their bound (polls_helpers reads "<1%" as 1)."""
    long = real_long.with_columns(
        pl.when(pl.col("share_lt")).then(pl.col("share") * 2).otherwise(pl.col("share")).alias("share"))
    mine = collapse_to_ballot(long, reg, year)
    ref = pl.read_csv(f"polling_data/{year}_first_round_polls.csv", try_parse_dates=True, infer_schema_length=None)
    key = ["pollster", "fieldwork_start", "fieldwork_end", "sample_size"]
    assert ref.height == mine.height
    joined = ref.join(mine, on=key, how="inner", suffix="_m", nulls_equal=True)
    assert joined.height == ref.height
    cands = [c.surname for c in reg.ballot(year)]
    differing = [r for r in joined.iter_rows(named=True)
                 if any(float(r[c] or 0) != float(r.get(c + "_m") or 0) for c in cands if c in ref.columns)]
    # Only multi-lineup polls may differ: the post's multi-way-ballot rule can
    # keep a different lineup than polls_helpers' "closest to the ballot".
    assert all(r["n_scenarios"] > 1 for r in differing)
    assert len(differing) <= 2

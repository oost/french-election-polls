"""The forecast as one PDF: headline numbers, charts and tables for the
presidential forecast, its scenarios, turnout, the legislative forecast and
the backtest. Built with reportlab (`pip install -e '.[pdf]'`); charts are
the notebook's matplotlib charts, embedded as images.

Layout: a title band and summary page (headline numbers, key points, the
main chart, how the model works, contents), then numbered sections with
ruled headings, tables with tinted headers, and a footer with the page count.
Text and charts share one typeface, the first of `FONT_FAMILIES` installed."""
from __future__ import annotations

import io
from datetime import date, timedelta
from functools import cache
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import font_manager
from matplotlib.colors import PowerNorm, to_hex
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    CondPageBreak, HRFlowable, Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

from forecast import paths
from forecast.backtest import charts as bt_charts
from forecast.backtest.charts import GRID, INK, MUTED
from forecast.backtest.metrics import calibration, coverage, scores, share_errors
from forecast.presidential.pipeline import CLEAR_SHARE
from forecast.report import charts

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
TOP, BOTTOM = 17 * mm, 20 * mm
FRAME_W = PAGE_W - 2 * MARGIN
BAND_H = 40 * mm                 # title band on the first page
ACCENT = "#0e3d7e"               # the darkest step of the charts' blue ramp
ACCENT_SOFT = "#c9dcf4"          # text on the band
TINT = "#f2f5fa"                 # tiles, table headers, the key-points box
FONT_FAMILIES = ("Arial", "Liberation Sans", "DejaVu Sans")   # first one installed; the last ships with matplotlib
_c = colors.HexColor
BLOCS = "EXD far right · GAU left · CEN centre · DRO right · EXG far left · DIV other."
UNITS = ("RN with its allies · EXD the rest of the far right · GAU left (without LFI when the left runs split) · "
         "RAD LFI, when the left runs split · CEN centre · DRO right · EXG far left · DIV other.")
_toc: list[str] = []             # numbered section titles of the report being built


@cache
def _font_family() -> str:
    """Register the report's typeface as "Body" and return the family name
    (also used for the charts, so text and figures match)."""
    variants = (("Body", "normal", "normal"), ("Body-Bold", "bold", "normal"),
                ("Body-Italic", "normal", "italic"), ("Body-BoldItalic", "bold", "italic"))
    for family in FONT_FAMILIES:
        try:
            files = [font_manager.findfont(font_manager.FontProperties(family=family, weight=w, style=st),
                                           fallback_to_default=False) for _, w, st in variants]
        except ValueError:
            continue
        if len(set(files)) < 4 or not all(f.lower().endswith(".ttf") for f in files):
            continue                 # a variant is missing, or the family is a collection file
        for (name, _, _), file in zip(variants, files):
            pdfmetrics.registerFont(TTFont(name, file))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
                                      boldItalic="Body-BoldItalic")
        return family
    raise RuntimeError("no usable font family found")


@cache
def _styles() -> dict[str, ParagraphStyle]:
    _font_family()
    base = ParagraphStyle("body", fontName="Body", fontSize=9.5, leading=13.5, textColor=INK)
    return {
        "body": base,
        "meta": ParagraphStyle("meta", base, fontSize=8.5, leading=12, textColor=MUTED, spaceAfter=10),
        "h1": ParagraphStyle("h1", base, fontName="Body-Bold", fontSize=15, leading=19, textColor=ACCENT,
                             spaceBefore=16, spaceAfter=3),
        "h2": ParagraphStyle("h2", base, fontName="Body-Bold", fontSize=11, leading=14.5, spaceBefore=12,
                             spaceAfter=5),
        "box_h": ParagraphStyle("box_h", base, fontName="Body-Bold", fontSize=8, leading=11, textColor=ACCENT,
                                spaceAfter=4),
        "note": ParagraphStyle("note", base, fontSize=8, leading=10.5, textColor=MUTED, spaceBefore=4,
                               spaceAfter=6),
        "toc": ParagraphStyle("toc", base, fontSize=8.5, leading=12, textColor=MUTED, spaceBefore=5),
        "tile_n": ParagraphStyle("tile_n", base, fontName="Body-Bold", fontSize=24, leading=27, textColor=ACCENT),
        "tile_l": ParagraphStyle("tile_l", base, fontSize=8.5, leading=10.5, textColor=MUTED),
        "mh": ParagraphStyle("mh", base, fontSize=7, leading=8.5, alignment=TA_CENTER),
    }


# ---------------------------------------------------------------- formatting

def _pct(v, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if 0 < v < 0.5 * 10 ** -(digits + 2):
        return f"<{10 ** -digits:g}%"
    return f"{100 * v:.{digits}f}%"


def _label(slot: str) -> str:
    return slot.upper() if len(slot) <= 3 else slot.replace("_", "/")


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, _styles()[style])


def _table(header: list[str], rows: list[list[str]], widths: list[float] | None = None,
           right: tuple[int, ...] = ()) -> Table:
    """A table in the report's style: tinted header over an accent rule,
    hairlines between rows, numbers right-aligned (columns in `right`)."""
    t = Table([header] + rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, 0), "Body-Bold", 8), ("TEXTCOLOR", (0, 0), (-1, 0), _c(ACCENT)),
        ("BACKGROUND", (0, 0), (-1, 0), _c(TINT)),
        ("FONT", (0, 1), (-1, -1), "Body", 8.5), ("TEXTCOLOR", (0, 1), (-1, -1), _c(INK)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, _c(ACCENT)), ("LINEBELOW", (0, 1), (-1, -2), 0.3, _c(GRID)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, _c(MUTED)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 4), ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 1), (-1, -1), 3), ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    style += [("ALIGN", (c, 0), (c, -1), "RIGHT") for c in right]
    t.setStyle(TableStyle(style))
    return t


def _figure(fig, width: float = FRAME_W) -> Image:
    """A matplotlib figure as an image `width` wide (the figure is closed)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return _image(buf, width)


def _image(buf: io.BytesIO, width: float = FRAME_W) -> Image:
    buf.seek(0)
    w, h = ImageReader(buf).getSize()
    buf.seek(0)
    img = Image(buf, width=width, height=width * h / w)
    img.hAlign = "LEFT"
    img.spaceBefore, img.spaceAfter = 6, 4
    return img


def _section(title: str, *first, level: str = "h1", numbered: bool = True):
    """A heading kept on the same page as what follows it. Top-level sections
    are numbered, ruled, and start a new page when little room is left."""
    if level != "h1":
        return KeepTogether([_p(escape(title), level), *first])
    if numbered:
        _toc.append(title)
        title = f"{len(_toc)}  {title}"
    rule = HRFlowable(width="100%", thickness=0.9, color=_c(ACCENT), spaceBefore=0, spaceAfter=8)
    return [CondPageBreak((75 if numbered else 35) * mm), KeepTogether([_p(escape(title), "h1"), rule, *first])]


def _flat(items) -> list:
    """A story with nested lists flattened."""
    out = []
    for x in items:
        out.extend(_flat(x) if isinstance(x, list) else [x])
    return out


def _bullets(items: list[str]) -> ListFlowable:
    return ListFlowable([ListItem(_p(i), leftIndent=11, value="•", spaceAfter=2) for i in items],
                        bulletType="bullet", start="•", leftIndent=11, bulletFontSize=8.5, bulletColor=_c(ACCENT))


def _box(title: str, *content) -> Table:
    """A tinted box with an accent bar on the left (key points)."""
    t = Table([[[_p(escape(title.upper()), "box_h"), *content]]], colWidths=[FRAME_W], hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), _c(TINT)), ("LINEBEFORE", (0, 0), (0, -1), 3, _c(ACCENT)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def _tiles(items: list[tuple[str, str]]) -> Table:
    """Headline numbers: a big figure over a short label, side by side."""
    gap = 3 * mm
    w = (FRAME_W - gap * (len(items) - 1)) / len(items)
    row, widths = [], []
    for k, (n, l) in enumerate(items):
        if k:
            row.append("")
            widths.append(gap)
        row.append([_p(escape(n), "tile_n"), _p(escape(l), "tile_l")])
        widths.append(w)
    t = Table([row], colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 9),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 10), ("LEFTPADDING", (0, 0), (-1, -1), 10),
             ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    for col in range(0, len(row), 2):
        style += [("BACKGROUND", (col, 0), (col, 0), _c(TINT)), ("LINEABOVE", (col, 0), (col, 0), 2.5, _c(ACCENT))]
    t.setStyle(TableStyle(style))
    return t


def _matrix(pres, slot: str, measure: str) -> Table:
    """`PresSummary.conditional_matrix` as a shaded table, the same colours as
    `charts.conditional_matrix`; "–": not on that ballot."""
    m = pres.conditional_matrix(slot, measure)
    heads = pres.conditional.filter(pl.col("slot") == slot).unique("ballot", maintain_order=True)
    ballots = heads["ballot"].to_list()
    head = [""] + [_p(escape(b).replace(" + ", " +<br/>") + f'<br/><font color="{MUTED}">{p:.0%} likely</font>', "mh")
                   for b, p in zip(ballots, heads["p_ballot"])]
    norm = PowerNorm(0.5, vmin=0, vmax=1)
    data, style = [head], []
    for i, r in enumerate(m.iter_rows(named=True), start=1):
        row = [r["name"]]
        for j, b in enumerate(ballots, start=1):
            v = r[b]
            if v is None:
                row.append("–")
                style.append(("TEXTCOLOR", (j, i), (j, i), _c(MUTED)))
                continue
            row.append("<1%" if 0 < v < 0.005 else f"{v:.0%}")
            x = float(norm(v))
            style += [("BACKGROUND", (j, i), (j, i), _c(to_hex(charts.BLUES(x)))),
                      ("TEXTCOLOR", (j, i), (j, i), colors.white if x > 0.6 else _c(INK))]
        data.append(row)
    name_w = 30 * mm
    col_w = min(20 * mm, (FRAME_W - name_w) / len(ballots))
    t = Table(data, colWidths=[name_w] + [col_w] * len(ballots), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 1), (-1, -1), "Body", 8.5), ("TEXTCOLOR", (0, 1), (0, -1), _c(INK)),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
        ("LEFTPADDING", (1, 0), (-1, -1), 1.5), ("RIGHTPADDING", (1, 0), (-1, -1), 1.5),
        ("TOPPADDING", (0, 1), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 1), (-1, -1), 3.5),
        ("GRID", (1, 1), (-1, -1), 1.5, colors.white),
        ("LINEABOVE", (0, -1), (-1, -1), 5, colors.white),   # sets the combined row apart
        *style,
    ]))
    return t


# ---------------------------------------------------------------- sections

def _headline(pres, legi) -> list:
    c, mu = pres.candidates, pres.matchups
    top = c.row(0, named=True)
    tiles = [(f"{top['p_win']:.0%}", f"{top['name']} wins the presidency"),
             (f"{top['p_runoff']:.0%}", f"{top['name']} reaches the runoff")]
    if mu.height:
        tiles.append((f"{mu['p'][0]:.0%}", f"Most likely runoff: {mu['matchup'][0]}"))
    if legi is not None:
        big = legi.totals.sort("p_largest", descending=True).row(0, named=True)
        tiles.append((f"{big['p_largest']:.0%}", f"{big['unit']} largest group in the next Assembly"))
    others = ", ".join(f"{r['name']} {r['p_win']:.0%}" for r in c.slice(1, 4).iter_rows(named=True))
    items = [f"<b>{escape(top['name'])}</b> wins in {_pct(top['p_win'])} of simulations and reaches the runoff in "
             f"{_pct(top['p_runoff'])}. Next most likely winners: {escape(others)}."]
    if mu.height:
        r = mu.row(0, named=True)
        first = r["matchup"].split(" vs ")[0]
        items.append(f"Most likely runoff: <b>{escape(r['matchup'])}</b> ({_pct(r['p'])} of simulations); "
                     f"{escape(first)} wins it {r['p_first_wins']:.0%} of the time.")
    if legi is not None:
        maj = legi.totals.sort("p_majority", descending=True).row(0, named=True)
        items.append(f"Next legislative election: {escape(big['unit'])} is the largest group in "
                     f"{_pct(big['p_largest'])} of simulations; a {escape(maj['unit'])} majority (289+ seats) "
                     f"in {_pct(maj['p_majority'])}.")
    if not pres.far_right.is_empty():
        r = pres.far_right.row(0, named=True)
        prob = pres.meta["far_right"][0]
        items.append(f"This includes a {prob:.0%} chance that the far right again does worse in the runoff than the "
                     f"polls say, as in 2002, 2017 and 2022: {escape(r['name'])} wins {_pct(r['p_win_if_shortfall'])} "
                     f"of the time if it does and {_pct(r['p_win_if_not'])} if it does not.")
    dropped = pres.meta.get("scenario_mass_dropped", 0.0)
    if dropped > 1e-3:
        items.append(f'<font color="#c0392b"><b>Warning:</b> {dropped:.0%} of the scenario probability was '
                     "dropped (lineups the now-cast cannot place) and the rest renormalised.</font>")
    return [_tiles(tiles), Spacer(1, 10), _box("Key points", _bullets(items)), Spacer(1, 4),
            _figure(charts.presidential_probabilities(pres.candidates, top=8), FRAME_W * 0.78)]


def _method() -> list:
    return [KeepTogether([_p("How the model works", "h2"), _p(
        "The model averages the polls, corrected for each pollster's lean, and combines them with a model "
        "based on past election results. That gives each candidate's expected first-round share, including "
        "for lineups the pollsters have not tested. It then simulates the election 40,000 times: each time it "
        "draws who is on the ballot from the candidate scenarios below, adds an error calibrated on its own "
        "mistakes in the 2002–2022 presidential elections, and plays both rounds. A candidate's chance of "
        "winning is the share of simulations they win.")])]


def _presidential(pres) -> list:
    c = pres.candidates.filter(pl.col("p_run") > 0)
    rows = [[r["name"], r["bloc"] or "", f"{r['p_run']:.0%}", _pct(r["p_runoff"]), _pct(r["p_win"]),
             f"{r['share_q50']:.1f}", f"{r['share_q05']:.1f}–{r['share_q95']:.1f}"] for r in c.iter_rows(named=True)]
    out = [
        _section("Presidential election", _p("All candidates", "h2"), _table(
            ["Candidate", "Bloc", "Runs", "Reaches runoff", "Wins", "Median %", "90% range"], rows,
            [46 * mm, 16 * mm, 18 * mm, 28 * mm, 20 * mm, 22 * mm, 24 * mm], right=(2, 3, 4, 5, 6)),
            _p("Shares are first-round % of votes cast, when the candidate runs. " + BLOCS, "note")),
        _figure(charts.presidential_shares(pres.candidates), FRAME_W * 0.92),
    ]
    if pres.matchups.height:
        mrows = [[r["matchup"], _pct(r["p"]), f"{r['p_first_wins']:.0%}"]
                 for r in pres.matchups.head(8).iter_rows(named=True)]
        out.append(_section("Most likely runoffs", _table(["Runoff", "Probability", "First named wins"], mrows,
                                                          [70 * mm, 26 * mm, 32 * mm], right=(1, 2)), level="h2"))
    out += _far_right(pres)
    if pres.blocs.height and "p_runoff" in pres.blocs.columns:
        brows = [[r["bloc"], _pct(r["p_runoff"]), _pct(r["p_win"])] for r in pres.blocs.iter_rows(named=True)]
        out.append(_section("By bloc", _table(["Bloc", "Reaches runoff", "Wins"], brows,
                                              [20 * mm, 30 * mm, 24 * mm], right=(1, 2)), level="h2"))
    return out


def _far_right(pres) -> list:
    """The far-right runoff scenario: its probability, size and what it changes."""
    if pres.far_right.is_empty():
        return []
    prob, mean, sd, by_year = pres.meta["far_right"]
    points = 100 * (0.55 - 1 / (1 + np.exp(-(np.log(0.55 / 0.45) + mean))))
    rows = [[r["name"], _pct(r["p_win"]), _pct(r["p_win_if_shortfall"]), _pct(r["p_win_if_not"])]
            for r in pres.far_right.iter_rows(named=True)]
    past = ", ".join(str(y) for y in by_year)
    return [_section("Does the far right fall short of its runoff polls again?", _p(
        f"In the three runoffs with a far-right finalist ({past}), that finalist did worse than the final polls "
        "each time, by 3 to 5 points. Other trailing finalists did not. Each time the far right was the underdog "
        f"facing a broad front; this time its candidate leads the runoff polls. So the model treats it as a scenario: "
        f"in {prob:.0%} of simulations (a judgment call) the far-right finalist loses about {points:.1f} points of "
        "runoff vote share, give or take half as much; in the rest, nothing."), Spacer(1, 5),
        _table(["Chance of winning", "Overall", "If it happens", "If it does not"], rows,
               [50 * mm, 24 * mm, 28 * mm, 28 * mm], right=(1, 2, 3)), level="h2")]


def _scenarios(pres) -> list:
    out = []
    spec = pres.meta.get("spec")
    names = dict(zip(pres.candidates["candidate_id"], pres.candidates["name"]))
    if spec is not None:
        def lab(opt):
            return " + ".join(names.get(c, c) for c in opt.split("+")) if opt else "nobody"
        rows = []
        for slot, opts in spec.slots.items():
            rows += [[_label(slot) if k == 0 else "", lab(o), f"{p:.0%}"] for k, (o, p) in enumerate(opts.items())]
        rows += [["runs or not" if k == 0 else "", names.get(c, c), f"{p:.0%}"]
                 for k, (c, p) in enumerate(spec.optional.items())]
        fixed = ", ".join(names.get(c, c) for c in spec.fixed)
        out.append(_section("Candidate scenarios", _p(
            "Each simulation first draws who is on the ballot. These probabilities are judgment calls set in "
            "<i>scenarios_pres_2027.yaml</i>, not estimates; the file's header dates and explains them."),
            Spacer(1, 4), _table(["Draw", "Option", "Probability"], rows, [30 * mm, 90 * mm, 24 * mm], right=(2,)),
            _p(f"On every ballot: {escape(fixed)}.", "note")))
    if pres.conditional.is_empty():
        return out
    slots = pres.conditional.group_by("slot", maintain_order=True).agg(pl.col("ballot").n_unique())
    for slot in slots.sort("ballot", descending=True, maintain_order=True)["slot"]:   # most options first
        heads = pres.conditional.filter(pl.col("slot") == slot).unique("ballot")
        n_min = int(heads["n_sims"].min())
        note = (f"Each column uses only the simulations that drew that ballot; weighting the columns by their "
                f"probability gives the headline numbers. The last row takes the {_label(slot)} candidates together. "
                f"The smallest column has {n_min:,} simulations, so its values can be off by up to "
                f"±{1.96 * 50 / np.sqrt(n_min):.1f} points.")
        out.append(_section(f"Given the {_label(slot)} ballot: chance of reaching the runoff",
                            _matrix(pres, slot, "p_runoff"), level="h2"))
        out.append(_section(f"Given the {_label(slot)} ballot: chance of winning", _matrix(pres, slot, "p_win"),
                            _p(note, "note"), level="h2"))
    return out


def _turnout(pres) -> list:
    if pres.turnout.is_empty():
        return []
    tm = pres.meta.get("turnout_model")
    basis = "past elections only: pollsters have not yet published abstention estimates for this election"
    if tm is not None and (tm.n_polls_t1 or tm.runoff_polls):
        basis = (f"past elections and pollsters' abstention estimates ({tm.n_polls_t1} first-round polls, "
                 f"{sum(e[2] for e in tm.runoff_polls.values())} runoff polls), corrected for how far off such "
                 "estimates were at the same distance from the vote in earlier campaigns")
    rows = [[r["measure"], f"{r['q50']:.1f}", f"{r['q05']:.1f}–{r['q95']:.1f}", f"{r['last']:.1f}"]
            for r in pres.turnout.iter_rows(named=True)]
    out = [_section("Turnout", _p(
        f"Turnout is the share of registered voters who vote. For now the forecast rests on {basis}. Runoff "
        "turnout comes from a model of who stays home: each first-round electorate chooses between the two "
        "finalists and not voting, depending on how far each finalist is from it. The model is fitted on the "
        "eleven runoffs since 1965 and on the 2012 and 2017 polls of how each electorate would split."),
        Spacer(1, 6),
        _table(["% of registered voters", "Median", "90% range", f"{pres.year - 5}"], rows,
               [50 * mm, 20 * mm, 28 * mm, 18 * mm], right=(1, 2, 3)),
        _p("Valid votes leave out blank and spoiled ballots.", "note")),
        _figure(charts.turnout_history(pres), FRAME_W * 0.85)]
    if pres.turnout_matchups.is_empty():
        return out
    m = pres.turnout_matchups
    mrows, unclear = [], []
    for i, r in enumerate(m.iter_rows(named=True), start=1):
        mrows.append([r["matchup"], _pct(r["p"]), f"{r['turnout_q50']:.1f}", f"{r['p_first_wins']:.0%}",
                      f"{r['p_first_wins_low']:.0%} ±{196 * r['p_first_wins_low_se']:.0f}",
                      f"{r['p_first_wins_high']:.0%} ±{196 * r['p_first_wins_high_se']:.0f}",
                      f"{r['share_per_3pts']:+.1f} ±{1.96 * r['share_per_3pts_se']:.1f}",
                      "" if r["clear_effect"] else "no clear effect"])
        if not r["clear_effect"]:
            unclear.append(i)
    table = _table(["Runoff (first named …)", "Prob.", "Turnout", "… wins", "turnout low", "turnout high",
                    "share / +3 pts", ""], mrows,
                   [40 * mm, 14 * mm, 16 * mm, 14 * mm, 23 * mm, 24 * mm, 25 * mm, 18 * mm], right=(1, 2, 3, 4, 5, 6))
    table.setStyle(TableStyle([("TEXTCOLOR", (4, i), (7, i), _c(MUTED)) for i in unclear]
                              + [("FONT", (7, 1), (7, -1), "Body-Italic", 6.5)]))
    out.append(_section("Does runoff turnout change the winner?", table, _p(
        f"“Low” and “high” are the thirds of that runoff's simulations where turnout comes in furthest below and "
        f"above what its first round would normally give: about {-m['surprise_low'].mean():.0f} points below and "
        f"{m['surprise_high'].mean():.0f} above. ± is the 95% simulation margin. The share column is the change in the "
        "first-named candidate's vote share that goes with three more points of turnout. A row is marked "
        f"“no clear effect” when that change is under {CLEAR_SHARE} points or within simulation noise; the gap "
        "between its low and high figures should then not be read as an effect of turnout. Turnout matters where the voters most likely to stay home lean "
        "heavily to one finalist. The model only lets more or fewer people from each electorate vote the way that "
        "electorate already leans; it cannot represent a surge driven by rejection of one candidate, as in 2002. "
        "No past runoff has opposed the far right and the radical left, so the figures for such a runoff rest on "
        "how other runoffs behaved.", "note"), level="h2"))
    return out


def _legislative(legi) -> list:
    nat = ", ".join(f"{u} {100 * s:.1f}%" for u, s in sorted(legi.national.items(), key=lambda x: -x[1]))
    scen = ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in legi.scenarios.items())
    rows = [[r["unit"], f"{r['mean_seats']:.0f}", f"{r['q05']:.0f}–{r['q95']:.0f}", _pct(r["p_majority"]),
             _pct(r["p_largest"])] for r in legi.totals.iter_rows(named=True)]
    widths = [18 * mm, 24 * mm, 24 * mm, 30 * mm, 30 * mm]
    turnout = ""
    if legi.turnout is not None:
        turnout = (f" Valid votes are assumed at {legi.turnout:.0%} of registered voters ({legi.turnout_last:.0%} "
                   "in the 2024 snap election); a candidate needs 12.5% of registered voters to reach the second round.")
    if legi.triangulaires is not None:
        turnout += (f" Three-way runoffs after withdrawals: median {legi.triangulaires[0]:.0f} seats "
                    f"(90% range {legi.triangulaires[1]:.0f}–{legi.triangulaires[2]:.0f}).")
    out = [_section("Next legislative election", _p(
        f"First round assumed on {legi.election_t1:%-d %B %Y}. National polling average: {escape(nat)}. "
        f"Latest national legislative poll: {legi.last_poll}. Alliance scenarios: {escape(scen)}." + turnout),
        Spacer(1, 6),
        _table(["Unit", "Mean seats", "90% range", "Majority (289+)", "Largest group"], rows, widths,
               right=(1, 2, 3, 4)))]
    out.append(_figure(charts.legislative_seats(legi.totals), FRAME_W * 0.85))
    for name, df in legi.by_scenario.items():
        srows = [[r["unit"], f"{r['mean_seats']:.0f}", f"{r['q05']:.0f}–{r['q95']:.0f}", _pct(r["p_majority"]),
                  _pct(r["p_largest"])] for r in df.iter_rows(named=True)]
        out.append(_section(f"If {name.replace('_', ' ')} ({legi.scenarios.get(name, float('nan')):.0%})",
                            _table(["Unit", "Mean seats", "90% range", "Majority (289+)", "Largest group"], srows,
                                   widths, right=(1, 2, 3, 4)), level="h2"))
    out.append(_p(UNITS, "note"))
    return out


def _turnout_backtest() -> list:
    """Turnout predicted for 2002–2022 from earlier elections and the polls of the time."""
    from forecast.config import default_config
    from forecast.simulate.turnout import backtest_turnout
    bt = backtest_turnout(default_config().turnout)
    rows = []
    for rnd, label, picks in ((1, "First round", (("history only", 2), ("with polls", 60), ("with polls", 7), ("with polls", 2))),
                              (2, "Runoff", (("history only", 2), ("with polls", 5), ("with polls", 2)))):
        for basis, days in picks:
            d = bt.filter((pl.col("round") == rnd) & (pl.col("basis") == basis) & (pl.col("days") == days))
            if d.is_empty():
                continue
            when = "any time" if basis == "history only" else f"{days} days before"
            same = bt.filter((pl.col("round") == rnd) & (pl.col("basis") == "history only") & (pl.col("days") == days)
                             & pl.col("year").is_in(d["year"].to_list()))
            rows.append([label, "Past elections only" if basis == "history only" else "With abstention polls", when,
                         ", ".join(str(y) for y in d["year"]), f"{d['error'].abs().mean():.1f}",
                         "" if basis == "history only" else f"{same['error'].abs().mean():.1f}",
                         f"{int(d['inside'].sum())} of {d.height}"])
    return [_section("Turnout predictions for past elections", _table(
        ["Round", "Based on", "When", "Elections", "Mean error", "No polls", "In range"], rows,
        [19 * mm, 37 * mm, 25 * mm, 40 * mm, 20 * mm, 17 * mm, 16 * mm], right=(4, 5, 6)),
        _p("Mean absolute error in points of turnout. Each prediction uses only earlier elections and the polls "
           "published by then; runoff predictions take the first-round result as known. “No polls” is the "
           "error on the same elections from past elections alone, and “in range” counts results inside the 90% range. Pollsters published first-round abstention "
           "estimates in 2012 and 2017 only; two months out they did worse than history alone.", "note"), level="h2")]


def _backtest(archive: pl.DataFrame) -> list:
    cycles = sorted(archive["cycle"].unique().to_list())
    n_snap = archive.select(pl.struct("cycle", "days_out").n_unique()).item()
    sc, cov = scores(archive), coverage(archive).row(0, named=True)
    err = share_errors(archive)
    erows = [[r["horizon"], f"{r['mae_polls']:.2f}", f"{r['mae_fundamentals']:.2f}", f"{r['mae_ensemble']:.2f}",
              f"{r['mean_w_polls']:.0%}", f"{r['n']:,}"] for r in err.iter_rows(named=True)]
    srows = [["Winner called (favourite won)", f"{sc['winner_called']:.0%}"],
             ["… excluding toss-ups (favourite below 65%)", f"{sc['winner_called_excl_tossups']:.0%}"],
             ["Brier score, P(reaches runoff)", f"{sc['brier_p_runoff']:.3f}"],
             ["Brier score, P(wins)", f"{sc['brier_p_win']:.3f}"],
             ["Results inside the 50% interval (target 50%)", f"{cov['cov50']:.0%}"],
             ["Results inside the 90% interval (target 90%)", f"{cov['cov90']:.0%}"]]
    cr, cw = calibration(archive, "p_runoff", "made_runoff"), calibration(archive, "p_win", "won")
    crows = [[a["bucket"], _pct(a["mean_p"], 0), _pct(a["freq"], 0), f"{a['n']:,}",
              _pct(b["mean_p"], 0), _pct(b["freq"], 0), f"{b['n']:,}"]
             for a, b in zip(cr.iter_rows(named=True), cw.iter_rows(named=True))]
    charts_png = []
    for fn in (bt_charts.error_by_horizon, bt_charts.calibration_chart):
        buf = io.BytesIO()
        fn(archive, buf)
        charts_png.append(_image(buf, FRAME_W * 0.8))
    return [
        _section("How the model did in past elections", _p(
            f"Strict out-of-sample backtest over the {cycles[0]}–{cycles[-1]} presidential elections "
            f"({len(cycles)} elections, {n_snap} forecast dates, {archive.height:,} candidate forecasts). Each "
            "forecast uses only polls published by its date and earlier elections' results."), Spacer(1, 6),
            _table(["Score", ""], srows, [100 * mm, 24 * mm], right=(1,))),
        _section("Error by days before the first round", _table(
            ["Days before", "Polls", "Fundamentals", "Ensemble", "Polls weight", "Forecasts"], erows,
            [26 * mm, 20 * mm, 26 * mm, 22 * mm, 24 * mm, 22 * mm], right=(1, 2, 3, 4, 5)),
            _p("Mean absolute error of first-round shares, in points.", "note"), level="h2"),
        charts_png[0],
        _section("Calibration", _table(
            ["Predicted", "Runoff: mean", "observed", "n", "Win: mean", "observed", "n"], crows,
            [24 * mm, 24 * mm, 20 * mm, 16 * mm, 22 * mm, 20 * mm, 16 * mm], right=(1, 2, 3, 4, 5, 6)),
            _p("Forecasts grouped by predicted probability, against how often the event happened.", "note"),
            charts_png[1], level="h2"),
    ]


def _notes(pres, archive: pl.DataFrame | None = None) -> list:
    n = pres.meta.get("n_sims") or 40_000
    extra = []
    if archive is not None and archive.height:
        fav = next((r for r in calibration(archive, "p_win", "won").iter_rows(named=True) if r["bucket"] == "50%–70%"), None)
        if fav and fav["n"] >= 20 and fav["freq"] > 0.75:
            extra.append(f"In the backtest the model was too cautious about favourites: candidates it gave a 50–70% "
                         f"chance of winning went on to win {fav['freq']:.0%} of the time ({fav['n']} forecasts, "
                         f"from {archive['cycle'].n_unique()} elections).")
    return [_section("Notes and limitations", numbered=False, *[_bullets(extra + [
        "Scenario probabilities (who is on the ballot) and the legislative alliance scenarios are judgment "
        "calls, not estimates.",
        f"With {n:,} simulations, a headline probability carries up to ±{1.96 * 50 / np.sqrt(n):.1f} points of "
        "simulation noise. The same inputs always give the same numbers.",
        "Only five past presidential elections train the error model, so the tails are wide. The backtest "
        "conditions on the actual ballot, because candidate declaration dates are not curated.",
        "A candidate who reaches the runoff by beating their first-round polls is stronger there only through "
        "their larger first-round vote. A further carry-over of momentum was tested and scored worse on past "
        "elections.",
    ])])]


def write_pdf(pres, legi=None, archive: pl.DataFrame | None = None, path: Path | None = None) -> Path:
    """Write the report and return its path (default:
    data/outputs/forecast_<year>_<as_of>.pdf). `legi` (a LegiSummary) and
    `archive` (the presidential backtest archive) are optional sections."""
    path = Path(path or paths.OUTPUTS / f"forecast_{pres.year}_{pres.as_of.isoformat()}.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    t1 = pres.as_of + timedelta(days=pres.days_out)
    title = "French election forecasts"
    sub = f"{pres.year} presidential election" + (" and the next legislative election" if legi is not None else "")
    as_of = f"{pres.as_of:%-d %B %Y}"
    meta = (f"Based on polls published by {as_of}, {pres.days_out} days before the first round on {t1:%-d %B %Y}. "
            f"{pres.meta.get('n_sims') or 0:,} simulations, {pres.meta.get('mode', 'polls-only')} model. "
            f"Generated {date.today():%-d %B %Y}.")
    _toc.clear()
    with plt.rc_context({"font.family": _font_family()}):     # charts in the report's typeface
        body = [*_presidential(pres), *_scenarios(pres), *_turnout(pres)]
        if legi is not None:
            body += _legislative(legi)
        if archive is not None and archive.height:
            body += _backtest(archive)
            if not pres.turnout.is_empty():
                body += _turnout_backtest()
        body += _notes(pres, archive)
        contents = "<b>Contents</b>&nbsp;&nbsp; " + " &nbsp;·&nbsp; ".join(
            f"{k}&nbsp;{escape(t)}" for k, t in enumerate(_toc, start=1))
        story = _flat([Spacer(1, BAND_H - TOP + 7 * mm), _p(escape(meta), "meta"), *_headline(pres, legi),
                       *_method(), HRFlowable(width="100%", thickness=0.6, color=_c(GRID), spaceBefore=10, spaceAfter=0),
                       _p(contents, "toc"), *body])

    def first_page(canvas, doc):
        """The title band."""
        canvas.saveState()
        canvas.setFillColor(_c(ACCENT))
        canvas.rect(0, PAGE_H - BAND_H, PAGE_W, BAND_H, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Body-Bold", 24)
        canvas.drawString(MARGIN, PAGE_H - 20 * mm, title)
        canvas.setFont("Body", 12)
        canvas.setFillColor(_c(ACCENT_SOFT))
        canvas.drawString(MARGIN, PAGE_H - 28.5 * mm, sub)
        canvas.setFont("Body", 7.5)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 17 * mm, "FORECAST AS OF")
        canvas.setFillColor(colors.white)
        canvas.setFont("Body-Bold", 12)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 22.5 * mm, as_of)
        canvas.restoreState()

    class Numbered(Canvas):
        """Adds the footer once the page count is known, and a short accent
        bar at the top of every page after the first."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            for state in self._saved:
                self.__dict__.update(state)
                self.setStrokeColor(_c(GRID))
                self.setLineWidth(0.6)
                self.line(MARGIN, 13.5 * mm, PAGE_W - MARGIN, 13.5 * mm)
                self.setFont("Body", 7.5)
                self.setFillColor(_c(MUTED))
                self.drawString(MARGIN, 9.5 * mm, f"{title}  ·  {sub}  ·  as of {as_of}")
                self.drawRightString(PAGE_W - MARGIN, 9.5 * mm, f"Page {self._pageNumber} of {len(self._saved)}")
                if self._pageNumber > 1:
                    self.setFillColor(_c(ACCENT))
                    self.rect(MARGIN, PAGE_H - 9 * mm, 22 * mm, 1.2 * mm, stroke=0, fill=1)
                super().showPage()
            super().save()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=TOP,
                            bottomMargin=BOTTOM, title=f"{title}: {sub}", subject=meta)
    doc.build(story, onFirstPage=first_page, canvasmaker=Numbered)
    return path

"""Build a one-file HTML site from the Markdown docs.

    python docs/build_site.py            # → docs/site/index.html (open in a browser)

Each Markdown file becomes a section of one page; the sidebar switches
between them with #anchors. Formulas are typeset by MathJax and the pipeline
diagram by Mermaid (both loaded from jsdelivr). `--fragment PATH` also writes
a variant without the <html> skeleton or the Mermaid script, for hosts that
supply both.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify
from pygments.formatters import HtmlFormatter

DOCS = Path(__file__).resolve().parent
PAGES = [  # (key, markdown file, nav label)
    ("home", "README.md", "Start here"),
    ("overview", "model-overview.md", "Model overview"),
    ("tutorial", "bayesian-tutorial.md", "Bayesian tutorial"),
    ("files", "file-guide.md", "File guide"),
]
FILE_TO_KEY = {f: k for k, f, _ in PAGES}

MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&"
         "family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;"
         "1,8..60,400&display=swap")


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Swap $…$ / $$…$$ (outside code) for placeholders so Markdown leaves
    underscores and asterisks inside formulas alone."""
    store: list[str] = []
    out, i = [], 0
    for block in re.split(r"(```.*?```)", text, flags=re.S):
        if block.startswith("```"):
            out.append(block)
            continue
        def keep(m):
            store.append(m.group(0))
            return f"@@MATH{len(store) - 1}@@"
        block = re.sub(r"\$\$.+?\$\$", keep, block, flags=re.S)
        block = re.sub(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", keep, block)
        # inline code spans may contain $; restore them if caught
        out.append(block)
    return "".join(out), store


def _restore_math(html_text: str, store: list[str]) -> str:
    def put(m):
        tex = store[int(m.group(1))]
        if tex.startswith("$$"):
            return f'<div class="math">\\[{html.escape(tex[2:-2])}\\]</div>'
        return f"\\({html.escape(tex[1:-1])}\\)"
    html_text = re.sub(r"<p>@@MATH(\d+)@@</p>", put, html_text)
    return re.sub(r"@@MATH(\d+)@@", put, html_text)


def _rewrite_links(html_text: str, key: str) -> str:
    def fix(m):
        href = m.group(1)
        if href.startswith("#"):
            return f'href="#{key}-{href[1:]}"'
        path, _, frag = href.partition("#")
        if path in FILE_TO_KEY:
            target = FILE_TO_KEY[path]
            return f'href="#{target}-{frag}"' if frag else f'href="#{target}"'
        if not href.startswith(("http://", "https://")):
            return 'data-local="1"'           # repo-relative file: shown as text, not a dead link
        return f'href="{href}" rel="noopener"'
    html_text = re.sub(r'href="([^"]+)"', fix, html_text)
    return re.sub(r"<a data-local=\"1\">(.*?)</a>", r'<span class="repo-path">\1</span>', html_text)


def render(key: str, path: Path) -> tuple[str, list[tuple[str, str]], str]:
    """(section html, [(id, h2 text)], page title)."""
    text = path.read_text(encoding="utf-8")
    text, store = _protect_math(text)
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list"],
        extension_configs={
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
            "toc": {"slugify": lambda v, sep: f"{key}-{slugify(v, sep)}", "toc_depth": "2-3"},
        },
    )
    body = md.convert(text)
    # mermaid fences come out as highlighted code; turn them back into diagrams
    body = re.sub(r'<div class="codehilite"><pre><span></span><code>(flowchart.*?)</code></pre></div>',
                  lambda m: f'<pre class="mermaid">{html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))}</pre>',
                  body, flags=re.S)
    body = _restore_math(body, store)
    body = _rewrite_links(body, key)
    title = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    heads = [(m.group(1), re.sub(r"<[^>]+>", "", m.group(2)))
             for m in re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', body, re.S)]
    return body, heads, re.sub(r"<[^>]+>", "", title.group(1)) if title else key


def _pygments_css() -> str:
    light = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    dark = HtmlFormatter(style="github-dark").get_style_defs(".codehilite")
    dark_rules = "\n".join(line for line in dark.splitlines() if line.startswith(".codehilite"))
    scoped = lambda sel: "\n".join(f"{sel} {r}" for r in dark_rules.splitlines())
    strip_bg = lambda css: re.sub(r"\.codehilite \{ background: [^;]+; \}", "", css)
    return (strip_bg(light) + "\n@media (prefers-color-scheme: dark) {\n"
            + strip_bg(scoped(':root:not([data-theme="light"])')) + "\n}\n"
            + strip_bg(scoped(':root[data-theme="dark"]')))


CSS = r"""
:root {
  --paper: #f6f7f9; --surface: #ffffff; --ink: #1b2130; --muted: #566074; --rule: #dde1e8;
  --accent: #2b4bb1; --accent-soft: #e7ecfa; --mark: #b8322a; --code-bg: #f0f2f6;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper: #11141b; --surface: #181c25; --ink: #e5e8ef; --muted: #9aa3b4; --rule: #2a303d;
    --accent: #93a9ff; --accent-soft: #1f2842; --mark: #ef7a70; --code-bg: #1d222c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper: #11141b; --surface: #181c25; --ink: #e5e8ef; --muted: #9aa3b4; --rule: #2a303d;
  --accent: #93a9ff; --accent-soft: #1f2842; --mark: #ef7a70; --code-bg: #1d222c;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink); font: 17px/1.62 var(--serif); }
.shell { display: grid; grid-template-columns: 17rem minmax(0, 1fr); gap: 3rem; max-width: 76rem;
         margin: 0 auto; padding-inline: 24px; padding-block: 2rem 5rem; }
nav.side { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 1.5rem); align-self: start;
           max-height: calc(100vh - 3rem); overflow-y: auto; font-family: var(--sans); font-size: 14px; }
.brand { display: grid; gap: .15rem; padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid var(--rule); }
.brand strong { font-size: 16px; font-weight: 600; letter-spacing: -.005em; }
.brand span { color: var(--muted); font-size: 13px; }
.tricolore { display: flex; height: 3px; width: 2.25rem; margin-bottom: .5rem; }
.tricolore i { flex: 1; } .tricolore i:nth-child(1) { background: #2b4bb1; }
.tricolore i:nth-child(2) { background: var(--rule); } .tricolore i:nth-child(3) { background: #c53a31; }
nav.side ol { list-style: none; margin: 0; padding: 0; display: grid; gap: .1rem; }
nav.side a { color: var(--muted); text-decoration: none; display: block; padding: .3rem .6rem; border-radius: 6px; }
nav.side a:hover { color: var(--ink); background: var(--accent-soft); }
nav.side a.current { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
nav.side .toc { margin: .2rem 0 .6rem .6rem; border-left: 1px solid var(--rule); }
nav.side .toc a { font-size: 13px; padding: .2rem .7rem; border-radius: 0 6px 6px 0; }
.facts { margin-top: 1.4rem; padding-top: 1rem; border-top: 1px solid var(--rule); display: grid; gap: .45rem;
         font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.facts b { color: var(--ink); font-weight: 600; font-family: var(--mono); font-size: 12.5px; }
main { min-width: 0; max-width: 46rem; }
section.doc[hidden] { display: none !important; }
h1, h2, h3 { font-family: var(--sans); line-height: 1.25; text-wrap: balance; color: var(--ink); }
h1 { font-size: 2.05rem; font-weight: 600; letter-spacing: -.015em; margin: .2rem 0 1.2rem; }
h2 { font-size: 1.4rem; font-weight: 600; margin: 2.8rem 0 .8rem; padding-top: .4rem; border-top: 1px solid var(--rule); }
h3 { font-size: 1.08rem; font-weight: 600; margin: 2rem 0 .6rem; }
p, li { max-width: 68ch; }
a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2.5rem 0; }
strong { font-weight: 600; }
code { font-family: var(--mono); font-size: .86em; background: var(--code-bg); padding: .1em .35em; border-radius: 4px; }
.repo-path { font-family: var(--mono); font-size: .86em; color: var(--muted); }
.codehilite { background: var(--code-bg); border-radius: 8px; overflow-x: auto; margin: 1.2rem 0; }
.codehilite pre { margin: 0; padding: 1rem 1.1rem; font: 13.5px/1.55 var(--mono); }
.codehilite code { background: none; padding: 0; font-size: inherit; }
.table-wrap { overflow-x: auto; margin: 1.2rem 0; }
table { border-collapse: collapse; font-family: var(--sans); font-size: 14.5px; line-height: 1.45; min-width: 100%; }
th, td { text-align: left; vertical-align: top; padding: .55rem .8rem .55rem 0; border-bottom: 1px solid var(--rule); }
th { font-weight: 600; font-size: 12.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
td code { white-space: nowrap; }
.math { overflow-x: auto; margin: 1.1rem 0; }
mjx-container[display="true"] { margin: 0 !important; }
pre.mermaid { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 1rem;
              overflow-x: auto; text-align: center; font-family: var(--sans); }
blockquote { margin: 1.2rem 0; padding: .2rem 1rem; border-left: 3px solid var(--mark); color: var(--muted); }
.pager { display: flex; justify-content: space-between; gap: 1rem; margin-top: 3.5rem; padding-top: 1.2rem;
         border-top: 1px solid var(--rule); font-family: var(--sans); font-size: 14px; flex-wrap: wrap; }
.pager a { text-decoration: none; display: grid; gap: .1rem; }
.pager a span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
.pager a.next { text-align: right; margin-left: auto; }
@media (max-width: 860px) {
  .shell { grid-template-columns: minmax(0, 1fr); gap: 1.5rem; padding-inline: 16px; padding-block: 1rem 4rem; }
  nav.side { position: static; max-height: none; }
  nav.side .toc, .facts { display: none; }
  nav.side > ol { display: flex; flex-wrap: wrap; gap: .3rem; }
  body { font-size: 16px; }
  h1 { font-size: 1.65rem; }
}
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
"""

JS = r"""
(function () {
  const docs = [...document.querySelectorAll('section.doc')];
  const links = [...document.querySelectorAll('nav.side > ol > li > a')];
  const tocs = [...document.querySelectorAll('nav.side .toc')];
  function show(hash) {
    const id = (hash || '').replace(/^#/, '');
    let doc = docs.find(d => d.id === id) || docs.find(d => id && d.querySelector('[id="' + id + '"]')) || docs[0];
    docs.forEach(d => { d.hidden = d !== doc; });
    links.forEach(a => a.classList.toggle('current', a.getAttribute('href') === '#' + doc.id));
    tocs.forEach(t => { t.hidden = t.dataset.doc !== doc.id; });
    const target = id && id !== doc.id ? document.getElementById(id) : null;
    if (target) target.scrollIntoView(); else window.scrollTo(0, 0);
    try { localStorage.setItem('docs-last', doc.id); } catch (e) {}
  }
  window.addEventListener('hashchange', () => show(location.hash));
  let start = location.hash;
  if (!start) { try { start = '#' + (localStorage.getItem('docs-last') || ''); } catch (e) {} }
  show(start);
})();
"""


def build(fragment: bool = False) -> str:
    sections, nav = [], []
    rendered = [(k, *render(k, DOCS / f), label) for k, f, label in PAGES]
    for i, (key, body, heads, title, label) in enumerate(rendered):
        body = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', body, flags=re.S)
        prev_ = rendered[i - 1] if i > 0 else None
        next_ = rendered[i + 1] if i + 1 < len(rendered) else None
        pager = '<nav class="pager" aria-label="Previous and next">'
        if prev_:
            pager += f'<a href="#{prev_[0]}"><span>Previous</span>{prev_[4]}</a>'
        if next_:
            pager += f'<a class="next" href="#{next_[0]}"><span>Next</span>{next_[4]}</a>'
        pager += "</nav>"
        sections.append(f'<section class="doc" id="{key}" aria-label="{html.escape(title)}">{body}{pager}</section>')
        toc = "".join(f'<li><a href="#{hid}">{text}</a></li>' for hid, text in heads)
        nav.append(f'<li><a href="#{key}">{label}</a>'
                   + (f'<ol class="toc" data-doc="{key}">{toc}</ol>' if heads else "") + "</li>")
    facts = """<div class="facts" aria-label="Model at a glance">
      <div><b>40,000</b> simulations per forecast</div>
      <div><b>5</b> presidential training cycles, 2002–2022</div>
      <div><b>ν = 5</b> Student-t degrees of freedom</div>
      <div><b>577</b> constituencies · majority <b>289</b></div>
      <div><b>n/(n+5)</b> house-effect shrinkage</div>
    </div>"""
    page = f"""<title>French Election Forecast Guide</title>
<meta name="description" content="How the FiftyPlusOne-style French election forecast works: model, files, and a Bayesian tutorial.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}
{_pygments_css()}</style>
<div class="shell">
  <nav class="side" aria-label="Documentation">
    <div class="brand"><div class="tricolore" aria-hidden="true"><i></i><i></i><i></i></div>
      <strong>French election forecast</strong><span>Model documentation</span></div>
    <ol>{''.join(nav)}</ol>
    {facts}
  </nav>
  <main>{''.join(sections)}</main>
</div>
<script>window.MathJax = {{ tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
  svg: {{ fontCache: 'global' }} }};</script>
<script src="{MATHJAX}" defer></script>
<script>{JS}</script>"""
    if fragment:
        return page
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head><body>
{page}
<script src="{MERMAID}"></script>
<script>mermaid.initialize({{ startOnLoad: true, theme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'neutral' }});</script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DOCS / "site" / "index.html")
    ap.add_argument("--fragment", type=Path, default=None, help="also write the skeleton-less variant here")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(), encoding="utf-8")
    print(f"wrote {args.out}")
    if args.fragment:
        args.fragment.parent.mkdir(parents=True, exist_ok=True)
        args.fragment.write_text(build(fragment=True), encoding="utf-8")
        print(f"wrote {args.fragment}")


if __name__ == "__main__":
    main()

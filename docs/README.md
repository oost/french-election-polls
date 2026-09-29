# Documentation

The code in `forecast/` forecasts French presidential and legislative elections. It follows the method of the FiftyPlusOne 2026 US forecast; [`PLAN.md`](../PLAN.md) records the design decisions and every place this implementation departs from that method.

| Document | Read it to… |
| --- | --- |
| [Model overview](model-overview.md) | Follow the whole pipeline, from Wikipedia tables to "Le Pen wins 49%", and see what each stage assumes. |
| [File guide](file-guide.md) | Find the file that does something, and see which inputs you are expected to edit. |
| [Bayesian modelling and Stan: a tutorial](bayesian-tutorial.md) | Learn the ideas the model relies on (priors, shrinkage, hierarchical models, random walks, MCMC, Stan, calibration), using this project's own models as the examples. |

The tutorial assumes you know regression, distributions and maximum likelihood, but no Bayesian statistics. If that describes you, read the overview first, then the tutorial, and keep the file guide open while reading the code.

## Quick reference

```bash
python -m forecast ingest                 # rebuild the poll tables from the cached Wikipedia pages
python -m forecast backtest --cycles 2002 2007 2012 2017 2022   # ≈ 40 min
python -m forecast run --as-of 2026-09-28 # 2027 presidential forecast
python -m forecast legi-backtest          # legislative backtest
python -m forecast legi --as-of 2026-09-28
pytest -q                                 # fast tests;  pytest -m slow  also fits the Stan models
```

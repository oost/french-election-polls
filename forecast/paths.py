"""Filesystem layout for the forecast pipeline (see PLAN.md › Architecture)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "config"
MODEL_CONFIG = CONFIG_DIR / "model.yaml"

WIKI_CACHE = ROOT / "wiki_cache"

DATA = ROOT / "data"
REFERENCE = DATA / "reference"      # hand-curated, committed
RAW = DATA / "raw"                  # downloads (gitignored)
PROCESSED = DATA / "processed"      # derived tables (gitignored)
FIT_CACHE = DATA / "cache" / "fits"
ARCHIVE = DATA / "archive"
OUTPUTS = DATA / "outputs"

STAN_DIR = Path(__file__).resolve().parent / "stan"
STAN_BUILD = STAN_DIR / "build"


def ensure_dirs() -> None:
    for d in (RAW, PROCESSED, FIT_CACHE, ARCHIVE, OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)

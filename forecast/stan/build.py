"""Compile Stan models once (into forecast/stan/build/) and cache fits.

Fits are cached by a hash of (Stan code, data, sampler settings), so the
backtest reuses a test cycle's fit across its weekly snapshots.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from forecast import paths


def _require_cmdstanpy():
    try:
        import cmdstanpy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("cmdstanpy is not installed: pip install '.[stan]' then "
                           "python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'") from exc
    import cmdstanpy
    return cmdstanpy


def stan_available() -> bool:
    try:
        cmdstanpy = _require_cmdstanpy()
        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


def compile_model(name: str):
    """CmdStanModel for forecast/stan/{name}.stan, compiled into build/."""
    cmdstanpy = _require_cmdstanpy()
    src = paths.STAN_DIR / f"{name}.stan"
    paths.STAN_BUILD.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha1(src.read_bytes()).hexdigest()[:10]
    staged = paths.STAN_BUILD / f"{name}_{code_hash}.stan"
    if not staged.exists():
        shutil.copyfile(src, staged)
    return cmdstanpy.CmdStanModel(stan_file=str(staged))


def _jsonable(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def cached_sample(name: str, data: dict[str, Any], variables: list[str], **sample_kwargs) -> dict[str, np.ndarray]:
    """Run NUTS (or reuse a cached run) and return the requested draws.

    Draws are read with `stan_variable`, avoiding cmdstanpy's pandas helpers."""
    src = paths.STAN_DIR / f"{name}.stan"
    key = hashlib.sha1(
        src.read_bytes() + json.dumps(_jsonable(data), sort_keys=True).encode()
        + json.dumps(sample_kwargs, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    cache = paths.FIT_CACHE / f"{name}_{key}.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    model = compile_model(name)
    kwargs = {"chains": 4, "parallel_chains": 4, "iter_warmup": 1000, "iter_sampling": 1000,
              "show_progress": False, "seed": 1234, **sample_kwargs}
    fit = model.sample(data=_jsonable(data), **kwargs)
    draws = {v: np.asarray(fit.stan_variable(v)) for v in variables}
    summary = fit.diagnose() or ""
    draws["_diagnostics"] = np.array([summary])
    paths.FIT_CACHE.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(draws, f)
    return draws

"""Optuna TPE backend for parameter sweeps (plan item R3).

Drop-in alternative to the grid-neighborhood sweep used in
`run_local_fullmetric_sweep_oos.py`. The TPE sampler typically finds equivalent
peaks in 30-50% of the trials a grid would burn through.

This module is **opt-in**. The sweep matrix workflow keeps the grid backend as
its default; flipping to TPE is a single env-var change:

    SWEEP_BACKEND=optuna python run_local_fullmetric_sweep_oos.py ...

Persistent storage: each category's study lives in
`optuna_studies/<study_name>.db` (sqlite) and resumes across cycles, so the
sampler keeps learning from previous evals.

The objective function passed in must accept a `dict[gene_name -> float|int]`
and return a **fitness scalar to maximize**. The backend does NOT know about
GA payloads — it just samples genes and calls back.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SWEEP_BACKEND_ENV = "SWEEP_BACKEND"
BACKEND_GRID = "grid"
BACKEND_OPTUNA = "optuna"
DEFAULT_BACKEND = BACKEND_GRID
VALID_BACKENDS = (BACKEND_GRID, BACKEND_OPTUNA)

DEFAULT_STUDIES_DIR = _REPO_ROOT / "optuna_studies"


def get_active_backend(override: Optional[str] = None) -> str:
    """Resolve the active sweep backend (env var + override)."""
    raw = override if override is not None else os.environ.get(SWEEP_BACKEND_ENV, DEFAULT_BACKEND)
    if raw is None:
        return DEFAULT_BACKEND
    value = str(raw).strip().lower()
    return value if value in VALID_BACKENDS else DEFAULT_BACKEND


GeneSpec = tuple  # (name, lo, hi, step, is_int)


def build_search_space(
    focus_genes: Iterable[str],
    all_specs: Iterable[GeneSpec],
) -> list[GeneSpec]:
    """Filter the full GLOBAL_PARAM_SPECS list to only the focus genes."""
    focus_set = set(focus_genes)
    seen: dict[str, GeneSpec] = {}
    for spec in all_specs:
        name = spec[0]
        if name in focus_set:
            seen[name] = spec
    out = [seen[g] for g in focus_genes if g in seen]
    if len(out) != len(list(focus_genes)) and len(out) == 0:
        raise ValueError(f"None of the focus genes were found in specs: {list(focus_genes)}")
    return out


def _suggest_from_spec(trial, spec: GeneSpec) -> float:
    """Suggest a value for one gene using Optuna's discrete sampler.

    Discrete (step-based) sampling mirrors how the grid backend already moves
    along `step` units, so the two backends share the same per-gene resolution.
    """
    name, lo, hi, step, is_int = spec
    lo_f, hi_f, step_f = float(lo), float(hi), float(step)
    if is_int:
        return float(trial.suggest_int(name, int(round(lo_f)), int(round(hi_f)), step=max(1, int(round(step_f)))))
    return float(trial.suggest_float(name, lo_f, hi_f, step=step_f))


def _study_storage_uri(study_name: str, *, studies_dir: Optional[Path] = None) -> str:
    base = Path(studies_dir) if studies_dir is not None else DEFAULT_STUDIES_DIR
    base.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(base / f'{study_name}.db').as_posix()}"


def run_optuna_study(
    objective_fn: Callable[[dict[str, float]], float],
    focus_genes: Iterable[str],
    all_specs: Iterable[GeneSpec],
    *,
    n_trials: int,
    study_name: str = "sweep",
    studies_dir: Optional[Path] = None,
    seed: Optional[int] = None,
    direction: str = "maximize",
    storage: Optional[str] = None,
) -> dict[str, Any]:
    """Run a TPE study and return `{best_params, best_value, n_trials}`.

    `studies_dir` defaults to `<repo>/optuna_studies` so consecutive runs of the
    same `study_name` resume from the prior trials.
    """
    try:
        import optuna
    except ImportError as e:  # pragma: no cover - optuna is in requirements.txt
        raise RuntimeError("optuna is required for the Optuna sweep backend") from e

    focus_list = list(focus_genes)
    specs = build_search_space(focus_list, list(all_specs))
    if not specs:
        raise ValueError("no genes resolved from focus list")

    storage_uri = storage if storage is not None else _study_storage_uri(study_name, studies_dir=studies_dir)
    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else optuna.samplers.TPESampler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
    )

    def _wrapped_objective(trial) -> float:
        params: dict[str, float] = {}
        for spec in specs:
            params[spec[0]] = _suggest_from_spec(trial, spec)
        value = float(objective_fn(params))
        return value

    try:
        study.optimize(_wrapped_objective, n_trials=int(n_trials), gc_after_trial=False)
        result = {
            "best_params": dict(study.best_params),
            "best_value": float(study.best_value),
            "n_trials": int(len(study.trials)),
            "study_name": study_name,
            "storage": storage_uri,
        }
    finally:
        # Release the SQLite file handle so callers can clean up the directory
        # (especially on Windows where the lock survives Python references).
        try:
            engine = getattr(study._storage, "_backend", study._storage)
            engine = getattr(engine, "engine", None) or getattr(engine, "_engine", None)
            if engine is not None and hasattr(engine, "dispose"):
                engine.dispose()
        except Exception:
            pass
    return result


def should_use_optuna(override: Optional[str] = None) -> bool:
    return get_active_backend(override) == BACKEND_OPTUNA

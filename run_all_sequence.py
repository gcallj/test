#!/usr/bin/env python3
"""Run notebooks in the recommended order with conservative resource usage.

Usage:
  python run_all_sequence.py

Notes:
- Intended for Colab/local execution. Installs are not performed automatically.
- Uses papermill to execute notebooks sequentially in isolated runs.
- Sets conservative thread limits to avoid overloading Colab resources.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

NOTEBOOKS = [
    "Copy of STOCK_ETL_v2.ipynb",
    "bin_Stock_modelos_individuais.ipynb",
    "reg_Stock_modelos_individuais.ipynb",
    "bin_Ensemble_stock.ipynb",
    "Final_stock_output.ipynb",
]


def _set_conservative_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _require_papermill() -> None:
    try:
        import papermill  # noqa: F401
    except Exception:  # pragma: no cover - runtime check
        print(
            "papermill não está instalado. Rode: pip -q install papermill nbclient",
            file=sys.stderr,
        )
        sys.exit(1)


def run_notebook(path: Path, output_dir: Path) -> None:
    import papermill as pm

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / path.name

    print(f"\n[RUN] {path.name}")
    pm.execute_notebook(
        input_path=str(path),
        output_path=str(output_path),
        kernel_name="python3",
        progress_bar=True,
        request_save_on_cell_execute=False,
    )


def main() -> int:
    _set_conservative_threads()
    _require_papermill()

    repo_root = Path(__file__).resolve().parent
    output_dir = repo_root / "executed_notebooks"

    for nb in NOTEBOOKS:
        nb_path = repo_root / nb
        if not nb_path.exists():
            print(f"Notebook não encontrado: {nb_path}", file=sys.stderr)
            return 1

    for nb in NOTEBOOKS:
        run_notebook(repo_root / nb, output_dir)
        # Respiro entre execuções para reduzir picos de memória
        time.sleep(5)

    print("\n✅ Execução concluída. Outputs em:", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

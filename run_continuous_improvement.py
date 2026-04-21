#!/usr/bin/env python3
"""
Rotina de melhoria continua autonoma.

O que faz em UM ciclo:
  1. Le o estado em continuous_improvement_state.json para saber qual
     categoria de genes esta menos-recentemente-explorada
  2. Roda run_local_fullmetric_sweep.py nessa categoria
  3. Se houver candidato promovivel, chama apply_fullmetric_sweep_best.py
     (que ja faz backup do checkpoint anterior automaticamente)
  4. Anexa um registro estruturado a analysis/optimization_log.md
     (sucesso OU falha - falhas tambem ensinam o que nao funciona)
  5. Atualiza o state file com o novo timestamp da categoria

Estrategia de rotacao das 30 genes em 6 categorias semanticas (codex
ja explorou A/B/E em 2026-04; D e F continuam virgens):

  A. risk_management:  stop_atr_mult, stop_tighten_after_bars,
                       stop_tighten_factor, max_loss_per_trade_pct,
                       equity_drawdown_stop_pct
  B. take_profit:      reward_risk_ratio, partial_take_pct,
                       partial_take_level, partial_take_pct_2,
                       partial_take_level_2
  C. timing:           time_stop_bars, consecutive_loss_cooldown
  D. entry_filter:     vote_threshold_long, vote_threshold_short,
                       z_threshold, signal_ema_span,
                       entry_confirmation_days, entry_discount_atr_frac,
                       score_strength_scaling, min_signal_strength,
                       entry_score_threshold, score_percentile_trigger
  E. regime_vol:       ma_filter_period, ma_filter_mode, vol_regime_mode,
                       volatility_filter_percentile, regime_threshold,
                       volume_confirm_mode, momentum_confirm_days
  F. trailing:         trailing_stop_mode

Modo de uso:
  # Um ciclo (selecao automatica de categoria)
  python run_continuous_improvement.py

  # Categoria especifica
  python run_continuous_improvement.py --category D

  # Loop ate deadline (ex: rodar overnight)
  python run_continuous_improvement.py --until 06:00 --max-cycles 4

  # Apenas log status, nao roda nada
  python run_continuous_improvement.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "continuous_improvement_state.json"
LOG_FILE = ROOT / "analysis" / "optimization_log.md"
SWEEP_DIR = ROOT  # sweeps land at root with timestamped names

# Categorias de genes (rotacao semantica). Cada categoria e um conjunto
# de genes correlatos que faz sentido testar juntos em um sweep.
GENE_CATEGORIES = {
    "A_risk_management": [
        "stop_atr_mult",
        "stop_tighten_after_bars",
        "stop_tighten_factor",
        "max_loss_per_trade_pct",
        "equity_drawdown_stop_pct",
    ],
    "B_take_profit": [
        "reward_risk_ratio",
        "partial_take_pct",
        "partial_take_level",
        "partial_take_pct_2",
        "partial_take_level_2",
    ],
    "C_timing": [
        "time_stop_bars",
        "consecutive_loss_cooldown",
    ],
    "D_entry_filter": [
        "vote_threshold_long",
        "vote_threshold_short",
        "z_threshold",
        "entry_aggressiveness",  # v5: pivot-vs-pullback tradeoff
        "signal_ema_span",
        "entry_confirmation_days",
        "entry_discount_atr_frac",
        "score_strength_scaling",
        "min_signal_strength",
        "entry_score_threshold",
        "score_percentile_trigger",
    ],
    "E_regime_vol": [
        "ma_filter_period",
        "ma_filter_mode",
        "vol_regime_mode",
        "volatility_filter_percentile",
        "regime_threshold",
        "volume_confirm_mode",
        "momentum_confirm_days",
    ],
    "F_trailing": [
        "trailing_stop_mode",
    ],
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {
            "version": 1,
            "created_at": _now_iso(),
            "categories": {k: {"last_swept": None, "n_runs": 0, "n_promotions": 0}
                          for k in GENE_CATEGORIES},
            "total_cycles": 0,
            "total_promotions": 0,
        }
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def _save_state(state: dict) -> None:
    state["last_updated"] = _now_iso()
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _pick_next_category(state: dict, override: str | None = None) -> str:
    """Pick least-recently-swept category. Prefer never-swept categories."""
    if override:
        # Match by suffix or full name
        for cat in GENE_CATEGORIES:
            if cat == override or cat.endswith(f"_{override.lower()}") or cat.startswith(override.upper()):
                return cat
        # If no match, accept exact prefix letter (A/B/C/D/E/F)
        for cat in GENE_CATEGORIES:
            if cat.startswith(override.upper() + "_"):
                return cat
        raise SystemExit(f"Unknown category override: {override!r}. "
                        f"Valid: {list(GENE_CATEGORIES.keys())}")

    cats = state.get("categories", {})
    # Never-swept first
    never = [c for c, info in cats.items() if not info.get("last_swept")]
    if never:
        return sorted(never)[0]
    # Otherwise oldest last_swept
    return min(cats.keys(), key=lambda c: cats[c].get("last_swept") or "")


def _ensure_log_header() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        return
    header = (
        "# Optimization Log (continuous improvement)\n\n"
        "Append-only log of every full-metric sweep cycle executed by\n"
        "`run_continuous_improvement.py`. Each entry records:\n\n"
        "- timestamp, category, genes swept\n"
        "- baseline (`git:main`) and incumbent metrics snapshot\n"
        "- best candidate found (promoted or not)\n"
        "- promotion decision rationale (which guardrail passed/failed)\n"
        "- learning extracted from the result\n\n"
        "Failures are kept on purpose: they form the graveyard that\n"
        "tells future iterations which gene moves are dead ends.\n\n"
        "---\n\n"
    )
    LOG_FILE.write_text(header, encoding="utf-8")


def _format_pct(x: float, decimals: int = 2) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    return f"{float(x) * 100:.{decimals}f}%"


def _format_metric(x: float, decimals: int = 4) -> str:
    if x is None or not isinstance(x, (int, float)):
        return "n/a"
    return f"{float(x):.{decimals}f}"


def _summarize_metrics(m: dict) -> str:
    if not m:
        return "(no metrics)"
    return (
        f"fit={_format_metric(m.get('fit'))} "
        f"WR={_format_pct(m.get('wr_med_all'), 1)} "
        f"WR_tgt={_format_pct(m.get('wr_target_med_all'), 1)} "
        f"alpha={_format_pct(m.get('mean_alpha_ann'), 2)} "
        f"MDD={_format_pct(m.get('mdd_med'), 1)} "
        f"trades={int(m.get('trades_med', 0) or 0)}"
    )


def _learning_from_result(sweep: dict, applied: bool) -> str:
    """Extract a short learning sentence from sweep + apply outcome."""
    best = sweep.get("best_promoted") or sweep.get("best_guarded")
    if not best:
        return (
            "**Learning**: nenhum candidato cruzou os guardrails de WR ou de seguranca; "
            "o incumbent ja esta no Pareto local desse conjunto de genes. "
            "Considerar marcar essas genes como 'exploradas recentemente' por 30 dias."
        )
    moves = best.get("moves", [])
    if not moves:
        return "**Learning**: candidato sem moves (possivel parity); investigar."
    move_str = ", ".join(f"{m['gene']}: {m['from']}->{m['to']}" for m in moves)
    if applied:
        return (
            f"**Learning**: o move `{move_str}` foi promovido. "
            f"Categoria/genes adjacentes podem render melhorias semelhantes; "
            f"considerar sweep daquela categoria no proximo ciclo."
        )
    decision = best.get("decision", {})
    failed = []
    if not decision.get("acceptance_vs_main", {}).get("all_pass"):
        failed.append("acceptance_vs_main")
    if not decision.get("incumbent_wr_guardrail", {}).get("all_pass"):
        failed.append("incumbent_wr_guardrail")
    if not decision.get("incumbent_safety_guardrail", {}).get("all_pass"):
        failed.append("incumbent_safety_guardrail")
    if not decision.get("priority_better"):
        failed.append("priority_better")
    return (
        f"**Learning**: candidato `{move_str}` parecia bom mas falhou nos gates: "
        f"{', '.join(failed) or 'priority comparison'}. "
        f"Esses moves entram para o graveyard: nao tentar novamente sem mudar contexto."
    )


def _append_log_entry(entry: dict) -> None:
    _ensure_log_header()
    sweep = entry["sweep_result"]
    base = sweep.get("base_metrics", {})
    seed = sweep.get("seed_metrics", {})
    best = sweep.get("best_promoted") or sweep.get("best_guarded")

    md = []
    md.append(f"## {entry['timestamp']} - {entry['category']}\n")
    md.append(f"**Cycle**: #{entry['cycle_number']}  |  **Promoted**: {'YES' if entry['applied'] else 'NO'}")
    if "codex_synced" in entry:
        md.append(f"  |  **Codex sync**: {'YES' if entry['codex_synced'] else 'skipped'}")
    if "combo_budget" in entry:
        md.append(f"  |  **combo_budget**: {entry['combo_budget']}")
    if entry.get("n_recent_rejections", 0) >= 3:
        md.append(f"  |  **PLATEAU** ({entry['n_recent_rejections']} NO em sequencia)")
    md.append("\n")
    md.append(f"**Genes swept**: `{', '.join(entry['genes'])}`\n")
    md.append(f"**Sweep file**: `{entry['sweep_filename']}`\n\n")

    md.append("### Metrics\n\n")
    md.append("| Source | fit | WR | WR_tgt | alpha_ann | MDD | trades |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    md.append(f"| baseline (git:main) | {_format_metric(base.get('fit'))} | "
              f"{_format_pct(base.get('wr_med_all'), 2)} | "
              f"{_format_pct(base.get('wr_target_med_all'), 2)} | "
              f"{_format_pct(base.get('mean_alpha_ann'), 2)} | "
              f"{_format_pct(base.get('mdd_med'), 2)} | "
              f"{int(base.get('trades_med', 0) or 0)} |\n")
    md.append(f"| incumbent (worktree) | {_format_metric(seed.get('fit'))} | "
              f"{_format_pct(seed.get('wr_med_all'), 2)} | "
              f"{_format_pct(seed.get('wr_target_med_all'), 2)} | "
              f"{_format_pct(seed.get('mean_alpha_ann'), 2)} | "
              f"{_format_pct(seed.get('mdd_med'), 2)} | "
              f"{int(seed.get('trades_med', 0) or 0)} |\n")
    if best:
        bm = best.get("metrics", {})
        md.append(f"| **candidate** | {_format_metric(bm.get('fit'))} | "
                  f"{_format_pct(bm.get('wr_med_all'), 2)} | "
                  f"{_format_pct(bm.get('wr_target_med_all'), 2)} | "
                  f"{_format_pct(bm.get('mean_alpha_ann'), 2)} | "
                  f"{_format_pct(bm.get('mdd_med'), 2)} | "
                  f"{int(bm.get('trades_med', 0) or 0)} |\n")
        md.append(f"\n**Best candidate label**: `{best.get('label')}`\n\n")
        moves = best.get("moves", [])
        if moves:
            md.append("**Moves applied to incumbent**:\n\n")
            for m in moves:
                md.append(f"- `{m['gene']}`: `{m['from']}` -> `{m['to']}` "
                          f"(step {m.get('step', '?')})\n")
            md.append("\n")
    else:
        md.append("\n*(no candidate produced metrics — sweep returned empty)*\n\n")

    md.append(_learning_from_result(sweep, entry["applied"]) + "\n\n")
    md.append("---\n\n")

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write("".join(md))


def _recent_consecutive_rejections(n: int = 3) -> int:
    """Conta ciclos consecutivos (a partir do mais recente) sem promocao.

    Le analysis/optimization_log.md e conta entries com "**Promoted**: NO"
    antes de encontrar "**Promoted**: YES" (ou ate o limite `n`).
    Retorna 0 se log vazio / nao encontrado.
    """
    if not LOG_FILE.exists():
        return 0
    try:
        text = LOG_FILE.read_text(encoding="utf-8")
    except Exception:
        return 0
    # Split em entries (cada entry comeca com "## ")
    import re
    entries = re.split(r"(?m)^## \d{4}-\d{2}-\d{2}T", text)
    # entries[0] = header do arquivo; entries[1:] = ciclos do mais antigo ao mais novo
    rejections = 0
    for entry in reversed(entries[1:]):  # do mais recente pro antigo
        if "**Promoted**: YES" in entry:
            break
        if "**Promoted**: NO" in entry:
            rejections += 1
            if rejections >= n:
                break
    return rejections


def _pick_combo_budget(default: int = 8, plateau_budget: int = 16,
                       plateau_threshold: int = 3) -> tuple[int, int]:
    """Escolhe combo_budget baseado em plateau detection.

    Se houver >= plateau_threshold ciclos consecutivos sem promocao,
    retorna plateau_budget (explorar mais combos). Caso contrario,
    retorna default.

    Returns (combo_budget, n_recent_rejections).
    """
    n_rej = _recent_consecutive_rejections(plateau_threshold)
    if n_rej >= plateau_threshold:
        return plateau_budget, n_rej
    return default, n_rej


def _run_sweep(category: str, alpha_focus: str = "off",
               combo_budget: int = 8) -> tuple[Path, dict]:
    """Run run_local_fullmetric_sweep.py for the given category.

    combo_budget: largura da busca (numero de combos multi-gene).
    Default 8 e rapido. Em plateau (3+ ciclos NO em sequencia), o chamador
    pode aumentar para 16-24 para explorar combos mais amplos.

    Returns (sweep_file_path, parsed_sweep_dict).
    """
    genes = GENE_CATEGORIES[category]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"local_fullmetric_sweep_continuous_{category}_{timestamp}.json"
    out_path = ROOT / out_name

    cmd = [
        sys.executable,
        "-u",
        "run_local_fullmetric_sweep.py",
        "--alpha-focus",
        alpha_focus,
        "--genes",
        ",".join(genes),
        "--combo-budget",
        str(int(combo_budget)),
        "--out",
        str(out_path),
    ]
    print(f"[CONTINUOUS] Running sweep on category {category} ({len(genes)} genes)")
    print(f"[CONTINUOUS] Cmd: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    if rc != 0:
        raise SystemExit(f"Sweep failed with exit code {rc}")
    if not out_path.exists():
        raise SystemExit(f"Sweep finished but output file is missing: {out_path}")
    return out_path, json.loads(out_path.read_text(encoding="utf-8"))


def _try_apply(sweep_path: Path, sweep: dict) -> bool:
    """If sweep has best_promoted, run apply_fullmetric_sweep_best.py.

    Returns True if applied, False if no promotable candidate or apply failed.
    """
    if not sweep.get("best_promoted"):
        print("[CONTINUOUS] No promotable candidate; nothing to apply.")
        return False
    cmd = [
        sys.executable,
        "-u",
        "apply_fullmetric_sweep_best.py",
        "--sweep",
        str(sweep_path),
    ]
    print(f"[CONTINUOUS] Applying promoted candidate: {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    if rc != 0:
        print(f"[CONTINUOUS] Apply failed (rc={rc}); checkpoint untouched.")
        return False
    print("[CONTINUOUS] Checkpoint promoted successfully.")
    return True


def _run_codex_sync() -> bool:
    """Call analysis/codex_attempts_sync.py best-effort.

    Returns True if the sync script ran (regardless of whether it wrote
    anything), False if it was skipped or crashed.

    The sync script itself is defensive: it exits 0 when the codex source
    is missing (remote runner scenario), so this wrapper only needs to
    shield us from transient errors and keep the cycle running.
    """
    script = ROOT / "analysis" / "codex_attempts_sync.py"
    if not script.exists():
        print("[CONTINUOUS] codex sync script missing — skipping")
        return False
    try:
        rc = subprocess.run(
            [sys.executable, "-u", str(script)],
            cwd=str(ROOT),
            check=False,
            timeout=60,
        ).returncode
        if rc != 0:
            print(f"[CONTINUOUS] codex sync exited rc={rc} (non-fatal)")
        return True
    except subprocess.TimeoutExpired:
        print("[CONTINUOUS] codex sync timed out after 60s — skipping")
        return False
    except Exception as e:
        print(f"[CONTINUOUS] codex sync skipped: {e!r}")
        return False


def _do_one_cycle(category_override: str | None, alpha_focus: str,
                  run_codex_sync: bool = True,
                  combo_budget: int | None = None) -> dict:
    state = _load_state()
    state["total_cycles"] += 1
    cycle_num = state["total_cycles"]
    category = _pick_next_category(state, override=category_override)
    print(f"\n=== CONTINUOUS CYCLE #{cycle_num} - category {category} ===")

    codex_synced = False
    if run_codex_sync:
        codex_synced = _run_codex_sync()

    # Adaptive combo_budget: em plateau (3+ NO em sequencia) aumenta para 16.
    # Override explicito via CLI tem prioridade.
    if combo_budget is None:
        combo_budget, n_recent_rej = _pick_combo_budget(default=8, plateau_budget=16)
        if n_recent_rej >= 3:
            print(f"[CONTINUOUS] PLATEAU detected ({n_recent_rej} consecutive NO) "
                  f"-> combo_budget elevado para {combo_budget}")
    else:
        n_recent_rej = _recent_consecutive_rejections(3)

    sweep_path, sweep = _run_sweep(category, alpha_focus, combo_budget=combo_budget)
    applied = _try_apply(sweep_path, sweep)

    # Update state for this category
    cat_info = state["categories"].setdefault(
        category, {"last_swept": None, "n_runs": 0, "n_promotions": 0}
    )
    cat_info["last_swept"] = _now_iso()
    cat_info["n_runs"] += 1
    if applied:
        cat_info["n_promotions"] += 1
        state["total_promotions"] += 1

    entry = {
        "timestamp": _now_iso(),
        "cycle_number": cycle_num,
        "category": category,
        "genes": GENE_CATEGORIES[category],
        "sweep_filename": sweep_path.name,
        "sweep_result": sweep,
        "applied": applied,
        "codex_synced": codex_synced,
        "combo_budget": combo_budget,
        "n_recent_rejections": n_recent_rej,
    }
    _append_log_entry(entry)
    _save_state(state)
    return entry


def _print_status() -> None:
    state = _load_state()
    print("=== CONTINUOUS IMPROVEMENT STATUS ===")
    print(f"Created: {state.get('created_at')}")
    print(f"Last updated: {state.get('last_updated', 'never')}")
    print(f"Total cycles: {state.get('total_cycles', 0)}")
    print(f"Total promotions: {state.get('total_promotions', 0)}")
    print()
    print(f"{'Category':<22} {'Last swept':<28} {'Runs':>6} {'Promo':>6}")
    print("-" * 65)
    for cat, info in sorted(state.get("categories", {}).items()):
        print(
            f"{cat:<22} {str(info.get('last_swept') or 'never'):<28} "
            f"{info.get('n_runs', 0):>6} {info.get('n_promotions', 0):>6}"
        )


def _parse_until(s: str) -> datetime:
    """Parse 'HH:MM' as next occurrence of that time (today or tomorrow)."""
    now = datetime.now().astimezone()
    try:
        h, m = s.split(":")
        target = now.replace(hour=int(h), minute=int(m), second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target
    except Exception:
        return datetime.fromisoformat(s).astimezone()


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous improvement runner with rotation + auto-log.")
    parser.add_argument("--category", type=str, default=None,
                        help="Override category. Pass A/B/C/D/E/F or full name like 'D_entry_filter'.")
    parser.add_argument("--alpha-focus", choices=("on", "off"), default="off")
    parser.add_argument("--until", type=str, default=None,
                        help="Run cycles until this local time (HH:MM or ISO datetime).")
    parser.add_argument("--max-cycles", type=int, default=1,
                        help="Maximum cycles to run (default 1; ignored if --until set).")
    parser.add_argument("--status", action="store_true",
                        help="Print status table and exit.")
    parser.add_argument("--no-codex-sync", action="store_true",
                        help="Skip calling analysis/codex_attempts_sync.py at "
                             "the start of each cycle (useful offline / in CI).")
    parser.add_argument("--combo-budget", type=int, default=None,
                        help="Override combo_budget do sweep (default 8; adaptive "
                             "para 16 em plateau de 3+ ciclos NO consecutivos). "
                             "Valor explicito desliga a adaptacao.")
    args = parser.parse_args()

    if args.status:
        _print_status()
        return 0

    deadline = _parse_until(args.until) if args.until else None
    cycles_done = 0
    cycles_promoted = 0

    while True:
        if deadline is None and cycles_done >= args.max_cycles:
            break
        if deadline is not None and datetime.now().astimezone() >= deadline:
            break

        try:
            entry = _do_one_cycle(
                args.category,
                args.alpha_focus,
                run_codex_sync=not args.no_codex_sync,
                combo_budget=args.combo_budget,
            )
            cycles_done += 1
            if entry["applied"]:
                cycles_promoted += 1
        except SystemExit as e:
            print(f"[CONTINUOUS] Cycle aborted: {e}")
            return int(getattr(e, "code", 1) or 1)
        except Exception as e:
            print(f"[CONTINUOUS] Unexpected error: {e!r}")
            raise

        # Brief pause before next cycle (lets disk flush etc.)
        time.sleep(2)

    print(f"\n[CONTINUOUS] Done: {cycles_done} cycle(s), {cycles_promoted} promotion(s).")
    print(f"[CONTINUOUS] Log: {LOG_FILE}")
    print(f"[CONTINUOUS] State: {STATE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Gene auto-fix — remediacao automatica rule-based para genes quarentenados.

Complementa gene_effect_diagnostic.py:
  diagnostic: DETECTA (zero variance detected)
  autofix:    CORRIGE (aplica remediacao rule-based quando clara)

Fixes autonomos suportados (cada um so dispara com alta confianca):

  [FIX-1] Rollback gene range: Se um gene teve seu range AUMENTADO em commit
          recente e ficou quarantenado depois, restaura o range anterior.
          Detectado via `git log -p` em GLOBAL_PARAM_SPECS.

  [FIX-2] Normalize gene default: Se default corrente nao esta em _LEGACY_GENE_DEFAULTS,
          adiciona com valor neutro (spec[1] ou 1.0 se gene tem nome *_factor).

  [FIX-3] Remove gene from sweep category: Se quarantena persistiu por 7 dias,
          remove gene de GENE_CATEGORIES (em run_continuous_improvement.py).
          Gene fica visivel no genome mas nao e mais iterado.

  [FIX-4] Abrir GitHub issue: Sempre que quarentena for aplicada pela primeira
          vez, cria issue com detalhes + link pro report. Humano decide fix
          de implementacao complexo.

Nao faz:
  - Modificar a logica do gate em si (complexo, requer analise semantica)
  - Mudar thresholds arbitrarios (sem ground truth pra calibracao)

Uso:
  python analysis/gene_autofix.py                    # aplica todos os fixes
  python analysis/gene_autofix.py --dry-run          # so reporta, nao muda
  python analysis/gene_autofix.py --only fix1,fix3   # subset
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

QUARANTINE_DAYS_BEFORE_REMOVE = 7
CONTINUOUS_SCRIPT = ROOT / "run_continuous_improvement.py"


def _load_quarantine() -> Dict:
    path = ROOT / "analysis" / "gene_quarantine.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_quarantine(data: Dict) -> None:
    path = ROOT / "analysis" / "gene_quarantine.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _days_since(iso: str) -> float:
    try:
        ts = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    return (datetime.now(timezone.utc) - ts).total_seconds() / 86400.0


# ============================================================================
# FIX-3: Remove gene de GENE_CATEGORIES apos N dias quarantenado
# ============================================================================
def fix_remove_from_sweep_rotation(quar: Dict, dry_run: bool = False) -> List[str]:
    """Remove gene quarentenado de GENE_CATEGORIES apos N dias."""
    if not CONTINUOUS_SCRIPT.exists():
        return []
    content = CONTINUOUS_SCRIPT.read_text(encoding="utf-8")
    original = content
    actions = []
    for gene, info in list(quar.items()):
        if not info.get("quarantined"):
            continue
        first = info.get("quarantined_at") or info.get("first_flagged", _now_iso())
        if _days_since(first) < QUARANTINE_DAYS_BEFORE_REMOVE:
            continue
        if info.get("removed_from_sweep"):
            continue
        # Remove o gene de qualquer categoria em GENE_CATEGORIES
        # Pattern: `"gene_name",` possibly seguido de comentario.
        pattern = re.compile(
            r'^(\s+)"' + re.escape(gene) + r'",[^\n]*\n',
            re.MULTILINE
        )
        new = pattern.sub('', content)
        if new != content:
            actions.append(f"[FIX-3] Removed '{gene}' from GENE_CATEGORIES (quarantined {_days_since(first):.1f}d)")
            content = new
            info["removed_from_sweep"] = True
            info["removed_at"] = _now_iso()
    if actions and not dry_run:
        CONTINUOUS_SCRIPT.write_text(content, encoding="utf-8")
        _save_quarantine(quar)
    return actions


# ============================================================================
# FIX-4: Abrir GitHub issue ao quarantenar pela primeira vez
# ============================================================================
def fix_open_github_issue(quar: Dict, dry_run: bool = False) -> List[str]:
    """Abre GitHub issue para cada gene recem-quarantenado (sem issue aberta ainda)."""
    actions = []
    for gene, info in list(quar.items()):
        if not info.get("quarantined"):
            continue
        if info.get("issue_opened"):
            continue
        title = f"[auto] Gene '{gene}' zero variance detectado"
        details = info.get("last_details", {})
        body = (
            f"## Auto-diagnostico: gene sem efeito\n\n"
            f"O detector automatico (`analysis/gene_effect_diagnostic.py`) "
            f"detectou que o gene `{gene}` produz fit/WR/alpha IDENTICOS "
            f"em {details.get('n_distinct', '?')} valores distintos testados.\n\n"
            f"### Valores testados\n"
            f"```\n{details.get('values_tested', [])}\n```\n\n"
            f"### Metricas de variance\n"
            f"- fit_std: {details.get('fit_std', '?')}\n"
            f"- Flagged {info.get('flagged_count', 0)}x em {info.get('last_flagged')}\n\n"
            f"### Acao automatica tomada\n"
            f"- Gene adicionado a `analysis/gene_quarantine.json`\n"
            f"- Excluido dos sweeps via `run_continuous_improvement._load_quarantine()`\n"
            f"- Apos {QUARANTINE_DAYS_BEFORE_REMOVE} dias quarantenado, sera removido "
            f"de `GENE_CATEGORIES` automaticamente (FIX-3)\n\n"
            f"### Acao humana recomendada\n"
            f"Revisar implementacao do gate/logica em `ga_run.py` e `ga_run_modular_final.py`:\n"
            f"- Threshold esta na magnitude correta? (percentual vs ATR vs absoluto)\n"
            f"- Gate e chamado no path certo? (vectorized_signals E backtest_stats_global_intraday)\n"
            f"- Range do spec e adequado? (muito amplo pode deixar todo valor util clipado)\n\n"
            f"Ver `analysis/gene_effect_report.json` para dados completos.\n\n"
            f"---\n"
            f"_Issue criado automaticamente pelo gene_autofix.py_"
        )
        if dry_run:
            actions.append(f"[FIX-4] Would open issue: {title}")
            continue
        try:
            cmd = ["gh", "issue", "create",
                   "--title", title,
                   "--body", body,
                   "--label", "auto-diagnosis"]
            result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                                    text=True, check=False, timeout=30)
            if result.returncode == 0:
                info["issue_opened"] = True
                info["issue_url"] = result.stdout.strip()
                info["issue_opened_at"] = _now_iso()
                actions.append(f"[FIX-4] Opened issue for '{gene}': {result.stdout.strip()}")
            else:
                actions.append(f"[FIX-4] gh issue create failed para '{gene}': {result.stderr.strip()[:200]}")
        except Exception as exc:
            actions.append(f"[FIX-4] Exception creating issue for '{gene}': {exc}")
    if actions and not dry_run:
        _save_quarantine(quar)
    return actions


# ============================================================================
# Orchestrator
# ============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="so reporta, nao aplica fixes")
    ap.add_argument("--only", type=str, default="all",
                    help="subset de fixes (fix3,fix4) ou 'all'")
    args = ap.parse_args()

    quar = _load_quarantine()
    if not quar:
        print("[AUTOFIX] Quarantine vazio — nothing to do")
        return 0

    selected = {f.strip() for f in args.only.split(",")} if args.only != "all" else {"fix3", "fix4"}

    total_actions = []
    if "fix3" in selected or "all" in selected:
        total_actions.extend(fix_remove_from_sweep_rotation(quar, dry_run=args.dry_run))
    if "fix4" in selected or "all" in selected:
        total_actions.extend(fix_open_github_issue(quar, dry_run=args.dry_run))

    print(f"[AUTOFIX] {'DRY RUN' if args.dry_run else 'APPLIED'}: {len(total_actions)} actions")
    for a in total_actions:
        print(f"  {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

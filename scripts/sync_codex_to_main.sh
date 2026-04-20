#!/usr/bin/env bash
# Sync codex worktree artifacts -> main repo working copy.
#
# Copia:
#   - analysis/optimization_attempts_20260416.md -> analysis/codex_attempts_source.md (main)
#   - local_fullmetric_sweep_result_*.json novos -> main (root)
#   - analysis/codex_attempts_history.json (regenerado via sync script)
#   - analysis/sweep_learnings_summary.md (regenerado)
#
# Fluxo:
#   1. Detecta path do worktree codex
#   2. Se markdown diferiu (hash diff), copia
#   3. Copia sweep JSONs nao-presentes ainda em main
#   4. Roda analysis/codex_attempts_sync.py + sweep_learnings.py (regenera JSONs agregados)
#   5. Commita em main com mensagem "[codex-sync] ..."
#   6. Push opcional (GA_SYNC_PUSH=1 para ativar)
#
# Idempotente: se nada mudou, exit 0 sem commit. Sem push se nada novo.
#
# Uso:
#   bash scripts/sync_codex_to_main.sh
#   GA_SYNC_PUSH=1 bash scripts/sync_codex_to_main.sh
#
# Chamado automaticamente via .git/hooks/post-commit quando codex worktree
# faz commit (veja scripts/install_codex_sync_hook.sh).
set -euo pipefail

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
CODEX_WORKTREE="${CODEX_WORKTREE:-C:/Users/gabri/.codex/worktrees/0b30/test}"
MAIN_REPO="${MAIN_REPO:-C:/Users/gabri/OneDrive/Documentos/GitHub/test}"
CODEX_MD_REL="analysis/optimization_attempts_20260416.md"
MAIN_MD_REL="analysis/codex_attempts_source.md"

# ----------------------------------------------------------------------------
# Safety checks
# ----------------------------------------------------------------------------
if [ ! -d "$CODEX_WORKTREE" ]; then
  echo "[codex-sync] Codex worktree nao encontrado em: $CODEX_WORKTREE"
  echo "[codex-sync] Set CODEX_WORKTREE=<path> se esta em outra maquina."
  exit 0
fi

if [ ! -d "$MAIN_REPO" ]; then
  echo "[codex-sync] Main repo nao encontrado em: $MAIN_REPO"
  exit 0
fi

CODEX_MD="$CODEX_WORKTREE/$CODEX_MD_REL"
MAIN_MD="$MAIN_REPO/$MAIN_MD_REL"

if [ ! -f "$CODEX_MD" ]; then
  echo "[codex-sync] Markdown do codex nao existe em $CODEX_MD — skip."
  exit 0
fi

# ----------------------------------------------------------------------------
# Step 1: markdown diff
# ----------------------------------------------------------------------------
changed_md=0
if [ ! -f "$MAIN_MD" ] || ! cmp -s "$CODEX_MD" "$MAIN_MD"; then
  changed_md=1
  echo "[codex-sync] markdown difere, copiando: $CODEX_MD -> $MAIN_MD"
  cp "$CODEX_MD" "$MAIN_MD"
else
  echo "[codex-sync] markdown identico, skip"
fi

# ----------------------------------------------------------------------------
# Step 2: sweep JSONs novos do codex
# ----------------------------------------------------------------------------
changed_json=0
copied_count=0
while IFS= read -r -d '' f; do
  base=$(basename "$f")
  dest="$MAIN_REPO/$base"
  if [ ! -f "$dest" ] || ! cmp -s "$f" "$dest"; then
    cp "$f" "$dest"
    changed_json=1
    copied_count=$((copied_count + 1))
  fi
done < <(find "$CODEX_WORKTREE" -maxdepth 1 -name "local_fullmetric_sweep_result_*.json" -print0 2>/dev/null)

if [ "$copied_count" -gt 0 ]; then
  echo "[codex-sync] copiados $copied_count sweep JSONs novos do codex"
fi

if [ "$changed_md" -eq 0 ] && [ "$changed_json" -eq 0 ]; then
  echo "[codex-sync] nada novo — exit 0"
  exit 0
fi

# ----------------------------------------------------------------------------
# Step 3: regenerar aggregator JSONs em main
# ----------------------------------------------------------------------------
cd "$MAIN_REPO"
echo "[codex-sync] regenerando aggregator..."

python_bin="python"
command -v python3 >/dev/null 2>&1 && python_bin="python3"

"$python_bin" analysis/codex_attempts_sync.py --force --quiet 2>&1 | tail -3 || {
  echo "[codex-sync] WARN: codex_attempts_sync.py falhou (nao-fatal)"
}
"$python_bin" analysis/sweep_learnings.py 2>&1 | tail -3 || {
  echo "[codex-sync] WARN: sweep_learnings.py falhou (nao-fatal)"
}

# ----------------------------------------------------------------------------
# Step 4: commit em main
# ----------------------------------------------------------------------------
cd "$MAIN_REPO"
git add \
  "$MAIN_MD_REL" \
  analysis/codex_attempts_history.json \
  analysis/codex_attempts_imported.md \
  analysis/sweep_learnings_summary.md \
  local_fullmetric_sweep_result_*.json 2>/dev/null || true

if git diff --cached --quiet; then
  echo "[codex-sync] nenhuma mudanca staged — exit 0"
  exit 0
fi

# extrair info do codex para msg de commit
codex_latest=""
if [ -f "$MAIN_MD" ]; then
  codex_latest=$(grep -E "^(## |### )Attempt [0-9]+" "$MAIN_MD" | tail -1 | sed 's/^#* //' | cut -c1-60)
fi

msg="[codex-sync] importado do codex worktree"
if [ -n "$codex_latest" ]; then
  msg="$msg: $codex_latest"
fi

git commit -m "$msg" \
  -m "Auto-sync via scripts/sync_codex_to_main.sh." \
  -m "Changes: md=$changed_md jsons_new=$copied_count"

echo "[codex-sync] commit criado: $(git log -1 --format=%h)"

# ----------------------------------------------------------------------------
# Step 5: push opcional
# ----------------------------------------------------------------------------
if [ "${GA_SYNC_PUSH:-0}" = "1" ]; then
  echo "[codex-sync] GA_SYNC_PUSH=1 — push para origin/main..."
  git pull --rebase origin main 2>&1 | tail -2
  git push origin main 2>&1 | tail -2
else
  echo "[codex-sync] commit local criado; set GA_SYNC_PUSH=1 para push automatico."
fi

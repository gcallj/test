#!/usr/bin/env python3
"""
Parse the codex `optimization_attempts_*.md` log into structured JSON so
the continuous-improvement aggregator can learn from it.

Design goals:
  - Defensive: if the source file is unreachable (remote runner, offline)
    exit 0 with a skip message. Never break the rotina.
  - Idempotent: running twice on unchanged source is a no-op.
  - Minimal: only extract the fields the aggregator needs (genes swept,
    promotion status, move list, sweep file name) plus enough context
    for humans (title, timestamp, learning sentence).

Output artifacts:
  - analysis/codex_attempts_history.json : structured data for aggregator
  - analysis/codex_attempts_imported.md  : raw copy with SHA1 header for traceability

Usage:
  python analysis/codex_attempts_sync.py                       # default source path
  python analysis/codex_attempts_sync.py --source PATH         # override source
  python analysis/codex_attempts_sync.py --force               # rewrite even if SHA1 unchanged
  python analysis/codex_attempts_sync.py --quiet               # only print errors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
HISTORY_JSON = ANALYSIS / "codex_attempts_history.json"
RAW_COPY = ANALYSIS / "codex_attempts_imported.md"

# In-repo source (committed copy of codex markdown). Preferida quando roda
# em runner remoto (GitHub Actions, trigger remoto Anthropic) onde o path
# local do codex worktree nao existe.
REPO_SOURCE = ANALYSIS / "codex_attempts_source.md"

# Local codex worktree path (so existe em maquinas de devs com codex
# instalado). Mantido como fallback se REPO_SOURCE estiver ausente.
LOCAL_SOURCE = Path(
    "C:/Users/gabri/.codex/worktrees/0b30/test/analysis/optimization_attempts_20260416.md"
)

# Default: tenta REPO_SOURCE, depois LOCAL_SOURCE. `--source` no CLI sobrescreve.
DEFAULT_SOURCE = REPO_SOURCE if REPO_SOURCE.exists() else LOCAL_SOURCE

PARSER_VERSION = 1

# ---------- Regex (compiled once) ----------

# Matches "## Attempt 6: Title" OR "### Attempt 6: Title" (any heading level >= 2)
ATTEMPT_HEADER_RE = re.compile(
    r"^(?P<hashes>#{2,4})\s+Attempt\s+(?P<id>\d+):\s*(?P<title>.+?)\s*$",
    re.MULTILINE,
)

TIMESTAMP_RE = re.compile(r"Timestamp:\s*`([^`]+)`")

# Command with --genes "gene1,gene2,..." — may span one line with args after
GENES_CMD_RE = re.compile(r'--genes\s+"([^"]+)"')

# Output file name: --out FILE.json (optional quotes)
OUT_FILE_RE = re.compile(r'--out\s+"?([^\s"]+\.json)"?')

# Default sweep output when --out is omitted
DEFAULT_SWEEP_FILE = "local_fullmetric_sweep_result.json"

# Single-change line: "- change: `gene: from -> to` ..."
SINGLE_CHANGE_RE = re.compile(
    r"^\s*-\s*change:\s*`([^:`]+):\s*([^`>]+?)\s*->\s*([^`]+?)\s*`",
    re.MULTILINE,
)

# Multi-change row under "- changes:" header: "  - `gene: from -> to`"
MULTI_CHANGE_ROW_RE = re.compile(
    r"^\s*-\s*`([^:`]+):\s*([^`>]+?)\s*->\s*([^`]+?)\s*`\s*$",
    re.MULTILINE,
)

# "- `promote = true`" or "- `promote = false`"
PROMOTE_RE = re.compile(r"`promote\s*=\s*(true|false)`", re.IGNORECASE)

# "What was learned:" or "What was learned" — paragraph after it
LEARNED_HEADER_RE = re.compile(r"^What was learned:?\s*$", re.MULTILINE | re.IGNORECASE)

# Known metric label -> canonical key (matches keys used by existing sweep JSON)
METRIC_LABEL_MAP = {
    "fit": "fit",
    "wr_med_all": "wr_med_all",
    "wr_target_med_all": "wr_target_med_all",
    "mean_alpha_ann": "mean_alpha_ann",
    "alpha_ann_pos_rate": "alpha_ann_pos_rate",
    "mdd_med": "mdd_med",
    "mdd_p75": "mdd_p75",
    "mdd_duration_med": "mdd_duration_med",
    "trades_med": "trades_med",
    "n_ge_70_target": "n_ge_70_target",
}

# ---------- Helpers ----------


def _sha1(content: bytes) -> str:
    return hashlib.sha1(content).hexdigest()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def _coerce_numeric(cell: str):
    """Convert a table cell to a number.

    - "33.33%" -> 0.3333
    - "-31.7891" -> -31.7891
    - "+12.5" -> 12.5
    - "147" -> 147.0
    - "n/a" / "" -> None
    """
    if cell is None:
        return None
    s = cell.strip().strip("`").replace(",", "")
    if not s or s.lower() in ("n/a", "na", "none", "-"):
        return None
    # Strip leading +
    if s.startswith("+"):
        s = s[1:]
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    try:
        val = float(s)
        if pct:
            val = val / 100.0
        return val
    except ValueError:
        return None


def _parse_metrics_table(section_text: str) -> tuple[dict, dict]:
    """Extract incumbent + candidate metrics from the first data table in section.

    Returns (incumbent_dict, candidate_dict). Empty dict if no table found.

    Handles these shapes:
      | Metric | `main` baseline | Prior incumbent | New incumbent | Delta ... |
      | Metric | Incumbent seed  | Best candidate  | Delta |
      | Metric | Incumbent       | Candidate       | Delta vs incumbent | Delta vs `main` |
    """
    lines = section_text.splitlines()
    # Find first header row containing "Metric"
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "Metric" in line and line.count("|") >= 3:
            header_idx = i
            break
    if header_idx is None:
        return {}, {}

    def _split_row(row: str) -> list[str]:
        # Strip trailing/leading '|' and split
        parts = [c.strip() for c in row.strip().strip("|").split("|")]
        return parts

    header_cells = _split_row(lines[header_idx])
    headers_lc = [h.strip().strip("`").lower() for h in header_cells]

    # Find incumbent + candidate column indices
    inc_col, cand_col = None, None
    for i, h in enumerate(headers_lc):
        if "delta" in h or "Î”" in h or "vs" in h or "Δ" in h:
            continue
        if "new incumbent" in h or "candidate" in h:
            cand_col = i
        elif "prior incumbent" in h or "incumbent seed" in h:
            inc_col = i
        elif "incumbent" in h and inc_col is None:
            # Plain "incumbent" column (no "prior" or "new" prefix)
            inc_col = i
    # Fallback: if candidate missing but incumbent found, assume next col is candidate
    if cand_col is None and inc_col is not None and inc_col + 1 < len(headers_lc):
        next_h = headers_lc[inc_col + 1]
        if "delta" not in next_h and "vs" not in next_h:
            cand_col = inc_col + 1

    if inc_col is None and cand_col is None:
        return {}, {}

    # Parse data rows (skip header + separator)
    incumbent: dict = {}
    candidate: dict = {}
    for line in lines[header_idx + 2:]:
        if not line.strip().startswith("|"):
            break
        cells = _split_row(line)
        if len(cells) != len(header_cells):
            continue
        label_raw = cells[0].strip().strip("`").lower()
        # Normalize: some labels have spaces like "mdd med" — keep only a-z_
        label = re.sub(r"[^a-z0-9_]", "_", label_raw).strip("_")
        key = METRIC_LABEL_MAP.get(label)
        if not key:
            continue
        if inc_col is not None and inc_col < len(cells):
            v = _coerce_numeric(cells[inc_col])
            if v is not None:
                incumbent[key] = v
        if cand_col is not None and cand_col < len(cells):
            v = _coerce_numeric(cells[cand_col])
            if v is not None:
                candidate[key] = v
    return incumbent, candidate


def _parse_moves(section_text: str) -> list[dict]:
    """Extract gene moves from 'Promoted candidate:' / 'change:' / 'changes:' sections."""
    moves: list[dict] = []
    seen = set()

    # Try single "- change: `gene: a -> b`"
    for m in SINGLE_CHANGE_RE.finditer(section_text):
        gene, frm, to = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        key = (gene, frm, to)
        if key not in seen:
            seen.add(key)
            moves.append({
                "gene": gene,
                "from": _coerce_numeric(frm) if _coerce_numeric(frm) is not None else frm,
                "to": _coerce_numeric(to) if _coerce_numeric(to) is not None else to,
            })

    # Try multi-change rows (nested under "- changes:")
    # We scan lines looking for a line that contains "changes:" (with colon),
    # then collect subsequent indented `- \`gene: a -> b\`` rows until blank/unindented.
    lines = section_text.splitlines()
    in_changes = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"-\s*changes:\s*$", stripped, re.IGNORECASE):
            in_changes = True
            continue
        if in_changes:
            if not stripped:
                in_changes = False
                continue
            m = re.match(r"-\s*`([^:`]+):\s*([^`>]+?)\s*->\s*([^`]+?)\s*`", stripped)
            if m:
                gene, frm, to = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                key = (gene, frm, to)
                if key not in seen:
                    seen.add(key)
                    moves.append({
                        "gene": gene,
                        "from": _coerce_numeric(frm) if _coerce_numeric(frm) is not None else frm,
                        "to": _coerce_numeric(to) if _coerce_numeric(to) is not None else to,
                    })
            else:
                # No longer a move row — stop parsing this block
                in_changes = False
    return moves


def _parse_learning(section_text: str) -> str:
    """Extract the 'What was learned' paragraph as a single string."""
    m = LEARNED_HEADER_RE.search(section_text)
    if not m:
        return ""
    tail = section_text[m.end():]
    out_lines: list[str] = []
    for raw in tail.splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):  # next section header
            break
        if stripped.startswith("Daily workbook") or stripped.startswith("Daily delivery"):
            break
        if stripped.lower().startswith("outcome:"):
            break
        if stripped.lower().startswith("decision:"):
            break
        if stripped:
            out_lines.append(stripped.lstrip("-").strip())
    # Join with spaces, trim
    text = " ".join(out_lines).strip()
    # Cap to ~500 chars to keep JSON compact
    return text[:500]


def _parse_one_attempt(attempt_id: str, title: str, body: str) -> dict:
    timestamp_match = TIMESTAMP_RE.search(body)
    timestamp = timestamp_match.group(1) if timestamp_match else ""

    genes_match = GENES_CMD_RE.search(body)
    genes: list[str] = []
    if genes_match:
        genes = [g.strip() for g in genes_match.group(1).split(",") if g.strip()]

    out_match = OUT_FILE_RE.search(body)
    sweep_file = out_match.group(1).strip() if out_match else ""
    # If command is a full-metric sweep without --out, use default filename
    if not sweep_file and "run_local_fullmetric_sweep" in body:
        sweep_file = DEFAULT_SWEEP_FILE

    moves = _parse_moves(body)

    # Derive promoted status. Priority:
    #   1. body `promote = true` / `promote = false` (most explicit, always correct)
    #   2. title "NOT PROMOTED" (overrides generic PROMOTED substring)
    #   3. title "PROMOTED"
    promote_match = PROMOTE_RE.search(body)
    title_upper = title.upper()
    if promote_match is not None:
        promoted = promote_match.group(1).lower() == "true"
    elif "NOT PROMOTED" in title_upper:
        promoted = False
    elif "PROMOTED" in title_upper:
        promoted = True
    else:
        promoted = False

    incumbent, candidate = _parse_metrics_table(body)
    learning = _parse_learning(body)

    # If genes missing but moves present, seed genes from moves
    if not genes and moves:
        genes = list(dict.fromkeys(m["gene"] for m in moves))

    return {
        "attempt_id": attempt_id,
        "section_title": title.strip(),
        "timestamp": timestamp,
        "sweep_file": sweep_file,
        "genes": genes,
        "moves": moves,
        "metrics": {
            "incumbent": incumbent,
            "candidate": candidate,
        },
        "promoted": promoted,
        "learning": learning,
    }


def _split_attempts(md_text: str) -> list[tuple[str, str, str]]:
    """Split markdown into (attempt_id, title, body) tuples.

    Body is everything from the header line down to the next Attempt header
    at same-or-higher level.
    """
    matches = list(ATTEMPT_HEADER_RE.finditer(md_text))
    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        body = md_text[start:end]
        out.append((m.group("id"), m.group("title"), body))
    return out


def _build_history(source_path: Path, md_text: str, source_sha1: str) -> dict:
    attempts_raw = _split_attempts(md_text)
    parsed = []
    errors = 0
    for attempt_id, title, body in attempts_raw:
        try:
            parsed.append(_parse_one_attempt(attempt_id, title, body))
        except Exception as e:
            errors += 1
            print(f"[CODEX SYNC] WARN: failed to parse Attempt {attempt_id}: {e!r}",
                  file=sys.stderr)

    # Dedup: sometimes an attempt_id appears twice (legit reuse with different title).
    # Keep all — they're distinct attempts from codex's perspective.
    n_promoted = sum(1 for a in parsed if a["promoted"])

    return {
        "codex_sync_metadata": {
            "source_file": str(source_path).replace("\\", "/"),
            "source_sha1": source_sha1,
            "last_parsed": _now_iso(),
            "parser_version": PARSER_VERSION,
            "n_attempts_parsed": len(parsed),
            "n_promoted": n_promoted,
            "n_parse_errors": errors,
        },
        "attempts": parsed,
    }


def _load_existing_sha1() -> str | None:
    if not HISTORY_JSON.exists():
        return None
    try:
        data = json.loads(HISTORY_JSON.read_text(encoding="utf-8"))
        return data.get("codex_sync_metadata", {}).get("source_sha1")
    except Exception:
        return None


def _write_raw_copy(source_path: Path, md_text: str, source_sha1: str) -> None:
    header = (
        f"<!-- Imported from: {source_path}\n"
        f"     SHA1: {source_sha1}\n"
        f"     Synced at: {_now_iso()}\n"
        f"     Do not edit this file by hand; it is regenerated by\n"
        f"     analysis/codex_attempts_sync.py on every continuous cycle. -->\n\n"
    )
    RAW_COPY.write_text(header + md_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync codex optimization_attempts markdown into structured JSON.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Path to codex optimization_attempts markdown.")
    parser.add_argument("--force", action="store_true",
                        help="Rewrite JSON even if source SHA1 is unchanged.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress info messages (only errors).")
    args = parser.parse_args()

    source: Path = args.source
    # Fallback chain: se --source nao existe, tenta REPO_SOURCE depois LOCAL_SOURCE.
    # Isso faz o script funcionar tanto localmente (dev com codex worktree) quanto
    # em runner remoto (so tem REPO_SOURCE, commitada em main).
    if not source.exists():
        for fallback in (REPO_SOURCE, LOCAL_SOURCE):
            if fallback != source and fallback.exists():
                _log(f"[CODEX SYNC] --source {source} not found, fallback -> {fallback}",
                     args.quiet)
                source = fallback
                break
        else:
            _log(f"[CODEX SYNC] no source available (tried {args.source}, "
                 f"{REPO_SOURCE}, {LOCAL_SOURCE}) — skipping (exit 0)",
                 args.quiet)
            return 0

    try:
        raw_bytes = source.read_bytes()
    except Exception as e:
        print(f"[CODEX SYNC] could not read source: {e!r}", file=sys.stderr)
        return 0  # non-fatal

    source_sha1 = _sha1(raw_bytes)
    md_text = raw_bytes.decode("utf-8", errors="replace")

    prev_sha1 = _load_existing_sha1()
    if prev_sha1 == source_sha1 and not args.force:
        _log(f"[CODEX SYNC] source SHA1 unchanged ({source_sha1[:10]}...) — no rewrite",
             args.quiet)
        return 0

    history = _build_history(source, md_text, source_sha1)

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    HISTORY_JSON.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_raw_copy(source, md_text, source_sha1)

    meta = history["codex_sync_metadata"]
    _log(
        f"[CODEX SYNC] parsed {meta['n_attempts_parsed']} attempts "
        f"({meta['n_promoted']} promoted, {meta['n_parse_errors']} parse errors); "
        f"wrote {HISTORY_JSON.relative_to(ROOT)} and {RAW_COPY.relative_to(ROOT)}",
        args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

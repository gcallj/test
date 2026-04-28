#!/usr/bin/env python3
"""Autonomous overnight ladder for swing Premium/HQ improvement.

The runner keeps discovery cheap and validation strict:
- rotates entry/regime/exit/timing sweeps until a local deadline;
- applies only candidates marked as promotable by the staged sweep;
- relies on apply_fullmetric_sweep_best.py for fail-closed overfit gates;
- regenerates signals at the end without repeating the heavy ETL.
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
LOG_DIR = ROOT / "logs"
RUN_STATE = ROOT / "analysis" / "swing_autonomous_state.json"

CATEGORIES: dict[str, list[str]] = {
    "D_entry_filter": [
        "vote_threshold_long",
        "vote_threshold_short",
        "z_threshold",
        "entry_aggressiveness",
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
        "resistance_overext_gate",
        "support_broken_gate",
    ],
    "G_exit_quality": [
        "min_hold_bars",
        "early_exit_block_bars",
        "profit_take_min_bars",
        "early_take_quality_gate",
        "early_take_score_decay_max",
    ],
    "C_timing": [
        "time_stop_bars",
        "consecutive_loss_cooldown",
    ],
}

ANCHORS: dict[str, str] = {
    "D_entry_filter": (
        "entry_score_threshold:+1,min_signal_strength:+1,"
        "score_percentile_trigger:+1,entry_aggressiveness:-1"
    ),
    "E_regime_vol": (
        "vol_regime_mode:+1,ma_filter_mode:+1,"
        "resistance_overext_gate:+1,support_broken_gate:+1"
    ),
    "G_exit_quality": (
        "profit_take_min_bars:+3,early_take_quality_gate:+4,"
        "min_hold_bars:+1,early_exit_block_bars:+2"
    ),
    "C_timing": "time_stop_bars:+1,consecutive_loss_cooldown:+1",
}


def _now() -> datetime:
    return datetime.now().astimezone()


def _stamp() -> str:
    return _now().strftime("%Y%m%d_%H%M%S")


def _parse_deadline(value: str) -> datetime:
    value = value.strip()
    now = _now()
    if "T" in value or "-" in value:
        dt = datetime.fromisoformat(value)
        return dt.astimezone() if dt.tzinfo else dt.replace(tzinfo=now.tzinfo)
    hh, mm = value.split(":", 1)
    dt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    if dt <= now:
        dt += timedelta(days=1)
    return dt


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def _run(cmd: list[str], *, timeout_s: int | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("GA_DISABLE_TELEGRAM_SEND", "1")
    env.setdefault("GA_KEEP_STORE", "1")
    env.setdefault("GA_ALPHA_FOCUS", "on")
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[{_now().isoformat(timespec='seconds')}] RUN: {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )


def _print_output(title: str, result: subprocess.CompletedProcess, tail_chars: int = 6000) -> None:
    print(f"\n--- {title} rc={result.returncode} ---", flush=True)
    out = result.stdout or ""
    if len(out) > tail_chars:
        print(out[-tail_chars:], flush=True)
    else:
        print(out, flush=True)


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"json_read_failed: {exc}"}


def _summarize_candidate(sweep: dict) -> dict:
    best = sweep.get("best_promoted") or sweep.get("best_guarded") or {}
    metrics = best.get("metrics", {}) if isinstance(best, dict) else {}
    decision = best.get("decision", {}) if isinstance(best, dict) else {}
    return {
        "label": best.get("label"),
        "promote": bool(decision.get("promote")),
        "fit": metrics.get("fit"),
        "wr_op": metrics.get("wr_med_operable"),
        "wr_target_op": metrics.get("wr_target_med_operable"),
        "alpha_op": metrics.get("alpha_ann_mean_operable"),
        "expectancy_op": metrics.get("expectancy_net_med_operable"),
        "hold_op": metrics.get("hold_med_operable"),
        "expected_hold_score_op": metrics.get("expected_hold_score_med_operable"),
        "wr_after_3_op": metrics.get("wr_after_3_bars_med_operable"),
        "n_buy_op": metrics.get("n_buy_operable"),
    }


def _run_sweep(category: str, combo_budget: int) -> tuple[Path, dict, subprocess.CompletedProcess]:
    out = ROOT / f"local_fullmetric_sweep_autonomous_{category}_{_stamp()}.json"
    genes = ",".join(CATEGORIES[category])
    cmd = [
        sys.executable,
        "run_local_fullmetric_sweep.py",
        "--alpha-focus",
        "on",
        "--genes",
        genes,
        "--combo-budget",
        str(combo_budget),
        "--combo-anchor-labels",
        ANCHORS.get(category, ""),
        "--out",
        str(out.name),
    ]
    result = _run(cmd, timeout_s=90 * 60)
    _print_output(f"sweep {category}", result)
    return out, _load_json(out), result


def _apply_if_promoted(sweep_path: Path, sweep: dict) -> tuple[bool, str]:
    if not sweep.get("best_promoted"):
        return False, "no_best_promoted"
    result = _run([sys.executable, "apply_fullmetric_sweep_best.py", str(sweep_path)], timeout_s=60 * 60)
    _print_output("apply candidate", result)
    return result.returncode == 0, f"apply_rc={result.returncode}"


def _finalize(run_predict: bool) -> None:
    print(f"[{_now().isoformat(timespec='seconds')}] Final validation", flush=True)
    tests = _run([sys.executable, "-m", "pytest", "tests\\test_ga_metrics.py", "-q"], timeout_s=10 * 60)
    _print_output("pytest", tests)

    overfit = _run(
        [
            sys.executable,
            "analysis\\overfitting_stage.py",
            "--holdout-days",
            "90",
            "--bootstrap",
            "100",
            "--rolling-folds",
            "2",
            "--embargo-days",
            "5",
            "--skip-sensitivity",
            "--negative-control-tickers",
            "40",
        ],
        timeout_s=40 * 60,
    )
    _print_output("overfit quick", overfit)

    if not run_predict:
        return

    py = (
        "from predict_daily import step_ga_load, step_refresh_close, "
        "step_guide_overlay, step_log_predictions; "
        "step_ga_load(); step_refresh_close(); step_guide_overlay(); step_log_predictions()"
    )
    pred = _run([sys.executable, "-u", "-c", py], timeout_s=60 * 60)
    _print_output("post-ETL predict", pred)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="08:00")
    parser.add_argument("--combo-budget", type=int, default=16)
    parser.add_argument("--min-minutes-left", type=int, default=35)
    parser.add_argument("--run-predict-final", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    deadline = _parse_deadline(args.until)
    print(f"[START] swing autonomous until {deadline.isoformat(timespec='seconds')}", flush=True)

    categories = list(CATEGORIES)
    cycle = 0
    promotions = 0
    while _now() + timedelta(minutes=args.min_minutes_left) < deadline:
        category = categories[cycle % len(categories)]
        cycle += 1
        entry = {
            "timestamp": _now().isoformat(timespec="seconds"),
            "cycle": cycle,
            "category": category,
            "deadline": deadline.isoformat(timespec="seconds"),
        }
        try:
            sweep_path, sweep, result = _run_sweep(category, args.combo_budget)
            candidate = _summarize_candidate(sweep)
            applied, apply_reason = _apply_if_promoted(sweep_path, sweep)
            promotions += int(applied)
            entry.update(
                {
                    "sweep_path": str(sweep_path),
                    "sweep_rc": result.returncode,
                    "candidate": candidate,
                    "applied": applied,
                    "apply_reason": apply_reason,
                }
            )
        except subprocess.TimeoutExpired as exc:
            entry.update({"error": f"timeout: {exc}"})
            print(f"[WARN] cycle timeout: {exc}", flush=True)
        except Exception as exc:
            entry.update({"error": repr(exc)})
            print(f"[WARN] cycle error: {exc!r}", flush=True)
        _append_jsonl(RUN_STATE, entry)

        if _now() + timedelta(minutes=args.min_minutes_left) >= deadline:
            break
        time.sleep(10)

    _append_jsonl(
        RUN_STATE,
        {
            "timestamp": _now().isoformat(timespec="seconds"),
            "event": "finalize_start",
            "cycles": cycle,
            "promotions": promotions,
        },
    )
    _finalize(args.run_predict_final)
    _append_jsonl(
        RUN_STATE,
        {
            "timestamp": _now().isoformat(timespec="seconds"),
            "event": "done",
            "cycles": cycle,
            "promotions": promotions,
        },
    )
    print(f"[DONE] cycles={cycle} promotions={promotions}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

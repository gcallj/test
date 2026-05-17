import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from analysis.exec_log import (
    STATUS_CANCELLED,
    STATUS_FILLED,
    STATUS_MISSED,
    STATUS_PENDING,
    aggregate_state,
    iter_log,
    record_fill,
    record_signal,
    record_status_change,
    summarize,
)
from analysis import exec_log_record


class _TmpLog:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "exec_log.jsonl"
        return self.path

    def __exit__(self, *exc):
        self._tmp.cleanup()


class RecordSignal(unittest.TestCase):
    def test_appends_pending_entry(self):
        with _TmpLog() as path:
            entry = record_signal(
                "2026-05-15", "PETR4.SA",
                desired_entry=38.50, desired_stop=37.95, desired_target=39.20,
                log_path=path,
            )
            self.assertIsNotNone(entry)
            self.assertEqual(entry["status"], STATUS_PENDING)
            self.assertEqual(entry["desired_entry"], 38.50)
            self.assertEqual(entry["fill_price"], None)
            self.assertEqual(entry["slippage_bps_actual"], None)
            # File contains exactly one line.
            self.assertEqual(len(list(iter_log(path))), 1)

    def test_idempotent_on_pending_duplicate(self):
        with _TmpLog() as path:
            first = record_signal("2026-05-15", "PETR4.SA", desired_entry=38.50, log_path=path)
            dup = record_signal("2026-05-15", "PETR4.SA", desired_entry=38.50, log_path=path)
            self.assertIsNotNone(first)
            self.assertIsNone(dup)
            self.assertEqual(len(list(iter_log(path))), 1)

    def test_different_dates_or_tickers_create_new_pendings(self):
        with _TmpLog() as path:
            record_signal("2026-05-15", "PETR4.SA", desired_entry=38.50, log_path=path)
            record_signal("2026-05-16", "PETR4.SA", desired_entry=38.60, log_path=path)
            record_signal("2026-05-15", "VALE3.SA", desired_entry=72.0, log_path=path)
            self.assertEqual(len(list(iter_log(path))), 3)

    def test_normalizes_date_to_yyyymmdd(self):
        with _TmpLog() as path:
            record_signal("2026-05-15T18:00:00Z", "PETR4.SA", log_path=path)
            row = next(iter(iter_log(path)))
            self.assertEqual(row["date"], "2026-05-15")

    def test_rejects_blank_ticker(self):
        with _TmpLog() as path:
            with self.assertRaises(ValueError):
                record_signal("2026-05-15", "  ", log_path=path)


class RecordFill(unittest.TestCase):
    def test_computes_positive_slippage_when_filled_above_desired(self):
        with _TmpLog() as path:
            record_signal("2026-05-15", "PETR4.SA", desired_entry=38.50, log_path=path)
            # Sleep 1s to ensure FILLED has a later recorded_at than PENDING
            # (the assertion in aggregate_state uses >= for stability).
            time.sleep(1)
            fill = record_fill("2026-05-15", "PETR4.SA", fill_price=38.55, log_path=path)
            self.assertEqual(fill["status"], STATUS_FILLED)
            # 5 cents on 38.50 = ~12.99 bps
            self.assertAlmostEqual(fill["slippage_bps_actual"], (0.05 / 38.50) * 10_000, places=4)
            # desired_entry copied from PENDING
            self.assertEqual(fill["desired_entry"], 38.50)

    def test_computes_negative_slippage_when_filled_below_desired(self):
        with _TmpLog() as path:
            record_signal("2026-05-15", "PETR4.SA", desired_entry=38.50, log_path=path)
            fill = record_fill("2026-05-15", "PETR4.SA", fill_price=38.40, log_path=path)
            self.assertLess(fill["slippage_bps_actual"], 0)

    def test_fill_without_pending_records_with_null_slippage(self):
        with _TmpLog() as path:
            fill = record_fill("2026-05-15", "PETR4.SA", fill_price=38.55, log_path=path)
            self.assertEqual(fill["status"], STATUS_FILLED)
            self.assertIsNone(fill["slippage_bps_actual"])
            self.assertIsNone(fill["desired_entry"])

    def test_rejects_non_finite_fill_price(self):
        with _TmpLog() as path:
            with self.assertRaises(ValueError):
                record_fill("2026-05-15", "PETR4.SA", fill_price="not a number", log_path=path)


class RecordStatusChange(unittest.TestCase):
    def test_missed_appends_entry(self):
        with _TmpLog() as path:
            record_signal("2026-05-15", "PETR4.SA", desired_entry=38.5, log_path=path)
            entry = record_status_change("2026-05-15", "PETR4.SA", STATUS_MISSED, log_path=path)
            self.assertEqual(entry["status"], STATUS_MISSED)
            # desired_entry preserved
            self.assertEqual(entry["desired_entry"], 38.5)

    def test_rejects_invalid_status(self):
        with _TmpLog() as path:
            with self.assertRaises(ValueError):
                record_status_change("2026-05-15", "PETR4.SA", "FILLED", log_path=path)


class AggregateState(unittest.TestCase):
    def test_latest_wins(self):
        with _TmpLog() as path:
            record_signal("2026-05-15", "PETR4.SA", desired_entry=38.5, log_path=path)
            time.sleep(1)
            record_fill("2026-05-15", "PETR4.SA", fill_price=38.55, log_path=path)
            state = aggregate_state(path)
            self.assertEqual(len(state), 1)
            key = ("2026-05-15", "PETR4.SA")
            self.assertIn(key, state)
            self.assertEqual(state[key]["status"], STATUS_FILLED)

    def test_independent_keys(self):
        with _TmpLog() as path:
            record_signal("2026-05-15", "PETR4.SA", desired_entry=38.5, log_path=path)
            record_signal("2026-05-16", "PETR4.SA", desired_entry=38.6, log_path=path)
            record_signal("2026-05-15", "VALE3.SA", desired_entry=70.0, log_path=path)
            state = aggregate_state(path)
            self.assertEqual(len(state), 3)


class Summarize(unittest.TestCase):
    def test_empty_log(self):
        with _TmpLog() as path:
            out = summarize(path)
            self.assertEqual(out["counts_by_status"][STATUS_PENDING], 0)
            self.assertEqual(out["n_filled_with_slippage"], 0)

    def test_counts_and_slippage_stats(self):
        with _TmpLog() as path:
            # 2 filled with slippage, 1 cancelled, 1 still pending
            record_signal("2026-05-15", "PETR4.SA", desired_entry=10.0, log_path=path)
            record_fill("2026-05-15", "PETR4.SA", fill_price=10.10, log_path=path)  # +100 bps
            record_signal("2026-05-15", "VALE3.SA", desired_entry=20.0, log_path=path)
            record_fill("2026-05-15", "VALE3.SA", fill_price=19.90, log_path=path)  # -50 bps
            record_signal("2026-05-15", "ITUB4.SA", desired_entry=30.0, log_path=path)
            record_status_change("2026-05-15", "ITUB4.SA", STATUS_CANCELLED, log_path=path)
            record_signal("2026-05-15", "BBDC4.SA", desired_entry=15.0, log_path=path)
            out = summarize(path)
            self.assertEqual(out["counts_by_status"][STATUS_FILLED], 2)
            self.assertEqual(out["counts_by_status"][STATUS_CANCELLED], 1)
            self.assertEqual(out["counts_by_status"][STATUS_PENDING], 1)
            self.assertEqual(out["n_filled_with_slippage"], 2)
            # mean(+100, -50) = +25
            self.assertAlmostEqual(out["slippage_mean_bps"], 25.0, places=4)


class CLIRoundtrip(unittest.TestCase):
    def _run(self, args):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = exec_log_record.main(args)
        return rc, buf.getvalue()

    def test_signal_then_fill_then_summary(self):
        with _TmpLog() as path:
            rc, out = self._run(["signal", "--date", "2026-05-15", "--ticker", "PETR4.SA",
                                 "--entry", "38.50", "--stop", "37.95", "--target", "39.20",
                                 "--log", str(path)])
            self.assertEqual(rc, 0)
            self.assertIn(STATUS_PENDING, out)

            rc, out = self._run(["fill", "--date", "2026-05-15", "--ticker", "PETR4.SA",
                                 "--price", "38.55", "--log", str(path)])
            self.assertEqual(rc, 0)
            self.assertIn(STATUS_FILLED, out)

            rc, out = self._run(["summary", "--log", str(path)])
            self.assertEqual(rc, 0)
            summary = json.loads(out)
            self.assertEqual(summary["counts_by_status"][STATUS_FILLED], 1)

    def test_state_command_dumps_keyed_json(self):
        with _TmpLog() as path:
            self._run(["signal", "--date", "2026-05-15", "--ticker", "PETR4.SA",
                       "--entry", "38.50", "--log", str(path)])
            rc, out = self._run(["state", "--log", str(path)])
            self.assertEqual(rc, 0)
            state = json.loads(out)
            self.assertIn("2026-05-15|PETR4.SA", state)


if __name__ == "__main__":
    unittest.main()

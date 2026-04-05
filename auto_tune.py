# -*- coding: utf-8 -*-
"""
auto_tune.py — AutoTuner, RuntimeGuard, memory utilities
=========================================================
Dynamically computes optimal workers / pop / ngen / windows
based on system RAM, CPU, and a time+memory budget.
"""

import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from payload_store import PayloadStore, WindowPlan


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class AutoTuneResult:
    """Output of the auto-tuner. Consumed by ga_run.run_global_ga_two_stage."""
    n_workers: int

    stage1_pop: int
    stage1_ngen: int
    stage1_window_count: int

    stage2_pop: int
    stage2_ngen: int
    stage2_window_count: int

    batch_size: int

    sec_per_eval: float
    estimated_total_hours: float
    time_budget_seconds: float
    ram_reserve_gb: float


# ---------------------------------------------------------------------------
# Memory utilities (cross-platform)
# ---------------------------------------------------------------------------
def _get_process_rss_gb() -> float:
    """Get current process RSS in GB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            import ctypes.wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(pmc), pmc.cb
            )
            return pmc.WorkingSetSize / (1024 ** 3)
        except Exception:
            return 0.5  # fallback estimate
    # Linux/Mac fallback
    try:
        with open(f"/proc/{os.getpid()}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB to GB
    except Exception:
        pass
    return 0.5


def _get_available_ram_gb() -> float:
    """Get available system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return mem.ullAvailPhys / (1024 ** 3)
        except Exception:
            return 4.0  # fallback
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 4.0


def _get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            return mem.ullTotalPhys / (1024 ** 3)
        except Exception:
            return 8.0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 8.0


# ---------------------------------------------------------------------------
# Probe worker (module-level for ProcessPoolExecutor on Windows/spawn)
# ---------------------------------------------------------------------------
_probe_store: Optional[PayloadStore] = None


def _probe_worker_init(store_path: str):
    global _probe_store
    _probe_store = PayloadStore(store_path, mode="r")


def _probe_worker_eval(args: Tuple[List[float], str, int, int]) -> float:
    """Evaluate one genome on one ticker slice. Returns wall-clock seconds."""
    genome, ticker, start_idx, end_idx = args
    t0 = time.perf_counter()

    assert _probe_store is not None
    arrays = _probe_store.get_ticker_arrays(ticker, start_idx, end_idx)

    # Inline a minimal version of backtest evaluation to measure cost
    o = np.asarray(arrays["open"], dtype=np.float64)
    h = np.asarray(arrays["high"], dtype=np.float64)
    l = np.asarray(arrays["low"], dtype=np.float64)
    c = np.asarray(arrays["close"], dtype=np.float64)
    atr_arr = np.asarray(arrays["atr"], dtype=np.float64)
    sm = np.asarray(arrays["score_matrix"], dtype=np.float64)

    # We don't import backtest_stats_global_intraday here to avoid
    # pulling in main_cell_v3; instead use array operations to estimate cost
    n = len(c)
    if n > 5 and sm.ndim == 2 and sm.shape[0] == n:
        feat_n = max(1, sm.shape[1])
        votes = (sm > 0.3).sum(axis=1) / feat_n
        _ = np.cumsum(votes)
        _ = np.std(c)

    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# AutoTuner
# ---------------------------------------------------------------------------
class AutoTuner:
    """
    Probes system resources and computes optimal GA configuration.
    """

    def __init__(
        self,
        store: PayloadStore,
        window_plans: List[WindowPlan],
        time_budget_hours: float = 6.0,
        ram_reserve_gb: float = 2.0,
        workers_override: Optional[int] = None,
    ):
        self.store = store
        self.window_plans = window_plans
        self.time_budget = time_budget_hours * 3600.0
        self.ram_reserve = ram_reserve_gb
        self.workers_override = workers_override

    def probe(self) -> Dict[str, float]:
        """
        Run a small evaluation probe to measure sec_per_eval and memory usage.
        """
        parent_rss = _get_process_rss_gb()
        available_before = _get_available_ram_gb()

        # Pick 2 tickers and create small tasks
        tickers = self.store.tickers[:min(3, len(self.store.tickers))]
        tasks = []
        dummy_genome = [random.uniform(0, 1) for _ in range(20)]

        for tk in tickers:
            d = self.store.get_ticker_dates_int64(tk)
            n = len(d)
            # Use up to 500 rows per ticker for the probe
            end = min(n, 500)
            tasks.append((dummy_genome, tk, 0, end))

        # Try ProcessPoolExecutor with 1 worker
        sec_per_eval = 0.1  # default if probe fails
        worker_peak = 0.15  # default GB

        try:
            t0 = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=1,
                initializer=_probe_worker_init,
                initargs=(self.store.store_path,),
            ) as executor:
                results = list(executor.map(_probe_worker_eval, tasks))
            elapsed = time.perf_counter() - t0

            if results:
                sec_per_eval = max(elapsed / len(results), 0.001)

            available_after = _get_available_ram_gb()
            worker_peak = max(0.05, available_before - available_after)
        except Exception as e:
            print(f"[AUTO_TUNE] probe failed ({e}), using defaults")

        # Scale sec_per_eval to full-size tickers and all tickers per window
        avg_ticker_rows = self.store.meta.total_rows / max(1, len(self.store.tickers))
        probe_rows = min(500, avg_ticker_rows)
        scale_factor = avg_ticker_rows / max(probe_rows, 1)
        sec_per_eval_full = sec_per_eval * scale_factor * len(self.store.tickers)

        return {
            "parent_rss_gb": parent_rss,
            "worker_peak_gb": worker_peak,
            "sec_per_eval": sec_per_eval_full,
            "available_ram_gb": available_before,
            "total_ram_gb": _get_total_ram_gb(),
        }

    def compute(self) -> AutoTuneResult:
        """
        Probe system, compute safe_workers, grid-search optimal config.
        """
        probe_data = self.probe()
        sec_per_eval = probe_data["sec_per_eval"]
        parent_rss = probe_data["parent_rss_gb"]
        worker_peak = probe_data["worker_peak_gb"]
        total_ram = probe_data["total_ram_gb"]
        available = probe_data["available_ram_gb"]

        print(f"[AUTO_TUNE] probe: sec_per_eval={sec_per_eval:.3f}s "
              f"parent_rss={parent_rss:.2f}GB worker_peak={worker_peak:.2f}GB "
              f"available={available:.2f}GB total={total_ram:.2f}GB")

        # Pre-flight RAM check
        reserve = max(self.ram_reserve, 0.25 * total_ram)
        if available < reserve + 1.0:
            print(f"[AUTO_TUNE] WARNING: available RAM ({available:.1f}GB) < "
                  f"reserve ({reserve:.1f}GB) + 1GB. Proceeding with 1 worker.")

        # Compute safe workers
        ram_usable = total_ram - reserve
        cpu_count = os.cpu_count() or 2
        if worker_peak > 0.01:
            safe_workers = int(math.floor((ram_usable - parent_rss) / worker_peak))
        else:
            safe_workers = cpu_count
        safe_workers = max(1, min(safe_workers, cpu_count))

        if self.workers_override is not None:
            safe_workers = min(safe_workers, int(self.workers_override))
            safe_workers = max(1, safe_workers)

        n_available_windows = len(self.window_plans)

        # Grid search
        s1_windows_cands = [w for w in [4, 6] if w <= n_available_windows]
        if not s1_windows_cands:
            s1_windows_cands = [min(n_available_windows, 4)]
        s1_pop_cands = [64, 96, 128]
        s1_ngen_cands = [10, 15, 20, 25]

        s2_windows_cands = [w for w in [8, 10] if w <= n_available_windows]
        if not s2_windows_cands:
            s2_windows_cands = [min(n_available_windows, 8)]
        s2_pop_cands = [24, 32, 48, 64]
        s2_ngen_cands = [12, 20, 30, 40]

        time_margin = 0.80  # 20% safety margin
        budget_effective = self.time_budget * time_margin

        best_config = None
        best_score = (-1, -1, -1, -1)  # (workers, s2_win, s1_pop, sum_ngen)

        for s1w in s1_windows_cands:
            for s1p in s1_pop_cands:
                for s1n in s1_ngen_cands:
                    for s2w in s2_windows_cands:
                        for s2p in s2_pop_cands:
                            for s2n in s2_ngen_cands:
                                # Estimate total evaluations
                                # Each gen: pop evals + offspring evals = ~2x pop
                                s1_evals = s1p * s1n * 2 * s1w
                                s2_evals = s2p * s2n * 2 * s2w
                                total_evals = s1_evals + s2_evals

                                # Time per eval scaled by windows
                                est_seconds = (total_evals * sec_per_eval) / max(1, safe_workers * len(self.store.tickers))
                                # Each eval is one genome on all tickers in all windows for that stage
                                est_seconds = (
                                    (s1p * s1n * 2 * sec_per_eval * s1w / max(n_available_windows, 1))
                                    + (s2p * s2n * 2 * sec_per_eval * s2w / max(n_available_windows, 1))
                                ) / max(1, safe_workers)

                                if est_seconds > budget_effective:
                                    continue

                                score = (safe_workers, s2w, s1p, s1n + s2n)
                                if score > best_score:
                                    best_score = score
                                    best_config = (s1w, s1p, s1n, s2w, s2p, s2n, est_seconds)

        if best_config is None:
            # Fallback: small pop but reasonable ngen for convergence
            best_config = (
                s1_windows_cands[0], s1_pop_cands[0], s1_ngen_cands[-1],  # max ngen for S1
                s2_windows_cands[0], s2_pop_cands[0], max(20, s2_ngen_cands[0]),  # at least 20 gen for S2
                budget_effective,
            )
            print("[AUTO_TUNE] no config fits budget, using small pop + extended ngen for convergence")

        s1w, s1p, s1n, s2w, s2p, s2n, est_sec = best_config

        batch_size = max(4, min(16, s1p // (safe_workers * 2)))

        result = AutoTuneResult(
            n_workers=safe_workers,
            stage1_pop=s1p,
            stage1_ngen=s1n,
            stage1_window_count=s1w,
            stage2_pop=s2p,
            stage2_ngen=s2n,
            stage2_window_count=s2w,
            batch_size=batch_size,
            sec_per_eval=sec_per_eval,
            estimated_total_hours=est_sec / 3600.0,
            time_budget_seconds=self.time_budget,
            ram_reserve_gb=self.ram_reserve,
        )

        print(f"[AUTO_TUNE] config: workers={safe_workers} "
              f"S1(win={s1w}, pop={s1p}, ngen={s1n}) "
              f"S2(win={s2w}, pop={s2p}, ngen={s2n}) "
              f"batch={batch_size} est={est_sec/3600:.2f}h")

        return result

    @staticmethod
    def manual_config(
        n_workers: int = 1,
        stage1_pop: int = 200,
        stage1_ngen: int = 60,
        stage1_windows: int = 4,
        stage2_pop: int = 0,
        stage2_ngen: int = 0,
        stage2_windows: int = 0,
        time_budget_hours: float = 6.0,
        ram_reserve_gb: float = 2.0,
    ) -> AutoTuneResult:
        """
        For GA_AUTO_TUNE=0: build AutoTuneResult from manual settings.
        If stage2_pop == 0, no Stage 2 is run.
        """
        return AutoTuneResult(
            n_workers=max(1, n_workers),
            stage1_pop=stage1_pop,
            stage1_ngen=stage1_ngen,
            stage1_window_count=stage1_windows,
            stage2_pop=stage2_pop,
            stage2_ngen=stage2_ngen,
            stage2_window_count=stage2_windows,
            batch_size=max(4, stage1_pop // max(1, n_workers * 2)),
            sec_per_eval=0.0,
            estimated_total_hours=0.0,
            time_budget_seconds=time_budget_hours * 3600.0,
            ram_reserve_gb=ram_reserve_gb,
        )


# ---------------------------------------------------------------------------
# RuntimeGuard
# ---------------------------------------------------------------------------
class RuntimeGuard:
    """
    Monitors GA execution and adjusts parameters in-flight.
    Called by ga_run after each generation.
    """

    def __init__(self, tune_result: AutoTuneResult):
        self.time_budget = tune_result.time_budget_seconds
        self.ram_reserve = tune_result.ram_reserve_gb
        self.n_workers = tune_result.n_workers
        self.gen_times: List[float] = []
        self._start_time = time.perf_counter()

    def after_generation(
        self, gen: int, ngen: int, gen_elapsed: float
    ) -> Dict[str, Any]:
        """
        Called after each generation. Returns adjustment dict.

        Keys:
          ngen_adjusted: int or None — new total ngen if budget is exceeded
          reduce_workers: int or None — new worker count if RAM is low
          checkpoint_and_exit: bool — save and exit if critically low on RAM
        """
        self.gen_times.append(gen_elapsed)

        adjustments: Dict[str, Any] = {
            "ngen_adjusted": None,
            "reduce_workers": None,
            "checkpoint_and_exit": False,
        }

        # Time projection after gen 3
        if gen >= 3:
            recent = self.gen_times[-min(5, len(self.gen_times)):]
            avg_gen_time = sum(recent) / len(recent)
            remaining_gens = ngen - gen
            total_elapsed = time.perf_counter() - self._start_time
            projected_remaining = avg_gen_time * remaining_gens
            projected_total = total_elapsed + projected_remaining

            if projected_total > self.time_budget:
                allowed_remaining = max(0, self.time_budget - total_elapsed)
                new_remaining = max(1, int(allowed_remaining / max(avg_gen_time, 0.01)))
                new_ngen = gen + new_remaining
                if new_ngen < ngen:
                    adjustments["ngen_adjusted"] = new_ngen
                    print(f"[GUARD] time projection {projected_total/3600:.2f}h > "
                          f"budget {self.time_budget/3600:.2f}h, "
                          f"reducing ngen {ngen} -> {new_ngen}")

        # RAM check
        available_gb = _get_available_ram_gb()
        if available_gb < self.ram_reserve:
            if self.n_workers > 1:
                self.n_workers -= 1
                adjustments["reduce_workers"] = self.n_workers
                print(f"[GUARD] RAM low ({available_gb:.2f}GB < {self.ram_reserve:.2f}GB), "
                      f"reducing workers to {self.n_workers}")
            else:
                adjustments["checkpoint_and_exit"] = True
                print(f"[GUARD] RAM critically low ({available_gb:.2f}GB), "
                      f"checkpointing and exiting")

        return adjustments

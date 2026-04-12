# -*- coding: utf-8 -*-
"""
payload_store.py — Memory-mapped PayloadStore + WindowPlan
==========================================================
Eliminates per-window array duplication by storing all ticker data once
in memmap files. Workers open read-only (zero-copy via OS page cache).
"""

import json
import os
import shutil
from bisect import bisect_right, insort
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (duplicated from main_cell_v3 to avoid importing the Colab module)
# ---------------------------------------------------------------------------
ATR_EPS = 1e-12


# ---------------------------------------------------------------------------
# Utility: rolling_percentile_rank (same as main_cell_v3.py line 381)
# ---------------------------------------------------------------------------
def _rolling_percentile_rank(arr: np.ndarray, window: int = 252) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    n = len(x)
    if n == 0:
        return out
    sorted_vals: list = []
    for i in range(n):
        xi = x[i]
        if np.isfinite(xi):
            insort(sorted_vals, float(xi))
        j_rm = i - window
        if j_rm >= 0 and valid[j_rm]:
            xrm = float(x[j_rm])
            k = bisect_right(sorted_vals, xrm) - 1
            if k >= 0:
                sorted_vals.pop(k)
        if len(sorted_vals) >= 20 and np.isfinite(xi):
            out[i] = bisect_right(sorted_vals, float(xi)) / float(len(sorted_vals))
    return out


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TickerSlice:
    """Index range into a ticker's base arrays within the PayloadStore."""
    ticker: str
    start_idx: int      # inclusive (relative to ticker's offset in the store)
    end_idx: int         # exclusive


@dataclass
class WindowPlan:
    """One walk-forward window: train and test slices per ticker, by index."""
    train_slices: Dict[str, TickerSlice]
    test_slices: Dict[str, TickerSlice]
    train_start: Any = None  # pd.Timestamp (for logging only)
    train_end: Any = None
    test_start: Any = None
    test_end: Any = None


@dataclass
class PayloadStoreMeta:
    """Metadata persisted as meta.json inside the store directory."""
    tickers: List[str]
    offsets: Dict[str, List[int]]      # ticker -> [start, end] in global arrays
    n_features: int
    feature_cols: List[str]
    total_rows: int
    dtypes: Dict[str, str]            # array_name -> dtype string


# ---------------------------------------------------------------------------
# PayloadStore
# ---------------------------------------------------------------------------
class PayloadStore:
    """
    Memory-mapped payload store for GA evaluation.

    Coordinator builds once via ``build_from_ticker_payloads``.
    Workers open in read-only mode via ``PayloadStore(path, mode='r')``.
    """

    _ARRAY_NAMES_1D = ("open", "high", "low", "close", "atr", "ma", "vol_rank",
                        "volume", "ret_fwd", "long_votes", "short_votes")
    _ARRAY_NAMES_EXTRA = ("dates", "valid_mask")
    _ARRAY_NAME_2D = "score_matrix"

    def __init__(self, store_dir: str, mode: str = "r"):
        self.store_dir = os.path.abspath(store_dir)
        self._mode = mode
        self._arrays: Dict[str, np.memmap] = {}

        meta_path = os.path.join(self.store_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.meta = PayloadStoreMeta(
            tickers=raw["tickers"],
            offsets={k: v for k, v in raw["offsets"].items()},
            n_features=raw["n_features"],
            feature_cols=raw["feature_cols"],
            total_rows=raw["total_rows"],
            dtypes=raw["dtypes"],
        )

        self._load_memmaps()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build_from_ticker_payloads(
        cls,
        ticker_payloads: Dict[str, Dict[str, Any]],
        feature_cols: List[str],
        store_dir: str,
    ) -> "PayloadStore":
        """
        Coordinator method: writes all ticker data into memmap files.
        Computes vol_rank once per ticker during construction.
        Casts market data + score_matrix to float32; dates to int64; valid_mask to uint8.
        """
        store_dir = os.path.abspath(store_dir)
        os.makedirs(store_dir, exist_ok=True)

        tickers = list(ticker_payloads.keys())
        n_features = len(feature_cols)

        # First pass: compute total rows and offsets
        offsets: Dict[str, List[int]] = {}
        total_rows = 0
        for tk in tickers:
            p = ticker_payloads[tk]
            n = len(p["close"])
            offsets[tk] = [total_rows, total_rows + n]
            total_rows += n

        if total_rows == 0:
            raise ValueError("No data rows found in ticker_payloads")

        # Create memmap files
        dtype_map = {
            "open": np.float32, "high": np.float32, "low": np.float32,
            "close": np.float32, "atr": np.float32, "ma": np.float32,
            "vol_rank": np.float32,
            "volume": np.float32, "ret_fwd": np.float32,
            "long_votes": np.float32, "short_votes": np.float32,
            "dates": np.int64,
            "valid_mask": np.uint8,
        }

        memmaps: Dict[str, np.memmap] = {}
        for name in cls._ARRAY_NAMES_1D + cls._ARRAY_NAMES_EXTRA:
            path = os.path.join(store_dir, f"{name}.dat")
            dt = dtype_map[name]
            memmaps[name] = np.memmap(path, dtype=dt, mode="w+", shape=(total_rows,))

        # score_matrix is 2D
        sm_path = os.path.join(store_dir, "score_matrix.dat")
        memmaps["score_matrix"] = np.memmap(
            sm_path, dtype=np.float32, mode="w+", shape=(total_rows, n_features)
        )

        # Second pass: write data
        for tk in tickers:
            p = ticker_payloads[tk]
            s, e = offsets[tk]
            n = e - s

            for name in ("open", "high", "low", "close", "atr", "ma", "volume"):
                arr = p.get(name)
                if arr is not None:
                    memmaps[name][s:e] = np.asarray(arr, dtype=np.float64).astype(np.float32)
                else:
                    memmaps[name][s:e] = np.nan

            # Phase-3 / optional 1D float32 arrays
            for name in ("ret_fwd", "long_votes", "short_votes"):
                arr = p.get(name)
                if arr is not None:
                    memmaps[name][s:e] = np.asarray(arr, dtype=np.float64).astype(np.float32)
                else:
                    memmaps[name][s:e] = np.nan

            # score_matrix
            sm = p.get("score_matrix")
            if sm is not None:
                sm_np = np.asarray(sm, dtype=np.float64)
                if sm_np.ndim == 2:
                    # Pad or truncate feature columns to n_features
                    actual_cols = sm_np.shape[1]
                    if actual_cols >= n_features:
                        memmaps["score_matrix"][s:e, :] = sm_np[:, :n_features].astype(np.float32)
                    else:
                        memmaps["score_matrix"][s:e, :actual_cols] = sm_np.astype(np.float32)
                        memmaps["score_matrix"][s:e, actual_cols:] = 0.0

            # vol_rank: compute once per ticker
            c = np.asarray(p["close"], dtype=np.float64)
            atr = np.asarray(p["atr"], dtype=np.float64)
            vr = _rolling_percentile_rank(atr / np.maximum(c, ATR_EPS), 252)
            memmaps["vol_rank"][s:e] = vr.astype(np.float32)

            # dates: convert to int64 nanoseconds via datetime64[ns] view
            dates_raw = p.get("dates")
            if dates_raw is not None:
                dates_dt = np.asarray(pd.to_datetime(dates_raw), dtype="datetime64[ns]")
                memmaps["dates"][s:e] = dates_dt.view(np.int64)
            else:
                memmaps["dates"][s:e] = 0

            # valid_mask
            vm = p.get("valid_mask")
            if vm is not None:
                memmaps["valid_mask"][s:e] = np.asarray(vm, dtype=bool).astype(np.uint8)
            else:
                memmaps["valid_mask"][s:e] = 1

        # Flush all memmaps
        for mm in memmaps.values():
            mm.flush()
            del mm
        memmaps.clear()

        # Write metadata
        dtypes_str = {name: str(np.dtype(dt)) for name, dt in dtype_map.items()}
        dtypes_str["score_matrix"] = str(np.dtype(np.float32))

        meta = PayloadStoreMeta(
            tickers=tickers,
            offsets=offsets,
            n_features=n_features,
            feature_cols=feature_cols,
            total_rows=total_rows,
            dtypes=dtypes_str,
        )
        meta_path = os.path.join(store_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

        return cls(store_dir, mode="r")

    # ------------------------------------------------------------------
    # Memmap loading
    # ------------------------------------------------------------------
    def _load_memmaps(self):
        mode = self._mode
        tr = self.meta.total_rows
        nf = self.meta.n_features

        dtype_map = {
            "open": np.float32, "high": np.float32, "low": np.float32,
            "close": np.float32, "atr": np.float32, "ma": np.float32,
            "vol_rank": np.float32,
            "volume": np.float32, "ret_fwd": np.float32,
            "long_votes": np.float32, "short_votes": np.float32,
            "dates": np.int64,
            "valid_mask": np.uint8,
        }
        for name, dt in dtype_map.items():
            path = os.path.join(self.store_dir, f"{name}.dat")
            if not os.path.exists(path):
                # Older stores may lack newer arrays; skip gracefully
                continue
            self._arrays[name] = np.memmap(path, dtype=dt, mode=mode, shape=(tr,))

        sm_path = os.path.join(self.store_dir, "score_matrix.dat")
        self._arrays["score_matrix"] = np.memmap(
            sm_path, dtype=np.float32, mode=mode, shape=(tr, nf)
        )

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------
    def get_ticker_arrays(
        self,
        ticker: str,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Returns dict of array views (memmap slices) for a ticker.
        If start_idx/end_idx provided, returns sub-slices within the ticker's range.
        """
        off = self.meta.offsets[ticker]
        s0, e0 = off[0], off[1]

        if start_idx is not None:
            s = s0 + start_idx
        else:
            s = s0
        if end_idx is not None:
            e = s0 + end_idx
        else:
            e = e0

        s = max(s, s0)
        e = min(e, e0)

        result: Dict[str, np.ndarray] = {}
        for name in self._ARRAY_NAMES_1D + self._ARRAY_NAMES_EXTRA:
            if name in self._arrays:
                result[name] = self._arrays[name][s:e]
        result["score_matrix"] = self._arrays["score_matrix"][s:e]
        return result

    def get_ticker_dates_int64(self, ticker: str) -> np.ndarray:
        """Returns the raw int64 dates array for a ticker (for np.searchsorted)."""
        s, e = self.meta.offsets[ticker]
        return self._arrays["dates"][s:e]

    def get_window_payloads(
        self, window_plan: WindowPlan
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        """
        Returns (train_payloads, test_payloads) for one WindowPlan.
        Each is Dict[ticker -> Dict[array_name -> np.ndarray view]].
        Zero-copy: all values are memmap slices.
        """
        train_payloads: Dict[str, Dict[str, np.ndarray]] = {}
        test_payloads: Dict[str, Dict[str, np.ndarray]] = {}

        for tk, ts in window_plan.train_slices.items():
            if ts.end_idx > ts.start_idx:
                train_payloads[tk] = self.get_ticker_arrays(tk, ts.start_idx, ts.end_idx)

        for tk, ts in window_plan.test_slices.items():
            if ts.end_idx > ts.start_idx:
                test_payloads[tk] = self.get_ticker_arrays(tk, ts.start_idx, ts.end_idx)

        return train_payloads, test_payloads

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def tickers(self) -> List[str]:
        return self.meta.tickers

    @property
    def store_path(self) -> str:
        return self.store_dir

    @property
    def min_date(self) -> pd.Timestamp:
        dates = self._arrays["dates"]
        valid = dates[dates > 0]
        if len(valid) == 0:
            return pd.Timestamp("1970-01-01")
        return pd.Timestamp(int(valid.min()), unit="ns")

    @property
    def max_date(self) -> pd.Timestamp:
        dates = self._arrays["dates"]
        valid = dates[dates > 0]
        if len(valid) == 0:
            return pd.Timestamp("1970-01-01")
        return pd.Timestamp(int(valid.max()), unit="ns")

    # ------------------------------------------------------------------
    # Close / Cleanup
    # ------------------------------------------------------------------
    def close(self):
        """Release open memmap handles without deleting the store directory."""
        for name in list(self._arrays.keys()):
            del self._arrays[name]
        self._arrays.clear()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        """Delete the store directory and all memmap files."""
        self.close()

        if os.path.isdir(self.store_dir):
            shutil.rmtree(self.store_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# WindowPlan builder
# ---------------------------------------------------------------------------
def build_window_plans(
    store: PayloadStore,
    windows: List[Tuple],
) -> List[WindowPlan]:
    """
    For each (tr_start, tr_end, te_start, te_end) window, computes
    index ranges per ticker using np.searchsorted on the sorted dates array.

    Returns list of WindowPlan objects (no data copies, just indices).
    """
    # Pre-cache int64 dates per ticker
    dates_cache: Dict[str, np.ndarray] = {}
    for tk in store.tickers:
        dates_cache[tk] = store.get_ticker_dates_int64(tk)

    plans: List[WindowPlan] = []
    for window in windows:
        tr_start, tr_end, te_start, te_end = window[0], window[1], window[2], window[3]

        # Convert timestamps to int64 nanoseconds for searchsorted
        tr_start_ns = np.int64(pd.Timestamp(tr_start).value)
        tr_end_ns = np.int64(pd.Timestamp(tr_end).value)
        te_start_ns = np.int64(pd.Timestamp(te_start).value)
        te_end_ns = np.int64(pd.Timestamp(te_end).value)

        train_slices: Dict[str, TickerSlice] = {}
        test_slices: Dict[str, TickerSlice] = {}

        for tk in store.tickers:
            d = dates_cache[tk]
            if len(d) == 0:
                continue

            # Train: dates >= tr_start and dates <= tr_end
            tr_s = int(np.searchsorted(d, tr_start_ns, side="left"))
            tr_e = int(np.searchsorted(d, tr_end_ns, side="right"))
            if tr_e > tr_s:
                train_slices[tk] = TickerSlice(ticker=tk, start_idx=tr_s, end_idx=tr_e)

            # Test: dates >= te_start and dates <= te_end
            te_s = int(np.searchsorted(d, te_start_ns, side="left"))
            te_e = int(np.searchsorted(d, te_end_ns, side="right"))
            if te_e > te_s:
                test_slices[tk] = TickerSlice(ticker=tk, start_idx=te_s, end_idx=te_e)

        plans.append(WindowPlan(
            train_slices=train_slices,
            test_slices=test_slices,
            train_start=tr_start,
            train_end=tr_end,
            test_start=te_start,
            test_end=te_end,
        ))

    return plans

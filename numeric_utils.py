#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pandas as pd


def _coerce_object_series_to_numeric(series: pd.Series) -> pd.Series:
    # fast path: already numeric
    if pd.api.types.is_numeric_dtype(series):
        out = series
    else:
        out = pd.to_numeric(series, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def to_float32_safe(obj):
    """Coerce strings/objects to numeric, normalize inf/-inf, then cast to float32."""
    if isinstance(obj, pd.Series):
        s = _coerce_object_series_to_numeric(obj)
        vals = np.array(s.to_numpy(dtype=np.float64, copy=False), dtype=np.float64, copy=True)
        vals[~np.isfinite(vals)] = np.nan
        return pd.Series(vals.astype(np.float32, copy=False), index=obj.index, name=obj.name)

    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return obj

        # Fast path when frame is already numeric-like
        if all(pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt) for dt in obj.dtypes):
            vals = np.array(obj.to_numpy(dtype=np.float64, copy=False), dtype=np.float64, copy=True)
            vals[~np.isfinite(vals)] = np.nan
            return pd.DataFrame(vals.astype(np.float32, copy=False), index=obj.index, columns=obj.columns)

        # Fallback for mixed dtypes (object/string columns)
        out = pd.DataFrame(index=obj.index)
        for col in obj.columns:
            s = pd.to_numeric(obj[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            vals = np.array(s.to_numpy(dtype=np.float64, copy=False), dtype=np.float64, copy=True)
            vals[~np.isfinite(vals)] = np.nan
            out[col] = vals.astype(np.float32, copy=False)
        return out

    # scalar / array-like
    arr = pd.to_numeric(obj, errors="coerce")
    if isinstance(arr, pd.Series):
        vals = np.array(arr.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64, copy=False), dtype=np.float64, copy=True)
        vals[~np.isfinite(vals)] = np.nan
        return pd.Series(vals.astype(np.float32, copy=False), index=arr.index, name=arr.name)

    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(np.float64, copy=False)
        arr[~np.isfinite(arr)] = np.nan
        return arr.astype(np.float32, copy=False)
    return arr


def normalize_before_save(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight save-time normalization: numeric inf cleanup + numeric-like object coercion."""
    if df is None or df.empty:
        return df

    # Keep this shallow: ETL frames can be very wide and a deep copy may require
    # another full-frame allocation just before saving.
    out = df.copy(deep=False)

    # numeric columns: only inf cleanup (cheap)
    num_cols = out.select_dtypes(include=[np.number, "bool", "boolean"]).columns
    for col in num_cols:
        s = out[col]
        vals = s.to_numpy(copy=False)
        if np.isfinite(vals).all():
            continue
        out[col] = pd.Series(np.where(np.isfinite(vals), vals, np.nan), index=out.index)

    # object/string columns: attempt numeric conversion only when mostly numeric-like
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        s = out[col]
        num = pd.to_numeric(s, errors="coerce")
        # convert only if strongly numeric-like; keeps true text columns intact
        ratio = float(num.notna().mean()) if len(num) else 0.0
        if ratio >= 0.98:
            out[col] = num.replace([np.inf, -np.inf], np.nan)

    return out

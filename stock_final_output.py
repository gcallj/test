#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated from notebook. Wrapped as callable function.
"""

import os
os.makedirs('./output/data', exist_ok=True)
os.makedirs('./output/data/models', exist_ok=True)
os.makedirs('./output', exist_ok=True)


def run_final_output():
    """Run the pipeline step."""
    # ============================================================
    # HISTORY (CHUNKED) — FIXED fund_score KeyError + EV simétrico
    # ============================================================
    # !pip -q install pyarrow fastparquet yfinance openpyxl


    import numpy as np
    import pandas as pd
    from pathlib import Path
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import shutil, gc, datetime as dt
    import time

    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from numeric_utils import normalize_before_save

    # ============================================================
    # PATHS
    # ============================================================
    HIST_ENS_PATH = Path("./output/ensemble_signals_history.parquet")
    HIST_REG_WIDE_PATH = Path("./output/forecast_history_wide.parquet")

    OUT_HIST_PARQUET = Path("./output/history_consolidated.parquet")

    # ============================================================
    # CHUNKING
    # ============================================================
    START_DATE = "2005-01-01"
    END_DATE   = "2026-12-31"
    CHUNK_DAYS = 60

    WRITE_PARQUET = True
    OVERWRITE_OUTPUTS = True

    # ============================================================
    # PARAMS
    # ============================================================
    UP20_THR = 0.55
    DD5_THR  = 0.55
    MIN_BUY_TRUST  = 0.15
    MIN_SELL_TRUST = 0.15

    UP20_PAYOFF = 0.20
    DD5_LOSS    = 0.05
    EV_W_REG    = 0.50

    ERR_CAP = 0.20
    MIN_DOWNSIDE_FOR_RR = 0.002
    RR_CAP = 50.0

    ALPHA_FUND = 0.50
    EPS = 1e-12



    # --- MENOS peso pro binário (pred bins) ---
    BIN_WEIGHT = 0.20   # 0.0 = ignora binário; 1.0 = só binário. Recomendo 0.10~0.25

    # --- MAIS peso pro risk_return ---
    RR_REG_MIX   = 0.65 # mistura do EV_reg com um EV derivado do risk_return (0..1)
    EV_W_REG_MIN = 0.65 # peso mínimo do REG no EV final (quando RR é ruim)
    EV_W_REG_MAX = 0.90 # peso máximo do REG no EV final (quando RR é ótimo)

    # --- Boost do RR nos EV_fund ---
    RR_BOOST_MIN   = 0.60  # multiplicador mínimo
    RR_BOOST_RANGE = 0.80  # +0.80 => max = 1.40 (0.60..1.40)

    # ============================================================
    # OUTPUT COLS
    # ============================================================
    FINAL_COLS = [
        "Date","ticker","split",
        "price","open","high","low","close",
        "best_buy_value","best_sell_value",
        "err_buy_pct","err_sell_pct",
        "downside_pct","upside_pct","risk_return",
        "p_up20","p_dd5","pred_up20","pred_dd5","up20_bin","dd5_bin",
        "buy_trust","sell_trust",
        "EV_buy_reg","EV_buy_ens","EV_buy",
        "fund_score","EV_buy_fund","EV_buy_fund_2","EV_buy_fund_3",
        "signal",
        "dividend_yield","trailing_pe","price_to_book","market_cap"
    ]

    # ============================================================
    # HELPERS
    # ============================================================
    def ensure_exists(path: Path, name: str):
        if not path.exists():
            raise FileNotFoundError(f"{name} não encontrado: {path}")

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def clip01(x):
        return np.clip(x, 0.0, 1.0)

    def norm_date_str(series):
        return pd.to_datetime(series, errors="coerce").dt.date.astype(str)

    def looks_like_equity_br(t: str) -> bool:
        t = str(t).upper().strip()
        return t.endswith(".SA") and ("=" not in t) and ("^" not in t) and ("-USD" not in t)

    def to_prob(series):
        s = series.astype(str).str.strip()
        s = s.str.replace("%", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        v = pd.to_numeric(s, errors="coerce")
        mx = v.max(skipna=True)
        if pd.notna(mx) and mx > 1.5:
            v = v / 100.0
        return v

    def pick_prob(df, candidates):
        for c in candidates:
            if c in df.columns:
                v = to_prob(df[c])
                if v.notna().mean() > 0.2 and v.abs().sum(skipna=True) > 0:
                    return c, v
        for c in candidates:
            if c in df.columns:
                return c, to_prob(df[c])
        return None, pd.Series(np.nan, index=df.index)

    def pick_pred(df, candidates):
        for c in candidates:
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                return c, v
        return None, pd.Series(np.nan, index=df.index)

    def arrow_date_scalar(date_str: str, pa_type):
        if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
            return date_str
        if pa.types.is_date32(pa_type) or pa.types.is_date64(pa_type):
            y, m, d = map(int, date_str.split("-"))
            return dt.date(y, m, d)
        if pa.types.is_timestamp(pa_type):
            return pd.Timestamp(date_str).to_pydatetime()
        return date_str

    def safe_to_pandas(table: pa.Table) -> pd.DataFrame:
        return table.to_pandas(split_blocks=True, self_destruct=True)

    # ============================================================
    # FUNDAMENTAIS (yfinance) — apenas .SA
    # ============================================================
    def fetch_one_fund(ticker: str):
        try:
            d = yf.Ticker(ticker).info
        except Exception:
            d = {}

        dy = d.get("dividendYield", np.nan)
        pe = d.get("trailingPE", np.nan)
        pb = d.get("priceToBook", np.nan)
        mc = d.get("marketCap", np.nan)

        dy = pd.to_numeric(pd.Series([dy]), errors="coerce").iloc[0]
        if pd.notna(dy):
            for _ in range(3):
                if dy > 1.2:
                    dy = dy / 100.0
            dy = float(np.clip(dy, 0.0, 1.0))

        pe = pd.to_numeric(pd.Series([pe]), errors="coerce").iloc[0]
        pb = pd.to_numeric(pd.Series([pb]), errors="coerce").iloc[0]
        mc = pd.to_numeric(pd.Series([mc]), errors="coerce").iloc[0]

        if pd.notna(pe) and pe <= 0: pe = np.nan
        if pd.notna(pb) and pb <= 0: pb = np.nan
        if pd.notna(mc) and mc <= 0: mc = np.nan

        return {
            "ticker": str(ticker).upper().strip(),
            "dividend_yield": dy,
            "trailing_pe": pe,
            "price_to_book": pb,
            "market_cap": mc,
        }
    def preload_yahoo_history(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
        print(f"[yahoo] Baixando OHLC de {len(tickers)} ativos ({start_date} a {end_date})...")

        sd = pd.to_datetime(start_date) - pd.Timedelta(days=15)
        ed = pd.to_datetime(end_date) + pd.Timedelta(days=5)

        try:
            # auto_adjust=False traz: Open, High, Low, Close, Adj Close, Volume
            data = yf.download(tickers, start=sd, end=ed, auto_adjust=False, progress=True, threads=True)

            if data.empty:
                return pd.DataFrame(columns=["Date", "ticker", "open", "high", "low", "close"])

            # Seleciona apenas as colunas de interesse
            # O yfinance retorna MultiIndex nas colunas: (Price, Ticker)
            target_cols = ["Open", "High", "Low", "Close"]

            # Verifica se as colunas existem (pode variar se baixou só 1 ticker ou vários)
            available_cols = [c for c in target_cols if c in data.columns.get_level_values(0)]
            if not available_cols:
                return pd.DataFrame(columns=["Date", "ticker", "open", "high", "low", "close"])

            df_slice = data[available_cols].copy()

            # O .stack() move o nível 'Ticker' das colunas para o índice das linhas
            # Resultado Index: (Date, Ticker) | Colunas: Open, High, Low, Close
            df_stacked = df_slice.stack(level=1, future_stack=True)

            # Se o future_stack falhar na sua versão, use: df_stacked = df_slice.stack()

            df_reset = df_stacked.reset_index()

            # Renomear para minúsculo para manter padrão do seu código
            df_reset.rename(columns={
                "Date": "Date",
                "Ticker": "ticker",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close"
            }, inplace=True)

            # Normalização final
            df_reset["Date"] = norm_date_str(df_reset["Date"])
            df_reset["ticker"] = df_reset["ticker"].astype(str).str.upper().str.strip()

            for c in ["open", "high", "low", "close"]:
                df_reset[c] = pd.to_numeric(df_reset[c], errors="coerce")

            print(f"[yahoo] Download concluído. Linhas carregadas: {len(df_reset)}")
            return df_reset[["Date", "ticker", "open", "high", "low", "close"]]

        except Exception as e:
            print(f"[yahoo] Erro crítico no download OHLC: {e}")
            return pd.DataFrame(columns=["Date", "ticker", "open", "high", "low", "close"])


    def build_fund_table(tickers, max_workers=8):
        tickers = [t for t in tickers if looks_like_equity_br(t)]
        tickers = sorted(set([str(t).upper().strip() for t in tickers]))
        if len(tickers) == 0:
            return pd.DataFrame(columns=["ticker","dividend_yield","trailing_pe","price_to_book","market_cap"])

        rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fetch_one_fund, t): t for t in tickers}
            n = len(futs)
            for i, fut in enumerate(as_completed(futs), start=1):
                if i % 25 == 0 or i == n:
                    print(f"[fund] {i}/{n} tickers processados")
                try:
                    rows.append(fut.result())
                except Exception:
                    pass

        fund = pd.DataFrame(rows)
        for c in ["dividend_yield","trailing_pe","price_to_book","market_cap"]:
            if c in fund.columns:
                fund[c] = pd.to_numeric(fund[c], errors="coerce")
        fund["ticker"] = fund["ticker"].astype(str).str.upper().str.strip()
        fund = fund.drop_duplicates("ticker", keep="last").reset_index(drop=True)
        return fund

    def pct_rank_score(series: pd.Series, higher_is_better: bool):
        x = pd.to_numeric(series, errors="coerce")
        r = x.rank(pct=True, ascending=True)
        return (r if higher_is_better else (1.0 - r))

    def compute_fund_score_per_ticker(fund_df: pd.DataFrame) -> pd.DataFrame:
        if fund_df is None or fund_df.empty:
            return pd.DataFrame(columns=["ticker","fund_score"])

        fu = fund_df.copy()
        fu["ticker"] = fu["ticker"].astype(str).str.upper().str.strip()

        dy_s = pct_rank_score(fu["dividend_yield"], higher_is_better=True)   # DY ↑
        pe_s = pct_rank_score(fu["trailing_pe"],     higher_is_better=False) # PE ↓
        pb_s = pct_rank_score(fu["price_to_book"],   higher_is_better=False) # PB ↓
        mc_s = pct_rank_score(fu["market_cap"],      higher_is_better=True)  # MC ↑

        scores = pd.concat([dy_s, pe_s, pb_s, mc_s], axis=1)
        scores.columns = ["s_dy","s_pe","s_pb","s_mc"]

        w = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
        mask = scores.notna().astype(float).values
        w_eff = (mask * w).sum(axis=1)
        num = (scores.fillna(0.0).values * w).sum(axis=1)

        fu["fund_score"] = np.where(w_eff > 0, num / np.maximum(w_eff, EPS), 0.5)
        fu["fund_score"] = np.clip(fu["fund_score"], 0.0, 1.0)
        return fu[["ticker","fund_score"]]

    def ensure_fund_score_column(fund_all: pd.DataFrame) -> pd.DataFrame:
        if fund_all is None or fund_all.empty:
            return pd.DataFrame(columns=["ticker","dividend_yield","trailing_pe","price_to_book","market_cap","fund_score"])

        fa = fund_all.copy()
        if "ticker" in fa.columns:
            fa["ticker"] = fa["ticker"].astype(str).str.upper().str.strip()

        if "fund_score" not in fa.columns:
            base_cols = ["dividend_yield","trailing_pe","price_to_book","market_cap"]
            if all(c in fa.columns for c in base_cols):
                fs = compute_fund_score_per_ticker(fa[["ticker"] + base_cols].copy())
                fa = fa.merge(fs, on="ticker", how="left")
            else:
                fa["fund_score"] = np.nan

        fa["fund_score"] = to_num(fa["fund_score"]).fillna(0.5).clip(0.0, 1.0)
        for c in ["dividend_yield","trailing_pe","price_to_book","market_cap"]:
            if c not in fa.columns:
                fa[c] = np.nan
        return fa[["ticker","dividend_yield","trailing_pe","price_to_book","market_cap","fund_score"]].copy()

    # ============================================================
    # REG (WIDE)
    # ============================================================
    def prepare_regression_wide(reg_wide: pd.DataFrame) -> pd.DataFrame:
        df = reg_wide.copy()
        df.columns = df.columns.astype(str).str.strip()

        if "date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"date":"Date"})
        if "Ticker" in df.columns and "ticker" not in df.columns:
            df = df.rename(columns={"Ticker":"ticker"})

        need_min = ["Date","ticker","best_buy_price","best_sell_price"]
        for c in need_min:
            if c not in df.columns:
                raise ValueError(f"REG_WIDE precisa ter '{c}'. Colunas atuais: {list(df.columns)[:50]}")

        df["Date"] = norm_date_str(df["Date"])
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

        df["best_buy_value"]  = to_num(df["best_buy_price"])
        df["best_sell_value"] = to_num(df["best_sell_price"])

        if "buy_err_lo" in df.columns and "buy_err_hi" in df.columns:
            buy_lo = to_num(df["buy_err_lo"])
            buy_hi = to_num(df["buy_err_hi"])
            buy_c  = df["best_buy_value"]
            err_buy_abs = np.maximum((buy_c - buy_lo).abs(), (buy_hi - buy_c).abs())
            df["err_buy_pct"] = err_buy_abs / np.maximum(buy_c.abs(), EPS)
        else:
            df["err_buy_pct"] = np.nan

        if "sell_err_lo" in df.columns and "sell_err_hi" in df.columns:
            sell_lo = to_num(df["sell_err_lo"])
            sell_hi = to_num(df["sell_err_hi"])
            sell_c  = df["best_sell_value"]
            err_sell_abs = np.maximum((sell_c - sell_lo).abs(), (sell_hi - sell_c).abs())
            df["err_sell_pct"] = err_sell_abs / np.maximum(sell_c.abs(), EPS)
        else:
            df["err_sell_pct"] = np.nan

        df["err_buy_pct"]  = to_num(df["err_buy_pct"]).abs().fillna(ERR_CAP).clip(0, 1)
        df["err_sell_pct"] = to_num(df["err_sell_pct"]).abs().fillna(ERR_CAP).clip(0, 1)

        e1 = df["err_buy_pct"].fillna(ERR_CAP)
        e2 = df["err_sell_pct"].fillna(ERR_CAP)
        w1 = 1.0 / np.maximum(e1, 1e-6)
        w2 = 1.0 / np.maximum(e2, 1e-6)
        est = (df["best_buy_value"] * w1 + df["best_sell_value"] * w2) / np.maximum((w1 + w2), EPS)
        est = est.fillna((df["best_buy_value"] + df["best_sell_value"]) / 2.0)
        est = est.fillna(df["best_buy_value"]).fillna(df["best_sell_value"])
        df["price"] = to_num(est)

        return df[["Date","ticker","price","best_buy_value","best_sell_value","err_buy_pct","err_sell_pct"]].copy()

    # ============================================================
    # ENS normalize
    # ============================================================
    def normalize_ensemble(ens: pd.DataFrame) -> pd.DataFrame:
        df = ens.copy()
        df.columns = df.columns.astype(str).str.strip()

        if "date" in df.columns and "Date" not in df.columns:
            df = df.rename(columns={"date":"Date"})
        if "Ticker" in df.columns and "ticker" not in df.columns:
            df = df.rename(columns={"Ticker":"ticker"})

        for col in ["Date","ticker"]:
            if col not in df.columns:
                raise ValueError(f"ENS precisa ter '{col}'. Colunas atuais: {list(df.columns)[:50]}")

        df["Date"] = norm_date_str(df["Date"])
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

        if "split" not in df.columns:
            df["split"] = np.nan
        else:
            df["split"] = df["split"].astype(str)

        _, up_v = pick_prob(df, ["p_up20"])
        _, dd_v = pick_prob(df, ["p_dd5"])
        df["p_up20"] = up_v.clip(0, 1)
        df["p_dd5"]  = dd_v.clip(0, 1)

        pred_up_col, pred_up_v = pick_pred(df, ["pred_up20"])
        pred_dd_col, pred_dd_v = pick_pred(df, ["pred_dd5"])

        if pred_up_col is not None:
            df["pred_up20"] = pd.to_numeric(pred_up_v, errors="coerce")
            up_bin = (pred_up_v.fillna(0) > 0).astype(int)
        else:
            up_bin = (df["p_up20"].fillna(0) >= UP20_THR).astype(int)
            df["pred_up20"] = up_bin.astype(float)

        if pred_dd_col is not None:
            df["pred_dd5"] = pd.to_numeric(pred_dd_v, errors="coerce")
            dd_bin = (pred_dd_v.fillna(0) > 0).astype(int)
        else:
            dd_bin = (df["p_dd5"].fillna(0) >= DD5_THR).astype(int)
            df["pred_dd5"] = dd_bin.astype(float)

        df["up20_bin"] = up_bin
        df["dd5_bin"]  = dd_bin

        return df[["Date","ticker","split","p_up20","p_dd5","pred_up20","pred_dd5","up20_bin","dd5_bin"]].copy()

    # ============================================================
    # MERGE + METRICS
    # ============================================================
    def merge_reg_ens(reg_p: pd.DataFrame, ens_k: pd.DataFrame) -> pd.DataFrame:
        df = reg_p.merge(ens_k, on=["Date","ticker"], how="left")
        df["split"] = df.get("split", np.nan)

        df["p_up20"] = to_num(df.get("p_up20", 0)).fillna(0.0).clip(0, 1)
        df["p_dd5"]  = to_num(df.get("p_dd5", 0)).fillna(0.0).clip(0, 1)
        df["pred_up20"] = to_num(df.get("pred_up20", np.nan))
        df["pred_dd5"]  = to_num(df.get("pred_dd5", np.nan))
        df["up20_bin"] = to_num(df.get("up20_bin", 0)).fillna(0).astype(int)
        df["dd5_bin"]  = to_num(df.get("dd5_bin", 0)).fillna(0).astype(int)
        return df

    def compute_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["err_buy_pct"]  = to_num(df["err_buy_pct"]).abs().fillna(ERR_CAP).clip(0, 1)
        df["err_sell_pct"] = to_num(df["err_sell_pct"]).abs().fillna(ERR_CAP).clip(0, 1)

        buy_quality  = clip01(1.0 - (df["err_buy_pct"]  / ERR_CAP))
        sell_quality = clip01(1.0 - (df["err_sell_pct"] / ERR_CAP))

        p_up20 = to_num(df["p_up20"]).fillna(0.0).clip(0, 1)
        p_dd5  = to_num(df["p_dd5"]).fillna(0.0).clip(0, 1)

        pred_up_bin = (to_num(df.get("pred_up20", np.nan)).fillna(0) > 0).astype(int)
        pred_dd_bin = (to_num(df.get("pred_dd5",  np.nan)).fillna(0) > 0).astype(int)

        p_up20_eff = clip01((1.0 - BIN_WEIGHT) * p_up20 + BIN_WEIGHT * pred_up_bin)
        p_dd5_eff  = clip01((1.0 - BIN_WEIGHT) * p_dd5  + BIN_WEIGHT * pred_dd_bin)


        df["buy_trust"]  = clip01(p_up20_eff * (1.0 - p_dd5_eff) * buy_quality)
        df["sell_trust"] = clip01(p_dd5_eff * sell_quality)

        price = to_num(df["price"])
        buyv  = to_num(df["best_buy_value"])
        sellv = to_num(df["best_sell_value"])

        df["downside_pct"] = ((price - buyv) / np.maximum(price, EPS)).where((price > 0) & (buyv < price), 0.0)
        df["upside_pct"]   = ((sellv - price) / np.maximum(price, EPS)).where((price > 0) & (sellv > price), 0.0)

        down = to_num(df["downside_pct"]).clip(lower=0.0)
        up   = to_num(df["upside_pct"]).clip(lower=0.0)

        den = np.maximum(down, MIN_DOWNSIDE_FOR_RR)
        rr = np.where(up <= 0, 0.0, up / np.maximum(den, EPS))
        df["risk_return"] = np.clip(rr, 0.0, RR_CAP)

        # ============================================================
        # EV BUY/SELL SIMÉTRICO (índice único em [-1, +1])
        #   Agora: EV_reg recebe componente do risk_return
        # ============================================================
        buy_ev  = (buy_quality  * up).astype(float)
        sell_ev = (sell_quality * down).astype(float)

        # EV_reg "base" (diferença normalizada)
        ev_reg_base = ((buy_ev - sell_ev) / (buy_ev + sell_ev + EPS)).astype(float)
        ev_reg_base = np.clip(ev_reg_base, -1.0, 1.0)

        # ---- score do risk_return em 0..1 (log pra não explodir com RR alto)
        rr_clip  = np.clip(to_num(df["risk_return"]).fillna(0.0).values, 0.0, RR_CAP)
        rr_score = (np.log1p(rr_clip) / np.log1p(RR_CAP + EPS)).astype(float)  # 0..1

        # EV vindo do RR (simétrico):
        #  - se lado BUY (buy_ev >= sell_ev): usa +rr_score
        #  - se lado SELL: usa -(1-rr_score) (forte sell quando RR é ruim)
        sign_reg = np.sign((buy_ev - sell_ev).values)
        ev_rr = np.where(sign_reg >= 0, rr_score, -(1.0 - rr_score)).astype(float)
        ev_rr = np.clip(ev_rr, -1.0, 1.0)

        # mistura: dá MAIS importância ao RR dentro do REG
        ev_reg_rr = (1.0 - RR_REG_MIX) * ev_reg_base + RR_REG_MIX * ev_rr
        df["EV_buy_reg"] = np.clip(ev_reg_rr, -1.0, 1.0)

        # Ensemble simétrico (menos binário por causa do BIN_WEIGHT acima)
        df["EV_buy_ens"] = np.clip((p_up20_eff - p_dd5_eff).astype(float), -1.0, 1.0)

        # peso do REG cresce com RR (mais RR => menos dependência do ensemble/binário)
        w_reg = EV_W_REG_MIN + (EV_W_REG_MAX - EV_W_REG_MIN) * rr_score
        df["EV_buy"] = np.clip(w_reg * df["EV_buy_reg"].values + (1.0 - w_reg) * df["EV_buy_ens"].values, -1.0, 1.0)


        return df

    def add_fund_and_3_evfund(df: pd.DataFrame, fund_all: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for c in ["dividend_yield","trailing_pe","price_to_book","market_cap","fund_score"]:
            if c not in df.columns:
                df[c] = np.nan

        fund_all = ensure_fund_score_column(fund_all)

        if fund_all is not None and not fund_all.empty:
            df = df.merge(
                fund_all[["ticker","dividend_yield","trailing_pe","price_to_book","market_cap","fund_score"]],
                on="ticker",
                how="left",
                suffixes=("","_fund")
            )
            for c in ["dividend_yield","trailing_pe","price_to_book","market_cap","fund_score"]:
                cf = f"{c}_fund"
                if cf in df.columns:
                    df[c] = to_num(df[cf]).combine_first(to_num(df[c]))
                    df.drop(columns=[cf], inplace=True)

        df["fund_score"] = to_num(df["fund_score"]).fillna(0.5).clip(0, 1)

        ev = to_num(df["EV_buy"]).fillna(0.0).astype(float).clip(-1.0, 1.0)
        fs = to_num(df["fund_score"]).fillna(0.5).astype(float).clip(0.0, 1.0)
        # ---- score do risk_return em 0..1 (log)
        rr_clip  = np.clip(to_num(df.get("risk_return", 0.0)).fillna(0.0).values, 0.0, RR_CAP)
        rr_score = (np.log1p(rr_clip) / np.log1p(RR_CAP + EPS)).astype(float)  # 0..1

        # boost simétrico:
        #  - BUY (ev>=0): boost cresce com rr_score
        #  - SELL (ev<0): boost cresce com (1-rr_score)
        rr_w = np.where(ev.values >= 0, rr_score, (1.0 - rr_score)).astype(float)
        rr_boost = (RR_BOOST_MIN + RR_BOOST_RANGE * rr_w).astype(float)  # ex: 0.60..1.40

        # fund_1 (SIMÉTRICO)
        # fund_1 (SIMÉTRICO) + RR boost
        scale = 1.0 + float(ALPHA_FUND) * (fs - 0.5) * np.sign(ev)
        scale = np.clip(scale, 0.2, 2.0)
        df["EV_buy_fund"] = np.clip((ev.values * scale.values) * rr_boost, -1.0, 1.0)

        # fund_2 (SIMÉTRICO) + RR boost
        base2 = (ev.values + float(ALPHA_FUND) * (fs.values - 0.5) * np.abs(ev.values))
        df["EV_buy_fund_2"] = np.clip(base2 * rr_boost, -1.0, 1.0)


        # -----------------------------
        # FIX: definir p_up20_f / p_dd5_f AQUI
        # -----------------------------
        p_up20 = to_num(df.get("p_up20", 0.0)).fillna(0.0).clip(0.0, 1.0)
        p_dd5  = to_num(df.get("p_dd5",  0.0)).fillna(0.0).clip(0.0, 1.0)

        pred_up_bin = (to_num(df.get("pred_up20", np.nan)).fillna(0) > 0).astype(int)
        pred_dd_bin = (to_num(df.get("pred_dd5",  np.nan)).fillna(0) > 0).astype(int)

        p_up20_eff = clip01((1.0 - BIN_WEIGHT) * p_up20 + BIN_WEIGHT * pred_up_bin)
        p_dd5_eff  = clip01((1.0 - BIN_WEIGHT) * p_dd5  + BIN_WEIGHT * pred_dd_bin)


        up_scale = (0.8 + 0.4 * fs)
        dd_scale = (1.2 - 0.4 * fs)

        p_up20_f = clip01(p_up20_eff * up_scale)
        p_dd5_f  = clip01(p_dd5_eff  * dd_scale)

        ens_sym_f = (p_up20_f - p_dd5_f).astype(float).clip(-1.0, 1.0)
        ev_reg    = to_num(df["EV_buy_reg"]).fillna(0.0).astype(float).clip(-1.0, 1.0)

        w_reg = EV_W_REG_MIN + (EV_W_REG_MAX - EV_W_REG_MIN) * rr_score
        df["EV_buy_fund_3"] = np.clip(w_reg * ev_reg.values + (1.0 - w_reg) * ens_sym_f.values, -1.0, 1.0)


        return df

    def finalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in FINAL_COLS:
            if c not in out.columns:
                out[c] = np.nan
        out = out[FINAL_COLS].copy()

        num_cols = [
            "price","open","high","low","close","best_buy_value","best_sell_value","err_buy_pct","err_sell_pct",
            "downside_pct","upside_pct","risk_return",
            "p_up20","p_dd5","pred_up20","pred_dd5",
            "buy_trust","sell_trust",
            "EV_buy_reg","EV_buy_ens","EV_buy",
            "fund_score","EV_buy_fund","EV_buy_fund_2","EV_buy_fund_3",
            "dividend_yield","trailing_pe","price_to_book","market_cap",
        ]
        for c in num_cols:
            if c in out.columns:
                out[c] = to_num(out[c]).round(6)
        return out

    # ============================================================
    # SCAN tickers .SA (streaming)
    # ============================================================
    def open_parquet_dataset_with_retry(path, format="parquet", retries=4, sleep_s=2.0):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                return ds.dataset(str(path), format=format)
            except Exception as e:
                last_exc = e
                print(f"[retry] erro abrindo dataset {path} (tentativa {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(sleep_s * attempt)
        raise last_exc


    def dataset_to_table_with_retry(dataset_obj: ds.Dataset, columns=None, filter_expr=None, retries=4, sleep_s=1.5):
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                if filter_expr is None:
                    return dataset_obj.to_table(columns=columns)
                return dataset_obj.to_table(columns=columns, filter=filter_expr)
            except Exception as e:
                last_exc = e
                print(f"[retry] erro lendo dataset (tentativa {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(sleep_s * attempt)
        raise last_exc


    def collect_sa_tickers_from_reg_dataset(reg_dataset: ds.Dataset, ticker_col="ticker", batch_size=500_000, retries=4) -> list:
        tickers_sa = set()
        if ticker_col not in reg_dataset.schema.names:
            return []

        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                scanner = reg_dataset.scanner(columns=[ticker_col], batch_size=batch_size)
                for batch in scanner.to_batches():
                    arr = batch.column(0).to_pylist()
                    for t in arr:
                        if t is None:
                            continue
                        ts = str(t).upper().strip()
                        if looks_like_equity_br(ts):
                            tickers_sa.add(ts)
                    del batch, arr
                return sorted(tickers_sa)
            except Exception as e:
                last_exc = e
                print(f"[retry] erro no scan de tickers .SA (tentativa {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(1.5 * attempt)

        raise last_exc

    # ============================================================
    # WINDOW GENERATOR
    # ============================================================
    def date_windows(start_date: str, end_date: str, chunk_days: int):
        start = pd.Timestamp(start_date)
        end   = pd.Timestamp(end_date)
        cur = start
        while cur < end:
            nxt = min(cur + pd.Timedelta(days=chunk_days), end)
            yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
            cur = nxt

    # ============================================================
    # MAIN (CHUNKED)
    # ============================================================
    ensure_exists(HIST_ENS_PATH, "HIST_ENS_PATH")
    ensure_exists(HIST_REG_WIDE_PATH, "HIST_REG_WIDE_PATH")

    ens_ds = open_parquet_dataset_with_retry(HIST_ENS_PATH, format="parquet")
    reg_ds = open_parquet_dataset_with_retry(HIST_REG_WIDE_PATH, format="parquet")

    ens_date_type = ens_ds.schema.field("Date").type if "Date" in ens_ds.schema.names else None
    reg_date_type = reg_ds.schema.field("Date").type if "Date" in reg_ds.schema.names else None

    if OVERWRITE_OUTPUTS and OUT_HIST_PARQUET.exists() and WRITE_PARQUET:
        OUT_HIST_PARQUET.unlink()

    print("[scan] coletando tickers .SA do REG (streaming)...")
    try:
        tickers_sa = collect_sa_tickers_from_reg_dataset(reg_ds, ticker_col="ticker", batch_size=500_000)
    except Exception as e:
        print(f"[warn] falha no scan inicial do REG: {e}. Reabrindo dataset e tentando novamente...")
        reg_ds = open_parquet_dataset_with_retry(HIST_REG_WIDE_PATH, format="parquet")
        tickers_sa = collect_sa_tickers_from_reg_dataset(reg_ds, ticker_col="ticker", batch_size=500_000)
    tickers_sa_set = set(tickers_sa)
    print(f"[scan] tickers .SA únicos: {len(tickers_sa)}")

    fund_all = pd.DataFrame(columns=["ticker","dividend_yield","trailing_pe","price_to_book","market_cap","fund_score"])

    if len(tickers_sa) > 0:
        fund = build_fund_table(tickers_sa, max_workers=8)
        if not fund.empty:
            fs = compute_fund_score_per_ticker(fund)
            fund_all = fund.merge(fs, on="ticker", how="left")
            fund_all = ensure_fund_score_column(fund_all)
            print("[fund] ok — fundamentos carregados para .SA")
        else:
            fund_all = ensure_fund_score_column(fund_all)
            print("[fund] nenhum .SA retornou fundamentos (fund_score neutro=0.5)")
    else:
        fund_all = ensure_fund_score_column(fund_all)
        print("[fund] nenhum .SA no histórico (fund_score neutro=0.5)")

    ens_cols = [c for c in ["Date","ticker","split","p_up20","p_dd5","pred_up20","pred_dd5"] if c in ens_ds.schema.names]
    reg_cols = [c for c in ["Date","ticker","best_buy_price","buy_err_lo","buy_err_hi","best_sell_price","sell_err_lo","sell_err_hi"] if c in reg_ds.schema.names]

    print(f"[cols] ens_cols={ens_cols}")
    print(f"[cols] reg_cols={reg_cols}")

    parquet_writer = None
    all_closes_df = pd.DataFrame()
    if len(tickers_sa) > 0:
        # Baixa de 2005 a 2026 de uma vez só
        all_closes_df = preload_yahoo_history(tickers_sa, START_DATE, END_DATE)
    rows_total = 0

    for s, e in date_windows(START_DATE, END_DATE, CHUNK_DAYS):
        ens_f = (ds.field("Date") >= arrow_date_scalar(s, ens_date_type)) & (ds.field("Date") < arrow_date_scalar(e, ens_date_type)) if ens_date_type is not None else None
        reg_f = (ds.field("Date") >= arrow_date_scalar(s, reg_date_type)) & (ds.field("Date") < arrow_date_scalar(e, reg_date_type)) if reg_date_type is not None else None

        reg_tbl = dataset_to_table_with_retry(reg_ds, columns=reg_cols, filter_expr=reg_f)
        if reg_tbl.num_rows == 0:
            del reg_tbl
            continue

        reg_chunk = safe_to_pandas(reg_tbl)
        del reg_tbl
        reg_p = prepare_regression_wide(reg_chunk)
        reg_p = reg_p[reg_p["ticker"].isin(tickers_sa_set)]
        del reg_chunk
        if reg_p.empty:
            del reg_p
            continue

        ens_tbl = dataset_to_table_with_retry(ens_ds, columns=ens_cols, filter_expr=ens_f)
        if ens_tbl.num_rows > 0:
            ens_chunk = safe_to_pandas(ens_tbl)
            ens_k = normalize_ensemble(ens_chunk)
            del ens_chunk
        else:
            ens_k = pd.DataFrame(columns=["Date","ticker","split","p_up20","p_dd5","pred_up20","pred_dd5","up20_bin","dd5_bin"])
        del ens_tbl

        df = merge_reg_ens(reg_p, ens_k)
        del reg_p, ens_k
        if not all_closes_df.empty:
            # Filtra fatia de data
            mask_ohlc = (all_closes_df["Date"] >= s) & (all_closes_df["Date"] < e)
            chunk_ohlc = all_closes_df[mask_ohlc]

            # Merge com todas as colunas (open, high, low, close)
            df = df.merge(chunk_ohlc, on=["Date", "ticker"], how="left")

        # Garante que as colunas existam mesmo se o merge falhar (preenche NaN)
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                df[col] = np.nan

        if "close" not in df.columns:
            df["close"] = np.nan

        df = df.dropna(subset=["open", "high", "low", "close"]).copy()
        if df.empty:
            continue

        df = compute_core_metrics(df)
        df = add_fund_and_3_evfund(df, fund_all)
        df["price"] = df["close"]
        out = finalize_chunk(df)
        del df

        n = len(out)
        rows_total += n
        print(f"[chunk] {s} -> {e} | rows={n:,} | total={rows_total:,}")

        if WRITE_PARQUET:
            out["Date"] = out["Date"].astype("string")
            out["ticker"] = out["ticker"].astype("string")
            if "split" in out.columns:
                out["split"] = out["split"].astype("string").fillna("")

            out = normalize_before_save(out)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(str(OUT_HIST_PARQUET), table.schema, compression="snappy")
            parquet_writer.write_table(table)
            del table

        del out
        gc.collect()

    if parquet_writer is not None:
        parquet_writer.close()

    print(f"[OK] Parquet salvo em: {OUT_HIST_PARQUET} | rows={rows_total:,}" if WRITE_PARQUET else "[OK] Parquet desativado")
    print("[DONE]")


    import pandas as pd

    OUT_HIST_PARQUET = "./output/history_consolidated.parquet"

    N = 20
    df_tail = pd.read_parquet(OUT_HIST_PARQUET).tail(N)
    print(df_tail.to_string(index=False))


    # ALTERNATIVA 2: DIAGNÓSTICO DO SCORE
    #
    # OBJETIVO: Descobrir se EV_buy_fund_3 realmente tem poder preditivo
    #
    # Testes:
    # 1. Correlação Score × Retorno Futuro
    # 2. Quintis de Score × Performance
    # 3. Information Coefficient (IC)
    # 4. Comparação Long-Only vs Long-Short

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr


    # ==============================================================================
    # CONFIG
    # ==============================================================================
    HISTORY_PARQUET_PATH = "./output/history_consolidated.parquet"
    SCORE_COL = "EV_buy_fund_3"
    ONLY_SA = True

    # ==============================================================================
    # 1. CORRELAÇÃO SCORE × RETORNO FUTURO
    # ==============================================================================
    def test_forward_correlation(df: pd.DataFrame, horizons=[5, 10, 20, 60]):
        """Testa se score prevê retornos futuros"""
        print("\n" + "="*80)
        print("TESTE 1: CORRELAÇÃO SCORE × RETORNO FUTURO")
        print("="*80)

        results = []

        for horizon in horizons:
            df[f"fwd_ret_{horizon}"] = df.groupby("ticker")["price"].pct_change(horizon).shift(-horizon)

            valid = df[[SCORE_COL, f"fwd_ret_{horizon}"]].dropna()

            if len(valid) < 100:
                continue

            # Pearson e Spearman
            pears_r, pears_p = pearsonr(valid[SCORE_COL], valid[f"fwd_ret_{horizon}"])
            spear_r, spear_p = spearmanr(valid[SCORE_COL], valid[f"fwd_ret_{horizon}"])

            results.append({
                "horizon": horizon,
                "pearson_r": pears_r,
                "pearson_pval": pears_p,
                "spearman_r": spear_r,
                "spearman_pval": spear_p,
                "n_samples": len(valid)
            })

            print(f"\nHorizonte: {horizon} dias")
            print(f"  Pearson  r={pears_r:.4f} (p={pears_p:.4f})")
            print(f"  Spearman r={spear_r:.4f} (p={spear_p:.4f})")
            print(f"  N={len(valid):,}")

        return pd.DataFrame(results)

    # ==============================================================================
    # 2. ANÁLISE POR QUINTIS
    # ==============================================================================
    def test_quintiles(df: pd.DataFrame, horizon=20):
        """Divide score em 5 grupos e compara performance"""
        print("\n" + "="*80)
        print(f"TESTE 2: PERFORMANCE POR QUINTIL (Horizon={horizon}d)")
        print("="*80)

        df[f"fwd_ret_{horizon}"] = df.groupby("ticker")["price"].pct_change(horizon).shift(-horizon)

        valid = df[[SCORE_COL, f"fwd_ret_{horizon}"]].dropna()

        # Cria quintis
        valid["quintile"] = pd.qcut(valid[SCORE_COL], q=5, labels=["Q1_Worst", "Q2", "Q3", "Q4", "Q5_Best"])

        # Agrupa por quintil
        summary = valid.groupby("quintile")[f"fwd_ret_{horizon}"].agg([
            ("mean_return", "mean"),
            ("median_return", "median"),
            ("std", "std"),
            ("count", "count")
        ]).round(4)

        print("\n" + summary.to_string())

        # Long-Short: Q5 - Q1
        q5 = valid[valid["quintile"] == "Q5_Best"][f"fwd_ret_{horizon}"].mean()
        q1 = valid[valid["quintile"] == "Q1_Worst"][f"fwd_ret_{horizon}"].mean()
        ls = q5 - q1

        print(f"\n📊 LONG-SHORT (Q5 - Q1): {ls*100:.3f}%")

        if ls > 0:
            print("✅ Score tem poder preditivo POSITIVO")
        else:
            print("❌ Score NÃO tem poder preditivo (ou é inverso!)")

        return summary

    # ==============================================================================
    # 3. INFORMATION COEFFICIENT (IC)
    # ==============================================================================
    def test_ic(df: pd.DataFrame, rolling_window=60):
        """Calcula IC ao longo do tempo"""
        print("\n" + "="*80)
        print("TESTE 3: INFORMATION COEFFICIENT (IC) ROLLING")
        print("="*80)

        df["fwd_ret_20"] = df.groupby("ticker")["price"].pct_change(20).shift(-20)

        # Calcula IC por data
        ic_by_date = []

        for date, group in df.groupby("Date"):
            valid = group[[SCORE_COL, "fwd_ret_20"]].dropna()

            if len(valid) < 10:
                continue

            ic, _ = spearmanr(valid[SCORE_COL], valid["fwd_ret_20"])

            ic_by_date.append({
                "Date": date,
                "IC": ic,
                "n_stocks": len(valid)
            })

        ic_df = pd.DataFrame(ic_by_date)
        ic_df["IC_MA"] = ic_df["IC"].rolling(rolling_window).mean()

        # Estatísticas
        mean_ic = ic_df["IC"].mean()
        std_ic = ic_df["IC"].std()
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0

        print(f"\nMean IC: {mean_ic:.4f}")
        print(f"Std IC:  {std_ic:.4f}")
        print(f"IC IR:   {ic_ir:.4f} (quanto maior, melhor)")
        print(f"IC > 0:  {(ic_df['IC'] > 0).sum() / len(ic_df) * 100:.1f}% das vezes")

        if mean_ic > 0.02:
            print("✅ IC consistentemente positivo - score funciona")
        elif mean_ic > 0:
            print("⚠️ IC fraco mas positivo - score tem algum poder")
        else:
            print("❌ IC negativo ou zero - score NÃO funciona")

        return ic_df

    # ==============================================================================
    # 4. LONG vs SHORT vs MARKET
    # ==============================================================================
    def test_long_short_strategies(df: pd.DataFrame, top_pct=0.2):
        """Compara Long-Only, Short-Only e Long-Short"""
        print("\n" + "="*80)
        print(f"TESTE 4: LONG vs SHORT (Top/Bottom {top_pct*100:.0f}%)")
        print("="*80)

        df["fwd_ret_20"] = df.groupby("ticker")["price"].pct_change(20).shift(-20)

        results_by_date = []

        for date, group in df.groupby("Date"):
            valid = group[[SCORE_COL, "fwd_ret_20"]].dropna()

            if len(valid) < 20:
                continue

            # Top e Bottom
            n_select = max(1, int(len(valid) * top_pct))

            top_stocks = valid.nlargest(n_select, SCORE_COL)
            bottom_stocks = valid.nsmallest(n_select, SCORE_COL)

            long_ret = top_stocks["fwd_ret_20"].mean()
            short_ret = bottom_stocks["fwd_ret_20"].mean()
            market_ret = valid["fwd_ret_20"].mean()
            ls_ret = long_ret - short_ret

            results_by_date.append({
                "Date": date,
                "Long": long_ret,
                "Short": short_ret,
                "Market": market_ret,
                "LongShort": ls_ret
            })

        strat_df = pd.DataFrame(results_by_date)

        # Agregados
        summary = strat_df[["Long", "Short", "Market", "LongShort"]].mean() * 100

        print("\nRETORNO MÉDIO (20 dias):")
        print(f"  Long (Top {top_pct*100:.0f}%):     {summary['Long']:.3f}%")
        print(f"  Short (Bottom {top_pct*100:.0f}%): {summary['Short']:.3f}%")
        print(f"  Market (All):       {summary['Market']:.3f}%")
        print(f"  Long-Short:         {summary['LongShort']:.3f}%")

        # Sharpe
        sharpe_long = (strat_df["Long"].mean() / strat_df["Long"].std()) * np.sqrt(252/20)
        sharpe_ls = (strat_df["LongShort"].mean() / strat_df["LongShort"].std()) * np.sqrt(252/20)

        print(f"\nSHARPE (anualizado):")
        print(f"  Long:       {sharpe_long:.2f}")
        print(f"  Long-Short: {sharpe_ls:.2f}")

        return strat_df

    # ==============================================================================
    # MAIN
    # ==============================================================================
    def main():
        print("Carregando dados...")
        df = pd.read_parquet(
            HISTORY_PARQUET_PATH,
            columns=["Date", "ticker", "price", SCORE_COL]
        )
        df["ticker"] = df["ticker"].astype("string")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna()

        if ONLY_SA:
            df = df[df["ticker"].str.endswith(".SA", na=False)]

        df = df.sort_values(["ticker", "Date"])

        print(f"\nDataset: {len(df):,} linhas, {df['ticker'].nunique()} tickers")

        # Executa testes
        corr_results = test_forward_correlation(df)
        quintile_results = test_quintiles(df)
        ic_results = test_ic(df)
        strat_results = test_long_short_strategies(df)

        # Conclusão
        print("\n" + "="*80)
        print("CONCLUSÃO")
        print("="*80)

        mean_ic = ic_results["IC"].mean()
        ls_ret = strat_results["LongShort"].mean()

        if mean_ic > 0.02 and ls_ret > 0:
            print("✅ Score EV_buy_fund_3 TEM poder preditivo")
            print("   → Prossiga com otimização de estratégia")
        elif mean_ic > 0 or ls_ret > 0:
            print("⚠️ Score EV_buy_fund_3 tem ALGUM poder (fraco)")
            print("   → Considere melhorar o score ou usar outro")
        else:
            print("❌ Score EV_buy_fund_3 NÃO funciona")
            print("   → PARE! Use outro score ou método")

    if __name__ == "__main__":
        main()



if __name__ == '__main__':
    run_final_output()

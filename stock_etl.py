#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated from notebook. Wrapped as callable function.
"""

import os
os.makedirs('./output/data', exist_ok=True)
os.makedirs('./output/data/models', exist_ok=True)
os.makedirs('./output', exist_ok=True)


def run_etl():
    """Run the pipeline step."""


    import numpy as np
    import pandas as pd
    import yfinance as yf
    from tqdm import tqdm
    import warnings
    import traceback
    warnings.filterwarnings("ignore")
    from collections import Counter

    import ta  # technical analysis
    from numeric_utils import to_float32_safe, normalize_before_save

    # ============================
    # Configurações gerais
    # ============================
    START_DATE = "2005-01-01"

    # Permite preencher lacunas de FEATURES com bfill (útil p/ calendários diferentes).
    # Isto pode introduzir leakage "quando inevitável".
    ALLOW_BFILL_EXOGENOUS = False

    # Defasagem das FEATURES (1 evita leakage trivial; 0 permite mais vazamento).
    SHIFT_FEATURES = 1

    # Médias móveis a usar (manteremos TODOS cruzamentos slow > fast)
    AVERAGES = [1, 2, 5, 10, 15, 20, 25, 50, 100]

    # Horizonte para cálculo de alvos
    HORIZON = 90
    UP_THR = 0.25   # +30%
    DD_THR = -0.10  # -10%


    SAVE_PARQUET = True
    SAVE_CSV_FALLBACK = False
    OUTPUT_PATH = "./output/data/expanded_stock.parquet"

    # ============================
    # Listas de tickers
    # ============================
    # ============================
    # Listas de tickers (curadas)
    # ============================

    # Ações Brasil (foco em liquidez/setores + alguns nomes clássicos)
    ibovespa_tickers = [
        # Bancos / financeiros
        "BBAS3.SA", "BBDC4.SA", "ITUB4.SA", "ITSA4.SA", "BBSE3.SA", "BPAC11.SA", "SANB11.SA",

        # Commodities / energia / materiais
        "VALE3.SA", "PETR4.SA", "PRIO3.SA", "BRKM5.SA",
        "GGBR4.SA", "GOAU4.SA", "CSNA3.SA", "USIM5.SA",
        "SUZB3.SA", "KLBN11.SA",

        # Utilities / infraestrutura
        "AXIA3.SA", "AXIA6.SA", "EQTL3.SA", "ENGI11.SA", "TAEE11.SA", "CMIG4.SA", "CPLE6.SA",
        "SBSP3.SA", "SAPR11.SA",

        # Consumo / varejo / saúde
        "ABEV3.SA", "ASAI3.SA", "RADL3.SA", "LREN3.SA", "MGLU3.SA", "RENT3.SA", "MULT3.SA", "HAPV3.SA",

        # Indústria / transporte / energia downstream
        "WEGE3.SA", "EMBR3.SA", "RAIL3.SA", "CCRO3.SA", "ECOR3.SA",
        "CSAN3.SA", "RAIZ4.SA", "UGPA3.SA", "VBBR3.SA", "SLCE3.SA",

        # Proteínas (mantém diversificação exportadora)
        "MBRF3.SA", "BEEF3.SA",

        # Telecom / real estate (opcionais, mas costumam ser úteis)
        "VIVT3.SA", "TIMS3.SA", "CYRE3.SA", "JHSF3.SA",

        # JBS: no Yahoo costuma funcionar melhor via BDR (mantém “JBS” sem depender do JBSS3.SA)
        "JBSS32.SA",

        # Incorporação / real estate (apareceram forte em DY 2025)
        "DIRR3.SA", "CURY3.SA", "LAVV3.SA", "SYNE3.SA",

        # Energia / utilities (tradicionalmente boas pagadoras)
        "ISAE4.SA", "TRPL4.SA", "EGIE3.SA", "ALUP11.SA", "CPFE3.SA", "NEOE3.SA", "ENBR3.SA",

        # Saneamento
        "CSMG3.SA",

        # Bancos / seguros / serviços financeiros
        "ABCB4.SA", "BRSR6.SA", "PSSA3.SA", "CXSE3.SA", "WIZC3.SA",

        # Saúde / consumo
        "FLRY3.SA", "ODPV3.SA", "MDIA3.SA", "GRND3.SA", "VULC3.SA",

        # Indústria / logística
        "POMO4.SA", "LEVE3.SA", "RANI3.SA", "TGMA3.SA",

        # Agro
        "SMTO3.SA", "AGRO3.SA",

        # Shoppings
        "IGTI11.SA",

        # Química / materiais
        "UNIP6.SA",

        # Construção civil / incorporação
        "EZTC3.SA",

        # Frigoríficos / alimentos
        "MLAS3.SA",

        # Turismo / lazer
        "CVCB3.SA",

        # BDR / ETF internacional
        "XPBR31.SA",

    ]

    # FIIs Brasil (mix: logística, shoppings, lajes, papel, FoF)
    fii_tickers = [
        # Tijolo - logística / renda
        "MXRF11.SA", "KNRI11.SA", "HGLG11.SA", "XPLG11.SA", "BTLG11.SA", "BRCO11.SA", "GGRC11.SA", "LVBI11.SA",
        # Tijolo - shoppings
        "VISC11.SA", "XPML11.SA", "HSML11.SA", "HGBS11.SA",
        # Tijolo - lajes / híbridos
        "BRCR11.SA", "HGRE11.SA", "HGPO11.SA", "PVBI11.SA", "RCRB11.SA", "VINO11.SA",
        # Híbridos / renda urbana
        "ALZR11.SA", "TRXF11.SA", "RBRP11.SA",
        # Papel / crédito
        "KNCR11.SA", "KNSC11.SA", "KNHY11.SA", "CPTS11.SA", "IRDM11.SA",
        # FoF
        "BCFF11.SA", "HFOF11.SA", "KFOF11.SA", "RBRF11.SA",
        # Saúde / biotech
        "RBRX11.SA",
        # Híbridos / diversificados
        "BCIA11.SA",
    ]

    # Índices globais + proxies macro (inclui Suécia e juros EUA)
    global_indices = [
        # EUA (equities)
        "^GSPC", "^DJI", "^IXIC", "^NDX", "^RUT",
        # Volatilidade / sentimento
        "^VIX", "^VVIX",
        # Juros EUA (yields)
        "^IRX", "^FVX", "^TNX", "^TYX",

        # Europa
        "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E", "^STOXX",

        # Ásia-Pacífico
        "^N225", "^HSI", "^KS11", "^AXJO", "^TWII", "^JKSE",

        # Emergentes / LatAm
        "^BVSP", "^BSESN", "^MXX",

        # China (muitas vezes o Yahoo usa “códigos de bolsa” como índices)
        "000001.SS", "399001.SZ",

        # Suécia (relevante pra você)
        "^OMXS30",
    ]

    # FX + commodities + cripto (mais “robusto” no Yahoo que XAUEUR/XAGEUR)
    currency_commodity_tickers = [
        # Dólar (DXY)
        "DX-Y.NYB",

        # FX principais (formas mais comuns no Yahoo)
        "EURUSD=X", "GBPUSD=X", "AUDUSD=X",
        "JPY=X", "CNY=X", "CHF=X", "BRL=X", "MXN=X", "SEK=X",

        # Metais / energia (futuros)
        "GC=F", "SI=F", "HG=F",
        "CL=F", "BZ=F", "NG=F", "HO=F",

        # Agricultura / pecuária
        "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "HE=F", "LE=F",

        # ETFs macro/temáticos (complemento)
        "UUP", "FXE", "FXY", "GLD", "USO", "COPX",

        # Crypto
        "BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD", "DOGE-USD","CRO-USD",
    ]

    ALL_TICKERS = ibovespa_tickers + fii_tickers + global_indices + currency_commodity_tickers

    # ============================
    # Fundamentals (opcional)
    # ============================
    ADD_FUNDAMENTALS_QUARTERLY = True      # recomendado (histórico)
    ADD_FUNDAMENTALS_SNAPSHOT  = False     # WARNING: snapshot atual (leakage no treino)

    # cache em memória (p/ não chamar yfinance toda hora)
    _FUND_Q_CACHE = {}
    _FUND_S_CACHE = {}

    # quais campos do .info você quer (somente numéricos)
    SNAPSHOT_FIELDS = [
        "trailingPE",              # ok
        "priceToSalesTrailing12Months",  # NEW (P/S)
        "priceToBook",             # ok
        "pegRatio",                # NEW
        "trailingEps",             # NEW
        "bookValue",               # NEW
        "sharesOutstanding",       # NEW
        "dividendYield", "payoutRatio",
        "beta", "marketCap", "enterpriseValue",
        "enterpriseToEbitda", "enterpriseToRevenue",
        "profitMargins", "operatingMargins", "grossMargins",
        "returnOnEquity", "returnOnAssets",
        "totalDebt", "totalCash", "freeCashflow", "operatingCashflow",
        "currentRatio", "quickRatio",     # NEW (se vier)
    ]

    # ============================
    # Utilidades de preenchimento/casting
    # ============================
    def fill_100pct(df: pd.DataFrame, allow_bfill=False) -> pd.DataFrame:
        """Garante 100% preenchido: Inf->NaN, ffill, bfill opcional, e NaN restantes->0."""
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill()
        if allow_bfill:
            df = df.bfill()
        # Se alguma coluna ficou toda NaN (pode ocorrer em padrões), zera
        all_nan_cols = df.columns[df.isna().all()].tolist()
        if all_nan_cols:
            df[all_nan_cols] = 0.0
        # NaN remanescentes -> 0
        df = df.fillna(0.0)
        return df

    def cast_int8_multi(df: pd.DataFrame, prefixes=(), suffixes=()):
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        level0 = pd.Index(df.columns.get_level_values(0).astype(str))
        mask = np.zeros(len(level0), dtype=bool)

        if prefixes:
            starts = level0.str.startswith(prefixes)          # array-like
            mask = np.logical_or(mask, np.asarray(starts, dtype=bool))

        if suffixes:
            ends = level0.str.endswith(suffixes)              # array-like
            mask = np.logical_or(mask, np.asarray(ends, dtype=bool))

        cols = df.columns[mask]
        if len(cols):
            df = df.copy()
            df.loc[:, cols] = df.loc[:, cols].astype('int8', copy=False)
        return df

    # ============================
    # Download dos dados
    # ============================
    print("Baixando cotações do Yahoo Finance...")
    data = yf.download(
        ALL_TICKERS,
        start=START_DATE,
        group_by='column',
        auto_adjust=True,
        progress=False,
        threads=True
    )

    # Somente colunas OHLCV relevantes e limpeza de levels
    allowed_columns = ['Open','High','Low','Close','Adj Close','Volume']
    data = data.loc[:, data.columns.get_level_values(0).isin(allowed_columns)].copy()
    data.columns = data.columns.remove_unused_levels()

    # Forward-fill para alinhar calendários; (bfill só nas FEATURES mais adiante)
    data = data.ffill()

    # Tickers efetivamente presentes
    # Tickers efetivamente presentes (e com preço real, não tudo-NaN)
    tickers_all = np.unique(data.columns.get_level_values(1))
    requested_set  = set(ALL_TICKERS)
    downloaded_set = set(tickers_all)

    missing_download = sorted(requested_set - downloaded_set)
    extra_download   = sorted(downloaded_set - requested_set)

    print(f"[AUDIT] requested={len(requested_set)} | downloaded={len(downloaded_set)}")
    print(f"[AUDIT] missing in download: {len(missing_download)}")
    if missing_download[:20]:
        print("  ex:", missing_download[:20])

    valid_data_tickers = []
    for tk in tickers_all:
        # precisa existir pelo menos Close/High/Low para o ticker
        if ('Close', tk) not in data.columns or ('High', tk) not in data.columns or ('Low', tk) not in data.columns:
            continue

        close_s = data[('Close', tk)]
        high_s  = data[('High', tk)]
        low_s   = data[('Low', tk)]

        # descarta tickers que vieram “presentes” mas sem nenhuma cotação útil
        if close_s.dropna().empty or high_s.dropna().empty or low_s.dropna().empty:
            continue

        # descarta se for tudo NaN nos 3 (proteção extra)
        if pd.concat([close_s, high_s, low_s], axis=1).isna().all().all():
            continue

        valid_data_tickers.append(tk)

    tickers = np.array(valid_data_tickers, dtype=object)

    valid_prices_set = set(tickers)
    dropped_after_valid_prices = sorted(downloaded_set - valid_prices_set)
    print(f"[AUDIT] valid_prices={len(valid_prices_set)} | dropped_valid_prices={len(dropped_after_valid_prices)}")
    if dropped_after_valid_prices[:20]:
        print("  ex:", dropped_after_valid_prices[:20])

    print(f"Período: {data.index.min().date()} → {data.index.max().date()}")
    print(f"Tickers com dados (válidos): {len(tickers)}")


    # ============================
    # Funções de padrões (com prefixos)
    # ============================
    def detect_head_shoulder(df, window=3, prefix="hs_"):
        out = pd.DataFrame(index=df.index)
        out[prefix+'high_roll_max'] = df['High'].rolling(window).max()
        out[prefix+'low_roll_min']  = df['Low'].rolling(window).min()
        mask_hs  = ((out[prefix+'high_roll_max'] > df['High'].shift(1)) &
                    (out[prefix+'high_roll_max'] > df['High'].shift(-1)) &
                    (df['High'] < df['High'].shift(1)) &
                    (df['High'] < df['High'].shift(-1)))
        mask_inv = ((out[prefix+'low_roll_min'] < df['Low'].shift(1)) &
                    (out[prefix+'low_roll_min'] < df['Low'].shift(-1)) &
                    (df['Low'] > df['Low'].shift(1)) &
                    (df['Low'] > df['Low'].shift(-1)))
        out[prefix+'pattern'] = 0
        out.loc[mask_hs,  prefix+'pattern'] = 1
        out.loc[mask_inv, prefix+'pattern'] = -1
        return out

    def detect_multiple_tops_bottoms(df, window=3, prefix="mtb_"):
        out = pd.DataFrame(index=df.index)
        out[prefix+'high_roll_max']  = df['High'].rolling(window).max()
        out[prefix+'low_roll_min']   = df['Low'].rolling(window).min()
        out[prefix+'close_roll_max'] = df['Close'].rolling(window).max()
        out[prefix+'close_roll_min'] = df['Close'].rolling(window).min()
        mask_top    = (out[prefix+'high_roll_max'] >= df['High'].shift(1)) & (out[prefix+'close_roll_max'] < df['Close'].shift(1))
        mask_bottom = (out[prefix+'low_roll_min']  <= df['Low'].shift(1))  & (out[prefix+'close_roll_min']  > df['Close'].shift(1))
        out[prefix+'pattern'] = 0
        out.loc[mask_top,    prefix+'pattern'] = 1
        out.loc[mask_bottom, prefix+'pattern'] = -1
        return out

    def calculate_support_resistance(df, window=3, prefix="sr_"):
        out = pd.DataFrame(index=df.index)
        mean_high = df['High'].rolling(window).mean()
        std_high  = df['High'].rolling(window).std()
        mean_low  = df['Low'].rolling(window).mean()
        std_low   = df['Low'].rolling(window).std()
        out[prefix+'support']     = mean_low - 2*std_low
        out[prefix+'resistance']  = mean_high + 2*std_high
        out[prefix+'diff_support']    = df['Close'] - out[prefix+'support']
        out[prefix+'diff_resistance'] = out[prefix+'resistance'] - df['Close']
        return out

    def detect_triangle_pattern(df, window=3, prefix="tri_"):
        out = pd.DataFrame(index=df.index)
        out[prefix+'high_roll_max'] = df['High'].rolling(window).max()
        out[prefix+'low_roll_min']  = df['Low'].rolling(window).min()
        mask_asc  = (out[prefix+'high_roll_max'] >= df['High'].shift(1)) & (out[prefix+'low_roll_min'] <= df['Low'].shift(1)) & (df['Close'] > df['Close'].shift(1))
        mask_desc = (out[prefix+'high_roll_max'] <= df['High'].shift(1)) & (out[prefix+'low_roll_min'] >= df['Low'].shift(1)) & (df['Close'] < df['Close'].shift(1))
        out[prefix+'pattern'] = 0
        out.loc[mask_asc,  prefix+'pattern'] = 1
        out.loc[mask_desc, prefix+'pattern'] = -1
        return out

    def detect_wedge(df, window=3, prefix="wed_"):
        out = pd.DataFrame(index=df.index)
        out[prefix+'high_roll_max'] = df['High'].rolling(window).max()
        out[prefix+'low_roll_min']  = df['Low'].rolling(window).min()
        trend_high = df['High'].rolling(window).apply(lambda x: 1 if (x[-1]-x[0])>0 else (-1 if (x[-1]-x[0])<0 else 0), raw=True)
        trend_low  = df['Low'].rolling(window).apply(lambda x: 1 if (x[-1]-x[0])>0 else (-1 if (x[-1]-x[0])<0 else 0), raw=True)
        mask_up   = (out[prefix+'high_roll_max'] >= df['High'].shift(1)) & (out[prefix+'low_roll_min'] <= df['Low'].shift(1)) & (trend_high == 1) & (trend_low == 1)
        mask_down = (out[prefix+'high_roll_max'] <= df['High'].shift(1)) & (out[prefix+'low_roll_min'] >= df['Low'].shift(1)) & (trend_high == -1) & (trend_low == -1)
        out[prefix+'pattern'] = 0
        out.loc[mask_up,   prefix+'pattern'] = 1
        out.loc[mask_down, prefix+'pattern'] = -1
        return out

    def detect_channel(df, window=3, prefix="chan_", channel_range=0.1):
        out = pd.DataFrame(index=df.index)
        out[prefix+'high_roll_max'] = df['High'].rolling(window).max()
        out[prefix+'low_roll_min']  = df['Low'].rolling(window).min()
        trend_high = df['High'].rolling(window).apply(lambda x: 1 if (x[-1]-x[0])>0 else (-1 if (x[-1]-x[0])<0 else 0), raw=True)
        trend_low  = df['Low'].rolling(window).apply(lambda x: 1 if (x[-1]-x[0])>0 else (-1 if (x[-1]-x[0])<0 else 0), raw=True)
        width = out[prefix+'high_roll_max'] - out[prefix+'low_roll_min']
        mid   = (out[prefix+'high_roll_max'] + out[prefix+'low_roll_min'])/2
        mask_up   = (out[prefix+'high_roll_max'] >= df['High'].shift(1)) & (out[prefix+'low_roll_min'] <= df['Low'].shift(1)) & (width <= channel_range*mid) & (trend_high==1) & (trend_low==1)
        mask_down = (out[prefix+'high_roll_max'] <= df['High'].shift(1)) & (out[prefix+'low_roll_min'] >= df['Low'].shift(1)) & (width <= channel_range*mid) & (trend_high==-1) & (trend_low==-1)
        out[prefix+'pattern'] = 0
        out.loc[mask_up,   prefix+'pattern'] = 1
        out.loc[mask_down, prefix+'pattern'] = -1
        return out

    def detect_double_top_bottom(df, window=3, threshold=0.05, prefix="dbl_"):
        out = pd.DataFrame(index=df.index)
        out[prefix+'high_roll_max'] = df['High'].rolling(window).max()
        out[prefix+'low_roll_min']  = df['Low'].rolling(window).min()
        mask_top = (out[prefix+'high_roll_max'] >= df['High'].shift(1)) & (out[prefix+'high_roll_max'] >= df['High'].shift(-1)) & \
                   (df['High'] < df['High'].shift(1)) & (df['High'] < df['High'].shift(-1)) & \
                   ((df['High'].shift(1)-df['Low'].shift(1)) <= threshold*(df['High'].shift(1)+df['Low'].shift(1))/2) & \
                   ((df['High'].shift(-1)-df['Low'].shift(-1)) <= threshold*(df['High'].shift(-1)+df['Low'].shift(-1))/2)
        mask_bottom = (out[prefix+'low_roll_min'] <= df['Low'].shift(1)) & (out[prefix+'low_roll_min'] <= df['Low'].shift(-1)) & \
                      (df['Low'] > df['Low'].shift(1)) & (df['Low'] > df['Low'].shift(-1)) & \
                      ((df['High'].shift(1)-df['Low'].shift(1)) <= threshold*(df['High'].shift(1)+df['Low'].shift(1))/2) & \
                      ((df['High'].shift(-1)-df['Low'].shift(-1)) <= threshold*(df['High'].shift(-1)+df['Low'].shift(-1))/2)
        out[prefix+'pattern'] = 0
        out.loc[mask_top,    prefix+'pattern'] = 1
        out.loc[mask_bottom, prefix+'pattern'] = -1
        return out

    def detect_trendline(df, window=2, prefix="trend_"):
        out = pd.DataFrame(index=df.index)
        slope = np.zeros(len(df), dtype='float64')
        intercept = np.zeros(len(df), dtype='float64')
        idx = np.arange(len(df))
        close = df['Close'].values
        for i in range(window, len(df)):
            x = idx[i-window:i].astype(float)
            y = close[i-window:i]
            A = np.vstack([x, np.ones_like(x)]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            slope[i] = m
            intercept[i] = c
        out[prefix+'slope'] = slope
        out[prefix+'intercept'] = intercept
        x_now = idx.astype(float)
        y_line = slope * x_now + intercept
        # Estes dois podem ficar inteiros NaN; trataremos depois com fill_100pct
        out[prefix+'support2'] = np.where(slope>0, y_line, np.nan)
        out[prefix+'resistance2'] = np.where(slope<0, y_line, np.nan)
        out[prefix+'diff_support2'] = df['Close'] - out[prefix+'support2']
        out[prefix+'diff_resistance2'] = out[prefix+'resistance2'] - df['Close']
        return out
    # ============================
    # New PATH-AWARE targets (split)
    # ============================
    def make_targets_up_down(df, horizon=30, up_thr=0.20, dd_thr=-0.05,
                             name_up='target_up20', name_dd='target_dd5',
                             keep_order=False, name_order='target_up_before_dd'):
        """
        For each day t:
          - target_up20[t] = 1 if any High in (t+1 ... t+horizon) >= Close[t]*(1+up_thr), else 0
          - target_dd5[t]  = 1 if any Low  in (t+1 ... t+horizon) <= Close[t]*(1+dd_thr), else 0
        If keep_order=True:
          - target_up_before_dd[t] = 1 if the first hit is the UP threshold, 0 if first is DOWN, -1 if none hit
        """
        close = df['Close'].values
        high  = df['High'].values
        low   = df['Low'].values
        n = len(df)

        up_hit = np.zeros(n, dtype='int8')
        dd_hit = np.zeros(n, dtype='int8')
        order  = np.full(n, -1, dtype='int8')  # -1 => none

        for t in range(n):
            end = min(n, t + horizon + 1)
            if end - t <= 1:
                continue

            entry = close[t]
            up_th = entry * (1.0 + up_thr)
            dd_th = entry * (1.0 + dd_thr)

            hseg = high[t+1:end]
            lseg = low[t+1:end]

            up_idx = np.where(hseg >= up_th)[0]
            dd_idx = np.where(lseg <= dd_th)[0]

            hit_up = up_idx[0] if len(up_idx) else None
            hit_dd = dd_idx[0] if len(dd_idx) else None

            if hit_up is not None:
                up_hit[t] = 1
            if hit_dd is not None:
                dd_hit[t] = 1

            if keep_order:
                if hit_up is None and hit_dd is None:
                    order[t] = -1
                elif hit_up is None:
                    order[t] = 0
                elif hit_dd is None:
                    order[t] = 1
                else:
                    order[t] = 1 if hit_up < hit_dd else 0

        s_up   = pd.Series(up_hit, index=df.index, name=name_up)
        s_down = pd.Series(dd_hit, index=df.index, name=name_dd)
        if keep_order:
            s_ord = pd.Series(order, index=df.index, name=name_order)
            return s_up, s_down, s_ord
        else:
            return s_up, s_down

    # ============================
    # FEATURES (ta + cruzamentos + padrões)
    # ============================
    def indicators_for_ticker(ohlcv: pd.DataFrame, shift_features: int = 1) -> pd.DataFrame:
        close_col = 'Adj Close' if 'Adj Close' in ohlcv.columns else 'Close'

        ohlcv_num = ohlcv.copy()
        base_cols = ["Open", "High", "Low", close_col, "Volume"]
        for col in base_cols:
            if col in ohlcv_num.columns:
                ohlcv_num[col] = pd.to_numeric(ohlcv_num[col], errors='coerce').astype(np.float64, copy=False)

        feats = ta.add_all_ta_features(
            ohlcv_num,
            open="Open", high="High", low="Low", close=close_col, volume="Volume",
            fillna=True
        )

        # SMAs sobre o mesmo close_col
        for avg in AVERAGES:
            feats[f'SMA_{avg}'] = ohlcv_num[close_col].rolling(avg).mean()

        # cruzamentos
        for fast in AVERAGES:
            for slow in AVERAGES:
                if slow > fast:
                    fcol = f'SMA_{fast}'
                    scol = f'SMA_{slow}'
                    prev_f = feats[fcol].shift(1)
                    prev_s = feats[scol].shift(1)
                    crossname = f"cross_{fast}_{slow}"
                    cross = pd.Series(0, index=feats.index, dtype='int8')
                    cross[(feats[fcol] < feats[scol]) & (prev_f >= prev_s)] = -1
                    cross[(feats[fcol] > feats[scol]) & (prev_f <= prev_s)] = 1
                    feats[crossname] = cross

        feats['pct_change'] = ohlcv_num[close_col].pct_change()

        if shift_features > 0:
            feats = feats.shift(shift_features)

        # >>> FIX: garantir que colunas discretas não tenham NaN antes do astype(int8)
        int_cols = [c for c in feats.columns if str(c).startswith("cross_")]
        feats[int_cols] = feats[int_cols].fillna(0).astype('int8')

        # floats podem ficar com NaN (você já zera depois no fill_100pct), mas pode manter assim:
        float_cols = [c for c in feats.columns if c not in int_cols]
        feats[float_cols] = to_float32_safe(feats[float_cols])

        return feats



    def patterns_for_ticker(ohlcv: pd.DataFrame, shift_features: int = 1) -> pd.DataFrame:
        parts = [
            detect_head_shoulder(ohlcv),
            detect_multiple_tops_bottoms(ohlcv),
            calculate_support_resistance(ohlcv),
            detect_triangle_pattern(ohlcv),
            detect_wedge(ohlcv),
            detect_channel(ohlcv),
            detect_double_top_bottom(ohlcv),
            detect_trendline(ohlcv),
        ]
        P = pd.concat(parts, axis=1)
        if shift_features > 0:
            P = P.shift(shift_features)

        patt_cols = [c for c in P.columns if c.endswith('pattern')]
        P[patt_cols] = P[patt_cols].fillna(0).astype('int8')  # <<< FIX

        other_cols = [c for c in P.columns if c not in patt_cols]
        P[other_cols] = to_float32_safe(P[other_cols])
        return P


    # ============================
    # TARGETS
    # ============================
    def make_target_path_aware(df, horizon=30, up=0.15, dd=-0.05, name='target_path'):
        close = df['Close'].values
        high  = df['High'].values
        low   = df['Low'].values
        n = len(df)
        tgt = np.zeros(n, dtype='int8')
        for t in range(n):
            end = min(n, t + horizon + 1)
            if end - t <= 1:
                tgt[t] = 0
                continue
            entry = close[t]
            up_th = entry * (1.0 + up)
            dd_th = entry * (1.0 + dd)
            hseg = high[t+1:end]
            lseg = low[t+1:end]
            hit_up_idx = np.where(hseg >= up_th)[0]
            hit_dd_idx = np.where(lseg <= dd_th)[0]
            hit_up = hit_up_idx[0] if len(hit_up_idx) else None
            hit_dd = hit_dd_idx[0] if len(hit_dd_idx) else None
            tgt[t] = 1 if (hit_up is not None and (hit_dd is None or hit_up < hit_dd)) else 0
        return pd.Series(tgt, index=df.index, name=name)

    def make_best_entry_sale(df, horizon=30):
        low = df['Low'].values
        high = df['High'].values
        n = len(df)
        best_entry = np.empty(n, dtype='float32')
        best_sale  = np.empty(n, dtype='float32')
        for t in range(n):
            end = min(n, t + horizon + 1)
            window_low = low[t:end]
            window_high = high[t:end]
            best_entry[t] = float(np.nanmin(window_low))
            best_sale[t]  = float(np.nanmax(window_high))
        s_entry = pd.Series(best_entry, index=df.index, name='target_best_entry')
        s_sale  = pd.Series(best_sale,  index=df.index, name='target_best_sale')
        return s_entry, s_sale

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.utils import check_random_state

    # ============================
    # Feature reduction helpers
    # ============================
    def _is_discrete_series(colname: str) -> bool:
        # treat crosses and pattern flags as discrete
        return colname.startswith("cross_") or colname.endswith("pattern")

    def _drop_near_constant(df_tk: pd.DataFrame, var_thr: float = 1e-12):
        variances = df_tk.var(axis=0).astype(float)
        keep = variances > var_thr
        return df_tk.loc[:, keep], keep.index[keep].tolist()

    def _mi_rank_per_ticker(X_tk: pd.DataFrame, y_up: pd.Series, y_dd: pd.Series,
                            random_state=42):
        """
        Compute MI vs both targets; take max(MI_up, MI_dd) per feature.
        """
        # mask discrete
        cols = X_tk.columns.tolist()
        discrete_mask = np.array([_is_discrete_series(c) for c in cols], dtype=bool)

        # y must be 1D arrays
        y_up_arr = y_up.astype('int8').values
        y_dd_arr = y_dd.astype('int8').values

        rs = check_random_state(random_state)
        # MI can fail on constant features; ensure X already filtered
        mi_up = mutual_info_classif(X_tk.values, y_up_arr,
                                    discrete_features=discrete_mask,
                                    random_state=rs)
        mi_dd = mutual_info_classif(X_tk.values, y_dd_arr,
                                    discrete_features=discrete_mask,
                                    random_state=rs)
        mi = np.maximum(mi_up, mi_dd)
        mi_s = pd.Series(mi, index=cols).sort_values(ascending=False)
        return mi_s

    def _greedy_cor_filter(X_tk: pd.DataFrame, ranking: pd.Series,
                           corr_thr: float = 0.995):
        """
        Keep features in order of 'ranking' (desc), discard any that
        correlate (|r| >= corr_thr) with a feature already kept.
        """
        if X_tk.shape[1] <= 1:
            return X_tk.columns.tolist()

        ordered = [c for c in ranking.index if c in X_tk.columns]
        keep = []
        # precompute correlation in chunks to save time
        # Pearson on normalized data
        Z = (X_tk - X_tk.mean()) / (X_tk.std(ddof=0) + 1e-12)

        for c in ordered:
            if not keep:
                keep.append(c)
                continue
            # correlate c with kept
            r = Z[keep].T.dot(Z[c]) / (len(Z) - 1)
            max_abs_r = np.abs(r.values).max()
            if not np.isfinite(max_abs_r) or max_abs_r < corr_thr:
                keep.append(c)
        return keep

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.utils import check_random_state

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.utils import check_random_state

    def reduce_features_automatic(
        X: pd.DataFrame,
        y_up: pd.DataFrame,
        y_dd: pd.DataFrame,
        top_fraction: float = 0.35,
        min_keep: int = 48,
        var_thr: float | None = None,     # <<< added back
        corr_thr: float = 0.995,
        always_keep_prefixes=("pct_change","SMA_","tri_","sr_","fundQ_","fundS_"),
        verbose: bool = True,
    ):
        """
        Robust per-ticker reduction:
          - drop constants (nunique > 1) and optionally low-variance (<= var_thr)
          - MI vs both targets; fallback to variance if MI is flat/constant
          - greedy correlation de-dup; pad back to min_keep
        """
        if not isinstance(X.columns, pd.MultiIndex):
            raise ValueError("X must use MultiIndex columns=(feature, ticker).")

        tickers = np.unique(X.columns.get_level_values(1))
        kept_cols = []
        before_cnt = X.shape[1]
        rs = check_random_state(42)

        up_tk_set = set(y_up.columns.get_level_values(1))
        dd_tk_set = set(y_dd.columns.get_level_values(1))

        for tk in tickers:
            X_tk = X.xs(tk, level=1, axis=1).copy()

            # numeric only + ensure no NaN/Inf
            X_tk = X_tk.apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan).fillna(0.0)

            # drop constants
            nunique = X_tk.nunique(dropna=False)
            X_tk_nc = X_tk.loc[:, nunique > 1]

            # optional variance threshold
            if var_thr is not None and X_tk_nc.shape[1] > 0:
                variances = X_tk_nc.var().astype(float)
                X_tk_nc = X_tk_nc.loc[:, variances > var_thr]

            if X_tk_nc.shape[1] == 0:
                base = [c for c in X_tk.columns if any(str(c).startswith(p) for p in always_keep_prefixes)]
                base = base[:min_keep] if base else X_tk.columns[:min_keep].tolist()
                kept_cols.extend([(c, tk) for c in base])
                continue

            # if labels missing for ticker, keep top variance
            if (tk not in up_tk_set) or (tk not in dd_tk_set):
                var_rank = X_tk_nc.var().sort_values(ascending=False)
                base = var_rank.index[:min(min_keep, len(var_rank))].tolist()
                kept_cols.extend([(c, tk) for c in base])
                continue

            y_up_tk = y_up.xs(tk, level=1, axis=1).iloc[:,0].astype('int8')
            y_dd_tk = y_dd.xs(tk, level=1, axis=1).iloc[:,0].astype('int8')

            cols = X_tk_nc.columns.tolist()
            discrete_mask = np.array([str(c).startswith('cross_') or str(c).endswith('pattern') for c in cols], dtype=bool)

            def safe_mi(Xarr, yarr):
                if np.unique(yarr).size < 2:
                    return np.zeros(Xarr.shape[1], dtype=float)
                try:
                    return mutual_info_classif(Xarr, yarr, discrete_features=discrete_mask, random_state=rs)
                except Exception:
                    return np.zeros(Xarr.shape[1], dtype=float)

            mi_up = safe_mi(X_tk_nc.values, y_up_tk.values)
            mi_dd = safe_mi(X_tk_nc.values, y_dd_tk.values)
            mi = np.maximum(mi_up, mi_dd)

            mi_rank = pd.Series(mi, index=cols).sort_values(ascending=False)
            k_top = min(len(mi_rank), max(min_keep, int(np.ceil(len(mi_rank) * top_fraction))))
            top_feats = mi_rank.index[:k_top].tolist()

            # ensure interpretable anchors
            for pref in always_keep_prefixes:
                top_feats.extend([c for c in X_tk_nc.columns if str(c).startswith(pref)])
            # de-dup order
            seen = set()
            top_feats = [c for c in top_feats if not (c in seen or seen.add(c))]

            # fallback if MI flat
            if len(top_feats) == 0 or (mi_rank.iloc[0] == 0 and mi_rank.sum() == 0):
                var_rank = X_tk_nc.var().sort_values(ascending=False)
                top_feats = var_rank.index[:min(min_keep, len(var_rank))].tolist()

            # greedy correlation de-dup
            X_top = X_tk_nc[top_feats]
            Z = (X_top - X_top.mean()) / (X_top.std(ddof=0) + 1e-12)
            keep = []
            for c in X_top.columns:
                if not keep:
                    keep.append(c); continue
                r = Z[keep].T.dot(Z[c]) / max(1, (len(Z) - 1))
                max_abs_r = np.abs(r.values).max() if hasattr(r, "values") else float(np.abs(r).max())
                if not np.isfinite(max_abs_r) or max_abs_r < corr_thr:
                    keep.append(c)

            # pad to min_keep
            if len(keep) < min_keep:
                for c in top_feats:
                    if c not in keep:
                        keep.append(c)
                        if len(keep) >= min_keep:
                            break

            kept_cols.extend([(c, tk) for c in keep])

        if len(kept_cols) == 0:
            for tk in tickers:
                cols_tk = X.xs(tk, level=1, axis=1).columns[:min_keep].tolist()
                kept_cols.extend([(c, tk) for c in cols_tk])

        kept_cols = pd.MultiIndex.from_tuples(kept_cols, names=X.columns.names)
        X_red = X.loc[:, kept_cols].copy()

        # compact dtypes
        X_red = cast_int8_multi(X_red, prefixes=("cross_",), suffixes=("pattern",))
        other0 = [c for c in X_red.columns.get_level_values(0)
                  if not (str(c).startswith("cross_") or str(c).endswith("pattern"))]
        if other0:
            X_red.loc[:, (other0, slice(None))] = to_float32_safe(X_red.loc[:, (other0, slice(None))])

        if verbose:
            after_cnt = X_red.shape[1]
            print(f"[Feature reduction] columns: {before_cnt} → {after_cnt} ({100.0*after_cnt/before_cnt:.1f}% kept)")
        return X_red
    import re

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

    def _stmt_get(stmt: pd.DataFrame, candidates, col):
        """
        stmt: DataFrame estilo yfinance (index = linhas, columns = datas)
        col: uma coluna (data)
        candidates: lista de nomes possíveis p/ a linha
        """
        if stmt is None or stmt.empty:
            return np.nan
        idx_map = {_norm(i): i for i in stmt.index}
        for name in candidates:
            k = _norm(name)
            if k in idx_map:
                return stmt.loc[idx_map[k], col]
        return np.nan

    def fetch_snapshot_fundamentals(tk: str) -> dict:
        # WARNING: snapshot atual (bom pra APPLY; ruim pra treino histórico)
        if tk in _FUND_S_CACHE:
            return _FUND_S_CACHE[tk]

        out = {}
        try:
            info = yf.Ticker(tk).info or {}
            for f in SNAPSHOT_FIELDS:
                v = info.get(f, np.nan)
                # deixa só numérico
                if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v):
                    out[f"fundS_{f}"] = float(v)
                else:
                    out[f"fundS_{f}"] = np.nan
        except Exception:
            # mantém NaN
            for f in SNAPSHOT_FIELDS:
                out[f"fundS_{f}"] = np.nan

        _FUND_S_CACHE[tk] = out
        return out

    def fetch_quarterly_fundamentals_df(tk: str) -> pd.DataFrame:
        """
        Retorna DF indexado por datas de trimestre (period end),
        com features numéricas derivadas de balanços.
        """
        if tk in _FUND_Q_CACHE:
            return _FUND_Q_CACHE[tk]

        try:
            t = yf.Ticker(tk)
            inc = t.quarterly_income_stmt
            bal = t.quarterly_balance_sheet
            cfs = t.quarterly_cashflow

            # algumas vezes vem vazio p/ índices/FX/cripto
            if (inc is None or inc.empty) and (bal is None or bal.empty) and (cfs is None or cfs.empty):
                qdf = pd.DataFrame()
                _FUND_Q_CACHE[tk] = qdf
                return qdf

            # junta colunas (datas) disponíveis
            cols = []
            for df0 in (inc, bal, cfs):
                if df0 is not None and not df0.empty:
                    cols.extend(list(df0.columns))
            cols = sorted(set(cols))
            if not cols:
                qdf = pd.DataFrame()
                _FUND_Q_CACHE[tk] = qdf
                return qdf

            rows = []
            for col in cols:
                rev = _stmt_get(inc, ["Total Revenue", "TotalRevenue"], col)
                ni  = _stmt_get(inc, ["Net Income", "NetIncome"], col)
                ebt = _stmt_get(inc, ["EBITDA"], col)
                gp  = _stmt_get(inc, ["Gross Profit", "GrossProfit"], col)
                op_inc = _stmt_get(inc, ["Operating Income", "OperatingIncome"], col)

                # EPS: Diluted EPS preferred, fallback to Basic EPS
                diluted_eps = _stmt_get(inc, ["Diluted EPS", "DilutedEPS"], col)
                basic_eps   = _stmt_get(inc, ["Basic EPS", "BasicEPS"], col)
                eps_val = diluted_eps if np.isfinite(diluted_eps) else basic_eps

                # Shares outstanding (for derived EPS if not available)
                shares = _stmt_get(inc, ["Diluted Average Shares", "DilutedAverageShares",
                                         "Basic Average Shares", "BasicAverageShares"], col)

                assets = _stmt_get(bal, ["Total Assets", "TotalAssets"], col)
                liab   = _stmt_get(bal, ["Total Liabilities Net Minority Interest", "TotalLiabilitiesNetMinorityInterest",
                                         "Total Liabilities", "TotalLiabilities"], col)
                debt   = _stmt_get(bal, ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"], col)
                cash   = _stmt_get(bal, ["Cash And Cash Equivalents", "CashAndCashEquivalents"], col)
                curr_assets = _stmt_get(bal, ["Current Assets", "CurrentAssets"], col)
                curr_liab   = _stmt_get(bal, ["Current Liabilities", "CurrentLiabilities"], col)

                ocf = _stmt_get(cfs, ["Operating Cash Flow", "OperatingCashFlow"], col)
                fcf = _stmt_get(cfs, ["Free Cash Flow", "FreeCashFlow"], col)

                equity = (assets - liab) if np.isfinite(assets) and np.isfinite(liab) else np.nan

                # Derive EPS from net_income / shares if not directly available
                if not np.isfinite(eps_val) and np.isfinite(ni) and np.isfinite(shares) and shares > 0:
                    eps_val = ni / shares

                rows.append({
                    "date": pd.to_datetime(col),
                    "fundQ_revenue": float(rev) if np.isfinite(rev) else np.nan,
                    "fundQ_net_income": float(ni) if np.isfinite(ni) else np.nan,
                    "fundQ_ebitda": float(ebt) if np.isfinite(ebt) else np.nan,
                    "fundQ_gross_profit": float(gp) if np.isfinite(gp) else np.nan,
                    "fundQ_op_income": float(op_inc) if np.isfinite(op_inc) else np.nan,
                    "fundQ_eps": float(eps_val) if np.isfinite(eps_val) else np.nan,
                    "fundQ_assets": float(assets) if np.isfinite(assets) else np.nan,
                    "fundQ_liabilities": float(liab) if np.isfinite(liab) else np.nan,
                    "fundQ_equity": float(equity) if np.isfinite(equity) else np.nan,
                    "fundQ_total_debt": float(debt) if np.isfinite(debt) else np.nan,
                    "fundQ_cash": float(cash) if np.isfinite(cash) else np.nan,
                    "fundQ_curr_assets": float(curr_assets) if np.isfinite(curr_assets) else np.nan,
                    "fundQ_curr_liab": float(curr_liab) if np.isfinite(curr_liab) else np.nan,
                    "fundQ_ocf": float(ocf) if np.isfinite(ocf) else np.nan,
                    "fundQ_fcf": float(fcf) if np.isfinite(fcf) else np.nan,
                })

            qdf = pd.DataFrame(rows).set_index("date").sort_index()

            # ratios básicos
            eps = 1e-12
            qdf["fundQ_net_margin"]    = qdf["fundQ_net_income"] / (qdf["fundQ_revenue"].abs() + eps)
            qdf["fundQ_ebitda_margin"] = qdf["fundQ_ebitda"] / (qdf["fundQ_revenue"].abs() + eps)
            qdf["fundQ_gross_margin"]  = qdf["fundQ_gross_profit"] / (qdf["fundQ_revenue"].abs() + eps)
            qdf["fundQ_op_margin"]     = qdf["fundQ_op_income"] / (qdf["fundQ_revenue"].abs() + eps)
            qdf["fundQ_debt_to_equity"]= qdf["fundQ_total_debt"] / (qdf["fundQ_equity"].abs() + eps)
            qdf["fundQ_cash_to_debt"]  = qdf["fundQ_cash"] / (qdf["fundQ_total_debt"].abs() + eps)
            qdf["fundQ_fcf_margin"]    = qdf["fundQ_fcf"] / (qdf["fundQ_revenue"].abs() + eps)

            # ROE, ROA (annualized: multiply by 4 since quarterly)
            qdf["fundQ_roe"] = 4.0 * qdf["fundQ_net_income"] / (qdf["fundQ_equity"].abs() + eps)
            qdf["fundQ_roa"] = 4.0 * qdf["fundQ_net_income"] / (qdf["fundQ_assets"].abs() + eps)

            # Current ratio (liquidity)
            qdf["fundQ_current_ratio"] = qdf["fundQ_curr_assets"] / (qdf["fundQ_curr_liab"].abs() + eps)

            # growth q/q
            qdf["fundQ_rev_qoq"] = qdf["fundQ_revenue"].pct_change()
            qdf["fundQ_ni_qoq"]  = qdf["fundQ_net_income"].pct_change()
            qdf["fundQ_eps_qoq"] = qdf["fundQ_eps"].pct_change()

            # growth YoY (pct_change(4) = same quarter last year)
            if len(qdf) >= 4:
                qdf["fundQ_rev_yoy"]  = qdf["fundQ_revenue"].pct_change(4)
                qdf["fundQ_ni_yoy"]   = qdf["fundQ_net_income"].pct_change(4)
                qdf["fundQ_eps_yoy"]  = qdf["fundQ_eps"].pct_change(4)
                qdf["fundQ_ebitda_yoy"] = qdf["fundQ_ebitda"].pct_change(4)
            else:
                for c in ["fundQ_rev_yoy", "fundQ_ni_yoy", "fundQ_eps_yoy", "fundQ_ebitda_yoy"]:
                    qdf[c] = np.nan

            # TTM aggregates (trailing 12 months = sum of last 4 quarters)
            for base_col, ttm_col in [
                ("fundQ_revenue", "fundQ_rev_ttm"),
                ("fundQ_net_income", "fundQ_ni_ttm"),
                ("fundQ_ebitda", "fundQ_ebitda_ttm"),
                ("fundQ_fcf", "fundQ_fcf_ttm"),
                ("fundQ_ocf", "fundQ_ocf_ttm"),
            ]:
                if base_col in qdf.columns:
                    qdf[ttm_col] = qdf[base_col].rolling(4, min_periods=2).sum()

            # TTM margins
            if "fundQ_rev_ttm" in qdf.columns:
                rev_ttm_abs = qdf["fundQ_rev_ttm"].abs() + eps
                if "fundQ_ni_ttm" in qdf.columns:
                    qdf["fundQ_net_margin_ttm"] = qdf["fundQ_ni_ttm"] / rev_ttm_abs
                if "fundQ_ebitda_ttm" in qdf.columns:
                    qdf["fundQ_ebitda_margin_ttm"] = qdf["fundQ_ebitda_ttm"] / rev_ttm_abs
                if "fundQ_fcf_ttm" in qdf.columns:
                    qdf["fundQ_fcf_margin_ttm"] = qdf["fundQ_fcf_ttm"] / rev_ttm_abs

            # Clip extreme growth ratios to prevent inf/extreme outliers
            growth_cols = [c for c in qdf.columns if "_qoq" in c or "_yoy" in c]
            for gc in growth_cols:
                qdf[gc] = qdf[gc].clip(-5.0, 10.0)  # -500% to +1000%

            _FUND_Q_CACHE[tk] = qdf
            return qdf

        except Exception:
            qdf = pd.DataFrame()
            _FUND_Q_CACHE[tk] = qdf
            return qdf

    def fundamentals_daily_for_ticker(tk: str, daily_index: pd.DatetimeIndex, shift_features: int = 1) -> pd.DataFrame:
        out = pd.DataFrame(index=daily_index)

        # (A) quarterly histórico (recomendado)
        if ADD_FUNDAMENTALS_QUARTERLY:
            qdf = fetch_quarterly_fundamentals_df(tk)
            if qdf is not None and not qdf.empty:
                # reindex diário + ffill
                q_daily = qdf.reindex(daily_index, method="ffill")
                out = pd.concat([out, q_daily], axis=1)

        # (B) snapshot atual (leakage no treino)
        if ADD_FUNDAMENTALS_SNAPSHOT:
            snap = fetch_snapshot_fundamentals(tk)
            if snap:
                for k, v in snap.items():
                    out[k] = v

        # shift p/ alinhar com seu esquema de features
        if shift_features > 0:
            out = out.shift(shift_features)

        # tipos
        if out.shape[1] > 0:
            out = to_float32_safe(out.replace([np.inf, -np.inf], np.nan))

        return out


    MIN_TARGET_RATE = 0.01  # pelo menos 1% de positivos
    MAX_TARGET_RATE = 0.98  # no máximo 98% de positivos

    skip_reasons = Counter()
    valid_tickers = []
    tickers_before_deg = tickers.copy()

    print("Pre-filtering tickers (targets degeneracy)...")
    for tk in tqdm(tickers, desc="Pre-filtering tickers"):
        try:
            ohlcv = data.xs(tk, level=1, axis=1).copy()

            have = set(ohlcv.columns.astype(str))
            req = {'Close', 'High', 'Low'}
            if not req.issubset(have):
                skip_reasons['prefilter_missing_basic_ohlc'] += 1
                continue

            # Proteção: ticker veio mas está “vazio”
            if ohlcv[['Close', 'High', 'Low']].isna().all().all():
                skip_reasons['prefilter_all_nan_prices'] += 1
                continue

            # Targets direto do OHLCV (sem depender de df/X/y)
            y_up20, y_dd5 = make_targets_up_down(
                ohlcv, horizon=HORIZON, up_thr=UP_THR, dd_thr=DD_THR, keep_order=False
            )

            y_up20 = y_up20.fillna(0).astype('int8')
            y_dd5  = y_dd5.fillna(0).astype('int8')

            rate_up = float(y_up20.mean()) if len(y_up20) else 0.0
            rate_dd = float(y_dd5.mean()) if len(y_dd5) else 0.0

            if not (MIN_TARGET_RATE <= rate_up <= MAX_TARGET_RATE):
                skip_reasons[f"degenerate_up20_{tk}"] += 1
                continue
            if not (MIN_TARGET_RATE <= rate_dd <= MAX_TARGET_RATE):
                skip_reasons[f"degenerate_dd5_{tk}"] += 1
                continue

            valid_tickers.append(tk)

        except Exception as e:
            skip_reasons[f"prefilter_exception:{type(e).__name__}"] += 1
            continue

    tickers = np.array(valid_tickers, dtype=object)
    deg_ok_set = set(tickers)
    before_deg_set = set(tickers_before_deg)

    dropped_by_degeneracy = sorted(before_deg_set - deg_ok_set)
    print(f"[AUDIT] after_degeneracy={len(deg_ok_set)} | dropped_by_degeneracy={len(dropped_by_degeneracy)}")
    if dropped_by_degeneracy[:20]:
        print("  ex:", dropped_by_degeneracy[:20])

    print(f"Tickers after degeneracy filter: {len(tickers)}")

    # Log resumido (não explode o output com 200 chaves diferentes)
    top = skip_reasons.most_common(12)
    if top:
        print("Pre-filter skip reasons (top 12):")
        for k, v in top:
            print(f"  - {k}: {v}")

    # A partir daqui, reinicia motivos de skip para a construção principal
    skip_reasons = Counter()



    # ============================
    # Construção por ticker
    # ============================
    DEBUG_STOP_ON_FIRST_TICKER_ERROR = os.getenv("DEBUG_STOP_ON_FIRST_TICKER_ERROR", "0") == "1"

    feat_frames = []
    tgt_frames  = []
    skip_reasons = Counter()

    print("Gerando features e targets por ticker...")
    for tk in tqdm(tickers):
        try:
            ohlcv = data.xs(tk, level=1, axis=1).copy()

            have = set(ohlcv.columns.astype(str))
            req = {'Close', 'High', 'Low'}
            if not req.issubset(have):
                skip_reasons['missing_basic_ohlc'] += 1
                continue

            # Fallbacks to avoid skipping indices/FX/crypto
            if 'Adj Close' not in have:
                ohlcv['Adj Close'] = to_float32_safe(ohlcv['Close'])
            if 'Open' not in have:
                ohlcv['Open'] = to_float32_safe(ohlcv['Close'])
            if 'Volume' not in have:
                ohlcv['Volume'] = 0.0

            # Extra safety: skip tickers with no real price data
            if ohlcv[['Close','High','Low']].isna().all().all():
                skip_reasons['all_nan_prices'] += 1
                continue

            # Build features
            feats = indicators_for_ticker(ohlcv, shift_features=SHIFT_FEATURES)
            pats  = patterns_for_ticker(ohlcv, shift_features=SHIFT_FEATURES)
            if feats is None or feats.shape[1] == 0:
                skip_reasons['empty_feats'] += 1
                continue

            fund = fundamentals_daily_for_ticker(tk, ohlcv.index, shift_features=SHIFT_FEATURES)

            X_tk = pd.concat([feats, pats, fund], axis=1)
            X_tk = fill_100pct(X_tk, allow_bfill=ALLOW_BFILL_EXOGENOUS)
            if X_tk.shape[1] == 0:
                skip_reasons['empty_after_fill'] += 1
                continue

            # Attach ticker level
            X_tk.columns = pd.MultiIndex.from_product([X_tk.columns, [tk]])
            feat_frames.append(X_tk)

            # ---- Targets (IMPORTANT: cast/fill BEFORE appending) ----
            y_up20, y_dd5 = make_targets_up_down(
                ohlcv, horizon=HORIZON, up_thr=UP_THR, dd_thr=DD_THR, keep_order=False
            )
            y_up20 = y_up20.fillna(0).astype('int8').rename(('target_up20', tk))
            y_dd5  = y_dd5.fillna(0).astype('int8').rename(('target_dd5',  tk))

            y_entry, y_sale = make_best_entry_sale(ohlcv, horizon=HORIZON)
            y_entry = to_float32_safe(y_entry).rename(('target_best_entry', tk))
            y_sale  = to_float32_safe(y_sale).rename(('target_best_sale',  tk))

            tgt_frames.extend([y_up20, y_dd5, y_entry, y_sale])

        except Exception as e:
            skip_reasons[f'exception:{type(e).__name__}'] += 1
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                last = tb[-1]
                if len(tb) > 1:
                    prev = tb[-2]
                    print(f"\n[{tk}] {type(e).__name__}: {e} @ {last.name}:{last.lineno} (caller: {prev.name}:{prev.lineno})")
                else:
                    print(f"\n[{tk}] {type(e).__name__}: {e} @ {last.name}:{last.lineno}")
            else:
                print(f"\n[{tk}] {type(e).__name__}: {e}")
            if DEBUG_STOP_ON_FIRST_TICKER_ERROR:
                raise
            continue

    # Safety + diagnostics
    if len(feat_frames) == 0:
        raise RuntimeError(f"No features built for any ticker. Skip reasons: {dict(skip_reasons)}")
    else:
        print("Tickers processados:", len(feat_frames), "| Motivos de skip:", dict(skip_reasons))


    # Concat globais
    X = pd.concat(feat_frames, axis=1).sort_index()
    y = pd.concat(tgt_frames,  axis=1).sort_index()

    # Cast consistente (atualizado p/ novos targets)
    X = cast_int8_multi(X, prefixes=("cross_",), suffixes=("pattern",))
    other0 = [c for c in X.columns.get_level_values(0)
              if not (str(c).startswith("cross_") or str(c).endswith("pattern"))]
    if other0:
        X.loc[:, (other0, slice(None))] = to_float32_safe(X.loc[:, (other0, slice(None))])

    for tname in ('target_up20','target_dd5'):
        if tname in y.columns.get_level_values(0):
            y.loc[:, (tname, slice(None))] = y.loc[:, (tname, slice(None))].astype('int8')
    for tname in ('target_best_entry','target_best_sale'):
        if tname in y.columns.get_level_values(0):
            y.loc[:, (tname, slice(None))] = to_float32_safe(y.loc[:, (tname, slice(None))])


    # Garantir 100% preenchido (inf/NaN) globalmente (features já estão ok; segurança adicional)
    X = fill_100pct(X, allow_bfill=ALLOW_BFILL_EXOGENOUS)
    y = fill_100pct(y, allow_bfill=True)

    # Verificações finais
    assert not X.isna().any().any(), "Ainda há NaN em X após preenchimento!"
    assert not y.isna().any().any(), "Ainda há NaN em y após preenchimento!"

    # ============================
    # Saída única combinada
    # ============================
    DATASET = pd.concat([X, y], axis=1).sort_index()
    # Segurança final (caso algum merge crie lacunas):
    DATASET = fill_100pct(DATASET, allow_bfill=ALLOW_BFILL_EXOGENOUS)

    # Sanidade: sem NaN
    assert not DATASET.isna().any().any(), "Ainda há NaN no dataset final!"

    # ============================
    # Feature reduction (per ticker, MI vs targets)
    # ============================
    # y_up and y_dd views (ensure they exist)
    y_up = y.loc[:, y.columns.get_level_values(0) == 'target_up20']
    y_dd = y.loc[:, y.columns.get_level_values(0) == 'target_dd5']

    X_reduced = reduce_features_automatic(
        X, y_up, y_dd,
        top_fraction=0.85,   # pega ~85% por MI antes de deduplicar
        min_keep=96,         # garanta pelo menos ~100 por ticker (se existirem)
        var_thr=None,        # não remova por variância agora
        corr_thr=0.9995,     # só remove quase idênticos
        always_keep_prefixes=("pct_change","SMA_","tri_","sr_","Close","Adj Close","close"),  # Close obrigatório para reg
        verbose=True
    )



    # ============================
    # Output (full and reduced)
    # ============================
    DATASET_FULL = pd.concat([X, y], axis=1).sort_index()
    DATASET_FULL = fill_100pct(DATASET_FULL, allow_bfill=ALLOW_BFILL_EXOGENOUS)
    assert not DATASET_FULL.isna().any().any(), "NaN in full dataset!"

    DATASET_REDUCED = pd.concat([X_reduced, y], axis=1).sort_index()
    DATASET_REDUCED = fill_100pct(DATASET_REDUCED, allow_bfill=ALLOW_BFILL_EXOGENOUS)
    assert not DATASET_REDUCED.isna().any().any(), "NaN in reduced dataset!"

    if SAVE_PARQUET:
        DATASET_FULL = normalize_before_save(DATASET_FULL)
        DATASET_FULL.to_parquet(OUTPUT_PATH, compression="snappy")
        OUT_REDUCED = OUTPUT_PATH.replace(".parquet", "_reduced.parquet")
        DATASET_REDUCED = normalize_before_save(DATASET_REDUCED)
        DATASET_REDUCED.to_parquet(OUT_REDUCED, compression="snappy")
        print(f"Saved FULL:     {OUTPUT_PATH}  shape={DATASET_FULL.shape}")
        print(f"Saved REDUCED:  {OUT_REDUCED}  shape={DATASET_REDUCED.shape}")
    else:
        print("FULL:", DATASET_FULL.shape, " | REDUCED:", DATASET_REDUCED.shape)

    print("X full:", X.shape)
    print("y:", y.shape)
    print("X_reduced:", X_reduced.shape)
    def tickers_in_dataset_cols(df: pd.DataFrame) -> set:
        if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
            return set()
        return set(df.columns.get_level_values(1).astype(str))

    expected_saved = tickers_in_dataset_cols(DATASET_FULL)
    print("[AUDIT] tickers in-memory FULL:", len(expected_saved))

    # Round-trip read
    re_full = pd.read_parquet(OUTPUT_PATH)
    full_file = tickers_in_dataset_cols(re_full)

    missing_in_file = sorted(expected_saved - full_file)
    extra_in_file   = sorted(full_file - expected_saved)

    print(f"[AUDIT] tickers in-file FULL: {len(full_file)}")
    print(f"[AUDIT] missing_after_save: {len(missing_in_file)} | extra_after_save: {len(extra_in_file)}")
    if missing_in_file[:20]:
        print("  missing ex:", missing_in_file[:20])

    # Reduced
    re_red = pd.read_parquet(OUT_REDUCED)
    red_file = tickers_in_dataset_cols(re_red)
    expected_red = tickers_in_dataset_cols(DATASET_REDUCED)

    miss_red = sorted(expected_red - red_file)
    print(f"[AUDIT] tickers in-file REDUCED: {len(red_file)} | missing_after_save_reduced: {len(miss_red)}")
    if miss_red[:20]:
        print("  missing reduced ex:", miss_red[:20])

    # opcional: falhar duro se perder ticker no save
    assert len(missing_in_file) == 0, "Perdeu tickers ao salvar FULL (round-trip falhou)!"
    assert len(miss_red) == 0, "Perdeu tickers ao salvar REDUCED (round-trip falhou)!"

    # Per-ticker kept counts
    kept_counts = pd.Series(dict(
        (tk, X_reduced.xs(tk, level=1, axis=1).shape[1])
        for tk in np.unique(X_reduced.columns.get_level_values(1))
    )).sort_values(ascending=True)
    print("Min/median/max kept per ticker:", kept_counts.min(), kept_counts.median(), kept_counts.max())

    processed = np.unique(X.columns.get_level_values(1))
    print("Processed tickers:", len(processed))
    missing = sorted(set(tickers) - set(processed))
    print("Missing tickers:", len(missing))

    DATASET_REDUCED.tail()

    # ============================
    # Diagnóstico de targets (0/1)
    # ============================

    def print_target_rates(y, targets=("target_up20", "target_dd5"), topk=10):
        if not isinstance(y.columns, pd.MultiIndex):
            raise ValueError("Esperado y com colunas MultiIndex (target, ticker).")

        lvl0 = y.columns.get_level_values(0).astype(str)

        for t in targets:
            if t not in set(lvl0):
                print(f"[WARN] Target '{t}' não encontrado em y.")
                continue

            yt = y.loc[:, lvl0 == t].copy()  # cols = (t, ticker)
            yt = yt.apply(pd.to_numeric, errors="coerce").fillna(0).astype("int8")

            # --- geral (todas as datas e tickers empilhados) ---
            arr = yt.to_numpy().ravel()
            rate1 = float(arr.mean()) if arr.size else np.nan
            n1 = int(arr.sum()) if arr.size else 0
            n = int(arr.size)

            print("\n" + "="*60)
            print(f"[{t}] Geral")
            print(f"N total: {n:,} | N(1): {n1:,} | N(0): {n-n1:,}")
            print(f"Taxa(1): {100*rate1:.2f}% | Taxa(0): {100*(1-rate1):.2f}%")

            # --- por ticker ---
            # mean por coluna => taxa de 1 por ticker
            per_ticker_rate1 = yt.mean(axis=0)
            per_ticker_rate1.index = per_ticker_rate1.index.get_level_values(1).astype(str)

            print(f"\n[{t}] Taxa(1) por ticker (top {topk} MAIORES):")
            print((per_ticker_rate1.sort_values(ascending=False).head(topk) * 100).round(2).astype(str) + "%")

            print(f"\n[{t}] Taxa(1) por ticker (top {topk} MENORES):")
            print((per_ticker_rate1.sort_values(ascending=True).head(topk) * 100).round(2).astype(str) + "%")

    print_target_rates(y, targets=("target_up20","target_dd5"), topk=12)




if __name__ == '__main__':
    run_etl()

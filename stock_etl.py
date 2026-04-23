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
    AVERAGES = [1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200]

    # Horizonte para cálculo de alvos
    HORIZON = 90
    UP_THR = 0.25   # +30%
    DD_THR = -0.10  # -10%


    SAVE_PARQUET = True
    SAVE_CSV_FALLBACK = False
    OUTPUT_PATH = "./output/data/expanded_stock.parquet"
    SKIP_MI_REDUCTION = True   # skip slow MI-based feature reduction; GA does its own Spearman selection

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

        # Construção civil (adicionados)
        "TCSA3.SA",

        # ETF internacional (adicionado)
        "IVVB11.SA",

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
        "KNCR11.SA", "KNSC11.SA", "KNHY11.SA", "CPTS11.SA", "IRDM11.SA", "IRIM11.SA",
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

        # v8 feature engineering: ETFs-proxy para regime/flow
        "EWZ",      # iShares MSCI Brazil — proxy foreign flow B3
        "EEM",      # MSCI Emerging Markets
        # Sector ETFs (Tier 2E: sector rotation)
        "XLE",      # Energy (proxy oil/Petrobras/Vale)
        "XLF",      # Financial (proxy banks like Itau/Bradesco)
        "XLI",      # Industrial (proxy WEG/Embraer)
        "XLB",      # Materials (proxy mining/Vale)
        "XLY",      # Consumer Discretionary
        "XLP",      # Consumer Staples
        "XLU",      # Utilities (proxy Eletrobras/CEMIG)
        "XLK",      # Technology
        "XLV",      # Healthcare
        "XLRE",     # Real Estate
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
    # yfinance uses sqlite-backed caches (timezone/cookies/ISIN) under the user's
    # cache directory. Some environments restrict writing there, causing
    # "OperationalError: unable to open database file". Pin caches to workspace.
    try:
        yf_cache_dir = os.path.abspath("./output/yfinance_cache")
        os.makedirs(yf_cache_dir, exist_ok=True)
        yf.cache.set_cache_location(yf_cache_dir)
    except Exception:
        pass

    data = yf.download(
        ALL_TICKERS,
        start=START_DATE,
        group_by='column',
        auto_adjust=True,
        progress=False,
        # Use single-threaded downloads to avoid sqlite cache contention
        # ("database is locked") on some Windows setups.
        threads=False,
    )

    # Somente colunas OHLCV relevantes e limpeza de levels
    allowed_columns = ['Open','High','Low','Close','Adj Close','Volume']
    data = data.loc[:, data.columns.get_level_values(0).isin(allowed_columns)].copy()
    data.columns = data.columns.remove_unused_levels()

    # ATENCAO: NAO forward-fill OHLCV - isso criaria precos fantasmas em
    # fins de semana e feriados (Apr 4-5 2026 etc). Cada ticker deve ter
    # close=NaN em dias nao-negociados; o apply phase filtra esses dias.
    # Features derivadas ainda podem usar ffill (veja fill_100pct abaixo).
    # data = data.ffill()  # REMOVIDO (gerava precos falsos em weekends/feriados)

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

    print(f"Período: {data.index.min().date()} -> {data.index.max().date()}")
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

        # Keep the last `horizon` rows as NaN: there is not enough future path
        # to assign a fully observed label without right-censoring.
        up_hit = np.full(n, np.nan, dtype=np.float32)
        dd_hit = np.full(n, np.nan, dtype=np.float32)
        order  = np.full(n, np.nan, dtype=np.float32)

        for t in range(n):
            if (t + horizon) >= n:
                continue
            end = t + horizon + 1

            entry = close[t]
            up_th = entry * (1.0 + up_thr)
            dd_th = entry * (1.0 + dd_thr)

            hseg = high[t+1:end]
            lseg = low[t+1:end]

            up_idx = np.where(hseg >= up_th)[0]
            dd_idx = np.where(lseg <= dd_th)[0]

            hit_up = up_idx[0] if len(up_idx) else None
            hit_dd = dd_idx[0] if len(dd_idx) else None

            up_hit[t] = 1.0 if hit_up is not None else 0.0
            dd_hit[t] = 1.0 if hit_dd is not None else 0.0

            if keep_order:
                if hit_up is None and hit_dd is None:
                    order[t] = -1.0
                elif hit_up is None:
                    order[t] = 0.0
                elif hit_dd is None:
                    order[t] = 1.0
                else:
                    order[t] = 1.0 if hit_up < hit_dd else 0.0

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

        # Cruzamentos CONTÍNUOS: distância % entre médias (float32, não binário)
        # Binários tinham |r|~0.01. Contínuos têm |r|~0.03 (3x melhor).
        # Apenas os 10 pares mais informativos (economia de memória)
        _KEY_CROSS_PAIRS = [(5,20),(5,50),(10,20),(10,50),(20,50),(20,100),(20,200),(50,100),(50,200),(100,200)]
        for fast, slow in _KEY_CROSS_PAIRS:
            fcol, scol = f'SMA_{fast}', f'SMA_{slow}'
            if fcol in feats.columns and scol in feats.columns:
                feats[f'cross_{fast}_{slow}'] = to_float32_safe(
                    ((feats[fcol] - feats[scol]) / feats[scol].replace(0, np.nan)).clip(-0.20, 0.20)
                )

        feats['pct_change'] = ohlcv_num[close_col].pct_change()

        # ── SMA distance features (continuous, replaces binary crosses) ──
        # Binary crosses have |r| ~0.01 (useless). Distance between SMAs
        # is continuous and measures trend strength, not just crossing events.
        _close_for_sma = ohlcv_num[close_col]
        for fast, slow in [(5,20), (10,50), (20,50), (20,200), (50,200)]:
            fcol, scol = f'SMA_{fast}', f'SMA_{slow}'
            if fcol in feats.columns and scol in feats.columns:
                # Distance: (SMA_fast - SMA_slow) / SMA_slow (positive = bullish trend)
                feats[f'sma_dist_{fast}_{slow}'] = (
                    (feats[fcol] - feats[scol]) / feats[scol].replace(0, np.nan)
                ).clip(-0.15, 0.15)
                # Acceleration: change in distance over 5 days (trend strengthening/weakening)
                feats[f'sma_accel_{fast}_{slow}'] = feats[f'sma_dist_{fast}_{slow}'].diff(5)
        # Price relative to key SMAs (continuous, not binary above/below)
        for avg in [20, 50, 200]:
            scol = f'SMA_{avg}'
            if scol in feats.columns:
                feats[f'price_vs_sma{avg}'] = (
                    (_close_for_sma - feats[scol]) / feats[scol].replace(0, np.nan)
                ).clip(-0.3, 0.3)

        # ── STRATEGY 1: Vol-normalized features (boost existing by +10-20% |r|) ──
        _cl = ohlcv_num[close_col]
        _hi = ohlcv_num['High']
        _lo = ohlcv_num['Low']
        _vo = ohlcv_num['Volume']
        _vol20 = _cl.pct_change().rolling(20, min_periods=5).std().replace(0, np.nan)
        _atr_pct = ((_hi - _lo) / _cl.replace(0, np.nan)).rolling(14, min_periods=5).mean().replace(0, np.nan)

        # Vol-adjusted momentum (ROC / volatility)
        for period in [5, 10, 20]:
            _roc = (_cl - _cl.shift(period)) / _cl.shift(period).replace(0, np.nan)
            feats[f'roc_voladj_{period}'] = to_float32_safe((_roc / _vol20).clip(-5, 5))

        # Vol-adjusted RSI
        if 'momentum_rsi' in feats.columns:
            feats['rsi_voladj'] = to_float32_safe(((feats['momentum_rsi'] - 50) / 50 / _vol20).clip(-5, 5))

        # Vol-adjusted volume surge
        _vol_ratio = _vo / _vo.rolling(20, min_periods=5).mean().replace(0, np.nan)
        feats['volume_surge_voladj'] = to_float32_safe((_vol_ratio / _atr_pct).clip(0, 20))

        # ── STRATEGY 2: Feature interactions (capture non-linearities) ──
        # RSI oversold + high volume = capitulation bottom
        if 'momentum_rsi' in feats.columns:
            _rsi_oversold = ((50 - feats['momentum_rsi']).clip(0, 50) / 50)
            feats['rsi_x_volume'] = to_float32_safe(_rsi_oversold * _vol_ratio.clip(0, 5))

        # Momentum × trend alignment
        for sma_p in [50, 200]:
            scol = f'SMA_{sma_p}'
            if scol in feats.columns:
                _trend = ((_cl - feats[scol]) / feats[scol].replace(0, np.nan)).clip(-0.3, 0.3)
                _mom = ((_cl - _cl.shift(10)) / _cl.shift(10).replace(0, np.nan)).clip(-0.2, 0.2)
                feats[f'mom_x_trend{sma_p}'] = to_float32_safe(_trend * _mom * 10)

        # Breakout × volume confirmation
        _price_near_high = ((_cl - _lo.rolling(20).min()) /
                           (_hi.rolling(20).max() - _lo.rolling(20).min()).replace(0, np.nan)).clip(0, 1)
        feats['breakout_x_volume'] = to_float32_safe(_price_near_high * _vol_ratio.clip(0, 5))

        # ── Phase 3.3: Microstructure features ──
        _close = ohlcv_num[close_col]
        _volume = ohlcv_num['Volume']
        _high = ohlcv_num['High']
        _low = ohlcv_num['Low']
        _open = ohlcv_num['Open']

        # Volume relative (volume / 20-day avg)
        vol_avg_20 = _volume.rolling(20, min_periods=5).mean()
        feats['vol_relative_20d'] = _volume / vol_avg_20.replace(0, np.nan)

        # Amihud illiquidity ratio: |return| / dollar_volume (rolling 20d avg)
        abs_ret = _close.pct_change().abs()
        dollar_vol = (_close * _volume).replace(0, np.nan)
        amihud_raw = abs_ret / dollar_vol
        feats['amihud_illiquidity_20d'] = amihud_raw.rolling(20, min_periods=5).mean()

        # Garman-Klass volatility (uses OHLC, better than close-to-close)
        log_hl = np.log(_high / _low.replace(0, np.nan)) ** 2
        log_co = np.log(_close / _open.replace(0, np.nan)) ** 2
        gk_daily = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        feats['gk_volatility_20d'] = gk_daily.rolling(20, min_periods=5).mean().apply(np.sqrt)

        # ── Minervini Trend Template (SEPA) criteria ──────────────────────
        # All 7 criteria from Mark Minervini's momentum strategy
        sma50  = feats.get('SMA_50',  _close.rolling(50, min_periods=30).mean())
        sma150 = feats.get('SMA_150', _close.rolling(150, min_periods=100).mean())
        sma200 = feats.get('SMA_200', _close.rolling(200, min_periods=130).mean())

        # 52-week (252 trading days) high and low
        high_52w = _high.rolling(252, min_periods=126).max()
        low_52w  = _low.rolling(252, min_periods=126).min()

        # 1. Price > SMA150 AND Price > SMA200
        feats['miner_price_above_150_200'] = ((_close > sma150) & (_close > sma200)).astype('int8')

        # 2. SMA150 > SMA200
        feats['miner_sma150_above_200'] = (sma150 > sma200).astype('int8')

        # 3. SMA200 trending up for at least 1 month (20 trading days)
        sma200_1m_ago = sma200.shift(20)
        feats['miner_sma200_trending_up'] = (sma200 > sma200_1m_ago).astype('int8')
        # Stronger version: trending up for 4-5 months (~100 trading days)
        sma200_5m_ago = sma200.shift(100)
        feats['miner_sma200_trend_5m'] = (sma200 > sma200_5m_ago).astype('int8')

        # 4. SMA50 > SMA150 AND SMA50 > SMA200
        feats['miner_sma50_above_150_200'] = ((sma50 > sma150) & (sma50 > sma200)).astype('int8')

        # 5. Price > SMA50
        feats['miner_price_above_50'] = (_close > sma50).astype('int8')

        # 6. Price >= 30% above 52-week low
        pct_above_low = (_close - low_52w) / low_52w.replace(0, np.nan)
        feats['miner_pct_above_52w_low'] = to_float32_safe(pct_above_low)
        feats['miner_30pct_above_low'] = (pct_above_low >= 0.30).astype('int8')

        # 7. Price within 25% of 52-week high (closer to high = better)
        pct_from_high = (high_52w - _close) / high_52w.replace(0, np.nan)
        feats['miner_pct_from_52w_high'] = to_float32_safe(pct_from_high)
        feats['miner_within_25pct_high'] = (pct_from_high <= 0.25).astype('int8')

        # COMPOSITE: All 7 Minervini criteria pass = 1 (strong trend template)
        feats['miner_trend_template'] = (
            feats['miner_price_above_150_200'] &
            feats['miner_sma150_above_200'] &
            feats['miner_sma200_trending_up'] &
            feats['miner_sma50_above_150_200'] &
            feats['miner_price_above_50'] &
            feats['miner_30pct_above_low'] &
            feats['miner_within_25pct_high']
        ).astype('int8')

        # Count of how many criteria pass (0-7, useful as continuous signal)
        feats['miner_criteria_count'] = (
            feats['miner_price_above_150_200'] +
            feats['miner_sma150_above_200'] +
            feats['miner_sma200_trending_up'] +
            feats['miner_sma50_above_150_200'] +
            feats['miner_price_above_50'] +
            feats['miner_30pct_above_low'] +
            feats['miner_within_25pct_high']
        ).astype('int8')

        # ── VCP (Volatility Contraction Pattern) — Minervini timing ──
        # VCP measures if volatility is CONTRACTING (each dip smaller than last)
        # This is the key timing signal: enter when VCP is tight + Stage 2
        for vcp_w in [20, 40]:
            # Range contraction: recent range vs prior range
            _rng_recent = _high.rolling(vcp_w//2, min_periods=3).max() - _low.rolling(vcp_w//2, min_periods=3).min()
            _rng_prior = _high.shift(vcp_w//2).rolling(vcp_w//2, min_periods=3).max() - _low.shift(vcp_w//2).rolling(vcp_w//2, min_periods=3).min()
            _contraction = (_rng_recent / _rng_prior.replace(0, np.nan)).clip(0.1, 2.0)
            feats[f'vcp_contraction_{vcp_w}d'] = to_float32_safe(_contraction)

            # Number of contracting dips (each low higher than previous low)
            _dip_size = (_close - _low.rolling(5, min_periods=2).min()) / _close.replace(0, np.nan)
            _prev_dip = _dip_size.shift(vcp_w//4)
            _contracting = (_dip_size < _prev_dip).rolling(vcp_w//4, min_periods=2).mean()
            feats[f'vcp_dip_contracting_{vcp_w}d'] = to_float32_safe(_contracting)

        # ── Minervini Stage Detection (continuous, 0-4) ──
        # Stage 2 (markup) is the buying zone: all MAs aligned + trending
        _stage = pd.Series(0.0, index=_close.index)
        # Stage 2 indicators (higher = more Stage 2 like)
        _s2_score = (
            (_close > sma50).astype(float) * 0.15 +
            (_close > sma150).astype(float) * 0.15 +
            (_close > sma200).astype(float) * 0.15 +
            (sma50 > sma150).astype(float) * 0.15 +
            (sma150 > sma200).astype(float) * 0.15 +
            (sma200 > sma200.shift(20)).astype(float) * 0.10 +  # MA200 trending up
            (pct_from_high <= 0.25).astype(float) * 0.15       # within 25% of high
        )
        feats['miner_stage2_score'] = to_float32_safe(_s2_score)

        # ── TIMING: VCP + Stage 2 = optimal entry ──
        # Interaction: contraction happening IN Stage 2 = Minervini buy signal
        feats['vcp_x_stage2'] = to_float32_safe(
            (1.0 - feats.get('vcp_contraction_20d', pd.Series(1.0, index=_close.index)).clip(0.3, 1.0))
            * _s2_score * 3.0
        )

        # Additional useful derived features
        # Distance from 52w high (0 = at high, negative = below)
        feats['dist_52w_high_pct'] = to_float32_safe((_close - high_52w) / high_52w.replace(0, np.nan))
        # Distance from 52w low (0 = at low, positive = above)
        feats['dist_52w_low_pct'] = to_float32_safe(pct_above_low)
        # SMA slope indicators
        feats['sma50_slope_20d'] = to_float32_safe((sma50 - sma50.shift(20)) / sma50.shift(20).replace(0, np.nan))
        feats['sma150_slope_20d'] = to_float32_safe((sma150 - sma150.shift(20)) / sma150.shift(20).replace(0, np.nan))
        feats['sma200_slope_20d'] = to_float32_safe((sma200 - sma200.shift(20)) / sma200.shift(20).replace(0, np.nan))

        if shift_features > 0:
            # ROOT FIX: shift APENAS colunas derivadas (indicadores tecnicos),
            # NAO as colunas OHLCV base. OHLCV base deve ficar no Date=T real
            # para que downstream (ga_run, xlsx Apply sheet) reporte o close
            # correto da barra. Shiftar OHLCV causava o bug historico de close
            # stale no Telegram/xlsx que exigia refresh_close.py como patch.
            #
            # Rationale: os indicadores shiftados garantem que a GA treine com
            # dados conhecidos ate o final de T-1 (evita leakage). Mas OHLCV
            # base sao fatos historicos da barra T, nao features preditivas —
            # sao a referencia temporal. Nao devem ser shiftados.
            BASE_OHLCV_COLS = {
                "Open", "High", "Low", "Close", "Adj Close", "Volume",
                "open", "high", "low", "close", "adj close", "volume",
            }
            non_base = [c for c in feats.columns if c not in BASE_OHLCV_COLS]
            if non_base:
                feats[non_base] = feats[non_base].shift(shift_features)

        # >>> FIX: garantir que colunas discretas não tenham NaN
        # cross_ agora são float contínuos, só miner_ precisa int8
        int_cols = [c for c in feats.columns if str(c).startswith("miner_")]
        if int_cols:
            feats[int_cols] = feats[int_cols].fillna(0).astype('int8')

        # floats podem ficar com NaN (você já zera depois no fill_100pct), mas pode manter assim:
        float_cols = [c for c in feats.columns if c not in int_cols]
        feats[float_cols] = to_float32_safe(feats[float_cols])

        return feats



    def patterns_for_ticker(ohlcv: pd.DataFrame, shift_features: int = 1) -> pd.DataFrame:
        """Generate CONTINUOUS pattern features (not binary) for ML.

        Key insight: binary pattern detection (0/1) has near-zero Spearman correlation
        with future returns. Continuous features measuring proximity, strength, and
        context of chart structures are much more predictive.
        """
        c = ohlcv['Close'].values.astype(float)
        h = ohlcv['High'].values.astype(float)
        l = ohlcv['Low'].values.astype(float)
        v = ohlcv['Volume'].values.astype(float) if 'Volume' in ohlcv.columns else np.ones(len(c))
        n = len(c)
        idx = ohlcv.index
        out = pd.DataFrame(index=idx)

        # Helper: rolling on array → Series with correct index
        def _roll(arr, w, fn='max', min_p=None):
            s = pd.Series(arr, index=idx)
            mp = min_p or max(3, w // 3)
            if fn == 'max': return s.rolling(w, min_periods=mp).max()
            elif fn == 'min': return s.rolling(w, min_periods=mp).min()
            elif fn == 'mean': return s.rolling(w, min_periods=mp).mean()
            elif fn == 'std': return s.rolling(w, min_periods=mp).std()
            return s

        # 1. SUPPORT/RESISTANCE DISTANCE (best features: r=0.04)
        for w in [20, 50]:
            support = _roll(l, w, 'min').values
            resistance = _roll(h, w, 'max').values
            mid = np.maximum((support + resistance) / 2, 1e-8)
            out[f'dist_support_{w}d'] = (c - support) / mid
            out[f'dist_resistance_{w}d'] = (resistance - c) / mid

        # 2. CUP AND HANDLE (r=0.03, statistically significant)
        for cup_len in [30, 50]:
            if n >= cup_len:
                start_price = pd.Series(c, index=idx).shift(cup_len).values
                min_in_cup = _roll(l, cup_len, 'min', min_p=cup_len//2).values
                cup_depth = np.where(start_price > 0, (start_price - min_in_cup) / start_price, 0)
                recovery = np.where(
                    (start_price > 0) & ((start_price - min_in_cup) > 0),
                    (c - min_in_cup) / np.maximum(start_price - min_in_cup, 1e-8), 0)
                cup_score = np.where(
                    (cup_depth > 0.03) & (cup_depth < 0.30),
                    np.clip(recovery, 0, 1.2) * cup_depth * 10, 0)
                out[f'cup_handle_{cup_len}d'] = np.clip(cup_score, 0, 3)

        # 3. CONSOLIDATION: range compression → breakout imminent
        for w in [10, 20]:
            recent_rng = _roll(h, w, 'max') - _roll(l, w, 'min')
            prior_rng = _roll(h, w*2, 'max') - _roll(l, w*2, 'min')
            out[f'consolidation_{w}d'] = (recent_rng / prior_rng.replace(0, np.nan)).clip(0, 2).values

        # 4. HIGHER LOWS: ascending bottoms = uptrend confirmation
        for w in [10, 20]:
            recent_low = _roll(l, w, 'min')
            prior_low = _roll(l, w, 'min').shift(w)
            out[f'higher_lows_{w}d'] = ((recent_low - prior_low) / prior_low.replace(0, np.nan)).clip(-0.2, 0.2).values

        # 5. PIVOT POINTS: classic floor pivots + distance features
        #    Pivot = (High_prev + Low_prev + Close_prev) / 3
        #    The GA learns: "buy when price > pivot" via positive dist_pivot
        #    Multiple timeframes: daily (previous day), weekly (previous 5d)
        for pivot_w, label in [(1, 'daily'), (5, 'weekly')]:
            if pivot_w == 1:
                _prev_h = pd.Series(h, index=idx).shift(1).values
                _prev_l = pd.Series(l, index=idx).shift(1).values
            else:
                _prev_h = _roll(h, pivot_w, 'max').shift(1).values
                _prev_l = _roll(l, pivot_w, 'min').shift(1).values
            _prev_c = pd.Series(c, index=idx).shift(1).values
            _pivot = (_prev_h + _prev_l + _prev_c) / 3.0
            _range = _prev_h - _prev_l

            # Distance from pivot (positive = above pivot = bullish)
            _mid = np.maximum(_pivot, 1e-8)
            out[f'dist_pivot_{label}'] = ((c - _pivot) / _mid).clip(-0.10, 0.10)

            # Classic support/resistance levels from pivot
            _r1 = 2 * _pivot - _prev_l  # Resistance 1
            _s1 = 2 * _pivot - _prev_h  # Support 1
            _r2 = _pivot + _range       # Resistance 2
            _s2 = _pivot - _range       # Support 2

            # Position relative to pivot zones (0 = at S2, 0.5 = at pivot, 1 = at R2)
            _full_range = np.maximum(_r2 - _s2, 1e-8)
            out[f'pivot_zone_{label}'] = ((c - _s2) / _full_range).clip(0, 1)

            # Breakout above R1 (strong bullish signal)
            out[f'above_r1_{label}'] = ((c - _r1) / _mid).clip(-0.05, 0.10)

            # Breakdown below S1 (bearish)
            out[f'below_s1_{label}'] = ((_s1 - c) / _mid).clip(-0.05, 0.10)

        # 6. FIBONACCI RETRACEMENT: position within recent swing
        for fib_w in [20, 50]:
            _swing_h = _roll(h, fib_w, 'max').values
            _swing_l = _roll(l, fib_w, 'min').values
            _swing_range = np.maximum(_swing_h - _swing_l, 1e-8)
            # Retracement level: 0 = at low, 0.382 = fib level, 0.618 = fib level, 1 = at high
            _retrace = (c - _swing_l) / _swing_range
            out[f'fib_retrace_{fib_w}d'] = np.clip(_retrace, 0, 1)
            # Distance from key fib levels (0.382 and 0.618)
            out[f'dist_fib382_{fib_w}d'] = (_retrace - 0.382).clip(-0.5, 0.5)
            out[f'dist_fib618_{fib_w}d'] = (_retrace - 0.618).clip(-0.5, 0.5)

        # 7. VOLUME at extremes: confirmation signals
        vol_ma = _roll(v, 20, 'mean')
        vol_ratio = pd.Series(v, index=idx) / vol_ma.replace(0, np.nan)
        # Breakout proximity as base
        bp = out.get('dist_resistance_20d', pd.Series(0.0, index=idx))
        bp_inv = out.get('dist_support_20d', pd.Series(0.0, index=idx))
        # Volume surge near resistance = breakout confirmation
        out['volume_breakout_confirm'] = (vol_ratio.values * np.clip(1 - bp.values, 0, 1)).clip(0, 5)
        # Volume surge near support = capitulation (potential bottom)
        out['volume_capitulation'] = (vol_ratio.values * np.clip(1 - bp_inv.values, 0, 1)).clip(0, 5)

        if shift_features > 0:
            out = out.shift(shift_features)

        out = to_float32_safe(out)
        return out


    def patterns_weekly_monthly(ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Detect chart patterns on weekly and monthly resampled OHLC candles,
        then forward-fill back to daily frequency for alignment."""
        out = pd.DataFrame(index=ohlcv.index)
        ohlcv_idx = ohlcv.copy()
        if not isinstance(ohlcv_idx.index, pd.DatetimeIndex):
            if 'Date' in ohlcv_idx.columns:
                ohlcv_idx = ohlcv_idx.set_index('Date')
            else:
                return out  # can't resample without dates

        for tf_label, tf_rule in [('W', 'W-FRI'), ('M', 'ME')]:
            try:
                resampled = ohlcv_idx.resample(tf_rule).agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min',
                    'Close': 'last', 'Volume': 'sum'
                }).dropna(subset=['Close'])
            except Exception:
                continue

            if len(resampled) < 5:
                continue

            # Run pattern detection on resampled data
            patt_funcs = [
                detect_head_shoulder(resampled, prefix=f"hs_{tf_label}_"),
                detect_triangle_pattern(resampled, prefix=f"tri_{tf_label}_"),
                detect_wedge(resampled, prefix=f"wed_{tf_label}_"),
                detect_channel(resampled, prefix=f"chan_{tf_label}_"),
                detect_double_top_bottom(resampled, prefix=f"dbl_{tf_label}_"),
            ]
            tf_patt = pd.concat(patt_funcs, axis=1)

            # Keep only pattern columns (not roll_max/min intermediates)
            keep_cols = [c for c in tf_patt.columns if c.endswith('pattern')]
            tf_patt = tf_patt[keep_cols].fillna(0).astype('int8')

            # Shift by 1 to avoid look-ahead
            tf_patt = tf_patt.shift(1)

            # Reindex to daily (forward-fill: weekly/monthly signal persists until next candle)
            tf_daily = tf_patt.reindex(ohlcv_idx.index, method='ffill').fillna(0).astype('int8')

            # Reset index to match original
            tf_daily.index = ohlcv.index
            out = pd.concat([out, tf_daily], axis=1)

        return out


    # ============================
    # TARGETS
    # ============================
    def make_target_path_aware(df, horizon=30, up=0.15, dd=-0.05, name='target_path'):
        close = df['Close'].values
        high  = df['High'].values
        low   = df['Low'].values
        n = len(df)
        tgt = np.full(n, np.nan, dtype=np.float32)
        for t in range(n):
            if (t + horizon) >= n:
                continue
            end = t + horizon + 1
            entry = close[t]
            up_th = entry * (1.0 + up)
            dd_th = entry * (1.0 + dd)
            hseg = high[t+1:end]
            lseg = low[t+1:end]
            hit_up_idx = np.where(hseg >= up_th)[0]
            hit_dd_idx = np.where(lseg <= dd_th)[0]
            hit_up = hit_up_idx[0] if len(hit_up_idx) else None
            hit_dd = hit_dd_idx[0] if len(hit_dd_idx) else None
            tgt[t] = 1.0 if (hit_up is not None and (hit_dd is None or hit_up < hit_dd)) else 0.0
        return pd.Series(tgt, index=df.index, name=name)

    def make_best_entry_sale(df, horizon=30):
        low = df['Low'].values
        high = df['High'].values
        n = len(df)
        # Targets are defined for the path after the signal date (D+1 onward).
        best_entry = np.full(n, np.nan, dtype='float32')
        best_sale  = np.full(n, np.nan, dtype='float32')
        for t in range(n):
            if (t + horizon) >= n:
                continue
            end = t + horizon + 1
            window_low = low[t+1:end]
            window_high = high[t+1:end]
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

        # compact dtypes (cross_ now continuous, don't cast to int8)
        other0 = [c for c in X_red.columns.get_level_values(0)
                  if not str(c).endswith("pattern")]
        if other0:
            X_red.loc[:, (other0, slice(None))] = to_float32_safe(X_red.loc[:, (other0, slice(None))])

        if verbose:
            after_cnt = X_red.shape[1]
            print(f"[Feature reduction] columns: {before_cnt} -> {after_cnt} ({100.0*after_cnt/before_cnt:.1f}% kept)")
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


    MIN_TARGET_RATE = 0.005  # pelo menos 0.5% de positivos (relaxed for FIIs/crypto)
    MAX_TARGET_RATE = 0.995  # no máximo 99.5% de positivos (relaxed for FIIs/crypto)

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

            y_up20 = pd.to_numeric(y_up20, errors='coerce')
            y_dd5  = pd.to_numeric(y_dd5, errors='coerce')

            rate_up = float(y_up20.dropna().mean()) if y_up20.notna().any() else 0.0
            rate_dd = float(y_dd5.dropna().mean()) if y_dd5.notna().any() else 0.0

            # FIIs (ending in 11.SA) are low-volatility: skip degeneracy for them
            is_fii = tk.endswith("11.SA")
            if not is_fii:
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
    # Pre-compute market reference data for cross-ticker features
    # ============================
    _ibov_sym = None
    for _try_sym in ['^BVSP', '^BVSP.SA']:
        try:
            _ibov_ohlcv = data.xs(_try_sym, level=1, axis=1)
            if 'Close' in _ibov_ohlcv.columns and _ibov_ohlcv['Close'].notna().sum() > 100:
                _ibov_sym = _try_sym
                break
        except (KeyError, ValueError):
            continue
    if _ibov_sym:
        _ibov_ret = data.xs(_ibov_sym, level=1, axis=1)['Close'].pct_change()
        print(f"[cross-features] IBOV reference: {_ibov_sym} ({_ibov_ret.notna().sum()} returns)")
    else:
        _ibov_ret = None
        print("[cross-features] WARNING: IBOV not found, skipping beta/relative-strength")

    _vix_data = None
    for _try_vix in ['^VIX', '^VIX.SA']:
        try:
            _vix_ohlcv = data.xs(_try_vix, level=1, axis=1)
            if 'Close' in _vix_ohlcv.columns and _vix_ohlcv['Close'].notna().sum() > 100:
                _vix_data = _vix_ohlcv['Close']
                break
        except (KeyError, ValueError):
            continue
    if _vix_data is not None:
        _vix_pctl_252 = _vix_data.rolling(252, min_periods=60).rank(pct=True)
        print(f"[cross-features] VIX reference: {_vix_data.notna().sum()} values")
    else:
        _vix_pctl_252 = None
        print("[cross-features] WARNING: VIX not found, skipping VIX percentile")

    # ============================================================
    # v8 FEATURE ENGINEERING — Tier 1A: Foreign flow + USDBRL
    # ============================================================
    def _safe_extract_close(sym):
        """Extract Close series from `data` MultiIndex, returning None if missing."""
        try:
            _oh = data.xs(sym, level=1, axis=1)
            if "Close" in _oh.columns and _oh["Close"].notna().sum() > 100:
                return _oh["Close"]
        except (KeyError, ValueError):
            pass
        return None

    _ewz_close = _safe_extract_close("EWZ")
    _ewz_ret = _ewz_close.pct_change() if _ewz_close is not None else None
    if _ewz_ret is not None:
        print(f"[v8] EWZ foreign flow proxy: {_ewz_ret.notna().sum()} returns")
    else:
        print("[v8] WARNING: EWZ not found")

    _usdbrl_close = _safe_extract_close("BRL=X")
    if _usdbrl_close is not None:
        _usdbrl_ret = _usdbrl_close.pct_change()
        _usdbrl_pctl_252 = _usdbrl_close.rolling(252, min_periods=60).rank(pct=True)
        print(f"[v8] USDBRL reference: {_usdbrl_close.notna().sum()} values")
    else:
        _usdbrl_ret = None
        _usdbrl_pctl_252 = None
        print("[v8] WARNING: USDBRL not found")

    _eem_close = _safe_extract_close("EEM")
    _eem_ret = _eem_close.pct_change() if _eem_close is not None else None

    # Sector ETF returns (Tier 2E sector rotation)
    _sector_rets = {}
    for _sec in ("XLE", "XLF", "XLI", "XLB", "XLY", "XLP", "XLU", "XLK", "XLV", "XLRE"):
        _sc = _safe_extract_close(_sec)
        if _sc is not None:
            _sector_rets[_sec] = _sc.pct_change()
    print(f"[v8] Sector ETFs loaded: {len(_sector_rets)}")

    # Ticker -> sector ETF heuristic mapping (B3 tickers to global sector ETFs).
    # This is approximate -- B3 names map to global sectors they most resemble.
    _SECTOR_MAP = {
        # Energia (oil & gas)
        "PETR3.SA": "XLE", "PETR4.SA": "XLE", "PRIO3.SA": "XLE",
        "RECV3.SA": "XLE", "RRRP3.SA": "XLE",
        # Materiais (mineracao + siderurgia)
        "VALE3.SA": "XLB", "CSNA3.SA": "XLB", "GGBR4.SA": "XLB",
        "GOAU4.SA": "XLB", "USIM5.SA": "XLB", "BRAP4.SA": "XLB",
        # Financeiro
        "ITUB4.SA": "XLF", "BBDC4.SA": "XLF", "BBAS3.SA": "XLF",
        "SANB11.SA": "XLF", "BPAC11.SA": "XLF", "ITSA4.SA": "XLF",
        "B3SA3.SA": "XLF",
        # Industrial
        "WEGE3.SA": "XLI", "EMBR3.SA": "XLI", "RAIL3.SA": "XLI",
        "CCRO3.SA": "XLI", "MYPK3.SA": "XLI",
        # Utilities
        "ELET3.SA": "XLU", "ELET6.SA": "XLU", "CMIG4.SA": "XLU",
        "EGIE3.SA": "XLU", "ENGI11.SA": "XLU", "TAEE11.SA": "XLU",
        "CPFE3.SA": "XLU", "SBSP3.SA": "XLU",
        # Consumer Discretionary
        "LREN3.SA": "XLY", "MGLU3.SA": "XLY", "AMER3.SA": "XLY",
        "VIIA3.SA": "XLY", "LOCA3.SA": "XLY", "PCAR3.SA": "XLY",
        # Consumer Staples
        "ABEV3.SA": "XLP", "JBSS3.SA": "XLP", "BRFS3.SA": "XLP",
        "MRFG3.SA": "XLP", "NTCO3.SA": "XLP",
        # Healthcare
        "HAPV3.SA": "XLV", "FLRY3.SA": "XLV", "RDOR3.SA": "XLV",
        "QUAL3.SA": "XLV",
        # Tech
        "TOTS3.SA": "XLK", "LWSA3.SA": "XLK",
        # Real Estate
        "CYRE3.SA": "XLRE", "MRVE3.SA": "XLRE", "EZTC3.SA": "XLRE",
    }

    # ============================================================
    # v8 Tier 1B — Enhanced Market Breadth (new highs/lows, AD line)
    # ============================================================
    _market_breadth_v8 = None  # new_highs_new_lows_ratio + AD line momentum
    if len(_sa_tickers) >= 10:
        try:
            _closes_list = []
            for _t in _sa_tickers[:80]:
                _tc = _safe_extract_close(_t)
                if _tc is not None and _tc.notna().sum() > 300:
                    _closes_list.append(_tc)
            if len(_closes_list) >= 10:
                _breadth_df = pd.concat(_closes_list, axis=1)
                # New highs (60d) - new lows (60d) ratio
                _rolling_max_60 = _breadth_df.rolling(60, min_periods=20).max()
                _rolling_min_60 = _breadth_df.rolling(60, min_periods=20).min()
                _is_new_high = (_breadth_df >= _rolling_max_60 * 0.999).sum(axis=1)
                _is_new_low = (_breadth_df <= _rolling_min_60 * 1.001).sum(axis=1)
                _nh_nl_ratio = (_is_new_high - _is_new_low) / _breadth_df.notna().sum(axis=1)
                # Advance-decline line (cumsum of ups - downs, then momentum)
                _daily_rets = _breadth_df.pct_change()
                _advances = (_daily_rets > 0).sum(axis=1)
                _declines = (_daily_rets < 0).sum(axis=1)
                _ad_line = (_advances - _declines).cumsum()
                # AD line momentum: slope over 20d normalized
                _ad_mom = _ad_line.diff(20) / _breadth_df.notna().sum(axis=1).rolling(20).mean()
                _market_breadth_v8 = pd.DataFrame({
                    "new_highs_new_lows_ratio": _nh_nl_ratio,
                    "ad_line_mom_20d": _ad_mom,
                })
                print(f"[v8] Enhanced breadth v8 computed: new_highs_new_lows + AD line mom")
                del _breadth_df, _rolling_max_60, _rolling_min_60, _daily_rets, _ad_line
        except Exception as _e:
            print(f"[v8] Enhanced breadth failed: {_e}")

    # ── STRATEGY 3: Market regime features (cross-sectional) ──
    # These measure overall market conditions, shared across all tickers
    _sa_tickers = [t for t in valid_tickers if t.endswith('.SA') and not any(t.startswith(x) for x in ['^'])]
    _market_breadth = None
    _market_dispersion = None
    if len(_sa_tickers) >= 10:
        try:
            # Collect closes for SA tickers
            _closes_list = []
            for _t in _sa_tickers[:60]:  # cap at 60 to save memory
                try:
                    _tc = data.xs(_t, level=1, axis=1)['Close']
                    if _tc.notna().sum() > 200:
                        _closes_list.append(_tc)
                except (KeyError, ValueError):
                    pass
            if len(_closes_list) >= 10:
                _closes_df = pd.concat(_closes_list, axis=1)
                # Market breadth: % of stocks above their 200d SMA
                _sma200s = _closes_df.rolling(200, min_periods=50).mean()
                _above_200 = (_closes_df > _sma200s).sum(axis=1) / _closes_df.notna().sum(axis=1)
                _market_breadth = _above_200.rolling(5, min_periods=1).mean()  # smooth
                # Market dispersion: std of 20d returns across stocks (high = risk-off)
                _rets_20d = _closes_df.pct_change(20)
                _market_dispersion = _rets_20d.std(axis=1).rolling(10, min_periods=3).mean()
                del _closes_df, _sma200s, _above_200, _rets_20d  # free memory
                print(f"[cross-features] Market regime: breadth + dispersion ({len(_closes_list)} tickers)")
        except Exception as _e:
            print(f"[cross-features] Market regime failed: {_e}")

    def compute_cross_ticker_features(tk, ohlcv, ibov_ret, vix_pctl, shift_features=1):
        """Compute cross-ticker features: rolling beta, relative strength, VIX percentile.

        v8 extension: foreign flow (EWZ/USDBRL), market breadth enhanced, sector rotation.
        """
        close_col = 'Adj Close' if 'Adj Close' in ohlcv.columns else 'Close'
        tk_ret = ohlcv[close_col].pct_change()
        cross_feats = pd.DataFrame(index=ohlcv.index)

        # Rolling beta with IBOV (60-day)
        if ibov_ret is not None:
            aligned = pd.DataFrame({'tk': tk_ret, 'ibov': ibov_ret}).dropna()
            if len(aligned) > 60:
                cov_roll = aligned['tk'].rolling(60, min_periods=30).cov(aligned['ibov'])
                var_roll = aligned['ibov'].rolling(60, min_periods=30).var()
                beta_raw = cov_roll / var_roll.replace(0, np.nan)
                cross_feats['beta_ibov_60d'] = beta_raw.reindex(ohlcv.index)

            # Relative strength vs IBOV: ticker return 20d / ibov return 20d
            tk_ret20 = ohlcv[close_col].pct_change(20)
            ibov_ret20 = ibov_ret.rolling(20).sum()  # approx 20d return
            ibov_ret20_aligned = ibov_ret20.reindex(ohlcv.index)
            cross_feats['rel_strength_ibov_20d'] = tk_ret20 - ibov_ret20_aligned

            # Correlation rolling 60d with IBOV
            if len(aligned) > 60:
                corr_roll = aligned['tk'].rolling(60, min_periods=30).corr(aligned['ibov'])
                cross_feats['corr_ibov_60d'] = corr_roll.reindex(ohlcv.index)

        # VIX percentile (252d)
        if vix_pctl is not None:
            cross_feats['vix_percentile_252d'] = vix_pctl.reindex(ohlcv.index)

        # Market regime features (cross-sectional, same for all tickers)
        if _market_breadth is not None:
            cross_feats['market_breadth'] = _market_breadth.reindex(ohlcv.index)
        if _market_dispersion is not None:
            cross_feats['market_dispersion'] = _market_dispersion.reindex(ohlcv.index)

        # ==================================================================
        # v8 TIER 1A — Foreign flow proxies (USDBRL + EWZ)
        # ==================================================================
        if _usdbrl_ret is not None:
            # USDBRL momentum: alta do dolar normalmente = saida de foreign capital = ruim p/ B3
            _usdbrl_aligned = _usdbrl_ret.reindex(ohlcv.index)
            cross_feats['usdbrl_ret_5d'] = _usdbrl_aligned.rolling(5).sum()
            cross_feats['usdbrl_ret_20d'] = _usdbrl_aligned.rolling(20).sum()
            if _usdbrl_pctl_252 is not None:
                cross_feats['usdbrl_pctl_252'] = _usdbrl_pctl_252.reindex(ohlcv.index)
            # Ticker vs USDBRL correlation (inverted for intuitiveness: negative = cambio alto = ticker cai)
            _alig = pd.DataFrame({'tk': tk_ret, 'fx': _usdbrl_aligned}).dropna()
            if len(_alig) > 60:
                cross_feats['corr_usdbrl_60d'] = _alig['tk'].rolling(60, min_periods=30).corr(_alig['fx']).reindex(ohlcv.index)

        if _ewz_ret is not None:
            # EWZ como proxy foreign investor view on Brazil
            _ewz_aligned = _ewz_ret.reindex(ohlcv.index)
            cross_feats['ewz_ret_5d'] = _ewz_aligned.rolling(5).sum()
            cross_feats['ewz_ret_20d'] = _ewz_aligned.rolling(20).sum()
            # Ticker correlation with EWZ (high corr = moves with foreign flow)
            _alig = pd.DataFrame({'tk': tk_ret, 'ewz': _ewz_aligned}).dropna()
            if len(_alig) > 60:
                cross_feats['corr_ewz_60d'] = _alig['tk'].rolling(60, min_periods=30).corr(_alig['ewz']).reindex(ohlcv.index)
                # Alpha vs EWZ (idiosyncratic performance after adjusting for EWZ)
                _ewz_ret20 = _ewz_aligned.rolling(20).sum().reindex(ohlcv.index)
                cross_feats['alpha_ewz_20d'] = ohlcv[close_col].pct_change(20) - _ewz_ret20

        if _eem_ret is not None:
            _eem_aligned = _eem_ret.reindex(ohlcv.index)
            cross_feats['eem_ret_20d'] = _eem_aligned.rolling(20).sum()

        # ==================================================================
        # v8 TIER 1B — Enhanced market breadth
        # ==================================================================
        if _market_breadth_v8 is not None:
            for _mb_col in _market_breadth_v8.columns:
                cross_feats[f'v8_{_mb_col}'] = _market_breadth_v8[_mb_col].reindex(ohlcv.index)

        # ==================================================================
        # v8 TIER 2E — Sector rotation features
        # ==================================================================
        _ticker_sector = _SECTOR_MAP.get(tk)
        if _ticker_sector and _ticker_sector in _sector_rets:
            _sec_ret = _sector_rets[_ticker_sector].reindex(ohlcv.index)
            # Sector momentum (where IS this ticker's sector going?)
            cross_feats['sector_ret_5d'] = _sec_ret.rolling(5).sum()
            cross_feats['sector_ret_20d'] = _sec_ret.rolling(20).sum()
            # Relative strength vs own sector (alpha idiossincratico)
            cross_feats['rel_strength_sector_20d'] = (
                ohlcv[close_col].pct_change(20) - _sec_ret.rolling(20).sum()
            )
            # Correlation com setor (se descolou = oportunidade ou risco)
            _alig = pd.DataFrame({'tk': tk_ret, 'sec': _sec_ret}).dropna()
            if len(_alig) > 60:
                cross_feats['corr_sector_60d'] = _alig['tk'].rolling(60, min_periods=30).corr(_alig['sec']).reindex(ohlcv.index)

        # Sector rotation leader: qual setor esta mais forte em 20d?
        # Uma feature unica: rank do proprio setor entre 10 setores (0..1)
        if _sector_rets and _ticker_sector in _sector_rets:
            _sector_20d_rets = {}
            for _sn, _sr in _sector_rets.items():
                _sector_20d_rets[_sn] = _sr.reindex(ohlcv.index).rolling(20).sum()
            _sec_rank_df = pd.DataFrame(_sector_20d_rets)
            _own_rank = _sec_rank_df.rank(axis=1, pct=True)[_ticker_sector]
            cross_feats['own_sector_rank_20d'] = _own_rank

        if shift_features > 0:
            # v8 ROOT FIX: shift so nao-OHLCV (que aqui ja nao inclui OHLCV)
            # Todas as colunas aqui sao derivadas, entao shift integral e OK.
            cross_feats = cross_feats.shift(shift_features)

        cross_feats = to_float32_safe(cross_feats)
        return cross_feats

    # ============================================================
    # v8 FEATURE ENGINEERING — Advanced per-ticker features
    # Tier 1C (mean reversion) + Tier 2D (vol regime) +
    # Tier 2F (candlestick continuous) + Tier 3H (microstructure)
    # ============================================================
    def compute_advanced_features_v8(ohlcv, shift_features=1):
        """Per-ticker advanced features. Tier 1C + 2D + 2F + 3H."""
        close_col = 'Adj Close' if 'Adj Close' in ohlcv.columns else 'Close'
        c = pd.to_numeric(ohlcv[close_col], errors='coerce').astype(np.float64)
        h = pd.to_numeric(ohlcv['High'], errors='coerce').astype(np.float64)
        l = pd.to_numeric(ohlcv['Low'], errors='coerce').astype(np.float64)
        o = pd.to_numeric(ohlcv['Open'], errors='coerce').astype(np.float64)
        v = pd.to_numeric(ohlcv.get('Volume', pd.Series(0, index=ohlcv.index)),
                          errors='coerce').astype(np.float64)
        idx = ohlcv.index
        af = pd.DataFrame(index=idx)

        # ==================================================================
        # TIER 1C — Mean reversion setups
        # ==================================================================
        # Distance to 52w high (negative = below high, 0 = at high)
        _roll_52w_high = c.rolling(252, min_periods=60).max()
        _roll_52w_low = c.rolling(252, min_periods=60).min()
        af['v8_dist_52w_high_pct'] = ((c / _roll_52w_high.replace(0, np.nan)) - 1.0).clip(-0.80, 0.0)
        af['v8_dist_52w_low_pct'] = ((c / _roll_52w_low.replace(0, np.nan)) - 1.0).clip(0.0, 3.0)

        # Bollinger squeeze index: BB_width vs historical min (squeeze = breakout prox)
        _ma20 = c.rolling(20, min_periods=10).mean()
        _sd20 = c.rolling(20, min_periods=10).std()
        _bb_upper = _ma20 + 2 * _sd20
        _bb_lower = _ma20 - 2 * _sd20
        _bb_width = (_bb_upper - _bb_lower) / _ma20.replace(0, np.nan)
        _bb_width_min_100 = _bb_width.rolling(100, min_periods=30).min()
        af['v8_bb_squeeze_idx'] = (_bb_width - _bb_width_min_100).clip(0.0, 0.30)
        af['v8_bb_width'] = _bb_width.clip(0.0, 0.30)
        # Position within bollinger band (0=at lower, 1=at upper)
        _bb_range = (_bb_upper - _bb_lower).replace(0, np.nan)
        af['v8_bb_position'] = ((c - _bb_lower) / _bb_range).clip(-0.20, 1.20)

        # RSI divergence (preco new high mas RSI mais baixo -> bearish div; oposto bullish)
        # Compute RSI(14) from scratch
        _delta = c.diff()
        _gain = _delta.clip(lower=0).rolling(14, min_periods=7).mean()
        _loss = (-_delta.clip(upper=0)).rolling(14, min_periods=7).mean()
        _rs = _gain / _loss.replace(0, np.nan)
        _rsi = 100 - (100 / (1 + _rs))
        # Bullish divergence: preco em new 20d low mas RSI nao em new 20d low
        _price_min_20 = c.rolling(20, min_periods=10).min()
        _rsi_min_20 = _rsi.rolling(20, min_periods=10).min()
        _price_at_low = (c <= _price_min_20 * 1.005).astype(float)
        _rsi_at_low = (_rsi <= _rsi_min_20 * 1.05).astype(float)
        af['v8_rsi_bull_divergence'] = (_price_at_low * (1 - _rsi_at_low)).clip(0.0, 1.0)
        # Bearish divergence: preco new high mas RSI nao
        _price_max_20 = c.rolling(20, min_periods=10).max()
        _rsi_max_20 = _rsi.rolling(20, min_periods=10).max()
        _price_at_high = (c >= _price_max_20 * 0.995).astype(float)
        _rsi_at_high = (_rsi >= _rsi_max_20 * 0.95).astype(float)
        af['v8_rsi_bear_divergence'] = (_price_at_high * (1 - _rsi_at_high)).clip(0.0, 1.0)

        # ==================================================================
        # TIER 2D — Volatility regime transitions (GARCH-like)
        # ==================================================================
        _ret = c.pct_change()
        _vol_20 = _ret.rolling(20, min_periods=10).std()
        _vol_60 = _ret.rolling(60, min_periods=20).std()
        _vol_ratio = (_vol_20 / _vol_60.replace(0, np.nan)).clip(0.3, 3.0)
        # Vol expandindo: ratio > 1.2
        af['v8_vol_expansion_ratio'] = _vol_ratio
        af['v8_vol_expanding'] = (_vol_ratio > 1.2).astype(float)
        af['v8_vol_contracting'] = (_vol_ratio < 0.8).astype(float)
        # Vol regime: low / normal / high baseado em percentile 252d
        _vol_pctl = _vol_20.rolling(252, min_periods=60).rank(pct=True)
        af['v8_vol_regime_pctl'] = _vol_pctl
        # Parkinson volatility (high-low range-based, low noise)
        _parkinson = ((np.log(h / l.replace(0, np.nan))) ** 2).rolling(20, min_periods=10).mean()
        _parkinson_pctl = _parkinson.rolling(252, min_periods=60).rank(pct=True)
        af['v8_parkinson_vol_pctl'] = _parkinson_pctl

        # ==================================================================
        # TIER 2F — Candlestick patterns continuous
        # ==================================================================
        _body = (c - o).abs()
        _range = (h - l).replace(0, np.nan)
        _upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
        _lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
        _body_ratio = (_body / _range).clip(0.0, 1.0)
        _upper_shadow_ratio = (_upper_shadow / _range).clip(0.0, 1.0)
        _lower_shadow_ratio = (_lower_shadow / _range).clip(0.0, 1.0)

        # Hammer strength: long lower shadow + small body + close>open (bullish)
        # Conditions: lower_shadow >= 2*body AND body < 0.3*range
        _is_hammer = ((_lower_shadow_ratio > 2 * _body_ratio) &
                      (_body_ratio < 0.3) &
                      (c > o)).astype(float)
        af['v8_hammer_continuous'] = _is_hammer * (_lower_shadow_ratio - _body_ratio).clip(0.0, 1.0)

        # Inverted hammer / shooting star
        _is_shooting = ((_upper_shadow_ratio > 2 * _body_ratio) &
                        (_body_ratio < 0.3) &
                        (c < o)).astype(float)
        af['v8_shooting_star_continuous'] = _is_shooting * (_upper_shadow_ratio - _body_ratio).clip(0.0, 1.0)

        # Doji (very small body) — consolidation
        _is_doji = (_body_ratio < 0.1).astype(float)
        af['v8_doji_cluster_5d'] = _is_doji.rolling(5, min_periods=3).mean()

        # Engulfing magnitude (body today > body yesterday, opposite direction)
        _prev_body = _body.shift(1)
        _prev_sign = np.sign(c.shift(1) - o.shift(1))
        _curr_sign = np.sign(c - o)
        _engulf_mag = ((_body / _prev_body.replace(0, np.nan)) - 1).clip(0.0, 5.0)
        _is_bullish_engulf = ((_curr_sign > 0) & (_prev_sign < 0) & (_body > _prev_body)).astype(float)
        _is_bearish_engulf = ((_curr_sign < 0) & (_prev_sign > 0) & (_body > _prev_body)).astype(float)
        af['v8_bullish_engulfing'] = _is_bullish_engulf * _engulf_mag
        af['v8_bearish_engulfing'] = _is_bearish_engulf * _engulf_mag

        # ==================================================================
        # TIER 3H — Microstructure liquidity
        # ==================================================================
        _dollar_vol = c * v
        _dv_pctl = _dollar_vol.rolling(252, min_periods=60).rank(pct=True)
        af['v8_dollar_volume_pctl_252'] = _dv_pctl
        # Volume consistency: inverse coefficient of variation
        _vol_mean_20 = v.rolling(20, min_periods=5).mean()
        _vol_std_20 = v.rolling(20, min_periods=5).std()
        _vol_cv_20 = _vol_std_20 / _vol_mean_20.replace(0, np.nan)
        af['v8_volume_consistency_20d'] = (1.0 / (1.0 + _vol_cv_20)).clip(0.0, 1.0)
        # Spread proxy (high-low range / close) — crude proxy para bid-ask spread
        _spread_proxy = ((h - l) / c.replace(0, np.nan)).rolling(20, min_periods=5).mean()
        af['v8_spread_proxy_20d'] = _spread_proxy.clip(0.0, 0.20)
        # Volume surge frequency (>2sigma days in last 20d)
        _z_vol = (v - _vol_mean_20) / _vol_std_20.replace(0, np.nan)
        _surge_days = (_z_vol > 2.0).astype(float)
        af['v8_vol_surge_freq_20d'] = _surge_days.rolling(20, min_periods=5).mean()

        # ==================================================================
        # TIER 3G — Earnings calendar proxies (best-effort, graceful fallback)
        # ==================================================================
        # yfinance calendar access is often flaky; put behind try/except, default 0.
        # Implementacao minima: nao fazemos calendar scraping per-ticker (custo alto em ETL).
        # Em vez disso, usamos proxy indireto: volatility spike frequency.
        # Esses dias "anormais" sao geralmente proximos a earnings announcements.
        _ret_vol_norm = _ret.abs() / _vol_20.replace(0, np.nan)
        af['v8_abnormal_return_freq_20d'] = (_ret_vol_norm > 2.0).rolling(20, min_periods=5).mean()

        # ==================================================================
        # Shift (apenas indicadores, NAO OHLCV — que nao esta em af)
        # ==================================================================
        if shift_features > 0:
            af = af.shift(shift_features)

        af = to_float32_safe(af)
        return af


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
            pats_wm = patterns_weekly_monthly(ohlcv)  # weekly + monthly pattern detection
            if feats is None or feats.shape[1] == 0:
                skip_reasons['empty_feats'] += 1
                continue

            fund = fundamentals_daily_for_ticker(tk, ohlcv.index, shift_features=SHIFT_FEATURES)
            cross = compute_cross_ticker_features(tk, ohlcv, _ibov_ret, _vix_pctl_252, shift_features=SHIFT_FEATURES)
            # v8 advanced features: Tier 1C + 2D + 2F + 3H (per-ticker)
            try:
                adv_v8 = compute_advanced_features_v8(ohlcv, shift_features=SHIFT_FEATURES)
            except Exception as _e:
                print(f"[v8] advanced features falhou para {tk}: {_e}")
                adv_v8 = pd.DataFrame(index=ohlcv.index)

            X_tk = pd.concat([feats, pats, pats_wm, fund, cross, adv_v8], axis=1)
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
            y_up20 = to_float32_safe(y_up20).rename(('target_up20', tk))
            y_dd5  = to_float32_safe(y_dd5).rename(('target_dd5',  tk))

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

    # Cast consistente (cross_ now continuous float, don't cast to int8)
    other0 = [c for c in X.columns.get_level_values(0)
              if not str(c).endswith("pattern")]
    if other0:
        X.loc[:, (other0, slice(None))] = to_float32_safe(X.loc[:, (other0, slice(None))])

    for tname in ('target_up20','target_dd5'):
        if tname in y.columns.get_level_values(0):
            y.loc[:, (tname, slice(None))] = to_float32_safe(y.loc[:, (tname, slice(None))])
    for tname in ('target_best_entry','target_best_sale'):
        if tname in y.columns.get_level_values(0):
            y.loc[:, (tname, slice(None))] = to_float32_safe(y.loc[:, (tname, slice(None))])


    # Garantir 100% preenchido (inf/NaN) globalmente (features já estão ok; segurança adicional)
    X = fill_100pct(X, allow_bfill=ALLOW_BFILL_EXOGENOUS)
    y = y.replace([np.inf, -np.inf], np.nan)

    # Verificações finais
    assert not X.isna().any().any(), "Ainda há NaN em X após preenchimento!"

    # ============================
    # Saída única combinada
    # ============================
    DATASET = pd.concat([X, y], axis=1).sort_index()
    # Segurança final (caso algum merge crie lacunas):
    DATASET.loc[:, X.columns] = fill_100pct(DATASET.loc[:, X.columns], allow_bfill=ALLOW_BFILL_EXOGENOUS)

    # Sanidade: sem NaN
    assert not DATASET.loc[:, X.columns].isna().any().any(), "Ainda ha NaN nas features do dataset final!"

    # ============================
    # Feature reduction (per ticker, MI vs targets)
    # ============================
    if SKIP_MI_REDUCTION:
        print("[ETL] SKIP_MI_REDUCTION=True -> using full feature set as 'reduced' (GA does Spearman selection)")
        X_reduced = X.copy()
    else:
        # y_up and y_dd views (ensure they exist)
        y_up = y.loc[:, y.columns.get_level_values(0) == 'target_up20']
        y_dd = y.loc[:, y.columns.get_level_values(0) == 'target_dd5']

        X_reduced = reduce_features_automatic(
            X, y_up, y_dd,
            top_fraction=0.85,   # pega ~85% por MI antes de deduplicar
            min_keep=96,         # garanta pelo menos ~100 por ticker (se existirem)
            var_thr=None,        # nao remova por variancia agora
            corr_thr=0.9995,     # so remove quase identicos
            always_keep_prefixes=("pct_change","SMA_","tri_","sr_","Close","Adj Close","close"),  # Close obrigatorio para reg
            verbose=True
        )



    # ============================
    # Output (full and reduced)
    # ============================
    DATASET_FULL = pd.concat([X, y], axis=1).sort_index()
    DATASET_FULL.loc[:, X.columns] = fill_100pct(DATASET_FULL.loc[:, X.columns], allow_bfill=ALLOW_BFILL_EXOGENOUS)
    assert not DATASET_FULL.loc[:, X.columns].isna().any().any(), "NaN in feature block of full dataset!"

    DATASET_REDUCED = pd.concat([X_reduced, y], axis=1).sort_index()
    DATASET_REDUCED.loc[:, X_reduced.columns] = fill_100pct(DATASET_REDUCED.loc[:, X_reduced.columns], allow_bfill=ALLOW_BFILL_EXOGENOUS)
    assert not DATASET_REDUCED.loc[:, X_reduced.columns].isna().any().any(), "NaN in feature block of reduced dataset!"

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
            yt = yt.apply(pd.to_numeric, errors="coerce")

            # --- geral (todas as datas e tickers empilhados) ---
            arr = yt.to_numpy().ravel()
            arr = arr[np.isfinite(arr)]
            rate1 = float(arr.mean()) if arr.size else np.nan
            n1 = int(arr.sum()) if arr.size else 0
            n = int(arr.size)

            print("\n" + "="*60)
            print(f"[{t}] Geral")
            print(f"N total: {n:,} | N(1): {n1:,} | N(0): {n-n1:,}")
            print(f"Taxa(1): {100*rate1:.2f}% | Taxa(0): {100*(1-rate1):.2f}%")

            # --- por ticker ---
            # mean por coluna => taxa de 1 por ticker
            per_ticker_rate1 = yt.mean(axis=0, skipna=True)
            per_ticker_rate1.index = per_ticker_rate1.index.get_level_values(1).astype(str)

            print(f"\n[{t}] Taxa(1) por ticker (top {topk} MAIORES):")
            print((per_ticker_rate1.sort_values(ascending=False).head(topk) * 100).round(2).astype(str) + "%")

            print(f"\n[{t}] Taxa(1) por ticker (top {topk} MENORES):")
            print((per_ticker_rate1.sort_values(ascending=True).head(topk) * 100).round(2).astype(str) + "%")

    print_target_rates(y, targets=("target_up20","target_dd5"), topk=12)




if __name__ == '__main__':
    run_etl()

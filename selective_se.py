#!/usr/bin/env python3
"""
Estrategia v3 do mercado sueco: selecao ativa por previsao de retorno.

Um modelo preve o retorno de cada acao nas proximas H pregoes contra o resto
do universo; outro preve o retorno do mercado. Toda sexta (ultimo pregao da
semana), depois do fechamento:

- sai de uma posicao que ja cumpriu H pregoes se a previsao dela caiu abaixo
  da mediana, ou se ja cumpriu 2H; com o filtro de mercado ligado e previsao de
  queda do mercado, sai de tudo (antecipar quedas, long-only);
- entra nas acoes do topo das previsoes (`entry_q`) ate `k` posicoes.

Ordens a mercado na abertura do pregao seguinte; custos por lado de
market_se.py; precos ajustados por dividendos (prices_se.py).

Os modelos sao treinados walk-forward: a cada 6 meses, so com amostras cujo
rotulo ja tinha se realizado (purga de H pregoes).

    python selective_se.py predict --horizon 20 --model lgbm   # previsoes walk-forward (cache)
    python selective_se.py dev                                  # grade de desenvolvimento (2010-2019)
    python selective_se.py holdout                              # avaliacao unica (ADR 0007)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import market_se  # noqa: E402

TRAIN_START = pd.Timestamp("2006-06-01")
DEV = ("2010-01-01", "2019-12-31")
HOLDOUT = ("2020-01-01", "2026-10-30")
RETRAIN_MONTHS = 6
MIN_TURNOVER = market_se.MIN_MEDIAN_TURNOVER_SEK
CACHE_DIR = REPO_ROOT / "output" / "v3"
REPORT_DIR = REPO_ROOT / "reports" / "se_v3"
SPEC_PATH = REPO_ROOT / "data" / "strategy_se_v3.json"

STOCK_FEATURES = ["r5", "r21", "r63", "r126", "mom12_1", "hi52", "lo52", "vol21", "vol63", "volratio",
                  "gap50", "gap200", "tsurge", "beta63", "resid21", "resid5"]
MARKET_FEATURES = ["m_r5", "m_r21", "m_r63", "m_vol21", "m_gap200", "m_breadth50", "m_disp21"]


# --------------------------------------------------------------------------
# Features and labels (everything at date t uses data up to the close of t)
# --------------------------------------------------------------------------
@dataclass
class Data:
    days: pd.DatetimeIndex
    tickers: List[str]
    adj: pd.DataFrame
    adj_open: pd.DataFrame
    eligible: pd.DataFrame
    stock: Dict[str, pd.DataFrame]       # ranked cross-sectionally, in [-0.5, 0.5]
    market: pd.DataFrame                 # rolling z-scores, one row per day
    mkt_ret: pd.Series                   # equal-weight daily return of the eligible universe
    bench: pd.Series


def build(px: Dict[str, pd.DataFrame]) -> Data:
    eq = [t for t in px["adj"].columns if market_se.is_equity(t)]
    raw = px["adj"][eq]
    adj = raw.ffill(limit=5)
    adj_open = px["adj_open"][eq]
    turn = px["turnover"][eq]
    ret = adj.pct_change(fill_method=None)
    logp = np.log(adj)

    liquid = turn.rolling(60, min_periods=40).median() >= MIN_TURNOVER
    history = raw.notna().rolling(260, min_periods=1).sum() >= 240
    eligible = liquid & history & raw.notna()

    ew = ret.where(eligible.shift(1, fill_value=False)).mean(axis=1).fillna(0.0)
    mlog = np.log((1 + ew).cumprod())

    f: Dict[str, pd.DataFrame] = {}
    f["r5"] = logp - logp.shift(5)
    f["r21"] = logp - logp.shift(21)
    f["r63"] = logp - logp.shift(63)
    f["r126"] = logp - logp.shift(126)
    f["mom12_1"] = logp.shift(21) - logp.shift(252)
    f["hi52"] = adj / adj.rolling(252, min_periods=200).max() - 1
    f["lo52"] = adj / adj.rolling(252, min_periods=200).min() - 1
    f["vol21"] = ret.rolling(21, min_periods=15).std()
    f["vol63"] = ret.rolling(63, min_periods=45).std()
    f["volratio"] = f["vol21"] / f["vol63"]
    f["gap50"] = adj / adj.rolling(50, min_periods=40).mean() - 1
    f["gap200"] = adj / adj.rolling(200, min_periods=150).mean() - 1
    f["tsurge"] = turn.rolling(5, min_periods=3).mean() / turn.rolling(63, min_periods=40).mean()
    cov = ret.mul(ew, axis=0).rolling(63, min_periods=45).mean() - ret.rolling(63, min_periods=45).mean().mul(
        ew.rolling(63, min_periods=45).mean(), axis=0)
    var = ew.rolling(63, min_periods=45).var(ddof=0)
    beta = cov.div(var, axis=0)
    f["beta63"] = beta
    m21 = mlog - mlog.shift(21)
    m5 = mlog - mlog.shift(5)
    f["resid21"] = f["r21"] - beta.mul(m21, axis=0)
    f["resid5"] = f["r5"] - beta.mul(m5, axis=0)

    stock = {k: v.where(eligible).rank(axis=1, pct=True) - 0.5 for k, v in f.items()}

    m = pd.DataFrame(index=adj.index)
    m["m_r5"] = m5
    m["m_r21"] = m21
    m["m_r63"] = mlog - mlog.shift(63)
    m["m_vol21"] = ew.rolling(21, min_periods=15).std() * np.sqrt(252)
    level = np.exp(mlog)
    m["m_gap200"] = level / level.rolling(200, min_periods=150).mean() - 1
    m["m_breadth50"] = (f["gap50"] > 0).where(eligible).mean(axis=1)
    m["m_disp21"] = f["r21"].where(eligible).std(axis=1)
    mz = (m - m.rolling(756, min_periods=252).mean()) / m.rolling(756, min_periods=252).std()

    bench = px["adj"][market_se.BENCHMARK_SYMBOL].ffill(limit=5) if market_se.BENCHMARK_SYMBOL in px["adj"] else None
    return Data(adj.index, eq, adj, adj_open, eligible, stock, mz, ew, bench)


def forward_return(d: Data, horizon: int) -> pd.DataFrame:
    """Retorno da abertura de t+1 ao fechamento de t+H (execucao realista do sinal de t)."""
    entry = d.adj_open.shift(-1).fillna(d.adj.shift(-1))
    return d.adj.shift(-horizon) / entry - 1.0


def weekly_dates(days: pd.DatetimeIndex, start, end) -> pd.DatetimeIndex:
    sel = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    s = pd.Series(sel, index=sel)
    iso = sel.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).max().sort_values().values)


def panel(d: Data, horizon: int) -> pd.DataFrame:
    """Linhas (data, ticker) elegiveis com features e rotulos."""
    fwd = forward_return(d, horizon)
    fwd_e = fwd.where(d.eligible)
    excess = fwd_e.sub(fwd_e.mean(axis=1), axis=0)
    cols = {k: v.stack(future_stack=True) for k, v in d.stock.items()}
    p = pd.DataFrame(cols)
    p["y"] = excess.stack(future_stack=True)
    p["fwd"] = fwd_e.stack(future_stack=True)
    p = p[d.eligible.stack(future_stack=True).reindex(p.index).fillna(False).values]
    p.index.names = ["date", "ticker"]
    mk = d.market.reindex(p.index.get_level_values("date"))
    for c in MARKET_FEATURES:
        p[c] = mk[c].values
    return p


# --------------------------------------------------------------------------
# Walk-forward models
# --------------------------------------------------------------------------
def _model(kind: str):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=50.0)
    if kind == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=400,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                             random_state=7, verbose=-1)
    raise ValueError(kind)


def retrain_points(start, end) -> List[pd.Timestamp]:
    pts, t = [], pd.Timestamp(start)
    while t <= pd.Timestamp(end):
        pts.append(t)
        t = t + pd.DateOffset(months=RETRAIN_MONTHS)
    return pts


def purged_cutoff(days: pd.DatetimeIndex, retrain_at: pd.Timestamp, horizon: int) -> pd.Timestamp:
    """Ultima data de amostra cujo rotulo (H pregoes + abertura seguinte) ja se realizou antes de retrain_at."""
    i = days.searchsorted(retrain_at) - 1           # last trading day before the retrain date
    return days[max(0, i - horizon - 1)]


def walk_forward(d: Data, p: pd.DataFrame, kind: str, horizon: int, start, end) -> pd.Series:
    """Previsoes fora da amostra nas datas semanais de [start, end]."""
    feats = STOCK_FEATURES + MARKET_FEATURES
    wd = weekly_dates(d.days, start, end)
    pts = retrain_points(start, end) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    dates = p.index.get_level_values("date")
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        cut = purged_cutoff(d.days, a, horizon)
        tr = p[(dates >= TRAIN_START) & (dates <= cut) & p["y"].notna()]
        tr = tr.dropna(subset=["y"])
        y = tr["y"].clip(tr["y"].quantile(0.01), tr["y"].quantile(0.99))
        m = _model(kind)
        m.fit(tr[feats].fillna(0.0).values, y.values)
        block = wd[(wd >= a) & (wd < b)]
        te = p[dates.isin(block)]
        if len(te):
            out.append(pd.Series(m.predict(te[feats].fillna(0.0).values), index=te.index))
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)


def market_walk_forward(d: Data, horizon: int, start, end) -> pd.Series:
    """Previsao do retorno do universo (peso igual) nas datas semanais."""
    from sklearn.linear_model import Ridge
    fwd = forward_return(d, horizon).where(d.eligible).mean(axis=1)
    X = d.market[MARKET_FEATURES]
    wd = weekly_dates(d.days, start, end)
    pts = retrain_points(start, end) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        cut = purged_cutoff(d.days, a, horizon)
        rows = (X.index >= TRAIN_START) & (X.index <= cut)
        tr = X[rows].join(fwd.rename("y")).dropna()
        m = Ridge(alpha=50.0).fit(tr[MARKET_FEATURES].values, tr["y"].values)
        block = wd[(wd >= a) & (wd < b)]
        xb = X.reindex(block).fillna(0.0)
        if len(xb):
            out.append(pd.Series(m.predict(xb.values), index=block))
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)


def cached_predictions(d: Data, kind: str, horizon: int, start, end) -> Tuple[pd.Series, pd.Series]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{kind}_H{horizon}_{start}_{end}"
    fp, fm = CACHE_DIR / f"pred_{tag}.pkl", CACHE_DIR / f"mkt_H{horizon}_{start}_{end}.pkl"
    if fp.exists() and fm.exists():
        return pd.read_pickle(fp), pd.read_pickle(fm)
    p = panel(d, horizon)
    pred = walk_forward(d, p, kind, horizon, start, end)
    mpred = market_walk_forward(d, horizon, start, end)
    pred.to_pickle(fp)
    mpred.to_pickle(fm)
    return pred, mpred


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Spec3:
    model: str = "lgbm"
    horizon: int = 20
    entry_q: float = 0.05        # entra no topo `entry_q` das previsoes
    timing: bool = False         # previsao de queda do mercado -> caixa
    k: int = market_se.MAX_POSITIONS
    exit_rank: float = 0.5       # sai (depois de H pregoes) se cair abaixo deste percentil
    max_hold_mult: float = 2.0

    def label(self) -> str:
        return f"{self.model}-H{self.horizon}-q{int(round(100 * self.entry_q))}{'-timing' if self.timing else ''}-K{self.k}"


@dataclass
class Result3:
    equity: pd.Series
    trades: pd.DataFrame
    decisions: List[Dict]
    invested: pd.Series
    skipped: int = 0
    slots_used: pd.Series = field(default_factory=pd.Series)

    @property
    def taken(self) -> pd.DataFrame:
        return self.trades


def simulate(d: Data, pred: pd.Series, mpred: pd.Series, spec: Spec3, start, end,
             capital: float = market_se.CAPITAL_SEK, brokerage: float = market_se.BROKERAGE_PCT_PER_ORDER,
             slippage: float = market_se.SLIPPAGE_PCT_PER_SIDE) -> Result3:
    days = d.days[(d.days >= pd.Timestamp(start)) & (d.days <= pd.Timestamp(end))]
    pos_of = {t: i for i, t in enumerate(d.days)}
    decide = set(weekly_dates(d.days, start, end)) & set(pred.index.get_level_values("date"))
    by_date = {dt: s.droplevel(0) for dt, s in pred.groupby(level=0)}
    cash = float(capital)
    pos: Dict[str, Dict] = {}
    trades, decisions = [], []
    equity, invested, slots = {}, {}, {}
    pending = None

    def opx(t, dt):
        v = d.adj_open.at[dt, t]
        if not np.isfinite(v) or v <= 0:
            v = d.adj.at[dt, t]
        return float(v) if np.isfinite(v) else np.nan

    for dt in days:
        if pending is not None:
            sells, buys, slot = pending
            pending = None
            for t in sells:
                if t not in pos:
                    continue
                p = opx(t, dt)
                if not np.isfinite(p):
                    continue
                proceeds = pos[t]["n"] * p * (1 - slippage) * (1 - brokerage)
                cash += proceeds
                b = pos.pop(t)
                trades.append({"ticker": t, "data_compra": b["date"], "preco_compra": b["px"],
                               "data_venda": str(dt.date()), "preco_venda": p,
                               "resultado_liq": proceeds / b["paid"] - 1.0, "status": "fechada",
                               "pregoes": pos_of[dt] - b["i"]})
            for t in buys:
                if len(pos) >= spec.k:
                    break
                p = opx(t, dt)
                amount = min(slot, cash)
                if not np.isfinite(p) or amount < 0.5 * slot:
                    continue
                pos[t] = {"n": amount * (1 - brokerage) / (p * (1 + slippage)), "date": str(dt.date()),
                          "px": p, "paid": amount, "i": pos_of[dt], "last": p}
                cash -= amount
        for t, b in pos.items():
            v = d.adj.at[dt, t]
            if np.isfinite(v):
                b["last"] = float(v)
        stock_val = sum(b["n"] * b["last"] for b in pos.values())
        eq = cash + stock_val
        equity[dt], invested[dt], slots[dt] = eq, (stock_val / eq if eq > 0 else 0.0), len(pos)

        if dt in decide and dt != days[-1]:
            s = by_date.get(dt, pd.Series(dtype=float))
            rank = s.rank(pct=True)
            risk_off = bool(spec.timing and float(mpred.get(dt, 0.0)) < 0.0)
            i_now = pos_of[dt]
            sells = []
            for t, b in pos.items():
                held = i_now - b["i"]
                r = float(rank.get(t, 0.0))
                if risk_off or (held >= spec.horizon and (r < spec.exit_rank or held >= spec.max_hold_mult * spec.horizon)):
                    sells.append(t)
            buys = []
            if not risk_off:
                cand = s[(rank >= 1.0 - spec.entry_q)].sort_values(ascending=False)
                free = spec.k - (len(pos) - len(sells))
                buys = [t for t in cand.index if t not in pos][:max(0, free)]
            decisions.append({"date": str(dt.date()), "risk_off": risk_off, "market_pred": float(mpred.get(dt, np.nan)),
                              "sell": sells, "buy": buys, "equity": round(eq, 2)})
            pending = (sells, buys, eq / spec.k)

    for t, b in pos.items():
        val = b["n"] * b["last"] * (1 - slippage) * (1 - brokerage)
        trades.append({"ticker": t, "data_compra": b["date"], "preco_compra": b["px"], "data_venda": "",
                       "preco_venda": b["last"], "resultado_liq": val / b["paid"] - 1.0, "status": "aberta",
                       "pregoes": pos_of[days[-1]] - b["i"]})
    return Result3(pd.Series(equity).sort_index(), pd.DataFrame(trades), decisions,
                   pd.Series(invested).sort_index(), 0, pd.Series(slots).sort_index())


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def ic_series(d: Data, pred: pd.Series, horizon: int) -> pd.Series:
    """Correlacao de Spearman, em cada data, entre previsao e retorno realizado em excesso."""
    fwd = forward_return(d, horizon).where(d.eligible)
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    out = {}
    for dt, s in pred.groupby(level=0):
        s = s.droplevel(0)
        y = excess.loc[dt].reindex(s.index)
        ok = y.notna()
        if ok.sum() >= 10:
            out[dt] = s[ok].rank().corr(y[ok].rank())
    return pd.Series(out).sort_index()


def newey_west_t(x: pd.Series, lags: int) -> float:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = np.dot(e, e) / n
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        s += 2 * w * np.dot(e[l:], e[:-l]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def stationary_bootstrap_p(x: pd.Series, mean_block: float = 20.0, reps: int = 5000, seed: int = 11) -> float:
    """p-valor unilateral de media <= 0 (Politis-Romano), por reamostragem da serie centrada."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    obs = x.mean()
    xc = x - obs
    p_new = 1.0 / mean_block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(n, size=reps)
    for i in range(1, n):  # vectorised across replications
        jump = rng.random(reps) < p_new
        idx[:, i] = np.where(jump, rng.integers(n, size=reps), (idx[:, i - 1] + 1) % n)
    means = xc[idx].mean(axis=1)
    return float((np.sum(means >= obs) + 1) / (reps + 1))


def deflated_sharpe(returns: pd.Series, trial_srs: List[float]) -> Dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado), com Sharpe por periodo (diario)."""
    from scipy.stats import kurtosis, norm, skew
    r = np.asarray(pd.Series(returns).dropna(), dtype=float)
    T = len(r)
    sr = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    N = max(2, len(trial_srs))
    v = float(np.var(trial_srs, ddof=1)) if len(trial_srs) > 1 else 0.0
    emc = 0.5772156649
    sr0 = math.sqrt(max(v, 1e-12)) * ((1 - emc) * norm.ppf(1 - 1 / N) + emc * norm.ppf(1 - 1 / (N * math.e)))
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
    return {"sr_daily": float(sr), "sr_annual": float(sr * math.sqrt(252)), "sr0_daily": float(sr0),
            "n_trials": N, "dsr": dsr}


def load() -> Data:
    import prices_se
    return build(prices_se.load())


# --------------------------------------------------------------------------
# Development, pre-registration and the single holdout run
# --------------------------------------------------------------------------
GRID = [Spec3(model=m, horizon=h, entry_q=q, timing=tm)
        for m in ("ridge", "lgbm") for h in (10, 20) for q in (0.03, 0.08) for tm in (False, True)]
MAX_TRADES_PER_YEAR = 80
MIN_DEV_IC_T = 2.0
# Selection rule (fixed before the development results were seen): among the
# variants whose model has a dev IC with Newey-West t >= 2 and that make at most
# 80 trades a year, the highest daily Sharpe of the excess over the benchmark
# (2013-03..2019-12, the part of the dev period the benchmark exists); ties ->
# fewer trades.


def excess_series(eq: pd.Series, bench: pd.Series) -> pd.Series:
    b = bench.reindex(eq.index).ffill()
    return (eq.pct_change() - b.pct_change()).dropna()


def run_metrics(d: Data, res: Result3, start, end) -> Dict[str, float]:
    eq = res.equity
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    out = {"cagr": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1), "mdd": float((eq / eq.cummax() - 1).min()),
           "trades_per_year": float(len(res.trades) / yrs), "invested": float(res.invested.mean())}
    ew = (1 + d.mkt_ret.reindex(eq.index).fillna(0)).cumprod()
    out["ew_cagr"] = float((ew.iloc[-1] / ew.iloc[0]) ** (1 / yrs) - 1)
    b = d.bench.reindex(eq.index).ffill().dropna()
    if len(b) > 250:
        e2 = eq.reindex(b.index)
        y2 = (b.index[-1] - b.index[0]).days / 365.25
        out["bench_cagr"] = float((b.iloc[-1] / b.iloc[0]) ** (1 / y2) - 1)
        out["cagr_vs_bench_window"] = float((e2.iloc[-1] / e2.iloc[0]) ** (1 / y2) - 1)
        out["bench_mdd"] = float((b / b.cummax() - 1).min())
        x = excess_series(e2, b)
        out["excess_sr_daily"] = float(x.mean() / x.std()) if x.std() > 0 else 0.0
    return out


def dev_grid(d: Data) -> pd.DataFrame:
    rows = []
    ic_stats = {}
    for spec in GRID:
        pred, mpred = cached_predictions(d, spec.model, spec.horizon, *DEV)
        key = (spec.model, spec.horizon)
        if key not in ic_stats:
            ic = ic_series(d, pred, spec.horizon)
            ic_stats[key] = (float(ic.mean()), newey_west_t(ic, max(1, spec.horizon // 5)))
        res = simulate(d, pred, mpred, spec, *DEV)
        m = run_metrics(d, res, *DEV)
        rows.append({"spec": spec.label(), **asdict(spec), "ic": ic_stats[key][0], "ic_t": ic_stats[key][1], **m})
    return pd.DataFrame(rows)


def select(grid: pd.DataFrame) -> pd.Series:
    ok = grid[(grid["ic_t"] >= MIN_DEV_IC_T) & (grid["trades_per_year"] <= MAX_TRADES_PER_YEAR)]
    if ok.empty:
        raise SystemExit("Nenhuma variacao passa nos filtros de desenvolvimento: nao ha o que congelar.")
    return ok.sort_values(["excess_sr_daily", "trades_per_year"], ascending=[False, True]).iloc[0]


def frozen() -> Tuple[Spec3, List[float]]:
    j = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    return Spec3(**j["spec"]), list(j["dev_trial_excess_sr_daily"])


def holdout(d: Data, spec: Spec3, trial_srs: List[float]) -> Dict:
    """Avaliacao unica (ADR 0007): previsoes, estatisticas e criterio 1 no holdout."""
    import portfolio_se as ps
    import rotation_se as rot
    import walkforward as wf

    pred, mpred = cached_predictions(d, spec.model, spec.horizon, *HOLDOUT)
    ic = ic_series(d, pred, spec.horizon)
    res = simulate(d, pred, mpred, spec, *HOLDOUT)
    m = run_metrics(d, res, *HOLDOUT)
    x = excess_series(res.equity, d.bench)
    dsr = deflated_sharpe(x, trial_srs)
    out = {"spec": spec.label(), "period": HOLDOUT, **m,
           "ic_mean": float(ic.mean()), "ic_t_nw": newey_west_t(ic, max(1, spec.horizon // 5)),
           "ic_weeks": int(len(ic)), "ic_positive_share": float((ic > 0).mean()),
           "excess_daily_mean": float(x.mean()), "excess_p_bootstrap": stationary_bootstrap_p(x),
           **{f"dsr_{k}": v for k, v in dsr.items()}}
    t = res.trades
    closed = t[t["status"] == "fechada"] if not t.empty else t
    out["trade_mean"] = float(closed["resultado_liq"].mean()) if len(closed) else float("nan")
    out["trade_p_bootstrap"] = stationary_bootstrap_p(closed["resultado_liq"], mean_block=3.0) if len(closed) > 20 else float("nan")
    out["trades_closed"] = int(len(closed))

    folds = wf.folds()
    rc = simulate(d, pred, mpred, spec, folds[0].trade_start, folds[-1].trade_end)
    rc.trades = rot.assign_folds(rc.trades)
    v = ps.verdict(rc, d.bench)
    out["criterion1"] = {k: val for k, val in v.items() if k not in ("bench_equity",)}
    out["tests"] = {
        "S1_ic_significant": bool(out["ic_t_nw"] >= 2.0),
        "S2_excess_significant": bool(out["excess_p_bootstrap"] < 0.05),
        "S3_deflated_sharpe": bool(out["dsr_dsr"] > 0.95),
        "C1_criterion1": bool(v["pass"]),
    }
    out["pass"] = all(out["tests"].values())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sistema": res.equity, "indice": d.bench.reindex(res.equity.index).ffill()}).to_csv(REPORT_DIR / "equity.csv")
    res.trades.to_csv(REPORT_DIR / "trades.csv", index=False)
    pd.DataFrame(res.decisions).to_csv(REPORT_DIR / "decisions.csv", index=False)
    ic.rename("ic").to_csv(REPORT_DIR / "ic_weekly.csv")
    (REPORT_DIR / "verdict.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    _chart(res, d.bench, REPORT_DIR / "equity.png", spec)
    return out


def _chart(res: Result3, bench: pd.Series, path: Path, spec: Spec3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq = res.equity
    b = bench.reindex(eq.index).ffill()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq.index, eq.values, label=f"Estrategia v3 ({spec.label()})", color="#1565c0")
    ax.plot(b.index, eq.iloc[0] * b / b.iloc[0], label=f"Indice {market_se.BENCHMARK_SYMBOL} (com dividendos)", color="#9e9e9e")
    ax.set_ylabel("SEK")
    ax.set_title("Estrategia v3 — holdout 2020-2026 vs index benchmark")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--model", default="lgbm")
    sub.add_parser("dev")
    sub.add_parser("holdout")
    args = ap.parse_args(argv)
    d = load()
    if args.cmd == "predict":
        pred, _ = cached_predictions(d, args.model, args.horizon, *DEV)
        ic = ic_series(d, pred, args.horizon)
        print(f"IC medio {ic.mean():.4f} t_NW {newey_west_t(ic, max(1, args.horizon // 5)):.2f}")
        return 0
    if args.cmd == "dev":
        g = dev_grid(d)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        g.to_csv(REPORT_DIR / "dev_grid.csv", index=False)
        print(g.round(4).to_string(index=False))
        print("\nescolhida pela regra:", select(g)["spec"])
        return 0
    spec, trials = frozen()
    out = holdout(d, spec, trials)
    print(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

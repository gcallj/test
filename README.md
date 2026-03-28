# Pipeline de Sinais de Trading — Mercado Brasileiro

Pipeline completo de geração de sinais de compra para ~94 ativos do mercado brasileiro (ações, FIIs, índices, moedas, commodities). Combina modelos de classificação binária, regressão de preços e otimização genética (GA) com walk-forward backtesting.

> **Ultima atualizacao (27/Mar/2026)**
> Win Rate: **59.5%** | Trades/ticker: **105** | MDD: **-21.1%** | Custos: **R$7 corretagem + B3 + IR 15%**

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Requisitos](#2-requisitos)
3. [Como Executar](#3-como-executar)
4. [Step 1 — ETL](#4-step-1--etl-stock_etlpy)
5. [Step 2 — Modelos Binários](#5-step-2--modelos-binários-stock_bin_modelspy)
6. [Step 3 — Regressão](#6-step-3--regressão-stock_reg_modelspy)
7. [Step 4 — Output Final](#7-step-4--output-final-stock_final_outputpy)
8. [Step 5 — GA (Otimização Genética)](#8-step-5--ga-otimização-genética-ga_runpy)
9. [Lógica de Entrada (Quando Comprar)](#9-lógica-de-entrada-quando-comprar)
10. [Lógica de Saída (Quando Vender)](#10-lógica-de-saída-quando-vender)
11. [Saídas no Modo Apply (Planilha)](#11-saídas-no-modo-apply-planilha)
12. [Função de Fitness](#12-função-de-fitness)
13. [Arquivos de Saída](#13-arquivos-de-saída)
14. [FAQ](#14-faq)

---

## 1. Visão Geral

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  Step 1      │    │  Step 2       │    │  Step 3      │    │  Step 4       │    │  Step 5       │
│  ETL         │───►│  BIN Models   │───►│  Regressão   │───►│  Final Output │───►│  GA           │
│              │    │              │    │              │    │              │    │              │
│ stock_etl.py │    │ stock_bin_   │    │ stock_reg_   │    │ stock_final_ │    │ ga_run.py     │
│              │    │ models.py    │    │ models.py    │    │ output.py    │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
   ~30 min            ~2.5 h             ~10 min            ~5 min              ~4-8 h (train)
                                                                                ~2 min (load)
```

**Fluxo de dados:**
1. **ETL** baixa OHLCV via yfinance, gera ~222 features técnicas/microestruturais/cross-ticker, cria targets binários e contínuos → `expanded_stock_reduced.parquet`
2. **BIN** treina ensemble de 4-6 classificadores (LGBM, HGB, RF, ET, XGB, CatBoost) para prever probabilidade de alta (+25%) e queda (-10%) → `ensemble_signals_history.parquet`
3. **REG** gera previsões de preço de compra/venda com bandas de erro calibradas (35% ML + 65% baseline) → `forecast_history_wide.parquet`
4. **FINAL** consolida BIN + REG + fundamentais, calcula Expected Value (EV) → `history_consolidated.parquet`
5. **GA** otimiza 26 parâmetros de trading via algoritmo genético com walk-forward → `summary_latest.xlsx` + `apply_last_5d__H5.csv`

**Premissa fundamental:** `LONG_ONLY = True` — o sistema só opera comprado. Não há short selling.

---

## 2. Requisitos

- **Python 3.10** (obrigatório — scipy ABI incompatível com 3.12)
- Bibliotecas: `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `optuna`, `yfinance`, `ta`, `openpyxl`
- Opcionais: `xgboost`, `catboost` (modelos adicionais no Step 2)

```bash
# Caminho recomendado no Windows:
/c/Users/gabri/AppData/Local/Programs/Python/Python310/python.exe
```

---

## 3. Como Executar

### 3.1 Pipeline completa (treina tudo — ~6-12 horas)

```bash
# Steps 1-4 (ETL + ML models + consolidação)
python run_pipeline.py                    # ~3-4 horas

# Step 5 (GA — otimização genética)
# Editar ga_run.py: RUN_MODE = "train"
python ga_run.py                          # ~4-8 horas
```

### 3.2 Atualização diária (sem retreinar — ~40 min)

```bash
# 1) Atualizar dados de mercado (reutiliza modelos ML salvos)
python run_pipeline.py --mode data-only   # ~35-40 min

# 2) Gerar sinais com GA salvo (modo load — sem retreinar)
# Verificar ga_run.py: RUN_MODE = "load"
python ga_run.py                          # ~2 min

# 3) Abrir summary_latest.xlsx → aba "Apply"
```

### 3.3 Steps específicos

```bash
python run_pipeline.py --steps 1          # Só ETL
python run_pipeline.py --steps 2,3,4      # Pular ETL
python run_pipeline.py --steps 1,4        # ETL + Final (= data-only)
```

### 3.4 Pré-requisitos para modo "load"

Todos os arquivos de cache devem existir (gerados por pelo menos uma execução completa):
- `./output/ensemble_signals_history.parquet` (BIN)
- `./output/forecast_history_wide.parquet` (REG)
- `./output/history_consolidated.parquet` (FINAL)
- `./global_ga_checkpoint.json` (GA)

---

## 4. Step 1 — ETL (`stock_etl.py`)

### Objetivo
Baixar dados OHLCV, criar ~222 features técnicas/microestruturais/cross-ticker, e gerar targets de classificação e regressão.

### Universo de ativos (~139 baixados, ~94 operados pelo GA)

| Categoria | Qtd | Exemplos |
|-----------|-----|---------|
| **Ações brasileiras** | ~75 | VALE3, PETR4, ITUB4, BBAS3, WEGE3 |
| **FIIs** | ~20 | KNRI11, MXRF11, HGLG11 |
| **Índices globais** | ~16 | ^GSPC (S&P), ^GDAXI (DAX), ^N225 (Nikkei) |
| **Moedas/commodities** | ~28 | EURUSD=X, GC=F (ouro), CL=F (petróleo), BTC-USD |

> O GA filtra para ~94 tickers `.SA` com mínimo de 350 barras de dados históricas.

### Features geradas (~222 colunas)

| Categoria | Exemplos | Qtd aprox |
|-----------|----------|-----------|
| **Indicadores técnicos** | RSI, MACD, Stochastic, Bollinger, ATR, ADX, CCI, Williams %R | ~80 |
| **Cruzamentos de médias** | SMA 1/2/5/10/15/20/25/50/100 — todas as combinações fast < slow | ~36 |
| **Padrões gráficos** | Head & Shoulders, Double Top/Bottom, Triângulos, Canais, Wedges | ~20 |
| **Microestrutura** | `vol_relative_20d`, `amihud_illiquidity_20d`, `gk_volatility_20d` | 3 |
| **Cross-ticker** | `beta_ibov_60d`, `rel_strength_ibov_20d`, `corr_ibov_60d`, `vix_percentile_252d` | 4 |
| **Fundamentais** | Dividend yield, P/L, P/VPA, market cap | 4 |
| **Derivados** | Returns multi-período, volume profiles, volatilidade realizada | ~75 |

**Detalhes das features de microestrutura:**

| Feature | Fórmula | Interpretação |
|---------|---------|---------------|
| `vol_relative_20d` | `volume_dia / média_volume_20d` | > 1 = volume acima da média (interesse crescente) |
| `amihud_illiquidity_20d` | `média₂₀(|retorno| / volume)` | Maior = menos líquido (mais impacto por unidade negociada) |
| `gk_volatility_20d` | `√(média₂₀(volatilidade_OHLC_diária))` | Volatilidade Garman-Klass (mais precisa que close-to-close) |

**Detalhes das features cross-ticker:**

| Feature | Janela | Interpretação |
|---------|--------|---------------|
| `beta_ibov_60d` | 60 dias | Sensibilidade do ativo ao IBOV (1.2 = amplifica 20% os movimentos) |
| `rel_strength_ibov_20d` | 20 dias | Retorno do ativo menos retorno do IBOV (positivo = outperformance) |
| `corr_ibov_60d` | 60 dias | Correlação com IBOV (-1 a +1) |
| `vix_percentile_252d` | 252 dias | Percentil do VIX (0 = VIX mínimo histórico, 1 = VIX máximo) |

### Targets (horizonte = 90 dias, calculados com SHIFT = 1 dia para evitar leakage)

| Target | Tipo | Definição exata |
|--------|------|-----------------|
| `target_up20` | Binário (0/1) | 1 se `max(High[t+1..t+90]) ≥ Close[t] × 1.25` |
| `target_dd5` | Binário (0/1) | 1 se `min(Low[t+1..t+90]) ≤ Close[t] × 0.90` |
| `target_best_entry` | Contínuo (%) | `(min(Low[t+1..t+90]) / Close[t]) - 1` |
| `target_best_sale` | Contínuo (%) | `(max(High[t+1..t+90]) / Close[t]) - 1` |

### Output principal

**`./output/data/expanded_stock_reduced.parquet`** (~314 MB)

Formato MultiIndex — nível 0 = variável, nível 1 = ticker, índice = DatetimeIndex (2005 até hoje).

```python
import pandas as pd
df = pd.read_parquet("./output/data/expanded_stock_reduced.parquet")
# Acessar Close da VALE3:
df["Close"]["VALE3.SA"].tail()
# Acessar target_up20 de todos os tickers:
df["target_up20"].tail()
# Todas as variáveis disponíveis:
df.columns.get_level_values(0).unique()
```

### Configurações-chave

```python
START_DATE     = "2005-01-01"    # Início dos dados históricos
HORIZON        = 90              # Janela forward para targets (dias úteis)
UP_THR         = 0.25            # +25% → target_up20 = 1
DD_THR         = -0.10           # -10% → target_dd5 = 1
SHIFT_FEATURES = 1               # Lag de 1 dia (evita vazamento futuro)
```

---

## 5. Step 2 — Modelos Binários (`stock_bin_models.py`)

### Objetivo
Treinar ensemble de classificadores para prever probabilidade de alta (+25%) e queda (-10%) em 90 dias.

### Modelos treinados

| Modelo | Biblioteca | Trials Optuna | Arquivo salvo |
|--------|-----------|---------------|--------------|
| **LightGBM** | `lightgbm` | 15 | `*_lgbm_*.txt` |
| **HistGradientBoosting** | `sklearn` | 15 | `*_hgb_*.joblib` |
| **RandomForest** | `sklearn` | 10 | `*_rf_*.joblib` |
| **ExtraTrees** | `sklearn` | 10 | `*_et_*.joblib` |
| **XGBoost** *(se instalado)* | `xgboost` | 10 | `*_xgb_*.json` |
| **CatBoost** *(se instalado)* | `catboost` | 8 | `*_catboost_*.cbm` |

Dois conjuntos de modelos por classificador:
- `_train_*` — treinado apenas em TRAIN (para avaliação em VALID)
- `_tv_*` — treinado em TRAIN+VALID (para APPLY — dados mais recentes)

### Mecanismos anti-overfitting

- **Walk-forward CV** temporal: purge gap ≥ 90 dias entre treino e validação
- **Recency weights**: dados mais recentes pesam mais no treino
- **Asset-class feature**: diferencia ações/FIIs/índices/moedas como feature categórica
- **Threshold calibrado**: por precision × F-beta (conservador, β < 1)
- **Meta-learner** `ens_stackLGBM`: LightGBM walk-forward com OOF sobre probabilidades dos modelos base

### Como o sinal ensemble é formado

```
BUY  = pred_up20 == 1  AND  pred_dd5 == 0   → alta provável, queda improvável
SELL = pred_up20 == 0  AND  pred_dd5 == 1   → alta improvável, queda provável
HOLD = qualquer outra combinação
```

> Em `LONG_ONLY = True`, sinais SELL são ignorados pelo GA. Não abre posição vendida.

### Outputs

**`./output/ensemble_signals_history.parquet`** (histórico completo, ~368 KB)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Date` | datetime | Data de referência |
| `ticker` | str | Código do ativo |
| `split` | str | TRAIN / VALID / TEST / APPLY |
| `p_up20` | float [0,1] | Probabilidade de alta ≥ +25% em 90d |
| `p_dd5` | float [0,1] | Probabilidade de queda ≥ -10% em 90d |

**`./output/apply_ensemble_signals.csv`** (apenas partição APPLY, ~25 colunas)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `p_up20`, `p_dd5` | float | Probabilidades brutas dos modelos |
| `thr_up20`, `thr_dd5` | float | Thresholds calibrados usados |
| `pred_up20`, `pred_dd5` | int (0/1) | Predição binária final |
| `buy`, `sell` | bool | Flags de compra/venda |
| `action` | str | BUY / SELL / HOLD |
| `buy_trust` | float | `p_up20 × (1 - p_dd5)` |
| `sell_trust` | float | `(1 - p_up20) × p_dd5` |
| `action_trust` | float | Confiança da ação escolhida |
| `trust_up20`, `margin_up20` | float | Confiança e margem vs threshold (up20) |
| `trust_dd5`, `margin_dd5` | float | Confiança e margem vs threshold (dd5) |

---

## 6. Step 3 — Regressão (`stock_reg_models.py`)

### Objetivo
Prever o melhor preço de entrada (compra) e saída (venda) dentro de 90 dias, com bandas de incerteza calibradas.

### Abordagem híbrida (35% ML + 65% Baseline)

| Componente | Método | Peso |
|-----------|--------|------|
| **Baseline** | Rolling quantile (janela=60) + EWM Gaussiana (halflife=252) | 65% |
| **ML Regression** | LightGBM com objetivo L1 (MAE), features do ETL | 35% |

Pipeline de calibração:
1. Blend das previsões: `0.65 × baseline + 0.35 × ML`
2. Cálculo dos resíduos históricos
3. **Banda assimétrica**: quantis [0.02, 0.98] dos resíduos
4. **Vol-adaptation**: escala das bandas ajusta conforme regime de volatilidade (clip 0.45–2.6×)
5. **WF-calibration**: escala final ajustada para cobertura-alvo de 97%

### Outputs principais

**`./output/forecast_history_wide.parquet`** (~46 MB)

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Date` | datetime | Data |
| `ticker` | str | Ativo |
| `best_buy_price` | float | Menor preço projetado para entrada (R$) |
| `buy_err_lo` | float | Banda inferior de confiança (compra) |
| `buy_err_hi` | float | Banda superior de confiança (compra) |
| `best_sell_price` | float | Maior preço projetado para saída (R$) |
| `sell_err_lo` | float | Banda inferior de confiança (venda) |
| `sell_err_hi` | float | Banda superior de confiança (venda) |

**`./output/data/models/v35_.../apply_forecast_with_error_band.csv`** (~45 colunas detalhadas)

| Coluna | Descrição |
|--------|-----------|
| `used_ml` | True se ML regression foi aplicado |
| `pred_pct_center` | Previsão central (% vs Close) |
| `pred_pct_lo/hi_uncapped` | Banda sem cap (%) |
| `pred_pct_lo/hi_capped` | Banda com cap operacional (%) |
| `op_pct` | Percentual operacional final |
| `pred_price_center` | Preço previsto central (R$) |
| `op_price` | Preço operacional final (R$) |
| `band_scale` | Fator de escala da banda calibrada |
| `mae_valid`, `mae_test` | Erro médio absoluto na validação/teste |
| `covC_valid`, `touchOp_valid` | Cobertura e taxa de toque na validação |

---

## 7. Step 4 — Output Final (`stock_final_output.py`)

### Objetivo
Consolidar BIN + REG + fundamentais em um único DataFrame com Expected Value (EV) calculado para cada ativo por dia.

### Cálculo do Expected Value

```python
# Componente regressão (upside potencial vs downside risco)
EV_buy_reg  = f(upside_pct, downside_pct, bandas de confiança)

# Componente ensemble (probabilidades × payoff)
EV_buy_ens  = p_up20 × payoff_up - p_dd5 × loss_dd

# Combinação
EV_buy      = w_reg × EV_buy_reg + w_bin × EV_buy_ens

# Score fundamentalista
fund_score  = f(dividend_yield, trailing_pe, price_to_book, market_cap)  # [-1, +1]

# Versões do EV com ajuste fundamentalista
EV_buy_fund   = EV_buy × (1 + α₁ × fund_score)   # v1
EV_buy_fund_2 = EV_buy × (1 + α₂ × fund_score)   # v2
EV_buy_fund_3 = EV_buy × (1 + α₃ × fund_score)   # v3 ← USADO PELO GA
```

### Output principal

**`./output/history_consolidated.parquet`** (~52 MB, ~35 colunas)

| Grupo | Colunas | Descrição |
|-------|---------|-----------|
| **Identificação** | `Date`, `ticker`, `split` | Data, ativo, partição (TRAIN/VALID/TEST/APPLY) |
| **Preços OHLC** | `price`, `open`, `high`, `low`, `close` | `price` = `close` |
| **Regressão** | `best_buy_value`, `best_sell_value` | Preços-alvo de compra/venda (R$) |
| | `err_buy_pct`, `err_sell_pct` | Incerteza das estimativas (%) |
| | `downside_pct`, `upside_pct` | Risco e potencial projetados (%) |
| | `risk_return` | Razão risco/retorno da regressão |
| **Classificação** | `p_up20`, `p_dd5` | Probabilidades brutas [0,1] |
| | `pred_up20`, `pred_dd5` | Predições binárias (0/1) |
| | `up20_bin`, `dd5_bin` | Classes finais calibradas |
| | `buy_trust`, `sell_trust` | Confiança de compra/venda |
| **Expected Value** | `EV_buy_reg`, `EV_buy_ens`, `EV_buy` | EV por componente e combinado |
| | `fund_score` | Score fundamentalista [-1, +1] |
| | `EV_buy_fund_3` | **EV final — principal input do GA** |
| | `signal` | Sinal consolidado (buy/sell/hold) |
| **Fundamentais** | `dividend_yield`, `trailing_pe` | Yield e P/L |
| | `price_to_book`, `market_cap` | P/VPA e valor de mercado |

---

## 8. Step 5 — GA (Otimização Genética) (`ga_run.py`)

### Objetivo
Encontrar os 26 melhores parâmetros de trading via algoritmo genético com walk-forward backtesting em 10 janelas históricas independentes.

### Configuração importante

```python
LONG_ONLY    = True       # Apenas compra (sem short selling)
ONLY_SA      = True       # Apenas tickers .SA (Brasil)
APPLY_DAYS   = 5          # Gera sinais dos últimos 5 dias de pregão
FWD_H        = 5          # Horizonte forward do sinal (dias)
MAX_FEATURES = 40         # Features mais relevantes por ticker (Spearman)
FAST_MODE    = False      # Modo completo (10 janelas walk-forward)
RUN_MODE     = "load"     # "train" = retreina GA | "load" = usa checkpoint
```

### 26 Genes otimizados pelo GA

| # | Gene | Faixa | Descrição |
|---|------|-------|-----------|
| 1 | `vote_threshold_long` | 0.15–0.55 | % mínimo de features votando bullish para comprar |
| 2 | `vote_threshold_short` | 0.10–0.55 | % mínimo votando bearish (ignorado em LONG_ONLY) |
| 3 | `z_threshold` | 0.15–0.80 | Z-score mínimo para feature emitir voto |
| 4 | `signal_ema_span` | 2–12 | Janela EMA de suavização do score |
| 5 | `entry_confirmation_days` | 1–3 | Dias consecutivos de sinal antes de entrar |
| 6 | `score_percentile_trigger` | 0.35–0.80 | Percentil mínimo do score (vs últimos 252d) |
| 7 | `stop_atr_mult` | 1.0–2.5 | Multiplicador ATR para stop loss |
| 8 | `stop_tighten_after_bars` | 3–15 | Barras para ativar aperto do stop |
| 9 | `stop_tighten_factor` | 0.40–0.85 | Fator de redução do stop (0.70 = -30%) |
| 10 | `max_loss_per_trade_pct` | 2%–10% | Hard stop de emergência por trade |
| 11 | `reward_risk_ratio` | 1.0–5.0 | Take profit = R/R × stop_abs |
| 12 | `partial_take_pct` | 0%–60% | % da posição para 1ª saída parcial |
| 13 | `partial_take_level` | 0.5–1.5× | Nível da 1ª saída parcial (× stop_abs) |
| 14 | `time_stop_bars` | 5–25 | Máximo de barras na posição antes de forçar saída |
| 15 | `entry_discount_atr_frac` | 0.0–0.5 | Desconto da ordem limitada (fração do ATR) |
| 16 | `volatility_filter_percentile` | 0%–40% | Ignora entradas em volatilidade muito baixa |
| 17 | `score_strength_scaling` | 0.0–1.0 | Escala o desconto de entrada pela força do sinal |
| 18 | `ma_filter_period` | 100–300 | Período da média móvel de tendência |
| 19 | `ma_filter_mode` | 0–2 | 0=off, 1=soft (eleva threshold), 2=hard (bloqueia) |
| 20 | `consecutive_loss_cooldown` | 5–20 | Barras de pausa após 2+ stops consecutivos |
| 21 | `equity_drawdown_stop_pct` | 8%–22% | Circuit-breaker de portfólio |
| 22 | `vol_regime_mode` | 0–2 | 0=off, 1=alarga stops em alta vol, 2=pula trades |
| 23 | `partial_take_pct_2` | 0%–40% | % da posição para 2ª saída parcial |
| 24 | `partial_take_level_2` | 1.0–3.0× | Nível da 2ª saída parcial (× stop_abs) |
| 25 | `min_signal_strength` | 0.0–0.40 | Força mínima do sinal (`|score_ev| / score95`) |
| 26 | `trailing_stop_mode` | 0–2 | 0=fixo, 1=breakeven, 2=trailing 50% |

### GA de 2 estágios

| Estágio | Op | Pop | Gens | Objetivo |
|---------|---|-----|------|----------|
| **Stage 1** | Exploração | 50 | 30 | Cobrir o espaço de 26 dimensões |
| **Stage 2** | Refinamento | 80 | 25 | Polir o melhor genoma encontrado |

- **Walk-forward**: 10 janelas (treino = 3 anos, teste = 6 meses, step = 6 meses)
- **Early stopping**: 15 gerações sem melhoria
- **Warm-start**: Checkpoint anterior alimenta a geração inicial
- **Operadores**: Crossover 70%, Mutação 40%, Torneio k=3

---

## 9. Lógica de Entrada (Quando Comprar)

O sinal de compra é gerado no **fechamento do dia i** e executado no **dia i+1** (ou i+2 se não preencher). Todas as condições abaixo devem ser satisfeitas:

### 9.1 Pré-filtros (verificados no fechamento do dia i)

```
✅ Posição = 0 (sem posição aberta)
✅ Cooldown = 0 (não em pausa pós-perda)
✅ Vol rank ≥ volatility_filter_percentile     (evita mercado "morto")
✅ Se vol_regime_mode == 2: vol rank ≤ 0.85    (evita mercado caótico)
✅ ATR[i] ≥ 0.01                               (liquidez mínima)
✅ |score_ev[i]| / score95 ≥ min_signal_strength
```

### 9.2 Construção do sinal direcional (votação de features)

Para cada uma das ~40 features selecionadas por Spearman, calcula-se o z-score rolling:

```python
z = (feature[i] - rolling_mean[i]) / rolling_std[i]
if z > z_threshold:   votes_long  += 1/N    # voto bullish
if z < -z_threshold:  votes_short += 1/N    # voto bearish
```

O score agregado:
```python
score_raw  = votes_long - votes_short          # [-1, +1]
score_ev   = EMA(score_raw, signal_ema_span)   # suavizado
score_pctl = rolling_quantile(score_ev, score_percentile_trigger, 252d)
```

**Condição de entrada (long):**
```python
votes_long[i] >= vote_threshold_long  AND  score_ev[i] >= score_pctl[i]
```

### 9.3 Confirmação temporal

O sinal deve persistir por `entry_confirmation_days` dias consecutivos (1, 2 ou 3).

### 9.4 Filtro de tendência (média móvel)

| Modo | Se preço < MA(ma_filter_period) | Comportamento |
|------|--------------------------------|---------------|
| 0 | Sem filtro | Entra normalmente |
| 1 (soft) | Exige dobro do `vote_threshold_long` | Entrada mais seletiva contra tendência |
| 2 (hard) | Bloqueia entrada | Sem entrada contra tendência principal |

### 9.5 Execução da ordem limitada

```python
limit_px = close[i] - entry_discount_atr_frac × ATR[i] × (1 - score_strength_scaling × strength)
```

- Se `low[i+1] ≤ limit_px` → preenchido em `limit_px` no dia i+1
- Caso contrário → tenta na abertura do dia i+2
- Se não preencher → sinal expira

---

## 10. Lógica de Saída (Quando Vender)

> **Resposta direta:** Uma posição comprada é vendida quando qualquer uma das 4 condições abaixo for atingida. Verificadas **a cada barra (dia)**, nesta ordem de prioridade.

**Variáveis calculadas a cada barra:**

```python
stop_abs  = stop_atr_mult × ATR[i] × tighten × vol_adj   # nível de stop (pontos)
take_abs  = reward_risk_ratio × stop_abs                   # nível de take profit
fav       = high[i] - entry_px    # excursão favorável do dia
adv       = entry_px - low[i]     # excursão adversa do dia
max_fav   = max(max_fav, fav)     # maior excursão favorável desde a entrada
trail_adv = max_fav - fav         # recuo desde o topo
```

---

### Saída 1 — Hard Loss (gap adverso na abertura)

**Quando:** O mercado abre com gap de baixa maior que `max_loss_per_trade_pct`.

```python
hard_loss = |open[i] / entry_px - 1|
if hard_loss > max_loss_per_trade_pct:
    VENDE NA ABERTURA
```

| Parâmetro | Faixa GA | Proteção |
|-----------|---------|---------|
| `max_loss_per_trade_pct` | 2%–10% | Gap de abertura catastrófico |

**Exemplo:** Comprou R$100. Parâmetro = 7%. Mercado abre em R$92 (gap -8%) → vende imediatamente em R$92 (-8%).

---

### Saída 2 — Stop Loss (com trailing opcional)

O comportamento depende do gene `trailing_stop_mode`:

#### Modo 0 — Stop Fixo

```python
stop_hit = (adv >= stop_abs)
# adv = entry_px - low[i]  →  vende se low do dia caiu mais de stop_abs
ideal_exit = entry_px - stop_abs
exit_px = min(ideal_exit, open[i])   # slippage: pode executar no open se gap
```

#### Modo 1 — Breakeven (stop move para o pico)

Após o lucro favorável superar `stop_abs`, o stop passa a ser de "não devolver o lucro máximo":

```python
if max_fav > stop_abs:
    effective_stop = max_fav          # trail igual ao pico
    stop_hit = (trail_adv >= effective_stop)
    # trail_adv = max_fav - fav → vende se recuo desde o topo ≥ max_fav
```

**Exemplo:**
```
Entrada: R$100 | stop_abs: R$5
Preço sobe para R$110 → max_fav = R$10 > R$5
effective_stop = R$10
Se preço recua R$10 desde o topo (volta a R$100) → VENDE em R$100 (breakeven ≥ 0%)
```

#### Modo 2 — Trailing 50%

Quando o lucro supera `2 × stop_abs`, o trailing acompanha em 50% do máximo:

```python
if max_fav > 2.0 * stop_abs:
    effective_stop = 0.5 * max_fav   # trail = 50% do pico
    stop_hit = (trail_adv >= effective_stop)
```

**Exemplo:**
```
Entrada: R$100 | stop_abs: R$5
Preço sobe para R$120 → max_fav = R$20 > 2 × R$5
effective_stop = 0.5 × R$20 = R$10
Se preço recua R$10 desde o pico (de R$120 para R$110) → VENDE em R$110 (+10%)
```

**Aperto progressivo do stop:**

Após `stop_tighten_after_bars` dias em posição:
```python
stop_abs *= stop_tighten_factor   # ex: 0.70 → stop reduz 30%
```

Se nos primeiros 3 dias o trade já acumula -0.2%, aplica `stop_tighten_factor` antecipadamente.

**Ajuste por regime de vol** (`vol_regime_mode == 1`):
- Vol rank > 75%: `stop × 1.15` (mais espaço em alta volatilidade)
- Vol rank < 25%: `stop × 0.90` (menos espaço em mercado tranquilo)

---

### Saída 3 — Take Profit (com saídas parciais)

```python
take_hit = (fav >= take_abs)    # fav = high[i] - entry_px
```

**Saídas parciais** (realizações antes do take total):

| Etapa | Gatilho | % vendido | O que ocorre |
|-------|---------|-----------|-------------|
| **1ª parcial** | `fav ≥ partial_take_level × stop_abs` | `partial_take_pct` (0–60%) | Realiza parte do lucro, reduz posição |
| **2ª parcial** | `fav ≥ partial_take_level_2 × stop_abs` | `partial_take_pct_2` (0–40%) | Nova realização, posição ainda menor |
| **Take total** | `fav ≥ reward_risk_ratio × stop_abs` | Restante | Fecha posição completamente |

**Exemplo com parciais:**
```
Compra: 100 ações a R$100
ATR = R$2 | stop_abs = 2×R$2 = R$4
take_abs = 3.0 × R$4 = R$12 (take total em R$112)
partial_take_level = 0.75 → 1ª parcial em fav=R$3 (preço R$103): vende 30% → fica 70 ações
partial_take_level_2 = 1.50 → 2ª parcial em fav=R$6 (preço R$106): vende 20% → fica 56 ações
take total em fav=R$12 (preço R$112): vende 56 ações restantes
```

---

### Saída 4 — Time Stop (capital estagnado)

```python
time_stop = (barras_na_posição >= time_stop_bars) AND (fav < 0.5 × stop_abs)
```

Encerra o trade se ficou `time_stop_bars` dias (5–25) sem lucro significativo.

**Lógica:** Capital preso em um trade sem movimento favorável não pode ser realocado para oportunidades melhores.

---

### Diagrama completo de decisão por barra

```
POSIÇÃO ABERTA → a cada fechamento de dia:
│
├─ 1. open_hoje gerou gap adverso ≥ max_loss_per_trade_pct?
│     SIM → VENDE NA ABERTURA (emergência)
│
├─ 2. Stop atingido no intraday?
│     ├─ Modo 0: adv ≥ stop_abs (stop fixo)
│     ├─ Modo 1: trail_adv ≥ max_fav (não devolve pico)
│     └─ Modo 2: trail_adv ≥ 0.5×max_fav (trailing 50%)
│     SIM → VENDE próximo do nível de stop
│
├─ 3. Lucro parcial atingido?
│     ├─ fav ≥ partial_take_level × stop_abs → vende partial_take_pct %
│     └─ fav ≥ partial_take_level_2 × stop_abs → vende partial_take_pct_2 %
│     (posição continua aberta, só reduzida)
│
├─ 4. Take profit total atingido?
│     fav ≥ reward_risk_ratio × stop_abs?
│     SIM → VENDE tudo no nível do take profit
│
└─ 5. Time stop expirou?
      barras ≥ time_stop_bars AND fav < 0.5 × stop_abs?
      SIM → VENDE no fechamento
```

### Cooldown pós-perda consecutiva

Após 2+ stops consecutivos com resultado negativo:

```python
cooldown = consecutive_loss_cooldown   # ex: 10 barras sem novas entradas
```

Protege contra regimes de mercado adversos onde os stops são acionados repetidamente.

---

## 11. Saidas no Modo Apply (Planilha)

O modo apply (`GA_RUN_MODE=load`) aplica o genoma treinado aos dados mais recentes e gera sinais operacionais. Inclui custos reais brasileiros (corretagem R$7, B3, IR 15%).

### 11.1 Colunas do `apply_last_5d__H5.csv`

| Coluna | Descricao | Interpretacao |
|--------|-----------|--------------|
| `Date` | Data do pregao | Ultimo dia de dados disponiveis |
| `ticker` | Codigo Yahoo Finance | Ex: `PETR4.SA`, `VALE3.SA` |
| `signal` | **Sinal de trading** | `buy` = comprar; `hold` = aguardar |
| `potencial` | **Classificacao do trade** | `alto` = lucro liq >=10% + WR >=60%; `medio` = liq >=5%; `baixo` = demais; `-` = hold |
| `rank` | **Score de ranking** (0-100) | Combina 30% confidence + 30% upside + 15% R:R + 10% win_rate + 10% regime + 5% stop quality |
| `confidence` | **Confianca no sinal** (0-100) | Qualidade do backtest: 20% win_rate + 15% distribuicao + 15% viabilidade + 10% backtest + 10% sinal + 10% acordo + 10% regime + 10% ML |
| `close` | Preco de fechamento | Ultimo preco de referencia |
| `entrada` | **Preco de compra** (ordem limite) | Estimado no P30 das dips intraday historicas (70% chance de execucao). Sempre abaixo do close (min 0.5% desconto). Piso: minima 20 dias |
| `stop` | **Stop loss** (R$) | ATR-based, minimo 2% do preco de entrada. Consistente com logica do backtest |
| `alvo` | **Take profit** (R$) | Baseado na resistencia 20 dias (vende a 98% da maxima). Deve cobrir custos + IR |
| `RR` | **Risk/Reward ratio** | Alvo / Stop. Minimo 1.5. Valores altos = maior assimetria |
| `stop_pct` | Stop em % da entrada | Quanto pode perder se parar no stop |
| `alvo_pct` | Alvo em % da entrada | Quanto pode ganhar se atingir o alvo |
| `win_rate` | Taxa de acerto historica (%) | Percentual de trades lucrativos no backtest. Todos tickers com >20 trades |
| `custo_pct` | Custo total por trade (%) | R$7 corretagem x2 + B3 0.065% + slippage 0.20% = ~0.55% |
| `lucro_liq_pct` | **Lucro liquido estimado** (%) | (alvo% - custo%) x (1 - IR 15%). O que realmente sobra no bolso |
| `queda_max` | Drawdown maximo esperado (%) | Estimativa baseada no MDD historico + volatilidade atual |
| `regime` | Condicao de mercado | `favoravel` = acima MA200+MA50; `neutro` = misto; `desfavoravel` = abaixo |

### 11.2 Classificacao de potencial

| Potencial | Criterio | Acao sugerida |
|-----------|----------|---------------|
| **alto** | `lucro_liq_pct >= 10%` E `win_rate >= 60%` | Prioridade maxima. Alto retorno + alto acerto |
| **medio** | `lucro_liq_pct >= 5%` | Bom risco-retorno, operar com tamanho normal |
| **baixo** | Demais buys | Cautela. Considerar apenas se diversificando |
| `-` | Hold signals | Nao operar |

### 11.3 Como operar

**1. Filtrar:** `signal == "buy"` + `potencial` = "alto" ou "medio"

**2. Montar ordem:**
- Tipo: **Ordem limitada** (BUY LIMIT)
- Preco: coluna `entrada`
- Stop Loss: coluna `stop`
- Take Profit: coluna `alvo`
- Validade: 1 dia (se nao preencher, reavaliar)

**3. Gerenciar posicao:**
- **Stop**: manter no nivel da coluna `stop`
- **Breakeven**: apos lucro >= 1x stop, mover stop para preco de entrada
- **Time stop**: se em 25 dias nao bateu stop nem alvo, fechar
- **Hard stop**: se abertura com gap > 2% contra, fechar imediatamente

**4. Custos ja considerados:**
- Corretagem: R$7 por ordem (compra + venda = R$14)
- Taxas B3: 0.065% (emolumentos + liquidacao)
- Slippage: 0.20% estimado
- IR: 15% sobre lucro liquido (swing trade)
- A coluna `lucro_liq_pct` ja desconta TUDO

### 11.4 Relacao entre niveis de preco

```
close (preco atual do dia)
  |
  +-- entrada = close - desconto (P30 das dips intraday 60d)
        |
        +-- stop  = entrada - max(ATR, 2% x entrada)
        +-- alvo  = 98% x maxima_20_dias (resistencia)
        |
        lucro_liq = (alvo - entrada - custos) x 0.85  (apos IR)
```

### 11.5 Selecao de ativos

Filtros recomendados (em ordem):

1. `signal == "buy"` — obrigatorio
2. `potencial == "alto"` — foco nos melhores trades
3. `rank >= 70` — score combinado alto
4. `win_rate >= 60` — acerto historico consistente
5. `regime == "favoravel"` — mercado a favor
6. `lucro_liq_pct >= 8` — retorno liquido atrativo

---

## 12. Função de Fitness

O GA maximiza uma função que balanceia **retorno**, **risco** e **consistência** ao longo de 10 janelas walk-forward independentes.

### Fórmula geral

```
fitness = retorno_excess + win_rate_bonus + sharpe_component + calmar_bonus
        - mdd_penalty - tail_penalty + consistency_bonus + trade_bonus
```

### Componentes de retorno e qualidade

| Componente | Peso | Descrição |
|-----------|------|-----------|
| Retorno excedente médio (vs B&H) | ×1.8 | Superar buy-and-hold em média nas 10 janelas |
| Retorno excedente mediana | ×1.3 | Robusto a janelas outlier |
| % janelas com excess > 0 | ×1.4 | Consistência de superar B&H |
| Retorno médio absoluto | ×0.5 | Retorno bruto médio |
| Retorno mediana absoluto | ×0.3 | Retorno bruto mediano |
| Sharpe médio | ×0.5 | Risk-adjusted médio |
| Sharpe mediana | ×0.4 | Risk-adjusted mediano |
| % retornos positivos | ×0.3 | % de janelas com retorno > 0 |
| Win rate médio | ×1.1 | Taxa de acerto média |
| Win rate mediana | ×0.8 | Taxa de acerto mediana |

### Bônus de win rate (não-linear, fortemente incentivado)

| Condição | Bônus acumulado |
|----------|----------------|
| win_rate > 0.60 | **+6.0** |
| win_rate > 0.56 | **+5.0** |
| win_rate > 0.52 | **+6.0** |
| win_rate < 0.50 | **−10.0** (penalidade forte) |

### Penalidade de MDD (progressiva, em escada)

O MDD de cada ticker em cada janela é calculado, depois toma-se a mediana (cap: −60%):

| Faixa de |MDD| | Penalidade por ponto acima do limiar |
|----------|------------------------------------|
| > 8% | ×8 |
| > 15% | ×8 + ×20 (adicional) |
| > 18% | ×8 + ×20 + ×45 ← **muro principal** |
| > 25% | + ×200 ← **barreira de segurança** |
| > 30% | + ×300 ← **penalidade nuclear** |

**Bônus por baixo drawdown:**
- MDD < 20% → **+3.0**
- MDD < 15% → **+2.0** adicional (total +5.0)

### Calmar Ratio (retorno/risco)

```python
calmar = retorno_médio / abs(median_mdd)
calmar_bonus = 0.8 × clip(calmar, 0, 5)   # máximo +4.0
```

### Penalidade de cauda (tail risk)

Janelas com retorno < −20% recebem penalidade adicional progressiva (inicia em −20%, pesos dobrados).

### Bônus de consistência inter-janelas

| Condição | Efeito |
|----------|--------|
| std(win_rates) < 0.08 entre janelas | **+0.3** |
| std(returns) < 0.10 entre janelas | **+0.2** |
| std(win_rates) > 0.15 entre janelas | **−2.0** (inconsistente) |

---

## 13. Arquivos de Saída

### 13.1 `summary_latest.xlsx` (3 abas)

**Aba "Summary"** — Backtest por ticker

| Coluna | Descrição |
|--------|-----------|
| `ticker` | Código do ativo |
| `test_return` | Retorno total no período de teste |
| `test_mdd` | Maximum drawdown no teste |
| `test_sharpe` | Sharpe ratio no teste |
| `test_trades` | Número total de trades |
| `test_win_rate` | % trades com retorno > 0 |
| `test_avg_trade` | Retorno médio por trade |
| `buy_hold_return` | Retorno buy-and-hold no mesmo período |

**Aba "Apply"** — Sinais dos últimos 5 dias (ver [Seção 11](#11-saídas-no-modo-apply-planilha))

**Aba "Feature_Importance"** — Features mais relevantes por ticker

| Coluna | Descrição |
|--------|-----------|
| `ticker` | Ativo |
| `rank` | Posição no ranking (1 = mais importante) |
| `feature` | Nome da feature do ETL |
| `spearman_abs_r` | Correlação de Spearman absoluta com retorno forward |
| `mean_abs_zscore` | Z-score médio absoluto (força histórica da feature) |
| `pct_days_active_%` | % dos dias em que `|z| > z_threshold` |

### 13.2 `apply_last_5d__H5.csv`

Mesmas colunas da aba "Apply", em CSV (pode ser lido mesmo com Excel aberto na planilha).

### 13.3 `global_ga_checkpoint.json`

```json
{
    "fitness": 10.5878,
    "genome": [0.25, 0.20, 0.35, 5, 2, 0.55, 1.5, 8, 0.65, 0.07, 3.0, ...]
}
```

26 genes do melhor indivíduo (na ordem de `GLOBAL_PARAM_SPECS`). Usado para warm-start no próximo treino e para modo `"load"`.

### 13.4 Métricas impressas ao final do GA

```
============================================================
=== OBJETIVOS (FULL MODE) ===
============================================================
  win_rate_mean   : 0.554  (median: 0.603)  >> objetivo: >= 0.52
  median_mdd      : -0.236                   >> objetivo: >= -25%
  beat buy&hold % : 55.3%  (retorno medio: 5385% vs B&H: 577%)
  mean trades/tkr : 169.3
  % ret_pos       : 76.6%  | % sharpe_pos: 78.7%
  GA fitness      : 10.5878
============================================================
  win_rate_mean>=0.52 : PASS
  median_mdd>=-25%    : PASS
  beats buy&hold      : PASS
  trades>=10/ticker   : PASS
  TODOS OBJETIVOS     : ALL PASS
============================================================
```

---

## 14. FAQ

### O `signal_eod` no apply diz quando VENDER uma posição existente?

**Não diretamente.** O sinal do apply indica **entradas** (quando comprar), não saídas de posições existentes. Para gerenciar saídas:

- Use os níveis calculados **no dia que você comprou**: `stop_loss` e `take_profit`
- Esses níveis usam os mesmos parâmetros do backtest histórico
- `signal_eod = "hold"` significa apenas "sem novo sinal de compra hoje" — não é ordem de venda
- `score_100 < 30` em um ativo em posição pode sinalizar deterioração do cenário

Para o **trailing stop** (gene `trailing_stop_mode` no checkpoint):
- Modo 0: saída fixa em `stop_loss`
- Modo 1: após lucro > stop, não venda abaixo do preço máximo atingido
- Modo 2: após lucro > 2× stop, saída se recuar 50% desde o pico

### Como saber o valor atual dos parâmetros do GA?

```python
import json
with open("global_ga_checkpoint.json") as f:
    chk = json.load(f)

genes = ["vote_threshold_long","vote_threshold_short","z_threshold","signal_ema_span",
         "entry_confirmation_days","score_percentile_trigger","stop_atr_mult",
         "stop_tighten_after_bars","stop_tighten_factor","max_loss_per_trade_pct",
         "reward_risk_ratio","partial_take_pct","partial_take_level","time_stop_bars",
         "entry_discount_atr_frac","volatility_filter_percentile","score_strength_scaling",
         "ma_filter_period","ma_filter_mode","consecutive_loss_cooldown",
         "equity_drawdown_stop_pct","vol_regime_mode","partial_take_pct_2",
         "partial_take_level_2","min_signal_strength","trailing_stop_mode"]

params = dict(zip(genes, chk["genome"]))
print(f"Stop mult: {params['stop_atr_mult']}x ATR")
print(f"R/R: {params['reward_risk_ratio']}")
print(f"Trailing: modo {int(params['trailing_stop_mode'])}")
print(f"Time stop: {int(params['time_stop_bars'])} barras")
```

### Como interpretar `best_buy_value` vs `entry_ref_price` vs `stop_loss`?

```
Exemplo (VALE3.SA):
  close           = R$100.00   ← preço atual
  best_buy_value  = R$99.50    ← ordem limitada de compra (50ct abaixo do close)
  entry_ref_price = R$99.50    ← base para calcular stop e take (= best_buy se buy)
  stop_loss       = R$96.50    ← R$3.00 abaixo do entry_ref (stop_atr_mult × ATR)
  take_profit     = R$108.50   ← R$9.00 acima do entry_ref (R/R × stop)

  Risco: R$99.50 - R$96.50 = R$3.00 por ação
  Retorno alvo: R$108.50 - R$99.50 = R$9.00 por ação (R/R = 3:1)
```

### O que é `EV_buy_fund_3` e como usá-lo diretamente?

É o **Expected Value** combinado (modelos binários + regressão + fundamentais), disponível em `history_consolidated.parquet`. Valores:
- `> 0.30` → expectativa favorável de compra
- `0.10 a 0.30` → expectativa levemente positiva
- `< 0.10` → sem edge claro

Pode ser usado para **ranquear** oportunidades quando múltiplos tickers mostram `signal_eod = "buy"`.

### Por que o sistema usa LONG_ONLY?

O mercado brasileiro tem baixa liquidez para short selling na maioria das ações. Alugar ações para vender a descoberto tem custo elevado e disponibilidade limitada. O sistema foi otimizado especificamente para mercado comprado, onde as restrições operacionais são menores.

### Como o GA evita overfitting?

1. **Walk-forward obrigatório**: 10 janelas independentes — treina em 3 anos, testa nos 6 meses seguintes
2. **Abrange múltiplas crises**: janelas incluem 2008 (subprime), 2015-16 (recessão BR), 2018 (eleições), 2020 (COVID)
3. **Fitness 100% OOS**: apenas os 6 meses de teste (fora da amostra) contam para o fitness
4. **Penalidades múltiplas**: MDD, win_rate baixo e inconsistência entre janelas são penalizados severamente
5. **Diversidade controlada**: métricas de diversidade da população evitam convergência prematura
6. **Early stopping**: para quando não há mais melhoria (evita overfit ao continuar)

### O sistema funciona em tempo real (intraday)?

Não. O sistema é **diário (end-of-day)**:
- Sinal gerado no fechamento do dia i
- Ordem de compra executada no dia i+1 (limitada)
- Monitoramento de stop/take durante o pregão do dia
- Tempo de posse típico: `time_stop_bars` (5–25 dias úteis)

### Como rodar atualização diária completa?

```bash
# 1. Atualizar dados (ETL + consolidação)
python run_pipeline.py --mode data-only     # ~35-40 min

# 2. Gerar sinais (sem retreinar GA)
# ga_run.py: RUN_MODE = "load"
python ga_run.py                            # ~2 min

# 3. Verificar resultados
# Abrir summary_latest.xlsx → aba "Apply"
# Ou ler: apply_last_5d__H5.csv
```

---

## Testes automatizados (GitHub Actions)

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Verifica: existência dos arquivos principais, estrutura de notebooks, contrato I/O do GA, schema mínimo do `history_consolidated`.

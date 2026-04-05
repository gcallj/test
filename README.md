# Pipeline de Sinais de Trading — B3

Sistema de geracao de sinais de compra/venda para ~119 ativos brasileiros (acoes, FIIs, crypto). Pipeline de 5 etapas com modelos ML, otimizacao genetica (GA) walk-forward, custos realistas (corretagem + B3 + slippage + IR 15%) e **auditoria causal** que bloqueia features contaminadas.

> **Ultima atualizacao: 06/Abr/2026**
> Win Rate: **63.7%** | MDD: **-20.4%** | Confidence: **61+** | GA Fitness: **4.66**
> Features: **48 causais** (222 geradas, 48 usadas, 48 bloqueadas por auditoria)
> Custos: R$7 corretagem + B3 0.065% + slippage 0.10%/lado + IR 15%

---

## Fluxo do Pipeline

### Operacional (diario) — ETL + GA direto

```
+-----------+                                        +--------------+
|  ETL      |--------------------------------------->|  GA          |
|stock_etl  |   48 features causais (OHLCV-derived)  | ga_run.py    |
|  .py      |   27 internas + 21 ETL                 |  (load mode) |
+-----------+                                        +--------------+
  ~30 min                                              ~10 min

Output: summary_latest.xlsx + apply_last_5d__H5.csv -> Telegram
```

O GA usa **apenas features causais** derivadas de OHLCV. Steps BIN/REG/FINAL sao
desnecessarios no fluxo diario porque seus outputs (p_up20, EV_buy, buy_trust, etc.)
sao **bloqueados pela auditoria causal**.

### Pesquisa (completo) — inclui BIN/REG/FINAL

```
 STEP 1          STEP 2            STEP 3          STEP 4            STEP 5
+-----------+   +--------------+   +-----------+   +--------------+   +--------------+
|  ETL      |-->|  BIN Models  |-->| Regressao |-->| Final Output |-->|  GA          |
|stock_etl  |   |stock_bin_    |   |stock_reg_ |   |stock_final_  |   | ga_run.py    |
|  .py      |   |models.py     |   |models.py  |   |output.py     |   |              |
+-----------+   +--------------+   +-----------+   +--------------+   +--------------+
  ~30 min          ~2.5 h            ~10 min          ~5 min            ~4-8h (train)
```

Usar `python predict_daily.py --full` ou `python run_pipeline.py` para rodar completo.
BIN/REG geram probabilidades e previsoes de preco que ficam em `history_consolidated.parquet`
para analise futura, mas **nao alimentam** a operacao diaria.

---

## Auditoria Causal

O GA usa **apenas features causais** (derivadas de OHLCV). Uma blocklist automatica impede contaminacao por model outputs:

| Categoria | Qtd | Status |
|-----------|-----|--------|
| Features tecnicas (RSI, MACD, BB, CCI, etc.) | 27 | Permitidas |
| Features ETL (Minervini, SMA slopes, patterns) | 21 | Permitidas |
| Model outputs (p_up20, EV_buy, buy_trust, etc.) | 22 | **Bloqueadas** |
| Snapshot fundamentals (PE, PB, DY, market_cap) | 16 | **Bloqueadas** |
| Metadata (Date, ticker, split, signal) | 10 | Excluidas |

Script de auditoria: `analysis/leakage_audit.py`

---

## Metricas de Performance (OOS)

### Global

| Metrica | Valor |
|---------|-------|
| Win Rate (mean / median) | 63.7% / 66.3% |
| MDD (median) | -20.4% |
| Confidence (median) | 61+ |
| GA Fitness | 4.66 |
| Trades/ticker (mean) | 140 |
| % tickers com retorno positivo | 74.6% |

### Por Segmento (equal-weight)

| Segmento | Tickers | Win Rate | Retorno | MDD |
|----------|---------|----------|---------|-----|
| Industrials/Exporters | 4 | 73.2% | +558% | -18.2% |
| Crypto | 6 | 67.9% | +394% | -28.1% |
| Defensives/Health | 8 | 66.0% | +257% | -22.0% |
| Domestic Cyclicals | 14 | 67.9% | +182% | -20.9% |
| Other Equity | 8 | 67.0% | +87% | -17.3% |
| Commodity Largecap | 13 | 65.2% | +69% | -20.9% |
| Tech/Growth | 5 | 67.8% | +43% | -21.2% |
| Utilities/Infra | 13 | 64.4% | +20% | -21.2% |
| Agro | 3 | 61.1% | +18% | -21.2% |
| Financials | 9 | 64.5% | +14% | -21.8% |
| FII | 35 | 66.5% | +11% | -13.3% |

Segmentacao definida em `ticker_metadata.py`.

---

## Arquivos do Projeto

### Pipeline principal

| Arquivo | Funcao | Step |
|---------|--------|------|
| `stock_etl.py` | Download OHLCV (yfinance), ~222 features, targets | 1 |
| `stock_bin_models.py` | Ensemble classificadores (LGBM, HGB, RF, ET, XGB, CatBoost) | 2 |
| `stock_reg_models.py` | Previsoes de preco com bandas de erro | 3 |
| `stock_final_output.py` | Consolida BIN + REG, calcula Expected Value | 4 |
| `ga_run.py` | GA 30-param, walk-forward 2-stage, sinais finais | 5 |
| `run_pipeline.py` | Orquestrador Steps 1-4 (`--steps`, `--mode`, `--ga-only`) | - |
| `predict_daily.py` | Pipeline diario: ETL + BIN refresh + REG + FINAL + GA load | - |
| `numeric_utils.py` | Conversao float32 segura | lib |
| `payload_store.py` | Payloads por ticker em memmap (economia de RAM) | lib |
| `auto_tune.py` | AutoTuner: ajusta pop/ngen/workers por RAM disponivel | lib |
| `ga_run_modular_final.py` | GA two-stage modular (S1 exploracao + S2 refinamento) | lib |
| `ticker_metadata.py` | Mapa setorial: 80+ tickers B3 em 11 segmentos | lib |

### Analise e auditoria

| Arquivo | Funcao |
|---------|--------|
| `analysis/leakage_audit.py` | Scanner de contaminacao: model outputs, snapshot fundamentals, split |

### Ferramentas opcionais

| Arquivo | Funcao |
|---------|--------|
| `run_local.py` | Runner local com metricas e envio Telegram |
| `run_stages.py` | GA em stages como subprocessos |
| `chart_petr4.py` / `chart_top_signals.py` | Graficos de sinais |
| `compare_v3_v4.py` | Compara versoes do pipeline |
| `retrain_codespace.py` | Retreino em GitHub Codespaces |
| `main_cell_v3.py` | Pipeline GA standalone (legado, importado por payload_store) |

### Testes

| Arquivo | Funcao |
|---------|--------|
| `tests/test_notebooks.py` | Contratos I/O: ga_run.py, notebooks, checkpoint format |

---

## Como Executar

### Pipeline completa (treina tudo)

```bash
# Steps 1-4 (ETL + ML models + consolidacao) — ~3-4 horas
python run_pipeline.py

# Step 5 (GA train — otimizacao genetica) — ~4-8 horas
python ga_run.py
```

### Atualizacao diaria (~40 min)

```bash
python predict_daily.py                   # ETL + GA load (default, rapido)
# Resultado: summary_latest.xlsx enviado ao Telegram
```

### Pipeline de pesquisa (~3.5h, inclui BIN/REG)

```bash
python predict_daily.py --full            # ETL + BIN + REG + FINAL + GA
python run_pipeline.py                    # Alternativa: Steps 1-4 + GA
python run_pipeline.py --mode full        # Treina tudo
python run_pipeline.py --steps 1,4        # Steps especificos
```

---

## Saidas

### `summary_latest.xlsx`

| Aba | Conteudo |
|-----|---------|
| **Apply** | Sinais do dia: ticker, signal, confidence, entrada (alinhada com S1), stop, alvo (R1), R:R, condicao unificada |
| **Summary** | Metricas OOS por ticker: test_return, test_mdd, test_sharpe, test_win_rate, segment_key |
| **Feature_Importance** | Top features por ticker com relevance_score |

### Formato da aba Apply

| Campo | Descricao |
|-------|-----------|
| `signal` | buy / hold / sell |
| `entrada` | Preco de compra (alinhado com S1 - suporte) |
| `stop` | Stop-loss (abaixo de S1) |
| `alvo` | Take-profit (proximo de R1 - resistencia) |
| `pivot`, `r1`, `s1` | Niveis de pivot diario |
| `condicao` | Campo unificado: `BUY 66.74 \| Acima Pivot \| Regime OK \| Stop 62.66 Alvo 75.11` |
| `confidence` | Score 0-100 (10 fatores: backtest, WR, timing, pivot, regime, R:R, liquidez) |
| `regime` | favoravel / neutro / desfavoravel |

---

## Custos Modelados

```
Corretagem:  R$7.00 por ordem (compra + venda = R$14 round-trip)
B3 fees:     0.065% round-trip (emolumentos + liquidacao)
Slippage:    0.10% por lado (estimado)
IR:          15% sobre lucro liquido (swing trade)
Total:       ~0.55% por trade (sem IR)
```

---

## GA: Algoritmo Genetico em 2 Estagios

```
Stage 1 (Exploracao)                    Stage 2 (Refinamento)
  pop=64, ngen=25                         pop=24, ngen=20
  4 walk-forward windows                  8 walk-forward windows
  Varre espaco amplo                      Valida robustez OOS
  Top 6 genomas transferidos  -------->   Refina com mais periodos
```

- **30 parametros** otimizados: stops, take-profits, trailing, partial takes, volatility filter, timing
- **Fitness multi-objetivo**: retorno excess, Sharpe, MDD (softplus), win rate, consistencia
- **Warm-start**: checkpoint permite retomar de run anterior
- **AutoTuner**: ajusta pop/ngen/workers automaticamente por RAM disponivel

---

## Telegram

O `ga_run.py` envia `summary_latest.xlsx` via Telegram automaticamente.

Configurar:
```bash
export TELEGRAM_BOT_TOKEN="seu_token"
export TELEGRAM_CHAT_ID="seu_chat_id"
```

Fallback hardcoded no codigo para uso sem env vars.

---

## GitHub Actions

Workflow: `.github/workflows/daily_pipeline.yml`

- **Schedule:** Dias uteis, 18:30 BRT (21:30 UTC)
- **Modo:** `predict_daily.py` (ETL + GA load — BIN/REG pulados)
- **Outputs:** summary_latest.xlsx -> Telegram + GitHub Artifacts (30 dias)
- **Retrain:** `.github/workflows/retrain_ga.yml` (manual, staged)

---

## Requisitos

- **Python 3.10+**
- Core: `pandas`, `numpy`, `scikit-learn`, `deap`, `numba`, `scipy`, `xlsxwriter`, `pyarrow`
- ML: `lightgbm`, `xgboost`, `catboost`
- Dados: `yfinance`, `ta`
- Opcional: `optuna`

```bash
pip install pandas numpy scikit-learn deap numba scipy xlsxwriter pyarrow lightgbm yfinance ta
```

---

## Estrutura

```
test/
|-- ga_run.py                    # GA optimization (pipeline principal)
|-- run_pipeline.py              # Orquestrador Steps 1-4
|-- predict_daily.py             # Pipeline diario
|-- stock_etl.py                 # Step 1: ETL
|-- stock_bin_models.py          # Step 2: Binary classification
|-- stock_reg_models.py          # Step 3: Regression
|-- stock_final_output.py        # Step 4: Final consolidation
|-- numeric_utils.py             # Utilidades numericas
|-- payload_store.py             # Memmap payload store
|-- auto_tune.py                 # AutoTuner RAM-aware
|-- ga_run_modular_final.py      # GA two-stage
|-- ticker_metadata.py           # Mapa setorial B3
|-- analysis/
|   |-- leakage_audit.py         # Scanner de contaminacao
|-- tests/
|   |-- test_notebooks.py        # Testes de contrato I/O
|-- .github/workflows/
|   |-- daily_pipeline.yml       # Diario 18:30 BRT
|   |-- retrain_ga.yml           # Retrain manual staged
|   |-- test.yml                 # CI tests
|-- global_ga_checkpoint.json    # Checkpoint GA
|-- summary_latest.xlsx          # Output principal
|-- apply_last_5d__H5.csv        # Sinais em CSV
|-- output/
|   |-- data/
|   |   |-- expanded_stock_reduced.parquet
|   |   |-- models/
|   |-- history_consolidated.parquet
|   |-- ga_memmap/               # Payloads memmap
|-- .env                         # Telegram + API keys (nao commitado)
```

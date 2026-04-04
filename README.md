# Pipeline de Sinais de Trading — B3

Sistema completo de geracao de sinais de compra/venda para ~119 ativos do mercado brasileiro. Combina modelos ML (classificacao + regressao), consolidacao de Expected Value e otimizacao genetica (GA) com walk-forward backtesting e custos realistas (corretagem + B3 + slippage + IR).

> **Ultima atualizacao: Abril/2026**
> Win Rate: **61.7%** | R:R: **3.5** | MDD: **-20.4%** | Confidence: **61.3** | GA Fitness: **3.00**
> Custos: R$7 corretagem + B3 0.065% + slippage 0.10%/lado + IR 15%

---

## Fluxo do Pipeline

```
 STEP 1          STEP 2            STEP 3          STEP 4            STEP 5
┌───────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌──────────────┐
│  ETL      │──>│  BIN Models  │──>│ Regressao │──>│ Final Output │──>│  GA          │
│           │   │              │   │           │   │              │   │              │
│stock_etl  │   │stock_bin_    │   │stock_reg_ │   │stock_final_  │   │ ga_run.py    │
│  .py      │   │models.py     │   │models.py  │   │output.py     │   │              │
└───────────┘   └──────────────┘   └───────────┘   └──────────────┘   └──────────────┘
  ~30 min          ~2.5 h            ~10 min          ~5 min            ~4-8h (train)
                                                                        ~10 min (load)
```

**Dados entre steps:**

```
expanded_stock_reduced.parquet  ──>  ensemble_signals_history.parquet
                                         ──>  forecast_history_wide.parquet
                                                   ──>  history_consolidated.parquet
                                                              ──>  summary_latest.xlsx
                                                                   apply_last_5d__H5.csv
                                                                   global_ga_checkpoint.json
```

---

## Arquivos do Projeto

### Pipeline principal (9 arquivos)

| Arquivo | Funcao | Step |
|---------|--------|------|
| `stock_etl.py` | Download OHLCV (yfinance), ~222 features tecnicas/microestruturais, targets | 1 |
| `stock_bin_models.py` | Ensemble 4-6 classificadores (LGBM, HGB, RF, ET, XGB, CatBoost) | 2 |
| `stock_reg_models.py` | Previsoes de preco com bandas de erro (35% ML + 65% baseline) | 3 |
| `stock_final_output.py` | Consolida BIN + REG + fundamentais, calcula Expected Value | 4 |
| `ga_run.py` | Otimizacao genetica (30 parametros), walk-forward, sinais finais | 5 |
| `run_pipeline.py` | Orquestrador dos Steps 1-4 | - |
| `predict_daily.py` | Atualizacao diaria (GitHub Actions, sem retreinar modelos) | - |
| `numeric_utils.py` | Conversao float32 segura, normalizacao | lib |
| `payload_store.py` | Armazena payloads por ticker em memmap (economia de RAM) | lib |

### Dependencias do GA (3 arquivos)

| Arquivo | Funcao |
|---------|--------|
| `auto_tune.py` | AutoTuner + RuntimeGuard para o GA |
| `ga_run_modular_final.py` | Funcao `run_global_ga_two_stage` (GA em 2 estagios) |
| `main_cell_v3.py` | Pipeline alternativo GA standalone (importado por payload_store) |

### Ferramentas opcionais (5 arquivos)

| Arquivo | Funcao |
|---------|--------|
| `run_local.py` | Runner local com metricas e envio Telegram |
| `chart_petr4.py` | Graficos de candlestick para PETR4 |
| `chart_top_signals.py` | Graficos dos top sinais do dia |
| `compare_v3_v4.py` | Compara versoes do pipeline |
| `run_stages.py` | Roda steps individuais do GA |
| `retrain_codespace.py` | Retreino em GitHub Codespaces |
| `workflow_orchestrator.py` | Orquestrador multi-AI (Claude + Codex) |

### Testes

| Arquivo | Funcao |
|---------|--------|
| `tests/test_notebooks.py` | Valida contratos de I/O dos notebooks |

---

## Como Executar

### Pipeline completa (treina tudo — ~6-12 horas)

```bash
# Steps 1-4 (ETL + ML models + consolidacao)
python run_pipeline.py                    # ~3-4 horas

# Step 5 (GA — otimizacao genetica)
python ga_run.py                          # ~4-8 horas (RUN_MODE=train)
```

### Atualizacao diaria (sem retreinar — ~40 min)

```bash
# 1) Atualizar dados de mercado (reutiliza modelos ML salvos)
python run_pipeline.py --mode data-only   # ~35-40 min

# 2) Gerar sinais com GA salvo
GA_RUN_MODE=load python ga_run.py         # ~10 min

# 3) Abrir summary_latest.xlsx → aba "Apply"
```

### Steps especificos

```bash
python run_pipeline.py --steps 1          # So ETL
python run_pipeline.py --steps 2,3,4      # Pular ETL
python run_pipeline.py --steps 1,4        # ETL + Final (= data-only)
```

### Pre-requisitos para modo "load"

Todos gerados por pelo menos uma execucao completa:
- `./output/data/expanded_stock_reduced.parquet` (ETL)
- `./output/ensemble_signals_history.parquet` (BIN)
- `./output/forecast_history_wide.parquet` (REG)
- `./output/history_consolidated.parquet` (FINAL)
- `./global_ga_checkpoint.json` (GA)

---

## Requisitos

- **Python 3.10** (scipy ABI incompativel com 3.12+)
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `deap`, `numba`, `scipy`, `xlsxwriter`, `pyarrow`
- ML: `lightgbm`, `xgboost`, `catboost` (opcionais para ensemble completo)
- Dados: `yfinance`, `ta` (technical analysis)
- Opcionais: `optuna` (hyperparameter tuning no Step 2)

```bash
# Caminho recomendado no Windows:
"C:\Users\gabri\AppData\Local\Programs\Python\Python310\python.exe"

# Instalar dependencias:
pip install pandas numpy scikit-learn deap numba scipy xlsxwriter pyarrow lightgbm yfinance ta
```

---

## Saidas Principais

### `summary_latest.xlsx`

| Aba | Conteudo |
|-----|---------|
| **Apply** | Sinais dos ultimos 5 dias: ticker, signal (buy/sell/hold), confidence, R:R, entrada, stop, alvo |
| **Summary** | Metricas OOS por ticker: test_return, test_mdd, test_sharpe, test_win_rate, test_trades |
| **Feature_Importance** | Top features por ticker com relevance_score |

### `apply_last_5d__H5.csv`

Mesmos dados da aba Apply em CSV para integracao com outros sistemas.

### `global_ga_checkpoint.json`

Genoma otimizado (30 parametros) + fitness. Usado no modo `load` para gerar sinais sem retreinar.

---

## Metricas de Performance (OOS)

| Metrica | Valor | Descricao |
|---------|-------|-----------|
| Win Rate (mean) | 61.7% | % de trades lucrativos |
| Win Rate (median) | 62.8% | Mediana por ticker |
| R:R (median) | 3.5:1 | Risk-Reward ratio |
| MDD (median) | -20.4% | Maximum Drawdown |
| Test Return (mean) | +57.4% | Retorno OOS medio |
| Confidence (median) | 61.3 | Score de confianca (0-100) |
| GA Fitness | 3.00 | Fitness multi-objetivo |
| Trades/ticker (mean) | 67 | Numero medio de trades |

---

## Custos Modelados

```
Corretagem:  R$7.00 por ordem (compra + venda = R$14 round-trip)
B3 fees:     0.065% round-trip (emolumentos + liquidacao)
Slippage:    0.10% por lado (estimado)
IR:          15% sobre lucro liquido (swing trade)
────────────────────────────────────
Total:       ~0.55% por trade (sem IR)
```

---

## Envio via Telegram

O `ga_run.py` envia automaticamente o `summary_latest.xlsx` via Telegram ao final da execucao.

Configurar variaveis de ambiente:
```bash
export TELEGRAM_BOT_TOKEN="seu_token"
export TELEGRAM_CHAT_ID="seu_chat_id"
```

Ou via GitHub Secrets no workflow `daily_pipeline.yml`.

---

## GitHub Actions (Execucao Automatica)

Workflow: `.github/workflows/daily_pipeline.yml`

- **Schedule:** Dias uteis, 18:30 BRT (21:30 UTC)
- **Modo:** `predict_daily.py` (ETL + forward-fill + FINAL + GA load)
- **Outputs:** summary_latest.xlsx → Telegram + GitHub Artifacts (7 dias)

---

## Estrutura de Diretorios

```
test/
├── ga_run.py                    # Step 5: GA optimization (pipeline principal)
├── run_pipeline.py              # Orquestrador Steps 1-4
├── predict_daily.py             # Atualizacao diaria (GitHub Actions)
├── stock_etl.py                 # Step 1: ETL
├── stock_bin_models.py          # Step 2: Binary classification
├── stock_reg_models.py          # Step 3: Regression
├── stock_final_output.py        # Step 4: Final consolidation
├── numeric_utils.py             # Utilidades numericas
├── payload_store.py             # Store de payloads por ticker
├── auto_tune.py                 # AutoTuner para GA
├── ga_run_modular_final.py      # GA two-stage modular
├── main_cell_v3.py              # Pipeline GA alternativo
├── run_local.py                 # Runner local com metricas
├── chart_petr4.py               # Graficos PETR4
├── chart_top_signals.py         # Graficos top sinais
├── compare_v3_v4.py             # Comparacao de versoes
├── run_stages.py                # Runner de stages individuais
├── retrain_codespace.py         # Retreino em Codespaces
├── workflow_orchestrator.py     # Orquestrador multi-AI
├── requirements.txt             # Dependencias Python
├── global_ga_checkpoint.json    # Checkpoint GA (genoma + fitness)
├── summary_latest.xlsx          # Excel com sinais (output principal)
├── apply_last_5d__H5.csv        # CSV com sinais
├── tests/
│   └── test_notebooks.py        # Testes de contrato I/O
├── output/
│   ├── data/
│   │   ├── expanded_stock_reduced.parquet
│   │   └── models/              # Modelos sklearn (.joblib)
│   ├── ensemble_signals_history.parquet
│   ├── forecast_history_wide.parquet
│   └── history_consolidated.parquet
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml   # GitHub Actions scheduler
└── .env                         # API keys (nao commitado)
```

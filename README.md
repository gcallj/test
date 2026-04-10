# Pipeline de Sinais de Trading — B3

Sistema de geracao de sinais de compra/venda para ~119 ativos brasileiros (acoes, FIIs, crypto). Pipeline com otimizacao genetica (GA) walk-forward, custos realistas (corretagem + B3 + slippage + IR 15%) e **auditoria causal** que bloqueia features contaminadas.

> **Ultima atualizacao: 10/Abr/2026**
> Win Rate: **64.2%** | WR Alvo: **61.1%** | MDD: **-13.0%** | Fitness: **18.45**
> Features: **80 causais** (OHLCV-derived + ETL)
> Custos: R$7 corretagem + B3 0.065% + slippage 0.10%/lado + IR 15%

---

## Metricas de Performance

### Global (full history, 119 tickers)

| Metrica | Valor |
|---------|-------|
| Win Rate (median) | 64.2% |
| Win Rate Alvo (% trades que atingem take-profit) | 61.1% |
| MDD (median) | -13.0% |
| GA Fitness | 18.45 |
| Trades/ticker (median) | 61 |

### Por Universe Quality

O modelo tem **alpha real** concentrado em 7 tickers premium:

| Universe | N | WR | Ret a.a. | Alpha | Beat B&H | MDD |
|----------|---|-----|----------|-------|----------|-----|
| **PREMIUM** | 7 | 72.2% | +3.4% | **+5.5pp** | **100%** | 13.6% |
| HIGH_QUAL | 15 | 70.3% | +1.6% | +5.0pp | 100% | 12.9% |
| ALL | 119 | 64.2% | - | -11.5pp | 19% | 13.0% |

**Recomendacao: priorizar sinais BUY em tickers PREMIUM e HIGH_QUAL.**

---

## Fluxo Operacional (diario ~30 min)

```
+-----------+         +--------------+         +-----------+
|  ETL      |-------->|  GA          |-------->| Telegram  |
|stock_etl  |  80 feat| ga_run.py    | xlsx    | summary_  |
|  .py      |  causais|  (load mode) |         | latest    |
+-----------+         +--------------+         +-----------+
```

```bash
python predict_daily.py     # ETL + GA load → summary_latest.xlsx → Telegram
```

---

## Sinais e Niveis Operacionais

### Classificacao Minervini Stage

| Stage | Sub-timing | Preco vs MA50 | Acao |
|-------|-----------|---------------|------|
| **Stage 2 Alta - ideal** | Close a 0-3% da MA50 | Colado a MA50 (melhor entrada) | **COMPRAR** |
| **Stage 2 Alta - momentum** | Close a 3-15% da MA50 | Tendencia confirmada | **COMPRAR** |
| Stage 2 Alta - atrasado | Close > 15% da MA50 | Esticado | AGUARDAR pullback |
| Stage 2 Alta - pullback | Close < MA50 > MA200 | Correcao saudavel | COMPRAR cautela |
| Stage 2 Alta - cedo | MA50 cruzou MA200 < 3% | Transicao recente | OBSERVAR |
| Stage 1 Base | MA50 plana | Lateralidade | OBSERVAR |
| Stage 3 Topo | MA50 caindo | Distribuicao | AGUARDAR / VENDER |
| Stage 4 Queda | Close < MA200 | Downtrend | VENDER |

### Niveis de Entrada/Stop/Alvo (pivot-based)

| Nivel | Calculo | Logica |
|-------|---------|--------|
| **Entrada** | Pivot (pullback) ou S1+0.2% | Comprar em suporte |
| **Stop** | S1-0.5% (quebra suporte) | ATR fallback se S1 muito tight |
| **Alvo** | R1-0.2% (resistencia) | ATR fallback se R1 muito perto |
| **R:R** | Informativo (sem forcagem) | Resultado natural dos levels |

### Quality Gates (BUY → HOLD)

| Gate | Criterio | Motivo |
|------|----------|--------|
| WR < 60% | Backtest win rate insuficiente | Nao compensa custos |
| Potencial < 5% | Alvo muito proximo | Custos reais 0.55%+IR consomem lucro |
| Confidence < MIN | GA + stage + pivot baixos | Sinal fraco |

### Promocao hold → buy

GA disse HOLD mas Stage 2 ideal/momentum indica oportunidade? Promove SE:
- confidence >= 75 (antes era 65 — era permissivo demais)
- score_strength > 0.2
- regime == "favoravel"

---

## Colunas do Apply (xlsx)

| Campo | Descricao |
|-------|-----------|
| `signal` | buy / hold / sell (final, pos-coerce + gates) |
| `signal_ga` | Sinal bruto do GA (antes dos overrides) |
| `recomendacao` | Texto unificado: [PREMIUM] MODO * Stage * potencial alvo (WR X% / alvo Y%) |
| `entrada` | Preco de compra (pivot/S1 based) |
| `stop` | Stop-loss (abaixo de S1) |
| `alvo` | Take-profit (proximo de R1) |
| `win_rate` | % trades positivos (inclui trailing stops) |
| `wr_alvo` | % trades que atingiram o alvo take-profit (estrito) |
| `universe` | PREMIUM / HIGH_QUAL / REGULAR (baseado em alpha historico) |
| `confidence` | 0-100 (11 fatores: WR, WR_target, timing, pivot, regime, R:R, etc.) |
| `rank` | Score ponderado (WR 20% + WR_target 15% + confidence + timing + regime) |

---

## Arquivos Principais

| Arquivo | Funcao |
|---------|--------|
| `predict_daily.py` | Pipeline diario: ETL + GA load → Telegram |
| `ga_run.py` | GA 30-param, walk-forward 2-stage, sinais finais, Telegram |
| `stock_etl.py` | Download OHLCV (yfinance), features tecnicas |
| `send_telegram.py` | Caption enriquecido: WR breakdown + subsets Premium/HQ |
| `run_local_ga_staged.py` | Retrain seeded (chunks × gens × pop × windows) |
| `ga_run_modular_final.py` | GA two-stage modular (exploração + refinamento) |
| `payload_store.py` | Payloads por ticker em memmap (economia de RAM) |
| `ticker_metadata.py` | Mapa setorial: 119 tickers B3 em 11 segmentos |
| `global_ga_checkpoint.json` | Checkpoint GA (30 genes, fitness) |

### Analise (analysis/)

| Arquivo | Funcao |
|---------|--------|
| `ticker_alpha_subset.py` | Identifica tickers PREMIUM/HIGH_QUAL por alpha historico |
| `win_rate_deep_dive.py` | Compara WR total vs WR alvo (exit type tracking) |
| `sweep_chunk3_variants.py` | Parameter sweep post-hoc (rr, trailing, partial) |
| `sweep_universe_subset.py` | Avalia modelo em subsets do universo |
| `validate_stage_filter.py` | Valida Minervini Stage gate no backtest |

---

## Como Executar

```bash
# Diario (~30 min)
python predict_daily.py

# Retrain seeded (usa checkpoint atual como semente)
python retrain_with_wr_target.py    # 3 chunks × 3 gens × 24 pop

# Pipeline completa (treina tudo, ~4-8h)
python ga_run.py
```

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

## Telegram

Caption enriquecido com:
- Sinais: X BUY | Y HOLD | Z SELL
- WR breakdown: WR total vs WR alvo + exit types (alvo/stop)
- Subsets: Premium (alpha>+2pp) e High-Quality (alpha>=0)
- Top BUYs por rank com entrada/alvo/potencial

Configurar:
```bash
export TELEGRAM_BOT_TOKEN="seu_token"
export TELEGRAM_CHAT_ID="seu_chat_id"
```

---

## GitHub Actions

Workflow: `.github/workflows/daily_pipeline.yml`
- **Schedule:** Diariamente 18:30 BRT (21:30 UTC) + fins de semana
- **Modo:** `predict_daily.py` (ETL + GA load)
- **Output:** summary_latest.xlsx → Telegram + GitHub Artifacts

---

## Requisitos

- **Python 3.10+**
- Core: `pandas`, `numpy`, `scikit-learn`, `deap`, `numba`, `scipy`, `xlsxwriter`, `pyarrow`
- ML: `lightgbm`, `xgboost`, `catboost`
- Dados: `yfinance`, `ta`

```bash
pip install pandas numpy scikit-learn deap numba scipy xlsxwriter pyarrow lightgbm yfinance ta
```

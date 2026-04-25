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
| `ticker_metadata.py` | Mapa setorial: 48 tickers OMX Stockholm em 8 segmentos (branch SE) |
| `global_ga_checkpoint.json` | Checkpoint GA (30 genes, fitness) |

### Analise (analysis/)

| Arquivo | Funcao |
|---------|--------|
| `ticker_alpha_subset.py` | Identifica tickers PREMIUM/HIGH_QUAL por alpha historico |
| `win_rate_deep_dive.py` | Compara WR total vs WR alvo (exit type tracking) |
| `sweep_chunk3_variants.py` | Parameter sweep post-hoc (rr, trailing, partial) |
| `sweep_universe_subset.py` | Avalia modelo em subsets do universo |
| `validate_stage_filter.py` | Valida Minervini Stage gate no backtest |
| `sweep_learnings.py` | **Agregador**: le todos os sweeps + state + checkpoint, produz `sweep_learnings_summary.md` com graveyard + hot zones + sugestao do proximo sweep |
| `optimization_log.md` | **Log estruturado** anexado por `run_continuous_improvement.py` a cada ciclo (sucessos E falhas) |
| `optimization_attempts_20260416.md` | Log humano dos 11 attempts do codex (Apr 16-18) |

### Rotina de melhoria continua (toolkit)

| Arquivo | Funcao |
|---------|--------|
| `run_continuous_improvement.py` | **Orquestrador**: rotaciona 6 categorias de genes, chama sweep + apply, anexa log markdown |
| `run_local_fullmetric_sweep.py` | Sweep +/- step na vizinhanca de N genes com acceptance gates |
| `apply_fullmetric_sweep_best.py` | Aplica candidato promovivel ao checkpoint (com backup automatico) |
| `run_local_ga_staged.py` | Staged GA com acceptance gates compartilhados (eval-only ou retrain) |
| `overnight_alpha_until_0600.py` | Runner contínuo ate hora limite (alternativa ao continuous) |
| `continuous_improvement_state.json` | State de rotacao (quando cada categoria foi swept) |

---

## Como Executar

### Diario (sinais + Telegram)
```bash
python predict_daily.py     # ~30 min: ETL + GA load → summary_latest.xlsx → Telegram
```

### Rotina de melhoria continua (manual no Claude)

A rotina chama `run_local_fullmetric_sweep.py` em vizinhanca de 1 categoria de
genes, aplica o candidato se passar nos guardrails, e anexa entry estruturada
ao log markdown.

```bash
# Status: quais categorias ja foram exploradas e quando
python run_continuous_improvement.py --status

# 1 ciclo automatico (escolhe categoria menos-recentemente-explorada)
python run_continuous_improvement.py

# Categoria especifica (A=risk, B=take_profit, C=timing,
# D=entry_filter, E=regime_vol, F=trailing)
python run_continuous_improvement.py --category D

# N ciclos consecutivos
python run_continuous_improvement.py --max-cycles 6

# Loop ate hora limite (overnight)
python run_continuous_improvement.py --until 06:00

# Atualizar agregador apos rodar ciclos (le todos os sweeps + state)
python analysis/sweep_learnings.py
```

Cada ciclo:
1. Le `continuous_improvement_state.json` para escolher proxima categoria
2. Roda `run_local_fullmetric_sweep.py` na vizinhanca daquela categoria (~30-60 min)
3. Se passar acceptance gates (vs git:main + vs incumbent), promove via
   `apply_fullmetric_sweep_best.py` (faz backup automatico)
4. Anexa entry ao `analysis/optimization_log.md` (sucesso OU falha — falhas
   formam o graveyard que ensina o que NAO funciona)
5. Atualiza state file

**Maximizar periodicidade**: encadeie ciclos com `--max-cycles 6` ou
`--until 06:00`. Cada categoria leva ~30-60 min; 6 categorias completam um
"giro" em 3-6 horas.

### Rotina automatica (Claude Code remote trigger — RECOMENDADO)
- Trigger: `trig_01CsyZBNYWPvAzu7CxRfGrpi` (GA continuous improvement)
- Periodicidade: **a cada 2 horas** (12x por dia, intervalo minimo da API)
- Ambiente: cloud isolada da Anthropic (CCR), nao depende da maquina local
- Acao: 1 ciclo de `run_continuous_improvement.py` + push automatico se promover
- UI: https://claude.ai/code/scheduled/trig_01CsyZBNYWPvAzu7CxRfGrpi

### Rotina automatica (GitHub Actions — backup)
- `.github/workflows/weekly_sweep.yml` — sabado 22:00 UTC, 1 ciclo, auto-commit
- `.github/workflows/daily_pipeline.yml` — todos os dias 21:30 UTC, predict_daily

### Retrain pesado (raro)
```bash
python retrain_with_wr_target.py    # 3 chunks × 3 gens × 24 pop (~1-2h)
python ga_run.py                    # Treina do zero (~4-8h)
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

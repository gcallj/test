# Guia — rodar o sistema sueco por conta própria

Este guia explica como o sistema decide, como instalar no Windows, como ver as
simulações e como operar no dia a dia. As decisões de projeto estão em
`docs/sweden/plano.md` e `docs/adr/`; o vocabulário em `CONTEXT.md`.

## 0. Estratégia atual: v2, rotação mensal por momentum

A v2 (`rotation_se.py`, ADR 0005) substituiu o swing trade das seções 1, 3 e
4, que ficam como histórico. Ela foi avaliada fora da amostra e **reprovada no
critério de retorno** (+35,7% contra +63,6% do índice em 3 anos), com drawdown
menor que o do índice — detalhes em `docs/sweden/plano.md`.

Depois dela, foram testadas e reprovadas:
- a v3 (`selective_se.py`, seleção semanal por previsão);
- a v4 (`events_se.py`, eventos de insiders, posições vendidas, tom dos
  comunicados e balanços, com dados de `altdata_se.py`; fontes em
  `docs/sweden/dados_alternativos.md`);
- a v5 (`revisions_se.py`, revisões de analistas do Börsvärlden coletadas por
  `analyst_se.py`);
- a v6 (`swing_se.py`, swing derivado da v1 numa conta única, com CMA-ES e as
  entradas da v4/v5).

A v2 continua sendo a referência. Para atualizar os dados alternativos e
refazer as contagens sem olhar retornos:

```bash
python altdata_se.py insiders && python altdata_se.py shorts && python altdata_se.py news && python altdata_se.py tone
```

```bash
python events_se.py dry-run
```

`python events_se.py evaluate` é a rodada registrada (ADR 0008); não refazer
para escolher variações.

### Como ela decide (no último pregão de cada mês, depois do fechamento)

1. **Quem pode entrar**: ações do universo com ~1 ano de histórico e giro
   mediano de 60 dias ≥ 10 MSEK.
2. **Ranking**: retorno dos últimos 12 meses, ignorando o último mês
   (momentum 12-1).
3. **Carteira**: as 6 melhores, peso igual. Uma ação já na carteira só sai
   quando cai para fora das 20 melhores — isso evita girar por pouca diferença.
4. **Exposição**: se a volatilidade de 63 dias da cesta passar de 20% ao ano,
   a parte em ações é reduzida na proporção (20% ÷ vol) e o resto fica em
   caixa. Posições mantidas só são ajustadas se se afastarem mais de 25% do
   tamanho-alvo.
5. **Execução**: ordens a mercado na abertura do pregão seguinte (custos:
   0,25% de courtage + 0,10% de slippage por ordem).

Preços ajustados por dividendos; vagas sem ação iriam para um fundo de índice.

### Comandos

```bash
python prices_se.py download                 # preços ajustados (~2 min)
python rotation_se.py run --start 2014-01-01 --end 2023-10-30   # simular um período
python rotation_se.py evaluate               # a avaliação registrada (não refazer: ADR 0006)
python rotation_se.py orders --start 2026-10-30 --capital 30000 # carteira e ordens do mês
```

`run` aceita `--spec '{"k": 8, "vol_target": 0.15}'` para estudar variações —
em períodos de desenvolvimento ou em dados novos, nunca para escolher pelo
resultado nos 3 anos já avaliados.

`reports/se_v2/rebalances.csv` mostra, mês a mês, o regime, a exposição, as
ações mantidas, compradas e vendidas e o ranking das 20 melhores — é o
"porquê" de cada decisão.

## 1. O que o sistema faz, em uma página (v1, histórico)

```
Yahoo Finance ──► ETL ──► payloads ──► motor ──► ordens do dia
 (preços)      (features   (z-scores   (regra única: backtest,
                causais)    móveis)     replay e operação diária)
```

1. **ETL** (`stock_etl.py`): baixa preços dos 120 tickers do universo
   (`data/universe_se.json`) e de índices/câmbio de contexto, e calcula
   features técnicas. Toda feature é deslocada um pregão: no dia *d* o
   sistema só vê dados até *d−1*.
2. **Payloads** (`replay_se.py store`): para cada ticker, transforma as
   features em z-scores móveis (comparados com o próprio histórico recente).
3. **Motor** (`ga_run_modular_final._backtest_stats_global_intraday`): a única
   implementação da regra de operação. O mesmo código treina o GA, roda os
   replays e gera as ordens diárias (ADRs 0001 e 0004).
4. **Genoma**: os 38 parâmetros que ajustam o motor (limiares, stops,
   saídas). O GA escolhe o genoma; você aceita um genoma para uso diário
   depois de ler o replay (ADR 0002).

### Como o motor decide comprar (no fechamento do dia *d*, para *d+1*)

Os filtros são verificados nesta ordem; o primeiro que falhar é o "motivo"
que aparece no replay.

| # | Filtro | Gene(s) | Motivo no replay |
|---|---|---|---|
| 1 | sem posição, sem ordem ativa, fora de pausa | `consecutive_loss_cooldown`, `equity_drawdown_stop_pct` | já comprado / ordem ativa / pausa |
| 2 | indicadores aquecidos (≈1 ano de histórico) | — | histórico insuficiente |
| 3 | volatilidade nem baixa nem alta demais | `volatility_filter_percentile`, `vol_regime_mode` | volatilidade baixa/alta demais |
| 4 | força do sinal acima dos limiares | `min_signal_strength`, `entry_score_threshold` | sinal fraco / abaixo do limiar |
| 5 | volume acima da média (se ligado) | `volume_confirm_mode` | volume abaixo da média |
| 6 | retorno dos últimos dias ≥ 0 | `momentum_confirm_days` | momento negativo |
| 7 | regime de mercado favorável | `regime_threshold` | regime desfavorável |
| 8 | indicadores votam compra **e** score acima do gatilho | `vote_threshold_long`, `z_threshold`, `score_percentile_trigger`, `signal_ema_span` | indicadores não votaram compra |
| 9 | sinal repetido N dias seguidos | `entry_confirmation_days` | aguardando confirmação |
| 10 | preço acima da média longa (se ligado) | `ma_filter_mode`, `ma_filter_period` | abaixo da média longa |
| 11 | pivô, resistência R1 e suporte S1 | `entry_aggressiveness`, `resistance_overext_gate`, `support_broken_gate` | abaixo do pivô / esticado / suporte rompido |

**Força do sinal** = |score| ÷ percentil 95 do |score| no último ano (0 a 1).

### A ordem

- **Limite** = fechamento − desconto × ATR. O desconto cresce com
  `entry_discount_atr_frac`, diminui com a força do sinal
  (`score_strength_scaling`) e se ajusta à distância até a mínima dos últimos
  10 pregões.
- **Stop** = limite − `stop_atr_mult` × ATR · **Alvo** = limite + `reward_risk_ratio` × distância do stop.
- **Validade**: 3 pregões. Executa na abertura se o mercado abrir abaixo do
  limite, ou no limite se o pregão tocar nele. Se expirar, é uma *missed
  entry*: **não compre a outro preço** — o replay não conta esse trade, e a
  operação real também não deve.
- **Cancelamento abaixo do stop**: a regra existe, mas na prática não dispara
  — para fechar abaixo do stop o preço precisa antes passar pelo limite (que
  fica acima do stop), e aí a ordem já executou. Por isso os replays mostram
  0 ordens canceladas.

### Com a posição aberta (níveis recalculados a cada fechamento)

- **Stop dinâmico**: aperta depois de `stop_tighten_after_bars` pregões
  (`stop_tighten_factor`) ou se estiver no prejuízo após 3 pregões.
- **Saídas parciais**: vende `partial_take_pct` ao atingir
  `partial_take_level` × distância do stop; uma segunda parcial com os genes `_2`.
- **Stop móvel** (`trailing_stop_mode`): 1 = vai para o zero-a-zero depois de
  ganhar uma distância de stop; 2 = além disso, protege 50% do maior ganho.
- **Saída por tempo**: depois de `time_stop_bars` pregões sem andar meia
  distância de stop, sai no fechamento.
- **Perda máxima**: se abrir com perda maior que `max_loss_per_trade_pct`, sai na abertura.
- **Pausas**: 2 stops seguidos → `consecutive_loss_cooldown` pregões sem
  operar; drawdown do ticker acima de `equity_drawdown_stop_pct` → 20 pregões.

### Custos (ISK)

0,25% de courtage por ordem (classe Mini) + 0,10% de slippage por lado =
**~0,70% ida e volta**. Sem imposto por trade (ISK). Cenário otimista com
courtage zero (Avanza Start abaixo de 50 kSEK; campanha Nordnet até
30/06/2027).

## 2. Instalação no Windows (WSL2)

No Windows nativo o GA roda com um único processo; no WSL2 usa todos os
núcleos e o ambiente é igual ao do CI.

1. No PowerShell como administrador: `wsl --install -d Ubuntu` e reinicie.
2. Abra o Ubuntu e instale o básico:
   ```bash
   sudo apt update && sudo apt install -y git build-essential
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   ```
3. Clone e prepare o ambiente (Python 3.11, como no CI):
   ```bash
   git clone https://github.com/gcallj/test.git ~/test && cd ~/test
   git checkout claude/jolly-dijkstra-8clsp4
   uv venv --python 3.11 .venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
4. Confira: `python -m unittest discover -s tests -p 'test_*.py'`.

Trabalhe dentro de `~/` (sistema de arquivos do Linux), não em `/mnt/c/`:
é muito mais rápido.

## 3. Ver as simulações (v1, histórico)

```bash
# 1) Dados e payloads (~15 min)
python -c "from stock_etl import run_etl; run_etl()"
python replay_se.py store

# 2) Genomas do walk-forward: baixe os do CI (git pull) ou treine localmente
python walkforward.py status
python walkforward.py train --fold 1        # ~3,5 h por fold com 4 núcleos

# 3) Replay por ticker: tabela pregão a pregão + gráfico + resumo
python replay_se.py ticker VOLV-B.ST        # -> reports/se/tickers/VOLV_B_ST.*

# 4) Todos os tickers e a carteira
python replay_se.py all
python portfolio_se.py                      # -> reports/se/portfolio.md / .png
```

Para estudar a lógica sem os genomas do walk-forward, rode o replay com um
genoma qualquer (modo **in-sample, ilustrativo** — os números não valem como
validação): `python replay_se.py ticker VOLV-B.ST --genome data/walkforward/live.json`.

### Como ler o replay de um ticker

`<TICKER>_diario.csv`, uma linha por pregão:

| Coluna | Significado |
|---|---|
| `situacao` | sem posição / ordem ativa / comprado, durante o pregão |
| `aconteceu_no_dia` | compra executada, venda (motivo e resultado líquido), ordem expirada ou cancelada |
| `decisao_no_fechamento` | a ordem para o próximo pregão (limite, stop, alvo) ou o motivo de não comprar |
| `votos` / `votos_min` | fração ponderada de indicadores votando compra vs o mínimo do genoma |
| `score` / `gatilho` | score suavizado vs o percentil que ele precisa superar |
| `forca` | força do sinal (0 a 1) |
| `regime` / `regime_min` | score de regime de mercado vs o mínimo |
| `vol_rank` | percentil da volatilidade no último ano |

`<TICKER>_ordens.csv`: uma linha por ordem, com status (executada / perdida /
cancelada / aberta), preços e resultado líquido. No gráfico: ▲ compra,
▼ venda (azul alvo, vermelho stop, laranja stop móvel, cinza tempo),
× ordem não executada; linhas tracejadas separam os folds.

### Como ler a carteira

`reports/se/portfolio.md` aplica o critério de sucesso do plano: render mais
que o fundo de índice (`^OMXSBGI`, com dividendos) depois de custos, ter
expectancy positiva em pelo menos 4 dos 6 folds e drawdown da conta menor que
o do índice. **Se não passar, a resposta honesta é o fundo de índice.**

## 4. Operar no dia a dia (v1, histórico)

1. **Genoma de uso diário** (a cada 6 meses): treine e, depois de ler o replay, aceite.
   ```bash
   python walkforward.py train --live          # ou o workflow SE Walk-forward no CI
   python walkforward.py accept                # copia para data/genome_se.json
   ```
2. **Iniciar a conta** (etapa 1 = paper): `python paper_se.py start`
3. **Todo dia depois do fechamento** (Estocolmo fecha 17:30 CET):
   ```bash
   python daily_se.py run      # -> reports/se/daily/<data>.md (+ Telegram)
   ```
   O relatório traz: novas ordens (limite, stop, alvo, validade), ordens de
   dias anteriores ainda ativas, posições abertas com o stop e o alvo de
   amanhã, e os sinais que ficaram de fora por falta de vaga.
4. **Registrar o que aconteceu** (paper ou real):
   ```bash
   python paper_se.py fill VOLV-B.ST 254.30
   python paper_se.py missed VOLV-B.ST
   python paper_se.py exit VOLV-B.ST 261.10 --reason alvo
   python paper_se.py check
   ```

### Ramp-up e regras de parada

| Etapa | Comando | Conta | Para avançar |
|---|---|---|---|
| 1. Paper (mechanics check) | `paper_se.py start` | 30.000 SEK fictícios | 10 pregões sem regra de parada |
| 2. Real reduzido | `paper_se.py start --stage real1` | 10.000 SEK, até 2 posições | 20 trades reais sem regra de parada |
| 3. Real cheio | `paper_se.py start --stage real2` | 30.000 SEK, até 6 posições | — |

Qualquer uma destas manda de volta ao paper trading: drawdown da conta pior
que o pior fold do replay; fill rate mais de 15 pp abaixo do replay; slippage
médio acima de 0,20% por lado. O paper trading confere a **mecânica**; quem
mede o edge é o replay walk-forward.

Telegram (opcional): crie um bot com o @BotFather, mande uma mensagem para
ele e defina `SE_TELEGRAM_BOT_TOKEN` e `SE_TELEGRAM_CHAT_ID` no ambiente.

## 5. Problemas conhecidos

- **Yahoo com HTTP 429**: o yfinance costuma funcionar mesmo quando um `curl`
  direto recebe 429; se o ETL falhar, espere alguns minutos e rode de novo.
- **`^OMXS30` não existe mais no Yahoo**: o sistema usa `^OMXSPI` para regime
  e `^OMXSBGI` (com dividendos) como benchmark.
- **Workflows do GitHub e branches**: cron e `workflow_dispatch` só funcionam
  com o workflow no branch padrão (`main`). O walk-forward deste branch é
  disparado por push em `data/walkforward/REQUEST`.
- **"WR 0%" em janela curta não é desastre**: é amostra vazia (nenhum trade
  na janela). Confira o número de trades antes de concluir qualquer coisa.

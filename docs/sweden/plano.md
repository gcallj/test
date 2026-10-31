# Plano — mercado sueco

> **Estado em 31/out/2026**: v1 (swing), v2 (rotação por momentum), v3 (seleção ativa por previsão) e v4 (eventos de informação pública: insiders, posições vendidas, tom dos comunicados, balanços) avaliadas; as quatro reprovadas. Ver "Estratégia v4", "Estratégia v3" e "Estratégia v2".

Decisões tomadas em sessão de design em 30/out/2026. Termos em negrito seguem
o glossário em `CONTEXT.md`; decisões difíceis de reverter estão em
`docs/adr/`.

## Objetivo

Um sistema de sinais para Nasdaq Stockholm que o dono entende, consegue rodar
sozinho, e cujo comportamento operacional foi validado por simulação honesta
antes de qualquer dinheiro real. O Brasil fica congelado como está.

## Decisões

| Área | Decisão |
|---|---|
| Onde mora | Branch deste repo criado a partir do `main` (`claude/jolly-dijkstra-8clsp4`, renomeável). `feature/sweden-omxs-market` (abr/2026) é só referência. |
| Universo | ADR 0003: snapshot Large Cap, classe mais líquida, giro mediano >= 10 MSEK/dia, sem cripto. |
| Conta | **ISK**: nenhum imposto por trade. |
| Custos | Brokerage Mini 0,25% por ordem + slippage 0,10% por lado = ~0,70% round-trip. Cenário otimista com brokerage zero (Avanza Start abaixo de 50 kSEK; campanha Nordnet até 30/jun/2027). |
| Carteira | 30.000 SEK, até 6 posições de 5.000 SEK, desempate por `rank`. |
| Execução | ADR 0001: limite na entrada, **order validity** de 3 pregões, cancela se fechar abaixo do stop; expirado = **missed entry**. O backtest usa a mesma regra. |
| Validação | ADR 0002: **walk-forward replay** — 3 anos, 6 **folds** de 6 meses, janela de treino móvel de 4 anos. Aceite manual do genoma. Sem loop contínuo de sweeps. |
| Retreinos | GitHub Actions, um job por fold em paralelo. |
| Ambiente local | WSL2 (Ubuntu) no Windows — no Windows nativo o GA roda com 1 processo só. |
| Dia a dia | Workflow novo no `main` que faz checkout deste branch às 18:30; sinais em bot/chat Telegram próprio + planilha nos artifacts. |

## Critério de sucesso

1. **Para ir ao paper trading** — o **portfolio replay** dos 3 anos precisa:
   render mais que o **index benchmark** (fundo do OMXSPI na mesma ISK) depois
   de custos; ter expectancy positiva em pelo menos 4 dos 6 folds; e ter max
   drawdown da conta menor que o do índice no mesmo período.
2. **Ramp-up**:
   1. Paper trading por 10 pregões — **mechanics check**, não prova de edge.
   2. Dinheiro real com 10.000 SEK (até 2 posições de 5.000).
   3. Capital cheio (30.000 SEK) após 20 trades reais sem stop rule disparada.
3. **Stop rules** (qualquer etapa → volta ao paper trading e investiga):
   drawdown da conta maior que o pior drawdown entre os folds do replay; fill
   rate mais de 15 pp abaixo do suposto pelo replay; slippage médio acima de
   0,20% por lado.
4. Se não bater o index benchmark, a resposta é comprar o fundo de índice.

## Fases

| # | Fase | Entrega |
|---|---|---|
| 0 | Investigar por que o Run 2 (abr/2026) ficou CLEAN | **Feito** — ver "Resultado da Fase 0" |
| 1 | Portar o mercado sueco para este branch | **Feito** — `market_se.py`, `data/universe_se.json`, ETL e metadados suecos |
| 2 | Regra de execução real no backtest | **Feito** — motor único, ADR 0001, `tests/test_execution_rule.py` |
| 3 | Walk-forward | **Feito** — 6 genomas + genoma diário (run 41139471106) |
| 4 | Replay por ticker | **Feito** — `replay_se.py` (tabela + gráfico + porquê) |
| 5 | Replay de carteira | **Feito** — veredito: **reprovado** (ver "Resultado do walk-forward") |
| 6 | Guia de execução local | **Feito** — `docs/sweden/guia.md` |
| 7 | Operação diária | **Feito** — `daily_se.py`; workflow diário em `docs/sweden/se_daily.yml`, aguarda ir para o `main` |
| 8 | Paper trading e ramp-up | **Feito** — `paper_se.py` (registro, stop rules, etapas) |

## Decisões tomadas durante a implementação

- **Motor único** (`ga_run_modular_final`): havia duas cópias do backtest, uma
  usada no treino do GA e outra na avaliação; agora o `ga_run` delega.
- **Sem informação do futuro no motor**: além da regra de execução, o stop do
  dia usa o ATR da véspera e a força do sinal usa o percentil do último ano
  (antes: percentil da série inteira).
- **Fundamentos trimestrais desligados no ETL**: eram indexados pela data do
  trimestre, não pela de divulgação (~30–60 dias depois).
- **Index benchmark = `^OMXSBGI`** (com dividendos). `^OMXSPI` é índice de
  preço e daria ~3 pp/ano de vantagem falsa à estratégia.
- **ADR 0004**: as ordens diárias são exatamente as ordens do motor; os
  ajustes pós-backtest do Brasil (Minervini, gates de WR/confiança, tier) não
  são usados.
- **Livro da conta sem estado escondido**: a conta de paper/real é a
  continuação do portfolio replay a partir da data de início.
- **Aquecimento igual em todo lugar**: o GA treinava em janelas fatiadas a
  frio (o primeiro mês de cada janela quase não operava) e o replay rodava
  desde 2005. Agora treino, replay e operação diária começam cada período com
  capital zerado e 260 pregões de aquecimento dos indicadores.
- **Trava do circuit breaker corrigida**: a pausa por drawdown se renovava
  para sempre (sem trades o capital não se recupera); agora a referência é
  rearmada na pausa e o ticker volta a operar depois de 20 pregões.
- **Corte de 80 features pela variância dos 3 primeiros anos**: antes usava o
  histórico inteiro, e um fold (histórico truncado) podia escolher features
  diferentes das do replay.

## Estratégia v4 — eventos de informação pública (31/out/2026)

Pedido: seguir a opção 3 (registros da Finansinspektionen) e investigar outros
dados públicos e análise de sentimento de notícias. O levantamento das fontes,
a cobertura medida e a literatura estão em `docs/sweden/dados_alternativos.md`.
O desenho está em `events_se.py` e `altdata_se.py` (ADR 0008).

| Hipótese | Dado | Evento | Prazo |
|---|---|---|---|
| H1 | registro de insiders (FI, desde 2016-07) | compras em bolsa ≥ 250 kSEK no dia, fora de programas de ações | 60 pregões |
| H2 | posições vendidas (FI, desde 2012-11) | 10% mais vendidas entre as cobertas (≥ 0,5%), previsto **negativo**; vira filtro | 20 pregões |
| H3 | comunicados Nasdaq + FinBERT | balanço/informação privilegiada com tom no terço superior do ano anterior | 20 pregões |
| H4 | datas dos balanços | reação de 3 dias no terço superior do ano anterior (deriva) | 60 pregões |

**Protocolo**:
- Hipóteses, prazos, limites e regra de aprovação foram tirados da
  literatura e congelados em `data/strategy_se_v4.json` antes de qualquer dado
  alternativo ser ligado a retornos.
- Rodada única em 2016-07 a 2026-10.
- Retorno anormal contra ações do mesmo quintil de retorno de 63 dias.
- Holm sobre as quatro hipóteses, t de Newey-West ≥ 2 e retorno médio acima de
  0,70% (ida e volta).
- A carteira (6 vagas, fundo de índice nas vagas livres) usa só o que passou e
  precisa cumprir o critério 1.

O dry-run (só contagens de eventos) achou dois problemas de dados, corrigidos
antes de calcular qualquer retorno e registrados no JSON:
- o FI deixava o tipo de instrumento vazio até 2018;
- detentores de posições vendidas aparecem com grafias diferentes, e faltam
  avisos de encerramento.

**Resultado** (`reports/se_v4/verdict.md`):

| Hipótese | Eventos | Retorno anormal médio | t | |
|---|---:|---:|---:|---|
| H1 compras de insiders | 1.253 | +0,05% | 1,45 | não passa |
| H2 posição vendida alta (previsto negativo) | 4.770 | −0,29% | 0,98 | não passa |
| H3 tom positivo | 1.349 | −0,32% | −0,09 | não passa |
| H4 reação ao balanço | 1.167 | −1,21% (sinal contrário) | −0,77 | não passa |

Nenhuma hipótese passa, então a carteira ficaria 100% no fundo de índice
(critério 1 reprovado: zero trades, drawdown igual ao do índice). **v4
reprovada.**

A carteira diagnóstica (todas as famílias, sem passar nos testes) vai de
+18% a +81% em 2023–2026, dependendo só da ordem de prioridade, contra +63,6%
do índice. Com a ordem da especificação, fica em +56,6%. Por trade, o retorno
bruto (+3,7%) empata com o do fundo de índice nas mesmas janelas (+3,6%).

**Conclusão das quatro tentativas**: nem preço e volume (v1–v3) nem os
registros públicos de insiders, posições vendidas, tom dos comunicados e
reação a balanços (v4) mostram, nas ~120 large caps suecas de 2016 a 2026,
uma vantagem que cubra 0,70% de custo de ida e volta. A alternativa que o
próprio critério de sucesso aponta é o fundo de índice.

## Estratégia v3 — seleção ativa por previsão (31/out/2026)

Pedido: estratégia ativa como a v1, que antecipe altas e quedas, faça poucos
trades e tenha relevância estatística. Desenho (`selective_se.py`, ADR 0007):
previsão semanal do retorno de cada ação nas próximas 2–4 semanas contra o
universo; entrada só no topo das previsões, permanência mínima de H pregões,
saída quando a previsão cai abaixo da mediana; filtro opcional de queda do
mercado (long-only).

**Protocolo**: sem histórico limpo restante, desenvolvimento só em 2010–2019
(40 variações, `reports/se_v3/dev_grid.csv`); especificação e 4 testes
congelados em `data/strategy_se_v3.json` (commit `6a6a7bb`); uma rodada no
holdout 2020-01 a 2026-10.

Desenvolvimento: os modelos ajustados (ridge, LightGBM) não passaram no teste
de previsão; composições de sinais da literatura com sinais fixos passaram
(t 3,2–3,7). A escolhida pela regra foi `combo5-H20-q3` (reversão de 1 semana
e de 1 mês, momentum 12-1, volatilidade baixa, giro sem picos; ~47 trades/ano).

| Holdout 2020–2026 | Resultado | Exigido | |
|---|---:|---:|---|
| S1 previsão (IC semanal) | 0,004, t = 0,35 | t ≥ 2 | não passa |
| S2 excesso sobre o índice | p = 0,88 | p < 0,05 | não passa |
| S3 Deflated Sharpe | 0,00 | > 0,95 | não passa |
| C1 critério 1 (2023–2026) | +34,2% vs +63,6%; MDD −20,8% vs −21,3%; 5/6 folds | os 3 | não passa |

CAGR no holdout: 4,8% contra 10,7% do índice e 12,0% do universo com peso
igual. **O sinal sumiu fora da amostra**: a correlação, positiva em todos os
10 anos do desenvolvimento, oscilou em torno de zero em 2020–2026 (50% das
semanas positivas). Explicações compatíveis: viés de sobrevivência inflando a
reversão no passado (as empresas que caíram e sumiram não estão no universo) e
o enfraquecimento documentado da reversão de curto prazo.

**Conclusão das três tentativas**: com preço e volume diários de ~120 large
caps suecas e custos de varejo (0,70% ida e volta), não há evidência de uma
vantagem ativa que sobreviva fora da amostra. Todo o histórico de 2010 a 2026
já foi usado para desenvolver ou julgar v1–v3; uma nova estratégia precisa
de **dados novos** (paper trading, o futuro) ou de **informação nova** que o
preço ainda não contém (por exemplo, os registros públicos da
Finansinspektionen de negociação de insiders e de posições vendidas).

## Estratégia v2 — rotação mensal por momentum (31/out/2026)

O swing trade (v1) foi reprovado por causas estruturais, medidas em
`docs/sweden/diagnostico.md`. A v2 (`rotation_se.py`, ADR 0005) troca o
estilo: rotação mensal entre as ações do universo pelo momentum de 12 meses,
6 posições, buffer de 20, volatility target de 20%, execução na abertura.

**Protocolo (ADR 0006)**: desenhada e escolhida só com dados de 2014 a
30/10/2023 (`reports/se_v2/dev_grid.csv`, 216 variações), congelada em
`data/strategy_se_v2.json` (commit `cf9e269`) e avaliada **uma vez** nos 3
anos fora da amostra.

| Critério 1 | v1 | **v2** | Índice `^OMXSBGI` | v2 |
|---|---:|---:|---:|---|
| Retorno em 3 anos, depois de custos | −26,2% | **+35,7%** | +63,6% | não passa |
| Folds com expectancy positiva | 2 de 6 | **5 de 6** | precisa ≥ 4 | passa |
| Max drawdown da conta | −45,6% | **−20,7%** | −21,3% | passa |

**Veredito: reprovada no critério de retorno.** Relatórios em `reports/se_v2/`.

O que a avaliação mostra:

- **A v2 corrigiu o que o diagnóstico apontou**: ~11 posições novas por ano
  (contra ~180 trades), 90% investida, quedas menores que as do índice.
- **Ficou à frente do índice por quase dois anos** (meados de 2024 a abril
  de 2026, pico de ~51 kSEK) e perdeu a vantagem no fold 6 (mai–out/2026,
  −14,3% contra +7,3%): uma quebra típica de momentum, quando as líderes
  recentes caem e o resto do mercado sobe. O fold 1 (rali de nov/2023) é o
  mesmo fenômeno ao contrário.
- **A falha é da família, não da escolha**: das 216 variações do
  desenvolvimento rodadas nos 3 anos de avaliação (só como distribuição,
  `reports/se_v2/eval_family_distribution.csv`), **3 (1%)** bateram o
  índice em retorno; a mediana fez +37,5% — a escolhida fez +35,7%. Já o
  drawdown menor que o do índice apareceu em 176 (81%). Escolher agora uma
  das 3 seria seleção pelo teste (ADR 0006).
- **O índice foi difícil de bater**: +17,8% ao ano, alta ampla (o universo
  com pesos iguais subiu igual ao índice, +63,9%), sem as grandes dispersões
  que favorecem momentum.

**Pela regra do plano, a resposta para o objetivo de retorno é o fundo de
índice.** O que a v2 entrega de forma consistente é retorno positivo com
drawdown menor que o do índice — um objetivo diferente do critério 1, que
só pode ser adotado como critério novo, validado em dados novos.

## Resultado do walk-forward (31/out/2026)

Execução oficial: run 41139471106 (genomas em `data/walkforward/`). Relatórios
em `reports/se/` — `portfolio.md` tem o veredito; `tickers/` tem ordens e
resumo de cada ticker, com tabela diária e gráfico para uma seleção.

| Critério 1 | Sistema | Índice `^OMXSBGI` | |
|---|---:|---:|---|
| Retorno em 3 anos, depois de custos | −26,2% | +63,6% | não passa |
| Folds com expectancy positiva | 2 de 6 | precisa ≥ 4 | não passa |
| Max drawdown da conta | −45,6% | −21,3% | não passa |

Com courtage zero: +16,4% contra +63,6% — também não passaria.

**Veredito: reprovado. Pelo critério fixado antes do teste, a resposta é o
fundo de índice, e o sistema não vai para paper trading.**

O que os números mostram:

- **Custos decidem boa parte**: ~550 trades em 3 anos a ~0,70% cada. O mesmo
  sistema vai de −26% para +16% só zerando a courtage.
- **Mesmo sem custos não chega perto do índice**: o motor captura movimentos
  curtos (alvos de 1–2 ATR, poucos pregões) e fica fora das tendências longas,
  que é de onde veio o retorno do mercado sueco no período (+17,8% ao ano).
- **Instável entre folds**: só os folds 1 e 6 têm expectancy positiva; o fold
  1 sozinho sobe a conta a ~36 kSEK e os folds 2–5 a derrubam.

Se você quiser continuar pesquisando (decisão sua): mudanças entram no
motor e passam por um novo walk-forward com critério fixado antes. Cada nova
tentativa sobre os mesmos 3 anos "gasta" esse período como fora da amostra;
depois de algumas, só um período novo (paper trading ou o futuro) valida.

## Walk-forward: execução oficial

Decidido em 30/out/2026, **antes de ver qualquer replay fora da amostra**:
a 1ª execução do workflow (run 41131117803) vale só como teste de pipeline. O
auto-tune estimou 3,2h por job e o GA terminou em 58 min, com o fitness ainda
melhorando na última geração. A execução oficial usa um GA de tamanho fixo
(estágio 1: população 80 × 25 gerações; estágio 2: 40 × 30), ~3h por job. Os
replays e o veredito da carteira usam somente os genomas da execução oficial;
escolher entre execuções pelo resultado fora da amostra contaminaria a
validação.

## Resultado da Fase 0 (30/out/2026)

Reproduzido neste container com o código exato do branch antigo (`a5606dd`),
dados de hoje.

- **Sem BIN/REG, o FINAL falha nas 309 janelas** (`Empty ENS chunk`), não
  grava `history_consolidated.parquet` e ainda assim sai com código 0. Nos
  Runs 0/1 o GA usou o histórico montado a partir do ETL.
- **Esse histórico do ETL é saudável**: 44 tickers `.ST`, 2005 até hoje, sem
  NaN no fim. Com o genoma de partida, as 3 janelas de holdout têm ~10 trades
  por ticker e WR 0,60–0,65.
- **"WR 0,000 no holdout" é um artefato de amostra vazia**: as medianas usam
  só tickers com trades na janela e devolvem 0 quando a lista fica vazia.
  Um genoma que opera pouco gera "WR 0" sem ter perdido trade algum.
- **O BIN/REG não alimenta a fitness** — `EV_*`/`p_*` são bloqueados. O que
  ele mudou no Run 2 foi **quais datas o GA viu**: só as cobertas pelos
  modelos, um histórico bem mais curto e recente que os 21 anos do ETL.

**Decisão**: BIN/REG/FINAL não são obrigatórios. A janela de treino é
controlada explicitamente pelo walk-forward (4 anos móveis), e os relatórios
suecos mostram o número de trades de cada janela, com "-" quando não houve
trade — nunca WR 0.

O genoma BR aplicado à Suécia hoje: fitness −97,9, WR 0,576, alpha −13,8%.
Não transfere; não deve ser o ponto de partida dos folds.

## Obstáculos conhecidos da execução anterior

- `^OMXS30` deslistado no Yahoo (HTTP 404) — usar `^OMXSPI`.
- Cron do GitHub só roda no branch padrão — resolvido pelo workflow no `main`.
- `workflow_dispatch` de workflow inexistente no `main` dá HTTP 404.
- Commit automático falhou com `git pull --rebase` e working tree sujo — usar
  `-c rebase.autoStash=true`.
- Run 2 levou 5h47 num job de limite 6h.
- A explicação do Run 2 ("EV_buy_fund alimentando a fitness") não bate com o
  código: o prefixo `EV_` é bloqueado no GA. Fase 0 resolve.

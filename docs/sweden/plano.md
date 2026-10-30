# Plano — mercado sueco

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
| 3 | Walk-forward | Código feito (`walkforward.py`, workflow `SE Walk-forward`); genomas: ver `walkforward.py status` |
| 4 | Replay por ticker | **Feito** — `replay_se.py` (tabela + gráfico + porquê) |
| 5 | Replay de carteira | **Feito** — `portfolio_se.py`, veredito em `reports/se/portfolio.md` |
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

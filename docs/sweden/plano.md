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
| 0 | Investigar por que o Run 2 (abr/2026) ficou CLEAN | Causa confirmada; decide se BIN/REG/FINAL é obrigatório |
| 1 | Portar o mercado sueco para este branch | Universo, custos, ISK, índice `^OMXSPI`, ETL `.ST` |
| 2 | Regra de execução real no backtest | ADR 0001 implementado, com testes |
| 3 | Walk-forward | 6 genomas, um por fold, treinados no CI |
| 4 | Replay por ticker | Tabela pregão a pregão + gráfico + porquê de cada sinal |
| 5 | Replay de carteira | Curva da conta vs index benchmark; veredito do critério 1 |
| 6 | Guia de execução local | Passo a passo em WSL2, em português |
| 7 | Operação diária | Workflow no `main`, Telegram sueco |
| 8 | Paper trading e ramp-up | Registro de execução e stop rules |

## Obstáculos conhecidos da execução anterior

- `^OMXS30` deslistado no Yahoo (HTTP 404) — usar `^OMXSPI`.
- Cron do GitHub só roda no branch padrão — resolvido pelo workflow no `main`.
- `workflow_dispatch` de workflow inexistente no `main` dá HTTP 404.
- Commit automático falhou com `git pull --rebase` e working tree sujo — usar
  `-c rebase.autoStash=true`.
- Run 2 levou 5h47 num job de limite 6h.
- A explicação do Run 2 ("EV_buy_fund alimentando a fitness") não bate com o
  código: o prefixo `EV_` é bloqueado no GA. Fase 0 resolve.
